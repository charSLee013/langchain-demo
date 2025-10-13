#!/usr/bin/env python3
"""
Common utilities for Markdown indexing and retrieval.

Single, focused place for:
- Discovering Markdown files
- Structured splitting
- Temp Chroma creation
- Stable chunk IDs and dedup add (skip re-embedding)
"""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core._api.deprecation import LangChainDeprecationWarning
import warnings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from siliconflow_embeddings import SiliconFlowEmbeddings


EXCLUDE_DIR_NAMES: Tuple[str, ...] = (".git", "__pycache__", ".venv")
EXCLUDE_DIR_PREFIXES: Tuple[str, ...] = ("chroma_db",)


def iter_markdown_files(root: Path) -> Iterable[Path]:
    """Yield Markdown file paths under `root`, skipping vectorstore/build dirs."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in EXCLUDE_DIR_NAMES and not any(d.startswith(p) for p in EXCLUDE_DIR_PREFIXES)
        ]
        for name in filenames:
            if name.lower().endswith(".md"):
                yield Path(dirpath) / name


def split_markdown(path: Path) -> List[Document]:
    """Split a Markdown file into structured chunks.

    Args:
        path: Markdown path.

    Returns:
        List of chunked Documents with source metadata.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    header_docs = header_splitter.split_text(text)
    for d in header_docs:
        d.metadata["source"] = str(path)

    chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return chunker.split_documents(header_docs)


def stable_chunk_id(source: str, content: str) -> str:
    sha = hashlib.sha256()
    sha.update(source.encode("utf-8", errors="ignore"))
    sha.update(b"\x00")
    sha.update(content.encode("utf-8", errors="ignore"))
    return sha.hexdigest()


def build_temp_chroma() -> Chroma:
    """Create an empty temp-persisted Chroma with SiliconFlow embeddings."""
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
    temp_dir = tempfile.mkdtemp(prefix="chroma_tmp_")
    atexit.register(shutil.rmtree, temp_dir, True)
    embeddings = SiliconFlowEmbeddings()
    return Chroma(
        collection_name="md_chunks",
        embedding_function=embeddings,
        persist_directory=temp_dir,
    )


def add_texts_skip_existing(
    vs: Chroma, *, texts: Sequence[str], metadatas: Sequence[dict], ids: Sequence[str]
) -> Tuple[int, int]:
    """Add texts skipping pre-existing IDs. Returns (added, skipped)."""
    existing: set[str]
    try:
        got = vs._collection.get(ids=list(ids), include=["metadatas"])  # type: ignore[attr-defined]
        existing = set(got.get("ids", []) or [])
    except Exception:
        existing = set()

    keep = [i for i, _id in enumerate(ids) if _id not in existing]
    if not keep:
        return 0, len(ids)

    vs.add_texts(
        texts=[texts[i] for i in keep],
        metadatas=[metadatas[i] for i in keep],
        ids=[ids[i] for i in keep],
    )
    return len(keep), len(ids) - len(keep)


@dataclass(frozen=True)
class IndexStats:
    chunks: int
    added: int
    skipped: int


def index_markdown(root: Path) -> Tuple[Chroma, IndexStats]:
    """Index all Markdown files under root into a temp Chroma store.

    Returns:
        (vectorstore, IndexStats)
    """
    paths = list(iter_markdown_files(root))
    if not paths:
        raise RuntimeError("no markdown files")

    docs: List[Document] = []
    for p in paths:
        docs.extend(split_markdown(p))
    if not docs:
        raise RuntimeError("no chunks")

    texts: List[str] = []
    metas: List[dict] = []
    ids: List[str] = []
    seen: set[str] = set()
    for d in docs:
        src = d.metadata.get("source", "unknown")
        cid = stable_chunk_id(src, d.page_content)
        if cid in seen:
            continue
        seen.add(cid)
        d.metadata["chunk_id"] = cid
        texts.append(d.page_content)
        metas.append(d.metadata)
        ids.append(cid)

    vs = build_temp_chroma()
    added, skipped = add_texts_skip_existing(vs, texts=texts, metadatas=metas, ids=ids)
    return vs, IndexStats(chunks=len(ids), added=added, skipped=skipped)
