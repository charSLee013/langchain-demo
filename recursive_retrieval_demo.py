#!/usr/bin/env python3
"""Recursive retrieval demo (two-stage)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retrieval_common import (
    IndexStats,
    add_texts_skip_existing,
    build_temp_chroma,
    index_markdown,
    stable_chunk_id,
)


def main() -> None:
    vs, stats = index_markdown(Path.cwd())
    query = sys.argv[1] if len(sys.argv) > 1 else "LangChain 如何组织检索?"

    stage1_hits = vs.similarity_search(query, k=5)
    top_sources = {
        doc.metadata.get("source")
        for doc in stage1_hits
        if doc.metadata.get("source")
    }

    parent_vs, parent_stats = _build_parent_store(top_sources)
    stage2_hits = parent_vs.similarity_search(query, k=5)

    print(f"query: {query}")
    print(f"stage1_chunks: {stats.chunks}, added: {stats.added}, skipped: {stats.skipped}")
    print(
        f"stage2_chunks: {parent_stats.chunks}, added: {parent_stats.added}, "
        f"skipped: {parent_stats.skipped}"
    )
    print("stage1 (top5):")
    for i, doc in enumerate(stage1_hits, 1):
        src = doc.metadata.get("source", "unknown")
        snippet = doc.page_content.replace("\n", " ")[:120]
        print(f"{i}. {src} | {snippet}")
    print("stage2 (top5):")
    for i, doc in enumerate(stage2_hits, 1):
        src = doc.metadata.get("source", "unknown")
        snippet = doc.page_content.replace("\n", " ")[:160]
        print(f"{i}. {src} | {snippet}")


def _build_parent_store(sources: Iterable[str]) -> Tuple[Chroma, IndexStats]:
    """Construct a coarse-grained vector store over parent documents."""

    docs = _load_parent_docs(sources)
    vs = build_temp_chroma()
    if not docs:
        return vs, IndexStats(chunks=0, added=0, skipped=0)

    texts, metas, ids = _prepare_vectors(docs)
    added, skipped = add_texts_skip_existing(vs, texts=texts, metadatas=metas, ids=ids)
    return vs, IndexStats(chunks=len(ids), added=added, skipped=skipped)


def _load_parent_docs(sources: Iterable[str]) -> List[Document]:
    """Load and split parent documents (larger blocks) for recursion."""
    chunker = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    parent_docs: List[Document] = []
    for src in {s for s in sources if s}:
        path = Path(src)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        base_doc = Document(page_content=text, metadata={"source": str(path), "level": "parent"})
        parent_docs.extend(chunker.split_documents([base_doc]))
    return parent_docs


def _prepare_vectors(docs: Sequence[Document]) -> Tuple[List[str], List[dict], List[str]]:
    """Prepare text, metadata, ids for vector insertion (deduped)."""
    texts: List[str] = []
    metas: List[dict] = []
    ids: List[str] = []
    seen: set[str] = set()
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        cid = stable_chunk_id(src, doc.page_content)
        if cid in seen:
            continue
        seen.add(cid)
        doc.metadata["chunk_id"] = cid
        texts.append(doc.page_content)
        metas.append(doc.metadata)
        ids.append(cid)
    return texts, metas, ids


if __name__ == "__main__":
    main()
