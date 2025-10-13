#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

import retrieval_common as rc


class _DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):  # type: ignore[override]
        return [[0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):  # type: ignore[override]
        return [0.0, 0.0, 0.0]


def test_iter_markdown_files_discovers_repo_md():
    root = Path.cwd()
    paths = list(rc.iter_markdown_files(root))
    assert any(p.name.lower() == "readme.md" for p in paths)


def test_split_markdown_returns_chunks():
    md = Path("libs/text-splitters/README.md")
    docs = rc.split_markdown(md)
    assert isinstance(docs, list)
    assert len(docs) > 0


def test_add_texts_skip_existing():
    temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    vs = Chroma(collection_name="t", embedding_function=_DummyEmbeddings(), persist_directory=temp_dir)

    texts = ["hello", "world"]
    ids = ["a", "b"]
    metas = [{"source": "s"}, {"source": "s"}]

    added1, skipped1 = rc.add_texts_skip_existing(vs, texts=texts, metadatas=metas, ids=ids)
    assert added1 == 2 and skipped1 == 0

    added2, skipped2 = rc.add_texts_skip_existing(vs, texts=texts, metadatas=metas, ids=ids)
    assert added2 == 0 and skipped2 == 2

