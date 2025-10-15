#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document

from retrieval_common import index_markdown


def siliconflow_rerank(query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
    load_dotenv()
    api_key = os.getenv("SILICONFLOW_API_KEY")
    base_url = os.getenv("SILICONFLOW_BASE_URL")
    model = os.getenv("RERANKING_MODEL")
    if not api_key or not base_url or not model:
        raise RuntimeError("Missing SILICONFLOW_API_KEY/SILICONFLOW_BASE_URL/RERANKING_MODEL in environment")

    payload = {
        "model": model,
        "instruction": "请根据查询对文档进行重排序",
        "query": query,
        "documents": [d.page_content for d in documents],
        "top_n": top_n,
        "return_documents": True,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(
        f"{base_url}/v1/rerank",
        headers=headers,
        data=json.dumps(payload).encode("utf-8"),
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    idx = [r.get("index") for r in body.get("results", []) if isinstance(r.get("index"), int)][:top_n]
    return [documents[i] for i in idx if 0 <= i < len(documents)]


def main() -> None:
    vs, stats = index_markdown(Path.cwd())
    query = sys.argv[1] if len(sys.argv) > 1 else "重排序技术"
    # 典型流程：先用向量召回更多候选（如50），再重排选Top-5
    initial = vs.similarity_search(query, k=50)
    reranked = siliconflow_rerank(query, initial, top_n=5)

    print(f"query: {query}")
    print(f"chunks: {stats.chunks}, added: {stats.added}, skipped: {stats.skipped}")
    print("initial (top5 of 50):")
    for i, d in enumerate(initial[:5], 1):
        src = d.metadata.get("source", "unknown")
        snippet = d.page_content.replace("\n", " ")[:120]
        print(f"{i}. {src} | {snippet}")
    print("reranked (top5):")
    for i, d in enumerate(reranked[:5], 1):
        src = d.metadata.get("source", "unknown")
        snippet = d.page_content.replace("\n", " ")[:120]
        print(f"{i}. {src} | {snippet}")



if __name__ == "__main__":
    main()
