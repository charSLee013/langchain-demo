#!/usr/bin/env python3
"""
HyDE Retrieval Demo (focused, real, minimal)
"""

from __future__ import annotations

import sys
from pathlib import Path

import json
import os
from retrieval_common import index_markdown
from dotenv import load_dotenv
import requests


def main() -> None:
    vs, stats = index_markdown(Path.cwd())
    query = sys.argv[1] if len(sys.argv) > 1 else "What is LangChain Text Splitters?"

    # HyDE: generate a hypothetical answer (re-planned string)
    hyde_text = _hyde_generate(query)

    base_hits = vs.similarity_search(query, k=5)
    hyde_hits = vs.similarity_search(hyde_text, k=5)

    print(f"query: {query}")
    print(f"hyde: {hyde_text[:200].replace('\n', ' ')}")
    print(f"chunks: {stats.chunks}, added: {stats.added}, skipped: {stats.skipped}")
    print("initial (top5):")
    for i, d in enumerate(base_hits, 1):
        src = d.metadata.get("source", "unknown")
        snippet = d.page_content.replace("\n", " ")[:120]
        print(f"{i}. {src} | {snippet}")
    print("hyde (top5):")
    for i, d in enumerate(hyde_hits, 1):
        src = d.metadata.get("source", "unknown")
        snippet = d.page_content.replace("\n", " ")[:120]
        print(f"{i}. {src} | {snippet}")


def _hyde_generate(query: str) -> str:
    """Call SiliconFlow Chat API to generate a concise hypothetical answer.

    Falls back to the original query silently if request fails.
    """
    load_dotenv()
    api_key = os.getenv("SILICONFLOW_API_KEY")
    base_url = os.getenv("SILICONFLOW_BASE_URL")
    model = os.getenv("LLM_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")
    if not api_key or not base_url:
        return query

    prompt = (
        "请针对以下问题写一段150-250字的假设性技术说明，"
        "用于RAG检索，不要列表，不要客套语，只保留核心概念与术语。\n\n问题: "
        + query
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 240,
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return content or query
    except Exception:
        return query


if __name__ == "__main__":
    main()
