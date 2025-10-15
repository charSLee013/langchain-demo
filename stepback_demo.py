#!/usr/bin/env python3
"""StepBack prompt retrieval demo."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

from retrieval_common import index_markdown


def main() -> None:
    vs, stats = index_markdown(Path.cwd())
    base_query = sys.argv[1] if len(sys.argv) > 1 else "LangChain 项目提供了什么?"
    abstract_query = _stepback_prompt(base_query)

    base_hits = vs.similarity_search(base_query, k=5)
    abstract_hits = vs.similarity_search(abstract_query, k=5)

    print(f"query: {base_query}")
    print(f"stepback: {abstract_query[:200].replace('\n', ' ')}")
    print(f"chunks: {stats.chunks}, added: {stats.added}, skipped: {stats.skipped}")
    print("initial (top5):")
    for i, doc in enumerate(base_hits, 1):
        src = doc.metadata.get("source", "unknown")
        snippet = doc.page_content.replace("\n", " ")[:120]
        print(f"{i}. {src} | {snippet}")
    print("stepback (top5):")
    for i, doc in enumerate(abstract_hits, 1):
        src = doc.metadata.get("source", "unknown")
        snippet = doc.page_content.replace("\n", " ")[:120]
        print(f"{i}. {src} | {snippet}")


def _stepback_prompt(query: str) -> str:
    """Produce 2–6 high-level concepts/principles (comma-separated)."""

    load_dotenv()
    api_key = os.getenv("SILICONFLOW_API_KEY")
    base_url = os.getenv("SILICONFLOW_BASE_URL")
    model = os.getenv("LLM_MODEL")  # strictly use .env, no default
    if not api_key or not base_url or not model:
        raise RuntimeError("Missing SILICONFLOW_API_KEY/SILICONFLOW_BASE_URL/LLM_MODEL in environment")

    prompt = (
        "抽象该问题背后的核心概念/原则/系统目标，输出2-6个术语（中文逗号分隔）。"
        "不要复述问题文本，不要句子、编号或解释。\n\n问题: "
        + query
        + "\n\n仅输出术语："
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 96,
        "temperature": 0.2,
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    return content


if __name__ == "__main__":
    main()
