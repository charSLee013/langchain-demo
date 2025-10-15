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
import re


def main() -> None:
    vs, stats = index_markdown(Path.cwd())
    query = sys.argv[1] if len(sys.argv) > 1 else "What is LangChain Text Splitters?"

    # HyDE: generate a hypothetical answer (must differ from raw query)
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
    """Generate hypothetical answer strictly via configured LLM (no degrade)."""
    load_dotenv()
    api_key = os.getenv("SILICONFLOW_API_KEY")
    base_url = os.getenv("SILICONFLOW_BASE_URL")
    model = os.getenv("LLM_MODEL")  # strictly use .env, no default
    attempts = max(1, int(os.getenv("HYDE_ATTEMPTS", "2")))
    lang = _detect_lang(query)
    if not api_key or not base_url or not model:
        raise RuntimeError("Missing SILICONFLOW_API_KEY/SILICONFLOW_BASE_URL/LLM_MODEL in environment")

    if lang == "zh":
        base_prompt = (
            "将用户问题纠正并改写为用于检索的专业关键词，要求：\n"
            "- 仅输出用中文逗号分隔的2-8个关键词/术语；\n"
            "- 纠正错别字与概念，统一为学科规范称谓；\n"
            "- 不要句子/说明/编号/括号/引号/前后缀。\n\n问题："
            + query
            + "\n\n仅输出关键词列表："
        )
        retry_prompt = (
            "请只给出2-8个更规范的中文检索术语（用中文逗号分隔），"
            "纠正常见表述并提升检索可区分度。不要句子或解释。\n\n问题："
            + query
        )
    else:
        base_prompt = (
            "Rewrite the user's question into 2-8 domain-specific search terms, comma-separated. "
            "Correct typos and normalize to canonical technical/legal terms. No sentences or extra text.\n\nQuestion: "
            + query
            + "\n\nOutput only the comma-separated terms:"
        )
        retry_prompt = (
            "Return only 2-8 refined search terms (comma-separated), more canonical and discriminative. "
            "No sentences, no explanations.\n\nQuestion: "
            + query
        )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def call_llm(text: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "stream": False,
            "max_tokens": 96,
            "temperature": 0.2,
        }
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    if os.getenv("HYDE_DEBUG", "0") == "1":
        print(f"[HyDE] calling model={model}", file=sys.stderr)
    content = _postprocess_terms(call_llm(base_prompt))
    if _is_valid_hypothesis(query, content):
        return content
    for _ in range(attempts - 1):
        content = _postprocess_terms(call_llm(retry_prompt))
        if _is_valid_hypothesis(query, content):
            return content
    # 返回最后一次的规范化关键词结果（已严格使用 .env 模型）
    return content


def _is_valid_hypothesis(query: str, text: str) -> bool:
    q = _norm(query)
    t = _norm(text)
    if not t or t == q:
        return False
    if not (6 <= len(t) <= 200):
        return False
    tokens = [x for x in re.split(r"[\s,，;/]+", t) if x]
    return 2 <= len(tokens) <= 12


def _postprocess_terms(raw: str) -> str:
    """Normalize model output to comma-separated terms for embedding."""
    parts = re.split(r"[\n,，;；/\\|]+", raw)
    cleaned = []
    seen = set()
    for p in parts:
        term = p.strip().strip("-·•:：、。.,")
        if not term or len(term) < 2:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(term)
    return ", ".join(cleaned[:8])


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _detect_lang(text: str) -> str:
    """Very light language hint: zh if contains CJK, else en."""
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"

    if __name__ == "__main__":
        main()
