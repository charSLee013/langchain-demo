#!/usr/bin/env python3
"""
Markdown Retrieval Demo (focused, real, minimal)
"""

from __future__ import annotations

import sys
from pathlib import Path

from retrieval_common import index_markdown


def main() -> None:
    vs, stats = index_markdown(Path.cwd())
    query = sys.argv[1] if len(sys.argv) > 1 else "What is LangChain Text Splitters?"
    results = vs.similarity_search(query, k=5)

    print(f"query: {query}")
    print(f"chunks: {stats.chunks}, added: {stats.added}, skipped: {stats.skipped}")
    for i, d in enumerate(results, 1):
        src = d.metadata.get("source", "unknown")
        snippet = d.page_content.replace("\n", " ")[:120]
        print(f"{i}. {src} | {snippet}")


if __name__ == "__main__":
    main()
