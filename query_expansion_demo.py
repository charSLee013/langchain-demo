#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from retrieval_common import index_markdown
from simple_query_expander import expand_query_simple


def main() -> None:
    vs, stats = index_markdown(Path.cwd())
    base = sys.argv[1] if len(sys.argv) > 1 else "这个项目是干什么的?"
    variants: List[str] = expand_query_simple(base)

    hits = []
    for q in variants:
        hits.extend(vs.similarity_search(q, k=2))

    # de-dupe by chunk_id
    seen: set[str] = set()
    uniq = []
    for d in hits:
        cid = d.metadata.get("chunk_id") or d.metadata.get("id") or d.page_content[:64]
        if cid in seen:
            continue
        seen.add(cid)
        uniq.append(d)

    print(f"query: {base}")
    print(f"variants: {len(variants)}")
    print(f"chunks: {stats.chunks}, added: {stats.added}, skipped: {stats.skipped}")
    for i, d in enumerate(uniq[:5], 1):
        src = d.metadata.get("source", "unknown")
        snippet = d.page_content.replace("\n", " ")[:120]
        print(f"{i}. {src} | {snippet}")


if __name__ == "__main__":
    main()
