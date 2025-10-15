## RAG Demos (HyDE / StepBack / Recursive Retrieval / Reranking)

This fork focuses on retrieval-only experiments built on LangChain. All four demos share one indexing pipeline and differ only in how they retrieve:

- `hyde_demo.py` – HyDE: generate hypothetical terms then retrieve
- `stepback_demo.py` – StepBack: abstract high-level concepts then retrieve
- `recursive_retrieval_demo.py` – two-stage parent/child retrieval
- `reranking_demo.py` – vector recall then external reranker

Design goals:

- Use repository Markdown as the only data source
- Structured splitting (headers + recursive chunking)
- Fresh, temporary Chroma DB per run
- Real embeddings via SiliconFlow
- Stable chunk IDs; skip re-embedding duplicates
- Minimal output; no noise

---

## Quick Start

Requirements:

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for fast env + installs

Setup:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Environment (`.env` in project root):

```bash
SILICONFLOW_API_KEY=sk-...
SILICONFLOW_BASE_URL=https://api.siliconflow.cn
EMBEDDING_MODEL=BAAI/bge-m3
RERANKING_MODEL=Qwen/Qwen3-Reranker-0.6B
LLM_MODEL=Qwen/Qwen3-VL-235B-A22B-Instruct
```

`.gitignore` already excludes `.env` and `.venv*`.

---

## Run Demos

HyDE (hypothetical terms → retrieve):

```bash
.venv/bin/python hyde_demo.py "What is LangChain Text Splitters?"
```

StepBack Prompt:

```bash
.venv/bin/python stepback_demo.py "LangChain 项目提供了什么?"
```

Reranking:

```bash
.venv/bin/python reranking_demo.py "重排序技术"
```

Recursive Retrieval (two-stage):

```bash
.venv/bin/python recursive_retrieval_demo.py "LangChain 如何组织检索?"
```

Output format (example):

```
query: ...
chunks: <total>, added: <embedded>, skipped: <dedup>
1. <path> | <snippet>
...
```

Each run builds a new temp Chroma index and persists it only for the process lifetime.

---

## How It Works

- `retrieval_common.py`
  - Discover all Markdown (excluding `.git`, `.venv`, `chroma_db*`)
  - Split with `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`
  - Create a temp Chroma store
  - Generate stable `chunk_id` and skip duplicate IDs on add
- `siliconflow_embeddings.py`
  - Calls SiliconFlow embeddings API
  - Batch size configurable via `EMBED_BATCH` (default 64)

---

## Tests

```bash
uv run --python .venv/bin/python --group test pytest tests/unit_tests/test_retrieval_common.py -q
```

---

## Notes

- Chroma import from `langchain_community.vectorstores` emits a deprecation warning in LangChain >=0.2.9; warnings are suppressed in code to keep output minimal.
- Do not commit `.env`. It’s already ignored.

---

## Chunking Demos

Lightweight scripts to showcase different splitting strategies against this repo’s `README.md`:

- Hierarchical (Markdown headers):

```bash
.venv/bin/python hierarchical_chunking.py
```

- Recursive character:

```bash
.venv/bin/python recursive_chunking.py
```

- Semantic separators:

```bash
.venv/bin/python semantic_chunking.py
```

- Structural example with sample text:

```bash
.venv/bin/python structural_chunking.py
```
