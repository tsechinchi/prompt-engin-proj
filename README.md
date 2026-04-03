# HKBU Study Companion

Project skeleton for a prompt engineering / RAG / agentic workflow assignment.

## Setup With `uv`

This project is designed to work well with [`uv`](https://docs.astral.sh/uv/).

1. Install `uv` if you do not already have it.
2. Sync the environment:

```bash
uv sync
```

3. Add a package when you need one:

```bash
uv add <package-name>
```

`uv sync` keeps the environment aligned with the project files, and `uv add` updates dependencies for you.

This project now prefers `faiss-gpu-cu12` on Linux and falls back to `faiss-cpu` on non-Linux platforms. That keeps `uv sync` working across your machine and a future CUDA setup.

## Project TODO

See [`PROJECT_TODO.md`](PROJECT_TODO.md) for the prioritized checklist.

## Structure

### Source code

- `src/ingestion/`: document loading and chunking
- `src/retrieval/`: BM25, vector search, and hybrid ranking
- `src/prompt/`: prompt assembly helpers and reusable templates
- `src/generation/`: Ollama wrapper and generation controls
- `src/memory/`: conversation state and history handling
- `src/evaluation/`: quality and token-usage tracking
- `src/agent/`: optional LangGraph orchestration, tools, and HITL

### Project files

- `notebooks/00_ollama_raw_template.ipynb`: provided raw Ollama completion template
- `notebooks/01_baseline_no_rag.ipynb`: prompt-only baseline
- `notebooks/02_rag_pipeline.ipynb`: hybrid RAG pipeline
- `notebooks/03_evaluation.ipynb`: baseline vs RAG evaluation
- `data/`: local source documents and raw ingested files
- `report/`: report notes and export helpers before the final PDF

## Next steps

1. Install dependencies.
2. Add source documents to `data/`.
3. Fill in the notebook TODOs and module implementations.
