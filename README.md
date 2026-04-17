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

## Quick Start

### Prerequisites

- Python 3.11 or higher
- `uv` package manager ([install here](https://docs.astral.sh/uv/))
- Ollama (optional, for local LLM; fallback to mock generation available)

### Installation

1. **Clone and enter the project directory:**

```bash
cd prompt-engin-proj
```

2. **Sync dependencies with `uv`:**

```bash
uv sync
```

This installs all required packages listed in `pyproject.toml`.

### Launch the Full System

#### Option 1: API Server + Frontend (Recommended)

**Terminal 1 – Backend API:**

```bash
python run_api.py
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 – Frontend UI:**

```bash
cd frontend
python -m http.server 4173
```

Expected output:
```
Serving HTTP on 0.0.0.0 port 4173
```

**Open in browser:**

- Frontend: http://localhost:4173

The frontend will connect to the API at `http://localhost:8000` by default.

#### Option 2: API Only (Development/Testing)

```bash
python run_api.py
```

Test the API health endpoint:

```bash
curl http://localhost:8000/api/health
# Response: {"status":"ok"}
```

### Run Tests

```bash
python -m pytest tests/ -v
```

Expected: **37 tests pass**

### Run Notebooks

Execute the evaluation notebooks:

```bash
# Baseline (no RAG)
jupyter notebook notebooks/01_baseline_no_rag.ipynb

# RAG pipeline
jupyter notebook notebooks/02_rag_pipeline.ipynb

# Evaluation & comparison
jupyter notebook notebooks/03_evaluation.ipynb
```

### Verify Installation

Quick sanity check:

```bash
python -c "
from src.ingestion import load_documents, chunk_documents
from src.retrieval import BM25Retriever, VectorRetriever, fuse_scores
from src.agent import build_graph
from src.api import app
print('All modules loaded successfully!')
"
```

## Using the Frontend

1. **Upload documents** (PDF, DOCX, PPTX) or use mock corpus
2. **Select retrieval mode:**
   - `Baseline` – No retrieval (prompt only)
   - `BM25` – Lexical search only
   - `Vector` – Semantic search only
   - `Hybrid` – BM25 + Vector (recommended)
3. **Adjust creativity slider** (temperature 0.0 to 1.0)
4. **Click "Generate Answer"** to get results with citations

### Environment Variables

Optional: Store backend URL in browser localStorage:

```javascript
// In browser console
localStorage.setItem("hkbu_api_base", "http://localhost:8000");
localStorage.setItem("hkbu_use_mock_corpus", "true");  // for demo mode
```

## Next Steps

1. Add custom source documents to `data/mock/` (PDF, DOCX, PPTX, TXT, MD format)
2. Connect live Ollama model (set `use_mock_generation: false` in frontend)
3. Write your report using `report/` templates
4. Run evaluation notebooks to measure quality metrics
