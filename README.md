# HKBU Study Companion

A lightweight prompt-engineering and retrieval-augmented generation (RAG) prototype.

This repository combines:

- a FastAPI backend in `src/api/server.py`
- Ollama-backed generation via `src/generation/ollama_client.py`
- lexical and semantic retrieval in `src/retrieval/`
- document ingestion and chunking in `src/ingestion/`
- conversation memory in `src/memory/`
- a frontend prototype in `frontend/`
- notebooks and experiments in `notebooks/`

## System setup

### 1. Install Python

Use Python 3.11 or 3.12. On Windows, create a virtual environment from the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install Python dependencies

```powershell
pip install -r requirements.txt
```

Optionally install the package in editable mode:

```powershell
pip install -e .
```

### 3. Install and run Ollama

This project uses `ollama` for local model generation. Make sure the Ollama runtime is installed and available on your PATH.

```powershell
ollama serve
ollama pull gemma3:4b
```

If you prefer a different local model, update `model` in requests or notebook cells.

### 4. Run the backend API

From the project root:

```powershell
python run_api.py
```

The backend listens on `http://localhost:8000` by default.

### 5. Run the frontend prototype

In a second terminal:

```powershell
cd frontend
python -m http.server 4173
```

Open `http://localhost:4173` in your browser.


## Quick start

- Start Ollama
- Start the backend with `python run_api.py`
- Serve the frontend from `frontend/`
- Open the UI at `http://localhost:4173`

## Notes

- The backend exposes `/api/ask` and `/api/compare`.
- The API uses uploaded documents and retrieval pipelines for hybrid study companion behavior.
- If Ollama is not available, API generation may fail and return an error-safe message.
- Use the notebooks in `notebooks/` for evaluation and experiments.
