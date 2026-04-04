# Frontend Prototype

A distinctive editorial "Signal Desk" UI for the HKBU Study Companion workflow.

## What it includes

- Query composer with mode selector (baseline / BM25 / vector / hybrid)
- Optional file upload list preview (ready to connect to backend upload API)
- Output panel with citations and token estimate
- Quality snapshot bars (BLEU / ROUGE-L placeholders)
- Strong visual identity (custom typography, atmospheric backgrounds, asymmetrical cards)
- Meaningful motion (staggered entry + interactive feedback)
- Responsive layout for desktop and mobile

## Run locally

From project root:

```powershell
# Terminal 1: backend graph API
python run_api.py

# Terminal 2: frontend static server
cd frontend
python -m http.server 4173
```

Then open:

- http://localhost:4173

Backend API default:

- http://localhost:8000

## Integration notes

- Frontend now calls `POST /api/ask` on `http://localhost:8000` by default.
- To change backend URL in browser console:
	- `localStorage.setItem("hkbu_api_base", "http://localhost:8000")`
	- refresh page.
- Mock corpus is OFF by default in real use. To enable it only for demo/testing:
	- `localStorage.setItem("hkbu_use_mock_corpus", "true")`
	- refresh page.
- To force real-case behavior again:
	- `localStorage.removeItem("hkbu_use_mock_corpus")`
	- refresh page.
- If backend is down, frontend falls back to local demo generation.
