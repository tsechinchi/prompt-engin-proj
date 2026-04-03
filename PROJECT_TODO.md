# HKBU Study Companion — Project TODO

### Ingestion + Retrieval  [MINIMUM] [DONE]
- [ ] loader.py — Load PDFs via pypdf/pdfplumber (syllabus, handbook, regulations)
- [ ] chunker.py — Sliding window chunks (~200 tok window, ~50 tok stride, sentence-boundary aware)
- [ ] bm25_retriever.py — rank-bm25 wrapper, index chunks, return top-k with scores
- [ ] vector_retriever.py — all-MiniLM-L6-v2 embeddings + FAISS flat index, cosine similarity
- [ ] hybrid_ranker.py — normalize both scores to [0,1], fuse 0.4×BM25 + 0.6×vector

### Prompt + Generation + Memory  [MINIMUM] [DONE, NEED REVIEW AND REFINE]
- [ ] assembler.py — Role + task + context snippets + constraints + output format
- [ ] ollama_client.py — ollama.generate(raw=True), justify temperature + num_predict in report
- [ ] conversation.py — Rolling message list, truncate oldest turns at token budget

### Orchestration + HITL  [MINIMUM — core to your architecture] [FINISHED]
- [ ] graph.py — LangGraph StateGraph: retrieve → aggregate → assemble → generate → postprocess → hitl → output
- [ ] hitl.py — CLI or widget approve/reject/regenerate loop

### Evaluation Notebooks  [MINIMUM]
- [ ] 01_baseline_no_rag.ipynb — 5–10 queries, no retrieval, log token counts
- [ ] 02_rag_pipeline.ipynb — BM25-only, vector-only, hybrid; cite retrieved snippets
- [ ] 03_evaluation.ipynb — No-RAG vs RAG, lexical vs neural, token usage charts

### Advanced (Optional +20%)
- [ ] tools.py — Playwright headless browser for live HKBU timetable/news
- [ ] quality_eval.py — BLEU/ROUGE or LLM-as-judge + token_tracker.py

### Report Workflow
- Draft the report in small sections first.
- Keep the final PDF as an export target, not the place where the writing happens.
- Organize content around problem statement, architecture, implementation, evaluation, and limitations.

