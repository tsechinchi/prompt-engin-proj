# HKBU Study Companion — Architecture

## System Overview

HKBU Study Companion is a **Retrieval-Augmented Generation (RAG) system** with human-in-the-loop (HITL) approval. It combines document ingestion, multi-modal retrieval, LLM generation, and quality evaluation into a single **LangGraph-orchestrated pipeline**.

### Core Philosophy

- **Modularity**: Each component (ingestion, retrieval, generation, evaluation) is independent and testable
- **Robustness**: Multiple fallbacks at every critical layer (PDF extraction, embeddings, generation modes)
- **Transparency**: Token counting, retrieval scores, and quality metrics tracked throughout
- **User Control**: HITL loop allows approval, rejection, and regeneration before output

---

## System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INPUT                                  │
│  (Question + Mode + Optional Documents)                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────┐
        │   INGESTION PIPELINE                         │
        │  (loader.py → chunker.py)                    │
        │  - Load PDFs/TXT/MD                          │
        │  - Sliding window chunking (200 tok window)  │
        └──────────────────────────┬───────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────┐
        │   RETRIEVAL PIPELINE (graph.py - retrieve)   │
        │  ┌─────────────────────────────────────────┐ │
        │  │ BM25 Retriever (lexical search)         │ │
        │  │ • Tokenizes query                       │ │
        │  │ • Ranks by BM25 scores                  │ │
        │  └─────────────────────────────────────────┘ │
        │  ┌─────────────────────────────────────────┐ │
        │  │ Vector Retriever (semantic search)      │ │
        │  │ • Encodes query with sentence-transformers
        │  │ • FAISS cosine similarity                │ │
        │  └─────────────────────────────────────────┘ │
        └──────────────────────────┬───────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────┐
        │   RANKING & AGGREGATION (graph.py - aggregate)
        │  (hybrid_ranker.py - fuse_scores)           │
        │  • Normalize BM25 scores [0,1]              │
        │  • Normalize vector scores [0,1]            │
        │  • Fuse: 0.4×BM25 + 0.6×vector              │
        │  • Return top-k results                      │
        └──────────────────────────┬───────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────┐
        │   RETRIEVAL ASSESSMENT (graph.py - assess)  │
        │  • Check: min_fused_score threshold         │
        │  • Check: query-term overlap in snippets    │
        │  • Decision: continue or abstain             │
        └──────────────────────────┬───────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
         MATCH                                    MISMATCH
              │                                         │
              ▼                                         ▼
     ┌─────────────────┐                   ┌────────────────────┐
     │  Assemble       │                   │  Retry Retrieval   │
     │  Prompt         │                   │  (if retries left) │
     │  (assembler.py) │                   │  or Abstain        │
     └────────┬────────┘                   └────────┬───────────┘
              │                                     │
              └─────────────────┬───────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────────┐
        │   PROMPT ENGINEERING (graph.py - assemble)  │
        │  (assembler.py)                             │
        │  Assemble:                                  │
        │  • Role: "Helpful study companion"          │
        │  • Task: User query                         │
        │  • Context: Retrieved snippets              │
        │  • Constraints: Domain rules                │
        │  • Output format: Citation style            │
        └──────────────────────────┬───────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────┐
        │   GENERATION (graph.py - generate)          │
        │  (ollama_client.py)                         │
        │  Tier 1: Ollama raw mode                    │
        │  Tier 2: Ollama non-raw mode                │
        │  Tier 3: Ollama chat mode                   │
        │  Tier 4: Mock generation (fallback)         │
        │  Controls: temperature, num_predict         │
        └──────────────────────────┬───────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────┐
        │   QUALITY & RETRIEVAL GUARD (graph.py)      │
        │  (quality_eval.py, token_tracker.py)        │
        │  • BLEU score                               │
        │  • ROUGE-1/L scores                         │
        │  • Token overlap metrics                    │
        │  • Total token count                        │
        │  • Abstention check                         │
        └──────────────────────────┬───────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────┐
        │   HUMAN-IN-THE-LOOP (graph.py - hitl)       │
        │  (hitl.py)                                  │
        │  User decides:                              │
        │  • [A]pprove → output final answer          │
        │  • [R]eject → discard                       │
        │  • [G]enerate → refine & retry              │
        └──────────────────────────┬───────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
         APPROVE             REGENERATE            REJECT
              │                    │                    │
              └────────────────────┴────────────────────┤
                                                        │
                                   ▼
        ┌──────────────────────────────────────────────┐
        │   OUTPUT NODE (graph.py - output)           │
        │  Prepare response:                          │
        │  • Final answer text                        │
        │  • Status flag                              │
        │  • Quality metrics                          │
        │  • Citations from context                   │
        └──────────────────────────┬───────────────────┘
                                   │
                                   ▼
                        ┌──────────────────┐
                        │   RESPONSE       │
                        │  (JSON for API)  │
                        └──────────────────┘
```

---

## Module Breakdown

### 1. Ingestion (`src/ingestion/`)

**Purpose:** Load and normalize documents from various sources.

#### `loader.py` (170 lines)
- **Inputs:** File paths (PDF, TXT, MD)
- **Outputs:** `LoadedDocument` TypedDict with text + metadata
- **Key Features:**
  - PDF extraction: `pypdf` (primary) → `pdfplumber` (fallback)
  - Per-page tracking with `page_number` metadata
  - Text normalization (whitespace collapse)
  - Document ID generation (SHA1 hash)
  
```python
LoadedDocument = {
    "text": "Document content...",
    "metadata": {
        "source_path": "/data/syllabus.pdf",
        "source_name": "syllabus.pdf",
        "source_type": "pdf",
        "document_id": "abc123def456...",
        "page_number": 1
    }
}
```

#### `chunker.py` (209 lines)
- **Inputs:** `LoadedDocument` list
- **Outputs:** `ChunkRecord` list (chunks with extended metadata)
- **Algorithm:**
  1. Split text into sentences (regex on `.!?`)
  2. Build "chunk units" (sentence + token count)
  3. Assemble units into windows respecting `max_tokens`:
     - If sentence > window: split with sliding window overlap
     - If sentence fits: add to current chunk
  4. Apply stride (~50 tokens overlap by default)
  5. Preserve metadata: add `chunk_id`, `chunk_index`, `token_count`

**Why sliding window with sentence boundaries?**
- Preserves semantic completeness
- Avoids mid-sentence chunks
- Enables overlap for context continuity
- Token counting via tiktoken (with fallback)

#### `file_upload.py` (76 lines)
- **Purpose:** Handle file uploads from frontend
- **Features:**
  - Filename sanitization (removes dangerous chars)
  - Unique destination handling (appends `_1`, `_2` if file exists)
  - Batch ingestion support

---

### 2. Retrieval (`src/retrieval/`)

**Purpose:** Find relevant document chunks for a query.

#### `bm25_retriever.py` (109 lines)
- **Algorithm:** BM25 (Okapi variant)
- **Two-tier fallback:**
  1. Try `rank-bm25` library
  2. Fall back to manual BM25 implementation (if library unavailable)
- **Tokenization:** Lowercase, regex-based word tokenization
- **Scoring:** BM25 formula with k1=1.5, b=0.75
- **Output:** List of (document_text, score) tuples

#### `vector_retriever.py` (118 lines)
- **Embedder:** Sentence-Transformers (`all-MiniLM-L6-v2`, 384 dims)
- **Index:** FAISS (IndexFlatIP for inner-product/cosine)
- **Two-tier fallback:**
  1. Try sentence-transformers
  2. Fall back to hash-based embeddings (deterministic, no model)
- **Search:** FAISS `.search()` or manual cosine similarity
- **Normalization:** L2 norm applied to all embeddings
- **Output:** List of (document_text, score) tuples

#### `hybrid_ranker.py` (59 lines)
- **Fusion Strategy:**
  ```
  fused_score = (0.4 × norm_bm25) + (0.6 × norm_vector)
  ```
- **Normalization:** Min-max to [0, 1] per retriever
- **Tie-breaking:** Uses original retrieval order
- **Output:** Sorted list of (document_text, fused_score)

**Design Choice:** Why 0.4 BM25 + 0.6 vector?
- Vector embeddings capture semantic similarity better
- BM25 catches exact term matches (important for factual Q&A)
- Empirically, 0.6 semantic + 0.4 lexical performs well for study questions

---

### 3. Prompt & Generation (`src/prompt/`, `src/generation/`)

#### `assembler.py` (46 lines)
- **Purpose:** Build structured prompts from components
- **Template:**
  ```
  Role:
  [user-provided role]

  Task:
  [user query]

  Context Snippets:
  1. [retrieved snippet 1]
  2. [retrieved snippet 2]
  ...

  Constraints:
  - [constraint 1]
  - [constraint 2]
  - [reviewer feedback, if regenerating]

  Output Format:
  [desired format]
  ```
- **Design:** Separates prompt logic from content, enabling reuse

#### `ollama_client.py` (150 lines)
- **Three-tier generation fallback:**
  1. **Raw mode**: `ollama.generate(raw=True)` — no system prompt injection
  2. **Non-raw mode**: `ollama.generate(raw=False)` — standard generation
  3. **Chat mode**: `ollama.chat()` — structured conversation API
- **Parameters:**
  - `temperature`: Control randomness (0.3 default for factual answers)
  - `num_predict`: Max output tokens (200 default)
- **Fallback:** If all modes fail, return `[Model returned empty response]`

**Why three tiers?**
- Different Ollama versions support different modes
- Raw mode gives most control, chat most compatibility
- Ensures *something* returns to prevent crashes

#### `templates.py` (8 lines)
- **Current state:** Minimal (only default template)
- **Purpose:** Extensible template storage for future role customization

---

### 4. Memory (`src/memory/`)

#### `conversation.py` (83 lines)
- **Purpose:** Rolling conversation buffer with token budgets
- **Data Structure:**
  ```python
  ConversationMessage = {
      "role": "user" | "assistant" | "system",
      "content": "message text"
  }
  ```
- **Truncation Strategy:** Two-fold
  1. **Count-based:** Drop oldest non-system messages if `len(messages) > max_messages` (default 12)
  2. **Token-based:** Drop oldest non-system messages if token count > `max_tokens` (default 1200)
- **Token Counting:** Via `tiktoken.get_encoding("cl100k_base")` with fallback to basic tokenizer
- **Why preserve system messages?** System instructions should never be trimmed

---

### 5. Agent Orchestration (`src/agent/`)

#### `graph.py` (376 lines) — The Heart of the System

**LangGraph StateGraph with 8 nodes + conditional routing:**

```
START
  │
  ├─→ retrieve (build BM25 + Vector indices, query both)
  │
  ├─→ aggregate (fuse scores, select top-k)
  │
  ├─→ assess_retrieval (check quality thresholds)
  │     ├─→ MISMATCH?
  │     │   └─→ retrieval_guard
  │     │       ├─→ Retry? → assemble (and loop back)
  │     │       └─→ MaxRetries? → abstain
  │     │
  │     └─→ MATCH → continue
  │
  ├─→ assemble (build prompt with role/task/context/constraints)
  │
  ├─→ generate (call Ollama or mock)
  │
  ├─→ postprocess (strip whitespace)
  │
  ├─→ retrieval_guard (final mismatch check)
  │
  ├─→ hitl (human approval/rejection/regeneration)
  │     ├─→ REGENERATE? → assemble (loop back)
  │     ├─→ APPROVE → output
  │     └─→ REJECT → output (with rejected status)
  │
  ├─→ output (prepare response)
  │
  └─→ END
```

**State Fields (AgentState TypedDict):**
- Input: `query`, `chunk_records`, `mode`, `temperature`, `model`
- Retrieval outputs: `bm25_hits`, `vector_hits`, `fused_hits`, `context_snippets`
- Generation: `prompt`, `generated_text`, `final_output`
- Control: `require_approval`, `max_regenerations`, `max_retrieval_retries`
- Quality: `retrieval_mismatch`, `retrieval_mismatch_reason`, `status`

**Retrieval Assessment Logic:**
```python
mismatch = False
if top_fused_score < min_fused_score:  # 0.15 default
    mismatch = True
if best_query_term_overlap < min_query_term_overlap:  # 1 default
    mismatch = True
```

**Abstention Message:**
```
"I do not have enough relevant context to answer confidently. 
Please provide more relevant documents or rephrase the question."
```

#### `hitl.py` (51 lines)
- **CLI-based human-in-the-loop:**
  ```
  Generated Output:
  [text]

  Approve, reject, or regenerate? [a/r/g]: 
  ```
- **Outputs:** `HITLDecision` with `action` + optional `feedback`
- **Feedback loop:** For regeneration, feedback added as constraint to next prompt

---

### 6. Evaluation (`src/evaluation/`)

#### `quality_eval.py` (88 lines)
- **Metrics Computed:**
  - **BLEU:** Sentence-level BLEU with smoothing (via NLTK)
  - **ROUGE-1/L:** Unigram + longest common subsequence (via rouge_score)
  - **Token Overlap:** Precision, Recall, F1 on token sets
  - **Exact Match:** Binary (reference == prediction)
  - **LLM Judge:** Optional custom scoring function
- **Range:** All metrics normalized to [0, 1]

```python
score_answer(
    reference="ground truth answer",
    prediction="model output",
    judge_fn=optional_llm_judge  # takes (ref, pred) and returns float [0,1]
)
# Returns: {
#     "bleu": 0.45,
#     "rouge1_f": 0.67,
#     "rougeL_f": 0.62,
#     "token_precision": 0.85,
#     "token_recall": 0.72,
#     "token_f1": 0.78,
#     "exact_match": 0.0,
#     "llm_judge": 0.8  # if judge_fn provided
# }
```

#### `token_tracker.py` (13 lines)
- **Simple passthrough:**
  ```python
  track_usage(prompt_tokens=N, completion_tokens=M)
  # Returns: {
  #     "prompt_tokens": N,
  #     "completion_tokens": M,
  #     "total_tokens": N + M
  # }
  ```
- **Purpose:** Hook point for future token logging/cost tracking

---

### 7. Tools (`src/agent/tools.py`)

#### External Integration (70 lines)
- **`fetch_live_page(url, selector, timeout_ms, max_chars)`**
  - Uses Playwright headless browser
  - Extracts text from CSS selector
  - Returns: `"Title: ...\nURL: ...\n\ntext..."`
- **`fetch_hkbu_updates(timetable_url, news_url)`**
  - Wrapper for fetching both timetable and news in parallel
  - Returns: `{"timetable": "...", "news": "..."}`

**Design:** Lazy imports (only import Playwright if needed) to avoid bottleneck

---

### 8. API Server (`src/api/`)

#### `server.py` (334 lines)
- **Framework:** FastAPI with CORS middleware (allow all origins for demo)
- **Endpoints:**
  - `GET /api/health` → `{"status": "ok"}`
  - `POST /api/ask` ← `AskRequest`

**Request Schema:**
```python
AskRequest = {
    "query": str,
    "mode": "baseline" | "bm25" | "vector" | "hybrid",
    "temperature": float = 0.3,
    "top_k": int = 5,
    "model": str = "gemma3:4b",
    "uploaded_docs": [{"name": "...", "text": "..."}],
    "use_mock_generation": bool = True,
    "use_mock_corpus": bool = False
}
```

**Response Schema:**
```python
AskResponse = {
    "answer": str,  # final output
    "status": "approved" | "abstained" | "rejected",
    "citations": [str],  # [0..4] snippets from context
    "quality": {"bleu": float, "rouge_l": float},
    "tokens": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
    "radar_snippets": [{"snippet": str, "source": str}]
}
```

**Mode Weighting:**
```python
mode_weights = {
    "baseline": (0, 0),      # no retrieval
    "bm25": (1.0, 0.0),      # lexical only
    "vector": (0.0, 1.0),    # semantic only
    "hybrid": (0.4, 0.6)     # 0.4×BM25 + 0.6×vector
}
```

**Mock Generation Fallback:**
- Extracts snippets from assembled prompt
- Generates human-readable summary
- Falls back to mock if Ollama unavailable

---

## Frontend Architecture (`frontend/`)

### Stack
- **HTML5** + **CSS3** + **Vanilla JavaScript** (no framework)
- **PDF.js** – PDF file parsing
- **Mammoth.js** – DOCX parsing  
- **JSZip** – ZIP/PPTX handling

### UI Components
- **Query Composer** (textarea + mode select + temperature slider)
- **Upload Manager** (file list with metadata)
- **Output Panel** (answer display with citations)
- **Live Radar** (quality metrics + source signals)

### Client-Side Retrieval
- Implements local BM25 + snippet matching
- Falls back to backend `/api/ask` call if documents provided

---

## Data Flow Example: A Query

### Scenario: Student asks "What are CS101 exam dates?"

1. **Frontend receives query** → `"What are CS101 exam dates?"`

2. **POST /api/ask** with:
   - `mode`: "hybrid"
   - `uploaded_docs`: [] (using mock corpus)
   - `use_mock_corpus`: true

3. **Server chains:**
   - Loads mock documents from `data/mock/`
   - Chunks them (200 tok windows)
   - Builds BM25 + Vector indices
   - Queries: BM25 finds "CS101 exam dates: June 15, 2024"
   - Queries: Vector finds "Final assessment occurs in June..."
   - **Fusion:** Scores and combines results
   - **Assessment:** Query terms "exam", "dates" found in top result → PASS
   - **Assemble:** Creates prompt with role + context
   - **Generate:** Ollama produces answer
   - **Evaluation:** Calculates BLEU/ROUGE
   - **HITL:** Auto-approves (or waits for human)
   - **Output:** Returns answer + citations

4. **Frontend displays:**
   ```
   Q: What are CS101 exam dates?
   A: CS101 final exam is scheduled for June 15, 2024...
   
   Citations:
   - Context snippet 1: "Final exam occurs in June 2024..."
   
   Quality: BLEU 0.67, ROUGE-L 0.62
   Tokens: 45 prompt + 87 completion
   ```

---

## Key Design Decisions

### 1. Why Sliding Window Chunking?
- Preserves semantic coherence (sentence boundaries)
- Enables overlap (~50 tokens) for context continuity
- Avoids mid-sentence cuts

### 2. Why Hybrid Retrieval (BM25 + Vector)?
- **BM25 (0.4):** Catches exact terminology matches (dates, names, codes)
- **Vector (0.6):** Captures semantic intent ("when are exams" → "exam dates")
- Together: Better recall + precision than either alone

### 3. Why Multiple Fallback Tiers?
- **Robustness:** System keeps working even if libraries fail
- **Graceful degradation:** Falls back to simpler algorithms
- **Production-ready:** No single point of failure

### 4. Why HITL?
- **Ensures quality:** Humans can catch hallucinations/mismatches
- **Enables regeneration:** Feedback improves subsequent attempts
- **Compliance:** Some domains require human approval before output

### 5. Why Mock Generation?
- **Frontend works offline:** Can test without Ollama/API
- **Development speed:** Don't need running LLM locally
- **Fallback:** If Ollama fails, system still responds

---

## Dependency Graph

```
frontend/app.js
├─→ /api/health
└─→ /api/ask
    └─→ src.api.server:app
        ├─→ src.agent.graph:build_graph
        │   ├─→ src.retrieval.bm25_retriever:BM25Retriever
        │   ├─→ src.retrieval.vector_retriever:VectorRetriever
        │   ├─→ src.retrieval.hybrid_ranker:fuse_scores
        │   ├─→ src.prompt.assembler:assemble_prompt
        │   ├─→ src.generation.ollama_client:generate_raw
        │   └─→ src.agent.hitl:review_output
        ├─→ src.ingestion.loader:load_documents
        ├─→ src.ingestion.chunker:chunk_documents
        └─→ src.evaluation.quality_eval:score_answer
```

---

## Testing Strategy

### Unit Tests (37 total)
- **Ingestion (8 tests):** loader, chunker, file upload
- **Retrieval (5 tests):** BM25, Vector, Hybrid fusion
- **Prompt/Generation/Memory (5 tests):** Assembly, Ollama, ConversationBuffer
- **Agent (7 tests):** Graph routing, HITL approval, regeneration, abstention
- **API (2 tests):** Health check, ask endpoint
- **Tools (2 tests):** URL validation, fetch helpers
- **Evaluation (3 tests):** Quality metrics, token tracking

### Integration Tests
- Full pipeline: query → retrieval → generation → approval → output
- Fallback chains: Each tier actually works
- Mode switching: baseline vs BM25 vs vector vs hybrid

---

## Performance Characteristics

| Component | Complexity | Latency | Notes |
|-----------|-----------|---------|-------|
| Load 1 document | O(1) | <100ms | PDF extraction |
| Chunk 1 document | O(n) | ~50ms | n = text length |
| Build BM25 index | O(n·m) | ~100ms | n = docs, m = vocab |
| Build Vector index | O(n·d) | ~200ms | n = docs, d = embedding dim |
| BM25 query | O(m) | <10ms | m = vocab size |
| Vector query | O(n·d) | ~5ms | FAISS is fast |
| Generate answer | O(output_tokens) | ~2-5s | LLM latency dominates |
| Total pipeline | | ~3-7s | Dominated by generation |

---

## Future Extensions

1. **Persistent Vector Index:** Cache embeddings to skip recomputation
2. **Streaming Generation:** Return tokens as they arrive (WebSocket)
3. **Multi-Turn Conversation:** Use `ConversationBuffer` for context
4. **Custom LLM Evaluation:** Implement LLM-as-judge for quality scoring
5. **Cross-lingual RAG:** Multi-language embedding models
6. **Async Pipeline:** Non-blocking retrieval + generation
7. **Analytics:** Log queries, metrics, user feedback for analysis

---

## Deployment

### Docker Deployment
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["python", "run_api.py"]
```

### Environment Variables
```bash
OLLAMA_HOST=http://localhost:11434  # Ollama API endpoint
HKBU_CORPUS_PATH=./data/mock        # Document source
MAX_CHUNK_TOKENS=200                # Chunking window size
```

### Scaling Considerations
- **Single-instance:** Works fine for <1000 docs, <10 concurrent users
- **Multi-instance:** Stateless API, scale horizontally with load balancer
- **Vector index:** Make persistent (e.g., Pinecone, Qdrant) for multi-instance

---

## Security Considerations

1. **Input Validation:** Pydantic models validate all API inputs
2. **File Upload:** Filename sanitization prevents path traversal
3. **CORS:** Currently allow-all for demo; restrict in production
4. **Ollama:** Assumes trusted local network; don't expose untrusted
5. **LLM Prompt Injection:** Use role/task/context separation to mitigate

