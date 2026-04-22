# 🌳 Tree-RAG — Tree-Based Vector-Less Document Q&A

A production-oriented document Q&A system where PDFs are parsed into a **4-level hierarchical tree** (Document → Section → Subsection → Leaf) and queries are answered by **LLM-guided BFS tree traversal with multi-hop reasoning** — no vector embeddings, no vector database.

---

## Table of Contents

- [Why Tree-Based RAG?](#why-tree-based-rag)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Data Flow](#data-flow)
  - [Upload](#1-upload)
  - [Build Index](#2-build-index)
  - [Query](#3-query--multi-hop-reasoning)
- [API Reference](#api-reference)
- [Streamlit UI](#streamlit-ui)
- [Configuration](#configuration)
- [Setup & Running](#setup--running)
- [Tech Stack](#tech-stack)

---

## Why Tree-Based RAG?

Standard vector RAG flattens every document into equal-weight chunks and retrieves by cosine similarity. This works for simple lookups but fails at:

| Problem | Vector RAG | Tree-RAG |
|---------|-----------|----------|
| Multi-section questions | Misses links between sections | Multi-hop traversal follows document structure |
| Structural awareness | Loses hierarchy | 4-level tree preserves Doc → Section → Subsection |
| Incomplete answers | No recovery mechanism | Built-in retry loop with query refinement |
| Explainability | Black-box similarity scores | Full audit trail: every scoring decision logged |
| Metadata filtering | Post-retrieval (wasteful) | Pre-retrieval (deterministic, fast) |
| No embedding GPU | Requires model + compute | Zero embeddings — pure LLM reasoning |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit UI (port 8501)                │
│  Sidebar: Upload · Build · Filters                       │
│  Tabs:    Chat · Tree View · History                     │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP (httpx)
┌──────────────────────▼──────────────────────────────────┐
│                  FastAPI (port 8000)                     │
│  POST /upload        POST /index/build                   │
│  POST /index/refresh GET  /index/status                  │
│  GET  /index/tree    POST /query                         │
│  GET  /history       DELETE /history                     │
└──────┬───────────────────────┬──────────────────────────┘
       │                       │
┌──────▼───────┐     ┌─────────▼────────────────────────┐
│  Ingestion   │     │       Reasoning (LangGraph)       │
│  pdf_parser  │     │  decompose → retrieve →           │
│  section_    │     │  sufficiency → re_retrieve →      │
│  detector    │     │  generate_answer                  │
│  metadata_   │     └─────────┬────────────────────────┘
│  extractor   │               │
└──────┬───────┘     ┌─────────▼────────────────────────┐
       │             │     Retrieval (BFS traversal)     │
       │             │  score all siblings in 1 LLM call │
       │             │  prune branches scoring < 3/5     │
       │             └─────────┬────────────────────────┘
       │                       │
┌──────▼───────────────────────▼──────────────────────────┐
│                    Custom Tree Index                     │
│  data/parsed/*.json  →  data/index_store/tree.json       │
│  TreeNode: leaf | subsection | section | document        │
│  Each node: full text + LLM summary + metadata           │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
tree_rag_project/
├── app/
│   ├── main.py                    # FastAPI entry point, CORS, router registration
│   ├── config.py                  # Pydantic settings, all paths & model config
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response models
│   ├── routes/
│   │   ├── upload.py              # POST /upload
│   │   ├── index.py               # POST /index/build, /refresh, GET /status, /tree
│   │   ├── query.py               # POST /query
│   │   └── history.py             # GET/DELETE /history
│   └── services/
│       ├── ingestion_service.py   # PDF → parsed JSON pipeline
│       ├── tree_index_service.py  # Build & load CustomTreeIndex
│       ├── retrieval_service.py   # BFS traversal + batch LLM scoring
│       ├── reasoning_service.py   # LangGraph multi-hop reasoning graph
│       └── history_service.py     # Read/write query history
├── ui/
│   └── streamlit_app.py           # Full Streamlit frontend
├── utils/
│   ├── tree_model.py              # TreeNode + CustomTreeIndex dataclasses
│   ├── pdf_parser.py              # PyPDFLoader + section-aware chunking
│   ├── section_detector.py        # Regex → LLM → page fallback heading detection
│   ├── metadata_extractor.py      # Attach doc_name, timestamp, tags to chunks
│   ├── chunk_summarizer.py        # Optional per-chunk summarization helper
│   ├── prompt_templates.py        # All LLM prompt strings
│   └── logging_utils.py           # Shared logger (YYYY-MM-DD HH:MM:SS | LEVEL | MODULE)
├── data/
│   ├── uploads/                   # Raw uploaded PDFs
│   ├── parsed/                    # Enriched chunk JSONs (one file per document)
│   └── index_store/
│       └── tree.json              # Persisted CustomTreeIndex
├── tests/
├── .env                           # GOOGLE_API_KEY (not committed)
└── requirements.txt
```

---

## Data Flow

### 1. Upload

```
User uploads PDF (+ optional tags) in Streamlit sidebar
          │
          ▼
POST /upload
          │
          ▼
ingest_document(file_bytes, filename, tags)
  │
  ├─ Write raw PDF → data/uploads/<filename>
  │
  ├─ pdf_parser.parse_pdf(file_path)
  │    └─ PyPDFLoader extracts all pages
  │    └─ section_detector.detect_sections(pages)
  │         ├─ Pass 1 — Regex: count headings across pages
  │         ├─ If ≥ 2 headings → regex-based section chunking
  │         ├─ Else → LLM fallback: Gemini infers structure (up to 12k chars)
  │         └─ Else → page fallback: one chunk per page, "Page N" labels
  │         → Returns chunks with: text, section, subsection, page_range
  │
  ├─ metadata_extractor.attach_metadata(chunks, filename, tags)
  │    └─ Adds: doc_name, upload_timestamp (UTC ISO-8601), tags[]
  │
  └─ Write enriched chunks → data/parsed/<stem>.json
```

No LLM calls happen at upload time. Upload is fast.

---

### 2. Build Index

```
User clicks "Build Index" or "Rebuild Index" in sidebar
          │
          ▼
POST /index/build  (or /index/refresh)
          │
          ▼
Read all data/parsed/*.json → flat chunk list
          │
          ▼
tree_index_service.build_index(chunks)
  │
  └─ _build_hierarchical_tree(chunks)
       │
       ├─ Group chunks: { doc → { section → { subsection → [chunks] } } }
       │
       ├─ LEAF NODES  (1 LLM call per chunk)
       │    └─ _summarize_leaf(text, section, doc_name)
       │    └─ TreeNode(node_type="leaf", text=full_text, summary=2-3 sentences, children=[])
       │
       ├─ SUBSECTION NODES  (0 LLM calls — concat leaf summaries)
       │    └─ TreeNode(node_type="subsection", text=concat_summaries, children=[leaf_ids])
       │
       ├─ SECTION NODES  (1 LLM call per section)
       │    └─ _summarize_group(subsection_texts, label)
       │    └─ TreeNode(node_type="section", summary=3-5 sentences, children=[subsection_ids])
       │
       └─ DOCUMENT ROOT NODES  (1 LLM call per document)
            └─ _summarize_group(section_texts, doc_name)
            └─ TreeNode(node_type="document", summary=3-5 sentences, children=[section_ids])
                           → Added to root_ids
       │
       ▼
CustomTreeIndex.save(data/index_store/tree.json)
  └─ JSON: { root_ids: [...], nodes: { uuid: {node_id, node_type, text, summary, metadata, children} } }
```

Build stats returned: `node_count`, `tokens_used`, `llm_calls` — displayed in UI sidebar.

---

### 3. Query & Multi-hop Reasoning

```
User types question (+ optional filters) in Chat tab
          │
          ▼
POST /query
          │
          ▼
reasoning_service.run_reasoning(query, document_filter, tags_filter)
          │
          ▼ LangGraph State Machine
          │
  ┌───────▼────────────────────────────────────────────────────┐
  │ [1] DECOMPOSE                                               │
  │  Gemini splits query into 1-3 focused sub-questions         │
  │  e.g. "What challenges exist and how are they addressed?" → │
  │       ["What are the key challenges?",                      │
  │        "How are those challenges addressed?"]               │
  └───────────────────────────┬────────────────────────────────┘
                              │
  ┌───────────────────────────▼────────────────────────────────┐
  │ [2] RETRIEVE                                                │
  │  For each sub-question:                                     │
  │    BFS from root_ids through tree:                          │
  │    ┌─────────────────────────────────────────────┐         │
  │    │  frontier = [root_node_1, root_node_2, ...]  │         │
  │    │  LOOP:                                        │         │
  │    │    _score_batch(frontier, query)              │         │
  │    │    → ONE LLM call scores all siblings:        │         │
  │    │      [{"index":0,"score":4}, {"index":1,      │         │
  │    │        "score":1}, ...]                       │         │
  │    │    score ≥ 3 + has children → add to frontier │         │
  │    │    score ≥ 3 + leaf        → collect as evidence       │
  │    │    score < 3               → prune branch     │         │
  │    └─────────────────────────────────────────────┘         │
  │  Deduplicate evidence by node_id                            │
  └───────────────────────────┬────────────────────────────────┘
                              │
  ┌───────────────────────────▼────────────────────────────────┐
  │ [3] SUFFICIENCY CHECK                                       │
  │  Gemini judges: "Is this context enough to answer?"         │
  │  Returns: { sufficient: bool, reasoning: string }           │
  └───────────┬───────────────────────────┬────────────────────┘
              │ sufficient                │ insufficient
              │                           │
              │              ┌────────────▼──────────────────┐
              │              │ hop_count < max_hops (3)?     │
              │              │  YES → [4] RE-RETRIEVE        │
              │              │    Gemini refines query        │
              │              │    based on what's missing     │
              │              │    Loop back to [2]           │
              │              │  NO  → INSUFFICIENT_NODE      │
              │              │    Return "insufficient        │
              │              │    context" + best evidence   │
              │              └───────────────────────────────┘
              │
  ┌───────────▼────────────────────────────────────────────────┐
  │ [5] GENERATE ANSWER                                         │
  │  Gemini generates answer using only retrieved context       │
  └───────────────────────────┬────────────────────────────────┘
                              │
                              ▼
        QueryResponse { answer, status, evidence[], retrieval_trace[], tokens_used, llm_calls }
                              │
                              ▼
             history_service.add_entry()  [non-blocking]
```

---

## API Reference

### `POST /upload`
Upload and parse a PDF.

**Request:** `multipart/form-data`
- `file` — PDF binary (required)
- `tags` — comma-separated string, e.g. `"finance,2024"` (optional)

**Response:**
```json
{
  "filename": "report.pdf",
  "chunk_count": 12,
  "upload_timestamp": "2026-04-22T10:30:00.123Z",
  "message": "Successfully ingested 'report.pdf' (12 pages)."
}
```

---

### `POST /index/build`
Build the tree index from all parsed documents. Returns `already_exists` if an index is already on disk — use `/index/refresh` to force a rebuild.

**Response:**
```json
{
  "status": "built",
  "node_count": 145,
  "message": "TreeIndex built with 145 nodes.",
  "tokens_used": 2340,
  "llm_calls": 24
}
```

---

### `POST /index/refresh`
Force-rebuild the index. Re-parses all uploaded PDFs first, then rebuilds the tree.

**Response:** same as `/index/build` with `"status": "built"`.

---

### `GET /index/status`
```json
{ "index_exists": true, "node_count": 145 }
```

---

### `GET /index/tree`
Returns all nodes and edges for tree visualization.

```json
{
  "nodes": [
    {
      "id": "uuid",
      "is_root": true,
      "node_type": "document",
      "text": "...",
      "summary": "...",
      "section": "Introduction",
      "subsection": "",
      "doc_name": "report.pdf",
      "page_number": 1,
      "page_range": [1, 5]
    }
  ],
  "edges": [{ "from": "parent-uuid", "to": "child-uuid" }],
  "root_count": 2
}
```

---

### `POST /query`

**Request:**
```json
{
  "query": "What are the key findings?",
  "document_filter": "report.pdf",
  "tags_filter": ["2024", "finance"]
}
```

**Response:**
```json
{
  "query": "What are the key findings?",
  "answer": "The key findings are...",
  "status": "success",
  "evidence": [
    {
      "node_id": "uuid",
      "summary": "This section discusses...",
      "metadata": {
        "doc_name": "report.pdf",
        "section": "Results",
        "page_range": [4, 6],
        "relevance_score": 5
      }
    }
  ],
  "retrieval_trace": [
    { "step": 1, "action": "metadata_filter", "notes": "Decomposed into 2 sub-questions..." },
    { "step": 2, "action": "tree_traversal", "notes": "Retrieved 5 unique nodes..." },
    { "step": 3, "action": "sufficiency_check", "notes": "sufficient=true. Context directly answers..." }
  ],
  "tokens_used": 1240,
  "llm_calls": 5
}
```

Status values: `success` | `insufficient_context` | `error`

---

### `GET /history?limit=50`
Returns the most recent N queries.

### `DELETE /history`
Clears all query history.

### `GET /health`
```json
{ "status": "ok" }
```

---

## Streamlit UI

### Header Bar
Four metrics updated each session:
- **Queries this session**
- **Tokens used this session**
- **LLM API calls this session**
- **Index status** — `Ready — N nodes` (green) or `Not built` (red)

---

### Sidebar

**Upload section**
- PDF file uploader (multiple uploads accumulate)
- Tags text field — `"finance, 2024, Q3"` — comma-separated
- **Upload** button → calls `POST /upload`, shows chunk count
- **Build Index** button → calls `POST /index/build`, shows node count
- **🔄 Rebuild Index** button → calls `POST /index/refresh`

**Index Status** — green ready indicator or warning

**Last Build** — 3-column metric panel showing:
- 🌳 Nodes built
- 🔤 Tokens consumed
- 🤖 LLM calls made

**Retrieval Filters**
- Document filter (text) — case-insensitive substring match on `doc_name`
- Tags filter (text) — comma-separated, AND logic (all tags must match)

---

### Tab 1 — 💬 Chat

- Chat history in Streamlit message bubbles (user / assistant alternating)
- Chat input: `"Ask a question about your documents…"`
- For each response:
  - Answer text (markdown rendered)
  - Status pill: `🟢 Success` / `⚠️ Insufficient context` / `🔴 Error`
  - Inline metrics: token count, LLM calls, evidence node count
  - **Evidence** expander — for each node: metadata (doc, section, pages, score) + summary
  - **Retrieval trace** expander — numbered steps with action icons:
    - 🏷️ `metadata_filter` — query decomposition
    - 🌳 `tree_traversal` — BFS retrieval pass
    - 🔄 `re_retrieve` — query refinement hop
    - ✅ `sufficiency_check` — LLM judgment

---

### Tab 2 — 🌳 Tree View

An interactive **Pyvis** network graph of the entire tree index.

- **Layout**: hierarchical, directed, top-down
- **Green circles** (28px) — document root nodes
- **Blue circles** (14px) — leaf, section, subsection nodes
- **Arrows** — parent → child edges, cubic Bezier, 1.5px grey
- **Hover tooltip** on each node shows: node type, section, subsection, page range, text preview
- Caption shows total node count, root count, leaf count
- **🔄 Refresh** button reloads from `/index/tree`

---

### Tab 3 — 🕐 History

- Shows all persisted queries (most recent first)
- Each row: timestamp + query preview (first 80 chars)
- Expand to see: status pill, tokens, calls, evidence count, full answer
- **Clear History** button → calls `DELETE /history`
- Stored in `data/history.json`, survives server restarts

---

## Configuration

All settings are in `app/config.py` and can be overridden via environment variables or a `.env` file in the project root.

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | *(required)* | Gemini API key |
| `model_name` | `gemini-1.5-flash` | LLM model for all calls |
| `upload_dir` | `data/uploads` | Raw PDF storage |
| `parsed_dir` | `data/parsed` | Parsed chunk JSON storage |
| `index_store_dir` | `data/index_store` | Tree JSON storage |
| `history_file` | `data/history.json` | Query history |
| `max_hops` | `3` | Max re-retrieval iterations |
| `api_base_url` | `http://localhost:8000` | FastAPI URL (used by Streamlit) |

Create a `.env` file:
```env
GOOGLE_API_KEY=your_key_here
model_name=gemini-1.5-flash
max_hops=3
```

---

## Setup & Running

### Prerequisites
- Python 3.11+
- A Google Gemini API key

### Install

```bash
git clone <repo>
cd tree_rag_project
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### Environment

```bash
# Create .env in project root
echo GOOGLE_API_KEY=your_key_here > .env
```

### Start the backend

```bash
uvicorn app.main:app --reload --reload-dir app --reload-dir utils --host 0.0.0.0 --port 8000
```

> Use `--reload-dir app --reload-dir utils` to avoid restart loops from `.venv` changes.

### Start the frontend

```bash
streamlit run ui/streamlit_app.py --server.port 8501
```

Open `http://localhost:8501`

### Typical workflow

1. **Upload** one or more PDFs using the sidebar
2. **Build Index** — wait for LLM summarization to complete (progress visible in logs)
3. **Ask questions** in the Chat tab
4. **Inspect the tree** in the Tree View tab
5. **Review history** in the History tab

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini (`gemini-1.5-flash`) via `langchain-google-genai` |
| Reasoning loop | LangGraph `StateGraph` |
| LLM orchestration | LangChain |
| PDF parsing | `langchain-community` PyPDFLoader |
| Tree model | Custom `TreeNode` + `CustomTreeIndex` dataclasses (plain JSON, no external dependency) |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Tree visualization | pyvis |
| Config management | Pydantic Settings |
| Data persistence | Plain JSON files on disk |

---

## Key Design Decisions

**No vector embeddings** — Retrieval is entirely LLM-based. The tree structure removes the need for semantic similarity search; the LLM scores structural relevance directly.

**Batch scoring at each tree level** — All sibling nodes at a frontier level are scored in a single LLM call. This minimizes API calls while enabling precise branch pruning.

**Upload vs Build separation** — Upload only parses and saves JSON (fast, no LLM). The Build step runs all LLM summarization. This means uploads are instant and you choose when to pay the LLM cost.

**Custom tree model** — `TreeNode` + `CustomTreeIndex` are plain Python dataclasses saved as JSON. No framework, no hidden state, fully inspectable and debuggable.

**Multi-fallback section detection** — Regex first (free), LLM fallback (handles messy PDFs), page fallback (guarantees a result). This handles both well-structured academic papers and unstructured reports.

**Full audit trail** — Every query returns a `retrieval_trace` with all steps (decomposition, scoring, sufficiency check, refinement hops) so results are explainable and debuggable.

  ├── routes/upload.py   ←  POST /upload
  ├── routes/index.py    ←  POST /index/build, /refresh, GET /status
  ├── routes/query.py    ←  POST /query
  └── routes/history.py  ←  GET/DELETE /history
        │
  services/
  ├── ingestion_service.py   ←  PDF save + parse + metadata
  ├── tree_index_service.py  ←  LlamaIndex TreeIndex build/load/persist
  ├── retrieval_service.py   ←  metadata filter + tree traversal
  ├── reasoning_service.py   ←  LangGraph multi-hop loop
  └── history_service.py     ←  in-memory + disk history

utils/
  ├── pdf_parser.py          ←  pdfplumber page extraction
  ├── metadata_extractor.py  ←  chunk enrichment
  ├── prompt_templates.py    ←  all LLM prompt strings
  └── logging_utils.py       ←  structured logger factory

data/
  ├── uploads/     ←  raw PDF files
  ├── parsed/      ←  per-doc JSON chunk files
  └── index_store/ ←  persisted LlamaIndex artefacts
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- An OpenAI API key

### 2. Install

```bash
cd tree_rag_project          # or wherever the repo lives
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Open .env and set OPENAI_API_KEY (and optionally MODEL_NAME, MAX_HOPS, …)
```

### 4. Run the backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The interactive API docs are available at http://localhost:8000/docs.

### 5. Run the Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

Open http://localhost:8501 in your browser.

### 6. Typical workflow

1. **Upload** a PDF via the sidebar (adds it to `data/uploads/` and saves parsed JSON to `data/parsed/`).
2. **Build Index** — calls `POST /index/build` to create the TreeIndex and persist it to `data/index_store/`.
3. **Ask a question** in the chat box.
4. The backend runs the LangGraph reasoning loop:
   - decompose query → tree traversal → sufficiency check → re-retrieve (up to `MAX_HOPS`) → answer
5. The response includes the **answer**, **evidence nodes**, and a **retrieval trace**.
6. All queries are saved to **Query History** at the bottom of the page.

## Running Tests

```bash
pytest tests/ -v
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI API key. |
| `MODEL_NAME` | `gpt-4o-mini` | OpenAI model to use for all LLM calls. |
| `UPLOAD_DIR` | `data/uploads` | Directory for raw PDF storage. |
| `PARSED_DIR` | `data/parsed` | Directory for parsed chunk JSON files. |
| `INDEX_STORE_DIR` | `data/index_store` | Directory for persisted TreeIndex artefacts. |
| `HISTORY_FILE` | `data/history.json` | Path for persistent query history. |
| `MAX_HOPS` | `3` | Maximum re-retrieval attempts before returning "insufficient context". |
| `API_BASE_URL` | `http://localhost:8000` | Base URL the Streamlit UI uses to reach the FastAPI backend. |



## To commplete
1. deepen the tree on subsections and check if it is hopping to child nodes
