# Tree-RAG · Tree-Based Vector-Less Document QA

A production-oriented document Q&A application.  PDFs are parsed into a hierarchical **LlamaIndex TreeIndex** and queries are answered by LLM-guided tree traversal with multi-hop reasoning — no vector search required.

## Architecture

```
ui/streamlit_app.py      ←  browser-facing chat UI
        │  httpx
app/main.py              ←  FastAPI entry point
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
