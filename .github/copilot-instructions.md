# Instruction.md — Tree-Based Vector-Less RAG Project

## Objective
Build a production-oriented document Q&A application that can ingest structured documents such as PDFs, convert them into a hierarchical tree structure, and answer user questions by traversing that tree with multi-hop reasoning. The system must work **without vector search** as the primary retrieval mechanism.

The final product must include:
- A document upload flow
- Tree-based indexing and persistence on disk
- Metadata-based filtering before retrieval
- Multi-hop reasoning for complex questions
- A simple Streamlit UI for end users
- Query history visible in the UI
- Optional FastAPI backend for modularity

---

## Core Functional Requirements

### 1) Document ingestion
- Accept PDF uploads from the UI.
- Extract text from PDFs.
- Split content into structured chunks.
- Attach metadata to each chunk where possible, such as:
  - document name
  - upload timestamp
  - page number
  - section title
  - source file path
  - user-provided tags if available
- Store raw uploaded documents separately from processed index artifacts.

### 2) Tree index creation
- Build a hierarchical tree index from the parsed document content using LlamaIndex.
- Use the tree as the primary retrieval structure.
- Persist the tree index to disk using `index.storage_context.persist()`.
- Implement a load path so the app can start without rebuilding the tree every time.

### 3) Query processing
- Accept a natural-language query from the UI.
- First filter candidate content using metadata if applicable.
- Traverse the tree to identify the most relevant nodes.
- Support multi-hop reasoning when one retrieval pass is not enough.
- After each retrieval step, evaluate whether the context is sufficient.
- If insufficient, re-retrieve with a refined query or expanded search scope.
- If still insufficient after retrying, return an explicit “insufficient context” response along with the best available supporting content.

### 4) Reasoning orchestration
- Use LangGraph to orchestrate the reasoning loop.
- Use LangChain components where useful for prompt templates, LLM calls, and tool orchestration.
- The reasoning flow should support:
  - query decomposition
  - retrieval
  - context sufficiency check
  - retry / re-retrieval
  - final answer generation

### 5) User interface
- Build a simple Streamlit app for end users.
- UI must support:
  - document upload
  - asking questions
  - showing the final answer
  - showing retrieved tree nodes or evidence used
  - showing query history for the current user session
- The UI should feel like a real application, not a developer demo.

### 6) Backend
- Use FastAPI if it simplifies separation of concerns.
- Expose endpoints for:
  - upload document
  - build / refresh index
  - ask question
  - fetch query history
  - load persisted index
- The backend should be optional but preferred if it improves maintainability.

---

## Non-Functional Requirements
- Keep the architecture modular and easy to extend.
- Make the code readable and production-friendly.
- Separate indexing, retrieval, reasoning, and UI logic.
- Add logging for ingestion and query flow.
- Add basic error handling for file upload failures, parse failures, and missing index states.
- Avoid hardcoding environment-specific values.
- Use environment variables or a config file for model names, file paths, and API URLs.

---

## Recommended Architecture

### Components
1. **Streamlit frontend**
   - handles upload and chat UI
   - displays history and results

2. **FastAPI service**
   - handles document ingestion and query APIs
   - serves as the interface between UI and backend logic

3. **Indexing layer**
   - parses PDFs
   - builds LlamaIndex Tree Index
   - persists and reloads index artifacts

4. **Reasoning layer**
   - LangGraph manages retrieval loops
   - decides when to re-retrieve
   - produces the final response

5. **Storage layer**
   - raw document storage
   - persisted index storage
   - session/query history storage if needed

---

## Suggested Project Structure

```text
tree_rag_project/
├── app/
│   ├── main.py
│   ├── routes/
│   │   ├── upload.py
│   │   ├── query.py
│   │   └── history.py
│   ├── services/
│   │   ├── ingestion_service.py
│   │   ├── tree_index_service.py
│   │   ├── retrieval_service.py
│   │   ├── reasoning_service.py
│   │   └── history_service.py
│   ├── models/
│   │   └── schemas.py
│   └── config.py
├── ui/
│   └── streamlit_app.py
├── data/
│   ├── uploads/
│   ├── parsed/
│   └── index_store/
├── utils/
│   ├── pdf_parser.py
│   ├── metadata_extractor.py
│   ├── prompt_templates.py
│   └── logging_utils.py
├── tests/
├── requirements.txt
└── instructions.md
```

---

## Implementation Rules for the Coding Agent

### General
- Start with a minimal working version.
- Build the ingestion flow first, then tree index creation, then query reasoning, then UI.
- Keep each module small and testable.
- Do not mix UI logic with retrieval logic.
- Write code that can be extended later to support more document types.

### Tree index logic
- Use LlamaIndex Tree Index as the retrieval backbone.
- Persist the created tree to disk.
- Add a loader that checks whether a persisted index exists before rebuilding.
- Do not use vector similarity search as the default retrieval path.

### Metadata filtering
- Before retrieval, use metadata filters to reduce the search space.
- Metadata filtering should be applied based on available document properties.
- Prefer deterministic filtering before LLM traversal.

### Multi-hop reasoning
- The system should not stop after one retrieval attempt if the answer is incomplete.
- Implement a retrieval loop that can:
  1. retrieve relevant tree nodes
  2. assess sufficiency
  3. refine the query or branch selection
  4. retrieve again
  5. terminate with either a final answer or an insufficient-context response

### Sufficiency check
- Add a dedicated step to judge whether the retrieved context is enough.
- The sufficiency check should consider:
  - answer completeness
  - presence of direct evidence
  - whether multiple hops are required
  - whether the retrieved nodes support a reliable answer

### Query history
- Store recent user queries and responses in session state or backend storage.
- Display the history in the Streamlit UI.
- Include the query text, timestamp, and status of the response.

---

## Suggested Development Phases

### Phase 1: Ingestion and parsing
- Upload PDFs
- Parse text
- Extract metadata
- Save raw files

### Phase 2: Tree index creation
- Build tree index from parsed docs
- Persist index to disk
- Reload index successfully

### Phase 3: Retrieval and reasoning
- Implement metadata filtering
- Implement tree traversal
- Implement sufficiency check
- Add re-retrieval loop

### Phase 4: Backend APIs
- Expose upload and query endpoints
- Return answer plus supporting evidence

### Phase 5: Streamlit UI
- Build upload widget
- Build chat interface
- Display answer and context
- Show history panel

### Phase 6: Hardening
- Add logging
- Add error handling
- Add tests
- Add configuration management

---

## Response Contract
Every query response should return a structured payload similar to:

```json
{
  "query": "user question",
  "answer": "final answer text",
  "status": "success | insufficient_context | error",
  "evidence": [
    {
      "node_id": "...",
      "summary": "...",
      "metadata": {}
    }
  ],
  "retrieval_trace": [
    {
      "step": 1,
      "action": "metadata_filter | tree_traversal | re_retrieve",
      "notes": "..."
    }
  ]
}
```

---

## Acceptance Criteria
The project is complete when:
- A user can upload a PDF through the UI.
- The PDF is parsed and converted into a tree index.
- The tree index is persisted and later reloaded from disk.
- Queries are answered by tree traversal and multi-hop reasoning.
- Metadata filtering occurs before retrieval.
- The system re-retrieves when the first result is insufficient.
- The UI displays answer, evidence, and query history.
- The app clearly reports when the content is insufficient.

---

## Coding Agent Output Expectations
When generating code, the agent should:
- Create the project skeleton first.
- Implement one layer at a time.
- Include clear function names and comments.
- Prefer simple, maintainable solutions over clever ones.
- Keep the first version runnable with minimal setup.
- Use placeholder/mock LLM logic only if necessary, but structure the code so a real model can be swapped in later.

---

## Final Goal
Deliver a working tree-based document QA system that behaves like a real product:
- upload document
- build tree
- persist tree
- ask question
- traverse tree with reasoning
- retry if needed
- answer clearly
- show history

