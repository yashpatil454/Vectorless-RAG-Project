from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import httpx
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _post(endpoint: str, **kwargs) -> Optional[dict]:
    try:
        resp = httpx.post(f"{API_BASE_URL}{endpoint}", timeout=1000, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", str(exc)) if exc.response.content else str(exc)
        st.error(f"API error ({exc.response.status_code}): {detail}")
    except Exception as exc:
        st.error(f"Connection error: {exc}")
    return None


def _get(endpoint: str, **kwargs) -> Optional[dict]:
    try:
        resp = httpx.get(f"{API_BASE_URL}{endpoint}", timeout=1000, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", str(exc)) if exc.response.content else str(exc)
        st.error(f"API error ({exc.response.status_code}): {detail}")
    except Exception as exc:
        st.error(f"Connection error: {exc}")
    return None


def _delete(endpoint: str) -> Optional[dict]:
    try:
        resp = httpx.delete(f"{API_BASE_URL}{endpoint}", timeout=1000)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Error: {exc}")
    return None


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _status_pill(status: str) -> str:
    colors = {
        "success": "#22c55e",
        "insufficient_context": "#f59e0b",
        "error": "#ef4444",
    }
    labels = {
        "success": "\u2713 Success",
        "insufficient_context": "\u26a0 Insufficient context",
        "error": "\u2717 Error",
    }
    color = colors.get(status, "#94a3b8")
    label = labels.get(status, status)
    return (
        f"<span style='background:{color};color:white;padding:2px 10px;"
        f"border-radius:12px;font-size:0.8em;font-weight:600;'>{label}</span>"
    )


def _render_response_meta(meta: dict) -> None:
    # Render status pill, token/call metrics, evidence, and retrieval trace.
    status = meta.get("status", "")
    tokens = meta.get("tokens_used", 0)
    calls = meta.get("llm_calls", 0)
    evidence = meta.get("evidence", [])

    st.markdown(
        f"{_status_pill(status)} &nbsp; "
        f"\U0001f524 **{tokens:,}** tokens &nbsp; "
        f"\U0001f916 **{calls}** LLM calls &nbsp; "
        f"\U0001f4c4 **{len(evidence)}** evidence nodes",
        unsafe_allow_html=True,
    )

    if evidence:
        with st.expander(f"\U0001f4da Evidence ({len(evidence)} nodes)"):
            for node in evidence:
                c1, c2 = st.columns([1, 3])
                with c1:
                    nid = node["node_id"]
                    st.markdown(f"**`{nid[:16]}{'...' if len(nid) > 16 else ''}`**")
                    for k, v in list((node.get("metadata") or {}).items())[:4]:
                        st.caption(f"{k}: {str(v)[:50]}")
                with c2:
                    st.text(node["summary"])
                st.divider()

    trace = meta.get("retrieval_trace", [])
    if trace:
        with st.expander("\U0001f50d Retrieval trace"):
            for step in trace:
                icon = {
                    "metadata_filter": "\U0001f3f7",
                    "tree_traversal": "\U0001f333",
                    "re_retrieve": "\U0001f504",
                    "sufficiency_check": "\u2705",
                }.get(step["action"], "\u2022")
                st.markdown(f"{icon} **Step {step['step']}** \u00b7 `{step['action']}`")
                st.caption(step["notes"])


def _render_tree() -> None:
    # Build an interactive pyvis graph from the persisted index structure.
    tree_resp = _get("/index/tree")
    if not tree_resp:
        st.warning("No tree data available. Build an index first.")
        return

    nodes = tree_resp.get("nodes", [])
    edges = tree_resp.get("edges", [])

    if not nodes:
        st.info("The index is empty.")
        return

    from pyvis.network import Network  # noqa: PLC0415

    net = Network(
        height="620px",
        width="100%",
        directed=True,
        bgcolor="#0e1117",
        font_color="white",
    )
    net.set_options(
        '{"layout": {"hierarchical": {"enabled": true, "direction": "UD", "sortMethod": "directed"}},'
        ' "physics": {"enabled": false},'
        ' "edges": {"smooth": {"type": "cubicBezier", "roundness": 0.4}}}'
    )

    root_count = sum(1 for n in nodes if n.get("is_root"))
    leaf_count = len(nodes) - root_count

    for node in nodes:
        nid = node["id"]
        is_root = node.get("is_root", False)
        section = (node.get("section") or "")[:30]
        subsection = (node.get("subsection") or "")[:30]
        page_range = node.get("page_range") or [node.get("page_number", "?")]
        page_str = f"p.{page_range[0]}" if len(page_range) == 1 else f"p.{page_range[0]}-{page_range[-1]}"
        preview = (node.get("text") or "")[:160].replace("\\n", " ")

        label = (section[:20] or "Root") + f"\\n{page_str}" if is_root else (subsection[:20] or section[:20] or page_str)
        title = (
            f"<b>{'ROOT' if is_root else 'LEAF'}</b><br>"
            f"Section: {section}<br>"
            + (f"Subsection: {subsection}<br>" if subsection else "")
            + f"Pages: {page_str}<br><br>"
            f"<i>{preview}</i>"
        )
        color = "#22c55e" if is_root else "#3b82f6"
        size = 28 if is_root else 14
        net.add_node(nid, label=label, title=title, color=color, shape="dot", size=size)

    for edge in edges:
        net.add_edge(edge["from"], edge["to"], color="#475569", arrows="to", width=1.5)

    st.caption(
        f"\U0001f333 **{len(nodes)}** total nodes \u2014 "
        f"\U0001f7e2 {root_count} root (green) \u2014 "
        f"\U0001f535 {leaf_count} leaf (blue) \u2014 "
        "hover a node for details"
    )
    html = net.generate_html()
    components.html(html, height=650, scrolling=False)


# ---------------------------------------------------------------------------
# Page config + session state
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Tree-RAG - Document QA",
    page_icon="\U0001f333",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_tokens" not in st.session_state:
    st.session_state.session_tokens = 0
if "session_llm_calls" not in st.session_state:
    st.session_state.session_llm_calls = 0
if "session_queries" not in st.session_state:
    st.session_state.session_queries = 0

# ---------------------------------------------------------------------------
# Header + session metrics bar
# ---------------------------------------------------------------------------

st.title("\U0001f333 Tree-RAG - Document Q&A")
st.caption(
    "Tree-based document QA: PDF \u2192 LlamaIndex TreeIndex \u2192 LangGraph multi-hop reasoning. No vector search."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Queries this session", st.session_state.session_queries)
m2.metric("Tokens used this session", f"{st.session_state.session_tokens:,}")
m3.metric("LLM API calls this session", st.session_state.session_llm_calls)
_idx_status = _get("/index/status")
m4.metric("Index", "Ready" if (_idx_status or {}).get("index_exists") else "Not built")

st.divider()

# ---------------------------------------------------------------------------
# Sidebar - Documents + filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("\U0001f4c4 Documents")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Multiple uploads accumulate into the same index.",
    )

    tags_input = st.text_input(
        "Tags (optional)",
        placeholder="e.g. finance, 2024, Q3",
        help="Comma-separated tags you can later use to filter retrieval.",
    )

    col_upload, col_build = st.columns(2)

    with col_upload:
        if st.button("Upload", use_container_width=True, disabled=(uploaded_file is None)):
            with st.spinner("Parsing..."):
                result = _post(
                    "/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    data={"tags": tags_input} if tags_input else {},
                )
            if result:
                st.success(f"Uploaded: {result['message']}")
                st.info(f"{result['chunk_count']} pages extracted.")

    with col_build:
        if st.button("Build Index", use_container_width=True):
            with st.spinner("Building..."):
                result = _post("/index/build")
            if result:
                label = {
                    "built": "Index built",
                    "loaded": "Loaded",
                    "already_exists": "Already exists",
                }.get(result.get("status", ""), "Done")
                st.success(f"{label} - {result['node_count']} nodes.")
                st.rerun()

    if st.button("\U0001f504 Rebuild Index", use_container_width=True):
        with st.spinner("Rebuilding..."):
            result = _post("/index/refresh")
        if result:
            st.success(f"Rebuilt - {result['node_count']} nodes.")
            st.rerun()

    st.divider()
    st.subheader("Index Status")
    status_resp = _get("/index/status")
    if status_resp:
        if status_resp.get("index_exists"):
            st.success(f"Ready \u2014 {status_resp.get('node_count', '?')} nodes")
        else:
            st.warning("No index. Upload a PDF and build.")

    st.divider()
    st.subheader("Retrieval Filters")
    doc_filter = st.text_input(
        "Filter by document name",
        placeholder="filename.pdf",
        help="Leave blank to search all documents.",
    )
    tags_filter_input = st.text_input(
        "Filter by tags",
        placeholder="finance, 2024",
        help="Comma-separated tags. Only chunks with ALL these tags will be used.",
    )

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_chat, tab_tree, tab_history = st.tabs(
    ["\U0001f4ac Chat", "\U0001f333 Tree View", "\U0001f554 History"]
)

# ---- Chat tab ---------------------------------------------------------------
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                _render_response_meta(msg["meta"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "meta": None})
        with st.chat_message("user"):
            st.markdown(prompt)

        req_body: dict = {"query": prompt}
        if doc_filter.strip():
            req_body["document_filter"] = doc_filter.strip()
        if tags_filter_input.strip():
            req_body["tags_filter"] = [t.strip() for t in tags_filter_input.split(",") if t.strip()]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = _post("/query", json=req_body)

            if result:
                st.markdown(result["answer"])
                _render_response_meta(result)

                st.session_state.session_tokens += result.get("tokens_used", 0)
                st.session_state.session_llm_calls += result.get("llm_calls", 0)
                st.session_state.session_queries += 1

                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"], "meta": result}
                )
                st.rerun()
            else:
                st.error("No response received from the backend.")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "*(No response from backend)*", "meta": None}
                )

# ---- Tree View tab ----------------------------------------------------------
with tab_tree:
    st.subheader("\U0001f333 Index Tree Structure")
    st.caption(
        "Each node is a document chunk. "
        "\U0001f7e2 Green = root summary nodes. "
        "\U0001f535 Blue = leaf page nodes. "
        "Hover over any node for its text preview."
    )
    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("\U0001f504 Refresh", key="refresh_tree"):
            st.rerun()
    _render_tree()

# ---- History tab ------------------------------------------------------------
with tab_history:
    hdr_col, clr_col = st.columns([5, 1])
    with hdr_col:
        st.subheader("Query History")
    with clr_col:
        st.write("")
        if st.button("Clear", use_container_width=True):
            _delete("/history")
            st.rerun()

    hist_resp = _get("/history")
    if hist_resp and hist_resp.get("entries"):
        entries = hist_resp["entries"]
        st.caption(f"{len(entries)} saved queries")
        for entry in entries:
            ts = entry.get("timestamp", "")
            try:
                ts_display = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                ts_display = ts

            status = entry.get("status", "")
            tokens = entry.get("tokens_used", 0)
            calls = entry.get("llm_calls", 0)
            evidence_count = entry.get("evidence_count", 0)

            with st.expander(f"{ts_display} \u2014 {entry.get('query', '')[:80]}"):
                st.markdown(
                    f"{_status_pill(status)} &nbsp; "
                    f"\U0001f524 **{tokens:,}** tokens &nbsp; "
                    f"\U0001f916 **{calls}** LLM calls &nbsp; "
                    f"\U0001f4c4 **{evidence_count}** nodes",
                    unsafe_allow_html=True,
                )
                st.markdown("**Answer:**")
                st.markdown(entry.get("answer", ""))
    else:
        st.info("No history yet. Ask a question to get started.")