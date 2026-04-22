from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from app.config import settings
from app.models.schemas import EvidenceNode, QueryResponse, RetrievalStep
from app.services.retrieval_service import retrieve_from_tree
from app.services.tree_index_service import load_index
from utils.logging_utils import get_logger
from utils.prompt_templates import (
    ANSWER_GENERATION_TEMPLATE,
    QUERY_DECOMPOSITION_TEMPLATE,
    QUERY_REFINEMENT_TEMPLATE,
    SUFFICIENCY_CHECK_TEMPLATE,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class ReasoningState(TypedDict):
    query: str
    document_filter: Optional[str]
    tags_filter: Optional[List[str]]
    sub_queries: List[str]
    contexts: List[str]           # accumulated text blocks from retrieval
    evidence: List[Dict[str, Any]]  # serialised EvidenceNode dicts
    trace: List[Dict[str, Any]]     # serialised RetrievalStep dicts
    hop_count: int
    tokens_used: int
    llm_calls: int
    answer: str
    status: str                   # "success" | "insufficient_context" | "error"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.google_api_key or os.environ.get("GOOGLE_API_KEY", ""),
        temperature=0,
    )


def _add_trace(state: ReasoningState, step_num: int, action: str, notes: str) -> None:
    state["trace"].append({"step": step_num, "action": action, "notes": notes})


def _extract_tokens(response: Any) -> int:
    """Gemini-native path first, then LangChain standard."""
    rm = getattr(response, "response_metadata", None) or {}
    rm_um = rm.get("usage_metadata", {}) if isinstance(rm, dict) else {}
    if rm_um:
        v = rm_um.get("total_token_count") or rm_um.get("totalTokenCount")
        if v:
            return int(v)
    um = getattr(response, "usage_metadata", None)
    if um is not None:
        v = um.get("total_tokens", 0) if isinstance(um, dict) else getattr(um, "total_tokens", 0)
        if v:
            return int(v)
    return 0


def _invoke_llm(llm: ChatGoogleGenerativeAI, prompt: str, state: ReasoningState) -> Any:
    """Invoke the LLM, accumulate token usage and call count into state."""
    response = llm.invoke(prompt)
    state["llm_calls"] += 1
    state["tokens_used"] += _extract_tokens(response)
    return response


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def decompose_query_node(state: ReasoningState) -> ReasoningState:
    """Decompose the user query into focused sub-questions."""
    llm = _get_llm()
    prompt = QUERY_DECOMPOSITION_TEMPLATE.format(query=state["query"])

    try:
        response = _invoke_llm(llm, prompt, state)
        sub_queries: List[str] = json.loads(response.content)
        if not isinstance(sub_queries, list):
            sub_queries = [state["query"]]
    except Exception as exc:
        logger.warning("Query decomposition failed (%s); using original query.", exc)
        sub_queries = [state["query"]]

    state["sub_queries"] = sub_queries
    _add_trace(
        state,
        step_num=len(state["trace"]) + 1,
        action="metadata_filter",
        notes=f"Decomposed into {len(sub_queries)} sub-question(s): {sub_queries}",
    )
    return state


def retrieve_node(state: ReasoningState) -> ReasoningState:
    """Retrieve evidence nodes from the TreeIndex for each sub-query."""
    index = load_index()
    if index is None:
        state["status"] = "error"
        state["answer"] = "No index is available. Please build the index first."
        return state

    all_evidence: List[EvidenceNode] = []
    for sub_q in state["sub_queries"]:
        nodes = retrieve_from_tree(index, sub_q)
        all_evidence.extend(nodes)

    # Deduplicate by node_id, preserving order
    seen_ids: set = set()
    unique_evidence: List[EvidenceNode] = []
    for node in all_evidence:
        if node.node_id not in seen_ids:
            seen_ids.add(node.node_id)
            unique_evidence.append(node)

    state["evidence"] = [n.model_dump() for n in unique_evidence]
    state["contexts"] = [n.summary for n in unique_evidence]

    _add_trace(
        state,
        step_num=len(state["trace"]) + 1,
        action="tree_traversal",
        notes=f"Retrieved {len(unique_evidence)} unique node(s) across {len(state['sub_queries'])} sub-queries.",
    )
    return state


def check_sufficiency_node(state: ReasoningState) -> ReasoningState:
    """Assess whether the current retrieved context is sufficient."""
    if state.get("status") == "error":
        return state

    context_text = "\n\n".join(state["contexts"]) if state["contexts"] else "(none)"
    llm = _get_llm()
    prompt = SUFFICIENCY_CHECK_TEMPLATE.format(
        query=state["query"],
        context=context_text,
    )

    try:
        response = _invoke_llm(llm, prompt, state)
        result = json.loads(response.content)
        is_sufficient: bool = result.get("sufficient", False)
        reasoning: str = result.get("reasoning", "")
    except Exception as exc:
        logger.warning("Sufficiency check failed (%s); assuming insufficient.", exc)
        is_sufficient = False
        reasoning = "Sufficiency check could not be completed."

    _add_trace(
        state,
        step_num=len(state["trace"]) + 1,
        action="sufficiency_check",
        notes=f"sufficient={is_sufficient}. {reasoning}",
    )
    state["status"] = "sufficient" if is_sufficient else "insufficient"
    return state


def re_retrieve_node(state: ReasoningState) -> ReasoningState:
    """Refine the query and perform an additional retrieval pass."""
    state["hop_count"] += 1
    context_text = "\n\n".join(state["contexts"]) if state["contexts"] else "(none)"
    llm = _get_llm()
    prompt = QUERY_REFINEMENT_TEMPLATE.format(
        query=state["query"],
        context=context_text,
    )

    try:
        response = _invoke_llm(llm, prompt, state)
        refined_query = response.content.strip()
    except Exception as exc:
        logger.warning("Query refinement failed (%s); reusing original query.", exc)
        refined_query = state["query"]

    state["sub_queries"] = [refined_query]

    _add_trace(
        state,
        step_num=len(state["trace"]) + 1,
        action="re_retrieve",
        notes=f"Hop {state['hop_count']}: refined query → '{refined_query}'",
    )
    return state


def generate_answer_node(state: ReasoningState) -> ReasoningState:
    """Generate the final answer from the accumulated context."""
    context_text = "\n\n".join(state["contexts"]) if state["contexts"] else "(none)"
    llm = _get_llm()
    prompt = ANSWER_GENERATION_TEMPLATE.format(
        query=state["query"],
        context=context_text,
    )

    try:
        response = _invoke_llm(llm, prompt, state)
        state["answer"] = response.content.strip()
        state["status"] = "success"
    except Exception as exc:
        logger.error("Answer generation failed: %s", exc)
        state["answer"] = f"Answer generation failed: {exc}"
        state["status"] = "error"

    return state


def insufficient_node(state: ReasoningState) -> ReasoningState:
    """Terminal node: emit an insufficient-context response."""
    context_text = "\n\n".join(state["contexts"]) if state["contexts"] else "(none)"
    state["answer"] = (
        "Insufficient context: the available documents do not contain enough "
        "information to answer this question confidently.\n\n"
        f"Best available context:\n{context_text[:1000]}"
    )
    state["status"] = "insufficient_context"
    return state


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def _route_after_sufficiency(state: ReasoningState) -> str:
    if state.get("status") == "error":
        return "generate_answer"
    if state["status"] == "sufficient":
        return "generate_answer"
    if state["hop_count"] >= settings.max_hops:
        return "insufficient"
    return "re_retrieve"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> Any:
    g = StateGraph(ReasoningState)

    g.add_node("decompose_query", decompose_query_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("check_sufficiency", check_sufficiency_node)
    g.add_node("re_retrieve", re_retrieve_node)
    g.add_node("generate_answer", generate_answer_node)
    g.add_node("insufficient", insufficient_node)

    g.set_entry_point("decompose_query")
    g.add_edge("decompose_query", "retrieve")
    g.add_edge("retrieve", "check_sufficiency")
    g.add_conditional_edges(
        "check_sufficiency",
        _route_after_sufficiency,
        {
            "generate_answer": "generate_answer",
            "re_retrieve": "re_retrieve",
            "insufficient": "insufficient",
        },
    )
    g.add_edge("re_retrieve", "retrieve")
    g.add_edge("generate_answer", END)
    g.add_edge("insufficient", END)

    return g.compile()


_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_reasoning(
    query: str,
    document_filter: Optional[str] = None,
    tags_filter: Optional[List[str]] = None,
) -> QueryResponse:
    """Execute the multi-hop reasoning loop and return a structured response."""
    initial_state: ReasoningState = {
        "query": query,
        "document_filter": document_filter,
        "tags_filter": tags_filter,
        "sub_queries": [],
        "contexts": [],
        "evidence": [],
        "trace": [],
        "hop_count": 0,
        "tokens_used": 0,
        "llm_calls": 0,
        "answer": "",
        "status": "",
    }

    logger.info("Starting reasoning for query: '%s'", query)
    final_state: ReasoningState = _graph.invoke(initial_state)
    logger.info("Reasoning complete. Status: %s", final_state["status"])

    evidence = [EvidenceNode(**e) for e in final_state["evidence"]]
    trace = [RetrievalStep(**t) for t in final_state["trace"]]

    status = final_state["status"]
    if status not in ("success", "insufficient_context", "error"):
        status = "error"

    return QueryResponse(
        query=query,
        answer=final_state["answer"],
        status=status,  # type: ignore[arg-type]
        evidence=evidence,
        retrieval_trace=trace,
        tokens_used=final_state["tokens_used"],
        llm_calls=final_state["llm_calls"],
    )
