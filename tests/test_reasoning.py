from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    return msg


def _make_index_mock() -> MagicMock:
    """Return a MagicMock that looks enough like a TreeIndex to pass through retrieve_from_tree."""
    mock_node = MagicMock()
    mock_node.node.node_id = "node-1"
    mock_node.node.get_content.return_value = "Test content about topic X."
    mock_node.node.metadata = {"doc_name": "test.pdf", "page_number": 1}

    mock_response = MagicMock()
    mock_response.source_nodes = [mock_node]

    mock_engine = MagicMock()
    mock_engine.query.return_value = mock_response

    mock_index = MagicMock()
    mock_index.as_query_engine.return_value = mock_engine
    return mock_index


# ---------------------------------------------------------------------------
# Reasoning service graph nodes — unit tests
# ---------------------------------------------------------------------------

class TestDecomposeQueryNode:
    def test_returns_sub_queries_list(self):
        from app.services.reasoning_service import ReasoningState, decompose_query_node

        state: ReasoningState = {
            "query": "What is the revenue for Q1?",
            "document_filter": None,
            "tags_filter": None,
            "sub_queries": [],
            "contexts": [],
            "evidence": [],
            "trace": [],
            "hop_count": 0,
            "answer": "",
            "status": "",
        }

        mock_resp = _mock_llm_response('["What is the Q1 revenue?"]')
        with patch("app.services.reasoning_service._get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = mock_resp
            result = decompose_query_node(state)

        assert len(result["sub_queries"]) >= 1
        assert len(result["trace"]) == 1

    def test_falls_back_to_original_query_on_bad_json(self):
        from app.services.reasoning_service import ReasoningState, decompose_query_node

        state: ReasoningState = {
            "query": "What is the plan?",
            "document_filter": None,
            "tags_filter": None,
            "sub_queries": [],
            "contexts": [],
            "evidence": [],
            "trace": [],
            "hop_count": 0,
            "answer": "",
            "status": "",
        }

        mock_resp = _mock_llm_response("not valid json")
        with patch("app.services.reasoning_service._get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = mock_resp
            result = decompose_query_node(state)

        assert result["sub_queries"] == ["What is the plan?"]


class TestCheckSufficiencyNode:
    def test_marks_sufficient(self):
        from app.services.reasoning_service import ReasoningState, check_sufficiency_node

        state: ReasoningState = {
            "query": "What is X?",
            "document_filter": None,
            "tags_filter": None,
            "sub_queries": ["What is X?"],
            "contexts": ["X is the answer."],
            "evidence": [],
            "trace": [],
            "hop_count": 0,
            "answer": "",
            "status": "",
        }

        mock_resp = _mock_llm_response('{"sufficient": true, "reasoning": "Direct answer found."}')
        with patch("app.services.reasoning_service._get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = mock_resp
            result = check_sufficiency_node(state)

        assert result["status"] == "sufficient"

    def test_marks_insufficient(self):
        from app.services.reasoning_service import ReasoningState, check_sufficiency_node

        state: ReasoningState = {
            "query": "What is Y?",
            "document_filter": None,
            "tags_filter": None,
            "sub_queries": [],
            "contexts": [],
            "evidence": [],
            "trace": [],
            "hop_count": 0,
            "answer": "",
            "status": "",
        }

        mock_resp = _mock_llm_response('{"sufficient": false, "reasoning": "No evidence found."}')
        with patch("app.services.reasoning_service._get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = mock_resp
            result = check_sufficiency_node(state)

        assert result["status"] == "insufficient"


class TestInsufficientNode:
    def test_sets_status_and_answer(self):
        from app.services.reasoning_service import ReasoningState, insufficient_node

        state: ReasoningState = {
            "query": "Unknown question",
            "document_filter": None,
            "tags_filter": None,
            "sub_queries": [],
            "contexts": ["Some partial context."],
            "evidence": [],
            "trace": [],
            "hop_count": 3,
            "answer": "",
            "status": "insufficient",
        }

        result = insufficient_node(state)
        assert result["status"] == "insufficient_context"
        assert "Insufficient context" in result["answer"]


class TestRunReasoning:
    def test_returns_query_response(self):
        """End-to-end test of run_reasoning with all LLM + index calls mocked."""
        from app.services.reasoning_service import run_reasoning

        sufficient_resp = _mock_llm_response('{"sufficient": true, "reasoning": "Found answer."}')
        decomp_resp = _mock_llm_response('["What is the revenue?"]')
        answer_resp = _mock_llm_response("The revenue was $10M.")

        mock_index = _make_index_mock()

        with (
            patch("app.services.reasoning_service._get_llm") as mock_get_llm,
            patch("app.services.reasoning_service.load_index", return_value=mock_index),
        ):
            call_count = {"n": 0}

            def side_effect(prompt: str):
                call_count["n"] += 1
                if "decompose" in prompt.lower() or call_count["n"] == 1:
                    return decomp_resp
                if "sufficient" in prompt.lower() or call_count["n"] == 2:
                    return sufficient_resp
                return answer_resp

            mock_get_llm.return_value.invoke.side_effect = side_effect
            response = run_reasoning("What is the revenue?")

        assert response.query == "What is the revenue?"
        assert response.status in ("success", "insufficient_context", "error")
        assert isinstance(response.answer, str)
