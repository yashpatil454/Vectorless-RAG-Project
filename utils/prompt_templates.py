"""Prompt templates used across the reasoning pipeline."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Query decomposition
# ---------------------------------------------------------------------------
QUERY_DECOMPOSITION_TEMPLATE = """\
You are a precise query analyst.

Given the user question below, decompose it into a list of 1–3 focused sub-questions
that together cover everything needed to answer the original question fully.

If the question is already self-contained and simple, return a single item list with
the original question unchanged.

User question: {query}

Respond with a JSON array of strings.  Example:
["sub-question 1", "sub-question 2"]

Respond ONLY with the JSON array. No prose, no markdown fences.
"""

# ---------------------------------------------------------------------------
# Context sufficiency check
# ---------------------------------------------------------------------------
SUFFICIENCY_CHECK_TEMPLATE = """\
You are a rigorous fact-checker.

Original question: {query}

Retrieved context:
{context}

Determine whether the context above contains enough information to answer the question
completely and accurately.

Respond with a JSON object with exactly two keys:
  "sufficient": true or false
  "reasoning": one sentence explaining your decision

Example:
{{"sufficient": true, "reasoning": "The context directly states the answer."}}

Respond ONLY with the JSON object.  No prose, no markdown fences.
"""

# ---------------------------------------------------------------------------
# Final answer generation
# ---------------------------------------------------------------------------
ANSWER_GENERATION_TEMPLATE = """\
You are a helpful and accurate assistant.

Answer the user's question using ONLY the information in the provided context.
Do not add information that is not present in the context.
If you are uncertain, say so explicitly.

Question: {query}

Context:
{context}

Answer:
"""

# ---------------------------------------------------------------------------
# Refined sub-query generation (used during re-retrieval)
# ---------------------------------------------------------------------------
QUERY_REFINEMENT_TEMPLATE = """\
You are a search query optimiser.

The following question was not answered sufficiently by the first retrieval attempt.

Original question: {query}
Previous context (insufficient): {context}

Generate a refined or alternative search query that is more likely to surface
the missing information.  The refined query should be concise (one sentence).

Respond with ONLY the refined query string.  No prose, no markdown fences.
"""
