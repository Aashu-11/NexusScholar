"""
pseudo_relevance_feedback.py — Pseudo-relevance feedback (PRF).
After initial retrieval, extract key terms from top results
to expand the query for a second retrieval pass.
"""

import re
from collections import Counter
from typing import List
from backend.retrieval.hybrid_recall import RetrievalCandidate


def extract_expansion_terms(
    original_query: str,
    top_candidates: List[RetrievalCandidate],
    max_terms: int = 8,
) -> list[str]:
    """
    Extract discriminative terms from top-ranked results
    that don't appear in the original query.
    """
    query_terms = set(re.findall(r'[a-zA-Z]{3,}', original_query.lower()))

    # Collect term frequencies from top results
    term_freq = Counter()
    for cand in top_candidates[:5]:
        text = cand.chunk.get("text", "")
        title = (cand.paper or {}).get("title", "")
        combined = f"{title} {text}".lower()
        tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9-]{2,}', combined)

        for token in tokens:
            if token not in query_terms and len(token) > 3:
                term_freq[token] += 1

    # Filter: terms appearing in 2+ top docs are likely relevant
    expansion = [
        term for term, count in term_freq.most_common(max_terms * 3)
        if count >= 2
    ]

    # Remove generic academic terms
    generic = {
        "paper", "study", "method", "approach", "results",
        "proposed", "based", "using", "also", "however",
        "shown", "figure", "table", "section", "model",
        "work", "data", "used", "show", "that", "with",
        "from", "this", "which", "these", "than", "more",
        "been", "have", "their", "they", "other", "such",
        "each", "between", "both", "into", "over", "where",
    }
    expansion = [t for t in expansion if t not in generic]

    return expansion[:max_terms]


def build_expanded_query(
    original_query: str,
    expansion_terms: list[str],
) -> str:
    """Build an expanded query by appending discriminative terms."""
    if not expansion_terms:
        return original_query
    return f"{original_query} {' '.join(expansion_terms)}"
