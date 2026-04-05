from __future__ import annotations

from dataclasses import dataclass, field

from backend.config import settings
from backend.retrieval.hybrid_recall import RetrievalCandidate


@dataclass
class TaskPlan:
    intent: str
    response_format: str
    confidence_level: str
    confidence_score: float
    is_retrieval_sufficient: bool
    should_abstain: bool
    reasoning: str
    key_points: list[str] = field(default_factory=list)


async def plan_response(
    query: str,
    intent: str,
    candidates: list[RetrievalCandidate],
    groq=None,
) -> TaskPlan:
    # Normalize confidence to 0-1 range using sigmoid of final_score
    # Cross-encoder scores can range roughly -10 to +10; RRF scores are small positive
    # We use final_score (which combines both) and normalize it
    import math

    def _normalize_score(score: float) -> float:
        """Sigmoid normalization to map arbitrary scores to 0-1."""
        return 1.0 / (1.0 + math.exp(-score))

    top_scores = [_normalize_score(c.final_score) for c in candidates[:5]] if candidates else []
    confidence_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
    source_count = len(candidates)
    peer_reviewed = sum(1 for c in candidates if c.paper.get("is_peer_reviewed"))
    has_recent = any((c.paper.get("year") or 0) >= 2022 for c in candidates[:5])

    # Count unique papers for diversity assessment
    unique_papers = len({c.chunk.get("paper_id") for c in candidates if c.chunk.get("paper_id")})

    # Multi-signal sufficiency: score + count + diversity + peer-review
    sufficient = (
        source_count >= 2
        and confidence_score >= settings.CONFIDENCE_THRESHOLD
        and unique_papers >= 2
    ) or (
        source_count >= 3
        and peer_reviewed >= 1
    ) or (
        source_count >= 5
    )
    should_abstain = not sufficient

    response_format = _response_format_for_intent(intent)
    confidence_level = _confidence_label(confidence_score, source_count, peer_reviewed)
    key_points = [
        f"Intent classified as `{intent}`.",
        f"Retrieved {source_count} final evidence candidates from {unique_papers} unique papers.",
        f"{peer_reviewed} of the retrieved candidates are peer-reviewed.",
    ]
    if has_recent:
        key_points.append("Recent papers (2022+) are present in the top evidence set.")

    reasoning = "\n".join(
        [
            f"Query: {query}",
            f"Response format: {response_format}",
            f"Confidence score: {confidence_score:.3f}",
            f"Confidence level: {confidence_level}",
            f"Unique papers: {unique_papers}",
            f"Peer-reviewed sources: {peer_reviewed}",
            f"Retrieval sufficient: {'yes' if sufficient else 'no'}",
            f"Abstain: {'yes' if should_abstain else 'no'}",
            f"INSTRUCTION: Write a deep, comprehensive research synthesis. Extract ALL specific metrics, results, and methodological details from the evidence. Minimum 800 words for survey/comparison queries.",
        ]
    )

    return TaskPlan(
        intent=intent,
        response_format=response_format,
        confidence_level=confidence_level,
        confidence_score=confidence_score,
        is_retrieval_sufficient=sufficient,
        should_abstain=should_abstain,
        reasoning=reasoning,
        key_points=key_points,
    )


def _response_format_for_intent(intent: str) -> str:
    mapping = {
        "benchmark_comparison": "markdown_table",
        "literature_survey": "survey_summary",
        "trend_analysis": "timeline_summary",
        "dataset_discovery": "dataset_list",
        "paper_lookup": "paper_profile",
        "author_search": "author_summary",
        "contradiction_check": "evidence_conflict_summary",
    }
    return mapping.get(intent, "structured_markdown")


def _confidence_label(score: float, source_count: int, peer_reviewed: int) -> str:
    if source_count == 0:
        return "low"
    if score >= 0.7 and peer_reviewed >= 2:
        return "high"
    if score >= settings.CONFIDENCE_THRESHOLD and source_count >= 3:
        return "medium"
    return "low"
