"""
resolver.py — Citation ID → paper metadata resolution.
Validates citations against the evidence table, computes NLI entailment scores,
and provides consensus assessment across multiple sources.
"""

from __future__ import annotations
import logging
from typing import Optional

from backend.config import settings
from backend.generation.evidence_builder import EvidenceRow, EvidenceTable

logger = logging.getLogger(__name__)

_nli_model = None


class CitationResolver:
    """Resolves citation IDs to full paper metadata and validates entailment."""

    def __init__(self, evidence_table: EvidenceTable):
        self.evidence_table = evidence_table
        self._row_map = {r.evidence_id: r for r in evidence_table.rows}

    def resolve(self, evidence_id: str) -> Optional[EvidenceRow]:
        """Look up an evidence row by ID."""
        return self._row_map.get(evidence_id)

    def resolve_numbered(self, citation_number: int, citation_map: dict) -> Optional[EvidenceRow]:
        """Resolve a display number [N] back to an evidence row."""
        for eid, num in citation_map.items():
            if num == citation_number:
                return self._row_map.get(eid)
        return None

    def validate_entailment(self, claim: str, evidence_id: str) -> tuple[bool, float]:
        """
        Check if a claim is entailed by the source evidence.
        Returns (is_valid, entailment_score).
        """
        row = self.resolve(evidence_id)
        if not row:
            return False, 0.0

        score = compute_entailment(claim, row.chunk_text)
        is_valid = score >= settings.NLI_ENTAILMENT_THRESHOLD

        if row.is_retracted:
            is_valid = False

        return is_valid, score

    def assess_consensus(self, claim: str) -> dict:
        """
        Assess evidence consensus for a claim across all rows.
        Returns support/contradict/neutral counts and strength label.
        """
        supporting = 0
        contradicting = 0
        neutral = 0

        for row in self.evidence_table.rows:
            score = compute_entailment(claim, row.chunk_text)
            contra_score = compute_entailment(
                f"It is not true that {claim}", row.chunk_text
            )
            if score > 0.65:
                supporting += 1
            elif contra_score > 0.65:
                contradicting += 1
            else:
                neutral += 1

        total = supporting + contradicting + neutral
        if total == 0:
            strength = "insufficient"
        elif contradicting > 0 and supporting > 0:
            strength = "contested"
        elif supporting >= 3:
            strength = "broadly_supported"
        elif supporting >= 1:
            strength = "preliminary"
        else:
            strength = "insufficient"

        return {
            "supporting": supporting,
            "contradicting": contradicting,
            "neutral": neutral,
            "strength": strength,
        }

    def get_retraction_warnings(self) -> list[str]:
        """Flag any retracted papers in the evidence set."""
        warnings = []
        for row in self.evidence_table.rows:
            if row.is_retracted:
                warnings.append(
                    f'RETRACTED: "{row.paper_title}" ({row.year}) — '
                    f'claims from this paper should not be treated as valid evidence.'
                )
        return warnings

    def get_evidence_badges(self) -> dict:
        """Compute summary badge data for the answer."""
        return {
            "total_sources": self.evidence_table.total_sources,
            "peer_reviewed": self.evidence_table.peer_reviewed_count,
            "preprints": self.evidence_table.preprint_count,
            "retracted": sum(1 for r in self.evidence_table.rows if r.is_retracted),
            "has_retraction_warning": self.evidence_table.has_retracted,
        }


def compute_entailment(claim: str, evidence: str) -> float:
    """
    Compute NLI entailment score between a claim and source evidence.
    Returns probability of entailment (0.0–1.0).
    """
    global _nli_model
    if _nli_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _nli_model = CrossEncoder(settings.NLI_MODEL)
            logger.info(f"Loaded NLI model: {settings.NLI_MODEL}")
        except Exception as e:
            logger.warning(f"NLI model not available: {e}")
            _nli_model = "fallback"

    if _nli_model == "fallback":
        # Return a neutral score below entailment threshold to indicate
        # "unverified" rather than optimistically claiming entailment.
        # This prevents false validation when NLI model is unavailable.
        return 0.40

    try:
        scores = _nli_model.predict([(evidence, claim)])
        if hasattr(scores, '__len__') and len(scores) > 0:
            if hasattr(scores[0], '__len__') and len(scores[0]) >= 3:
                return float(scores[0][2])  # entailment probability
            return float(scores[0])
        return float(scores)
    except Exception as e:
        logger.warning("NLI scoring failed: %s", e)
        return 0.40  # unverified — below entailment threshold