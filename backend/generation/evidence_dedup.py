"""
evidence_dedup.py — Evidence-level deduplication.
Removes near-duplicate evidence rows that waste context window
without adding new information.
"""

import re
from typing import List
from backend.generation.evidence_builder import EvidenceRow


def deduplicate_evidence(rows: List[EvidenceRow], threshold: float = 0.75) -> List[EvidenceRow]:
    """
    Remove evidence rows that are near-duplicates of higher-ranked rows.
    Uses Jaccard similarity on normalized token sets.
    """
    if len(rows) <= 1:
        return rows

    kept = [rows[0]]
    for candidate in rows[1:]:
        is_duplicate = False
        cand_tokens = _tokenize(candidate.chunk_text)

        for existing in kept:
            existing_tokens = _tokenize(existing.chunk_text)
            similarity = _jaccard(cand_tokens, existing_tokens)

            if similarity >= threshold:
                # Same paper, same content -> skip
                if candidate.paper_id == existing.paper_id:
                    is_duplicate = True
                    break
                # Different paper, very high similarity -> also skip
                elif similarity >= 0.90:
                    is_duplicate = True
                    break

        if not is_duplicate:
            kept.append(candidate)

    return kept


def _tokenize(text: str) -> set:
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "that", "this", "with", "by"}
    tokens = set(re.findall(r'\w+', text.lower()))
    return tokens - stop


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0
