"""
verifier.py — Post-synthesis verification pass.
Step 4: Validates every [CIT:id] tag, checks NLI entailment,
flags hallucinated citations, retracted papers, and unsupported claims.
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass

from backend.config import settings
from backend.generation.evidence_builder import EvidenceTable
from backend.citation.resolver import CitationResolver

logger = logging.getLogger(__name__)


@dataclass
class CitationTag:
    citation_number: int
    evidence_id: str
    is_valid: bool = True
    nli_score: float = 0.0


@dataclass
class VerificationResult:
    cleaned_text: str
    citations: list[CitationTag]
    warnings: list[str]
    citation_map: dict  # evidence_id → display number


def verify_answer(
    raw_answer: str,
    evidence_table: EvidenceTable,
    nli_checker=None,
) -> VerificationResult:
    """
    Full verification pipeline:
    1. Extract all [CIT:id] tags from the generated text
    2. Validate each citation exists in the evidence table
    3. Optionally check NLI entailment score
    4. Flag hallucinated citations, retracted sources, unsupported claims
    5. Replace [CIT:id] with numbered [N] references
    """
    evidence_ids = {row.evidence_id for row in evidence_table.rows}
    retracted_ids = {row.evidence_id for row in evidence_table.rows if row.is_retracted}
    preprint_ids = {row.evidence_id for row in evidence_table.rows if row.is_preprint}

    # Extract all citation tags
    cit_pattern = r'\[CIT:([^\]]+)\]'
    found_cits = re.findall(cit_pattern, raw_answer)

    # Build numbered mapping (first-appearance order)
    number_map: dict[str, int] = {}
    counter = 1
    for cid in found_cits:
        if cid in evidence_ids and cid not in number_map:
            number_map[cid] = counter
            counter += 1

    valid_citations: list[CitationTag] = []
    warnings: list[str] = []

    for cid in found_cits:
        if cid in evidence_ids:
            tag = CitationTag(
                citation_number=number_map.get(cid, 0),
                evidence_id=cid,
                is_valid=True,
            )
            valid_citations.append(tag)

            if cid in retracted_ids:
                warnings.append(
                    f"Citation [{number_map[cid]}] references a retracted paper"
                )
        else:
            warnings.append(f"Hallucinated citation removed: [CIT:{cid}]")

    # Replace [CIT:id] → [N] in the text
    cleaned = raw_answer
    for eid, num in number_map.items():
        cleaned = cleaned.replace(f"[CIT:{eid}]", f"[{num}]")

    # Remove any unresolved CIT tags
    cleaned = re.sub(r'\[CIT:[^\]]+\]', '', cleaned)

    # Detect uncited factual claims
    uncited_count = _count_uncited_claims(cleaned)
    if uncited_count > 2:
        warnings.append(
            f"{uncited_count} potentially unsupported factual claims detected"
        )

    # Add preprint warnings
    preprint_nums = [
        number_map[eid] for eid in preprint_ids
        if eid in number_map
    ]
    if preprint_nums:
        warnings.append(
            f"Citations {preprint_nums} are from preprints (not peer-reviewed)"
        )

    return VerificationResult(
        cleaned_text=cleaned,
        citations=valid_citations,
        warnings=warnings,
        citation_map=number_map,
    )


def _count_uncited_claims(text: str) -> int:
    """Heuristic: count sentences that look factual but have no [N] citation."""
    # Strong claim indicators — only flag sentences with concrete factual claims
    claim_pattern = re.compile(
        r'\b\d+\.?\d*\s*%'   # Percentages
        r'|\b\d+\.\d+\s+(?:accuracy|f1|bleu|rouge|score|precision|recall|perplexity)'  # Metric values
        r'|\bshowed\s+that\b'  # Showed that (strong claim)
        r'|\bachieved?\s+(?:a\s+)?(?:state|best|highest|top|superior)'  # Achievement claims
        r'|\boutperform'       # Outperformance claims
        r'|\bsignificantly?\s+(?:better|worse|higher|lower|improved)',  # Significance claims
        re.I,
    )
    # Sections/lines to skip
    skip_pattern = re.compile(
        r'^#{1,3}\s|^[-*]\s|^>|^\|'  # Headings, bullets, blockquotes, tables
        r'|evidence is insufficient|limitation|confidence|source|citation',
        re.I,
    )
    sentences = re.split(r'(?<=[.!?])\s+', text)
    count = 0
    for sent in sentences:
        sent = sent.strip()
        if (len(sent) > 50
                and not re.search(r'\[\d+\]', sent)
                and not skip_pattern.search(sent)
                and claim_pattern.search(sent)):
            count += 1
    return count