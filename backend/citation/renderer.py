"""
renderer.py — Citation markup generation.
Converts verified answer text into structured data for frontend rendering:
citation hover cards, consensus badges, evidence panel links.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

from backend.generation.evidence_builder import EvidenceRow, EvidenceTable
from backend.generation.verifier import VerificationResult


@dataclass
class CitationHoverCard:
    """Data for the citation hover tooltip."""
    number: int
    paper_title: str
    authors: str
    year: int | None
    venue: str | None
    passage_preview: str
    section: str
    is_peer_reviewed: bool
    is_retracted: bool
    source_url: str | None
    pdf_url: str | None


@dataclass
class ConsensusBadge:
    """Data for the consensus indicator shown after multi-source claims."""
    claim_text: str
    supporting: int
    contradicting: int
    strength: str  # broadly_supported | contested | preliminary


@dataclass
class RenderedAnswer:
    """Complete rendered answer with all citation metadata for the frontend."""
    markdown_text: str
    citation_cards: list[CitationHoverCard] = field(default_factory=list)
    consensus_badges: list[ConsensusBadge] = field(default_factory=list)
    source_badges: dict = field(default_factory=dict)
    uncertainty_flags: list[str] = field(default_factory=list)
    is_abstention: bool = False

    def to_dict(self) -> dict:
        return {
            "markdown_text": self.markdown_text,
            "citation_cards": [
                {
                    "number": c.number,
                    "paper_title": c.paper_title,
                    "authors": c.authors,
                    "year": c.year,
                    "venue": c.venue,
                    "passage_preview": c.passage_preview,
                    "section": c.section,
                    "is_peer_reviewed": c.is_peer_reviewed,
                    "is_retracted": c.is_retracted,
                    "source_url": c.source_url,
                    "pdf_url": c.pdf_url,
                }
                for c in self.citation_cards
            ],
            "consensus_badges": [
                {
                    "claim_text": cb.claim_text,
                    "supporting": cb.supporting,
                    "contradicting": cb.contradicting,
                    "strength": cb.strength,
                }
                for cb in self.consensus_badges
            ],
            "source_badges": self.source_badges,
            "uncertainty_flags": self.uncertainty_flags,
            "is_abstention": self.is_abstention,
        }


def render_citations(
    verification: VerificationResult,
    evidence_table: EvidenceTable,
) -> RenderedAnswer:
    """
    Build the full rendered answer with citation metadata.
    Called after verification; produces all data the frontend needs.
    """
    row_map = {r.evidence_id: r for r in evidence_table.rows}

    # Build citation hover cards
    cards: list[CitationHoverCard] = []
    for eid, num in verification.citation_map.items():
        row = row_map.get(eid)
        if not row:
            continue
        cards.append(CitationHoverCard(
            number=num,
            paper_title=row.paper_title,
            authors=row.authors,
            year=row.year,
            venue=row.venue,
            passage_preview=row.chunk_text[:200],
            section=row.section_tag,
            is_peer_reviewed=row.is_peer_reviewed,
            is_retracted=row.is_retracted,
            source_url=row.source_url or None,
            pdf_url=row.pdf_url or None,
        ))

    cards.sort(key=lambda c: c.number)

    # Compute source badges
    source_badges = {
        "total_sources": evidence_table.total_sources,
        "peer_reviewed": evidence_table.peer_reviewed_count,
        "preprints": evidence_table.preprint_count,
        "retracted": sum(1 for r in evidence_table.rows if r.is_retracted),
    }

    return RenderedAnswer(
        markdown_text=verification.cleaned_text,
        citation_cards=cards,
        source_badges=source_badges,
        uncertainty_flags=verification.warnings,
    )
