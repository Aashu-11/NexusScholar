"""
evidence_builder.py — Evidence table construction.
Step 2: Before the LLM writes a single word, an evidence table is built
from all retrieved chunks. This is the "truth substrate" that constrains synthesis.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Optional

from backend.retrieval.hybrid_recall import RetrievalCandidate
from backend.integrations.source_urls import build_source_url, build_pdf_url


@dataclass
class EvidenceRow:
    evidence_id: str = ""
    chunk_id: str = ""
    paper_id: str = ""
    paper_title: str = ""
    authors: str = ""
    year: Optional[int] = None
    venue: Optional[str] = None
    section_tag: str = "unknown"
    chunk_text: str = ""
    relevance_score: float = 0.0
    is_peer_reviewed: bool = False
    is_preprint: bool = False
    is_retracted: bool = False
    source_url: str = ""
    pdf_url: str = ""
    supporting_span: str = ""
    parent_context: str = ""
    nli_entailment_score: float = 0.0

    def __post_init__(self):
        if not self.evidence_id:
            self.evidence_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "section_tag": self.section_tag,
            "chunk_text": self.chunk_text,
            "relevance_score": self.relevance_score,
            "is_peer_reviewed": self.is_peer_reviewed,
            "is_preprint": self.is_preprint,
            "is_retracted": self.is_retracted,
            "source_url": self.source_url,
            "pdf_url": self.pdf_url,
            "supporting_span": self.supporting_span,
            "parent_context": self.parent_context,
            "nli_entailment_score": self.nli_entailment_score,
        }


@dataclass
class EvidenceTable:
    answer_id: str = ""
    query: str = ""
    intent: str = "general"
    rows: list[EvidenceRow] = field(default_factory=list)
    confidence_score: float = 0.0

    def __post_init__(self):
        if not self.answer_id:
            self.answer_id = uuid.uuid4().hex[:12]

    @property
    def total_sources(self) -> int:
        return len(self.rows)

    @property
    def peer_reviewed_count(self) -> int:
        return sum(1 for r in self.rows if r.is_peer_reviewed)

    @property
    def preprint_count(self) -> int:
        return sum(1 for r in self.rows if r.is_preprint)

    @property
    def has_retracted(self) -> bool:
        return any(r.is_retracted for r in self.rows)

    def to_dict(self) -> dict:
        return {
            "answer_id": self.answer_id,
            "query": self.query,
            "intent": self.intent,
            "rows": [r.to_dict() for r in self.rows],
            "confidence_score": self.confidence_score,
        }

    def to_llm_context(self) -> list[dict]:
        """Format for injection into the synthesizer prompt — maximizes evidence available to LLM."""
        return [
            {
                "evidence_id": r.evidence_id,
                "paper_title": r.paper_title,
                "authors": r.authors,
                "year": r.year,
                "venue": r.venue or "Unknown",
                "section": r.section_tag,
                "text": r.chunk_text[:2000],
                "parent_context": r.parent_context[:1000] if r.parent_context else None,
                "relevance": round(r.relevance_score, 3),
                "peer_reviewed": r.is_peer_reviewed,
                "retracted": r.is_retracted,
                "source_url": r.source_url,
                "pdf_url": r.pdf_url,
            }
            for r in self.rows
        ]


def build_evidence_table(
    query: str,
    intent: str,
    candidates: list[RetrievalCandidate],
) -> EvidenceTable:
    """
    Construct the evidence table from retrieval candidates.
    Each candidate becomes a row with full metadata.
    """
    rows: list[EvidenceRow] = []

    for cand in candidates:
        paper = cand.paper or {}
        authors_list = paper.get("authors", [])

        # Format author string
        if isinstance(authors_list, list):
            names = []
            for a in authors_list[:3]:
                if isinstance(a, dict):
                    names.append(a.get("name", ""))
                elif isinstance(a, str):
                    names.append(a)
            authors_str = ", ".join(n for n in names if n)
            if len(authors_list) > 3:
                authors_str += " et al."
        else:
            authors_str = str(authors_list)

        is_peer_reviewed = paper.get("is_peer_reviewed", False)
        source_url = build_source_url(paper)
        pdf_url = build_pdf_url(paper)

        rows.append(EvidenceRow(
            chunk_id=cand.chunk["chunk_id"],
            paper_id=cand.chunk["paper_id"],
            paper_title=paper.get("title", "Unknown"),
            authors=authors_str,
            year=paper.get("year"),
            venue=paper.get("venue"),
            section_tag=cand.chunk.get("section_tag", "unknown"),
            chunk_text=cand.chunk["text"],
            relevance_score=cand.final_score,
            is_peer_reviewed=is_peer_reviewed,
            is_preprint=not is_peer_reviewed,
            is_retracted=paper.get("is_retracted", False),
            source_url=source_url,
            pdf_url=pdf_url,
            parent_context=cand.chunk.get("parent_context", ""),
            supporting_span=cand.chunk["text"][:200],
        ))

    # Compute confidence: sigmoid-normalized average of top-5 relevance scores
    # This maps arbitrary score ranges to a consistent 0-1 scale
    import math

    confidence = 0.0
    if rows:
        top_scores = [1.0 / (1.0 + math.exp(-r.relevance_score)) for r in rows[:5]]
        confidence = sum(top_scores) / len(top_scores)

    return EvidenceTable(
        query=query,
        intent=intent,
        rows=rows,
        confidence_score=confidence,
    )
