"""
chunker.py — Multi-granular chunking strategy.
Maintains 5 parallel chunk representations of every paper:
  1. Document-level (abstract + conclusion)
  2. Section-level (full section)
  3. Passage-level (~256 token sliding window)
  4. Claim-level (individual claim-bearing sentences)
  5. Table-level (structured tables extracted from markdown)
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Optional

from backend.config import settings
from backend.ingestion.pdf_parser import ParsedPaper
from backend.ingestion.claim_extractor import extract_claims
from backend.ingestion.table_extractor import extract_tables_from_markdown


@dataclass
class Chunk:
    chunk_id: str = ""
    paper_id: str = ""
    granularity: str = ""      # document | section | passage | claim
    section_tag: str = "unknown"
    text: str = ""
    token_count: int = 0
    embedding: Optional[list[float]] = None
    start_char: int = 0
    end_char: int = 0

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = uuid.uuid4().hex[:12]
        if not self.token_count:
            self.token_count = estimate_tokens(self.text)


def chunk_paper(paper: ParsedPaper) -> list[Chunk]:
    """
    Generate all 4 granularity levels for a parsed paper.
    Returns a flat list; downstream processes index by granularity.
    """
    chunks: list[Chunk] = []

    # ── Level 1: Document-level ───────────────────────────
    doc_text = paper.abstract
    conclusion = paper.sections.get("conclusion", "")
    if conclusion:
        doc_text += "\n\n" + conclusion
    if doc_text.strip():
        chunks.append(Chunk(
            paper_id=paper.paper_id,
            granularity="document",
            section_tag="abstract",
            text=doc_text.strip(),
        ))

    # ── Level 2: Section-level ────────────────────────────
    for tag, text in paper.sections.items():
        if text.strip():
            chunks.append(Chunk(
                paper_id=paper.paper_id,
                granularity="section",
                section_tag=tag,
                text=text.strip(),
            ))

    # ── Level 3: Passage-level (sliding window) ──────────
    for tag, text in paper.sections.items():
        passages = sliding_window_chunk(
            text,
            window_tokens=settings.PASSAGE_CHUNK_TOKENS,
            stride_tokens=settings.PASSAGE_STRIDE_TOKENS,
        )
        for passage_text, start, end in passages:
            chunks.append(Chunk(
                paper_id=paper.paper_id,
                granularity="passage",
                section_tag=tag,
                text=passage_text,
                start_char=start,
                end_char=end,
            ))

    # ── Level 4: Claim-level ─────────────────────────────
    for tag, text in paper.sections.items():
        claims = extract_claims(text)
        for claim_text in claims:
            chunks.append(Chunk(
                paper_id=paper.paper_id,
                granularity="claim",
                section_tag=tag,
                text=claim_text,
            ))

    # ── Level 5: Table-level ─────────────────────────────
    # Extract structured tables from markdown-formatted sections.
    # Tables are serialized to key:value text for BM25/dense indexing.
    for tag, text in paper.sections.items():
        try:
            tables = extract_tables_from_markdown(text, section_tag=tag)
            for table in tables:
                chunks.append(Chunk(
                    paper_id=paper.paper_id,
                    granularity="table",
                    section_tag=tag,
                    text=table["text"],
                ))
        except Exception:
            # Table extraction is best-effort — never block chunking
            pass

    return chunks


def sliding_window_chunk(
    text: str,
    window_tokens: int = 256,
    stride_tokens: int = 128,
) -> list[tuple[str, int, int]]:
    """
    Produce overlapping passage chunks using a sliding window.
    Returns list of (text, start_char, end_char).
    """
    words = text.split()
    window_words = token_to_word_count(window_tokens)
    stride_words = token_to_word_count(stride_tokens)

    if len(words) <= window_words:
        return [(text.strip(), 0, len(text))] if text.strip() else []

    # Pre-compute word start positions for accurate char offsets
    word_starts = []
    pos = 0
    for word in words:
        idx = text.find(word, pos)
        word_starts.append(idx if idx >= 0 else pos)
        pos = (idx if idx >= 0 else pos) + len(word)

    results: list[tuple[str, int, int]] = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + window_words]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            start = word_starts[i]
            last_word_idx = min(i + window_words - 1, len(words) - 1)
            end = word_starts[last_word_idx] + len(words[last_word_idx])
            results.append((chunk_text, start, end))
        i += stride_words

    return results


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 words per token for English."""
    return max(1, int(len(text.split()) * 4 / 3))


def token_to_word_count(tokens: int) -> int:
    """Convert token count to approximate word count."""
    return max(1, int(tokens * 3 / 4))