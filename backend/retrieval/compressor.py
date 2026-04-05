"""
compressor.py — Contextual compression via TF-IDF sentence scoring.

Key design rules:
1. ZERO LLM calls — eliminates all 429/413 errors from compression stage.
2. TF-IDF cosine similarity scores each sentence against the query.
3. Entity decisions from the reranker are respected (fast path for wrong/ambiguous).
4. Chunk text truncated to MAX_CHUNK_CHARS before processing large section chunks.
"""

from __future__ import annotations
import logging
import re
from typing import Optional

from backend.retrieval.hybrid_recall import RetrievalCandidate
from backend.retrieval.entity_extractor import QueryEntityProfile

logger = logging.getLogger(__name__)

# Maximum characters processed per chunk.
_MAX_CHUNK_CHARS = 3000

# Minimum TF-IDF cosine score for a sentence to be kept.
_SENTENCE_SCORE_THRESHOLD = 0.05

# Minimum sentence length (chars) — filters out headers, labels, etc.
_MIN_SENTENCE_LEN = 30

# Sentence splitter: split on sentence-ending punctuation followed by whitespace.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _tfidf_extract(query: str, text: str) -> str:
    """
    Return the sentences from text most relevant to the query (TF-IDF cosine).
    Preserves original sentence order. Falls back to original text on any error.
    """
    sentences = [
        s.strip() for s in _SENTENCE_RE.split(text)
        if len(s.strip()) >= _MIN_SENTENCE_LEN
    ]
    if len(sentences) <= 2:
        return text  # too short to compress — return as-is

    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = [query] + sentences
        vec = TfidfVectorizer(
            max_features=512,
            ngram_range=(1, 2),
            sublinear_tf=True,
        ).fit_transform(corpus).toarray()  # type: ignore[union-attr]

        query_vec = vec[0]
        sent_vecs = vec[1:]

        q_norm = float(np.linalg.norm(query_vec)) or 1.0
        row_norms = np.linalg.norm(sent_vecs, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        scores = (sent_vecs / row_norms) @ (query_vec / q_norm)

        selected = [
            sentences[i] for i, s in enumerate(scores)
            if s >= _SENTENCE_SCORE_THRESHOLD
        ]

        if not selected:
            return text  # nothing cleared threshold — keep original

        return " ".join(selected)

    except Exception as exc:
        logger.debug("TF-IDF extraction failed (%s) — keeping original text", exc)
        return text


def _compress_one(
    query: str,
    cand: RetrievalCandidate,
) -> RetrievalCandidate:
    """
    Compress a single candidate synchronously via TF-IDF.

    Fast paths (no scoring):
    - entity_decision="wrong"     → multiply score × 0.1, return
    - entity_decision="ambiguous" → multiply score × 0.5, return
    - text ≤ 2 sentences          → return as-is (nothing to compress)

    All other cases: TF-IDF sentence extraction.
    """
    chunk_id = cand.chunk.get("chunk_id", "?")

    # Fast path: reranker already decided this chunk is the wrong entity
    if cand.entity_decision == "wrong":
        logger.debug("Compressor: chunk=%s wrong entity — penalising", chunk_id)
        cand.final_score *= 0.1
        return cand

    # Fast path: ambiguous entity — mild penalty, no extraction needed
    if cand.entity_decision == "ambiguous":
        cand.final_score *= 0.5
        return cand

    original_text = cand.chunk.get("text", "")
    if not original_text.strip():
        return cand

    # Truncate oversized chunks before TF-IDF to keep memory bounded
    text = original_text[:_MAX_CHUNK_CHARS]
    if len(original_text) > _MAX_CHUNK_CHARS:
        logger.debug(
            "Compressor: chunk=%s truncated %d→%d chars before scoring",
            chunk_id, len(original_text), _MAX_CHUNK_CHARS,
        )

    compressed = _tfidf_extract(query, text)

    if compressed == text:
        return cand  # no change

    return RetrievalCandidate(
        chunk={**cand.chunk, "text": compressed},
        paper=cand.paper,
        rrf_score=cand.rrf_score,
        rerank_score=cand.rerank_score,
        final_score=cand.final_score,
        entity_decision=cand.entity_decision,
    )


async def compress_chunks(
    query: str,
    candidates: list[RetrievalCandidate],
    groq=None,  # noqa: ARG001  # kept for backward-compat; not used (TF-IDF is synchronous)
    max_candidates: int = 20,
    entity_profile: Optional[QueryEntityProfile] = None,  # noqa: ARG001  # reranker decisions already applied
) -> list[RetrievalCandidate]:
    """
    Compress up to max_candidates chunks via TF-IDF sentence scoring.
    Synchronous internally — no asyncio concurrency needed since there are no I/O calls.
    """
    batch = candidates[:max_candidates]
    if not batch:
        return []

    compressed: list[RetrievalCandidate] = []
    for cand in batch:
        try:
            compressed.append(_compress_one(query, cand))
        except Exception as exc:
            logger.warning(
                "Compressor: exception for chunk=%s: %s",
                cand.chunk.get("chunk_id"), exc,
            )
            compressed.append(cand)

    wrong_count = sum(1 for c in compressed if c.entity_decision == "wrong")
    logger.info(
        "Compression complete: %d candidates processed (%d wrong-entity penalised)",
        len(compressed), wrong_count,
    )
    return compressed or candidates[:max_candidates]
