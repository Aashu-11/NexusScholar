"""
graph_expander.py — Citation graph expansion after re-ranking.
Expands the top-15 evidence set via:
  - cited-by expansion
  - references expansion (seminal works)
  - co-citation clusters
  - shared dataset neighbors
  - same venue + topic cluster
"""

from __future__ import annotations
import logging
from typing import Optional  # noqa: F401

from backend.config import settings
from backend.indexing.metadata_store import MetadataStore
from backend.ingestion.graph_builder import (
    get_cited_by, get_references, get_co_citation_cluster, get_pagerank,
)
from backend.retrieval.hybrid_recall import RetrievalCandidate

logger = logging.getLogger(__name__)


async def graph_expand(
    top_candidates: list[RetrievalCandidate],
    store: MetadataStore,
    limit: Optional[int] = None,
) -> list[RetrievalCandidate]:
    """
    Expand evidence via citation graph neighbors of the top re-ranked results.
    Returns additional candidates (up to GRAPH_EXPANSION_LIMIT) from:
      1. Papers that cite the top results (cited-by)
      2. Papers cited by top results (references / seminal works)
      3. Co-citation clusters
    """
    limit = limit or settings.GRAPH_EXPANSION_LIMIT
    seen_papers = {c.chunk["paper_id"] for c in top_candidates}
    expansion_candidates: list[RetrievalCandidate] = []

    seed_papers = list({c.chunk["paper_id"] for c in top_candidates[:8]})

    for paper_id in seed_papers:
        if len(expansion_candidates) >= limit:
            break

        # 1. Cited-by expansion
        try:
            cited_by_ids = get_cited_by(paper_id)
        except Exception as exc:
            logger.debug("graph_expand: get_cited_by(%s) failed: %s", paper_id, exc)
            cited_by_ids = []
        for citer_id in cited_by_ids[:5]:
            if citer_id in seen_papers:
                continue
            seen_papers.add(citer_id)
            chunks = await _get_top_chunks(citer_id, store)
            for chunk in chunks:
                expansion_candidates.append(_make_candidate(chunk, is_expanded=True))
            if len(expansion_candidates) >= limit:
                break

        # 2. References expansion (seminal works)
        try:
            ref_ids = get_references(paper_id)
        except Exception as exc:
            logger.debug("graph_expand: get_references(%s) failed: %s", paper_id, exc)
            ref_ids = []
        for ref_id in ref_ids[:5]:
            if ref_id in seen_papers:
                continue
            seen_papers.add(ref_id)
            chunks = await _get_top_chunks(ref_id, store)
            for chunk in chunks:
                expansion_candidates.append(_make_candidate(chunk, is_expanded=True))
            if len(expansion_candidates) >= limit:
                break

        # 3. Co-citation clusters
        try:
            co_cited = get_co_citation_cluster(paper_id, max_size=10)
        except Exception as exc:
            logger.debug("graph_expand: get_co_citation_cluster(%s) failed: %s", paper_id, exc)
            co_cited = []
        for cc_id in co_cited[:3]:
            if cc_id in seen_papers:
                continue
            seen_papers.add(cc_id)
            chunks = await _get_top_chunks(cc_id, store)
            for chunk in chunks:
                expansion_candidates.append(_make_candidate(chunk, is_expanded=True))
            if len(expansion_candidates) >= limit:
                break

    logger.info("Graph expansion yielded %d additional candidates", len(expansion_candidates))
    return expansion_candidates[:limit]


async def _get_top_chunks(paper_id: str, store: MetadataStore, max_chunks: int = 3) -> list[dict]:
    """Get the most relevant passage-level chunks for a paper."""
    try:
        chunks = await store.get_chunks_by_paper(paper_id, granularity="passage")
        return chunks[:max_chunks]
    except Exception as exc:
        logger.debug("graph_expand: get_chunks_by_paper(%s) failed: %s", paper_id, exc)
        return []


def _make_candidate(chunk: dict, is_expanded: bool = False) -> RetrievalCandidate:
    import math
    paper_id = chunk.get("paper_id", "")
    # FIX: Use PageRank to give expanded nodes a non-zero base score weighted by authority.
    # Previously all expanded candidates had rrf_score=0.0 which made them rank last.
    pr_score = get_pagerank(paper_id)
    pr_boost = 1.0 + math.log1p(pr_score * 10)  # log-scale so outliers don't dominate
    return RetrievalCandidate(
        chunk=chunk,
        rrf_score=0.05 * pr_boost,
        is_graph_expanded=is_expanded,
    )