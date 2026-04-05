"""
hybrid_recall.py — Parallel BM25 + Dense retrieval with RRF fusion.
Runs both modalities in parallel, then fuses via Reciprocal Rank Fusion.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.config import settings
from backend.indexing.bm25_index import BM25Index
from backend.indexing.dense_index import DenseIndex
from backend.retrieval.query_rewriter import QueryAnalysis
from backend.indexing.colbert_index import colbert_search, is_available as colbert_available
from backend.retrieval.entity_extractor import QueryEntityProfile

logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
    """A single retrieval candidate with multi-signal scores."""
    chunk: dict
    paper: dict = field(default_factory=dict)
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    is_graph_expanded: bool = False
    # Set by reranker when entity grounding is active.
    # Values: "correct" | "wrong" | "ambiguous" | "neutral" | None
    entity_decision: Optional[str] = None


def hybrid_recall(
    query_analysis: QueryAnalysis,
    bm25_index: BM25Index,
    dense_index: DenseIndex,
) -> list[RetrievalCandidate]:
    """
    Stage B: Run parallel BM25 + Dense retrieval, then fuse via RRF.
    Returns top FUSED_TOP_K candidates.
    """
    # BM25 recall on keyword query
    bm25_queries = query_analysis.bm25_queries or [query_analysis.bm25_query or query_analysis.original_query]
    dense_queries = query_analysis.dense_queries or [query_analysis.dense_query or query_analysis.original_query]

    bm25_results = _merge_ranked_results(
        [bm25_index.search(query, top_k=settings.BM25_TOP_K) for query in bm25_queries],
        top_k=settings.BM25_TOP_K,
    )
    logger.info("BM25 recall returned %s hits across %s expanded queries", len(bm25_results), len(bm25_queries))

    # Dense recall on semantic query
    dense_results = _merge_ranked_results(
        [dense_index.search(query, top_k=settings.DENSE_TOP_K) for query in dense_queries],
        top_k=settings.DENSE_TOP_K,
    )
    logger.info("Dense recall returned %s hits across %s expanded queries", len(dense_results), len(dense_queries))

    # HyDE retrieval lane (if available)
    hyde_results: list[tuple[dict, float, int]] = []
    if query_analysis.hyde_embedding is not None:
        hyde_results = _hyde_search(dense_index, query_analysis.hyde_embedding, top_k=settings.DENSE_TOP_K)
        logger.info("HyDE recall returned %s hits", len(hyde_results))

    # RRF Fusion — include HyDE and ColBERT lanes when available
    all_result_lists = [bm25_results, dense_results]
    if hyde_results:
        all_result_lists.append(hyde_results)

    # ColBERT late-interaction lane (optional — only if index built and enabled)
    if settings.COLBERT_ENABLED and colbert_available():
        try:
            # Build a chunk_lookup from existing results for score assembly
            chunk_lookup = {
                c["chunk_id"]: c
                for results in [bm25_results, dense_results]
                for c, _, _ in results
            }
            colbert_results = colbert_search(
                query_analysis.dense_query or query_analysis.original_query,
                top_k=settings.COLBERT_TOP_K,
                chunk_lookup=chunk_lookup,
            )
            if colbert_results:
                logger.info("ColBERT recall returned %s hits", len(colbert_results))
                all_result_lists.append(colbert_results)
        except Exception as e:
            logger.warning("ColBERT recall failed: %s", e)

    fused = reciprocal_rank_fusion(
        all_result_lists,
        k=settings.RRF_K,
        top_k=settings.FUSED_TOP_K,
    )
    logger.info("RRF fused candidate count: %s", len(fused))

    # MMR deduplication — balance relevance vs diversity
    fused = mmr_dedup(fused, lambda_param=0.7, top_k=settings.FUSED_TOP_K)
    logger.info("MMR dedup kept %s candidates", len(fused))

    for idx, cand in enumerate(fused[:5], start=1):
        logger.info(
            "Fused candidate %s: chunk=%s paper=%s bm25_rank=%s dense_rank=%s rrf=%.4f",
            idx,
            cand.chunk.get("chunk_id"),
            cand.chunk.get("paper_id"),
            cand.bm25_rank,
            cand.dense_rank,
            cand.rrf_score,
        )
    return fused


def _hyde_search(
    dense_index: DenseIndex,
    hyde_embedding: np.ndarray,
    top_k: int = 100,
) -> list[tuple[dict, float, int]]:
    """Search the dense index directly with a precomputed HyDE embedding."""
    if not dense_index.chunks or dense_index.vectors is None:
        return []

    q_vec = hyde_embedding.astype(np.float32).reshape(1, -1)

    if dense_index._use_faiss and dense_index._faiss_index:
        import faiss
        faiss.normalize_L2(q_vec)
        scores, indices = dense_index._faiss_index.search(q_vec, min(top_k, len(dense_index.chunks)))
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx >= 0:
                results.append((dense_index.chunks[idx], float(score), rank))
        return results
    else:
        return dense_index._numpy_search(q_vec[0], top_k)


def _merge_ranked_results(
    result_lists: list[list[tuple[dict, float, int]]],
    top_k: int,
) -> list[tuple[dict, float, int]]:
    merged: dict[str, tuple[dict, float, int]] = {}
    for results in result_lists:
        for chunk, score, rank in results:
            chunk_id = chunk["chunk_id"]
            current = merged.get(chunk_id)
            if current is None or score > current[1]:
                merged[chunk_id] = (chunk, float(score), rank)
    sorted_results = sorted(
        merged.values(),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        (chunk, score, idx)
        for idx, (chunk, score, _rank) in enumerate(sorted_results[:top_k])
    ]


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[dict, float, int]]],
    k: int = 60,
    top_k: int = 200,
    alpha: float | None = None,
) -> list[RetrievalCandidate]:
    """
    Fuse multiple ranked lists using weighted Reciprocal Rank Fusion.

    list_idx=0 is BM25 (weight = 1-alpha).
    list_idx=1+ are dense lanes (HyDE, ColBERT — weight = alpha).
    Formula: RRF(d) = Σ w_i / (k + rank_in_modality_i)
    """
    if alpha is None:
        alpha = settings.HYBRID_ALPHA

    scores: dict[str, dict] = {}

    for list_idx, results in enumerate(result_lists):
        # BM25 lane gets (1-alpha) weight; dense/HyDE/ColBERT lanes get alpha weight
        modality_weight = (1.0 - alpha) if list_idx == 0 else alpha

        for chunk, score, rank in results:
            cid = chunk["chunk_id"]
            if cid not in scores:
                scores[cid] = {
                    "chunk": chunk,
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "dense_rank": None,
                }
            scores[cid]["rrf_score"] += modality_weight / (k + rank)

            if list_idx == 0:
                scores[cid]["bm25_rank"] = rank
            elif list_idx == 1:
                scores[cid]["dense_rank"] = rank

    sorted_items = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]

    return [
        RetrievalCandidate(
            chunk=item["chunk"],
            bm25_rank=item["bm25_rank"],
            dense_rank=item["dense_rank"],
            rrf_score=item["rrf_score"],
        )
        for item in sorted_items
    ]


def mmr_dedup(
    candidates: list[RetrievalCandidate],
    lambda_param: float = 0.7,
    top_k: int = 80,
    entity_profile: Optional[QueryEntityProfile] = None,
) -> list[RetrievalCandidate]:
    """
    Maximal Marginal Relevance — balance relevance vs diversity.

    Vectorized implementation: one TF-IDF fit over ALL texts, then pure
    matrix operations per iteration.  Complexity: O(n²) one-time setup +
    O(n × selected) per step — vs the previous O(n² × k) TF-IDF fits.
    For n=231 and k=80 that is ~80× fewer vectorizer calls.
    """
    if not candidates:
        return []

    if entity_profile and entity_profile.requires_entity_grounding:
        lambda_param = max(lambda_param, 0.85)

    n = len(candidates)
    rrf_scores = np.array([c.rrf_score for c in candidates], dtype=np.float32)

    # ── Build normalized TF-IDF matrix once ──────────────────────────
    texts = [c.chunk.get("text", "") or "" for c in candidates]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf_mat = TfidfVectorizer(
            max_features=256,
            sublinear_tf=True,
        ).fit_transform(texts).toarray()  # type: ignore[union-attr]
        norms = np.linalg.norm(tfidf_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = (tfidf_mat / norms).astype(np.float32)
        # Full cosine similarity matrix — O(n²) one-time cost
        # For n=300 this is 300×300 float32 = 360 KB, trivial.
        sim_matrix: np.ndarray = mat @ mat.T
    except Exception as exc:
        logger.debug("MMR vectorization failed (%s) — falling back to RRF sort", exc)
        return sorted(candidates, key=lambda c: c.rrf_score, reverse=True)[:top_k]

    # ── Greedy MMR selection using pre-computed similarity ────────────
    selected_idx: list[int] = [0]
    remaining_mask = np.ones(n, dtype=bool)
    remaining_mask[0] = False

    while remaining_mask.any() and len(selected_idx) < top_k:
        rem = np.where(remaining_mask)[0]
        # Max similarity of each remaining candidate to any already-selected doc
        sel_arr = np.array(selected_idx, dtype=np.intp)
        # sim_matrix[rem][:, sel_arr] → rows=remaining, cols=selected → shape (|rem|, |sel|)
        max_sim_to_selected = sim_matrix[rem][:, sel_arr].max(axis=1)
        mmr_scores = lambda_param * rrf_scores[rem] - (1 - lambda_param) * max_sim_to_selected
        best_local = int(np.argmax(mmr_scores))
        best_global = rem[best_local]
        selected_idx.append(int(best_global))
        remaining_mask[best_global] = False

    return [candidates[i] for i in selected_idx]
