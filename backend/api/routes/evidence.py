"""
/api/evidence — Evidence table retrieval for past answers.
/api/conversations — Conversation history management.
/api/indexes — Index management.
/api/health/pipeline — Pipeline health check.
/api/audit/missing-years — Corpus year coverage audit.
"""

from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/evidence/{answer_id}")
async def get_evidence(answer_id: str):
    """Return the complete evidence table for a past answer."""
    from backend.main import get_store
    store = get_store()

    et = await store.get_evidence_table(answer_id)
    if not et:
        raise HTTPException(404, "Evidence table not found")
    return et


@router.get("/conversations")
async def list_conversations():
    """List all conversations."""
    from backend.main import get_store
    store = get_store()
    return await store.get_conversations()


@router.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str):
    """Get all messages for a conversation."""
    from backend.main import get_store
    store = get_store()
    return await store.get_messages(conversation_id)


@router.post("/indexes/rebuild")
async def rebuild_indexes():
    """Manually rebuild all retrieval indexes."""
    from backend.main import get_store, get_bm25, get_dense
    store = get_store()
    bm25 = get_bm25()
    dense = get_dense()

    await bm25.build(store, granularity="passage", extra_granularities=["table"])
    await dense.build(store, granularity="passage")

    return {
        "status": "rebuilt",
        "bm25_size": bm25.size,
        "dense_size": dense.size,
    }


@router.get("/health/pipeline")
async def pipeline_health():
    """
    Detailed pipeline health check. Reports status of all components.
    Use this to verify all improvements are active after deployment.
    """
    from backend.main import get_store, get_bm25, get_dense, get_reranker
    from backend.indexing.colbert_index import is_available as colbert_available
    from backend.ingestion.graph_builder import get_graph, compute_pagerank
    from backend.config import settings

    store = get_store()
    bm25 = get_bm25()
    dense = get_dense()
    graph = get_graph()

    try:
        pr_scores = compute_pagerank()
    except Exception as e:
        logger.warning("PageRank computation failed in health check: %s", e)
        pr_scores = {}

    papers = await store.get_all_papers()
    all_chunks = await store.get_all_chunks()
    table_chunks = [c for c in all_chunks if c.get("granularity") == "table"]

    return {
        "status": "healthy",
        "indexes": {
            "bm25_size": bm25.size,
            "dense_size": dense.size,
            "colbert_available": colbert_available(),
            "colbert_enabled": settings.COLBERT_ENABLED,
        },
        "corpus": {
            "total_papers": len(papers),
            "papers_with_year": sum(1 for p in papers if p.get("year")),
            "papers_missing_year": sum(1 for p in papers if not p.get("year")),
            "total_chunks": len(all_chunks),
            "table_chunks": len(table_chunks),
            "chunk_granularities": {
                g: sum(1 for c in all_chunks if c.get("granularity") == g)
                for g in ("document", "section", "passage", "claim", "table")
            },
        },
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "pagerank_computed": len(pr_scores) > 0,
            "top_pagerank_paper": (
                max(pr_scores.items(), key=lambda x: x[1])[0] if pr_scores else None
            ),
        },
        "config": {
            "tavily_auto_fetch": settings.TAVILY_AUTO_FETCH,
            "s2_auto_fetch": settings.S2_AUTO_FETCH,
            "tavily_topic": settings.TAVILY_TOPIC,
            "reranker_model": settings.RERANKER_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "synthesis_temperature": settings.SYNTHESIS_TEMPERATURE,
            "final_evidence_top_k": settings.FINAL_EVIDENCE_TOP_K,
        },
        "recency": {
            "papers_2024_or_newer": sum(1 for p in papers if (p.get("year") or 0) >= 2024),
            "papers_2023": sum(1 for p in papers if (p.get("year") or 0) == 2023),
            "papers_pre_2020": sum(
                1 for p in papers if 0 < (p.get("year") or 9999) < 2020
            ),
        },
    }


@router.get("/audit/missing-years")
async def audit_missing_years():
    """
    Return all papers in the corpus that are missing year metadata.
    Use this to identify papers that need manual metadata correction or re-ingestion.
    """
    from backend.main import get_store
    store = get_store()
    all_papers = await store.get_all_papers()

    missing = [
        {
            "paper_id": p["paper_id"],
            "title": p["title"],
            "venue": p.get("venue"),
            "arxiv_id": p.get("arxiv_id"),
            "source_url": p.get("source_url"),
            "ingested_at": p.get("ingested_at"),
        }
        for p in all_papers
        if not p.get("year")
    ]

    with_year = [p for p in all_papers if p.get("year")]
    year_distribution: dict[str, int] = {}
    for p in with_year:
        y = str(p["year"])
        year_distribution[y] = year_distribution.get(y, 0) + 1

    return {
        "total_papers": len(all_papers),
        "papers_with_year": len(with_year),
        "papers_missing_year": len(missing),
        "missing_year_fraction": round(len(missing) / max(len(all_papers), 1), 3),
        "year_distribution": dict(sorted(year_distribution.items(), reverse=True)),
        "papers_missing_year_list": missing,
    }


@router.patch("/papers/{paper_id}/year")
async def patch_paper_year(paper_id: str, year: int):
    """
    Manually set the year for a paper that is missing year metadata.
    Use in conjunction with /audit/missing-years to repair the corpus.
    """
    from backend.main import get_store
    store = get_store()

    paper = await store.get_paper(paper_id)
    if not paper:
        raise HTTPException(404, "Paper not found")

    if not (1900 <= year <= 2030):
        raise HTTPException(400, f"Year {year} is out of valid range (1900-2030)")

    await store.update_paper_metadata(paper_id, {"year": year})
    logger.info("Manually patched year for paper %s: year=%s", paper_id, year)
    return {"paper_id": paper_id, "year": year, "status": "updated"}