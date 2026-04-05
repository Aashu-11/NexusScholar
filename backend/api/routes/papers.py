"""
/api/papers — Paper search, metadata, and discovery endpoints.
"""

from __future__ import annotations
from fastapi import APIRouter, Query
from typing import Optional

from backend.integrations.source_urls import build_source_url, build_pdf_url

router = APIRouter()


@router.get("/papers/search")
async def search_papers(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    peer_reviewed_only: bool = False,
    venue: Optional[str] = None,
):
    """Search indexed papers by title/abstract with filters."""
    from backend.main import get_store
    store = get_store()

    papers = await store.search_papers(
        q, limit=limit, year_min=year_min, year_max=year_max,
        peer_reviewed_only=peer_reviewed_only, venue=venue,
    )
    return {
        "total": len(papers),
        "papers": [
            {
                "paper_id": p["paper_id"],
                "title": p["title"],
                "authors": [
                    a.get("name", a) if isinstance(a, dict) else a
                    for a in p.get("authors", [])
                ],
                "year": p.get("year"),
                "venue": p.get("venue"),
                "abstract": p.get("abstract", "")[:300],
                "citation_count": p.get("citation_count", 0),
                "is_peer_reviewed": p.get("is_peer_reviewed", False),
                "arxiv_id": p.get("arxiv_id"),
                "source_url": build_source_url(p),
                "pdf_url": build_pdf_url(p),
            }
            for p in papers
        ],
    }


@router.get("/papers/{paper_id}")
async def get_paper(paper_id: str):
    """Get full metadata for a single paper."""
    from backend.main import get_store
    store = get_store()
    paper = await store.get_paper(paper_id)
    if not paper:
        from fastapi import HTTPException
        raise HTTPException(404, "Paper not found")
    paper["source_url"] = build_source_url(paper)
    paper["pdf_url"] = build_pdf_url(paper)
    return paper


@router.get("/papers")
async def list_papers(limit: int = Query(50, ge=1, le=500)):
    """List all ingested papers."""
    from backend.main import get_store
    store = get_store()
    papers = await store.get_all_papers()
    return {
        "total": len(papers),
        "papers": [
            {
                "paper_id": p["paper_id"],
                "title": p["title"],
                "year": p.get("year"),
                "authors": p.get("authors", []),
                "arxiv_id": p.get("arxiv_id"),
                "source_url": build_source_url(p),
                "pdf_url": build_pdf_url(p),
            }
            for p in papers[:limit]
        ],
    }
