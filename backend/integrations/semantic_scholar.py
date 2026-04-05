"""
semantic_scholar.py — Semantic Scholar API client.
Provides structured metadata: citation counts, year, venue, DOIs, open access PDFs.
"""

from __future__ import annotations
import logging
from typing import Optional

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = (
    "title,authors,year,venue,citationCount,abstract,"
    "externalIds,isOpenAccess,openAccessPdf"
)


async def search_semantic_scholar(
    query: str,
    limit: int = 10,
) -> list[dict]:
    """Search Semantic Scholar for papers matching *query*.

    Returns a list of normalised paper dicts with keys:
        title, year, venue, citation_count, abstract, arxiv_id, doi,
        is_peer_reviewed, pdf_url, authors.
    """
    headers: dict[str, str] = {}
    if settings.S2_API_KEY:
        headers["x-api-key"] = settings.S2_API_KEY

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{S2_API_BASE}/paper/search",
                params={
                    "query": query,
                    "limit": limit,
                    "fields": S2_FIELDS,
                },
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("Semantic Scholar search failed for %r: %s", query, exc)
        return []

    papers: list[dict] = []
    for p in data.get("data", []):
        external_ids = p.get("externalIds") or {}
        open_access_pdf = p.get("openAccessPdf") or {}
        authors_raw = p.get("authors") or []

        papers.append({
            "title": p.get("title", ""),
            "year": p.get("year"),
            "venue": p.get("venue"),
            "citation_count": p.get("citationCount", 0),
            "abstract": p.get("abstract", ""),
            "arxiv_id": external_ids.get("ArXiv"),
            "doi": external_ids.get("DOI"),
            "is_peer_reviewed": bool(p.get("venue")),
            "pdf_url": open_access_pdf.get("url", ""),
            "authors": [a.get("name", "") for a in authors_raw],
        })

    logger.info(
        "Semantic Scholar returned %s results for %r",
        len(papers), query,
    )
    return papers
