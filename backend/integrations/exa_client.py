from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

EXA_SEARCH_URL = "https://api.exa.ai/search"

# Authoritative academic domains — applied for all research queries
RESEARCH_DOMAINS = [
    "arxiv.org",
    "nature.com",
    "pubmed.ncbi.nlm.nih.gov",
    "researchgate.net",
    "openreview.net",
    "aclanthology.org",
    "proceedings.mlr.press",
    "papers.nips.cc",
    "jmlr.org",
    "semanticscholar.org",
    "doi.org",
    "science.org",
    "biorxiv.org",
    "medrxiv.org",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "springer.com",
    "wiley.com",
    "journals.plos.org",
    "frontiersin.org",
    "huggingface.co",
]

# Every intent that benefits from the "research paper" Exa category + domain filter
RESEARCH_INTENTS = {
    "technical_explanation",
    "benchmark_comparison",
    "trend_analysis",
    "methodology",
    "literature_review",
    "factual_lookup",
    "general",
}


@dataclass
class ExaPaperResult:
    result_id: str = ""
    title: str = ""
    source_url: str = ""
    pdf_url: str = ""
    snippet: str = ""
    content: str = ""
    highlights: list[str] = field(default_factory=list)
    score: float = 0.0
    published_date: str = ""
    doi: str = ""
    arxiv_id: str = ""
    authors: list[dict] = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    citation_count: int = 0
    is_peer_reviewed: bool = False
    is_retracted: bool = False

    def to_metadata_override(self) -> dict:
        # Prefer highlights-joined snippet as abstract — more targeted than raw content head
        if self.highlights:
            abstract = " ... ".join(self.highlights[:3])
        else:
            abstract = self.snippet or self.content[:1200]
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "abstract": abstract,
            "citation_count": self.citation_count,
            "is_peer_reviewed": self.is_peer_reviewed,
            "is_retracted": self.is_retracted,
            "source_url": self.source_url,
            "pdf_url": self.pdf_url,
        }


async def search_papers(
    query: str,
    num_results: int = 8,
    intent: str = "general",
    entity_profile=None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> list[ExaPaperResult]:
    """
    Search Exa for research papers — enterprise configuration.

    Features enabled:
    - Neural/auto search type for semantic retrieval
    - category="research paper" + includeDomains for all research intents
    - Highlights extraction for targeted passage retrieval
    - Year constraints pushed to Exa API for server-side filtering
    - Livecrawl fallback for freshness on key academic hosts
    - useAutoprompt for Exa query optimisation
    - Entity-aware query prepending when entity_profile is present
    """
    if not settings.EXA_API_KEY:
        logger.warning("Exa search skipped: EXA_API_KEY is not set")
        return []

    # ── Entity-aware query override ────────────────────────────────
    search_query = query
    if entity_profile and getattr(entity_profile, "primary_subject", None):
        search_query = entity_profile.primary_subject + " " + query
        logger.info("Exa entity-override: query=%r", search_query)

    # ── Category + domain filtering: applied to ALL research intents ──
    is_research = intent in RESEARCH_INTENTS or (
        entity_profile and getattr(entity_profile, "requires_entity_grounding", False)
    )

    payload: dict = {
        "query": search_query,
        "type": settings.EXA_SEARCH_TYPE,
        "numResults": max(1, min(num_results, 50)),
        "useAutoprompt": settings.EXA_USE_AUTOPROMPT,
        "contents": {
            "text": {
                "maxCharacters": settings.EXA_MAX_CHARACTERS,
            },
            "highlights": {
                "numSentences": settings.EXA_HIGHLIGHT_SENTENCES,
                "highlightsPerUrl": settings.EXA_HIGHLIGHTS_PER_URL,
                "query": search_query,
            },
            "livecrawl": "fallback",
            "livecrawlTimeout": settings.EXA_LIVECRAWL_TIMEOUT_MS,
        },
    }

    if is_research:
        payload["category"] = "research paper"
        payload["includeDomains"] = list(RESEARCH_DOMAINS)

    # ── Year-range filtering at API level ─────────────────────────
    current_year = datetime.now().year
    if year_min:
        payload["startPublishedDate"] = f"{year_min}-01-01T00:00:00.000Z"
    if year_max and year_max < current_year:
        payload["endPublishedDate"] = f"{year_max}-12-31T23:59:59.999Z"

    logger.info(
        "Exa search: query=%r type=%s category=%s domains=%s highlights=True "
        "autoprompt=%s year_min=%s year_max=%s num_results=%s",
        search_query,
        settings.EXA_SEARCH_TYPE,
        payload.get("category", "none"),
        is_research,
        settings.EXA_USE_AUTOPROMPT,
        year_min,
        year_max,
        num_results,
    )

    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.post(
                EXA_SEARCH_URL,
                json=payload,
                headers={
                    "x-api-key": settings.EXA_API_KEY,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            if response.status_code >= 400:
                logger.warning("Exa error %s: %s", response.status_code, response.text[:500])
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("Exa search failed: %s", exc)
        return []

    parsed: list[ExaPaperResult] = []
    for item in data.get("results", []) or []:
        result = _parse_result(item)
        if result:
            parsed.append(result)

    logger.info("Exa search returned %s results", len(parsed))
    for idx, r in enumerate(parsed[:5], start=1):
        logger.info(
            "Exa hit %s: %s | %s | score=%.3f highlights=%s",
            idx, r.title, r.source_url, r.score, len(r.highlights),
        )
    return parsed


def _parse_result(item: dict) -> Optional[ExaPaperResult]:
    title = (item.get("title") or "").strip()
    source_url = (item.get("url") or "").strip()
    if not title or not source_url:
        return None

    raw_content = _normalize(item.get("text") or "")
    published_date = item.get("publishedDate") or item.get("published_date") or ""

    # Extract highlights — targeted relevant passages Exa identified
    raw_highlights: list[str] = []
    highlights_data = item.get("highlights") or []
    if isinstance(highlights_data, list):
        for h in highlights_data:
            if isinstance(h, str) and h.strip():
                raw_highlights.append(h.strip())
            elif isinstance(h, dict):
                text = (h.get("text") or h.get("highlight") or "").strip()
                if text:
                    raw_highlights.append(text)

    # If content is sparse, prepend highlights so downstream ingestion gets quality passages
    enriched_content = raw_content
    if raw_highlights and len(raw_content) < 3000:
        enriched_content = " ".join(raw_highlights) + "\n\n" + raw_content

    doi = _extract_doi(f"{title}\n{source_url}\n{raw_content[:2000]}")
    arxiv_id = _extract_arxiv_id(f"{title}\n{source_url}\n{raw_content[:1000]}")
    pdf_url = _detect_pdf_url(source_url, arxiv_id)
    year = _extract_year(published_date, title, raw_content)
    venue = _extract_venue(source_url)
    is_peer_reviewed = _is_peer_reviewed_host(source_url)

    return ExaPaperResult(
        result_id=_stable_id(source_url),
        title=title,
        source_url=source_url,
        pdf_url=pdf_url,
        snippet=raw_highlights[0] if raw_highlights else raw_content[:1500],
        content=enriched_content[:40000],
        highlights=raw_highlights,
        score=float(item.get("score") or 0.0),
        published_date=published_date,
        doi=doi,
        arxiv_id=arxiv_id,
        year=year,
        venue=venue,
        is_peer_reviewed=is_peer_reviewed,
    )


# ── Parsing utilities ──────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _detect_pdf_url(source_url: str, arxiv_id: str) -> str:
    if source_url.lower().endswith(".pdf"):
        return source_url
    if "arxiv.org/abs/" in source_url and arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return ""


def _extract_doi(text: str) -> str:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, re.I)
    return match.group(0) if match else ""


def _extract_arxiv_id(text: str) -> str:
    match = re.search(r"(?:arxiv[:/ ]|abs/)(\d{4}\.\d{4,5}(?:v\d+)?)", text, re.I)
    return match.group(1) if match else ""


def _extract_year(published_date: str, title: str, content: str) -> Optional[int]:
    from datetime import datetime as _dt
    current_year = _dt.now().year

    # ISO / structured date
    iso = re.search(r"\b(20[0-3]\d)[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b", published_date or "")
    if iso:
        return int(iso.group(1))

    plain = re.search(r"\b(20[0-3]\d|19\d{2})\b", published_date or "")
    if plain:
        return int(plain.group(1))

    # arXiv URL signal
    arxiv_url = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{2})(\d{2})\.\d{4,5}", content or "")
    if arxiv_url:
        yy = int(arxiv_url.group(1))
        return 2000 + yy if yy <= current_year % 100 + 1 else 1900 + yy

    # Scan title + content
    all_found = []
    for src in (title or "", (content or "")[:2000]):
        all_found.extend(int(m) for m in re.findall(r"\b(20[0-3]\d|19\d{2})\b", src))
    valid = [y for y in all_found if 1800 <= y <= current_year + 1]
    return max(valid) if valid else None


def _extract_venue(source_url: str) -> str:
    u = source_url.lower()
    if "aclanthology.org" in u:
        return "ACL Anthology"
    if "openreview.net" in u:
        return "OpenReview"
    if "proceedings.mlr.press" in u:
        return "PMLR"
    if "papers.nips.cc" in u or "neurips.cc" in u:
        return "NeurIPS"
    if "arxiv.org" in u:
        return "arXiv"
    if "jmlr.org" in u:
        return "JMLR"
    if "nature.com" in u:
        return "Nature"
    if "science.org" in u:
        return "Science"
    if "pubmed.ncbi.nlm.nih.gov" in u:
        return "PubMed"
    if "ieeexplore.ieee.org" in u:
        return "IEEE Xplore"
    if "dl.acm.org" in u:
        return "ACM DL"
    if "springer.com" in u:
        return "Springer"
    if "wiley.com" in u:
        return "Wiley"
    if "journals.plos.org" in u:
        return "PLOS"
    if "frontiersin.org" in u:
        return "Frontiers"
    if "biorxiv.org" in u:
        return "bioRxiv"
    if "medrxiv.org" in u:
        return "medRxiv"
    if "semanticscholar.org" in u:
        return "Semantic Scholar"
    if "huggingface.co" in u:
        return "HuggingFace"
    return ""


def _is_peer_reviewed_host(source_url: str) -> bool:
    host = source_url.lower()
    # Preprints and community repos are not peer-reviewed
    non_peer = ("arxiv.org", "biorxiv.org", "medrxiv.org", "openreview.net", "huggingface.co")
    return not any(p in host for p in non_peer)


def _stable_id(source_url: str) -> str:
    return hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]
