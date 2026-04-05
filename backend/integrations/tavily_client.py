from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
SCHOLARLY_DOMAINS = [
    "arxiv.org",
    "openreview.net",
    "aclanthology.org",
    "proceedings.mlr.press",
    "papers.nips.cc",
    "jmlr.org",
    "semanticscholar.org",
    "doi.org",
    "nature.com",
    "science.org",
    "pubmed.ncbi.nlm.nih.gov",
    "biorxiv.org",
    "medrxiv.org",
]


@dataclass
class TavilyPaperResult:
    result_id: str = ""
    title: str = ""
    source_url: str = ""
    pdf_url: str = ""
    snippet: str = ""
    content: str = ""
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
    max_results: int = 10,
    topic: str = "general",
) -> list[TavilyPaperResult]:
    if not settings.TAVILY_API_KEY:
        logger.warning("Tavily search skipped: TAVILY_API_KEY is not set")
        return []

    payload = {
        "query": query,
        "search_depth": "advanced",
        "topic": topic,
        "max_results": max(1, min(max_results, 20)),
        "include_raw_content": True,
        "include_answer": False,
        "include_images": False,
        "include_domains": settings.TAVILY_INCLUDE_DOMAINS or SCHOLARLY_DOMAINS,
    }
    logger.info("Tavily search: query=%r max_results=%s", query, max_results)

    try:
        async with httpx.AsyncClient(
            timeout=settings.TAVILY_HTTP_TIMEOUT,
            follow_redirects=True,
        ) as client:
            response = await client.post(
                TAVILY_SEARCH_URL,
                json=payload,
                headers={
                    "User-Agent": settings.TAVILY_USER_AGENT,
                    "Authorization": f"Bearer {settings.TAVILY_API_KEY}",
                },
            )
            if response.status_code >= 400:
                logger.warning("Tavily error %s: %s", response.status_code, response.text)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return []

    parsed: list[TavilyPaperResult] = []
    for item in data.get("results", []) or []:
        result = _parse_result(item)
        if result:
            parsed.append(result)

    logger.info("Tavily search returned %s results", len(parsed))
    for idx, result in enumerate(parsed[:5], start=1):
        logger.info("Tavily hit %s: %s | %s | score=%.3f", idx, result.title, result.source_url, result.score)
    return parsed


def _parse_result(item: dict) -> Optional[TavilyPaperResult]:
    title = (item.get("title") or "").strip()
    source_url = (item.get("url") or "").strip()
    raw_content = _normalize_content(item.get("raw_content") or item.get("content") or "")
    snippet = _normalize_content(item.get("content") or "")
    if not title or not source_url:
        return None

    doi = _extract_doi(f"{title}\n{source_url}\n{raw_content[:2000]}")
    arxiv_id = _extract_arxiv_id(f"{title}\n{source_url}\n{raw_content[:1000]}")
    pdf_url = _detect_pdf_url(item, source_url, arxiv_id)
    year = _extract_year(item.get("published_date"), title, raw_content)
    # FIX: fallback to content signal inference when _extract_year returns None
    if year is None:
        year = _infer_year_from_content_signals(raw_content, source_url)
    venue = _extract_venue(source_url)
    is_peer_reviewed = _is_peer_reviewed_host(source_url)

    return TavilyPaperResult(
        result_id=_stable_result_id(source_url),
        title=title,
        source_url=source_url,
        pdf_url=pdf_url,
        snippet=snippet[:1500],
        content=raw_content[:40000],
        score=float(item.get("score") or 0.0),
        published_date=(item.get("published_date") or ""),
        doi=doi,
        arxiv_id=arxiv_id,
        year=year,
        venue=venue,
        is_peer_reviewed=is_peer_reviewed,
    )


def _normalize_content(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned


def _detect_pdf_url(item: dict, source_url: str, arxiv_id: str) -> str:
    for key in ("pdf_url", "download_url"):
        value = (item.get(key) or "").strip()
        if value:
            return value
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
    """
    Robustly extract publication year from Tavily result metadata.
    Handles: ISO dates, relative dates, long-form dates, and content scanning.
    Priority: published_date > title > content (most recent plausible year wins).
    """
    from datetime import datetime as _dt
    current_year = _dt.now().year

    # ── Tier 1: Relative date strings Tavily returns ──────────────
    relative = (published_date or "").lower().strip()
    if relative:
        if any(tok in relative for tok in (
            "just now", "moment ago", "hour", "minute", "second", "today"
        )):
            return current_year
        if any(tok in relative for tok in (
            "yesterday", "1 day ago", "2 days ago", "3 days ago",
            "4 days ago", "5 days ago", "6 days ago", "7 days ago",
            "this week", "last week", "a week ago"
        )):
            return current_year
        if any(tok in relative for tok in (
            "this month", "last month", "a month ago",
            "2 months ago", "3 months ago", "4 months ago",
            "5 months ago", "6 months ago", "7 months ago",
            "8 months ago", "9 months ago", "10 months ago",
            "11 months ago", "months ago"
        )):
            return current_year

        # "N years ago" — e.g. "2 years ago"
        years_ago_match = re.search(r"(\d+)\s+year", relative)
        if years_ago_match:
            n = int(years_ago_match.group(1))
            return max(2000, current_year - n)

        # "last year" / "a year ago"
        if "last year" in relative or "a year ago" in relative:
            return current_year - 1

    # ── Tier 2: ISO and structured date strings ───────────────────
    # "2024-03-15", "2024/03/15", "2024.03.15"
    iso_match = re.search(r"\b(20[0-3]\d)[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b", published_date or "")
    if iso_match:
        return int(iso_match.group(1))

    # "2024-03", "2024/03" (year-month only)
    ym_match = re.search(r"\b(20[0-3]\d)[/\-]\d{1,2}\b", published_date or "")
    if ym_match:
        return int(ym_match.group(1))

    # "March 15, 2024" / "15 March 2024" / "Mar 2024" / "March 2024"
    month_names = (
        r"january|february|march|april|may|june|july|august|"
        r"september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
    )
    long_date_match = re.search(
        rf"(?:{month_names})[\s,]*(\d{{1,2}})?[\s,]*(20[0-3]\d)",
        (published_date or "").lower()
    )
    if long_date_match:
        return int(long_date_match.group(long_date_match.lastindex))

    # "15 Mar 2024" / "15 March 2024"
    day_month_year = re.search(
        rf"\d{{1,2}}\s+(?:{month_names})\s+(20[0-3]\d)",
        (published_date or "").lower()
    )
    if day_month_year:
        return int(day_month_year.group(1))

    # ── Tier 3: Plain 4-digit year anywhere in published_date ─────
    plain_year = re.search(r"\b(20[0-3]\d|19\d{2})\b", published_date or "")
    if plain_year:
        return int(plain_year.group(1))

    # ── Tier 4: Scan title and content, prefer most recent valid year ─
    all_found: list[int] = []
    for source in (title or "", (content or "")[:2000]):
        all_found.extend(int(m) for m in re.findall(r"\b(20[0-3]\d|19\d{2})\b", source))

    valid = [y for y in all_found if 1800 <= y <= current_year + 1]
    if valid:
        # For publication context: most recent year found is usually the publication year
        return max(valid)

    return None


def _infer_year_from_content_signals(content: str, source_url: str) -> Optional[int]:
    """
    Extract year from semantic publication signals in content.
    Called when _extract_year() returns None.
    """
    from datetime import datetime as _dt
    current_year = _dt.now().year

    # arXiv ID in URL: arxiv.org/abs/2401.xxxxx → year=2024, month=01
    arxiv_url_match = re.search(
        r"arxiv\.org/(?:abs|pdf)/(\d{2})(\d{2})\.\d{4,5}", source_url or ""
    )
    if arxiv_url_match:
        yy = int(arxiv_url_match.group(1))
        # arXiv uses 2-digit year: 24 → 2024
        return 2000 + yy if yy <= current_year % 100 + 1 else 1900 + yy

    text = (content or "")[:3000].lower()

    # "published in 2024", "accepted in 2024", "submitted in 2024"
    pub_match = re.search(
        r"(?:published|accepted|submitted|appeared|presented|released)\s+(?:in|at|to)?\s*(20[0-3]\d)",
        text
    )
    if pub_match:
        return int(pub_match.group(1))

    # "neurips 2024", "icml 2024", "iclr 2024", etc.
    conf_match = re.search(
        r"(?:neurips|nips|icml|iclr|acl|emnlp|naacl|cvpr|iccv|eccv|aaai|ijcai|"
        r"nature|science|cell|lancet|jmlr|tacl)\s+(20[0-3]\d)",
        text
    )
    if conf_match:
        return int(conf_match.group(1))

    # Copyright symbol: "© 2024", "(c) 2024"
    copyright_match = re.search(r"(?:©|\(c\)|copyright)\s*(20[0-3]\d)", text)
    if copyright_match:
        return int(copyright_match.group(1))

    return None


def _extract_venue(source_url: str) -> str:
    if "aclanthology.org" in source_url:
        return "ACL Anthology"
    if "openreview.net" in source_url:
        return "OpenReview"
    if "proceedings.mlr.press" in source_url:
        return "PMLR"
    if "papers.nips.cc" in source_url:
        return "NeurIPS"
    if "arxiv.org" in source_url:
        return "arXiv"
    if "jmlr.org" in source_url:
        return "JMLR"
    return ""


def _is_peer_reviewed_host(source_url: str) -> bool:
    host = source_url.lower()
    return not any(preprint_host in host for preprint_host in ("arxiv.org", "biorxiv.org", "medrxiv.org", "openreview.net"))


def _stable_result_id(source_url: str) -> str:
    import hashlib

    return hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]
