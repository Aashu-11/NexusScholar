"""
pdf_parser.py — PDF → structured Research Object.
Uses PyMuPDF as primary parser. Grobid TEI XML support when available.
"""

from __future__ import annotations
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ParsedPaper:
    """Raw parse output before normalization."""
    paper_id: str = ""
    title: str = ""
    authors: list[dict] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    abstract: str = ""
    sections: dict[str, str] = field(default_factory=dict)  # heading → text
    references: list[dict] = field(default_factory=list)
    full_text: str = ""
    pdf_path: str = ""
    is_peer_reviewed: bool = False
    source_url: str = ""
    pdf_url: str = ""


def parse_pdf(pdf_path: str) -> ParsedPaper:
    """
    Extract text, metadata, and structure from a PDF.
    Tries in order: Grobid TEI → Marker → PyMuPDF (three-tier fallback).
    """
    grobid_result = _try_grobid(pdf_path)
    if grobid_result:
        return grobid_result

    marker_result = _try_marker(pdf_path)
    if marker_result:
        return marker_result

    return _parse_with_pymupdf(pdf_path)


def _try_marker(pdf_path: str) -> Optional[ParsedPaper]:
    """Attempt structured extraction with marker-pdf. Returns None if unavailable."""
    try:
        from marker.convert import convert_single_pdf
        from marker.models import load_all_models
        models = load_all_models()
        # marker returns (markdown_text, metadata, images)
        result = convert_single_pdf(pdf_path, models)
        if isinstance(result, tuple):
            markdown_text = result[0] if result else ""
        else:
            markdown_text = str(result)

        if not markdown_text or len(markdown_text) < 100:
            return None

        paper = ParsedPaper(pdf_path=pdf_path)
        paper.full_text = markdown_text
        paper.sections = _parse_markdown_to_sections(markdown_text)
        paper.abstract = paper.sections.pop("abstract", "")
        if not paper.abstract:
            paper.abstract = _extract_abstract_heuristic(markdown_text)

        # Extract metadata from markdown header
        lines = markdown_text.strip().split("\n")
        for line in lines[:5]:
            line = line.lstrip("#").strip()
            if len(line) > 10:
                paper.title = line
                break

        paper.year = _detect_year(markdown_text, {})
        paper.references = _extract_references(markdown_text)
        paper.paper_id = _generate_paper_id(paper.title, paper.year)
        doi_match = re.search(r"10\.\d{4,}/\S+", markdown_text[:3000])
        if doi_match:
            paper.doi = doi_match.group().rstrip(".,;)")
        arxiv_match = re.search(r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)", markdown_text[:3000])
        if arxiv_match:
            paper.arxiv_id = arxiv_match.group(1)

        logger.info("Marker extraction succeeded for %s: %s chars", pdf_path, len(markdown_text))
        return paper
    except ImportError:
        return None
    except Exception as e:
        logger.warning("Marker extraction failed for %s: %s", pdf_path, e)
        return None


def _parse_markdown_to_sections(markdown_text: str) -> dict[str, str]:
    """
    Parse Marker's markdown output (which uses # headers) into sections dict.
    Maps markdown headers to standard section tags.
    """
    sections: dict[str, str] = {}
    current_tag = "unknown"
    current_lines: list[str] = []

    for line in markdown_text.split("\n"):
        # Detect markdown heading
        header_match = re.match(r"^#{1,4}\s+(.+)$", line.strip())
        if header_match:
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections[current_tag] = sections.get(current_tag, "") + "\n" + content
            heading_text = header_match.group(1).strip()
            current_tag = _map_heading_to_section_tag(heading_text)
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections[current_tag] = sections.get(current_tag, "") + "\n" + content

    return {k: v.strip() for k, v in sections.items() if v.strip()}


def _map_heading_to_section_tag(heading: str) -> str:
    """Map a heading string to one of the standard section tags."""
    lower = heading.lower()
    if re.match(r"^abstract", lower):
        return "abstract"
    if re.match(r"^(1[\.\s]?\s*)?introduction", lower):
        return "introduction"
    if re.match(r"^(\d[\.\s]?\s*)?(background|related\s+work|literature|prior\s+work)", lower):
        return "background"
    if re.match(r"^(\d[\.\s]?\s*)?(method|approach|methodology|framework|model|proposed)", lower):
        return "methods"
    if re.match(r"^(\d[\.\s]?\s*)?(results?|experiments?|evaluation|empirical)", lower):
        return "results"
    if re.match(r"^(\d[\.\s]?\s*)?(discussion|analysis|limitations)", lower):
        return "discussion"
    if re.match(r"^(\d[\.\s]?\s*)?(conclusion|summary|future\s+work)", lower):
        return "conclusion"
    if re.match(r"^(appendix|supplementary)", lower):
        return "appendix"
    return "unknown"


def _try_grobid(pdf_path: str) -> Optional[ParsedPaper]:
    """Attempt Grobid TEI XML extraction. Returns None if Grobid unavailable."""
    parsed_dir = Path(settings.PARSED_PATH)
    tei_path = parsed_dir / (Path(pdf_path).stem + ".tei.xml")

    if not tei_path.exists():
        # Could attempt HTTP call to running Grobid service here
        # For now, fall back to PyMuPDF
        return None

    try:
        return _parse_tei_xml(str(tei_path), pdf_path)
    except Exception as e:
        logger.warning(f"Grobid TEI parse failed for {pdf_path}: {e}")
        return None


def _parse_tei_xml(tei_path: str, pdf_path: str) -> ParsedPaper:
    """Parse a Grobid TEI XML file into a ParsedPaper."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(tei_path)
    root = tree.getroot()
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    paper = ParsedPaper(pdf_path=pdf_path)

    # Title
    title_el = root.find(".//tei:titleStmt/tei:title", ns)
    paper.title = (title_el.text or "").strip() if title_el is not None else ""

    # Authors
    for author_el in root.findall(".//tei:sourceDesc//tei:author", ns):
        forename = author_el.findtext(".//tei:forename", "", ns).strip()
        surname = author_el.findtext(".//tei:surname", "", ns).strip()
        affiliation = author_el.findtext(".//tei:affiliation/tei:orgName", "", ns).strip()
        if forename or surname:
            paper.authors.append({
                "name": f"{forename} {surname}".strip(),
                "affiliation": affiliation or None,
            })

    # Abstract
    abstract_el = root.find(".//tei:profileDesc/tei:abstract", ns)
    if abstract_el is not None:
        paper.abstract = " ".join(abstract_el.itertext()).strip()

    # Body sections
    for div in root.findall(".//tei:body/tei:div", ns):
        head = div.findtext("tei:head", "", ns).strip()
        body_text = " ".join(div.itertext()).strip()
        if head:
            paper.sections[head] = body_text

    # References
    for ref in root.findall(".//tei:listBibl/tei:biblStruct", ns):
        ref_title = ref.findtext(".//tei:title", "", ns).strip()
        ref_year = ref.findtext(".//tei:date/@when", "", ns)[:4] if ref.find(".//tei:date") is not None else ""
        paper.references.append({
            "ref_id": str(len(paper.references) + 1),
            "title": ref_title,
            "year": int(ref_year) if ref_year.isdigit() else None,
        })

    paper.full_text = paper.abstract + "\n\n" + "\n\n".join(paper.sections.values())
    paper.paper_id = _generate_paper_id(paper.title, paper.year)

    return paper


def _parse_with_pymupdf(pdf_path: str) -> ParsedPaper:
    """Fallback: extract with PyMuPDF layout hints. Handles two-column academic layouts."""
    doc = fitz.open(pdf_path)
    metadata = doc.metadata or {}

    full_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        # FIX: Use "blocks" mode to get position-aware text blocks, then sort by position.
        # This correctly orders two-column text: left column top-to-bottom, then right column.
        # The old page.get_text("text") loses column ordering in two-column layouts.
        blocks = page.get_text("blocks")  # returns (x0,y0,x1,y1,text,block_no,block_type)
        # Sort by horizontal band (y0 // 50), then by x0 within the band
        sorted_blocks = sorted(blocks, key=lambda b: (b[1] // 50, b[0]))
        page_text = "\n".join(b[4] for b in sorted_blocks if b[6] == 0)  # block_type 0 = text
        full_text += page_text + "\n"
    doc.close()

    paper = ParsedPaper(pdf_path=pdf_path)

    # Title
    paper.title = metadata.get("title", "").strip()
    if not paper.title:
        paper.title = _extract_title_heuristic(full_text)

    # Authors
    authors_raw = metadata.get("author", "")
    if authors_raw:
        parts = re.split(r"[,;]|\band\b", authors_raw)
        paper.authors = [{"name": p.strip(), "affiliation": None}
                         for p in parts if p.strip()]

    # Year
    paper.year = _detect_year(full_text, metadata)

    # Sections
    paper.sections = _split_into_sections(full_text)

    # Abstract
    paper.abstract = paper.sections.pop("abstract", "")
    if not paper.abstract:
        paper.abstract = _extract_abstract_heuristic(full_text)

    # References
    paper.references = _extract_references(full_text)

    paper.full_text = full_text
    paper.paper_id = _generate_paper_id(paper.title, paper.year)

    # Extract DOI if present
    doi_match = re.search(r"10\.\d{4,}/\S+", full_text[:3000])
    if doi_match:
        paper.doi = doi_match.group().rstrip(".,;)")

    # Extract arXiv ID if present
    arxiv_match = re.search(r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)", full_text[:3000])
    if arxiv_match:
        paper.arxiv_id = arxiv_match.group(1)

    # Peer-review detection: DOI alone is NOT sufficient (arXiv papers can have DOIs).
    # Use heuristics: look for journal/conference indicators in the text.
    # Final peer-review status is best determined by OpenAlex enrichment later.
    if paper.doi and not paper.arxiv_id:
        # Has DOI but no arXiv ID — likely published in a venue
        paper.is_peer_reviewed = True
    elif _has_venue_indicators(full_text[:5000]):
        paper.is_peer_reviewed = True
    else:
        paper.is_peer_reviewed = False

    return paper


# ── Heuristics ────────────────────────────────────────────────────

SECTION_PATTERNS = {
    "abstract": r"(?i)^abstract",
    "introduction": r"(?i)^(1[\.\s]?\s*)?introduction",
    "background": r"(?i)^(\d[\.\s]?\s*)?(background|related\s+work|literature\s+review|prior\s+work)",
    "methods": r"(?i)^(\d[\.\s]?\s*)?(method|approach|methodology|framework|model\s+architecture|proposed)",
    "results": r"(?i)^(\d[\.\s]?\s*)?(results?|experiments?|evaluation|empirical)",
    "discussion": r"(?i)^(\d[\.\s]?\s*)?(discussion|analysis|limitations)",
    "conclusion": r"(?i)^(\d[\.\s]?\s*)?(conclusion|summary|future\s+work)",
    "appendix": r"(?i)^(appendix|supplementary)",
}


def _split_into_sections(text: str) -> dict[str, str]:
    """Split full text into sections by heading detection."""
    sections: dict[str, str] = {}
    current_tag = "unknown"
    current_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            current_lines.append("")
            continue

        detected = None
        if len(stripped) < 100:
            for tag, pattern in SECTION_PATTERNS.items():
                if re.match(pattern, stripped):
                    detected = tag
                    break

        if detected:
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections[current_tag] = sections.get(current_tag, "") + "\n" + content
            current_tag = detected
            current_lines = []
        else:
            current_lines.append(stripped)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections[current_tag] = sections.get(current_tag, "") + "\n" + content

    return {k: v.strip() for k, v in sections.items() if v.strip()}


def _extract_title_heuristic(text: str) -> str:
    lines = text.strip().split("\n")
    for line in lines[:10]:
        line = line.strip()
        if len(line) > 10 and not re.match(r"^(arxiv|preprint|journal|vol\.|page)", line, re.I):
            return line
    return "Untitled Paper"


def _detect_year(text: str, metadata: dict) -> Optional[int]:
    """
    Detect publication year from PDF. Prioritizes content signals over metadata.
    PDF creationDate is often the compilation date, not the publication year.
    """
    from collections import Counter
    import datetime
    current_year = datetime.datetime.now().year

    # Priority 1: arXiv ID embedded in the text (most reliable for preprints)
    # arXiv IDs since 2007 start with YYMM — e.g. 2401 = January 2024
    arxiv_match = re.search(r"arXiv:(\d{2})(\d{2})\.\d{4,5}", text[:3000])
    if arxiv_match:
        yy = int(arxiv_match.group(1))
        return 2000 + yy if yy <= current_year % 100 + 1 else 1900 + yy

    # Priority 2: Conference/journal year in first 3000 chars
    conf_match = re.search(
        r"(?:NeurIPS|NIPS|ICML|ICLR|ACL|EMNLP|NAACL|CVPR|ICCV|ECCV|AAAI|IJCAI|"
        r"JMLR|TACL|Nature|Science|Cell|Lancet|IEEE|ACM)\s+(20[0-2]\d|19\d{2})",
        text[:3000]
    )
    if conf_match:
        return int(conf_match.group(1))

    # Priority 3: "© 2024" or "Copyright 2024" near top of paper
    copyright_match = re.search(
        r"©\s*(20[0-2]\d|19\d{2})|[Cc]opyright\s+(20[0-2]\d|19\d{2})", text[:5000]
    )
    if copyright_match:
        year_str = copyright_match.group(1) or copyright_match.group(2)
        if year_str:
            return int(year_str)

    # Priority 4: "Submitted / Accepted / Published: MONTH YEAR"
    submitted_match = re.search(
        r"(?:submitted|accepted|published|received)[:\s]+\w+\s+(20[0-2]\d|19\d{2})",
        text[:5000], re.IGNORECASE
    )
    if submitted_match:
        return int(submitted_match.group(1))

    # Priority 5: PDF creationDate — only use if it looks like a publication year
    # (i.e. not obviously a recent conversion date for an old paper)
    for date_field in ("creationDate", "modDate"):
        date_str = (metadata.get(date_field) or "").strip()
        date_match = re.search(r"(20[0-2]\d|19\d{2})", date_str)
        if date_match:
            candidate = int(date_match.group(1))
            # Sanity check: don't trust if the paper text contains an earlier year
            # that looks more like a conference year
            text_years = [int(m) for m in re.findall(r"\b(20[0-2]\d|19\d{2})\b", text[:5000])]
            plausible = [y for y in text_years if 1990 <= y <= current_year]
            if plausible and min(plausible) < candidate - 1:
                # PDF metadata year is newer than what appears in content — likely wrong
                logger.debug(
                    "Ignoring PDF creationDate=%s; content suggests earlier year=%s",
                    candidate, min(plausible)
                )
            else:
                return candidate

    # Priority 6: Most common year in first 5000 chars of text
    text_years = [int(m) for m in re.findall(r"\b(20[0-2]\d|19\d{2})\b", text[:5000])]
    valid = [y for y in text_years if 1990 <= y <= current_year]
    if valid:
        # Most frequent year is usually the paper's own year (appears in citations as most recent)
        year_counts = Counter(valid)
        return year_counts.most_common(1)[0][0]

    return None


def _extract_abstract_heuristic(text: str) -> str:
    match = re.search(
        r"(?i)abstract[:\s]*\n?(.*?)(?=\n\s*\n|\n\d+[\.\s]*introduction)",
        text[:5000], re.DOTALL,
    )
    if match:
        return match.group(1).strip()[:2000]
    for p in text.split("\n\n")[:5]:
        p = p.strip()
        if len(p) > 100:
            return p[:2000]
    return ""


def _extract_references(text: str) -> list[dict]:
    refs = []
    match = re.search(r"(?i)(references|bibliography)\s*\n(.*)", text, re.DOTALL)
    if not match:
        return refs
    ref_section = match.group(2)
    for ref_id, ref_text in re.findall(r"\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)", ref_section, re.DOTALL):
        ref_text = ref_text.strip().replace("\n", " ")
        refs.append({"ref_id": ref_id, "title": ref_text[:200]})
        if len(refs) >= 100:
            break
    return refs


def _has_venue_indicators(text: str) -> bool:
    """Check if text contains indicators of journal/conference publication."""
    venue_patterns = [
        r"(?i)\b(proceedings|conference|journal|transactions|letters|review)\b",
        r"(?i)\b(IEEE|ACM|Springer|Elsevier|Nature|Science|PNAS|AAAI|NeurIPS|ICML|ICLR|CVPR|EMNLP|ACL)\b",
        r"(?i)\bvol\.\s*\d+",
        r"(?i)\bpp\.\s*\d+",
        r"(?i)\bISSN\b",
        r"(?i)\baccepted\s+(for|to|at|by)\b",
        r"(?i)\bpublished\s+(in|by)\b",
    ]
    matches = sum(1 for p in venue_patterns if re.search(p, text))
    return matches >= 2


def _generate_paper_id(title: str, year: Optional[int]) -> str:
    raw = f"{title}|{year or 'unknown'}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]