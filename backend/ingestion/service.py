from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from backend.indexing.bm25_index import BM25Index
from backend.indexing.dense_index import DenseIndex
from backend.indexing.metadata_store import MetadataStore
from backend.ingestion.chunker import chunk_paper
from backend.ingestion.graph_builder import add_paper_to_graph
from backend.ingestion.normalizer import normalize_text
from backend.ingestion.pdf_parser import ParsedPaper, parse_pdf
from backend.integrations.source_urls import build_source_url, build_pdf_url

logger = logging.getLogger(__name__)


async def ingest_pdf_file(
    pdf_path: str,
    store: MetadataStore,
    bm25: BM25Index,
    dense: DenseIndex,
    metadata_override: dict | None = None,
    skip_index_rebuild: bool = False,
) -> dict:
    """
    Ingest a single PDF file into the system.
    Set skip_index_rebuild=True when batch-ingesting multiple papers
    (caller should rebuild indexes once at the end).
    """
    parsed = parse_pdf(str(pdf_path))
    _apply_metadata_override(parsed, metadata_override or {})

    parsed.full_text = normalize_text(parsed.full_text)
    parsed.abstract = normalize_text(parsed.abstract)
    parsed.sections = {k: normalize_text(v) for k, v in parsed.sections.items()}

    paper_dict = {
        "paper_id": parsed.paper_id,
        "title": parsed.title,
        "authors": parsed.authors,
        "year": parsed.year,
        "venue": parsed.venue,
        "doi": parsed.doi,
        "arxiv_id": parsed.arxiv_id,
        "openalex_id": metadata_override.get("openalex_id") if metadata_override else None,
        "abstract": parsed.abstract,
        "sections": parsed.sections,
        "references": parsed.references,
        "is_peer_reviewed": parsed.is_peer_reviewed,
        "source_url": metadata_override.get("source_url") if metadata_override else None,
        "pdf_url": metadata_override.get("pdf_url") if metadata_override else None,
        "citation_count": metadata_override.get("citation_count", 0) if metadata_override else 0,
        "pdf_path": str(pdf_path),
    }
    # Compute source/pdf URLs using the fallback chain
    paper_dict["source_url"] = build_source_url(paper_dict)
    paper_dict["pdf_url"] = build_pdf_url(paper_dict)
    await store.insert_paper(paper_dict)

    chunks = chunk_paper(parsed)
    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "paper_id": c.paper_id,
            "granularity": c.granularity,
            "section_tag": c.section_tag,
            "text": c.text,
            "token_count": c.token_count,
            "start_char": c.start_char,
            "end_char": c.end_char,
        }
        for c in chunks
    ]
    await store.insert_chunks(chunk_dicts)
    await add_paper_to_graph(parsed.paper_id, parsed.references, store)

    if not skip_index_rebuild:
        await bm25.build(store, granularity="passage")
        await dense.build(store, granularity="passage")

    logger.info("Ingested %s (%s chunks, rebuild=%s)", parsed.title, len(chunks), not skip_index_rebuild)
    return {
        "paper_id": parsed.paper_id,
        "title": parsed.title,
        "chunks_created": len(chunks),
        "arxiv_id": parsed.arxiv_id,
        "openalex_id": metadata_override.get("openalex_id") if metadata_override else None,
        "pdf_path": str(pdf_path),
    }


async def rebuild_indexes(bm25: BM25Index, dense: DenseIndex, store: MetadataStore):
    """Rebuild all retrieval indexes. Call after batch ingestion."""
    await bm25.build(store, granularity="passage")
    await dense.build(store, granularity="passage")
    logger.info("Indexes rebuilt: BM25=%s, Dense=%s", bm25.size, dense.size)


async def ingest_text_document(
    title: str,
    content: str,
    store: MetadataStore,
    bm25: BM25Index,
    dense: DenseIndex,
    metadata_override: dict | None = None,
    skip_index_rebuild: bool = False,
) -> dict:
    """
    Ingest a web-retrieved text document into the same paper/chunk pipeline.
    """
    metadata_override = metadata_override or {}
    normalized_content = normalize_text(content or "")
    if not normalized_content.strip():
        raise ValueError("Cannot ingest empty text document")

    abstract = normalize_text(metadata_override.get("abstract") or normalized_content[:2000])
    sections = {
        "abstract": abstract,
        "web_content": normalized_content,
    }
    paper = ParsedPaper(
        paper_id=_generate_text_paper_id(title, metadata_override.get("source_url", "")),
        title=title or "Untitled Web Paper",
        authors=metadata_override.get("authors", []),
        year=metadata_override.get("year"),
        venue=metadata_override.get("venue"),
        doi=metadata_override.get("doi"),
        arxiv_id=metadata_override.get("arxiv_id"),
        abstract=abstract,
        sections=sections,
        references=metadata_override.get("references", []),
        full_text=normalized_content,
        pdf_path="",
        is_peer_reviewed=bool(metadata_override.get("is_peer_reviewed", False)),
        source_url=metadata_override.get("source_url", ""),
        pdf_url=metadata_override.get("pdf_url", ""),
    )

    paper_dict = {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "venue": paper.venue,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "openalex_id": None,
        "abstract": paper.abstract,
        "sections": paper.sections,
        "references": paper.references,
        "is_peer_reviewed": paper.is_peer_reviewed,
        "source_url": paper.source_url,
        "pdf_url": paper.pdf_url,
        "citation_count": metadata_override.get("citation_count", 0),
        "pdf_path": None,
    }
    paper_dict["source_url"] = build_source_url(paper_dict)
    paper_dict["pdf_url"] = build_pdf_url(paper_dict)
    await store.insert_paper(paper_dict)

    chunks = chunk_paper(paper)
    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "paper_id": c.paper_id,
            "granularity": c.granularity,
            "section_tag": c.section_tag,
            "text": c.text,
            "token_count": c.token_count,
            "start_char": c.start_char,
            "end_char": c.end_char,
        }
        for c in chunks
    ]
    await store.insert_chunks(chunk_dicts)
    await add_paper_to_graph(paper.paper_id, paper.references, store)

    if not skip_index_rebuild:
        await bm25.build(store, granularity="passage")
        await dense.build(store, granularity="passage")

    logger.info("Ingested web document %s (%s chunks, rebuild=%s)", paper.title, len(chunks), not skip_index_rebuild)
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "chunks_created": len(chunks),
        "pdf_path": None,
    }


def _apply_metadata_override(parsed: ParsedPaper, override: dict) -> None:
    if not override:
        return

    for field in ("title", "year", "venue", "doi", "arxiv_id", "abstract", "is_peer_reviewed", "source_url", "pdf_url"):
        value = override.get(field)
        if value not in (None, ""):
            setattr(parsed, field, value)

    authors = override.get("authors")
    if authors:
        parsed.authors = authors

    references = override.get("references")
    if references:
        parsed.references = references


def _generate_text_paper_id(title: str, source_url: str) -> str:
    raw = f"{title}|{source_url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
