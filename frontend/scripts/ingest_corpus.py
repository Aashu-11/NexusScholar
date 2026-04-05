#!/usr/bin/env python3
"""
ingest_corpus.py — Batch ingestion CLI.
Ingest all PDFs from a directory into the NexusScholar corpus.

Usage:
    python scripts/ingest_corpus.py ./papers/
    python scripts/ingest_corpus.py ./papers/ --rebuild-indexes
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.indexing.metadata_store import MetadataStore
from backend.indexing.bm25_index import BM25Index
from backend.indexing.dense_index import DenseIndex
from backend.ingestion.pdf_parser import parse_pdf
from backend.ingestion.normalizer import normalize_text
from backend.ingestion.chunker import chunk_paper
from backend.ingestion.graph_builder import add_paper_to_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_corpus")


async def ingest_directory(pdf_dir: str, rebuild: bool = False):
    store = MetadataStore()
    await store.init_db()

    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        logger.error(f"Directory not found: {pdf_dir}")
        return

    pdf_files = list(pdf_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")

    success = 0
    failed = 0

    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"[{i}/{len(pdf_files)}] Ingesting: {pdf_file.name}")
        try:
            parsed = parse_pdf(str(pdf_file))
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
                "abstract": parsed.abstract,
                "sections": parsed.sections,
                "references": parsed.references,
                "is_peer_reviewed": parsed.is_peer_reviewed,
                "pdf_path": str(pdf_file),
            }
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

            logger.info(f"  → {parsed.title[:60]} ({len(chunks)} chunks)")
            success += 1

        except Exception as e:
            logger.error(f"  → Failed: {e}")
            failed += 1

    logger.info(f"\nIngestion complete: {success} succeeded, {failed} failed")

    if rebuild or True:
        logger.info("Rebuilding indexes...")
        bm25 = BM25Index()
        dense = DenseIndex()
        await bm25.build(store, granularity="passage")
        await dense.build(store, granularity="passage")
        logger.info(f"Indexes rebuilt: BM25={bm25.size}, Dense={dense.size}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_corpus.py <pdf_directory> [--rebuild-indexes]")
        sys.exit(1)

    pdf_dir = sys.argv[1]
    rebuild = "--rebuild-indexes" in sys.argv

    asyncio.run(ingest_directory(pdf_dir, rebuild))