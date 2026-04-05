#!/usr/bin/env python3
"""
build_indexes.py — Rebuild all retrieval indexes.

Usage:
    python scripts/build_indexes.py
    python scripts/build_indexes.py --save-faiss
"""

import asyncio
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.indexing.metadata_store import MetadataStore
from backend.indexing.bm25_index import BM25Index
from backend.indexing.dense_index import DenseIndex
from backend.ingestion.graph_builder import build_graph_from_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_indexes")


async def main(save_faiss: bool = False):
    store = MetadataStore()
    await store.init_db()

    # BM25
    logger.info("Building BM25 index...")
    bm25 = BM25Index()
    await bm25.build(store, granularity="passage")
    logger.info(f"BM25 index: {bm25.size} chunks")

    # Dense / FAISS
    logger.info("Building dense (FAISS) index...")
    dense = DenseIndex()
    await dense.build(store, granularity="passage")
    logger.info(f"Dense index: {dense.size} vectors")

    if save_faiss:
        faiss_path = str(Path(settings.INDEX_PATH) / "dense.faiss")
        dense.save(faiss_path)

    # Citation graph
    logger.info("Building citation graph...")
    await build_graph_from_db(store)

    logger.info("All indexes rebuilt successfully.")


if __name__ == "__main__":
    save = "--save-faiss" in sys.argv
    asyncio.run(main(save_faiss=save))