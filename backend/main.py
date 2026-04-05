"""
main.py — NexusScholar FastAPI entrypoint.
Initializes DB, indexes, Groq client; mounts all route modules.
"""

import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings, validate_settings
from backend.indexing.metadata_store import MetadataStore
from backend.indexing.bm25_index import BM25Index
from backend.indexing.dense_index import DenseIndex
from backend.retrieval.reranker import Reranker
from backend.generation.groq_client import GroqClient
from backend.indexing.embedding_cache import EmbeddingCache
from backend.ingestion.graph_builder import build_graph_from_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nexus_scholar")

# ── Global singletons ────────────────────────────────────────────

_store = MetadataStore()
_groq = GroqClient()
_bm25 = BM25Index()
_dense = DenseIndex()
_reranker = Reranker()
_embedding_cache = EmbeddingCache()


def get_store() -> MetadataStore:
    return _store

def get_groq() -> GroqClient:
    return _groq

def get_bm25() -> BM25Index:
    return _bm25

def get_dense() -> DenseIndex:
    return _dense

def get_reranker() -> Reranker:
    return _reranker

def get_embedding_cache() -> EmbeddingCache:
    return _embedding_cache


# ── Lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NexusScholar starting up...")

    # Validate settings at startup — log warnings, crash on critical misconfigurations
    try:
        config_warnings = validate_settings(settings)
        for warning in config_warnings:
            logger.warning("Config warning: %s", warning)
    except ValueError as e:
        logger.error("Critical config error: %s", e)
        raise

    # Initialize database and embedding cache
    await _store.init_db()
    await _embedding_cache.init()
    logger.info("Database and embedding cache initialized")

    # Build indexes (with embedding cache for faster restarts)
    # Include table chunks in BM25 for richer retrieval of benchmark tables
    try:
        await _bm25.build(_store, granularity="passage", extra_granularities=["table"])
        await _dense.build(_store, granularity="passage", embedding_cache=_embedding_cache)
        logger.info(f"Indexes built: BM25={_bm25.size}, Dense={_dense.size}")
        cache_stats = await _embedding_cache.stats()
        logger.info(f"Embedding cache: {cache_stats['cached_embeddings']} cached vectors")
    except Exception as e:
        logger.warning(f"Index build deferred: {e}")

    # Build citation graph (also computes PageRank eagerly)
    try:
        await build_graph_from_db(_store)
    except Exception as e:
        logger.warning(f"Graph build deferred: {e}")

    # Audit year coverage at startup — warn about papers missing year metadata
    try:
        papers_no_year = [p for p in await _store.get_all_papers() if not p.get("year")]
        if papers_no_year:
            logger.warning(
                "%s papers in corpus are missing year metadata. "
                "Run GET /api/audit/missing-years for the full list. "
                "These papers will not benefit from recency filtering or recency boosts.",
                len(papers_no_year)
            )
    except Exception as e:
        logger.warning("Year coverage audit failed: %s", e)

    # Build ColBERT index if enabled
    if settings.COLBERT_ENABLED:
        try:
            from backend.indexing.colbert_index import build_colbert_index
            all_chunks = await _store.get_all_chunks(granularity="passage")
            success = await build_colbert_index(all_chunks)
            if success:
                logger.info("ColBERT index built successfully")
            else:
                logger.warning("ColBERT index build returned False — check ragatouille installation")
        except Exception as e:
            logger.warning("ColBERT index build skipped: %s", e)

    logger.info("NexusScholar ready")
    yield
    logger.info("NexusScholar shutting down")


# ── App ───────────────────────────────────────────────────────────

app = FastAPI(
    title="NexusScholar",
    description="Enterprise AI Research Intelligence Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount route modules
from backend.api.routes.chat import router as chat_router
from backend.api.routes.ingest import router as ingest_router
from backend.api.routes.papers import router as papers_router
from backend.api.routes.evidence import router as evidence_router

app.include_router(chat_router, prefix="/api")
app.include_router(ingest_router, prefix="/api")
app.include_router(papers_router, prefix="/api")
app.include_router(evidence_router, prefix="/api")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "nexus-scholar",
        "version": "1.0.0",
        "indexes": {
            "bm25": _bm25.size,
            "dense": _dense.size,
        },
    }


# Serve frontend build if it exists
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
