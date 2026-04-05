"""
dense_index.py — FAISS-backed dense vector index with BGE-large embeddings.
Falls back to in-memory cosine similarity if FAISS unavailable.
"""

from __future__ import annotations
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Optional

from backend.config import settings
from backend.indexing.metadata_store import MetadataStore

logger = logging.getLogger(__name__)

# ── Embedding Model ───────────────────────────────────────────────

_embedder = None
_embedder_lock = threading.Lock()  # Guards lazy initialization against concurrent requests


def _get_device() -> str:
    """Pick the best available compute device for embeddings."""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            logger.info("CUDA GPU detected: %s (%d GB VRAM) — using GPU for embeddings", name, vram)
            return device
    except ImportError:
        pass
    logger.info("No CUDA GPU detected — using CPU for embeddings")
    return "cpu"


def get_embedder():
    global _embedder
    # Fast path: model already loaded (no lock needed for read after init)
    if _embedder is not None:
        return _embedder
    # Slow path: acquire lock so only one thread initializes the model
    with _embedder_lock:
        if _embedder is not None:  # re-check after acquiring lock
            return _embedder
        try:
            from sentence_transformers import SentenceTransformer
            device = _get_device()
            _embedder = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)
            logger.info("Loaded embedding model: %s on %s", settings.EMBEDDING_MODEL, device)
            return _embedder
        except Exception as e:
            logger.warning("Could not load embedding model: %s", e)
            return None


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode passage texts (no query prefix)."""
    model = get_embedder()
    if model:
        return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return _tfidf_fallback(texts)


def embed_query(query: str, entity_hint: Optional[str] = None) -> np.ndarray:
    """Encode a query with the BGE retrieval prefix for asymmetric search.

    Args:
        query: The query text to encode.
        entity_hint: Optional entity type string to prepend for better retrieval
                     specificity (e.g. "RBMK reactor"). Improves BGE model precision.
    """
    model = get_embedder()
    if model:
        if entity_hint:
            prefixed = (
                f"Represent this sentence for searching relevant passages about "
                f"{entity_hint}: {query}"
            )
        else:
            prefixed = settings.EMBEDDING_QUERY_PREFIX + query
        return model.encode([prefixed], show_progress_bar=False, normalize_embeddings=True)[0]
    return _tfidf_fallback([query])[0]


def _tfidf_fallback(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    dim = 1024  # match BGE-large output dimension
    v = TfidfVectorizer(max_features=dim, stop_words="english")
    try:
        return v.fit_transform(texts).toarray()
    except ValueError:
        return np.zeros((len(texts), dim))


# ── FAISS Index ───────────────────────────────────────────────────

class DenseIndex:
    def __init__(self):
        self.chunks: list[dict] = []
        self.vectors: Optional[np.ndarray] = None
        self._faiss_index = None
        self._use_faiss = False

    async def build(self, store: MetadataStore, granularity: str = "passage", embedding_cache=None):
        """Load chunks and build FAISS (or fallback numpy) index. Uses embedding cache when available."""
        self.chunks = await store.get_all_chunks(granularity=granularity)

        if not self.chunks:
            logger.warning(f"No {granularity} chunks found for dense index")
            return

        # Check for precomputed embeddings and cache
        texts_to_embed = []
        all_vecs = [None] * len(self.chunks)
        cache_hits = 0

        for i, c in enumerate(self.chunks):
            if c.get("embedding"):
                all_vecs[i] = np.array(c["embedding"], dtype=np.float32)
                continue
            texts_to_embed.append((i, c["text"]))

        # Try embedding cache for uncached texts
        if texts_to_embed and embedding_cache:
            cache_texts = [text for _, text in texts_to_embed]
            cached = await embedding_cache.get_batch(cache_texts)
            still_need = []
            for idx, text in texts_to_embed:
                cached_emb = cached.get(text)
                if cached_emb is not None:
                    all_vecs[idx] = cached_emb
                    cache_hits += 1
                else:
                    still_need.append((idx, text))
            texts_to_embed = still_need
            logger.info(f"Embedding cache: {cache_hits} hits, {len(texts_to_embed)} misses")

        # Compute remaining embeddings
        if texts_to_embed:
            logger.info(f"Computing embeddings for {len(texts_to_embed)} chunks...")
            raw_texts = [text for _, text in texts_to_embed]
            new_embeddings = embed_texts(raw_texts)
            cache_items = []
            for (idx, text), emb in zip(texts_to_embed, new_embeddings):
                vec = emb.astype(np.float32)
                all_vecs[idx] = vec
                cache_items.append((text, vec))
            # Persist to cache
            if embedding_cache and cache_items:
                await embedding_cache.put_batch(cache_items)

        # Stack all vectors
        valid_vecs = [v for v in all_vecs if v is not None]
        if not valid_vecs:
            self.vectors = embed_texts([c["text"] for c in self.chunks]).astype(np.float32)
        else:
            self.vectors = np.stack(valid_vecs)

        # Try FAISS
        try:
            import faiss
            dim = self.vectors.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)  # inner product (cosine on normalized vecs)
            faiss.normalize_L2(self.vectors)
            self._faiss_index.add(self.vectors)
            self._use_faiss = True
            logger.info(f"FAISS index built: {len(self.chunks)} vectors, dim={dim}")
        except ImportError:
            self._use_faiss = False
            logger.info(f"FAISS not available, using numpy cosine fallback ({len(self.chunks)} vectors)")

    def search(
        self,
        query: str,
        top_k: int = 100,
        entity_hint: Optional[str] = None,
    ) -> list[tuple[dict, float, int]]:
        """
        Search dense index. Returns (chunk_dict, score, rank).

        Args:
            entity_hint: Optional entity type hint to improve retrieval specificity.
        """
        if not self.chunks or self.vectors is None:
            return []

        q_vec = embed_query(query, entity_hint=entity_hint).astype(np.float32).reshape(1, -1)

        if self._use_faiss and self._faiss_index:
            import faiss
            faiss.normalize_L2(q_vec)
            scores, indices = self._faiss_index.search(q_vec, min(top_k, len(self.chunks)))
            results = []
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
                if idx >= 0:
                    results.append((self.chunks[idx], float(score), rank))
            return results
        else:
            return self._numpy_search(q_vec[0], top_k)

    def _numpy_search(self, q_vec: np.ndarray, top_k: int) -> list[tuple[dict, float, int]]:
        """Fallback cosine similarity search."""
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = self.vectors / norms
        scores = normalized @ q_norm
        ranked = np.argsort(-scores)[:top_k]
        return [
            (self.chunks[i], float(scores[i]), rank)
            for rank, i in enumerate(ranked)
        ]

    def save(self, path: str):
        """Persist FAISS index to disk."""
        if self._use_faiss and self._faiss_index:
            import faiss
            faiss.write_index(self._faiss_index, path)
            logger.info(f"FAISS index saved to {path}")

    def load(self, path: str):
        """Load FAISS index from disk."""
        if not Path(path).exists():
            return
        try:
            import faiss
            self._faiss_index = faiss.read_index(path)
            self._use_faiss = True
            logger.info(f"FAISS index loaded from {path}")
        except ImportError:
            logger.warning("FAISS not available for loading")

    @property
    def size(self) -> int:
        return len(self.chunks)