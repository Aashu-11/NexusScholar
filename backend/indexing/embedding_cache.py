"""
embedding_cache.py — Persistent embedding cache using SQLite.
Avoids recomputing embeddings for chunks that haven't changed.
"""

import hashlib
import logging
import aiosqlite
from pathlib import Path
from typing import Optional

import numpy as np

from backend.config import settings

logger = logging.getLogger(__name__)


class EmbeddingCache:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(
            Path(settings.INDEX_PATH) / "embedding_cache.db"
        )

    async def init(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        db = await aiosqlite.connect(self.db_path)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                model_name TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()
        await db.close()
        logger.info("Embedding cache initialized at %s", self.db_path)

    async def get(self, text: str) -> Optional[np.ndarray]:
        text_hash = self._hash(text)
        db = await aiosqlite.connect(self.db_path)
        try:
            cur = await db.execute(
                "SELECT embedding FROM embeddings WHERE text_hash = ? AND model_name = ?",
                (text_hash, settings.EMBEDDING_MODEL),
            )
            row = await cur.fetchone()
            if row:
                return np.frombuffer(row[0], dtype=np.float32)
            return None
        finally:
            await db.close()

    async def put(self, text: str, embedding: np.ndarray):
        text_hash = self._hash(text)
        db = await aiosqlite.connect(self.db_path)
        try:
            await db.execute(
                "INSERT OR REPLACE INTO embeddings (text_hash, embedding, model_name) VALUES (?, ?, ?)",
                (text_hash, embedding.astype(np.float32).tobytes(), settings.EMBEDDING_MODEL),
            )
            await db.commit()
        finally:
            await db.close()

    async def get_batch(self, texts: list[str]) -> dict[str, Optional[np.ndarray]]:
        """Batch lookup — returns {text: embedding_or_None}."""
        results = {}
        hashes = {self._hash(t): t for t in texts}

        db = await aiosqlite.connect(self.db_path)
        try:
            placeholders = ','.join('?' for _ in hashes)
            cur = await db.execute(
                f"SELECT text_hash, embedding FROM embeddings WHERE text_hash IN ({placeholders}) AND model_name = ?",
                list(hashes.keys()) + [settings.EMBEDDING_MODEL],
            )
            found = {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in await cur.fetchall()}

            for h, text in hashes.items():
                results[text] = found.get(h)
        finally:
            await db.close()

        return results

    async def put_batch(self, items: list[tuple[str, np.ndarray]]):
        """Batch insert — items is list of (text, embedding)."""
        if not items:
            return
        db = await aiosqlite.connect(self.db_path)
        try:
            await db.executemany(
                "INSERT OR REPLACE INTO embeddings (text_hash, embedding, model_name) VALUES (?, ?, ?)",
                [(self._hash(text), emb.astype(np.float32).tobytes(), settings.EMBEDDING_MODEL)
                 for text, emb in items],
            )
            await db.commit()
            logger.info("Cached %d embeddings", len(items))
        finally:
            await db.close()

    async def stats(self) -> dict:
        """Return cache statistics."""
        db = await aiosqlite.connect(self.db_path)
        try:
            cur = await db.execute("SELECT COUNT(*) FROM embeddings WHERE model_name = ?", (settings.EMBEDDING_MODEL,))
            count = (await cur.fetchone())[0]
            return {"cached_embeddings": count, "model": settings.EMBEDDING_MODEL}
        finally:
            await db.close()

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]
