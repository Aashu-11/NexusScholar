"""
colbert_index.py — ColBERT late-interaction retrieval via RAGatouille.
Optional third retrieval lane. Degrades gracefully if ragatouille not installed.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)

_ragatouille_model = None
_index_path: Optional[str] = None


def is_available() -> bool:
    try:
        import ragatouille  # noqa: F401
        return True
    except ImportError:
        return False


def get_model():
    global _ragatouille_model
    if _ragatouille_model is not None:
        return _ragatouille_model
    if not is_available():
        return None
    try:
        from ragatouille import RAGPretrainedModel
        _ragatouille_model = RAGPretrainedModel.from_pretrained(
            settings.COLBERT_MODEL,
            index_root=str(Path(settings.INDEX_PATH) / "colbert"),
        )
        logger.info("ColBERT model loaded: %s", settings.COLBERT_MODEL)
        return _ragatouille_model
    except Exception as e:
        logger.warning("ColBERT load failed: %s", e)
        return None


async def build_colbert_index(chunks: list[dict]) -> bool:
    """Index all passage-level chunks. Returns True on success."""
    model = get_model()
    if not model:
        return False
    try:
        texts = [c["text"] for c in chunks if c.get("text")]
        doc_ids = [c["chunk_id"] for c in chunks if c.get("text")]
        if not texts:
            return False
        model.index(
            collection=texts,
            document_ids=doc_ids,
            index_name="nexus_passages",
            max_document_length=512,
            split_documents=False,
            overwrite_index=True,
        )
        logger.info("ColBERT index built: %s passages", len(texts))
        return True
    except Exception as e:
        logger.warning("ColBERT index build failed: %s", e)
        return False


def colbert_search(
    query: str,
    top_k: int = 50,
    chunk_lookup: dict = None,
) -> list[tuple[dict, float, int]]:
    """
    Search ColBERT index. Returns (chunk_dict, score, rank).
    chunk_lookup maps chunk_id → chunk dict for score assembly.
    """
    model = get_model()
    if not model or chunk_lookup is None:
        return []
    try:
        results = model.search(query=query, k=top_k)
        output = []
        for rank, r in enumerate(results):
            chunk_id = r.get("document_id")
            score = r.get("score", 0.0)
            chunk = chunk_lookup.get(chunk_id)
            if chunk:
                output.append((chunk, float(score), rank))
        return output
    except Exception as e:
        logger.warning("ColBERT search failed: %s", e)
        return []
