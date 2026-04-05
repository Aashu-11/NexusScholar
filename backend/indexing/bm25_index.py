"""
bm25_index.py — BM25 sparse retrieval index.
Built in-memory from passage-level chunks. Rebuilt on corpus changes.
Enhanced tokenizer with scientific text support and lightweight stemming.
"""

from __future__ import annotations
import re
import logging
from typing import Optional
from rank_bm25 import BM25Okapi

from backend.config import settings
from backend.indexing.metadata_store import MetadataStore

logger = logging.getLogger(__name__)

# Lightweight stemming rules for scientific text
_SUFFIX_RULES = [
    (r'ies$', 'y'),
    (r'ves$', 'f'),
    (r'(ss|zz|sh|ch|x)es$', r'\1'),
    (r'ses$', 's'),
    (r'([^aeiou])es$', r'\1e'),
    (r'([^s])s$', r'\1'),
    (r'ing$', ''),
    (r'tion$', 't'),
    (r'ment$', ''),
    (r'ness$', ''),
    (r'able$', ''),
    (r'ible$', ''),
    (r'ally$', ''),
]


class BM25Index:
    def __init__(self):
        self.index: Optional[BM25Okapi] = None
        self.chunks: list[dict] = []
        self._tokenized: list[list[str]] = []

    async def build(
        self,
        store: MetadataStore,
        granularity: str = "passage",
        extra_granularities: list[str] = None,
    ):
        """Load chunks and build BM25 index. Optionally include additional granularities."""
        self.chunks = await store.get_all_chunks(granularity=granularity)

        # Merge in additional granularities (e.g. tables)
        for extra in (extra_granularities or []):
            extra_chunks = await store.get_all_chunks(granularity=extra)
            if extra_chunks:
                logger.info("Adding %s '%s' chunks to BM25 index", len(extra_chunks), extra)
                self.chunks.extend(extra_chunks)

        if not self.chunks:
            logger.warning(f"No {granularity} chunks found for BM25 index")
            self.index = None
            return

        self._tokenized = [tokenize(c["text"]) for c in self.chunks]
        self.index = BM25Okapi(self._tokenized, k1=settings.BM25_K1, b=settings.BM25_B)
        logger.info(
            "BM25 index built: %d chunks (primary=%s, k1=%.2f, b=%.2f)",
            len(self.chunks), granularity, settings.BM25_K1, settings.BM25_B,
        )

    def search(self, query: str, top_k: int = 100) -> list[tuple[dict, float, int]]:
        """
        Search the BM25 index.
        Returns list of (chunk_dict, score, rank).
        """
        if not self.index or not self.chunks:
            return []

        tokens = tokenize(query)
        scores = self.index.get_scores(tokens)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            (self.chunks[i], float(scores[i]), rank)
            for rank, i in enumerate(ranked)
            if scores[i] > 0
        ]

    @property
    def size(self) -> int:
        return len(self.chunks)


# Scientific compound terms that must NOT be split at hyphens
# These are exact entity names where splitting would destroy their identity
PROTECTED_COMPOUNDS = {
    # Nuclear reactor types
    "rbmk", "rbmk-1000", "rbmk-1500", "triga", "vver", "vver-1000", "candu",
    "pressurized-water", "boiling-water", "scram", "xenon-poisoning",
    # ML models
    "bert-base", "bert-large", "gpt-4", "gpt-3.5", "llama-2", "llama-3",
    "claude-3", "gemini-pro", "mistral-7b", "gpt-4o", "llama-3.1",
    # Chemistry / Biology
    "sars-cov-2", "sars-cov-1", "covid-19", "h2o2", "co2",
}


def tokenize(text: str) -> list[str]:
    """
    Enhanced tokenizer for scientific text:
    - Lowercases
    - Preserves hyphens in compound terms (state-of-the-art)
    - Preserves decimal numbers (98.5%)
    - Lightweight suffix normalization
    - Removes pure-numeric tokens under 4 digits
    - Protects known scientific compound names from being split
    """
    text_lower = text.lower()

    # Pre-extract protected compounds before general tokenization
    # Replace hyphens in protected compounds with underscores so they survive splitting
    protected_found: list[str] = []
    for compound in PROTECTED_COMPOUNDS:
        if compound in text_lower:
            protected_found.append(compound)
            # Also keep as single token (underscore variant for BM25)
            text_lower = text_lower.replace(compound, compound.replace("-", "_"))

    # Extract tokens preserving hyphens and decimals
    tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9-]*[a-zA-Z0-9]|[a-zA-Z]|\d+\.\d+', text_lower)

    result = []
    # Add back protected compound originals (with hyphens restored)
    for compound in protected_found:
        result.append(compound)  # Keep "rbmk-1000" as single token

    for token in tokens:
        if len(token) < 2:
            continue
        # Restore underscore-encoded protected compounds to hyphenated form
        if '_' in token:
            hyphenated = token.replace("_", "-")
            result.append(hyphenated)
            continue
        # Keep hyphenated terms as-is AND split them
        if '-' in token and len(token) > 3:
            result.append(token)  # Keep compound: "state-of-the-art"
            parts = token.split('-')
            result.extend(p for p in parts if len(p) > 1)
        else:
            result.append(token)
            # Add stem variant
            stemmed = _light_stem(token)
            if stemmed != token and len(stemmed) > 2:
                result.append(stemmed)

    return result


def _light_stem(word: str) -> str:
    for pattern, replacement in _SUFFIX_RULES:
        new_word = re.sub(pattern, replacement, word)
        if new_word != word and len(new_word) > 2:
            return new_word
    return word
