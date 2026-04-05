"""
chunk_expander.py — Parent-child chunk expansion.
When a passage chunk scores high, fetch its parent section chunk
and sibling passages to provide full context to the synthesizer.
"""

import logging
from backend.indexing.metadata_store import MetadataStore
from backend.retrieval.hybrid_recall import RetrievalCandidate

logger = logging.getLogger(__name__)


async def expand_with_parents(
    candidates: list[RetrievalCandidate],
    store: MetadataStore,
    max_parent_chars: int = 3000,
) -> list[RetrievalCandidate]:
    """
    For each passage-level candidate, fetch its parent section
    and inject truncated section context into the chunk metadata.
    """
    for cand in candidates:
        chunk = cand.chunk
        if chunk.get("granularity") != "passage":
            continue

        paper_id = chunk.get("paper_id")
        section_tag = chunk.get("section_tag")
        if not paper_id or not section_tag:
            continue

        try:
            section_chunks = await store.get_chunks_by_paper(
                paper_id, granularity="section"
            )
        except Exception as exc:
            logger.debug("expand_with_parents: store call failed for %s: %s", paper_id, exc)
            continue

        parent = next(
            (sc for sc in section_chunks if sc.get("section_tag") == section_tag),
            None,
        )
        if parent:
            parent_text = parent["text"][:max_parent_chars]
            chunk["parent_context"] = parent_text
            chunk["has_parent"] = True

    return candidates


async def fetch_sibling_passages(
    candidates: list[RetrievalCandidate],
    store: MetadataStore,
    window: int = 1,
) -> list[RetrievalCandidate]:
    """
    For top candidates, fetch adjacent passage chunks by char offset.
    This recovers context that the sliding window cut off.
    """
    expanded = list(candidates)
    seen_ids = {c.chunk["chunk_id"] for c in candidates}

    for cand in candidates[:8]:  # Only expand top candidates
        chunk = cand.chunk
        if chunk.get("granularity") != "passage":
            continue

        paper_id = chunk["paper_id"]
        section_tag = chunk.get("section_tag", "")

        try:
            all_passages = await store.get_chunks_by_paper(
                paper_id, granularity="passage"
            )
        except Exception as exc:
            logger.debug("fetch_sibling_passages: store call failed for %s: %s", paper_id, exc)
            continue
        # Find siblings in same section, sorted by position
        siblings = sorted(
            [p for p in all_passages if p.get("section_tag") == section_tag],
            key=lambda p: p.get("start_char", 0),
        )

        # Find current position
        current_idx = None
        for i, sib in enumerate(siblings):
            if sib["chunk_id"] == chunk["chunk_id"]:
                current_idx = i
                break

        if current_idx is None:
            continue

        # Add adjacent siblings
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            idx = current_idx + offset
            if 0 <= idx < len(siblings):
                sib = siblings[idx]
                if sib["chunk_id"] not in seen_ids:
                    expanded.append(RetrievalCandidate(
                        chunk=sib,
                        paper=cand.paper,
                        rrf_score=cand.rrf_score * 0.6,
                    ))
                    seen_ids.add(sib["chunk_id"])

    return expanded
