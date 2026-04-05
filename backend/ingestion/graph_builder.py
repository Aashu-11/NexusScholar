"""
graph_builder.py — Citation graph construction and persistence.
Uses NetworkX for in-memory graph ops + SQLite for edge persistence.
Supports: cited-by, references, co-citation clusters, shared dataset neighbors.
"""

from __future__ import annotations
import logging
from typing import Optional
import networkx as nx

from backend.indexing.metadata_store import MetadataStore

logger = logging.getLogger(__name__)

# Global in-memory citation graph
_graph: Optional[nx.DiGraph] = None

# PageRank cache
_pagerank_cache: dict[str, float] = {}
_pagerank_dirty: bool = True


def get_graph() -> nx.DiGraph:
    global _graph
    if _graph is None:
        _graph = nx.DiGraph()
    return _graph


def compute_pagerank(alpha: float = 0.85) -> dict[str, float]:
    """Compute PageRank scores for all papers in the citation graph."""
    global _pagerank_cache, _pagerank_dirty
    if not _pagerank_dirty and _pagerank_cache:
        return _pagerank_cache

    g = get_graph()
    if g.number_of_nodes() == 0:
        return {}
    try:
        scores = nx.pagerank(g, alpha=alpha, max_iter=200, tol=1e-6)
        _pagerank_cache = scores
        _pagerank_dirty = False
        top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("PageRank computed for %s nodes. Top 5: %s", len(scores), top5)
        return scores
    except Exception as e:
        logger.warning("PageRank computation failed: %s", e)
        return {}


def get_pagerank(paper_id: str) -> float:
    """Get normalized PageRank score for a paper (0.0 if unknown)."""
    scores = compute_pagerank()
    if not scores:
        return 0.0
    raw = scores.get(paper_id, 0.0)
    # Normalize against max score so boosts are consistent regardless of corpus size
    max_score = max(scores.values()) if scores else 1.0
    return raw / max_score if max_score > 0 else 0.0


async def build_graph_from_db(store: MetadataStore):
    """Rebuild the in-memory citation graph from the database."""
    global _graph, _pagerank_dirty
    _graph = nx.DiGraph()

    edges = await store.get_all_citation_edges()
    for src, tgt in edges:
        _graph.add_edge(src, tgt)

    papers = await store.get_all_papers()
    for p in papers:
        _graph.add_node(p["paper_id"], **{
            "title": p["title"],
            "year": p.get("year"),
            "venue": p.get("venue"),
            "citation_count": p.get("citation_count", 0),
        })

    logger.info(f"Citation graph built: {_graph.number_of_nodes()} nodes, {_graph.number_of_edges()} edges")

    # Eagerly compute PageRank so it is cached at startup
    _pagerank_dirty = True
    compute_pagerank()


async def add_paper_to_graph(paper_id: str, references: list[dict], store: MetadataStore):
    """
    Insert a newly ingested paper into the citation graph.
    Resolves reference edges where target papers already exist.
    """
    global _pagerank_dirty
    g = get_graph()
    g.add_node(paper_id)

    for ref in references:
        # Try to resolve the reference to an existing paper in the DB
        resolved_id = await _resolve_reference(ref, store)
        if resolved_id:
            g.add_edge(paper_id, resolved_id)  # paper cites resolved_id
            await store.insert_citation_edge(paper_id, resolved_id)

    # Mark PageRank as dirty — new edges change the graph structure
    _pagerank_dirty = True


async def _resolve_reference(ref: dict, store: MetadataStore) -> Optional[str]:
    """
    Attempt to match a reference entry to a paper already in the corpus.
    Uses title similarity matching.
    """
    ref_title = ref.get("title", "").strip()
    if not ref_title or len(ref_title) < 10:
        return None

    # Search for matching paper by title
    matches = await store.search_papers_by_title(ref_title, limit=1)
    if matches:
        return matches[0]["paper_id"]
    return None


def get_cited_by(paper_id: str) -> list[str]:
    """Papers that cite this one (incoming edges)."""
    g = get_graph()
    if paper_id not in g:
        return []
    return list(g.predecessors(paper_id))


def get_references(paper_id: str) -> list[str]:
    """Papers this one cites (outgoing edges)."""
    g = get_graph()
    if paper_id not in g:
        return []
    return list(g.successors(paper_id))


def get_co_citation_cluster(paper_id: str, max_size: int = 20) -> list[str]:
    """
    Papers frequently co-cited alongside this paper.
    Two papers are co-cited if a third paper cites both.
    """
    g = get_graph()
    if paper_id not in g:
        return []

    citers = set(g.predecessors(paper_id))
    co_cited_counts: dict[str, int] = {}

    for citer in citers:
        for also_cited in g.successors(citer):
            if also_cited != paper_id:
                co_cited_counts[also_cited] = co_cited_counts.get(also_cited, 0) + 1

    sorted_peers = sorted(co_cited_counts.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in sorted_peers[:max_size]]


def get_neighborhood(paper_id: str, hops: int = 1) -> set[str]:
    """Get all papers within N citation hops."""
    g = get_graph()
    if paper_id not in g:
        return set()

    visited = {paper_id}
    frontier = {paper_id}

    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            for neighbor in list(g.predecessors(node)) + list(g.successors(node)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier

    visited.discard(paper_id)
    return visited