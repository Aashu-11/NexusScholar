"""
reranker.py — Cross-encoder re-ranking.
Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to re-score RRF candidates.
Multi-signal scoring: relevance, recency, citation count, section weight.
"""

from __future__ import annotations
import logging
import math
import re
from typing import Optional

from backend.config import settings
from backend.retrieval.hybrid_recall import RetrievalCandidate
from backend.ingestion.graph_builder import get_pagerank
from backend.retrieval.entity_extractor import QueryEntityProfile

logger = logging.getLogger(__name__)


def _get_reranker_device() -> str:
    """Pick CUDA if available, fall back to CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA available — loading reranker on GPU")
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class Reranker:
    def __init__(self):
        self._model = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            from sentence_transformers import CrossEncoder
            device = _get_reranker_device()
            self._model = CrossEncoder(settings.RERANKER_MODEL, device=device)
            logger.info("Loaded reranker: %s on %s", settings.RERANKER_MODEL, device)
        except Exception as e:
            logger.warning(f"Reranker not available: {e}. Using RRF fallback.")
            self._model = None
        self._loaded = True

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int = 15,
        intent: str = "general",
        entity_profile: Optional[QueryEntityProfile] = None,
    ) -> list[RetrievalCandidate]:
        """
        Re-rank candidates using cross-encoder + multi-signal scoring.
        """
        if not candidates:
            return []

        self._load()

        if self._model is None:
            # Fallback: use RRF scores only
            sorted_cands = sorted(candidates, key=lambda c: c.rrf_score, reverse=True)
            for i, c in enumerate(sorted_cands):
                c.rerank_score = c.rrf_score
                c.final_score = c.rrf_score
            logger.info("Reranker unavailable, using RRF fallback for %s candidates", len(candidates))
            return sorted_cands[:top_k]

        # Cross-encoder scoring
        # Truncate chunk text to 512 chars before encoding — the model's
        # tokenizer handles max_length internally, but shorter strings
        # reduce Python-side overhead and tokenization time significantly.
        _CE_TEXT_LIMIT = 512
        pairs = [(query, c.chunk["text"][:_CE_TEXT_LIMIT]) for c in candidates]
        try:
            # batch_size=64 balances GPU utilisation vs memory; show_progress_bar
            # disabled to avoid stdout noise in production.
            scores = self._model.predict(
                pairs,
                batch_size=64,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.warning(f"Cross-encoder prediction failed: {e}")
            sorted_cands = sorted(candidates, key=lambda c: c.rrf_score, reverse=True)
            return sorted_cands[:top_k]

        for cand, score in zip(candidates, scores):
            cand.rerank_score = float(score)

            # Multi-signal final score
            cand.final_score = self._compute_final_score(query, cand, intent=intent, entity_profile=entity_profile)

        reranked = sorted(candidates, key=lambda c: c.final_score, reverse=True)

        # ── Hard cutoff: drop candidates whose CE score is below threshold ──
        # Applied on sigmoid(raw CE score) so it's model-calibrated regardless of other signals.
        threshold = settings.RERANKER_SCORE_THRESHOLD
        min_keep = max(3, top_k // 4)
        filtered = [c for c in reranked if (1.0 / (1.0 + math.exp(-c.rerank_score))) >= threshold]
        if len(filtered) >= min_keep:
            dropped = len(reranked) - len(filtered)
            if dropped:
                logger.info("Hard cutoff (CE threshold=%.2f) dropped %d low-relevance candidates", threshold, dropped)
            reranked = filtered
        else:
            logger.info(
                "Hard cutoff skipped — would leave only %d candidates (min_keep=%d)", len(filtered), min_keep
            )

        # ── Elbow method: cut at the first large score drop between consecutive results ──
        elbow_drop = settings.RERANKER_ELBOW_DROP
        if len(reranked) > 2:
            for i in range(1, len(reranked)):
                drop = reranked[i - 1].final_score - reranked[i].final_score
                if drop >= elbow_drop:
                    logger.info(
                        "Elbow method cut at position %d (score drop=%.3f >= threshold=%.2f)",
                        i, drop, elbow_drop,
                    )
                    reranked = reranked[:i]
                    break

        logger.info("Reranker produced %s candidates after cutoffs; top_k=%s", len(reranked), top_k)
        for idx, cand in enumerate(reranked[:40], start=1):
            logger.info(
                "Reranked %s: chunk=%s rerank=%.4f final=%.4f paper=%s",
                idx,
                cand.chunk.get("chunk_id"),
                cand.rerank_score,
                cand.final_score,
                cand.chunk.get("paper_id"),
            )
        return reranked[:top_k]

    def _entity_consistency_score(
        self,
        entity_profile: Optional[QueryEntityProfile],
        cand: RetrievalCandidate,
    ) -> float:
        """
        Score how well a candidate matches the queried entity.

        Co-occurrence aware: a chunk that discusses BOTH the primary subject
        and an exclusion entity (common in survey/comparison papers) is NOT
        hard-rejected — only chunks where the exclusion entity appears without
        any trace of the primary subject are penalised to 0.0.

        Returns:
            1.0 — primary subject (or alias) confirmed present
            0.9 — only alias found
            0.6 — domain terminology present, no entity name
            0.5 — neutral (no entity signals)
            0.4 — both primary AND exclusion entity present (co-occurrence)
            0.0 — ONLY exclusion entity present, primary subject absent
        """
        if entity_profile is None or not entity_profile.requires_entity_grounding:
            cand.entity_decision = "neutral"
            return 1.0

        # Build search corpus from chunk + paper metadata
        corpus_parts = [
            cand.chunk.get("text", ""),
            cand.paper.get("title", ""),
            cand.paper.get("abstract", ""),
        ]
        corpus = " ".join(p for p in corpus_parts if p).lower()

        primary = entity_profile.primary_subject

        # 1. Check if primary subject (or aliases) is present
        primary_present = False
        if primary and re.search(r'\b' + re.escape(primary.lower()) + r'\b', corpus):
            primary_present = True
        if not primary_present:
            for alias in entity_profile.entity_aliases:
                if alias and re.search(r'\b' + re.escape(alias.lower()) + r'\b', corpus):
                    primary_present = True
                    break

        # 2. Check for exclusion entities
        exclusion_found: Optional[str] = None
        for excl in entity_profile.exclusion_entities:
            if excl and re.search(r'\b' + re.escape(excl.lower()) + r'\b', corpus):
                exclusion_found = excl
                break

        if exclusion_found:
            if primary_present:
                # Co-occurrence: paper discusses BOTH — mild penalty, not hard reject
                logger.info(
                    "Entity consistency: chunk=%s co-occurrence (primary=%r + excl=%r) — score=0.4",
                    cand.chunk.get("chunk_id"), primary, exclusion_found,
                )
                cand.entity_decision = "ambiguous"
                return 0.4
            else:
                # Exclusion entity only — hard reject
                logger.info(
                    "Entity consistency: chunk=%s ONLY exclusion=%r, no primary — score=0.0",
                    cand.chunk.get("chunk_id"), exclusion_found,
                )
                cand.entity_decision = "wrong"
                return 0.0

        if not primary:
            cand.entity_decision = "neutral"
            return 0.5

        if primary_present:
            cand.entity_decision = "correct"
            return 1.0

        # 3. Domain presence check
        domain_terms = {
            "nuclear_physics": ["reactor", "nuclear", "neutron", "fission", "moderator", "coolant"],
            "machine_learning": ["model", "training", "neural", "embedding", "attention", "transformer"],
            "medicine": ["drug", "patient", "clinical", "trial", "dose", "treatment", "therapy"],
            "chemistry": ["compound", "molecule", "reaction", "synthesis", "polymer", "element"],
            "biology": ["protein", "gene", "cell", "virus", "bacteria", "genome", "sequence"],
        }
        terms = domain_terms.get(entity_profile.domain, [])
        if any(t in corpus for t in terms):
            cand.entity_decision = "neutral"
            return 0.6

        cand.entity_decision = "neutral"
        return 0.5

    def _compute_final_score(
        self,
        query: str,
        cand: RetrievalCandidate,
        intent: str = "general",
        entity_profile: Optional[QueryEntityProfile] = None,
    ) -> float:
        """
        Combine cross-encoder relevance with auxiliary signals.
        Cross-encoder dominates (75%) — it's the strongest relevance signal.
        Auxiliary signals provide tiebreaking and diversity.
        """
        # Normalize cross-encoder score to 0-1 via sigmoid for consistent weighting
        ce_normalized = 1.0 / (1.0 + math.exp(-cand.rerank_score))
        score = ce_normalized * 0.75

        # Section boost: evidence-bearing sections weighted higher
        section = cand.chunk.get("section_tag", "unknown")
        section_weights = {
            "results": 0.10, "abstract": 0.09, "methods": 0.08,
            "conclusion": 0.07, "discussion": 0.06, "introduction": 0.03,
            "web_content": 0.04,
        }
        score += section_weights.get(section, 0.02)

        # Recency boost — more aggressive for benchmark/trend queries
        year = cand.paper.get("year")
        if year is None:
            # FIX: log missing year so we can identify papers that need metadata repair
            paper_id = cand.chunk.get("paper_id", "unknown")
            logger.debug("Paper %s has no year metadata — recency boost skipped", paper_id)
        if intent in ("benchmark_comparison", "trend_analysis"):
            if year and year >= 2024:
                score += 0.12
            elif year and year >= 2023:
                score += 0.08
            elif year and year <= 2020:
                score -= 0.10  # actively penalize stale results
        elif year and year >= 2023:
            score += 0.05
        elif year and year >= 2021:
            score += 0.03
        elif year and year >= 2018:
            score += 0.01

        # Citation count boost (highly cited = more authoritative)
        citations = cand.paper.get("citation_count", 0)
        if citations and citations > 500:
            score += 0.06
        elif citations and citations > 100:
            score += 0.04
        elif citations and citations > 20:
            score += 0.02

        # Topic alignment
        paper_text = " ".join(
            [
                cand.paper.get("title", ""),
                cand.paper.get("abstract", ""),
                cand.chunk.get("section_tag", ""),
            ]
        )
        score += 0.12 * _topic_alignment(query, paper_text)

        # Evidence density — chunks with numbers, metrics, and claims
        score += 0.10 * _evidence_density(cand.chunk.get("text", ""))

        # Granularity bonus
        granularity = cand.chunk.get("granularity")
        if granularity == "claim":
            score += 0.04
        elif granularity == "document":
            score += 0.03
        elif granularity == "section":
            score += 0.02

        # Peer-reviewed boost
        if cand.paper.get("is_peer_reviewed"):
            score += 0.03

        # PageRank authority boost (normalized 0-1 scale)
        # log-scale contribution so highly-cited papers don't overwhelm other signals
        pr_score = get_pagerank(cand.chunk.get("paper_id", ""))
        if pr_score > 0:
            score += 0.05 * pr_score   # capped contribution — PageRank is auxiliary signal

        # Entity consistency scoring — prevents wrong-entity chunks from surfacing
        entity_score = self._entity_consistency_score(entity_profile, cand)
        chunk_id = cand.chunk.get("chunk_id", "?")
        paper_title = cand.paper.get("title", "?")[:60]
        if entity_score == 0.0:
            # Hard wrong: ONLY exclusion entity present, primary absent
            score -= settings.ENTITY_GROUNDING_PENALTY
            logger.info("Entity: chunk=%s paper=%s score=0.0 → hard penalise", chunk_id, paper_title)
        elif entity_score >= 0.9:
            # Confirmed correct entity
            score += settings.ENTITY_GROUNDING_BOOST
            logger.info("Entity: chunk=%s paper=%s score=%.1f → boost", chunk_id, paper_title, entity_score)
        elif entity_score == 0.4:
            # Co-occurrence: both primary + exclusion present — mild penalty only
            score -= 0.06
            logger.info("Entity: chunk=%s paper=%s score=0.4 → co-occurrence mild penalty", chunk_id, paper_title)
        elif entity_score < 0.6:
            score -= 0.10
            logger.info("Entity: chunk=%s paper=%s score=%.2f → ambiguous mild penalty", chunk_id, paper_title, entity_score)

        return score


    async def listwise_rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        groq,
        top_k: int = 12,
    ) -> list[RetrievalCandidate]:
        """
        Second-stage listwise reranking via LLM.
        The LLM sees all candidates simultaneously and can reason
        about cross-document relevance, complementarity, and coverage.
        """
        if len(candidates) <= top_k:
            return candidates

        # Format candidates for LLM
        candidate_descriptions = []
        for i, cand in enumerate(candidates[:50]):
            paper_title = (cand.paper or {}).get("title", "Unknown")
            section = cand.chunk.get("section_tag", "unknown")
            preview = cand.chunk["text"][:300].replace('\n', ' ')
            candidate_descriptions.append(
                f"[{i}] {paper_title} ({section}): {preview}"
            )

        prompt = f"""Given this research query: "{query}"

Here are {len(candidate_descriptions)} candidate evidence passages. Rank the TOP {top_k} most relevant by their index numbers. Consider: direct relevance, specificity of evidence, complementarity (don't pick 5 passages saying the same thing).

Candidates:
{chr(10).join(candidate_descriptions)}

Return ONLY a JSON list of index numbers in order of relevance: [most_relevant_idx, second_idx, ...]"""

        try:
            response = await groq.complete_fast(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            import json
            raw = response.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            indices = json.loads(raw)

            reranked = []
            seen = set()
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                    reranked.append(candidates[idx])
                    seen.add(idx)

            # Add any remaining candidates not selected by LLM
            for i, cand in enumerate(candidates):
                if i not in seen and len(reranked) < top_k:
                    reranked.append(cand)

            return reranked[:top_k]
        except Exception as e:
            logger.warning("Listwise reranking failed: %s", e)
            return candidates[:top_k]


def _topic_alignment(query: str, text: str) -> float:
    stop = {"what", "are", "the", "recent", "advances", "for", "and", "with", "about", "a", "an", "of", "in", "to", "is", "how", "does"}
    q_terms = {
        token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", query.lower())
        if token not in stop
    }
    if not q_terms:
        return 0.0
    t_lower = text.lower()
    t_terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", t_lower))
    overlap = len(q_terms & t_terms) / len(q_terms)
    return max(0.0, min(1.0, overlap))


def _evidence_density(text: str) -> float:
    text_lower = text.lower()
    score = 0.0
    if re.search(r"\b(outperform|improv|achiev|state-of-the-art|sota|significant|better than)\b", text_lower):
        score += 0.45
    if re.search(r"\b\d+(?:\.\d+)?%|\b\d+\.\d+\b", text_lower):
        score += 0.3
    if re.search(r"\b(accuracy|f1|precision|recall|bleu|rouge|mmlu|glue|auc)\b", text_lower):
        score += 0.25
    return max(0.0, min(1.0, score))
