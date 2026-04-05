"""
/api/chat — SSE streaming endpoint.
Orchestrates the full pipeline: query understanding → retrieval → generation → verification.
"""

from __future__ import annotations
import json
import uuid
import logging
import re
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from backend.config import settings
from backend.retrieval.query_classifier import classify_intent
from backend.retrieval.query_rewriter import rewrite_query
from backend.retrieval.hybrid_recall import RetrievalCandidate, hybrid_recall
from backend.retrieval.reranker import Reranker
from backend.retrieval.graph_expander import graph_expand
from backend.generation.planner import plan_response
from backend.generation.evidence_builder import build_evidence_table
from backend.generation.synthesizer import synthesize
from backend.generation.verifier import verify_answer
from backend.generation.entity_verifier import verify_entity_consistency
from backend.citation.renderer import render_citations
from backend.integrations.tavily_client import search_papers as search_tavily
from backend.integrations.exa_client import search_papers as search_exa
from backend.integrations.semantic_scholar import search_semantic_scholar
from backend.retrieval.compressor import compress_chunks
from backend.retrieval.chunk_expander import expand_with_parents, fetch_sibling_passages
from backend.retrieval.pseudo_relevance_feedback import extract_expansion_terms, build_expanded_query
from backend.generation.markdown_fixer import ensure_table_in_synthesis
from backend.generation.evidence_dedup import deduplicate_evidence
from backend.generation.self_evaluator import evaluate_answer
from backend.ingestion.service import ingest_pdf_file, ingest_text_document, rebuild_indexes
from backend.retrieval.entity_extractor import QueryEntityProfile
from backend.generation.question_decomposer import decompose_question, QuestionDecomposition
from backend.generation.coverage_verifier import verify_coverage

logger = logging.getLogger(__name__)
router = APIRouter()


@dataclass(frozen=True)
class YearConstraint:
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    source: str = "none"


class ChatRequest(BaseModel):
    query: str
    conversation_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    corpus_id: str = "default"
    recency_filter: str = "any"  # any | 1y | 3y
    intent_override: Optional[str] = None


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint — streams SSE events through the full pipeline."""
    from backend.main import get_store, get_groq, get_bm25, get_dense, get_reranker

    store = get_store()
    groq = get_groq()
    bm25 = get_bm25()
    dense = get_dense()
    reranker = get_reranker()

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            logger.info("Chat request: conversation=%s query=%r", request.conversation_id, request.query)
            # Upsert conversation
            await store.upsert_conversation(request.conversation_id, corpus_id=request.corpus_id)
            await store.insert_message({
                "message_id": uuid.uuid4().hex[:12],
                "conversation_id": request.conversation_id,
                "role": "user",
                "content": request.query,
            })

            # ── Stage 0: Compound Question Decomposition ──
            decomposition = await decompose_question(request.query, groq)
            if decomposition.is_compound:
                logger.info(
                    "Compound question detected: %d sub-questions — %s",
                    decomposition.count,
                    decomposition.sub_questions,
                )
                yield _sse("question_decomposition", {
                    "is_compound": True,
                    "sub_questions": decomposition.sub_questions,
                    "reasoning": decomposition.reasoning,
                })
            else:
                logger.info("Single-question query — skipping decomposition")

            # ── Stage A: Query Understanding ──────────────
            yield _sse("intent", {"status": "analyzing_query"})

            intent = request.intent_override or await classify_intent(request.query, groq)

            # For compound queries: run query rewriting for each sub-question
            # then merge all query forms to cast the widest retrieval net.
            if decomposition.is_compound and decomposition.count >= 2:
                query_analysis = await _merge_sub_question_analyses(
                    decomposition.sub_questions, intent, groq
                )
            else:
                query_analysis = await rewrite_query(request.query, intent, groq)

            entity_profile = query_analysis.entity_profile
            year_constraint = _resolve_year_constraint(request.query, request.recency_filter)

            # Auto-tighten recency for benchmark/trend queries when user didn't specify one.
            # Skip if the query is clearly about historical/foundational work.
            if (
                intent in ("benchmark_comparison", "trend_analysis")
                and year_constraint.year_min is None
                and year_constraint.year_max is None
                and request.recency_filter == "any"
                and not _is_historical_query(request.query)
            ):
                year_constraint = _resolve_year_constraint(request.query, "2y")
                logger.info(
                    "Auto-applied 2y recency constraint for intent=%s (user specified 'any')",
                    intent,
                )

            logger.info(
                "Pipeline stage=query_understanding intent=%s dense_query=%r bm25_query=%r year_min=%r year_max=%r year_source=%s",
                intent,
                query_analysis.dense_query,
                query_analysis.bm25_query,
                year_constraint.year_min,
                year_constraint.year_max,
                year_constraint.source,
            )

            allowed_paper_ids: set[str] = set()
            exa_paper_ids: set[str] = set()
            use_exa_only = False

            # ── External retrieval: Exa primary (§7.4) ───────────
            if settings.EXA_AUTO_FETCH:
                exa_paper_ids = await _hydrate_from_exa(
                    query_analysis,
                    store,
                    bm25,
                    dense,
                    year_constraint=year_constraint,
                    intent=intent,
                    entity_profile=entity_profile,
                    decomposition=decomposition,
                )
                use_exa_only = len(exa_paper_ids) >= 2
                allowed_paper_ids |= exa_paper_ids

            # ── Tavily: strict fallback — only when Exa insufficient (§7.4.2) ──
            if not use_exa_only and settings.TAVILY_AUTO_FETCH:
                tavily_paper_ids = await _hydrate_from_tavily(
                    query_analysis,
                    store,
                    bm25,
                    dense,
                    year_constraint=year_constraint,
                    intent=intent,
                )
                allowed_paper_ids |= tavily_paper_ids
                logger.info("Tavily fallback hydration: %s papers", len(tavily_paper_ids))

            logger.info(
                "External retrieval: source=%s exa_results=%s tavily_used=%s",
                "exa" if use_exa_only else "exa+tavily",
                len(exa_paper_ids),
                not use_exa_only and settings.TAVILY_AUTO_FETCH,
            )

            s2_paper_ids: set[str] = set()
            if settings.S2_AUTO_FETCH:
                s2_paper_ids = await _hydrate_from_semantic_scholar(
                    query_analysis, store, bm25, dense, year_constraint=year_constraint,
                    intent=intent, decomposition=decomposition,
                )
                allowed_paper_ids |= s2_paper_ids
                logger.info("Semantic Scholar hydration added %s papers", len(s2_paper_ids))

            if allowed_paper_ids:
                yield _sse("retrieval", {
                    "status": "sources_ingested",
                    "papers_downloaded": len(allowed_paper_ids),
                })

            yield _sse("intent", {
                "intent": intent,
                "dense_query": query_analysis.dense_query,
                "bm25_query": query_analysis.bm25_query,
                "year_min": year_constraint.year_min,
                "year_max": year_constraint.year_max,
            })

            # Emit entity grounding event when a specific entity was detected
            if entity_profile and entity_profile.primary_subject:
                yield _sse("entity_grounding", {
                    "primary_subject": entity_profile.primary_subject,
                    "entity_type": entity_profile.entity_type,
                    "exclusion_count": len(entity_profile.exclusion_entities),
                    "requires_grounding": entity_profile.requires_entity_grounding,
                })

            # ── Stage B: Hybrid Recall ────────────────────
            yield _sse("retrieval", {"status": "searching", "phase": "hybrid_recall"})

            candidates = hybrid_recall(query_analysis, bm25, dense)
            if allowed_paper_ids:
                # Prioritize Tavily-fetched papers but don't discard local corpus entirely
                tavily_cands = [c for c in candidates if c.chunk.get("paper_id") in allowed_paper_ids]
                other_cands = [c for c in candidates if c.chunk.get("paper_id") not in allowed_paper_ids]
                # Boost Tavily candidates' RRF scores for priority
                for c in tavily_cands:
                    c.rrf_score *= 1.5
                candidates = tavily_cands + other_cands[:max(20, len(other_cands) // 2)]
                logger.info("Prioritized %s Tavily papers, kept %s local corpus candidates", len(tavily_cands), len(candidates) - len(tavily_cands))

            yield _sse("retrieval", {
                "status": "fused",
                "candidates_found": len(candidates),
            })

            if not candidates:
                yield _sse("error", {
                    "message": "No relevant papers found in the indexed corpus.",
                    "suggestion": "Try uploading relevant papers or broadening your query.",
                })
                return

            # ── Stage C: Re-ranking ───────────────────────
            yield _sse("retrieval", {"status": "reranking"})

            # Resolve paper metadata — single batch query instead of N sequential calls
            _paper_ids = [c.chunk["paper_id"] for c in candidates]
            _papers_map = await store.get_papers_batch(_paper_ids)
            for c in candidates:
                c.paper = _papers_map.get(c.chunk["paper_id"]) or {}
            logger.info("Resolved metadata for %s fused candidates (batch)", len(candidates))
            candidates = _filter_candidates_by_year(candidates, year_constraint)
            logger.info("Year filtering kept %s candidates", len(candidates))
            candidates = _filter_candidates_by_paper(query_analysis, candidates)
            logger.info("Paper-first filtering kept %s candidates", len(candidates))
            candidates = await _augment_with_selected_paper_chunks(request.query, query_analysis, candidates, store)
            logger.info("Paper-context augmentation expanded candidate pool to %s", len(candidates))

            reranked = reranker.rerank(request.query, candidates, top_k=settings.RERANKED_TOP_K, intent=intent, entity_profile=entity_profile)
            logger.info("Pipeline stage=rerank kept %s candidates", len(reranked))

            # ── Pseudo-Relevance Feedback ─────────────────
            expansion_terms = extract_expansion_terms(request.query, reranked)
            if expansion_terms:
                expanded_query = build_expanded_query(query_analysis.bm25_query or request.query, expansion_terms)
                logger.info("PRF expansion terms: %s", expansion_terms)
                prf_results = bm25.search(expanded_query, top_k=30)
                prf_seen = {c.chunk["chunk_id"] for c in reranked}
                prf_new = [
                    (chunk, score)
                    for chunk, score, _ in prf_results
                    if chunk["chunk_id"] not in prf_seen
                ][:10]
                if prf_new:
                    prf_batch_ids = [chunk["paper_id"] for chunk, _ in prf_new]
                    prf_papers_map = await store.get_papers_batch(prf_batch_ids)
                    for chunk, score in prf_new:
                        reranked.append(RetrievalCandidate(
                            chunk=chunk,
                            paper=prf_papers_map.get(chunk["paper_id"]) or {},
                            rrf_score=score * 0.5,
                        ))
                        prf_seen.add(chunk["chunk_id"])
                logger.info("PRF added %s candidates", len(prf_new))

            # ── Parent-Child Chunk Expansion ──────────────
            reranked = await expand_with_parents(reranked, store)
            reranked = await fetch_sibling_passages(reranked, store)
            logger.info("Post parent-child expansion: %s candidates", len(reranked))

            # ── Stage D: Graph Expansion ──────────────────
            expanded = await graph_expand(reranked, store)
            logger.info("Pipeline stage=graph_expand added %s candidates", len(expanded))

            # Merge + re-rank final set
            seen = {c.chunk["chunk_id"] for c in reranked}
            new_expanded = [ec for ec in expanded if ec.chunk["chunk_id"] not in seen]
            if new_expanded:
                _exp_ids = [ec.chunk["paper_id"] for ec in new_expanded]
                _exp_map = await store.get_papers_batch(_exp_ids)
                for ec in new_expanded:
                    ec.paper = _exp_map.get(ec.chunk["paper_id"]) or {}
                    reranked.append(ec)
                    seen.add(ec.chunk["chunk_id"])

            final_candidates = reranker.rerank(
                request.query, reranked, top_k=settings.FINAL_EVIDENCE_TOP_K, intent=intent,
                entity_profile=entity_profile,
            )
            final_candidates = _filter_candidates_by_year(final_candidates, year_constraint)
            final_candidates = _limit_chunks_per_paper(final_candidates, max_chunks_per_paper=3)
            logger.info("Pipeline stage=final_rerank final_candidates=%s", len(final_candidates))

            # ── Listwise LLM Re-ranking (second stage) ───
            try:
                final_candidates = await reranker.listwise_rerank(
                    request.query, final_candidates, groq, top_k=min(12, len(final_candidates))
                )
                logger.info("Listwise reranking refined to %s candidates", len(final_candidates))
            except Exception as e:
                logger.warning("Listwise reranking skipped: %s", e)
            for idx, cand in enumerate(final_candidates[:5], start=1):
                logger.info(
                    "Final candidate %s: paper=%s title=%s section=%s final=%.4f",
                    idx,
                    cand.chunk.get("paper_id"),
                    cand.paper.get("title"),
                    cand.chunk.get("section_tag"),
                    cand.final_score,
                )

            # ── Contextual Compression ────────────────────
            # max_candidates matches actual list size so every candidate is compressed.
            final_candidates = await compress_chunks(
                request.query, final_candidates, groq,
                max_candidates=len(final_candidates),
                entity_profile=entity_profile,
            )
            logger.info("Post-compression candidate count: %s", len(final_candidates))

            # ── Corpus Coverage Check ─────────────────────
            if settings.CORPUS_GAP_ABSTENTION_ENABLED:
                coverage = _check_entity_coverage(final_candidates, entity_profile, reranker)
                if not coverage["has_coverage"]:
                    missing = coverage.get("missing_entity", "the requested entity")
                    yield _sse("corpus_gap", {
                        "missing_entity": missing,
                        "consistent_count": coverage.get("consistent_count", 0),
                        "total_count": coverage.get("total_count", len(final_candidates)),
                    })
                    abstention_text = _build_corpus_gap_abstention(request.query, missing)
                    yield _sse("synthesis_token", {"token": abstention_text})
                    yield _sse("answer_complete", {
                        "answer_id": "",
                        "query": request.query,
                        "intent": intent,
                        "markdown_text": abstention_text,
                        "citations": [],
                        "is_abstention": True,
                        "uncertainty_flags": [f"Entity '{missing}' not found in corpus"],
                        "total_sources": 0,
                        "peer_reviewed_count": 0,
                        "preprint_count": 0,
                    })
                    return

            yield _sse("retrieval", {
                "status": "complete",
                "final_candidates": len(final_candidates),
            })

            # ── Step 1: Planner ───────────────────────────
            yield _sse("planning", {"status": "planning"})

            task_plan = await plan_response(request.query, intent, final_candidates, groq)
            logger.info(
                "Planner: format=%s confidence=%s sufficient=%s abstain=%s",
                task_plan.response_format,
                task_plan.confidence_level,
                task_plan.is_retrieval_sufficient,
                task_plan.should_abstain,
            )

            yield _sse("planning", {
                "intent": intent,
                "response_format": task_plan.response_format,
                "confidence": task_plan.confidence_level,
                "is_sufficient": task_plan.is_retrieval_sufficient,
            })

            # ── Step 2: Evidence Table ────────────────────
            evidence_table = build_evidence_table(request.query, intent, final_candidates)
            # Deduplicate evidence rows to avoid wasting context window
            original_count = len(evidence_table.rows)
            evidence_table.rows = deduplicate_evidence(evidence_table.rows)
            if len(evidence_table.rows) < original_count:
                logger.info("Evidence dedup removed %s duplicate rows", original_count - len(evidence_table.rows))
            logger.info(
                "Evidence table built: rows=%s confidence=%.4f",
                evidence_table.total_sources,
                evidence_table.confidence_score,
            )
            for idx, row in enumerate(evidence_table.rows[:5], start=1):
                logger.info(
                    "Evidence %s: %s | %s | %s",
                    idx,
                    row.paper_title,
                    row.source_url,
                    row.section_tag,
                )

            yield _sse("evidence", {
                "answer_id": evidence_table.answer_id,
                "total_rows": evidence_table.total_sources,
                "confidence": evidence_table.confidence_score,
                "rows": [
                    {
                        "evidence_id": r.evidence_id,
                        "paper_title": r.paper_title,
                        "authors": r.authors,
                        "year": r.year,
                        "venue": r.venue,
                        "section": r.section_tag,
                        "passage_preview": r.chunk_text[:200],
                        "relevance": round(r.relevance_score, 3),
                        "is_peer_reviewed": r.is_peer_reviewed,
                        "is_retracted": r.is_retracted,
                        "source_url": r.source_url,
                        "pdf_url": r.pdf_url,
                    }
                    for r in evidence_table.rows
                ],
            })

            # ── Step 3: Synthesis (streaming) ─────────────
            full_text = ""
            gen = await synthesize(
                request.query, intent, evidence_table, task_plan, groq, stream=True,
                entity_profile=entity_profile,
                decomposition=decomposition,
            )
            async for token in gen:
                full_text += token
                yield _sse("synthesis_token", {"token": token})

            # ── Post-synthesis: Fix markdown tables ──────
            full_text = ensure_table_in_synthesis(full_text)
            logger.info("Markdown table formatting applied")

            # ── Self-evaluation + conditional regeneration ──────────
            MAX_REGENERATION_ATTEMPTS = 2
            regen_attempt = 0

            while regen_attempt <= MAX_REGENERATION_ATTEMPTS:
                try:
                    quality_score = await evaluate_answer(
                        request.query, full_text, evidence_table.total_sources, groq
                    )
                    logger.info(
                        "Self-evaluation attempt=%s: overall=%s composite=%.2f issues=%s",
                        regen_attempt, quality_score.overall, quality_score.composite,
                        quality_score.issues,
                    )
                except Exception as e:
                    logger.warning("Self-evaluation skipped: %s", e)
                    break

                if not quality_score.needs_regeneration or regen_attempt >= MAX_REGENERATION_ATTEMPTS:
                    break

                # Quality insufficient — regenerate with targeted instructions
                regen_attempt += 1
                logger.info(
                    "Regenerating answer (attempt %s): issues=%s",
                    regen_attempt, quality_score.issues
                )
                yield _sse("regenerating", {
                    "attempt": regen_attempt,
                    "issues": quality_score.issues,
                    "previous_score": quality_score.overall,
                })

                # Build regeneration hint from identified issues
                issue_hint = ""
                if quality_score.issues:
                    issue_hint = (
                        "\n\nPREVIOUS ATTEMPT ISSUES (fix these):\n"
                        + "\n".join(f"- {issue}" for issue in quality_score.issues)
                    )

                # Re-synthesize with the hint injected into the task plan
                from backend.generation.planner import TaskPlan
                patched_plan = TaskPlan(
                    intent=task_plan.intent,
                    response_format=task_plan.response_format,
                    confidence_level=task_plan.confidence_level,
                    confidence_score=task_plan.confidence_score,
                    is_retrieval_sufficient=task_plan.is_retrieval_sufficient,
                    should_abstain=False,
                    reasoning=task_plan.reasoning + issue_hint,
                    key_points=task_plan.key_points,
                )

                new_full_text = ""
                gen = await synthesize(
                    request.query, intent, evidence_table, patched_plan, groq, stream=True,
                    entity_profile=entity_profile,
                    decomposition=decomposition,
                )
                async for token in gen:
                    new_full_text += token
                    yield _sse("synthesis_token", {"token": token, "is_regeneration": True})

                full_text = ensure_table_in_synthesis(new_full_text)

            # ── Post-synthesis: Entity verification ──────
            if settings.ENTITY_VERIFY_POST_SYNTHESIS and entity_profile and entity_profile.requires_entity_grounding:
                try:
                    entity_verification = await verify_entity_consistency(
                        request.query, full_text, entity_profile, groq
                    )
                    if not entity_verification.entity_correct:
                        logger.warning(
                            "Entity verification failed: substituted=%s confidence=%.2f issues=%s",
                            entity_verification.substituted_entity,
                            entity_verification.confidence,
                            entity_verification.issues,
                        )
                        yield _sse("entity_warning", {
                            "entity_correct": False,
                            "substituted_entity": entity_verification.substituted_entity,
                            "confidence": entity_verification.confidence,
                            "issues": entity_verification.issues,
                        })
                        if entity_verification.confidence > settings.ENTITY_VERIFY_CONFIDENCE_THRESHOLD:
                            full_text = _build_entity_warning_prefix(
                                entity_verification, entity_profile
                            ) + "\n\n" + full_text
                except Exception as ev_exc:
                    logger.warning("Entity verification skipped: %s", ev_exc)

            # ── Post-synthesis: Coverage verification + gap-fill ────────
            # For compound queries: LLM audits whether each sub-question is
            # FULLY / PARTIAL / MISSING in the synthesized answer.
            #
            # MISSING → targeted Tavily re-search + full mini-synthesis per sub-question.
            # PARTIAL → lightweight supplement search (fewer papers, shorter synthesis).
            #
            # Gap-fill evidence rows are merged into evidence_table BEFORE
            # verify_answer so that [CIT:evidence_id] tags resolve correctly.
            if decomposition is not None:
                try:
                    coverage = await verify_coverage(decomposition, full_text, groq)
                    logger.info(
                        "Coverage check: fully=%d partial=%d missing=%d",
                        len(coverage.fully_covered), len(coverage.partial), len(coverage.missing),
                    )

                    needs_gap_fill = bool(coverage.missing or coverage.partial)

                    if needs_gap_fill:
                        yield _sse("coverage_gaps", {
                            "missing": [
                                {"sub_question": i.sub_question, "reason": i.reason}
                                for i in coverage.missing
                            ],
                            "partial": [
                                {"sub_question": i.sub_question, "reason": i.reason}
                                for i in coverage.partial
                            ],
                            "status": "gap_fill_starting",
                        })

                        gap_result = await _targeted_gap_fill(
                            coverage.missing,
                            coverage.partial,
                            store, bm25, dense, groq, reranker,
                            year_constraint, intent,
                        )

                        # ── Merge gap-fill evidence into the main evidence table ──
                        # This ensures [CIT:evidence_id] tags in supplementary text
                        # are resolvable by verify_answer and render_citations.
                        if gap_result.new_evidence_rows:
                            evidence_table.rows.extend(gap_result.new_evidence_rows)
                            logger.info(
                                "Gap fill: merged %d new evidence rows into evidence_table "
                                "(total now %d)",
                                len(gap_result.new_evidence_rows), len(evidence_table.rows),
                            )

                        # ── Append supplementary text + stream to client ─────────
                        if gap_result.supplementary_text:
                            full_text = full_text + gap_result.supplementary_text
                            for chunk in [
                                gap_result.supplementary_text[i:i + 80]
                                for i in range(0, len(gap_result.supplementary_text), 80)
                            ]:
                                yield _sse("synthesis_token", {"token": chunk, "is_gap_fill": True})

                        # ── Report final gap-fill outcome ────────────────────────
                        yield _sse("coverage_gaps", {
                            "status": "gap_fill_complete",
                            "filled": gap_result.filled,
                            "still_missing": gap_result.still_missing,
                            "new_evidence_rows": len(gap_result.new_evidence_rows),
                        })

                        # ── Append a compact summary notice for any sub-questions
                        #    that are STILL unanswered after all gap-fill attempts ──
                        if gap_result.still_missing:
                            still_notice = (
                                "\n\n---\n## Evidence Gaps\n\n"
                                "The following sub-questions could not be answered even after "
                                "targeted web search. Uploading relevant papers will resolve this:\n\n"
                            )
                            for sq in gap_result.still_missing:
                                still_notice += f"- **{sq}**\n"
                            full_text = full_text + still_notice

                except Exception as cov_exc:
                    logger.warning("Coverage verification/gap-fill skipped: %s", cov_exc)

            # ── Step 4: Verification ──────────────────────
            verification = verify_answer(full_text, evidence_table)
            rendered = render_citations(verification, evidence_table)
            logger.info(
                "Verification complete: citations=%s warnings=%s",
                len(verification.citations),
                verification.warnings,
            )

            # Save to DB
            answer_data = {
                "answer_id": evidence_table.answer_id,
                "query": request.query,
                "intent": intent,
                "markdown_text": verification.cleaned_text,
                "citations": [
                    {"citation_number": c.citation_number, "evidence_id": c.evidence_id}
                    for c in verification.citations
                ],
                "is_abstention": task_plan.should_abstain,
                "uncertainty_flags": verification.warnings,
                "total_sources": evidence_table.total_sources,
                "peer_reviewed_count": evidence_table.peer_reviewed_count,
                "preprint_count": evidence_table.preprint_count,
            }
            await store.save_evidence_table(evidence_table.to_dict())
            await store.save_answer(answer_data)

            await store.insert_message({
                "message_id": uuid.uuid4().hex[:12],
                "conversation_id": request.conversation_id,
                "role": "assistant",
                "content": verification.cleaned_text,
                "answer_id": evidence_table.answer_id,
            })

            yield _sse("answer_complete", {
                **answer_data,
                "citation_cards": rendered.to_dict().get("citation_cards", []),
            })

        except Exception as e:
            logger.exception("Chat pipeline error")
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _merge_sub_question_analyses(
    sub_questions: list[str],
    intent: str,
    groq,
):
    """
    Run query rewriting for each sub-question then merge all query forms
    into a single QueryAnalysis that covers the full compound question.

    This ensures the retrieval stage casts a wide net across all sub-topics
    while still using one unified retrieval pass (more efficient than N passes).
    """
    from backend.retrieval.query_rewriter import rewrite_query, QueryAnalysis, _unique_nonempty
    import numpy as np

    analyses = []
    for sq in sub_questions:
        try:
            a = await rewrite_query(sq, intent, groq)
            analyses.append(a)
        except Exception as e:
            logger.warning("Sub-question rewrite failed for %r: %s", sq, e)

    if not analyses:
        # Fallback: rewrite the full original query
        return await rewrite_query(" ".join(sub_questions), intent, groq)

    # Merge: take the first analysis as base and union all query forms
    base = analyses[0]

    # Combine all dense_queries from all sub-question analyses
    all_dense = _unique_nonempty([a.dense_query for a in analyses] + [a.acronym_expanded for a in analyses])
    all_bm25 = _unique_nonempty([a.bm25_query for a in analyses])
    all_arxiv = _unique_nonempty([q for a in analyses for q in a.arxiv_queries])

    base.dense_queries = _unique_nonempty(all_dense + base.dense_queries)[:8]
    base.bm25_queries = _unique_nonempty(all_bm25 + base.bm25_queries)[:6]
    base.arxiv_queries = all_arxiv[:15]  # up to 15 for compound questions
    base.all_queries = _unique_nonempty([
        *base.arxiv_queries,
        *base.dense_queries,
        base.original_query,
    ])

    # Merge HyDE: average embeddings across sub-questions that produced one
    hyde_embeddings = [a.hyde_embedding for a in analyses if a.hyde_embedding is not None]
    if hyde_embeddings:
        base.hyde_embedding = np.mean(np.stack(hyde_embeddings, axis=0), axis=0)

    # Merge topic terms
    base.topic_terms = _unique_nonempty([t for a in analyses for t in a.topic_terms])

    # Use entity profile from whichever sub-question had one (prefer first)
    if base.entity_profile is None:
        for a in analyses[1:]:
            if a.entity_profile is not None:
                base.entity_profile = a.entity_profile
                break

    logger.info(
        "Merged %d sub-question analyses: dense_queries=%d bm25_queries=%d arxiv_queries=%d",
        len(analyses),
        len(base.dense_queries),
        len(base.bm25_queries),
        len(base.arxiv_queries),
    )
    return base


def _check_entity_coverage(
    candidates: list,
    entity_profile: Optional[QueryEntityProfile],
    reranker,
) -> dict:
    """
    Check whether there are enough entity-consistent candidates to answer confidently.

    Returns a dict with:
    - has_coverage: bool
    - reason: str
    - missing_entity: Optional[str]
    - consistent_count: int
    - total_count: int
    """
    if not entity_profile or not entity_profile.requires_entity_grounding:
        return {"has_coverage": True, "reason": "no_entity_constraint"}

    consistent_count = sum(
        1 for c in candidates
        if reranker._entity_consistency_score(entity_profile, c) >= 0.5
    )

    if consistent_count < settings.MIN_ENTITY_CONSISTENT_CANDIDATES:
        return {
            "has_coverage": False,
            "reason": "entity_not_in_corpus",
            "missing_entity": entity_profile.primary_subject,
            "consistent_count": consistent_count,
            "total_count": len(candidates),
        }

    return {
        "has_coverage": True,
        "consistent_count": consistent_count,
        "total_count": len(candidates),
    }


def _build_corpus_gap_abstention(query: str, missing_entity: Optional[str]) -> str:
    """Build a structured corpus gap abstention message."""
    entity_str = f'**{missing_entity}**' if missing_entity else "the requested entity"
    return f"""## Entity Not Found in Corpus

The indexed corpus does not contain papers about {entity_str}.

**Query:** {query}

### Why This Happened
NexusScholar retrieves answers exclusively from your indexed document corpus. \
The corpus does not appear to contain papers specifically about {entity_str}.

### Suggested Actions
- **Upload relevant papers:** Use the upload feature to add papers about {entity_str}
- **Try a broader query:** Search for the general topic area instead of the specific entity
- **Check spelling:** Verify the entity name is spelled correctly

*NexusScholar abstains rather than substituting information from a similar entity.*
"""


def _build_entity_warning_prefix(entity_verification, entity_profile: QueryEntityProfile) -> str:
    """Build a warning banner for entity mismatch."""
    substituted = entity_verification.substituted_entity or "a different entity"
    primary = entity_profile.primary_subject or "the requested entity"
    return (
        f"> **⚠ Entity mismatch detected:** This answer may contain information about "
        f"**{substituted}** rather than the requested **{primary}**. "
        f"The indexed corpus may not contain papers specifically about **{primary}**. "
        f"Please verify carefully or upload relevant documents."
    )


@dataclass
class _GapFillResult:
    """
    Bundle returned by _targeted_gap_fill.

    supplementary_text : Markdown to append to the main answer.
    new_evidence_rows  : EvidenceRow objects to merge into the main
                         evidence_table BEFORE verify_answer / render_citations,
                         so that [CIT:evidence_id] tags in supplementary_text
                         resolve correctly.
    filled             : Sub-question strings that received a substantive answer.
    still_missing      : Sub-question strings that remain unanswered despite search.
    """
    supplementary_text: str = ""
    new_evidence_rows: list = dc_field(default_factory=list)
    filled: list = dc_field(default_factory=list)
    still_missing: list = dc_field(default_factory=list)


async def _gap_fill_one(
    sub_q: str,
    store,
    bm25,
    dense,
    groq,
    reranker,
    year_constraint: "YearConstraint",
    intent: str,
    paper_limit: int,
    rerank_top_k: int,
    max_tokens: int,
) -> tuple[str, list]:
    """
    Full pipeline for a single sub-question:
      Tavily search → ingest → hybrid recall → rerank → build evidence → synthesize.

    Returns (answer_markdown, list_of_EvidenceRow).
    Returns ("", []) when no useful evidence was found.
    """
    from backend.retrieval.query_rewriter import rewrite_query
    from backend.generation.evidence_builder import build_evidence_table as _build_et

    # ── 1. Targeted Tavily search ────────────────────────────────────────
    ingested_ids: set[str] = set()
    try:
        papers = await search_tavily(sub_q, max_results=settings.TAVILY_MAX_RESULTS)
    except Exception as exc:
        logger.warning("Gap fill Tavily search failed for %r: %s", sub_q, exc)
        papers = []

    for paper in papers[:paper_limit]:
        try:
            existing = await store.search_papers_by_title(paper.title, limit=1)
            if existing:
                ingested_ids.add(existing[0]["paper_id"])
                continue
            result = await ingest_text_document(
                title=paper.title,
                content=paper.content or paper.snippet,
                store=store,
                bm25=bm25,
                dense=dense,
                metadata_override=paper.to_metadata_override(),
                skip_index_rebuild=True,
            )
            if result.get("paper_id"):
                ingested_ids.add(result["paper_id"])
                # Persist metadata so recency/citation signals work downstream
                if paper.year or paper.venue or paper.citation_count:
                    meta = {k: v for k, v in {
                        "year": paper.year, "venue": paper.venue,
                        "citation_count": paper.citation_count,
                        "is_peer_reviewed": paper.is_peer_reviewed,
                    }.items() if v}
                    try:
                        await store.update_paper_metadata(result["paper_id"], meta)
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("Gap fill ingest failed for %r: %s", getattr(paper, "title", "?"), exc)

    if ingested_ids:
        await rebuild_indexes(bm25, dense, store)
        logger.info("Gap fill: indexes rebuilt (+%d papers) for %r", len(ingested_ids), sub_q)

    # ── 2. Targeted hybrid recall ────────────────────────────────────────
    try:
        sub_analysis = await rewrite_query(sub_q, intent, groq)
    except Exception as exc:
        logger.warning("Gap fill query rewrite failed for %r: %s", sub_q, exc)
        return "", []

    sub_candidates = hybrid_recall(sub_analysis, bm25, dense)

    # Freshly ingested papers get priority; fall back to rest of corpus
    if ingested_ids:
        priority = [c for c in sub_candidates if c.chunk.get("paper_id") in ingested_ids]
        other = [c for c in sub_candidates if c.chunk.get("paper_id") not in ingested_ids]
        sub_candidates = priority + other[:15]

    # Resolve paper metadata in a single batch query
    _sub_ids = [c.chunk["paper_id"] for c in sub_candidates]
    _sub_map = await store.get_papers_batch(_sub_ids)
    for c in sub_candidates:
        c.paper = _sub_map.get(c.chunk["paper_id"]) or {}

    sub_candidates = _filter_candidates_by_year(sub_candidates, year_constraint)
    sub_reranked = reranker.rerank(sub_q, sub_candidates, top_k=rerank_top_k, intent=intent)

    if not sub_reranked:
        logger.info("Gap fill: no candidates after reranking for %r", sub_q)
        return "", []

    # ── 3. Build evidence table ──────────────────────────────────────────
    sub_evidence = _build_et(sub_q, intent, sub_reranked)

    # ── 4. Focused synthesis ─────────────────────────────────────────────
    evidence_json = json.dumps(sub_evidence.to_llm_context(), indent=2)
    try:
        sub_text = await groq.complete_primary(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research synthesizer answering a specific sub-question. "
                        "Use ONLY the evidence JSON provided. Rules:\n"
                        "1. Cite EVERY factual claim with [CIT:evidence_id] using the exact "
                        "   evidence_id values from the JSON.\n"
                        "2. Report specific numbers, methods, and results from the evidence.\n"
                        "3. If evidence is genuinely insufficient, state that clearly — do NOT "
                        "   hallucinate or use parametric knowledge.\n"
                        "4. Write in Markdown. Do NOT repeat content from the main answer — "
                        "   this is a supplementary section that fills a gap.\n"
                        "5. Aim for 200-400 words (or 400-600 for complex sub-questions)."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Sub-question to answer:\n{sub_q}\n\n"
                        f"Evidence (JSON):\n{evidence_json}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=max_tokens,
            stream=False,
        )
    except Exception as exc:
        logger.warning("Gap fill synthesis failed for %r: %s", sub_q, exc)
        return "", []

    # Treat short abstentions as failure so still_missing is populated correctly
    sub_text = (sub_text or "").strip()
    is_abstention = (
        len(sub_text) < 250
        and any(kw in sub_text.lower() for kw in ("insufficient", "not found", "no evidence", "abstain"))
    )
    if is_abstention:
        logger.info("Gap fill: synthesis abstained for %r", sub_q)
        return "", []

    return sub_text, sub_evidence.rows


async def _targeted_gap_fill(
    missing_items: list,
    partial_items: list,
    store,
    bm25,
    dense,
    groq,
    reranker,
    year_constraint: "YearConstraint",
    intent: str,
) -> "_GapFillResult":
    """
    Comprehensive gap-fill for MISSING and PARTIAL sub-questions.

    MISSING items receive an aggressive gap-fill:
      - Up to 6 papers ingested, top_k=8, 800 output tokens
      - Each gets a full supplementary section or an explicit "not found" notice

    PARTIAL items receive a lightweight supplement:
      - Up to 3 papers ingested, top_k=5, 400 output tokens
      - Skipped silently if no improvement found (partial evidence already exists)

    The returned _GapFillResult.new_evidence_rows MUST be merged into the
    caller's evidence_table before calling verify_answer / render_citations,
    so that [CIT:evidence_id] tags in the supplementary text can be resolved.
    """
    all_parts: list[str] = []
    all_rows: list = []
    filled: list[str] = []
    # Pre-populate still_missing with items that exceed the attempt cap so they
    # appear in the final "Evidence Gaps" notice even if never attempted.
    still_missing: list[str] = [item.sub_question for item in missing_items[3:]]

    # ── MISSING: full gap-fill (up to 3 items) ────────────────────────
    for item in missing_items[:3]:
        sub_q = item.sub_question
        logger.info("Gap fill [MISSING]: %r", sub_q)
        try:
            text, rows = await _gap_fill_one(
                sub_q, store, bm25, dense, groq, reranker,
                year_constraint, intent,
                paper_limit=6, rerank_top_k=8, max_tokens=800,
            )
        except Exception as exc:
            logger.warning("Gap fill [MISSING] exception for %r: %s", sub_q, exc)
            text, rows = "", []

        if text:
            all_parts.append(f"\n\n---\n## {sub_q}\n\n{text}")
            all_rows.extend(rows)
            filled.append(sub_q)
            logger.info("Gap fill [MISSING] ✓ %r — %d chars, %d rows", sub_q, len(text), len(rows))
        else:
            still_missing.append(sub_q)
            all_parts.append(
                f"\n\n---\n## {sub_q}\n\n"
                "*After a targeted web search no sufficient evidence was found to answer "
                "this sub-question. Consider uploading relevant papers directly.*"
            )
            logger.info("Gap fill [MISSING] ✗ %r — no answer found", sub_q)

    # ── PARTIAL: lightweight supplement ───────────────────────────────
    for item in partial_items[:2]:
        sub_q = item.sub_question
        logger.info("Gap fill [PARTIAL]: %r", sub_q)
        try:
            text, rows = await _gap_fill_one(
                sub_q, store, bm25, dense, groq, reranker,
                year_constraint, intent,
                paper_limit=3, rerank_top_k=5, max_tokens=400,
            )
        except Exception as exc:
            logger.warning("Gap fill [PARTIAL] exception for %r: %s", sub_q, exc)
            text, rows = "", []

        if text:
            all_parts.append(f"\n\n---\n## Additional Evidence: {sub_q}\n\n{text}")
            all_rows.extend(rows)
            filled.append(sub_q)
            logger.info("Gap fill [PARTIAL] ✓ %r — %d chars", sub_q, len(text))
        # Silent skip if no improvement — partial evidence already exists in main answer

    return _GapFillResult(
        supplementary_text="".join(all_parts),
        new_evidence_rows=all_rows,
        filled=filled,
        still_missing=still_missing,
    )


def _adaptive_exa_query_count(intent: str, decomposition=None) -> int:
    """
    Return how many arxiv_queries to fire at Exa, scaled by question complexity.

    - Compound questions: all 15 available — each sub-question contributes its own
      set of queries, merged together, so we want maximum coverage.
    - Wide-scope intents (survey, trend, benchmark): 12 queries.
    - Focused intents (definition, paper_lookup, author_search): 8 queries.
    - Default: 10 queries.
    """
    if decomposition and decomposition.is_compound:
        return 15  # use all available queries from merged sub-question analyses
    if intent in ("literature_survey", "benchmark_comparison", "trend_analysis", "contradiction_check"):
        return 12
    if intent in ("paper_lookup", "author_search", "definition"):
        return 8
    return 10


def _adaptive_ingest_limit(intent: str, decomposition=None) -> int:
    """
    Return how many Exa results to actually ingest (after ranking).
    Scaled up for complex / compound queries.
    """
    base = settings.EXA_NUM_RESULTS  # now 40
    if decomposition and decomposition.is_compound:
        return min(base * 2, 80)
    if intent in ("literature_survey", "benchmark_comparison", "trend_analysis"):
        return int(base * 1.5)
    return base


async def _hydrate_from_exa(
    query_analysis, store, bm25, dense,
    year_constraint: YearConstraint,
    intent: str = "general",
    entity_profile=None,
    decomposition=None,
) -> set[str]:
    """
    Hydrate corpus from Exa — the primary external retrieval source (§7.4.1).

    Year constraints, entity grounding, category filtering, domain filtering,
    highlights and autoprompt are all pushed into the Exa API call for
    server-side precision before local re-ranking.

    Query count and ingest limit are adaptive: compound questions and wide-scope
    intents get more queries to ensure full topic coverage.
    """
    _log_search_query_box(query_analysis.arxiv_queries, source_name="Exa")
    selected_papers = []

    query_count = _adaptive_exa_query_count(intent, decomposition)
    ingest_limit = _adaptive_ingest_limit(intent, decomposition)
    logger.info(
        "Exa hydration: query_count=%d ingest_limit=%d intent=%s compound=%s",
        query_count, ingest_limit, intent,
        decomposition.is_compound if decomposition else False,
    )

    for search_query in query_analysis.arxiv_queries[:query_count]:
        try:
            papers = await search_exa(
                search_query,
                num_results=settings.EXA_NUM_RESULTS,
                intent=intent,
                entity_profile=entity_profile,
                year_min=year_constraint.year_min,
                year_max=year_constraint.year_max,
            )
            selected_papers.extend(papers)
        except Exception as exc:
            logger.warning("Exa search failed for %r: %s", search_query, exc)

    # Rank and filter using same pipeline as Tavily (term overlap + recency + authority)
    ranked_papers = _rank_exa_papers(query_analysis, selected_papers, intent=intent)
    ranked_papers = _filter_papers_by_year(ranked_papers, year_constraint)
    allowed_paper_ids: set[str] = set()
    ingested = 0

    for paper in ranked_papers[:ingest_limit]:
        existing = await store.search_papers_by_title(paper.title, limit=1)
        if existing:
            logger.info("Using existing Exa paper for query scope: %s", paper.title)
            allowed_paper_ids.add(existing[0]["paper_id"])
            continue

        try:
            if paper.pdf_url:
                pdf_path = await _download_pdf(paper.pdf_url, paper.result_id)
                result = await ingest_pdf_file(
                    str(pdf_path),
                    store,
                    bm25,
                    dense,
                    metadata_override=paper.to_metadata_override(),
                    skip_index_rebuild=True,
                )
            else:
                result = await ingest_text_document(
                    title=paper.title,
                    content=paper.content or paper.snippet,
                    store=store,
                    bm25=bm25,
                    dense=dense,
                    metadata_override=paper.to_metadata_override(),
                    skip_index_rebuild=True,
                )

            if result.get("paper_id"):
                metadata_to_persist = {}
                if paper.year:
                    metadata_to_persist["year"] = paper.year
                if paper.venue:
                    metadata_to_persist["venue"] = paper.venue
                if paper.citation_count:
                    metadata_to_persist["citation_count"] = paper.citation_count
                if paper.is_peer_reviewed:
                    metadata_to_persist["is_peer_reviewed"] = paper.is_peer_reviewed
                if metadata_to_persist:
                    try:
                        await store.update_paper_metadata(result["paper_id"], metadata_to_persist)
                        logger.info(
                            "Persisted Exa metadata for %s: %s",
                            result["paper_id"], metadata_to_persist,
                        )
                    except Exception as meta_exc:
                        logger.warning("Failed to persist Exa metadata: %s", meta_exc)

            logger.info("Ingested Exa result: %s | %s", paper.result_id, paper.title)
            allowed_paper_ids.add(result["paper_id"])
            ingested += 1
        except Exception as exc:
            logger.warning("Exa ingest failed for %s: %s", paper.source_url, exc)

    if ingested > 0:
        await rebuild_indexes(bm25, dense, store)
        logger.info("Indexes rebuilt after ingesting %s Exa papers", ingested)

    return allowed_paper_ids


def _rank_exa_papers(query_analysis, papers, intent: str = "general") -> list:
    """Rank Exa results using the same multi-signal scoring as Tavily."""
    unique = {}
    query_terms = set(query_analysis.topic_terms) | set(query_analysis.original_query.lower().split())
    arxiv_terms = set()
    for aq in query_analysis.arxiv_queries:
        arxiv_terms.update(aq.lower().split())
    all_signal_terms = query_terms | arxiv_terms
    stop = {
        "what", "are", "the", "recent", "advances", "for", "and", "with",
        "about", "using", "how", "does", "a", "an", "of", "in", "to", "on",
    }
    all_signal_terms -= stop
    seen_titles: set[str] = set()

    for paper in papers:
        if not paper or paper.source_url in unique:
            continue
        norm_title = re.sub(r"[^a-z0-9]", "", (paper.title or "").lower())[:60]
        if norm_title in seen_titles:
            continue
        seen_titles.add(norm_title)

        title_lower = (paper.title or "").lower()
        combined = f"{paper.title} {paper.snippet} {paper.content[:5000]}".lower()

        title_overlap = sum(1.5 for term in all_signal_terms if term and term in title_lower)
        body_overlap = sum(0.5 for term in all_signal_terms if term and term in combined)
        term_score = title_overlap + body_overlap

        exa_score = paper.score * 5.0

        content_len = len(paper.content or "")
        if content_len > 5000:
            richness = 3.0
        elif content_len > 2000:
            richness = 2.0
        elif content_len > 500:
            richness = 1.0
        else:
            richness = 0.0

        authority = 0.0
        url_lower = paper.source_url.lower()
        if "nature.com" in url_lower or "science.org" in url_lower:
            authority = 3.0
        elif "aclanthology.org" in url_lower or "proceedings.mlr.press" in url_lower:
            authority = 2.5
        elif "openreview.net" in url_lower:
            authority = 2.0
        elif "arxiv.org" in url_lower:
            authority = 1.5
        elif "semanticscholar.org" in url_lower:
            authority = 1.0

        recency_score = 0.0
        paper_year = getattr(paper, "year", None)
        if paper_year:
            age = datetime.now().year - paper_year
            if intent in ("benchmark_comparison", "trend_analysis"):
                if age == 0:
                    recency_score = 5.0
                elif age == 1:
                    recency_score = 3.5
                elif age == 2:
                    recency_score = 1.5
                elif age >= 4:
                    recency_score = -3.0
            else:
                if age <= 1:
                    recency_score = 1.5
                elif age <= 2:
                    recency_score = 0.8
                elif age <= 3:
                    recency_score = 0.3

        score = term_score + exa_score + richness + authority + recency_score
        unique[paper.source_url] = (score, paper)

    ranked = [p for _, p in sorted(unique.values(), key=lambda x: x[0], reverse=True)]
    for idx, paper in enumerate(ranked[:8], start=1):
        logger.info(
            "Ranked Exa paper %s (score=%.2f year=%s): %s | %s",
            idx, unique[paper.source_url][0], getattr(paper, "year", None),
            paper.source_url, paper.title,
        )
    return ranked


async def _hydrate_from_tavily(
    query_analysis, store, bm25, dense,
    year_constraint: YearConstraint,
    intent: str = "general",
    decomposition=None,
) -> set[str]:
    _log_search_query_box(query_analysis.arxiv_queries, source_name="Tavily")
    selected_papers = []

    # Use "news" topic for benchmark/trend queries to surface recent content
    tavily_topic = "news" if intent in ("benchmark_comparison", "trend_analysis") else "general"

    # Adaptive query count: Tavily is fallback so keep slightly below Exa count
    tavily_query_count = max(4, _adaptive_exa_query_count(intent, decomposition) - 2)

    for search_query in query_analysis.arxiv_queries[:tavily_query_count]:
        try:
            papers = await search_tavily(
                search_query,
                max_results=settings.TAVILY_MAX_RESULTS,
                topic=tavily_topic,
            )
            selected_papers.extend(papers)
        except Exception as exc:
            logger.warning("Tavily search failed for %r: %s", search_query, exc)

    ranked_papers = _rank_tavily_papers(query_analysis, selected_papers, intent=intent)
    ranked_papers = _filter_papers_by_year(ranked_papers, year_constraint)
    allowed_paper_ids: set[str] = set()
    ingested = 0

    tavily_ingest_limit = _adaptive_ingest_limit(intent, decomposition)
    for paper in ranked_papers[:tavily_ingest_limit]:
        existing = await store.search_papers_by_title(paper.title, limit=1)
        if existing:
            logger.info("Using existing Tavily paper for query scope: %s", paper.title)
            allowed_paper_ids.add(existing[0]["paper_id"])
            continue

        try:
            if paper.pdf_url:
                pdf_path = await _download_pdf(paper.pdf_url, paper.result_id)
                result = await ingest_pdf_file(
                    str(pdf_path),
                    store,
                    bm25,
                    dense,
                    metadata_override=paper.to_metadata_override(),
                    skip_index_rebuild=True,
                )
            else:
                result = await ingest_text_document(
                    title=paper.title,
                    content=paper.content or paper.snippet,
                    store=store,
                    bm25=bm25,
                    dense=dense,
                    metadata_override=paper.to_metadata_override(),
                    skip_index_rebuild=True,
                )

            # FIX: Persist Tavily-extracted metadata that the PDF parser may have missed.
            # Year is the most critical field for recency filtering.
            if result.get("paper_id"):
                metadata_to_persist = {}
                if paper.year:
                    metadata_to_persist["year"] = paper.year
                if paper.venue:
                    metadata_to_persist["venue"] = paper.venue
                if paper.citation_count:
                    metadata_to_persist["citation_count"] = paper.citation_count
                if paper.is_peer_reviewed:
                    metadata_to_persist["is_peer_reviewed"] = paper.is_peer_reviewed
                if metadata_to_persist:
                    try:
                        await store.update_paper_metadata(result["paper_id"], metadata_to_persist)
                        logger.info(
                            "Persisted Tavily metadata for %s: %s",
                            result["paper_id"], metadata_to_persist
                        )
                    except Exception as meta_exc:
                        logger.warning("Failed to persist Tavily metadata: %s", meta_exc)

            logger.info("Ingested Tavily result: %s | %s", paper.result_id, paper.title)
            allowed_paper_ids.add(result["paper_id"])
            ingested += 1
        except Exception as exc:
            logger.warning("Tavily ingest failed for %s: %s", paper.source_url, exc)

    if ingested > 0:
        await rebuild_indexes(bm25, dense, store)
        logger.info("Indexes rebuilt after ingesting %s Tavily papers", ingested)

    return allowed_paper_ids


async def _hydrate_from_semantic_scholar(
    query_analysis, store, bm25, dense, year_constraint: YearConstraint,
    intent: str = "general",
    decomposition=None,
) -> set[str]:
    """Fetch structured metadata from Semantic Scholar and ingest as text documents."""
    allowed: set[str] = set()

    # Adaptive S2 query count — compound/wide-scope queries get more
    if decomposition and decomposition.is_compound:
        s2_query_count = 6
    elif intent in ("literature_survey", "benchmark_comparison", "trend_analysis"):
        s2_query_count = 5
    else:
        s2_query_count = 4

    # If entity profile has a primary subject, prioritize that as the search term
    entity_profile = query_analysis.entity_profile
    if entity_profile and entity_profile.primary_subject and entity_profile.requires_entity_grounding:
        primary_queries = [
            entity_profile.primary_subject + " " + query_analysis.original_query,
            entity_profile.primary_subject,
        ]
        fallback_queries = query_analysis.arxiv_queries[:s2_query_count - 2]
        search_queries = (primary_queries + fallback_queries)[:s2_query_count]
        logger.info("S2 entity-targeted search: primary=%r", entity_profile.primary_subject)
    else:
        search_queries = query_analysis.arxiv_queries[:s2_query_count] or [query_analysis.original_query]

    for sq in search_queries:
        try:
            papers = await search_semantic_scholar(sq, limit=6)
        except Exception as exc:
            logger.warning("Semantic Scholar search failed for %r: %s", sq, exc)
            continue

        for paper in papers:
            if not paper.get("abstract") or not paper.get("title"):
                continue
            year = paper.get("year")
            if year_constraint.year_min and year and year < year_constraint.year_min:
                continue
            if year_constraint.year_max and year and year > year_constraint.year_max:
                continue

            # Skip if already ingested
            existing = await store.search_papers_by_title(paper["title"], limit=1)
            if existing:
                allowed.add(existing[0]["paper_id"])
                continue

            try:
                result = await ingest_text_document(
                    title=paper["title"],
                    content=paper["abstract"],
                    store=store,
                    bm25=bm25,
                    dense=dense,
                    metadata_override={
                        "authors": paper.get("authors", []),
                        "year": paper.get("year"),
                        "venue": paper.get("venue"),
                        "doi": paper.get("doi"),
                        "arxiv_id": paper.get("arxiv_id"),
                        "abstract": paper.get("abstract"),
                        "citation_count": paper.get("citation_count", 0),
                        "is_peer_reviewed": paper.get("is_peer_reviewed", False),
                        "pdf_url": paper.get("pdf_url", ""),
                    },
                    skip_index_rebuild=True,
                )
                allowed.add(result["paper_id"])

                # FIX: S2 metadata is authoritative — always persist it after ingest.
                # The PDF parser may not have extracted these fields reliably.
                if result.get("paper_id"):
                    s2_metadata = {
                        k: v for k, v in {
                            "year": paper.get("year"),
                            "venue": paper.get("venue"),
                            "citation_count": paper.get("citation_count", 0),
                            "is_peer_reviewed": paper.get("is_peer_reviewed", False),
                            "doi": paper.get("doi"),
                        }.items() if v not in (None, "", 0, False)
                    }
                    if s2_metadata:
                        try:
                            await store.update_paper_metadata(result["paper_id"], s2_metadata)
                            logger.info(
                                "Persisted S2 metadata for %s: %s",
                                result["paper_id"], s2_metadata
                            )
                        except Exception as meta_exc:
                            logger.warning("Failed to persist S2 metadata: %s", meta_exc)
            except Exception as exc:
                logger.warning("S2 ingest failed for %s: %s", paper["title"], exc)

    if allowed:
        await rebuild_indexes(bm25, dense, store)
    return allowed


def _rank_tavily_papers(query_analysis, papers, intent: str = "general") -> list:
    """
    Enterprise-grade Tavily result ranking using multi-signal scoring:
    - Semantic term overlap (query terms + topic terms + arxiv terms)
    - Tavily relevance score
    - Content richness (length, structure)
    - Source authority (venue prestige)
    - Duplicate suppression by fuzzy title matching
    """
    unique = {}
    query_terms = set(query_analysis.topic_terms) | set(query_analysis.original_query.lower().split())
    arxiv_terms = set()
    for aq in query_analysis.arxiv_queries:
        arxiv_terms.update(aq.lower().split())
    all_signal_terms = query_terms | arxiv_terms
    # Remove stopwords from signal
    stop = {"what", "are", "the", "recent", "advances", "for", "and", "with", "about", "using", "how", "does", "a", "an", "of", "in", "to", "on"}
    all_signal_terms -= stop

    seen_titles: set[str] = set()

    for paper in papers:
        if not paper or paper.source_url in unique:
            continue
        # Fuzzy duplicate suppression: normalize title
        norm_title = re.sub(r"[^a-z0-9]", "", (paper.title or "").lower())[:60]
        if norm_title in seen_titles:
            continue
        seen_titles.add(norm_title)

        title_lower = (paper.title or "").lower()
        combined = f"{paper.title} {paper.snippet} {paper.content[:5000]}".lower()

        # 1. Deep term overlap — weighted by where the term appears
        title_overlap = sum(1.5 for term in all_signal_terms if term and term in title_lower)
        body_overlap = sum(0.5 for term in all_signal_terms if term and term in combined)
        term_score = title_overlap + body_overlap

        # 2. Tavily relevance score (0-1 range, weight it up)
        tavily_score = paper.score * 5.0

        # 3. Content richness — prefer papers with substantial content
        content_len = len(paper.content or "")
        richness = 0.0
        if content_len > 5000:
            richness = 3.0
        elif content_len > 2000:
            richness = 2.0
        elif content_len > 500:
            richness = 1.0

        # 4. Source authority boost
        authority = 0.0
        url_lower = paper.source_url.lower()
        if "nature.com" in url_lower or "science.org" in url_lower:
            authority = 3.0
        elif "aclanthology.org" in url_lower or "proceedings.mlr.press" in url_lower:
            authority = 2.5
        elif "openreview.net" in url_lower:
            authority = 2.0
        elif "arxiv.org" in url_lower:
            authority = 1.5
        elif "semanticscholar.org" in url_lower:
            authority = 1.0

        # 5. Evidence signals in content
        evidence_bonus = 0.0
        if re.search(r"\b\d+(?:\.\d+)?%", combined):
            evidence_bonus += 1.0
        if re.search(r"\b(outperform|state-of-the-art|sota|achiev|improv)\b", combined):
            evidence_bonus += 1.0
        if re.search(r"\b(table|figure|experiment|evaluation|results|benchmark)\b", combined):
            evidence_bonus += 0.5

        # 6. Recency signal — weighted heavily for benchmark/trend queries
        recency_score = 0.0
        current_year = datetime.now().year
        paper_year = getattr(paper, "year", None)
        if paper_year:
            age = current_year - paper_year
            if intent in ("benchmark_comparison", "trend_analysis"):
                if age == 0:
                    recency_score = 5.0
                elif age == 1:
                    recency_score = 3.5
                elif age == 2:
                    recency_score = 1.5
                elif age == 3:
                    recency_score = 0.0
                elif age >= 4:
                    recency_score = -3.0   # Actively penalise — benchmarks from 4+ years ago are superseded
            else:
                # Mild preference for recency in all other intents
                if age <= 1:
                    recency_score = 1.5
                elif age <= 2:
                    recency_score = 0.8
                elif age <= 3:
                    recency_score = 0.3

        score = term_score + tavily_score + richness + authority + evidence_bonus + recency_score
        unique[paper.source_url] = (score, paper)

    ranked = [paper for _, paper in sorted(unique.values(), key=lambda item: item[0], reverse=True)]
    for idx, paper in enumerate(ranked[:8], start=1):
        logger.info(
            "Ranked Tavily paper %s (score=%.2f year=%s): %s | %s",
            idx, unique[paper.source_url][0], getattr(paper, "year", None),
            paper.source_url, paper.title
        )
    return ranked


def _filter_papers_by_year(papers, year_constraint: YearConstraint):
    if year_constraint.year_min is None and year_constraint.year_max is None:
        return papers

    filtered = []
    dropped = 0
    for paper in papers:
        year = getattr(paper, "year", None)
        if _year_matches(year, year_constraint):
            filtered.append(paper)
        else:
            dropped += 1
    logger.info(
        "Year filter applied to Tavily papers: source=%s year_min=%s year_max=%s kept=%s dropped=%s",
        year_constraint.source,
        year_constraint.year_min,
        year_constraint.year_max,
        len(filtered),
        dropped,
    )
    return filtered


def _limit_chunks_per_paper(candidates, max_chunks_per_paper: int = 2):
    per_paper: dict[str, int] = {}
    filtered = []
    for candidate in candidates:
        paper_id = candidate.chunk.get("paper_id")
        count = per_paper.get(paper_id, 0)
        if count >= max_chunks_per_paper:
            continue
        per_paper[paper_id] = count + 1
        filtered.append(candidate)
    return filtered


def _filter_candidates_by_year(candidates, year_constraint: YearConstraint):
    if year_constraint.year_min is None and year_constraint.year_max is None:
        return candidates

    filtered = []
    dropped = 0
    for candidate in candidates:
        year = (candidate.paper or {}).get("year")
        if _year_matches(year, year_constraint):
            filtered.append(candidate)
        else:
            dropped += 1

    logger.info(
        "Year filter applied to candidates: source=%s year_min=%s year_max=%s kept=%s dropped=%s",
        year_constraint.source,
        year_constraint.year_min,
        year_constraint.year_max,
        len(filtered),
        dropped,
    )
    return filtered


async def _augment_with_selected_paper_chunks(query: str, query_analysis, candidates, store, per_paper_limit: int = 6):
    if not candidates:
        return []

    selected: list = list(candidates)
    seen_chunk_ids = {candidate.chunk.get("chunk_id") for candidate in candidates}
    paper_order: list[str] = []
    for candidate in candidates:
        paper_id = candidate.chunk.get("paper_id")
        if paper_id and paper_id not in paper_order:
            paper_order.append(paper_id)

    for paper_id in paper_order[:6]:
        paper = next((c.paper for c in candidates if c.chunk.get("paper_id") == paper_id), {}) or {}
        extra_chunks = []
        for granularity in ("document", "section", "claim", "passage"):
            extra_chunks.extend(await store.get_chunks_by_paper(paper_id, granularity=granularity))

        ranked_chunks = sorted(
            extra_chunks,
            key=lambda chunk: _paper_chunk_priority(query, query_analysis, paper, chunk),
            reverse=True,
        )

        added = 0
        for chunk in ranked_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id in seen_chunk_ids:
                continue
            selected.append(
                RetrievalCandidate(
                    chunk=chunk,
                    paper=paper,
                    rrf_score=max(0.01, _paper_chunk_priority(query, query_analysis, paper, chunk)),
                )
            )
            seen_chunk_ids.add(chunk_id)
            added += 1
            if added >= per_paper_limit:
                break

    return selected


def _filter_candidates_by_paper(query_analysis, candidates, max_papers: int = 8):
    if not candidates:
        return []

    paper_scores: dict[str, float] = {}
    paper_meta: dict[str, dict] = {}
    for candidate in candidates:
        paper_id = candidate.chunk.get("paper_id")
        paper = candidate.paper or {}
        paper_meta[paper_id] = paper
        score = _paper_relevance_score(query_analysis, paper, candidate.chunk)
        paper_scores[paper_id] = max(paper_scores.get(paper_id, float("-inf")), score)

    ranked_papers = sorted(paper_scores.items(), key=lambda item: item[1], reverse=True)
    selected_paper_ids = {paper_id for paper_id, _score in ranked_papers[:max_papers]}

    for idx, (paper_id, score) in enumerate(ranked_papers[:max_papers], start=1):
        paper = paper_meta.get(paper_id, {})
        logger.info(
            "Paper-ranked %s: paper=%s score=%.4f title=%s",
            idx,
            paper_id,
            score,
            paper.get("title"),
        )

    return [candidate for candidate in candidates if candidate.chunk.get("paper_id") in selected_paper_ids]


def _paper_relevance_score(query_analysis, paper: dict, chunk: dict) -> float:
    combined = " ".join(
        [
            paper.get("title", ""),
            paper.get("abstract", ""),
            chunk.get("section_tag", ""),
        ]
    ).lower()
    query_terms = _normalized_terms(query_analysis.original_query)
    topic_terms = _normalized_terms(" ".join(query_analysis.topic_terms))
    search_terms = _normalized_terms(" ".join(query_analysis.arxiv_queries))

    overlap = _overlap_ratio(query_terms, combined)
    topic_overlap = _overlap_ratio(topic_terms, combined)
    search_overlap = _overlap_ratio(search_terms, combined)

    score = 0.55 * overlap + 0.3 * topic_overlap + 0.15 * search_overlap

    title = (paper.get("title") or "").lower()
    if any(term in title for term in topic_terms):
        score += 0.15
    if "code generation" in combined and "code" not in query_terms:
        score -= 0.2
    if "image generation" in combined and "image" not in query_terms:
        score -= 0.2
    if "scientific" in query_terms and "scientific" in combined:
        score += 0.1
    if "qa" in query_terms and ("question answering" in combined or "qa" in combined):
        score += 0.1

    return score


def _paper_chunk_priority(query: str, query_analysis, paper: dict, chunk: dict) -> float:
    text = chunk.get("text", "")
    section = (chunk.get("section_tag") or "").lower()
    text_lower = text.lower()
    query_terms = _normalized_terms(query)
    topic_terms = _normalized_terms(" ".join(query_analysis.topic_terms))

    overlap = _overlap_ratio(query_terms, text_lower)
    topic_overlap = _overlap_ratio(topic_terms, text_lower)
    score = 0.5 * overlap + 0.2 * topic_overlap

    section_boost = {
        "abstract": 0.28,
        "results": 0.24,
        "discussion": 0.18,
        "conclusion": 0.18,
        "methods": 0.14,
        "introduction": 0.08,
        "web_content": 0.08,
    }
    score += section_boost.get(section, 0.03)

    if chunk.get("granularity") == "claim":
        score += 0.12
    elif chunk.get("granularity") == "document":
        score += 0.10
    elif chunk.get("granularity") == "section":
        score += 0.08

    if re.search(r"\b(outperform|improv|achiev|state-of-the-art|sota|accuracy|f1|bleu|mmlu|glue)\b", text_lower):
        score += 0.12
    if re.search(r"\b\d+(?:\.\d+)?%|\b\d+\.\d+\b", text_lower):
        score += 0.08
    if paper.get("title"):
        score += 0.08 * _overlap_ratio(query_terms, paper["title"].lower())

    length = len(text.split())
    if 40 <= length <= 260:
        score += 0.06
    elif length < 20:
        score -= 0.06

    return score


def _normalized_terms(text: str):
    stop = {"what", "are", "the", "recent", "for", "and", "with", "about", "a", "an", "of", "in", "to", "is", "how", "does", "do"}
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", text.lower())
        if token not in stop
    }


def _overlap_ratio(terms: set[str], combined_text: str) -> float:
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if term in combined_text)
    return hits / len(terms)


_HISTORICAL_PATTERN = re.compile(
    r"\b("
    r"original|seminal|foundational|classic|landmark|pioneering|"
    r"first paper|first proposed|invented|introduced in|"
    r"paxos|raft consensus|lamport|leslie lamport|"
    r"black.scholes|merton|fisher|turing|von neumann|"
    r"pagerank|backpropagation|perceptron|hidden markov|"
    r"19\d{2}"          # any explicit 19xx year in the query
    r")\b",
    re.IGNORECASE,
)


def _is_historical_query(query: str) -> bool:
    """Return True when the query is clearly about pre-2000 / foundational work."""
    return bool(_HISTORICAL_PATTERN.search(query or ""))


def _year_min_for_filter(recency_filter: str):
    current_year = datetime.now().year
    normalized = (recency_filter or "any").strip().lower()
    if normalized == "1y":
        return current_year - 1
    if normalized == "3y":
        return current_year - 3
    return None


def _resolve_year_constraint(query: str, recency_filter: str) -> YearConstraint:
    """
    Resolve year constraint from explicit filter, query year mentions,
    AND implicit recency language ("latest", "recent", "newest", "current SOTA").
    """
    explicit = _extract_year_constraint_from_query(query)
    if explicit.year_min is not None or explicit.year_max is not None:
        return explicit

    derived_min = _year_min_for_filter(recency_filter)
    if derived_min is not None:
        return YearConstraint(year_min=derived_min, source=f"recency:{recency_filter}")

    # NEW: Detect implicit recency language in query even without explicit year numbers
    implicit = _extract_implicit_recency_constraint(query)
    if implicit.year_min is not None or implicit.year_max is not None:
        return implicit

    return YearConstraint()


def _extract_implicit_recency_constraint(query: str) -> YearConstraint:
    """
    Detect recency intent from natural language without explicit year numbers.
    Examples: "latest models", "recent advances", "newest benchmarks", "state of the art"
    """
    current_year = datetime.now().year
    text = (query or "").lower()

    # "latest", "newest", "most recent", "current" → last 1 year
    strong_recency = re.compile(
        r"\b(latest|newest|most recent|just released|just published|"
        r"brand new|cutting.edge|state.of.the.art|current best|"
        r"top performing|leading|best performing)\b"
    )
    if strong_recency.search(text):
        logger.info(
            "Implicit strong recency constraint detected in query: year_min=%s",
            current_year - 1
        )
        return YearConstraint(year_min=current_year - 1, source="query:implicit_strong_recency")

    # "recent", "new", "modern", "contemporary" → last 2 years
    moderate_recency = re.compile(
        r"\b(recent|recently|new|newly|modern|contemporary|up.to.date|"
        r"up to date|advances in|progress in|developments in)\b"
    )
    if moderate_recency.search(text):
        logger.info(
            "Implicit moderate recency constraint detected in query: year_min=%s",
            current_year - 2
        )
        return YearConstraint(year_min=current_year - 2, source="query:implicit_moderate_recency")

    # "this year" → current year only
    if "this year" in text:
        return YearConstraint(year_min=current_year, year_max=current_year, source="query:this_year")

    # "last year" → previous year only
    if "last year" in text:
        return YearConstraint(
            year_min=current_year - 1,
            year_max=current_year - 1,
            source="query:last_year"
        )

    return YearConstraint()


def _extract_year_constraint_from_query(query: str) -> YearConstraint:
    text = (query or "").lower()
    years = [int(value) for value in re.findall(r"\b(19\d{2}|20[0-3]\d)\b", text)]
    if not years:
        return YearConstraint()

    between_match = re.search(r"\bbetween\s+(19\d{2}|20[0-3]\d)\s+and\s+(19\d{2}|20[0-3]\d)\b", text)
    if between_match:
        year_a = int(between_match.group(1))
        year_b = int(between_match.group(2))
        return YearConstraint(year_min=min(year_a, year_b), year_max=max(year_a, year_b), source="query:between")

    from_to_match = re.search(r"\bfrom\s+(19\d{2}|20[0-3]\d)\s+to\s+(19\d{2}|20[0-3]\d)\b", text)
    if from_to_match:
        year_a = int(from_to_match.group(1))
        year_b = int(from_to_match.group(2))
        return YearConstraint(year_min=min(year_a, year_b), year_max=max(year_a, year_b), source="query:from_to")

    after_match = re.search(r"\b(after|since|newer than|later than)\s+(19\d{2}|20[0-3]\d)\b", text)
    if after_match:
        return YearConstraint(year_min=int(after_match.group(2)), source=f"query:{after_match.group(1)}")

    before_match = re.search(r"\b(before|earlier than|older than)\s+(19\d{2}|20[0-3]\d)\b", text)
    if before_match:
        return YearConstraint(year_max=int(before_match.group(2)), source=f"query:{before_match.group(1)}")

    from_match = re.search(r"\bfrom\s+(19\d{2}|20[0-3]\d)\b", text)
    if from_match:
        return YearConstraint(year_min=int(from_match.group(1)), source="query:from")

    until_match = re.search(r"\buntil\s+(19\d{2}|20[0-3]\d)\b", text)
    if until_match:
        return YearConstraint(year_max=int(until_match.group(1)), source="query:until")

    if len(years) == 1:
        return YearConstraint(year_min=years[0], year_max=years[0], source="query:single_year")

    return YearConstraint(year_min=min(years), year_max=max(years), source="query:multi_year")


def _year_matches(year: Optional[int], constraint: YearConstraint) -> bool:
    if year is None:
        return True
    if constraint.year_min is not None and year < constraint.year_min:
        return False
    if constraint.year_max is not None and year > constraint.year_max:
        return False
    return True


def _log_search_query_box(queries: list[str], source_name: str = "Search") -> None:
    if not queries:
        logger.info("........................................")
        logger.info(". search queries: none available        .")
        logger.info("........................................")
        return

    title = f" {source_name} keyword queries "
    width = max(len(title) + 4, max(len(query) for query in queries) + 8)
    border = "." * width
    logger.info(border)
    logger.info(".%s.", title.center(width - 2))
    logger.info(border)
    for idx, query in enumerate(queries, start=1):
        line = f" {idx}. {query} "
        logger.info(".%s.", line.ljust(width - 2))
    logger.info(border)


async def _download_pdf(pdf_url: str, source_id: str) -> str:
    import httpx
    from pathlib import Path

    target_dir = Path(settings.PDF_STORE_PATH)
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_id = (source_id or "web").rsplit("/", 1)[-1].replace(":", "_").replace("/", "_")
    target = target_dir / f"{safe_id}.pdf"
    if target.exists():
        return str(target)

    async with httpx.AsyncClient(timeout=settings.TAVILY_HTTP_TIMEOUT, follow_redirects=True) as client:
        response = await client.get(pdf_url, headers={"User-Agent": settings.TAVILY_USER_AGENT})
        response.raise_for_status()
        target.write_bytes(response.content)
    return str(target)
