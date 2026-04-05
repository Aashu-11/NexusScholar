"""
query_rewriter.py — Multi-form query generation.
Rewrites each classified query into 4-6 parallel retrieval forms
optimized for different search strategies (dense, BM25, acronym, etc.).
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.generation.groq_client import GroqClient
from backend.retrieval.entity_extractor import QueryEntityProfile, extract_query_entities

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """You are an expert scientific literature search query optimizer.
Given a user query, generate multiple retrieval forms that maximize recall across different search strategies.
Think deeply about what the user is really asking — consider synonyms, related methods, foundational concepts, and the full research landscape.

User query: "{query}"
Detected intent: {intent}

Respond with ONLY a JSON object:
{{
  "dense_query": "<rich natural language sentence that captures the full semantic meaning, including related concepts and methodology — write as if explaining the research question to a colleague>",
  "bm25_query": "<all key noun phrases, method names, dataset names, benchmark names, metric names, and technical terms separated by spaces — be exhaustive with terminology>",
  "acronym_expanded": "<query with ALL acronyms expanded AND include alternate names for methods/models, e.g. BERT → bidirectional encoder representations from transformers, GPT → generative pre-trained transformer>",
  "paper_title_query": "<if user references a specific paper or seminal work, extract title; else null>",
  "author_query": "<if user mentions an author, extract name; else null>",
  "citation_graph_query": "<the most influential/foundational paper or method name to seed citation graph expansion — pick the single most important anchor paper; else null>"
}}
"""

ARXIV_DECOMPOSITION_PROMPT = """
Convert the research question into 8–12 diverse arXiv keyword search queries that together maximize full coverage of the research landscape. More queries = better recall; aim for the upper end.

STRATEGY:
1. Core concept queries — direct keywords from the question (2-3 queries)
2. Methodology queries — specific techniques, architectures, algorithms mentioned or implied (2-3 queries)
3. Evaluation queries — benchmarks, datasets, metrics relevant to the topic (1-2 queries)
4. Foundational/survey queries — broader topic area to capture survey papers and seminal works (1-2 queries)
5. Application/downstream queries — real-world use cases or task-specific angles (1 query)
6. Related technique queries — adjacent methods that often appear in the same papers (1 query)

RULES:
- Return ONLY valid JSON. No prose, no markdown fences.
- Each query must be 2–7 words in keyword style.
- Strip filler words only: "what", "are", "the", "compare", "explain", "how", "does", "a", "an", "of", "in".
- KEEP domain words: "models", "learning", "generation", "benchmark", "evaluation", "detection", "classification".
- KEEP all acronyms: BERT, GLUE, RAG, RLHF, ViT, NLU, QA, MMLU, SQuAD, NER.
- If the query compares multiple models/methods/datasets:
    * Generate SEPARATE queries per entity (1–2 queries each).
    * Add 1–2 cross-cutting queries about the shared task/benchmark.
    * NEVER put two competing entities in the same query.
- Include at least one query targeting survey/review papers on the topic.

FEW-SHOT EXAMPLES:

Input: "Compare BERT, RoBERTa, and DeBERTa on GLUE benchmark"
Output:
{{
  "arxiv_queries": [
    "bert language model glue benchmark",
    "roberta pretraining natural language understanding",
    "deberta disentangled attention glue",
    "transformer pretraining glue evaluation",
    "language model fine-tuning nlp benchmark",
    "pretrained language model comparison survey",
    "glue superglue benchmark leaderboard"
  ]
}}

Input: "How does RAG improve open-domain QA?"
Output:
{{
  "arxiv_queries": [
    "retrieval augmented generation open domain qa",
    "dense passage retrieval question answering",
    "rag knowledge grounded generation",
    "retrieval augmented language model",
    "open domain question answering survey",
    "knowledge intensive nlp tasks retrieval"
  ]
}}

Input: "What are recent advances in diffusion models for image generation?"
Output:
{{
  "arxiv_queries": [
    "diffusion models image generation",
    "denoising diffusion probabilistic models",
    "latent diffusion image synthesis",
    "score-based generative models",
    "diffusion model text-to-image",
    "generative model image synthesis survey",
    "conditional image generation diffusion guidance"
  ]
}}

NOW PROCESS:
User query: "{query}"
Detected intent: {intent}

Respond ONLY with (aim for 10-12 queries to maximize coverage):
{{"arxiv_queries": ["query 1", "query 2", ...]}}
"""


@dataclass
class QueryAnalysis:
    """Complete query understanding result."""
    original_query: str
    intent: str = "general"
    dense_query: str = ""
    bm25_query: str = ""
    acronym_expanded: str = ""
    paper_title_query: Optional[str] = None
    author_query: Optional[str] = None
    citation_graph_query: Optional[str] = None
    arxiv_queries: list[str] = field(default_factory=list)
    topic_terms: list[str] = field(default_factory=list)
    dense_queries: list[str] = field(default_factory=list)
    bm25_queries: list[str] = field(default_factory=list)
    all_queries: list[str] = field(default_factory=list)
    hyde_embedding: Optional[np.ndarray] = None
    freshness_required: bool = False
    entity_profile: Optional[QueryEntityProfile] = None


HYDE_PROMPT = """Write a 3-sentence academic abstract that directly answers this research question.
Use precise technical language and include specific numbers/methods if relevant.

Question: {query}

Abstract:"""


async def _generate_hyde(query: str, groq: GroqClient) -> Optional[np.ndarray]:
    """Generate a Hypothetical Document Embedding (HyDE) for the query.

    Asks the LLM to write a short fake abstract that would answer the query,
    then embeds that text for dense retrieval. This collapses the query-document
    vocabulary gap — the single most common reason dense retrieval fails on
    academic queries.
    """
    try:
        prompt = HYDE_PROMPT.format(query=query)
        response = await groq.complete_fast(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        hyde_text = response.strip()
        if not hyde_text or len(hyde_text) < 20:
            return None

        from backend.indexing.dense_index import embed_texts
        embedding = embed_texts([hyde_text])[0]
        logger.info("HyDE embedding generated (%d chars)", len(hyde_text))
        return embedding
    except Exception as e:
        logger.warning("HyDE generation failed: %s", e)
        return None


async def rewrite_query(query: str, intent: str, groq: GroqClient) -> QueryAnalysis:
    """
    Rewrite a query into multiple parallel retrieval forms.
    Uses Groq for diversity-optimized rewriting.
    """
    analysis = QueryAnalysis(original_query=query, intent=intent)

    # Flag queries that need freshness
    if intent in ("benchmark_comparison", "trend_analysis"):
        analysis.freshness_required = True

    try:
        logger.info("Query rewrite started: intent=%s query=%r", intent, query)
        prompt = REWRITE_PROMPT.format(query=query, intent=intent)
        response = await groq.complete_fast(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # lower temp for faithful rewrites, not creative ones
            max_tokens=500,
        )
        raw = response.strip()
        # Strip markdown fences if model wraps output
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        logger.info("Query rewrite raw response: %s", raw)

        analysis.dense_query = parsed.get("dense_query", query)
        analysis.bm25_query = parsed.get("bm25_query", query)
        analysis.acronym_expanded = parsed.get("acronym_expanded", query)
        analysis.paper_title_query = parsed.get("paper_title_query")
        analysis.author_query = parsed.get("author_query")
        analysis.citation_graph_query = parsed.get("citation_graph_query")

    except Exception as e:
        logger.warning("Query rewriting failed: %s — using original query", e)
        analysis.dense_query = query
        analysis.bm25_query = query
        analysis.acronym_expanded = query

    await _augment_analysis(analysis, groq)

    # Generate HyDE embedding for dense retrieval
    analysis.hyde_embedding = await _generate_hyde(query, groq)

    # Extract entity profile for downstream grounding
    from backend.config import settings
    if settings.ENTITY_EXTRACTION_ENABLED:
        analysis.entity_profile = await extract_query_entities(query, groq)
        logger.info(
            "Entity profile: subject=%r type=%s grounding=%s specificity=%.2f",
            analysis.entity_profile.primary_subject,
            analysis.entity_profile.entity_type,
            analysis.entity_profile.requires_entity_grounding,
            analysis.entity_profile.specificity_score,
        )

    # dense_queries: clean paraphrases for embedding search — no generic suffixes
    analysis.dense_queries = _unique_nonempty([
        analysis.dense_query,
        analysis.acronym_expanded,
        analysis.original_query,
    ])

    # bm25_queries: keyword-style, no injected generic sentences
    analysis.bm25_queries = _unique_nonempty([
        analysis.bm25_query,
        analysis.original_query,
    ])

    # all_queries: arXiv queries first (primary retrieval signal),
    # then dense paraphrases, then acronym expansion
    analysis.all_queries = _unique_nonempty([
        *analysis.arxiv_queries,
        analysis.dense_query,
        analysis.acronym_expanded,
        analysis.original_query,
    ])

    logger.info(
        "Query rewrite result: dense=%r bm25=%r acronym=%r title=%r author=%r "
        "graph=%r arxiv_queries=%r dense_queries=%r bm25_queries=%r",
        analysis.dense_query,
        analysis.bm25_query,
        analysis.acronym_expanded,
        analysis.paper_title_query,
        analysis.author_query,
        analysis.citation_graph_query,
        analysis.arxiv_queries,
        analysis.dense_queries,
        analysis.bm25_queries,
    )

    return analysis


async def _augment_analysis(analysis: QueryAnalysis, groq: GroqClient) -> None:
    """
    Augment analysis with topic terms and arXiv queries.
    Domain-specific boosts are OPT-IN — only applied when the query
    actually contains the relevant concept.
    """
    base_query = analysis.original_query.strip()
    lower = base_query.lower()

    topic_terms = _extract_topic_terms(base_query)

    # Domain boosting is conditional — never injected unconditionally
    if "rag" in lower or "retrieval-augmented generation" in lower or "retrieval augmented" in lower:
        topic_terms.extend([
            "retrieval augmented generation",
            "knowledge grounded generation",
        ])

    if "scientific" in lower or "biomedical" in lower:
        topic_terms.extend([
            "scientific literature",
            "biomedical question answering",
        ])

    if "question answering" in lower or " qa" in lower:
        topic_terms.extend([
            "question answering",
            "open domain qa",
        ])

    topic_terms = _unique_nonempty(topic_terms)
    analysis.topic_terms = topic_terms
    keyword_terms = _extract_keyword_terms(base_query, topic_terms)

    # BM25: extracted keywords + topic terms, no generic sentences
    analysis.bm25_query = " ".join(_unique_nonempty(
        [analysis.bm25_query or base_query, *topic_terms[:6]]
    ))

    # Dense: use LLM output as-is; fall back to original if empty
    if not analysis.dense_query:
        analysis.dense_query = base_query

    # Acronym expansion: only patch if LLM missed a known acronym
    if "rag" in lower and "retrieval-augmented generation" not in analysis.acronym_expanded.lower():
        analysis.acronym_expanded = (
            f"{analysis.acronym_expanded or base_query} retrieval-augmented generation"
        ).strip()

    analysis.arxiv_queries = await _build_arxiv_queries(
        base_query,
        intent=analysis.intent,
        groq=groq,
        keyword_terms=keyword_terms,
        topic_terms=topic_terms,
    )


def _extract_topic_terms(text: str) -> list[str]:
    """
    Extract meaningful topic terms from query text.
    Preserves multi-word phrases before falling back to unigrams.
    """
    STOPWORDS = {
        "what", "are", "the", "recent", "advances", "in", "for", "of", "and",
        "to", "with", "a", "an", "on", "about", "how", "does", "do", "is",
        "be", "compare", "comparison", "between", "using", "based",
    }

    lower = text.lower()
    terms: list[str] = []

    # Phase 1: extract known multi-word phrases first
    KNOWN_PHRASES = [
        "retrieval augmented generation",
        "natural language understanding",
        "natural language processing",
        "question answering",
        "machine translation",
        "named entity recognition",
        "text classification",
        "language model",
        "language models",
        "image generation",
        "image classification",
        "object detection",
        "knowledge graph",
        "graph neural network",
        "large language model",
        "instruction tuning",
        "fine-tuning",
        "pre-training",
        "benchmark evaluation",
        "transfer learning",
        "zero-shot learning",
        "few-shot learning",
        "open domain",
        "passage retrieval",
        "dense retrieval",
        "sparse retrieval",
    ]
    for phrase in KNOWN_PHRASES:
        if phrase in lower:
            terms.append(phrase)

    # Phase 2: capitalised tokens are likely model/dataset names — keep them
    capitalised = re.findall(r"\b[A-Z][A-Za-z0-9-]{1,}\b", text)
    terms.extend([t.lower() for t in capitalised])

    # Phase 3: remaining content words as unigrams
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", lower)
    terms.extend([t for t in tokens if t not in STOPWORDS and len(t) > 2])

    return _unique_nonempty(terms)


def _extract_keyword_terms(text: str, topic_terms: list[str]) -> list[str]:
    """
    Extract keyword terms optimised for arXiv-style search.
    Prioritises acronyms and domain phrases.
    """
    # All-caps acronyms (BERT, GLUE, RAG, ViT, etc.)
    acronyms = re.findall(r"\b[A-Z][A-Z0-9-]{1,}\b", text)

    # Known benchmark / dataset names
    KNOWN_BENCHMARKS = {
        "glue", "superglue", "squad", "mmlu", "hellaswag", "arc",
        "winogrande", "truthfulqa", "humaneval", "mbpp", "coco",
        "imagenet", "cifar", "pascal",
    }
    lower = text.lower()
    benchmarks = [b for b in KNOWN_BENCHMARKS if b in lower]

    # Comparison / benchmark intent signals
    comparison_terms: list[str] = []
    if re.search(r"\bcompar\w*\b|\bbenchmark\b|\bevaluat\w*\b", lower):
        comparison_terms.extend(["benchmark", "evaluation"])

    return _unique_nonempty([
        *[a.lower() for a in acronyms],
        *benchmarks,
        *topic_terms[:8],
        *comparison_terms,
    ])


async def _build_arxiv_queries(
    base_query: str,
    intent: str,
    groq: GroqClient,
    keyword_terms: list[str],
    topic_terms: list[str],
) -> list[str]:
    """
    Build arXiv-optimised keyword queries.
    Groq LLM is the primary path; term-based heuristic is the fallback.
    The fallback NEVER injects domain concepts not present in the query.
    """
    groq_queries = await _build_arxiv_queries_with_groq(base_query, intent, groq)
    if groq_queries:
        return groq_queries

    logger.warning("Groq arXiv decomposition failed — using term-based fallback")

    # Fallback: build queries purely from what the query contains
    terms = _unique_nonempty(keyword_terms)
    if not terms:
        return [" ".join(_extract_topic_terms(base_query)[:5]) or base_query]

    queries: list[str] = []

    # Sliding windows over top terms
    for window in (4, 3, 2):
        for start in range(0, min(len(terms), 10) - window + 1, 2):
            chunk = terms[start: start + window]
            queries.append(" ".join(chunk))

    # For comparison intent: per-entity queries
    if intent == "benchmark_comparison":
        # Capitalised tokens are likely model names
        entities = re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}\b", base_query)
        for entity in _unique_nonempty([e.lower() for e in entities])[:4]:
            queries.append(f"{entity} benchmark evaluation")
            queries.append(f"{entity} language model performance")

    # Filter to valid arXiv query lengths
    cleaned = [q for q in _unique_nonempty(queries) if 2 <= len(q.split()) <= 7]
    return cleaned[:8]


async def _build_arxiv_queries_with_groq(
    query: str, intent: str, groq: GroqClient
) -> list[str]:
    try:
        prompt = ARXIV_DECOMPOSITION_PROMPT.format(query=query, intent=intent)
        response = await groq.complete_fast(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        raw = response.strip()
        # Strip markdown fences if model wraps output
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        logger.info("arXiv decomposition raw response: %s", raw)
        parsed = json.loads(raw)
        queries = parsed.get("arxiv_queries", [])
        sanitized = _sanitize_arxiv_queries(queries)
        if sanitized:
            logger.info("arXiv decomposition accepted: %s", sanitized)
            return sanitized
    except Exception as exc:
        logger.warning("arXiv decomposition via Groq failed: %s", exc)
    return []


def _sanitize_arxiv_queries(values: list[str]) -> list[str]:
    """
    Sanitise LLM-generated arXiv queries.
    Only strips true filler words — NOT domain content words like
    "models", "using", "learning", "generation", "evaluation".
    """
    # Minimal filler-only stoplist — do NOT add domain words here
    FILLER = {
        "what", "are", "the", "compare", "explain", "tell",
        "show", "give", "a", "an", "of", "in", "is", "does",
    }
    cleaned_queries: list[str] = []
    for value in values:
        query = (value or "").strip().lower()
        # Remove characters not useful in arXiv search
        query = re.sub(r"[^a-z0-9+\-. ]", " ", query)
        query = re.sub(r"\s+", " ", query).strip()
        terms = [t for t in query.split() if t not in FILLER]
        if len(terms) < 2:
            continue
        cleaned_queries.append(" ".join(terms[:7]))
    return _unique_nonempty(cleaned_queries)[:12]


def _unique_nonempty(values: list[str]) -> list[str]:
    """Deduplicate a list of strings, preserving order, ignoring case."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = " ".join((value or "").split()).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(cleaned)
    return result