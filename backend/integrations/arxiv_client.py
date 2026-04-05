"""
arxiv_client.py — arXiv search and PDF download.

Key design decisions in _build_search_expression:
  - Field-qualified queries (ti:BERT AND abs:GLUE) from query_rewriter.py
    are passed through as-is after light normalisation. This is path A.
  - Plain keyword queries are decomposed into entity terms (model/method names
    → ti:) and task/benchmark terms (→ abs:) and combined with AND/OR.
    Never a naive multi-term AND-chain across all words in both fields.
    This is path B.

Root cause of the previous bad results:
  The old _build_search_expression would take "bert language model glue benchmark"
  and produce:
    (ti:"bert language model glue benchmark" OR abs:"bert language model glue benchmark")
    OR (ti:bert AND ti:glue)
    OR (abs:bert AND abs:glue)
  The exact-phrase search never matched anything useful. The ti: AND-chain
  required ALL terms in the title. And _select_search_terms was incorrectly
  stripping domain words like "language", "benchmark", "model" as "generic".

  Fixed output for "bert language model glue benchmark":
    ti:BERT AND abs:GLUE OR abs:BERT AND abs:GLUE OR ti:BERT
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# ---------------------------------------------------------------------------
# Known benchmark / dataset names — these go into abs:, never ti:
# ---------------------------------------------------------------------------
_KNOWN_BENCHMARKS: frozenset[str] = frozenset({
    "glue", "superglue", "squad", "squadv2", "mmlu", "hellaswag", "arc",
    "winogrande", "truthfulqa", "humaneval", "mbpp", "coco", "imagenet",
    "cifar", "pascal", "voc", "wmt", "snli", "mnli", "qnli", "wnli",
    "sst", "mrpc", "qqp", "rte", "cola", "stsb", "swag", "piqa",
    "boolq", "cb", "copa", "multirc", "record", "wic", "wsc",
    "naturalquestions", "triviaqa", "webquestions", "hotpotqa",
    "nq", "msmarco", "beir", "trec", "fever", "vqa", "gqa",
    "nocaps", "flickr30k", "mscoco",
})

# Known model/method names matched case-insensitively.
# Keys: lowercase. Values: canonical casing used in arXiv titles.
# NOTE: generic words ("attention", "transformer", "model") are intentionally
# excluded — they would create over-broad ti: matches.
_KNOWN_MODELS: dict[str, str] = {
    # BERT family
    "bert": "BERT",
    "roberta": "RoBERTa",
    "deberta": "DeBERTa",
    "debertav2": "DeBERTaV2",
    "debertav3": "DeBERTaV3",
    "distilbert": "DistilBERT",
    "tinybert": "TinyBERT",
    "albert": "ALBERT",
    "electra": "ELECTRA",
    "xlnet": "XLNet",
    "longformer": "Longformer",
    "bigbird": "BigBird",
    "reformer": "Reformer",
    # GPT / decoder family
    "gpt2": "GPT-2",
    "gpt3": "GPT-3",
    "gpt4": "GPT-4",
    "llama": "LLaMA",
    "llama2": "LLaMA-2",
    "llama3": "LLaMA-3",
    "mistral": "Mistral",
    "falcon": "Falcon",
    "alpaca": "Alpaca",
    "vicuna": "Vicuna",
    "palm": "PaLM",
    "gemini": "Gemini",
    "chinchilla": "Chinchilla",
    "instructgpt": "InstructGPT",
    "chatgpt": "ChatGPT",
    # Seq2seq / encoder-decoder
    "t5": "T5",
    "bart": "BART",
    "pegasus": "PEGASUS",
    "mt5": "mT5",
    # Vision / multimodal
    "vit": "ViT",
    "deit": "DeiT",
    "swin": "Swin",
    "clip": "CLIP",
    "blip": "BLIP",
    "flamingo": "Flamingo",
    "dalle": "DALL-E",
    # Retrieval models
    "colbert": "ColBERT",
    "splade": "SPLADE",
    "contriever": "Contriever",
    "dpr": "DPR",
    "realm": "REALM",
    "atlas": "ATLAS",
    "bm25": "BM25",
    # PEFT methods
    "lora": "LoRA",
    "qlora": "QLoRA",
}

# arXiv field qualifier detection pattern
_FIELD_RE = re.compile(r"\b(ti|abs|au|cat|id|jr|rn|all)\s*:", re.IGNORECASE)
_OPERATOR_WORDS: frozenset[str] = frozenset({"AND", "OR", "NOT", "ANDNOT"})


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    summary: str
    authors: list[dict]
    published_year: int | None
    updated: str | None
    pdf_url: str
    abs_url: str

    def metadata_override(self) -> dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.published_year,
            "venue": "arXiv",
            "arxiv_id": self.arxiv_id,
            "abstract": self.summary,
            "is_peer_reviewed": False,
            "source_url": self.abs_url or f"https://arxiv.org/abs/{self.arxiv_id}",
            "pdf_url": self.pdf_url,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def search_arxiv(query: str, max_results: int | None = None) -> list[ArxivPaper]:
    max_results = max_results or settings.ARXIV_MAX_RESULTS
    search_expr = _build_search_expression(query)
    url = (
        "https://export.arxiv.org/api/query"
        f"?search_query={quote_plus(search_expr)}"
        f"&start=0&max_results={max_results}"
        "&sortBy=relevance&sortOrder=descending"
    )
    logger.info("arXiv search: query=%r max_results=%s expr=%s", query, max_results, search_expr)
    async with httpx.AsyncClient(
        timeout=settings.ARXIV_HTTP_TIMEOUT,
        follow_redirects=True,
    ) as client:
        response = await client.get(url, headers={"User-Agent": settings.ARXIV_USER_AGENT})
        response.raise_for_status()
    papers = _parse_arxiv_feed(response.text)
    logger.info("arXiv search returned %s papers", len(papers))
    for idx, paper in enumerate(papers[:5], start=1):
        logger.info(
            "arXiv hit %s: %s | %s | %s",
            idx,
            paper.arxiv_id,
            paper.title,
            paper.pdf_url,
        )
    return papers


async def download_arxiv_pdf(paper: ArxivPaper, target_dir: str | Path) -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^0-9A-Za-z._-]", "_", paper.arxiv_id)
    target = target_dir / f"arxiv_{safe_id}.pdf"
    if target.exists():
        return target

    logger.info("Downloading arXiv PDF: %s -> %s", paper.pdf_url, target)
    async with httpx.AsyncClient(
        timeout=settings.ARXIV_HTTP_TIMEOUT,
        follow_redirects=True,
    ) as client:
        response = await client.get(paper.pdf_url, headers={"User-Agent": settings.ARXIV_USER_AGENT})
        response.raise_for_status()
        target.write_bytes(response.content)
    logger.info("Downloaded PDF for %s", paper.arxiv_id)
    return target


def arxiv_abs_url(arxiv_id: str | None) -> str | None:
    if not arxiv_id:
        return None
    return f"https://arxiv.org/abs/{arxiv_id}"


def arxiv_pdf_url(arxiv_id: str | None) -> str | None:
    if not arxiv_id:
        return None
    return f"https://arxiv.org/pdf/{arxiv_id}"


# ---------------------------------------------------------------------------
# Search expression builder — the core of precision retrieval
# ---------------------------------------------------------------------------

def _build_search_expression(query: str) -> str:
    """
    Convert a query string into an arXiv API search_query expression.

    Path A — field-qualified query (ti:BERT AND abs:GLUE):
        Produced by query_rewriter.py LLM path. Pass through after normalising
        boolean operator casing and whitespace. Highest-precision path.

    Path B — plain keyword query:
        Decompose into entity_terms (model/method → ti:) and abs_targets
        (benchmark/task → abs:), then build three tiers OR-ed together:
          1. ti:Entity AND abs:Task   — finds the model paper in task context
          2. abs:Entity AND abs:Task  — finds survey/comparison papers
          3. ti:Entity                — catches the original model paper alone

    Examples:
        "ti:BERT AND abs:GLUE"
            → path A → "ti:BERT AND abs:GLUE"

        "bert language model glue benchmark"
            → path B → "ti:BERT AND abs:GLUE OR abs:BERT AND abs:GLUE OR ti:BERT"

        "roberta pretraining natural language understanding"
            → path B → 'ti:RoBERTa AND abs:"natural language understanding"
                        OR abs:RoBERTa AND abs:"natural language understanding"
                        OR ti:RoBERTa'

        "colbert late interaction passage retrieval"
            → path B → 'ti:ColBERT AND abs:"passage retrieval"
                        OR abs:ColBERT AND abs:"passage retrieval"
                        OR ti:ColBERT'
    """
    query = query.strip()
    if not query:
        return 'abs:"deep learning"'

    if _is_field_qualified(query):
        return _normalise_field_query(query)

    return _build_plain_query_expression(query)


def _is_field_qualified(query: str) -> bool:
    """Return True if query contains arXiv field syntax (ti:, abs:, etc.)."""
    return bool(_FIELD_RE.search(query))


def _normalise_field_query(query: str) -> str:
    """
    Normalise a field-qualified query: uppercase AND/OR/ANDNOT, collapse spaces.
    Does NOT re-quote or restructure — query_rewriter already handles that.
    """
    q = re.sub(r"\bAND\b", "AND", query, flags=re.IGNORECASE)
    q = re.sub(r"\bOR\b", "OR", q, flags=re.IGNORECASE)
    q = re.sub(r"\bANDNOT\b", "ANDNOT", q, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", q).strip()


def _build_plain_query_expression(query: str) -> str:
    """
    Build a targeted arXiv search expression from a plain keyword query.

    1. Extract entity_terms (model/method names) → go into ti:
    2. Extract abs_targets (benchmarks, task phrases) → go into abs:
    3. Build three tiers and OR them together.
    """
    lower = query.lower()

    entity_terms = _extract_entity_terms(query)
    task_phrases = _extract_task_phrases(query)
    benchmark_terms = [
        b for b in _KNOWN_BENCHMARKS
        if re.search(r"\b" + re.escape(b) + r"\b", lower)
    ]

    # abs: targets — benchmarks first (exact acronym), then task phrases
    abs_targets: list[str] = []
    abs_targets.extend([b.upper() for b in benchmark_terms])
    abs_targets.extend([f'"{p}"' for p in task_phrases if len(p.split()) >= 2])

    # Fallback abs: — content words excluding entity names already in ti:
    if not abs_targets:
        entity_lowers = {e.lower() for e in entity_terms}
        content = [w for w in _extract_content_words(query) if w not in entity_lowers]
        if content:
            abs_targets.append(content[0])

    sub_expressions: list[str] = []

    if entity_terms and abs_targets:
        # Tier 1: ti:Entity AND abs:Task
        for entity in entity_terms[:3]:
            ti_term = _quoted_if_multiword(entity)
            for abs_target in abs_targets[:2]:
                sub_expressions.append(f"ti:{ti_term} AND abs:{abs_target}")
        # Tier 2: abs:Entity AND abs:Task
        for entity in entity_terms[:2]:
            for abs_target in abs_targets[:1]:
                sub_expressions.append(f"abs:{entity} AND abs:{abs_target}")
        # Tier 3: ti:Entity alone
        for entity in entity_terms[:3]:
            ti_term = _quoted_if_multiword(entity)
            sub_expressions.append(f"ti:{ti_term}")

    elif entity_terms:
        for entity in entity_terms[:3]:
            ti_term = _quoted_if_multiword(entity)
            sub_expressions.append(f"ti:{ti_term}")
            sub_expressions.append(f"abs:{entity}")

    elif abs_targets:
        for target in abs_targets[:3]:
            sub_expressions.append(f"abs:{target}")

    else:
        content = _extract_content_words(query)
        if content:
            sub_expressions.append(f'abs:"{" ".join(content[:4])}"')
        else:
            sub_expressions.append(f'abs:"{_escape_phrase(query)}"')

    # Deduplicate preserving order, case-insensitively
    seen: set[str] = set()
    deduped: list[str] = []
    for expr in sub_expressions:
        key = expr.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(expr)

    return " OR ".join(deduped)


# ---------------------------------------------------------------------------
# Term extraction helpers
# ---------------------------------------------------------------------------

def _extract_entity_terms(query: str) -> list[str]:
    """
    Extract model/method entity names from a query string.
    Results go into ti: queries.

    Three passes in priority order:
    1. ALL-CAPS tokens in original query (BERT, RAG, ViT)
    2. CamelCase tokens in original query (RoBERTa, DeBERTa, ColBERT)
    3. Known model names matched case-insensitively — handles plain lowercase
       queries like "bert glue benchmark" → BERT

    Benchmarks are excluded (they go to abs:).
    Generic words ("attention", "transformer") excluded from the known-models
    lookup to prevent over-broad ti: matches.
    """
    entities: list[str] = []

    # Pass 1: ALL-CAPS tokens
    for token in re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", query):
        if token in _OPERATOR_WORDS:
            continue
        if token.lower() in _KNOWN_BENCHMARKS:
            continue
        entities.append(token)

    # Pass 2: CamelCase tokens
    for token in re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b", query):
        if token.lower() not in _KNOWN_BENCHMARKS:
            entities.append(token)

    # Pass 3: known model names, longest match first (prevents "bert" matching
    # inside "deberta" before the full "deberta" pattern fires)
    lower = query.lower()
    for model_lower, model_canonical in sorted(_KNOWN_MODELS.items(), key=lambda x: -len(x[0])):
        if model_lower in _KNOWN_BENCHMARKS:
            continue
        if re.search(r"\b" + re.escape(model_lower) + r"\b", lower):
            entities.append(model_canonical)

    # Deduplicate preserving order, case-insensitively
    seen: set[str] = set()
    result: list[str] = []
    for e in entities:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result


def _extract_task_phrases(query: str) -> list[str]:
    """
    Extract multi-word NLP/ML task and method phrases for abs: queries.
    Ordered from most specific to most general; sub-phrases are suppressed
    if a longer phrase already covers their words.
    """
    lower = query.lower()
    TASK_PHRASES = [
        "open domain question answering",
        "retrieval augmented generation",
        "parameter efficient fine-tuning",
        "natural language understanding",
        "natural language processing",
        "natural language inference",
        "language model pretraining",
        "knowledge grounded generation",
        "visual question answering",
        "question answering",
        "machine translation",
        "named entity recognition",
        "semantic textual similarity",
        "textual entailment",
        "reading comprehension",
        "coreference resolution",
        "text classification",
        "sentiment analysis",
        "information retrieval",
        "passage retrieval",
        "dense retrieval",
        "sparse retrieval",
        "image generation",
        "image classification",
        "object detection",
        "image captioning",
        "instruction following",
        "instruction tuning",
        "fine-tuning",
        "code generation",
        "text summarization",
        "dialogue generation",
    ]
    found: list[str] = []
    seen_words: set[str] = set()
    for phrase in TASK_PHRASES:
        if phrase in lower:
            phrase_words = set(phrase.split())
            if not phrase_words.issubset(seen_words):
                found.append(phrase)
                seen_words.update(phrase_words)
    return found


def _extract_content_words(query: str) -> list[str]:
    """
    Extract meaningful content words from a plain query.
    Last-resort fallback for abs: when no entities or task phrases are found.
    """
    STOPWORDS = {
        "what", "are", "the", "recent", "advances", "in", "for", "of",
        "and", "to", "with", "a", "an", "on", "about", "how", "does",
        "do", "is", "be", "compare", "comparison", "between", "using",
        "based", "vs", "versus", "method", "methods", "approach",
        "approaches", "paper", "papers", "study",
    }
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", query.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _quoted_if_multiword(term: str) -> str:
    """Wrap multi-word terms in double quotes for arXiv field search."""
    term = term.strip()
    if " " in term:
        return f'"{_escape_phrase(term)}"'
    return term


def _escape_phrase(value: str) -> str:
    """Escape characters that break arXiv API phrase queries."""
    return value.replace("\\", " ").replace('"', " ").strip()


# ---------------------------------------------------------------------------
# Feed parser (unchanged from original)
# ---------------------------------------------------------------------------

def _parse_arxiv_feed(xml_text: str) -> list[ArxivPaper]:
    root = ET.fromstring(xml_text)
    papers: list[ArxivPaper] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        entry_id = _get_text(entry, "atom:id")
        arxiv_id = entry_id.rsplit("/", 1)[-1] if entry_id else ""
        title = _normalize_space(_get_text(entry, "atom:title"))
        summary = _normalize_space(_get_text(entry, "atom:summary"))
        published = _get_text(entry, "atom:published")
        year = int(published[:4]) if published and published[:4].isdigit() else None
        authors = []
        for author_el in entry.findall("atom:author", ATOM_NS):
            name = _normalize_space(_get_text(author_el, "atom:name"))
            if name:
                authors.append({"name": name, "affiliation": None})

        pdf_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break
        if not pdf_url and arxiv_id:
            pdf_url = arxiv_pdf_url(arxiv_id) or ""

        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                summary=summary,
                authors=authors,
                published_year=year,
                updated=_get_text(entry, "atom:updated"),
                pdf_url=pdf_url,
                abs_url=entry_id,
            )
        )
    return papers


def _get_text(node: ET.Element, path: str) -> str:
    found = node.find(path, ATOM_NS)
    return found.text.strip() if found is not None and found.text else ""


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()