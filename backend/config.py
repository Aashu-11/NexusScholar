from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

load_dotenv(ROOT_DIR / ".env")
load_dotenv(ROOT_DIR / "backend" / ".env")


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    HOST: str = _env_str("HOST", "127.0.0.1")
    PORT: int = _env_int("PORT", 8000)
    FRONTEND_URL: str = _env_str("FRONTEND_URL", "http://localhost:5173")

    GROQ_API_KEY: str = _env_str("GROQ_API_KEY", "")
    GROQ_API_BASE: str = _env_str("GROQ_API_BASE", "https://api.groq.com/openai/v1")
    GROQ_MODEL_PRIMARY: str = _env_str("GROQ_MODEL_PRIMARY", "llama-3.3-70b-versatile")
    GROQ_MODEL_FAST: str = _env_str("GROQ_MODEL_FAST", "llama-3.1-8b-instant")

    DB_PATH: str = _env_str("DB_PATH", str(DATA_DIR / "db" / "nexus.db"))
    INDEX_PATH: str = _env_str("INDEX_PATH", str(DATA_DIR / "indexes"))
    PDF_STORE_PATH: str = _env_str("PDF_STORE_PATH", str(DATA_DIR / "pdfs"))
    PARSED_PATH: str = _env_str("PARSED_PATH", str(DATA_DIR / "parsed"))

    BM25_TOP_K: int = _env_int("BM25_TOP_K", 500)
    DENSE_TOP_K: int = _env_int("DENSE_TOP_K", 500)
    FUSED_TOP_K: int = _env_int("FUSED_TOP_K", 300)
    RRF_K: int = _env_int("RRF_K", 60)
    RERANKED_TOP_K: int = _env_int("RERANKED_TOP_K", 100)
    FINAL_EVIDENCE_TOP_K: int = _env_int("FINAL_EVIDENCE_TOP_K", 25)
    GRAPH_EXPANSION_LIMIT: int = _env_int("GRAPH_EXPANSION_LIMIT", 25)

    # Chunking — 512 tokens with 20% overlap (stride = 512 - 102 = 410)
    PASSAGE_CHUNK_TOKENS: int = _env_int("PASSAGE_CHUNK_TOKENS", 512)
    PASSAGE_STRIDE_TOKENS: int = _env_int("PASSAGE_STRIDE_TOKENS", 410)

    # BM25 tuning — k1=1.2 (term saturation), b=0.85 (length penalty; higher for long docs)
    BM25_K1: float = _env_float("BM25_K1", 1.2)
    BM25_B: float = _env_float("BM25_B", 0.85)

    # Hybrid search — alpha=0.7 gives 70% weight to dense, 30% to BM25
    HYBRID_ALPHA: float = _env_float("HYBRID_ALPHA", 0.7)

    # Reranker cutoffs — hard threshold on sigmoid(CE score); elbow = max score drop between consecutive results
    RERANKER_SCORE_THRESHOLD: float = _env_float("RERANKER_SCORE_THRESHOLD", 0.30)
    RERANKER_ELBOW_DROP: float = _env_float("RERANKER_ELBOW_DROP", 0.25)

    SYNTHESIS_TEMPERATURE: float = _env_float("SYNTHESIS_TEMPERATURE", 0.15)
    CONFIDENCE_THRESHOLD: float = _env_float("CONFIDENCE_THRESHOLD", 0.40)
    NLI_ENTAILMENT_THRESHOLD: float = _env_float("NLI_ENTAILMENT_THRESHOLD", 0.55)

    EMBEDDING_MODEL: str = _env_str(
        "EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"
    )
    EMBEDDING_QUERY_PREFIX: str = _env_str(
        "EMBEDDING_QUERY_PREFIX",
        "Represent this sentence for searching relevant passages: ",
    )
    RERANKER_MODEL: str = _env_str(
        "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
    )
    NLI_MODEL: str = _env_str("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
    PIPELINE_VERBOSE: bool = _env_bool("PIPELINE_VERBOSE", True)

    S2_AUTO_FETCH: bool = _env_bool("S2_AUTO_FETCH", True)
    S2_API_KEY: str = _env_str("S2_API_KEY", "")

    # Entity grounding
    ENTITY_EXTRACTION_ENABLED: bool = _env_bool("ENTITY_EXTRACTION_ENABLED", True)
    ENTITY_GROUNDING_PENALTY: float = _env_float("ENTITY_GROUNDING_PENALTY", 0.20)
    ENTITY_GROUNDING_BOOST: float = _env_float("ENTITY_GROUNDING_BOOST", 0.12)
    ENTITY_SPECIFICITY_THRESHOLD: float = _env_float("ENTITY_SPECIFICITY_THRESHOLD", 0.60)

    # Corpus coverage
    MIN_ENTITY_CONSISTENT_CANDIDATES: int = _env_int("MIN_ENTITY_CONSISTENT_CANDIDATES", 2)
    CORPUS_GAP_ABSTENTION_ENABLED: bool = _env_bool("CORPUS_GAP_ABSTENTION_ENABLED", True)

    # Synthesis grounding
    ENTITY_VERIFY_POST_SYNTHESIS: bool = _env_bool("ENTITY_VERIFY_POST_SYNTHESIS", True)
    ENTITY_VERIFY_CONFIDENCE_THRESHOLD: float = _env_float("ENTITY_VERIFY_CONFIDENCE_THRESHOLD", 0.70)

    EXA_API_KEY: str = _env_str("EXA_API_KEY", "")
    EXA_AUTO_FETCH: bool = _env_bool("EXA_AUTO_FETCH", True)
    EXA_NUM_RESULTS: int = _env_int("EXA_NUM_RESULTS", 40)
    EXA_MAX_CHARACTERS: int = _env_int("EXA_MAX_CHARACTERS", 40000)
    EXA_SEARCH_TYPE: str = _env_str("EXA_SEARCH_TYPE", "auto")
    EXA_USE_AUTOPROMPT: bool = _env_bool("EXA_USE_AUTOPROMPT", True)
    EXA_HIGHLIGHT_SENTENCES: int = _env_int("EXA_HIGHLIGHT_SENTENCES", 5)
    EXA_HIGHLIGHTS_PER_URL: int = _env_int("EXA_HIGHLIGHTS_PER_URL", 3)
    EXA_LIVECRAWL_TIMEOUT_MS: int = _env_int("EXA_LIVECRAWL_TIMEOUT_MS", 8000)

    TAVILY_API_KEY: str = _env_str("TAVILY_API_KEY", "")
    TAVILY_AUTO_FETCH: bool = _env_bool("TAVILY_AUTO_FETCH", True)
    TAVILY_MAX_RESULTS: int = _env_int("TAVILY_MAX_RESULTS", 8)
    TAVILY_HTTP_TIMEOUT: float = _env_float("TAVILY_HTTP_TIMEOUT", 60.0)
    TAVILY_SEARCH_DEPTH: str = _env_str("TAVILY_SEARCH_DEPTH", "advanced")
    TAVILY_TOPIC: str = _env_str("TAVILY_TOPIC", "general")
    TAVILY_USER_AGENT: str = _env_str(
        "TAVILY_USER_AGENT",
        "NexusScholar/1.0 (local development contact: admin@example.com)",
    )
    TAVILY_INCLUDE_DOMAINS: tuple[str, ...] = tuple(
        domain.strip()
        for domain in _env_str(
            "TAVILY_INCLUDE_DOMAINS",
            "arxiv.org,openreview.net,aclanthology.org,proceedings.mlr.press,papers.nips.cc,jmlr.org,semanticscholar.org,doi.org,nature.com,science.org,pubmed.ncbi.nlm.nih.gov,biorxiv.org,medrxiv.org,researchgate.net",
        ).split(",")
        if domain.strip()
    )

    # ColBERT late-interaction retrieval (optional — opt-in until index built)
    COLBERT_MODEL: str = _env_str("COLBERT_MODEL", "colbert-ir/colbertv2.0")
    COLBERT_ENABLED: bool = _env_bool("COLBERT_ENABLED", False)
    COLBERT_TOP_K: int = _env_int("COLBERT_TOP_K", 50)


settings = Settings()


def validate_settings(s: Settings) -> list[str]:
    """
    Validate settings at startup. Returns list of warning messages.
    Critical issues raise ValueError. Warnings are logged and returned.
    """
    warnings_list = []

    if not s.GROQ_API_KEY:
        warnings_list.append("GROQ_API_KEY not set — synthesis will use fallback mode")
    if not s.EXA_API_KEY:
        warnings_list.append("EXA_API_KEY not set — Exa primary search disabled, falling back to Tavily")
    if not s.TAVILY_API_KEY:
        warnings_list.append("TAVILY_API_KEY not set — Tavily fallback search disabled")
    if s.PASSAGE_CHUNK_TOKENS < 128:
        warnings_list.append(
            f"PASSAGE_CHUNK_TOKENS={s.PASSAGE_CHUNK_TOKENS} is very small — may hurt recall"
        )
    if s.RERANKED_TOP_K > s.FUSED_TOP_K:
        raise ValueError(
            f"RERANKED_TOP_K ({s.RERANKED_TOP_K}) > FUSED_TOP_K ({s.FUSED_TOP_K}) — "
            "reranker can't select more candidates than were fused"
        )
    if s.FINAL_EVIDENCE_TOP_K > s.RERANKED_TOP_K:
        raise ValueError(
            f"FINAL_EVIDENCE_TOP_K ({s.FINAL_EVIDENCE_TOP_K}) > RERANKED_TOP_K ({s.RERANKED_TOP_K})"
        )
    if s.NLI_ENTAILMENT_THRESHOLD > 0.9:
        warnings_list.append(
            "NLI_ENTAILMENT_THRESHOLD is very high — most citations will be flagged as unverified"
        )

    return warnings_list

for path in (Path(settings.DB_PATH).parent, Path(settings.INDEX_PATH), Path(settings.PDF_STORE_PATH), Path(settings.PARSED_PATH)):
    path.mkdir(parents=True, exist_ok=True)
