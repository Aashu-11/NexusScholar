# NexusScholar — Complete System Architecture & Technical Reference

> **Version:** Production-1.0 | **Date:** 2026-04-07  
> **Stack:** FastAPI · React/TypeScript · SQLite · Groq LLaMA 3 · BAAI/bge-large · FAISS · NetworkX  
> **Purpose:** Enterprise-grade Retrieval-Augmented Generation (RAG) platform for scientific literature synthesis

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture Diagram](#2-high-level-architecture-diagram)
3. [Technology Stack — Every Dependency Explained](#3-technology-stack--every-dependency-explained)
4. [Configuration Layer](#4-configuration-layer)
5. [Stage 1 — PDF Ingestion Pipeline](#5-stage-1--pdf-ingestion-pipeline)
   - 5.1 Three-Tier PDF Parsing
   - 5.2 Multi-Granular Chunking (5 Levels)
   - 5.3 Table Extraction
   - 5.4 Claim Extraction
   - 5.5 Metadata Normalization
   - 5.6 Citation Graph Construction & PageRank
6. [Stage 2 — Indexing Layer](#6-stage-2--indexing-layer)
   - 6.1 SQLite Metadata Store
   - 6.2 BM25 Sparse Index
   - 6.3 Dense Vector Index (FAISS + BGE-Large)
   - 6.4 Embedding Cache
   - 6.5 ColBERT Late-Interaction Index (Optional)
7. [Stage 3 — Query Understanding Pipeline](#7-stage-3--query-understanding-pipeline)
   - 7.1 Compound Question Decomposition
   - 7.2 Intent Classification (10 Types)
   - 7.3 Query Rewriting (5 Parallel Forms)
   - 7.4 HyDE — Hypothetical Document Embedding
   - 7.5 Entity Extraction & Grounding
   - 7.6 Year/Recency Constraint Resolution
8. [Stage 4 — Hybrid Retrieval](#8-stage-4--hybrid-retrieval)
   - 8.1 Parallel Retrieval Lanes
   - 8.2 Weighted RRF Fusion (Reciprocal Rank Fusion)
   - 8.3 Graph Expansion (Citation Neighborhood)
   - 8.4 Parent/Sibling Chunk Expansion
   - 8.5 Pseudo-Relevance Feedback (PRF)
   - 8.6 External Web Retrieval (Exa + Tavily + Semantic Scholar)
9. [Stage 5 — Multi-Signal Reranking](#9-stage-5--multi-signal-reranking)
   - 9.1 Cross-Encoder (BAAI/bge-reranker-v2-m3)
   - 9.2 Multi-Signal Final Score Computation
   - 9.3 Entity Consistency Scoring
   - 9.4 Hard Cutoff & Elbow Method
   - 9.5 Listwise LLM Reranking (Second-Stage)
   - 9.6 Context Compression (TF-IDF Sentence Scoring)
10. [Stage 6 — Evidence Table Construction](#10-stage-6--evidence-table-construction)
11. [Stage 7 — Response Planning & Trust-Tier Admission](#11-stage-7--response-planning--trust-tier-admission)
12. [Stage 8 — Answer Synthesis (LLaMA 3.3 70B)](#12-stage-8--answer-synthesis-llama-33-70b)
    - 12.1 System Prompt Architecture (10 Hard Rules)
    - 12.2 Mathematical Computation — LaTeX + Python Sandbox
    - 12.3 Streaming via Server-Sent Events (SSE)
13. [Stage 9 — Post-Synthesis Verification](#13-stage-9--post-synthesis-verification)
    - 13.1 Citation Validation & NLI Entailment
    - 13.2 Entity Identity Verification
    - 13.3 Coverage Verification & Gap-Fill
    - 13.4 Self-Evaluation + Conditional Regeneration Loop
14. [Stage 10 — Citation Rendering & Response Delivery](#14-stage-10--citation-rendering--response-delivery)
15. [API Layer — All Endpoints](#15-api-layer--all-endpoints)
16. [Frontend Architecture](#16-frontend-architecture)
17. [Database Schema — Complete](#17-database-schema--complete)
18. [End-to-End Data Flow Trace](#18-end-to-end-data-flow-trace)
19. [Performance & Tuning Parameters](#19-performance--tuning-parameters)
20. [Security Considerations](#20-security-considerations)

---

## 1. System Overview

NexusScholar is a **production-grade Retrieval-Augmented Generation (RAG) system** built specifically for scientific literature synthesis. Unlike general-purpose chat applications, every architectural decision is optimized for the hard problem of faithfully answering research questions from a corpus of uploaded academic papers — with zero hallucination on specific claims, full citation traceability, and graded confidence signals.

### Why RAG over a General LLM?

A large language model (LLM) such as LLaMA 3.3 70B has extensive parametric knowledge encoded in its weights, but this knowledge has a training cutoff and — critically — cannot be attributed to specific sources. For research use, this creates two fatal problems:

1. **Hallucination of citations:** The model may confidently cite papers that do not exist, or attribute findings to the wrong authors.
2. **Temporal staleness:** Papers published after the training cutoff are invisible to the model.

RAG resolves this by **separating retrieval from generation**. The LLM is given only a curated, cited evidence table extracted from documents uploaded by the user. Every claim it writes must be grounded in — and tagged to — a specific evidence row. The system then programmatically validates every citation tag it produces.

### Core Design Principles

| Principle | Implementation |
| --- | --- |
| Grounded-only generation | 10 hard rules in system prompt; post-synthesis NLI verification |
| Multi-granular retrieval | 5 chunk levels (document, section, passage, claim, table) |
| Evidence diversity | 5+ parallel retrieval lanes fused via weighted RRF |
| Entity precision | Named entity extraction prevents cross-entity contamination |
| Mathematical integrity | Isolated Python subprocess for arithmetic; no LLM estimation |
| Authority ranking | PageRank on citation graph; peer-review tier weighting |
| Graceful abstention | Corpus gap detection; explicit "insufficient evidence" paths |

### Scale

- ~9,100 lines of Python across backend
- ~2,100 lines in the main chat orchestrator (`chat.py`) alone
- 51 configuration parameters
- 10 sequential pipeline stages
- 5+ parallel retrieval lanes per query

---

## 2. High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            USER (Browser)                                    │
│                    React + TypeScript + Zustand                              │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │ HTTP POST /api/chat
                                 │ SSE stream back (tokens + metadata)
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (uvicorn / asyncio)                       │
│                                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐   ┌───────────────┐  │
│  │  /api/chat  │   │ /api/ingest │   │ /api/papers  │   │ /api/evidence │  │
│  │  (SSE orch) │   │ (PDF upload)│   │ (search/meta)│   │ (audit/hover) │  │
│  └──────┬──────┘   └─────────────┘   └──────────────┘   └───────────────┘  │
│         │                                                                    │
│  ┌──────▼──────────────────────────────────────────────────────────────┐    │
│  │              QUERY UNDERSTANDING PIPELINE                            │    │
│  │  1. Compound Decomposition → 2. Intent Classification               │    │
│  │  3. Query Rewriting (5 forms) → 4. HyDE → 5. Entity Extraction     │    │
│  └──────┬──────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│  ┌──────▼──────────────────────────────────────────────────────────────┐    │
│  │                  HYBRID RETRIEVAL (parallel)                         │    │
│  │  BM25 │ Dense-FAISS │ HyDE │ ColBERT │ Graph-Expand │ Web(Exa/Tav) │    │
│  └──────┬──────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ▼ Reciprocal Rank Fusion (Weighted RRF)                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                   MULTI-SIGNAL RERANKING                             │   │
│  │  Cross-Encoder + Section Weight + Recency + Citation Count          │   │
│  │  + PageRank + Entity Consistency + Evidence Density                  │   │
│  └──────┬───────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│  ┌──────▼───────────────────────────────────────────────────────────────┐   │
│  │              EVIDENCE TABLE CONSTRUCTION                              │   │
│  │  Top-25 candidates → EvidenceRow objects with full metadata          │   │
│  └──────┬───────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│  ┌──────▼───────────────────────────────────────────────────────────────┐   │
│  │                 SYNTHESIS (LLaMA 3.3 70B via Groq)                   │   │
│  │  10 hard rules · LaTeX math · Python sandbox · streaming tokens      │   │
│  └──────┬───────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│  ┌──────▼───────────────────────────────────────────────────────────────┐   │
│  │              POST-SYNTHESIS VERIFICATION                              │   │
│  │  Citation validation · NLI · Entity check · Coverage · Self-eval    │   │
│  └──────┬───────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼ SSE token stream → client                                         │
└──────────────────────────────────────────────────────────────────────────────┘
         ▲                         ▲                         ▲
         │                         │                         │
┌────────┴────────┐    ┌──────────┴──────────┐   ┌─────────┴──────────┐
│  SQLite (WAL)   │    │  FAISS Index        │   │  NetworkX Graph    │
│  papers/chunks  │    │  BAAI/bge-large-en  │   │  PageRank cache    │
│  embeddings     │    │  1024-dim, cosine   │   │  citation edges    │
│  conversations  │    │                     │   │                    │
└─────────────────┘    └─────────────────────┘   └────────────────────┘
```

---

## 3. Technology Stack — Every Dependency Explained

### Backend Runtime

| Component | Library / Version | Why Chosen |
| --- | --- | --- |
| Web framework | FastAPI 0.115.8 | Async-native, automatic OpenAPI docs, Pydantic validation |
| ASGI server | uvicorn | Production-grade, supports HTTP/2, needed for SSE streaming |
| Async DB driver | aiosqlite | Non-blocking SQLite reads/writes on the async event loop |
| Settings | python-dotenv + dataclass | Frozen dataclass prevents accidental mutation of config |

### AI / ML Models

| Role | Model | Rationale |
| --- | --- | --- |
| Primary LLM | `llama-3.3-70b-versatile` (Groq) | 70B parameters for deep analytical synthesis; Groq inference hardware achieves ~400 tok/s |
| Fast LLM | `llama-3.1-8b-instant` (Groq) | Classification and rewriting tasks where latency matters more than depth |
| Dense embedding | `BAAI/bge-large-en-v1.5` | 1024-dimensional sentence embeddings; trained explicitly for retrieval with instruction prefixing; state-of-the-art on MTEB |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` | Multilingual cross-encoder that jointly encodes query+document pairs — far more accurate than bi-encoder but slower |
| NLI verification | `cross-encoder/nli-deberta-v3-base` | DeBERTa-based NLI model checks entailment between claims and cited passages |
| ColBERT (optional) | `colbert-ir/colbertv2.0` via `ragatouille` | Late-interaction: per-token MaxSim scoring between query and document token sets |

### Retrieval Infrastructure

| Component | Library | Details |
| --- | --- | --- |
| Vector search | FAISS (`IndexFlatIP`) | Exact inner-product search on L2-normalized vectors ≡ cosine similarity; no approximation error |
| Sparse retrieval | `rank-bm25==0.2.2` (BM25Okapi) | Classic probabilistic term-matching with k1=1.2, b=0.85 tuned for scientific text |
| Citation graph | `networkx` (DiGraph) | Directed graph for citation edges; PageRank via `nx.pagerank(alpha=0.85)` |
| Web search primary | Exa API | Neural web search — unlike keyword search, Exa understands semantic similarity |
| Web search fallback | Tavily API | Broad web search with domain filtering to academic sources |
| Paper metadata | Semantic Scholar API | Auto-hydration of citation counts, peer-review status, DOIs |
| arXiv integration | arXiv API | Direct paper search for freshness when corpus is thin |

### Frontend

| Component | Library | Details |
| --- | --- | --- |
| Framework | React + TypeScript | Type-safe component tree |
| State management | Zustand | Lightweight, no boilerplate; separate chat and evidence stores |
| Streaming | Custom SSE hook | Parses `text/event-stream` for token-by-token rendering |
| Bundler | Vite | Fast HMR during development |

---

## 4. Configuration Layer

**File:** `backend/config.py`

All 51 settings are defined as a frozen dataclass loaded from environment variables at startup. The `frozen=True` constraint means the settings object is immutable — no component can accidentally mutate global config at runtime.

```python
@dataclass(frozen=True)
class Settings:
    # LLM
    GROQ_MODEL_PRIMARY: str = "llama-3.3-70b-versatile"   # 70B for deep synthesis
    GROQ_MODEL_FAST:    str = "llama-3.1-8b-instant"      # 8B for classification
    SYNTHESIS_TEMPERATURE: float = 0.15                    # Near-deterministic generation

    # Retrieval funnel sizes
    BM25_TOP_K:           int   = 500    # BM25 initial pool
    DENSE_TOP_K:          int   = 500    # Dense initial pool
    FUSED_TOP_K:          int   = 300    # After RRF fusion
    RERANKED_TOP_K:       int   = 100    # After cross-encoder rerank
    FINAL_EVIDENCE_TOP_K: int   = 25     # Fed into LLM context

    # Chunking
    PASSAGE_CHUNK_TOKENS: int = 512      # Window size
    PASSAGE_STRIDE_TOKENS: int = 410     # Stride (≈20% overlap)

    # Hybrid weighting
    HYBRID_ALPHA: float = 0.7            # 70% dense, 30% BM25

    # Cross-encoder cutoffs
    RERANKER_SCORE_THRESHOLD: float = 0.30  # sigmoid(CE score) minimum
    RERANKER_ELBOW_DROP:      float = 0.25  # score drop for elbow cut

    # Entity grounding
    ENTITY_GROUNDING_PENALTY: float = 0.20  # wrong-entity penalty
    ENTITY_GROUNDING_BOOST:   float = 0.12  # correct-entity boost
    ENTITY_SPECIFICITY_THRESHOLD: float = 0.60  # minimum to trigger grounding

    # NLI verification
    NLI_ENTAILMENT_THRESHOLD: float = 0.55

    # BM25 tuning
    BM25_K1: float = 1.2   # term saturation
    BM25_B:  float = 0.85  # length normalization
```

**Startup validation** (`validate_settings`) enforces hard constraints — e.g., `RERANKED_TOP_K ≤ FUSED_TOP_K` — and raises `ValueError` for impossible configurations.

The funnel cascade must satisfy: `BM25_TOP_K` → `FUSED_TOP_K` → `RERANKED_TOP_K` → `FINAL_EVIDENCE_TOP_K`, i.e., each stage narrows the candidate set.

---

## 5. Stage 1 — PDF Ingestion Pipeline

**Entry point:** `POST /api/ingest/upload` → `backend/ingestion/service.py`

When a user uploads a PDF, the ingestion pipeline transforms raw binary into structured, indexed, searchable chunks. This is an 8-step sequential process.

### 5.1 Three-Tier PDF Parsing

**File:** `backend/ingestion/pdf_parser.py`

Scientific PDFs are notoriously difficult to parse because they use complex two-column layouts, mathematical equations, embedded figures, and non-standard fonts. The system uses a cascading 3-tier parser where each tier is tried in order, falling back to the next if it fails:

#### Tier 1: Grobid TEI XML (Preferred)

**What it is:** Grobid (GeneRation Of BIbliographic Data) is a machine-learning-based PDF-to-TEI-XML parser specifically trained on scientific literature. It segments papers into structured components: header, abstract, body sections, reference lists, footnotes.

**Why first:** It produces the richest structured output — section headings are correctly identified, references are parsed into individual fields (author, title, year, venue), and figure/table captions are separated from body text.

**How it works:** If a pre-parsed TEI XML file exists in `data/parsed/` (from a previous Grobid run), it is loaded directly. The XML is then parsed with Python's `xml.etree.ElementTree` to extract `<title>`, `<author>`, `<date>`, `<abstract>`, `<body>` divs, and `<biblStruct>` reference entries.

**Output:** `ParsedPaper` dataclass with `title`, `authors`, `year`, `venue`, `doi`, `arxiv_id`, `abstract`, `full_text`, `sections` (dict of heading → text), `references` (list of dicts).

#### Tier 2: Marker (Transformer-Based Markdown Extraction)

**What it is:** Marker is a modern deep-learning PDF parser that converts PDFs to Markdown, preserving tables, equations, and layout structure better than traditional heuristics.

**Why second:** Better at handling modern PDF formats with complex layouts but requires the `ragatouille` package, which is optional.

**How it works:** Returns markdown text that is then parsed with regex-based section detection (lines starting with `#`, `##`, bold headers).

#### Tier 3: PyMuPDF / fitz (Universal Fallback)

**What it is:** PyMuPDF (`fitz`) is a Python binding for the MuPDF rendering library. It provides direct, low-level access to PDF text streams.

**Why third:** Works on any PDF but produces flat text without structure — no section headings, no table semantics.

**Year detection:** Applies multiple regex patterns in order:
1. 4-digit year in PDF metadata properties
2. arXiv ID pattern `(\d{4})\.\d{4,5}` 
3. First occurrence of a plausible year (1990–2030) in the text body

### 5.2 Multi-Granular Chunking (5 Levels)

**File:** `backend/ingestion/chunker.py`

A single fixed-size chunking strategy cannot serve all retrieval needs. Long passages capture context but miss precise atomic claims. Short sentences are precise but lack context. The system creates **5 granularity levels simultaneously** for every paper, stored as separate chunk rows in the database.

**Token approximation:** `tokens ≈ words / 1.3` (lightweight — no tokenizer loaded at chunk time)

**Configuration:** `PASSAGE_CHUNK_TOKENS=512`, `PASSAGE_STRIDE_TOKENS=410` (≈20% overlap)

```
Level 1 — Document  : Entire abstract + conclusion combined. 
                       Used for document-level queries ("what is this paper about?")

Level 2 — Section   : Each section (Methods, Results, etc.) as a whole unit.
                       Used for section-targeted queries ("describe the method")

Level 3 — Passage   : 512-token sliding window with 410-token stride.
                       Primary retrieval granularity — balances context with precision.
                       Overlap (102 tokens) prevents evidence from falling at chunk boundaries.

Level 4 — Claim     : Individual sentences extracted by regex sentence boundary detection.
                       Used when precision is paramount (e.g., exact statistic lookup)

Level 5 — Table     : Markdown table → serialized text rows.
                       Allows BM25 and dense search to match tabular data.
```

**Why 20% overlap on passage chunks?** If a key finding spans a chunk boundary, neither chunk alone contains it. The overlap ensures that any 512-token span in the original document is fully represented in at least one chunk.

**Section tagging:** Every chunk is tagged with its section name (e.g., `"results"`, `"methods"`, `"abstract"`) because the section is used as a signal in the reranker's multi-signal score — results sections receive the highest boost (0.10) since they contain the most evidence-dense content.

### 5.3 Table Extraction

**File:** `backend/ingestion/table_extractor.py`

Tables in scientific papers (benchmark comparison tables, ablation studies, hyperparameter tables) are among the most information-dense content. However, raw markdown table syntax (`| col | col |`) is not easily retrievable by either BM25 or dense models because the syntax is sparse and the semantic content is distributed across rows.

**Detection:** Regex matching for pipe-delimited markdown table patterns.

**Serialization to natural language:**
```
Table from results section. Columns: Method, Dataset, Accuracy, F1.
Row 1: Method: BERT-base, Dataset: GLUE, Accuracy: 79.6%, F1: 0.803.
Row 2: Method: RoBERTa, Dataset: GLUE, Accuracy: 88.5%, F1: 0.891.
```

This serialization makes each cell value retrievable via keyword (BM25 matches "79.6%" or "BERT-base") and each row semantically embeddable as a sentence.

**Storage:** Each table becomes a separate chunk with `granularity="table"` and the appropriate `section_tag`.

### 5.4 Claim Extraction

**File:** `backend/ingestion/claim_extractor.py`

Individual atomic claims (single sentences expressing one finding) are extracted for high-precision retrieval. These are stored as `granularity="claim"` chunks.

**Method:** Regex-based sentence boundary detection (split on `.`, `!`, `?` with heuristics to avoid splitting on abbreviations like "e.g.", "et al.", decimal numbers).

**Use case:** When the query asks for a very specific statistic or finding, claim-level chunks can directly surface the exact sentence without surrounding context.

### 5.5 Metadata Normalization

**File:** `backend/ingestion/normalizer.py`

Ensures all metadata fields are extracted, cleaned, and stored consistently regardless of which parser tier was used.

**Year extraction cascade:** Grobid date → PDF metadata year → arXiv ID prefix → regex scan of first 1000 chars.

**Venue normalization:** Cleans and normalizes conference/journal names (removes extra whitespace, standardizes capitalization).

**Author parsing:** Handles "Firstname Lastname", "Lastname, Firstname", and affiliation-attached formats.

**External enrichment (Semantic Scholar):** If `S2_AUTO_FETCH=True` and an S2 API key is configured, the system automatically fetches:
- `citation_count` (number of citing papers)
- `is_peer_reviewed` (journal/conference publication flag)
- `is_retracted` (retraction status)
- `openalex_id`
- Canonical `doi`

This enrichment is critical for the trust-tier corpus admission and the reranker's citation authority signal.

### 5.6 Citation Graph Construction & PageRank

**File:** `backend/ingestion/graph_builder.py`

Every paper carries a reference list. When a paper is ingested, its references are resolved against papers already in the corpus. This builds an explicit **directed citation graph** that can be traversed at query time to find related papers.

**Data structure:**
```python
_graph: nx.DiGraph
# Nodes: paper_id (str), attributes: title, year, venue, citation_count
# Edges: (citing_paper_id → cited_paper_id)
```

**Reference resolution:** For each reference dict `{title, authors, year, venue}`, the system queries SQLite for a matching paper by fuzzy title match. If found, a directed edge is added: `citing_paper → cited_paper`.

**Persistence:** All resolved citation edges are stored in the `citation_edges` table in SQLite so the graph can be rebuilt on restart.

**PageRank computation:**

PageRank is Google's original link analysis algorithm adapted here for scientific citation authority. In the citation graph, a paper with many highly-cited papers citing it has high PageRank — it is considered authoritative.

```
Formula: PR(u) = (1 - α) + α · Σ_{v→u} PR(v) / out_degree(v)
Parameters: α = 0.85 (damping factor), max_iter=200, tol=1e-6
```

**Eager computation:** PageRank is computed at startup and cached in `_pagerank_cache`. It is marked dirty (needing recomputation) whenever a new paper is added. At query time, `get_pagerank(paper_id)` returns a normalized score in [0, 1] (normalized against the max score in the corpus so contributions are consistent regardless of corpus size).

**Usage in reranking:** The PageRank score contributes a log-scaled boost to the reranker's multi-signal score (up to +0.05), acting as an authority prior that complements cross-encoder relevance.

---

## 6. Stage 2 — Indexing Layer

After parsing and chunking, all chunks are embedded and indexed. Three independent index structures are maintained in parallel.

### 6.1 SQLite Metadata Store

**File:** `backend/indexing/metadata_store.py`

SQLite with WAL (Write-Ahead Logging) mode serves as the persistent ground truth for all metadata. WAL mode enables concurrent reads while a write is in progress — important for a system where background ingestion and foreground queries can overlap.

**Key settings:**
```sql
PRAGMA journal_mode=WAL;       -- concurrent reads during writes
PRAGMA foreign_keys=ON;        -- enforce referential integrity
```

**Schema:** See [Section 17](#17-database-schema--complete) for full schema.

**Embedding storage:** Chunk embeddings are stored as BLOB columns (binary-serialized numpy float32 arrays). This means the FAISS index can be rebuilt from the database on restart without re-running the embedding model.

**Batch queries:** When the reranker needs metadata for N papers, a single `SELECT * FROM papers WHERE paper_id IN (?, ?, ...)` query is issued rather than N individual queries — critical for performance at scale.

### 6.2 BM25 Sparse Index

**File:** `backend/indexing/bm25_index.py`  
**Library:** `rank-bm25==0.2.2` (BM25Okapi algorithm)

**What BM25 is:** BM25 (Best Match 25) is a bag-of-words retrieval function that ranks documents based on the frequency of query terms appearing in each document, adjusted for document length. It is the gold standard for sparse keyword retrieval.

**BM25Okapi formula:**
```
Score(D, Q) = Σ_{t ∈ Q} IDF(t) · (f(t,D) · (k1 + 1)) / (f(t,D) + k1 · (1 - b + b · |D|/avgDL))

Where:
  f(t, D)  = term frequency in document D
  |D|      = document length
  avgDL    = average document length in corpus
  k1 = 1.2 (term saturation: diminishing returns for repeated terms)
  b  = 0.85 (length penalty: longer docs are penalized more than shorter ones)
  IDF(t)   = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
             where N = corpus size, n(t) = documents containing term t
```

**Why k1=1.2, b=0.85?** Scientific papers are typically longer than web pages. Higher `b` (closer to 1.0) means more aggressive length normalization — a term appearing once in a 5000-word methods section is not treated as more important than the same term appearing once in a 200-word abstract. `k1=1.2` limits term saturation so a term appearing 10 times isn't 10x more important than one appearing once.

**Custom scientific tokenizer:**
```python
def tokenize(text: str) -> list[str]:
    # 1. Preserve hyphens in compound terms: "state-of-the-art" → kept as one token
    # 2. Preserve decimals: "98.5" → not split at decimal point
    # 3. Lightweight suffix normalization:
    #    -ing → '' (running → runn)
    #    -tion → 't' (activation → activat)
    #    -ity → '' (similarity → similar)
    # 4. Protect known scientific compounds: RBMK, BERT, COVID-19
    # 5. Remove pure numeric tokens < 4 digits (avoids matching on "1", "10", etc.)
```

**Why not use NLTK/spaCy stemming?** Heavy stemming can collapse distinct scientific terms. The lightweight custom normalizer is enough to improve recall without losing precision on method names like "RoBERTa" or "BERT-large".

**Index build:** At startup or on corpus change, all passage-level chunks (plus optionally table chunks) are loaded from SQLite, tokenized, and fed to `BM25Okapi([tokenized_doc for doc in chunks])`. The built index is stored in memory.

**Search:** `bm25.get_scores(tokenized_query)` returns a numpy array of scores. The top-`BM25_TOP_K` (500) chunks are returned as `(chunk_dict, bm25_score, rank)` tuples.

### 6.3 Dense Vector Index (FAISS + BGE-Large)

**File:** `backend/indexing/dense_index.py`

Dense retrieval uses neural embeddings to find semantically similar chunks even when there is no keyword overlap. This is critical for scientific retrieval where the same concept can be described in many ways.

**Embedding model: `BAAI/bge-large-en-v1.5`**
- **Architecture:** Fine-tuned BERT-style transformer
- **Embedding dimension:** 1024
- **Training:** Contrastive learning on retrieval pairs — the model is explicitly trained so that query embeddings are close to relevant document embeddings in L2 space
- **Instruction prefixing:** BGE-large uses task-specific instruction prefixes. For queries: `"Represent this sentence for searching relevant passages: {query}"`. For documents: no prefix. This asymmetric approach significantly improves retrieval quality.

**FAISS index type: `IndexFlatIP`**
- Inner Product (IP) = dot product
- On L2-normalized vectors, `dot(a, b) = cos(a, b)` — so inner product ≡ cosine similarity
- `IndexFlat` = exact search, no approximation (HNSW or IVF would be faster but less accurate)
- At typical corpus sizes (<100K papers), exact search is fast enough

**Entity-hint injection:**
```python
def embed_query(query: str, entity_hint: Optional[str] = None) -> np.ndarray:
    if entity_hint:
        prefixed = f"Represent this sentence for searching relevant passages about {entity_hint}: {query}"
    else:
        prefixed = settings.EMBEDDING_QUERY_PREFIX + query
```

When the entity extractor identifies a specific named entity (e.g., "RBMK reactor"), that entity is injected into the embedding prefix. This biases the query embedding toward the entity's semantic neighborhood, improving precision for entity-specific queries.

**GPU acceleration:** If CUDA is available, the SentenceTransformer model is loaded to GPU. Batch embedding of 500 chunks takes ~2-3 seconds on GPU vs. ~30 seconds on CPU.

### 6.4 Embedding Cache

**File:** `backend/indexing/embedding_cache.py`

Embedding computation is the most expensive per-chunk operation (~0.5ms/chunk on GPU, ~10ms/chunk on CPU). The cache stores computed embeddings in SQLite BLOB columns so they survive restarts.

**Key design:** On startup, the dense index loads all chunk embeddings from the DB (bypassing model inference) and builds the FAISS index from stored BLOBs. Only truly new chunks trigger model inference.

### 6.5 ColBERT Late-Interaction Index (Optional)

**File:** `backend/indexing/colbert_index.py`  
**Library:** `ragatouille`  
**Model:** `colbert-ir/colbertv2.0`  
**Enabled by:** `COLBERT_ENABLED=True`

**What ColBERT is:** Unlike bi-encoders (which compress query and document each into a single vector), ColBERT retains per-token embeddings for both query and document. At retrieval time, it computes a **MaxSim** score:

```
ColBERT_Score(Q, D) = Σ_{qi ∈ Q} max_{dj ∈ D} (qi · dj)
```

For each query token, find the most similar document token; sum these maxima. This allows fine-grained token-level matching — far more expressive than a single-vector similarity.

**Trade-off:** ColBERT is more accurate than bi-encoders but requires storing per-token vectors (much larger index size). At `COLBERT_TOP_K=50`, it provides a third retrieval lane that is fused via RRF.

**Graceful degradation:** If `ragatouille` is not installed, ColBERT silently disables itself and the system continues with BM25 + Dense only.

---

## 7. Stage 3 — Query Understanding Pipeline

Before any retrieval occurs, the system performs extensive query analysis. This pipeline runs 5 sequential sub-stages, most calling the Groq API.

### 7.1 Compound Question Decomposition

**File:** `backend/generation/question_decomposer.py`

Scientific queries are frequently compound — users ask multiple questions in one message. If treated as a single query, some sub-questions will get poor coverage in retrieval.

**Detection:** LLM classification with the fast model (LLaMA 3.1 8B, temp=0.0).

**Output dataclass:**
```python
@dataclass
class QuestionDecomposition:
    original_query: str
    is_compound: bool
    sub_questions: list[str]   # 1-5 sub-questions
    count: int
    reasoning: str
```

**Example:**
```
Input:  "What is LoRA? How does it compare to full fine-tuning? What datasets are used for PEFT evaluation?"
Output: is_compound=True, count=3
        sub_questions=["What is LoRA?",
                       "How does LoRA compare to full fine-tuning?",
                       "What datasets are used to evaluate PEFT methods?"]
```

**Downstream impact:** If compound, query rewriting runs **independently** for each sub-question, and all resulting retrieval forms are merged into a union set before hybrid recall. This casts the widest possible retrieval net.

### 7.2 Intent Classification (10 Types)

**File:** `backend/retrieval/query_classifier.py`  
**Model:** LLaMA 3.1 8B (temp=0.1)

The query's intent determines the **response format** and **retrieval strategy** used downstream.

| Intent | Example Query | Impact |
| --- | --- | --- |
| `literature_survey` | "What are recent advances in protein folding?" | 800-2000 word synthesis; no table required |
| `benchmark_comparison` | "Compare BERT, RoBERTa, DeBERTa on GLUE" | Mandatory comparison table; auto 2y recency filter |
| `method_explanation` | "How does attention mechanism work?" | Detailed technical explanation; equations required |
| `paper_lookup` | "Find the LoRA paper by Hu et al. 2021" | Title/author search prioritized |
| `trend_analysis` | "How has NLP evolved from 2018 to 2024?" | Chronological synthesis; auto 2y recency filter |
| `dataset_discovery` | "What datasets are available for NER?" | Dataset metadata prioritized |
| `author_search` | "What has Yann LeCun published on CNNs?" | Author query form used |
| `definition` | "What is perplexity in NLP?" | Short precise definition; equations |
| `contradiction_check` | "Do papers agree on dropout regularization effectiveness?" | Conflict synthesis; both sides cited |
| `general` | Fallback | Standard retrieval path |

**Auto-recency tightening:** For `benchmark_comparison` and `trend_analysis` intents, if the user didn't explicitly specify a recency filter and the query isn't about historical work, the system automatically applies a 2-year recency constraint. This prevents outdated benchmark numbers from dominating results about rapidly-evolving fields like LLMs.

### 7.3 Query Rewriting (5 Parallel Forms)

**File:** `backend/retrieval/query_rewriter.py`

A single query string is simultaneously suboptimal for BM25 (needs keywords) and dense retrieval (needs semantic richness). The rewriter generates 5 specialized forms from one LLM call.

**LLM prompt strategy:**
```
dense_query:     "Rich natural language capturing full semantic meaning, 
                  related concepts, methodology — as if explaining to a colleague"
bm25_query:      "All key noun phrases, method names, dataset names, benchmark names, 
                  metric names, technical terms — be exhaustive with terminology"
acronym_expanded:"ALL acronyms expanded + alternate names: 
                  BERT → bidirectional encoder representations from transformers"
paper_title_query: (if query references a specific paper)
author_query:    (if query mentions an author name)
```

**ArXiv query generation (8-12 forms):** A second LLM call using `ARXIV_DECOMPOSITION_PROMPT` generates 8-12 diverse keyword queries for arXiv search, following a structured strategy:
1. Core concept queries (2-3)
2. Methodology queries (2-3)
3. Evaluation/benchmark queries (1-2)
4. Survey/foundational queries (1-2)
5. Application queries (1)
6. Related technique queries (1)

**Result:** `QueryAnalysis` dataclass containing all rewritten forms, the entity profile, HyDE embedding, and freshness signals.

### 7.4 HyDE — Hypothetical Document Embedding

**What it is:** HyDE (Hypothetical Document Embedding) is a technique to bridge the vocabulary gap between queries and documents. A user's query is typically short and informal; indexed document chunks are long, formal, and dense with technical language. Direct query-to-document similarity is suboptimal.

**How it works:**
1. The fast LLM generates a hypothetical 3-sentence abstract that **would answer the query** if it were a real paper.
2. That hypothetical abstract is embedded using the same `BAAI/bge-large-en-v1.5` model.
3. The hypothetical embedding becomes an **additional retrieval lane** in hybrid recall.

**Why it works:** The hypothetical abstract uses the vocabulary and style of real academic papers. Its embedding is therefore much closer to actual relevant document embeddings than the raw query embedding.

**Example:**
```
Query: "How does LoRA reduce GPU memory during fine-tuning?"
HyDE abstract: "Low-Rank Adaptation (LoRA) reduces the memory footprint of large 
language model fine-tuning by decomposing weight update matrices into low-rank 
factors. By freezing pre-trained weights and only training two small matrices A 
and B where W_update = BA, LoRA reduces trainable parameters by 10,000x compared 
to full fine-tuning, enabling 65B parameter models to run on a single GPU. 
Experiments on GPT-3 demonstrate that LoRA matches full fine-tuning quality while 
reducing memory by 67%."
```

This hypothetical abstract has embeddings close to the actual LoRA paper's methods section chunks.

### 7.5 Entity Extraction & Grounding

**File:** `backend/retrieval/entity_extractor.py`

When a query asks about a **specific named entity** (a specific reactor design, a specific drug compound, a specific algorithm, a specific dataset), correct answers require that retrieved evidence actually discusses *that exact entity* — not a similar one in the same category.

**QueryEntityProfile:**
```python
@dataclass
class QueryEntityProfile:
    primary_subject:          Optional[str]        # e.g., "RBMK reactor"
    entity_type:              str                  # reactor_type | chemical | drug | 
                                                   # organism | technique | material |
                                                   # algorithm | dataset | model | general
    entity_aliases:           list[str]            # e.g., ["RBMK-1000", "Soviet reactor"]
    exclusion_entities:       list[str]            # e.g., ["TRIGA", "PWR", "CANDU"]
    domain:                   str                  # nuclear_physics | chemistry | biology |
                                                   # machine_learning | medicine | etc.
    requires_entity_grounding: bool                # True only when specificity_score >= 0.60
    specificity_score:        float                # 0.0–1.0
```

**Extraction:** LLM call with the fast model (temp=0.0 for determinism). The model identifies the primary subject, its aliases, and — critically — **exclusion entities** (other entities in the same category that should NOT appear in the answer).

**The entity grounding cascade:** This profile propagates through the entire downstream pipeline:
- **Dense index:** Entity hint injected into query embedding prefix
- **Reranker:** `_entity_consistency_score()` per candidate
- **Compressor:** Entity-wrong chunks multiplied by 0.1
- **Synthesizer:** RULE 7 and RULE 8 in system prompt
- **Verifier:** Post-synthesis entity consistency check

**Why this matters:** Consider the query "What is the positive void coefficient in RBMK reactors?" Without entity grounding, chunks about PWR or TRIGA reactors (also nuclear reactors with void coefficients) would be retrieved and could contaminate the answer with properties of the wrong reactor type. Entity grounding hard-penalizes such chunks in the reranker and prohibits the LLM from using parametric knowledge to fill gaps.

### 7.6 Year/Recency Constraint Resolution

**Function:** `_resolve_year_constraint()` in `chat.py`

Parses year constraints from:
1. User's recency filter dropdown (`any`, `1y`, `3y`)
2. Year references in the query text ("papers from 2023", "recent 2024 work")

**Output:** `YearConstraint(year_min, year_max, source)` applied as a hard filter in retrieval.

---

## 8. Stage 4 — Hybrid Retrieval

**File:** `backend/retrieval/hybrid_recall.py`

The retrieval stage runs **multiple parallel lanes** simultaneously and fuses their results. No single retrieval method is optimal for all queries:

- BM25 excels at exact keyword matches (method names, dataset names, model architectures)
- Dense retrieval excels at semantic similarity (synonyms, paraphrases, concept-level matching)
- HyDE bridges the vocabulary gap
- Graph expansion finds related papers not directly retrieved
- Web retrieval provides freshness beyond the local corpus

### 8.1 Parallel Retrieval Lanes

All lanes execute concurrently (async/await with `asyncio.gather`):

**Lane 1 — BM25:**
- Multiple BM25 queries (from `query_analysis.bm25_queries`) sent to the BM25 index
- Each returns up to `BM25_TOP_K=500` results
- Results from all sub-queries are merged (union) before RRF

**Lane 2 — Dense (Primary):**
- Multiple semantic queries (from `query_analysis.dense_queries`) embedded and searched
- FAISS inner-product search against 1024-dim indexed vectors
- Returns up to `DENSE_TOP_K=500` results

**Lane 3 — HyDE:**
- Pre-computed HyDE embedding (from query rewriting stage) searched against FAISS
- Separate from the primary dense lane so it can be weighted independently

**Lane 4 — ColBERT (Optional):**
- MaxSim retrieval if `COLBERT_ENABLED=True`
- `COLBERT_TOP_K=50` results

**Lane 5 — Corpus BM25 expansion:**
- Additional BM25 search using PRF-expanded query terms
- `pseudo_relevance_feedback.py` extracts high-frequency discriminative terms from initial top results
- These terms are added back as a new BM25 query for a second-pass retrieval

**Lane 6 — Web retrieval (async, parallel):**
- **Exa (primary):** Neural web search across academic domains, `EXA_NUM_RESULTS=40`, live-crawl with `EXA_LIVECRAWL_TIMEOUT_MS=8000`
- **Tavily (fallback):** Broad web search restricted to academic domains (arxiv.org, openreview.net, aclanthology.org, pubmed, biorxiv, etc.), `TAVILY_MAX_RESULTS=8`, `TAVILY_SEARCH_DEPTH="advanced"`
- **Semantic Scholar:** Citation count and metadata hydration
- **arXiv API:** Direct paper search using 8-12 generated arXiv queries

Web results are converted to synthetic chunk-like dicts with `section_tag="web_content"` and merged into the candidate pool before RRF.

### 8.2 Weighted RRF Fusion (Reciprocal Rank Fusion)

**What RRF is:** RRF is a rank combination method that is robust to different score scales across retrieval systems. Instead of combining raw scores (which are incomparable across BM25, cosine similarity, and ColBERT), it uses only **rank positions**.

**Standard RRF formula:**
```
RRF(d) = Σ_i 1 / (k + rank_i(d))
where k=60 is a constant that controls the influence of high-rank documents
```

**NexusScholar's weighted RRF:**
```python
def reciprocal_rank_fusion(
    result_lists: list,   # list_idx=0 is BM25, list_idx=1+ are dense lanes
    k: int = 60,
    top_k: int = 200,
    alpha: float = 0.7,   # HYBRID_ALPHA from config
) -> list[RetrievalCandidate]:
    for i, (chunk, score, rank) in enumerate(all_results):
        if list_idx == 0:  # BM25 lane
            weight = 1.0 - alpha  # 0.30
        else:              # Dense / HyDE / ColBERT lanes
            weight = alpha         # 0.70
        rrf_score = weight / (k + rank)
        combined[chunk_id] += rrf_score
```

**Why `alpha=0.7`?** Dense retrieval generally outperforms BM25 on scientific text because it captures semantic similarity. The 70/30 split gives dense retrieval priority while retaining BM25's strength on exact keyword queries. This parameter is tunable per deployment.

**Why `k=60`?** At k=60, the contribution from the top-ranked document is 1/(60+1)≈0.016. The difference between rank 1 and rank 100 is significant but the difference between rank 200 and rank 300 is negligible. This smoothly blends high-rank results.

**Output:** Up to `FUSED_TOP_K=300` `RetrievalCandidate` objects with their `rrf_score`, `bm25_rank`, and `dense_rank` stored.

### 8.3 Graph Expansion (Citation Neighborhood)

**File:** `backend/retrieval/graph_expander.py`

After initial retrieval, the top `GRAPH_EXPANSION_LIMIT=25` papers' citation neighborhoods are explored. Papers that cite or are cited by retrieved papers are likely topically related and may contain relevant evidence not captured by the initial retrieval.

**Expansion strategies:**
1. **Cited-by:** Papers that cite a retrieved paper (later work building on it)
2. **References:** Papers cited by a retrieved paper (foundational work)
3. **Co-citation cluster:** Papers frequently cited alongside a retrieved paper

**Graph traversal:** Single-hop BFS in the NetworkX DiGraph. Each expanded paper's chunks are fetched from SQLite and added to the candidate pool with `is_graph_expanded=True`.

**Why graph expansion?** A user asking about "attention mechanisms in transformers" will retrieve the Transformer paper (Vaswani et al. 2017). Graph expansion then surfaces BERT, GPT, and other papers that cite it — which the user also wants.

### 8.4 Parent/Sibling Chunk Expansion

**File:** `backend/retrieval/chunk_expander.py`

When a claim-level chunk (a single sentence) is retrieved as relevant, that sentence may lack context. The chunk expander fetches the **parent passage chunk** (the 512-token window containing this sentence) and adjacent chunks for richer context.

**Operations:**
- `expand_with_parents(chunks)` — for each claim/small chunk, fetch the containing passage
- `fetch_sibling_passages(chunk)` — fetch the passage immediately before and after in the paper

This ensures the LLM receives sufficient context around each retrieved evidence point.

### 8.5 Pseudo-Relevance Feedback (PRF)

**File:** `backend/retrieval/pseudo_relevance_feedback.py`

PRF assumes the top-k initially retrieved documents are relevant and uses their content to improve the query.

**Process:**
1. Take the top-10 documents from initial BM25 retrieval
2. Extract the most frequent and discriminative terms (TF-IDF-like scoring against the full corpus)
3. Add these terms to the BM25 query for a second-pass retrieval

**Why PRF?** If a user asks about "transformer attention" and many top results mention "self-attention mechanism" and "multi-head attention", adding these terms to the BM25 query improves recall of additional relevant chunks that use this specific terminology.

**Gate:** PRF is only applied if the initial retrieval found enough high-confidence candidates — if retrieval is already poor, PRF can amplify noise.

### 8.6 External Web Retrieval

**Exa (`backend/integrations/exa_client.py`):**
- Neural search engine trained for semantic similarity (not keyword matching)
- Configuration: `EXA_NUM_RESULTS=40`, `EXA_MAX_CHARACTERS=40000`, `EXA_SEARCH_TYPE="auto"`, `EXA_USE_AUTOPROMPT=True`
- Highlights extraction: up to 5 highlights per result, 3 per URL, providing pre-extracted relevant sentences
- Live-crawl mode: fetches and parses pages in real time with `EXA_LIVECRAWL_TIMEOUT_MS=8000`
- Returns paper metadata + content snippets converted to evidence rows

**Tavily (`backend/integrations/tavily_client.py`):**
- Fallback when Exa is unavailable or for broader coverage
- `TAVILY_SEARCH_DEPTH="advanced"` (deeper crawl)
- Hard domain whitelist: `arxiv.org`, `openreview.net`, `aclanthology.org`, `proceedings.mlr.press`, `jmlr.org`, `nature.com`, `pubmed.ncbi.nlm.nih.gov`, `biorxiv.org`, and 6 others
- `TAVILY_MAX_RESULTS=8` per query

**Semantic Scholar (`backend/integrations/semantic_scholar.py`):**
- Used primarily for metadata hydration (citation counts, peer-review status)
- Also invoked for paper lookup when `intent="paper_lookup"`

---

## 9. Stage 5 — Multi-Signal Reranking

**File:** `backend/retrieval/reranker.py`  
**Model:** `BAAI/bge-reranker-v2-m3`

After RRF fusion produces 300 candidates, the reranker dramatically narrows this to the 100 most relevant using a **cross-encoder** — a fundamentally different and more accurate approach than the bi-encoder used for initial retrieval.

### 9.1 Cross-Encoder (BAAI/bge-reranker-v2-m3)

**What a cross-encoder does:** Unlike bi-encoders (which encode query and document separately), a cross-encoder processes the `[query, document]` pair **jointly**. The query and document tokens attend to each other through full self-attention at every layer.

This joint encoding captures:
- Which specific query terms match which document terms
- Whether the document actually *answers* the query vs. merely sharing vocabulary
- Subtle relevance signals that are lost when encoding independently

**Why it's more accurate but slower:** Cross-encoders cannot pre-compute document representations. Every (query, candidate) pair must be encoded from scratch. At 300 candidates with 512-char text limits, this is ~300 forward passes through a transformer — expensive, but worthwhile given the quality gain.

**GPU batching:** `batch_size=64` balances GPU utilization vs. memory. On an A100, 300 pairs take ~0.5 seconds; on a CPU, ~15 seconds.

**Text truncation:** Chunk text is truncated to 512 characters before the tokenizer, reducing Python-side overhead while the model's tokenizer handles max_length internally.

### 9.2 Multi-Signal Final Score Computation

The raw cross-encoder score (logit) is combined with 8 auxiliary signals to produce the final ranking score:

```
final_score = CE_score × 0.75                   # Cross-encoder dominates (75%)
            + section_weight                      # Section authority
            + recency_boost                       # Publication year
            + citation_count_boost                # Highly-cited = authoritative
            + topic_alignment × 0.12             # Keyword overlap with query
            + evidence_density × 0.10            # Numbers, metrics, claims
            + granularity_bonus                   # Claim > Document > Section
            + peer_review_boost                   # +0.03 if peer-reviewed
            + pagerank_contribution               # Citation graph authority
            + entity_consistency_adjustment       # Entity grounding signal
```

**Section weights:**
```
results:     +0.10   (most evidence-dense)
abstract:    +0.09   (concise summary)
methods:     +0.08   (technical detail)
conclusion:  +0.07   (synthesis)
discussion:  +0.06   (interpretation)
web_content: +0.04   (external source)
introduction:+0.03   (framing, less evidence)
unknown:     +0.02
```

**Recency boosts (intent-dependent):**
- `benchmark_comparison` / `trend_analysis`: ≥2024 → +0.12, ≥2023 → +0.08, ≤2020 → **-0.10**
- General: ≥2023 → +0.05, ≥2021 → +0.03, ≥2018 → +0.01

**Evidence density scoring:** The `_evidence_density()` function returns a 0–1 score:
```python
if re.search(r"\b(outperform|improv|achiev|state-of-the-art|sota)\b", text):
    score += 0.45   # Result-bearing language
if re.search(r"\b\d+(?:\.\d+)?%|\b\d+\.\d+\b", text):
    score += 0.30   # Numbers (metrics, percentages)
if re.search(r"\b(accuracy|f1|precision|recall|bleu|rouge|auc)\b", text):
    score += 0.25   # Metric names
```

### 9.3 Entity Consistency Scoring

**Method:** `_entity_consistency_score()` in `reranker.py`

For queries with `requires_entity_grounding=True`, every candidate is scored for how well its content aligns with the target entity:

```
Score 1.0 — Primary subject explicitly confirmed in chunk text or paper title/abstract
Score 0.9 — Only an alias found (not primary name, but recognized equivalent)
Score 0.6 — Domain terminology present but no entity name found
Score 0.5 — No entity signals (neutral)
Score 0.4 — Co-occurrence: BOTH primary subject AND an exclusion entity present
             (e.g., a survey paper comparing multiple reactor types)
Score 0.0 — ONLY exclusion entity present, primary subject completely absent
```

**The co-occurrence case (0.4):** A survey paper comparing RBMK and PWR reactors is not *wrong* — it does discuss the primary entity (RBMK) but also discusses exclusion entities (PWR). It receives a mild penalty (-0.06 to final score) rather than hard rejection (which would remove potentially useful comparative information).

**Translation to final score:**
```python
if entity_score == 0.0:  score -= 0.20  # Hard wrong: penalize heavily
elif entity_score >= 0.9: score += 0.12  # Confirmed correct: boost
elif entity_score == 0.4: score -= 0.06  # Co-occurrence: mild penalty
elif entity_score < 0.6:  score -= 0.10  # Ambiguous: moderate penalty
```

The `entity_decision` field (`"correct"`, `"wrong"`, `"ambiguous"`, `"neutral"`) is stored on each `RetrievalCandidate` and propagated to the compressor and synthesizer.

### 9.4 Hard Cutoff & Elbow Method

After multi-signal scoring, two sequential pruning operations are applied:

**Hard cutoff (sigmoid threshold):**
```python
threshold = 0.30   # RERANKER_SCORE_THRESHOLD
min_keep = max(3, top_k // 4)

sigmoid_ce = 1 / (1 + exp(-raw_ce_score))
filtered = [c for c in reranked if sigmoid_ce >= threshold]

if len(filtered) >= min_keep:
    reranked = filtered  # Drop low-relevance candidates
```

The `sigmoid(raw_CE_score)` calibration makes the threshold model-independent: regardless of the cross-encoder's output scale, sigmoid maps scores to [0, 1].

**Elbow method:**
```python
for i in range(1, len(reranked)):
    drop = reranked[i-1].final_score - reranked[i].final_score
    if drop >= RERANKER_ELBOW_DROP:  # 0.25
        reranked = reranked[:i]       # Cut at the relevance cliff
        break
```

The elbow method detects a "relevance cliff" — a large score gap between consecutive results indicating the transition from relevant to irrelevant. Cutting there avoids forcing low-quality evidence into the context.

### 9.5 Listwise LLM Reranking (Second-Stage)

**Method:** `listwise_rerank()` in `reranker.py`

After cross-encoder reranking, an optional second-stage reranking lets the LLM holistically assess candidate diversity and complementarity.

**How it works:**
1. Format top-50 candidates as numbered summaries: `[i] {paper_title} ({section}): {preview}`
2. Prompt the fast LLM: "Given query X, rank the TOP {top_k} most relevant by index number. Consider: direct relevance, specificity of evidence, complementarity (avoid 5 passages saying the same thing)."
3. Parse the returned JSON index list
4. Reorder candidates accordingly

**Why listwise?** Pointwise ranking (cross-encoder) evaluates each candidate independently and cannot reason about redundancy. If 5 retrieved passages all say "BERT achieves 79.6% on GLUE", only one is useful. The listwise LLM can see all candidates simultaneously and prefer diverse, complementary evidence.

### 9.6 Context Compression (TF-IDF Sentence Scoring)

**File:** `backend/retrieval/compressor.py`

Before building the evidence table, each passage chunk is compressed to its most relevant sentences. This reduces the amount of text fed to the LLM while preserving the highest-value content.

**Method:** TF-IDF-like sentence scoring
1. Extract all sentences from the chunk
2. Score each sentence by term overlap with the query
3. Apply entity decision weights: `entity_decision="wrong"` → multiply score × 0.1; `"ambiguous"` → × 0.5
4. Return the top-scoring N sentences (typically 3-5)

**Why compress?** The LLM has a finite context window. At `FINAL_EVIDENCE_TOP_K=25` passages of 512 tokens each, that's 12,800 tokens of evidence alone — near the limit. Compression can reduce this by 60-70% while keeping the most relevant sentences.

---

## 10. Stage 6 — Evidence Table Construction

**File:** `backend/generation/evidence_builder.py`

The top `FINAL_EVIDENCE_TOP_K=25` reranked candidates are transformed into a structured `EvidenceTable` — the single artifact passed to the LLM for generation.

**EvidenceRow dataclass:**
```python
@dataclass
class EvidenceRow:
    evidence_id:    str          # Unique ID referenced in [CIT:id] tags
    paper_title:    str
    authors:        list[str]
    year:           Optional[int]
    venue:          str
    doi:            Optional[str]
    arxiv_id:       Optional[str]
    chunk_text:     str          # Compressed passage content
    section_tag:    str          # "results", "methods", "abstract", etc.
    relevance_score: float       # Final reranker score
    is_peer_reviewed: bool
    is_retracted:   bool
    is_preprint:    bool
    citation_count: int
    source_url:     str          # Clickable URL for citation hover cards
    pdf_url:        str
```

**EvidenceTable:**
```python
@dataclass
class EvidenceTable:
    answer_id:         str
    query:             str
    intent:            str
    rows:              list[EvidenceRow]
    total_sources:     int
    confidence_score:  float   # Aggregate signal based on number/quality of sources
```

**Confidence scoring:**
The confidence score (0.0–1.0) is computed from:
- Number of evidence rows (more = higher confidence)
- Proportion of peer-reviewed sources
- Average relevance score of top rows
- Presence of retracted papers (reduces confidence)

**Evidence deduplication (`backend/generation/evidence_dedup.py`):**
Before building the table, duplicate or near-duplicate evidence rows are removed. Duplicates arise when:
- The same chunk appears in multiple retrieval lanes (BM25 + Dense)
- Graph expansion retrieves the same paper as initial retrieval

Deduplication uses exact `evidence_id` matching and near-duplicate text detection (Jaccard similarity on bigrams > 0.85 threshold).

---

## 11. Stage 7 — Response Planning & Trust-Tier Admission

**File:** `backend/generation/planner.py`

Before synthesis, the planner evaluates the evidence table and decides the response strategy.

**TaskPlan dataclass:**
```python
@dataclass
class TaskPlan:
    intent:                 str
    response_format:        str     # "prose" | "comparison_table" | "structured_list" | "equation"
    confidence_level:       str     # "high" | "medium" | "low" | "insufficient"
    confidence_score:       float
    is_retrieval_sufficient: bool
    should_abstain:         bool    # True when evidence is too thin
    reasoning:              str
    key_points:             list[str]
```

**Trust-tier corpus admission:**

Papers are evaluated for quality along three dimensions before their evidence rows are included in synthesis:

| Tier | Criteria | Treatment |
| --- | --- | --- |
| Peer-reviewed | `is_peer_reviewed=True` | Full trust; used without qualification |
| Preprint | `is_peer_reviewed=False`, `is_retracted=False` | Used with `[Preprint]` label; synthesis must hedge |
| Retracted | `is_retracted=True` | Included in evidence table but flagged; synthesis must warn |

**Abstention triggers:**

The system explicitly abstains (refuses to answer) rather than hallucinate when:
- `confidence_score < CONFIDENCE_THRESHOLD` (0.40) — too few relevant sources
- All retrieved chunks have entity decisions of `"wrong"` — corpus doesn't contain the right entity
- `CORPUS_GAP_ABSTENTION_ENABLED=True` and `MIN_ENTITY_CONSISTENT_CANDIDATES` (2) correct-entity chunks not found
- `is_retrieval_sufficient=False` and no web results available

**Abstention response:** "The indexed corpus does not contain sufficient evidence to answer this question with confidence. Please upload papers directly addressing [topic]."

**Format selection:**
- `benchmark_comparison` → `comparison_table` (mandatory Markdown table)
- `method_explanation` → `prose` with equations
- `literature_survey` → `prose` (800-2000 words)
- `definition` → `structured_list` (short precise definition)

---

## 12. Stage 8 — Answer Synthesis (LLaMA 3.3 70B)

**File:** `backend/generation/synthesizer.py`  
**Model:** `llama-3.3-70b-versatile` via Groq API  
**Temperature:** 0.15 (near-deterministic; minimal creativity needed)

### 12.1 System Prompt Architecture (10 Hard Rules)

The system prompt is the most critical artifact for answer quality and safety. It encodes 10 rules that are treated as absolute constraints — violation of any rule is declared "a system failure" in the prompt itself.

**RULE 1 — Citation mandate:** Every factual sentence must end with `[CIT:evidence_id]`. No citation = uncited claim = rule violation.

**RULE 2 — No bare attribution:** Paper titles, author names, statistics, and method names cannot appear without a corresponding evidence row. Prevents the LLM from hallucinating paper names from parametric memory.

**RULE 3 — Conflict synthesis:** Conflicting evidence rows must be explicitly noted with both sides cited: "Paper A finds X [CIT:A], while Paper B finds Y [CIT:B], likely due to [analysis]."

**RULE 4 — Abstention mandate:** If evidence is insufficient, say so explicitly. Do not fabricate plausible-sounding answers.

**RULE 5 — Preprint labeling:** Claims from preprints must be labeled `[Preprint]` to signal to readers that the finding hasn't survived peer review.

**RULE 6 — Hedging language:** Claims must use calibrated hedging based on evidence consensus: "broadly supported" (5+ papers agree), "contested" (conflicting papers), "preliminary" (single preprint only).

**RULE 7 — Entity Identity Lock:** Before writing any factual sentence, verify that the evidence row describes the *exact* entity the user asked about. If a mismatch is found, flag it explicitly and abstain on entity-specific claims.

**RULE 8 — Parametric Knowledge Prohibition for Entities:** For specific named entities (reactor design, drug compound, algorithm, dataset), the LLM is **forbidden** from using its training knowledge to fill gaps. Only evidence rows count.

**RULE 9 — Entity Consistency Check:** Before finalizing, scan the entire answer and verify no entity properties have been cross-attributed.

**RULE 10 — Python Sandbox Mandate:** For arithmetic, write a fenced Python code block. Assign the final answer to `result`. Print it. The sandbox executes this code and replaces the block with actual output.

**Depth requirements:** The prompt mandates 800-2000 words for survey/comparison queries, specific numbers and metrics extracted from evidence, synthesis of findings across papers (not a sequential list), and identification of research gaps.

**Output format template:**
```
## Overview (2-3 sentence executive summary)
## Detailed Analysis (multiple themed subsections)
### [Subsection 1]
### [Subsection 2]
## Key Findings & Metrics (table or bullet points with exact numbers)
## Research Gaps & Future Directions
## Sources (each cited paper with URL)
## Limitations & Confidence
```

### 12.2 Mathematical Computation — LaTeX + Python Sandbox

**File:** `backend/generation/math_sandbox.py`

Scientific queries often require arithmetic computation (efficiency metrics, parameter counts, energy calculations). The LLM is unreliable at arithmetic. The system uses a two-step approach:

**Step 1 — LaTeX formula display:**
The synthesizer writes all mathematical expressions in LaTeX:
- Inline: `$F_1 = 2 \cdot \frac{P \cdot R}{P + R}$`
- Block: `$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$`

**Step 2 — Python sandbox execution:**

When the LLM needs to perform a numerical computation, it writes a Python code block:
```python
# From evidence: "512 documents, 18432 total entities"
documents = 512
total_entities = 18432
result = total_entities / documents
print(f"Entity density = {total_entities} / {documents} = {result:.2f} entities/document")
```

The sandbox executes this code in an isolated subprocess:

```python
# Isolation approach:
# 1. Write code to a tempfile
# 2. Launch subprocess: [sys.executable, tempfile_path]
# 3. Timeout: 8 seconds
# 4. Double-verification: run twice; divergence = failure
# 5. Evidence variable injection: replace LLM-hardcoded values with actual evidence values
# 6. Safe preamble: imports math, numpy, sympy, fractions, decimal, statistics
```

**Evidence variable override:** To prevent the LLM from hallucinating intermediate values, the sandbox extracts numeric variables from the evidence table using regex patterns:
```python
# Pattern 1: "accuracy = 0.95"  or  "accuracy: 0.95"
# Pattern 2: "18432 entities"  or  "512 documents"
# Pattern 3: "accuracy of 94.5%"
```

These extracted values are injected into the sandbox's global namespace, overriding any values the LLM hardcoded. This ensures computations use evidence-grounded numbers.

**Math web verification:** For high-stakes computations, `backend/generation/math_web_verifier.py` optionally calls the WolframAlpha API to cross-verify numerical results.

**Pre-Calculation Audit mandate:** Before any calculation, the system prompt requires the LLM to produce an audit block:
1. List all required variables
2. Cite the evidence row for each variable
3. If any variable is missing → abstain ("Calculation aborted: Variable [X] not in corpus")
4. Dimensional check: verify unit consistency
5. First-principles validation: "Is this result physically possible?"

### 12.3 Streaming via Server-Sent Events (SSE)

The synthesis response is streamed token-by-token to the frontend using **Server-Sent Events** (SSE).

**SSE format:**
```
data: {"type": "synthesis_token", "token": "Low-rank"}
data: {"type": "synthesis_token", "token": " adaptation"}
data: {"type": "answer_complete", "answer_id": "abc123"}
```

**Full set of SSE event types:**
```
question_decomposition  — compound query detected, sub-questions sent
intent                  — query intent classification result
currentIntent           — intent name for UI display
question                — query analysis complete
recall                  — candidate retrieval complete (count)
evidence                — evidence table built (rows, confidence, metadata)
synthesis_token         — one streamed token of the answer
math_results            — updated text after math sandbox execution
regenerating            — quality insufficient; regeneration attempt N
entity_warning          — post-synthesis entity mismatch detected
answer_complete         — final verified answer; citations JSON attached
```

The streaming approach means the user sees the answer being written in real time (latency feels lower) while heavy post-processing (verification, entity check) runs after the token stream completes.

---

## 13. Stage 9 — Post-Synthesis Verification

Four independent verification passes run after synthesis completes.

### 13.1 Citation Validation & NLI Entailment

**File:** `backend/generation/verifier.py`

**Citation validation:**
1. Extract all `[CIT:evidence_id]` tags from the generated text via regex
2. Check each `evidence_id` against the evidence table's row IDs
3. Hallucinated citations (IDs not in the evidence table) → removed and logged as warnings
4. Replace valid `[CIT:id]` → `[N]` (numbered in first-appearance order)

**NLI entailment check:**
The `cross-encoder/nli-deberta-v3-base` model (when available) checks whether each cited passage *entails* the claim it's attached to.

```
Hypothesis: The sentence containing [CIT:N] (after removing the citation tag)
Premise:    The chunk_text of evidence row N
Result:     entailment / neutral / contradiction
Threshold:  NLI_ENTAILMENT_THRESHOLD = 0.55
```

Claims where the passage score falls below threshold are flagged in `warnings`.

**Retracted paper detection:**
Evidence rows with `is_retracted=True` are tracked. Any citation to a retracted paper triggers a warning in the response.

**Uncited claim detection:**
The `_count_uncited_claims()` function applies regex to find sentences that:
- Contain percentages (`\b\d+\.?\d*\s*%`)
- Reference metric values (`\b\d+\.\d+\s+accuracy`)
- Use strong claim language (`showed that`, `outperform`, `significantly`)
- Have no `[N]` citation tag

If >2 such sentences are found, a warning is added.

### 13.2 Entity Identity Verification

**File:** `backend/generation/entity_verifier.py`

A post-synthesis LLM check (fast model, temp=0.0) reads the final answer and verifies:
1. Does the answer actually discuss the entity the user asked about?
2. Has any entity substitution occurred (properties of entity B attributed to entity A)?

**Output:** `EntityVerificationResult(entity_correct, substituted_entity, confidence, issues)`

**Trigger for action:** If `entity_correct=False` and `confidence > ENTITY_VERIFY_CONFIDENCE_THRESHOLD` (0.70), a warning prefix is prepended to the answer:
```
⚠️ ENTITY MISMATCH DETECTED: This answer may discuss [substituted_entity] 
rather than the requested [primary_subject]. Please verify against the cited sources.
```

An `entity_warning` SSE event is also sent so the frontend can display a banner.

### 13.3 Coverage Verification & Gap-Fill

**File:** `backend/generation/coverage_verifier.py`

For compound queries, the coverage verifier audits whether the synthesized answer addresses each sub-question.

**LLM audit:** The fast model classifies each sub-question as:
- `FULLY` — completely answered with evidence
- `PARTIAL` — mentioned but not fully developed
- `MISSING` — not addressed at all

**Gap-fill pipeline:**
For `MISSING` sub-questions:
1. Targeted Tavily re-search using the sub-question as query
2. Additional Semantic Scholar search
3. Mini-synthesis: dedicated LLM call for just this sub-question
4. New evidence rows merged into `evidence_table`
5. Gap-fill answer appended to the main answer with a `### [Sub-question]` heading

For `PARTIAL` sub-questions:
1. Lightweight supplement search (fewer results)
2. Shorter synthesis (2-3 paragraphs)
3. Appended as a supplement section

**Why this matters:** Without coverage verification, compound queries reliably under-serve their later sub-questions. The first sub-question dominates retrieval and synthesis; later ones get superficial treatment. The gap-fill loop ensures every sub-question receives dedicated evidence.

### 13.4 Self-Evaluation + Conditional Regeneration Loop

**File:** `backend/generation/self_evaluator.py`

After synthesis (and after all verification passes), the fast LLM evaluates the answer quality on 8 dimensions:

```python
@dataclass
class QualityScore:
    factual_support:   float    # 0-1: are claims grounded in evidence?
    citation_density:  float    # 0-1: are factual sentences cited?
    depth:             float    # 0-1: comprehensive vs. superficial?
    organization:      float    # 0-1: logical structure?
    math_correctness:  float    # 0-1: no arithmetic errors?
    entity_accuracy:   float    # 0-1: correct entity discussed?
    hedging_quality:   float    # 0-1: appropriate uncertainty expressions?
    completeness:      float    # 0-1: all sub-questions addressed?
    composite:         float    # weighted average
    overall:           str      # "excellent" | "good" | "acceptable" | "poor"
    needs_regeneration: bool
    issues:            list[str]
```

**Regeneration loop:**
```python
MAX_REGENERATION_ATTEMPTS = 2
while regen_attempt <= MAX_REGENERATION_ATTEMPTS:
    quality_score = await evaluate_answer(query, full_text, sources, groq)
    
    if not quality_score.needs_regeneration or regen_attempt >= MAX_REGENERATION_ATTEMPTS:
        break
    
    # Build issue hint from identified problems
    issue_hint = "\n\nPREVIOUS ATTEMPT ISSUES (fix these):\n" + "\n".join(quality_score.issues)
    
    # Patch task plan with the issue hint
    patched_plan = TaskPlan(..., reasoning=task_plan.reasoning + issue_hint)
    
    # Re-synthesize with targeted corrections
    full_text = await synthesize(query, intent, evidence_table, patched_plan, ...)
    regen_attempt += 1
```

The client is notified via `regenerating` SSE events so the UI can show "Improving answer quality...".

---

## 14. Stage 10 — Citation Rendering & Response Delivery

**File:** `backend/citation/renderer.py`

**Citation hover cards:** When the frontend renders a `[N]` citation, it fetches the corresponding evidence row and displays a hover card with:
- Paper title
- Authors
- Year, venue
- Abstract preview
- Clickable source URL / PDF URL
- Peer-review status badge
- Citation count

**Citation resolver (`backend/citation/resolver.py`):** Resolves `evidence_id` → full paper metadata from the database. Supports both local papers (by `paper_id`) and web-retrieved papers (by URL).

**Answer persistence:** The final answer is stored in the `answers` table:
```sql
INSERT INTO answers (
    answer_id, query, intent, markdown_text, citations_json,
    is_abstention, abstention_reason, uncertainty_flags,
    total_sources, peer_reviewed_count, preprint_count
)
```

The `evidence_tables` table also receives the full evidence table for the `/api/evidence` audit endpoint.

---

## 15. API Layer — All Endpoints

**Backend root:** `backend/main.py` (FastAPI lifespan, dependency injection)

### POST /api/chat
**File:** `backend/api/routes/chat.py`

The main chat endpoint. Accepts `ChatRequest`, returns `StreamingResponse` (SSE).

```python
class ChatRequest(BaseModel):
    query:           str
    conversation_id: str        # UUID hex (auto-generated if omitted)
    corpus_id:       str = "default"
    recency_filter:  str = "any"  # "any" | "1y" | "3y"
    intent_override: Optional[str] = None
```

Returns `text/event-stream` with the full pipeline SSE events.

### POST /api/ingest/upload
**File:** `backend/api/routes/ingest.py`

Accepts multipart form upload of a PDF file. Triggers the full ingestion pipeline (parse → chunk → embed → index) and returns `{paper_id, title, chunk_count, status}`.

Also supports: `POST /api/ingest/text` (plain text ingestion), `POST /api/ingest/rebuild-indexes` (full index rebuild).

### GET /api/papers/search
**File:** `backend/api/routes/papers.py`

Searches the papers table by title, abstract, or author keywords. Returns paginated paper metadata.

### GET /api/papers/{paper_id}
Returns full metadata for a single paper, including all chunks.

### GET /api/evidence/{answer_id}
**File:** `backend/api/routes/evidence.py`

Returns the full evidence table for a past answer — the audit endpoint. Used by the frontend's "Evidence Explorer" panel to show which papers supported each answer.

### GET /api/evidence/{answer_id}/audit
Returns a detailed audit report: citation count, entity decisions, NLI scores, any warnings.

### GET /health
Returns `{"status": "ok", "models_loaded": [...]}` — used for uptime monitoring.

---

## 16. Frontend Architecture

**Directory:** `frontend/src/`

### Component Tree

```
App.tsx
├── ChatPanel
│   ├── QueryInput           — text area + recency filter + submit
│   ├── MessageList          — scrollable conversation history
│   │   ├── UserMessage
│   │   └── AssistantMessage
│   │       ├── MarkdownRenderer  — renders Markdown + LaTeX
│   │       ├── CitationTag       — [N] clickable badges
│   │       └── CitationCard      — hover card with paper metadata
│   └── StreamingIndicator   — "Thinking..." / "Retrieving..." / "Writing..."
└── EvidencePanel
    ├── EvidenceTable        — displays evidence rows (paper, section, relevance)
    └── EvidenceRow          — individual evidence item with preview
```

### State Management (Zustand)

**Chat store:**
```typescript
interface ChatStore {
    conversations:   Conversation[]
    currentConversation: string | null
    messages:        Message[]
    isStreaming:     boolean
    streamingTokens: string
    intent:          string | null
    evidenceRows:    EvidenceRow[]
    confidence:      number
    // Actions
    sendMessage:     (query: string) => void
    appendToken:     (token: string) => void
    setEvidence:     (rows: EvidenceRow[]) => void
}
```

**Evidence store:** Holds the full evidence table for the Evidence Explorer panel.

### SSE Streaming Hook (`frontend/src/hooks/useSSE.ts`)

```typescript
// Connects to /api/chat via EventSource
// Parses each event type and dispatches to Zustand store
const useChat = () => {
    const response = await fetch('/api/chat', {
        method: 'POST',
        body: JSON.stringify(request),
        headers: { 'Content-Type': 'application/json' }
    });
    
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        // Parse "data: {...}\n\n" events and dispatch
        parseSSEChunk(chunk);
    }
};
```

### Citation Rendering

Citation tags `[N]` in the Markdown are intercepted by the renderer and converted to clickable badges. On click/hover, the corresponding evidence row is fetched from the evidence store and displayed in a hover card.

---

## 17. Database Schema — Complete

**File:** `backend/indexing/metadata_store.py`  
**Engine:** SQLite 3 with WAL mode

```sql
-- Core paper metadata
CREATE TABLE papers (
    paper_id       TEXT PRIMARY KEY,    -- SHA256 hash of title+authors or DOI
    title          TEXT NOT NULL,
    authors        TEXT NOT NULL,       -- JSON: ["First Last", ...]
    year           INTEGER,
    venue          TEXT,
    doi            TEXT,
    arxiv_id       TEXT,
    openalex_id    TEXT,
    abstract       TEXT,
    sections       TEXT,               -- JSON: {"Introduction": "text", ...}
    references_json TEXT,              -- JSON: [{"title": ..., "year": ...}, ...]
    is_peer_reviewed INTEGER DEFAULT 0,
    is_retracted   INTEGER DEFAULT 0,
    citation_count INTEGER DEFAULT 0,
    source_url     TEXT,
    pdf_url        TEXT,
    pdf_path       TEXT,
    ingested_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Text chunks with embeddings
CREATE TABLE chunks (
    chunk_id     TEXT PRIMARY KEY,      -- UUID
    paper_id     TEXT NOT NULL,
    granularity  TEXT NOT NULL,         -- document|section|passage|claim|table
    section_tag  TEXT,                  -- results|methods|abstract|etc.
    text         TEXT NOT NULL,
    token_count  INTEGER,
    embedding    BLOB,                  -- numpy float32 array, binary serialized
    start_char   INTEGER,
    end_char     INTEGER,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);
CREATE INDEX idx_chunks_paper ON chunks(paper_id);
CREATE INDEX idx_chunks_gran  ON chunks(granularity);

-- Citation edges for the graph
CREATE TABLE citation_edges (
    source_paper_id TEXT NOT NULL,
    target_paper_id TEXT NOT NULL,
    PRIMARY KEY (source_paper_id, target_paper_id)
);

-- Conversation sessions
CREATE TABLE conversations (
    conversation_id TEXT PRIMARY KEY,
    title           TEXT,
    corpus_id       TEXT DEFAULT 'default',
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Individual messages (user + assistant)
CREATE TABLE messages (
    message_id      TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,      -- "user" | "assistant"
    content         TEXT NOT NULL,
    answer_id       TEXT,               -- links to answers table
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

-- Evidence tables (for audit)
CREATE TABLE evidence_tables (
    answer_id        TEXT PRIMARY KEY,
    query            TEXT,
    intent           TEXT,
    rows_json        TEXT,              -- JSON array of EvidenceRow objects
    confidence_score REAL,
    created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Full answers with citation metadata
CREATE TABLE answers (
    answer_id          TEXT PRIMARY KEY,
    query              TEXT,
    intent             TEXT,
    markdown_text      TEXT,
    citations_json     TEXT,            -- JSON: {"N": {paper_id, title, url}}
    is_abstention      INTEGER DEFAULT 0,
    abstention_reason  TEXT,
    uncertainty_flags  TEXT,            -- JSON: list of warning strings
    total_sources      INTEGER,
    peer_reviewed_count INTEGER,
    preprint_count     INTEGER,
    created_at         TEXT DEFAULT CURRENT_TIMESTAMP
);
```

---

## 18. End-to-End Data Flow Trace

This section traces a single query through the entire system with exact function calls and data transformations.

**Example query:** "How does LoRA reduce memory usage during LLM fine-tuning?"

```
User types query → React calls POST /api/chat
    {query: "How does LoRA reduce memory usage during LLM fine-tuning?",
     conversation_id: "a1b2c3", recency_filter: "any"}

── Stage 0: Compound Decomposition ───────────────────────────────────────────
decompose_question(query, groq)
  → LLaMA 3.1 8B: "is_compound=False" (single question)
  → decomposition.sub_questions = ["How does LoRA reduce memory usage..."]

── Stage A: Intent ────────────────────────────────────────────────────────────
classify_intent(query, groq)
  → "method_explanation"

── Stage B: Query Rewriting ───────────────────────────────────────────────────
rewrite_query(query, "method_explanation", groq)
  → dense_query: "Low-Rank Adaptation mechanism reducing GPU memory footprint 
                  during large language model fine-tuning by decomposing weight 
                  matrices into low-rank factors"
  → bm25_query:  "LoRA low-rank adaptation fine-tuning memory GPU parameter 
                  efficient PEFT rank decomposition"
  → acronym_expanded: "LoRA low-rank adaptation reduce memory GPU fine-tuning 
                       large language model LLM parameter efficient"
  → arxiv_queries: ["lora low-rank adaptation fine-tuning", 
                    "parameter efficient fine-tuning memory",
                    "PEFT LoRA transformer memory reduction", ...]
  → entity_profile: {primary_subject: "LoRA", entity_type: "algorithm",
                     entity_aliases: ["Low-Rank Adaptation"],
                     exclusion_entities: ["QLoRA", "Prefix Tuning", "Adapter"],
                     requires_entity_grounding: True, specificity_score: 0.75}
  → hyde_embedding: embed("LoRA (Low-Rank Adaptation) reduces GPU memory by 
                           freezing pre-trained weights and training small matrices 
                           A and B...") → 1024-dim vector

── Stage C: Hybrid Retrieval (parallel) ──────────────────────────────────────
await asyncio.gather(
    bm25.search("LoRA low-rank adaptation fine-tuning memory GPU"),
    dense.search(embed("LoRA mechanism reducing GPU memory...")),
    dense.search(hyde_embedding),
    exa.search("How does LoRA reduce memory during LLM fine-tuning"),
    tavily.search("LoRA low-rank adaptation memory reduction LLM"),
    arxiv.search(["lora low-rank adaptation fine-tuning", ...])
)
  → BM25 returns 500 candidates (ranked by BM25Okapi score)
  → Dense returns 500 candidates (ranked by cosine similarity)
  → HyDE returns 500 candidates (ranked by cosine similarity to HyDE vector)
  → Web returns ~48 results (40 Exa + 8 Tavily)
  → Total pool: ~300 unique chunks after dedup

── Stage D: RRF Fusion ────────────────────────────────────────────────────────
reciprocal_rank_fusion([bm25_results, dense_results, hyde_results], k=60, alpha=0.7)
  → For chunk from LoRA paper methods section:
    rrf_score = 0.3/(60+1) + 0.7/(60+1) + 0.7/(60+3) = 0.00492 + 0.01148 + 0.01087 = 0.02727
  → 300 candidates sorted by rrf_score

── Stage E: Graph Expansion ──────────────────────────────────────────────────
graph_expand(top_25_papers)
  → LoRA paper cites "Intrinsic Dimensionality" paper → add chunks
  → QLoRA paper cites LoRA paper → check if QLoRA chunks are needed
  → 25 additional candidates added

── Stage F: Cross-Encoder Reranking ──────────────────────────────────────────
reranker.rerank(query, 325_candidates, top_k=100)
  → 325 pairs fed to BAAI/bge-reranker-v2-m3 in batches of 64
  → CE score for LoRA methods section: raw=3.21 → sigmoid=0.961
  → Multi-signal final_score:
      CE_normalized: 0.961 × 0.75  = 0.721
      section "methods": +0.08
      year 2021: +0.01
      citation 6000+: +0.06
      topic_alignment: 0.85 × 0.12 = 0.102
      evidence_density: 0.75 × 0.10 = 0.075
      granularity "passage": +0.00
      peer_reviewed: +0.03
      pagerank (0.82 normalized): +0.041
      entity (LoRA found): +0.12
      ─────────────────────────────
      final_score: 1.239  (winner)
  → 100 candidates returned

── Stage G: Compression ──────────────────────────────────────────────────────
compress_chunks(top_100)
  → Each passage compressed to top-3 TF-IDF sentences vs. query
  → "wrong" entity chunks × 0.1 penalty

── Stage H: Evidence Table ────────────────────────────────────────────────────
build_evidence_table(top_25_compressed)
  → EvidenceTable(
        answer_id="d4e5f6",
        total_sources=25,
        confidence_score=0.82,
        rows=[
            EvidenceRow(evidence_id="ev_001", paper_title="LoRA: Low-Rank 
                         Adaptation of Large Language Models", year=2021,
                         chunk_text="LoRA decomposes the weight update matrix 
                         ΔW into two matrices A and B, where ΔW = BA...",
                         is_peer_reviewed=True, citation_count=6000,
                         source_url="https://arxiv.org/abs/2106.09685"),
            ...
        ]
    )

── Stage I: Synthesis ────────────────────────────────────────────────────────
synthesize(query, "method_explanation", evidence_table, task_plan, groq, stream=True)
  → 10-rule system prompt injected
  → Evidence table JSON included in user prompt
  → LLaMA 3.3 70B streams response tokens via Groq API
  → Each token: yield SSE("synthesis_token", {"token": token})
  → "## Overview\nLoRA (Low-Rank Adaptation) [CIT:ev_001] addresses..."
  → Math block detected → sandbox executes → verified result

── Stage J: Post-Synthesis ───────────────────────────────────────────────────
verify_answer(full_text, evidence_table)
  → [CIT:ev_001] → valid → [1]
  → 0 hallucinated citations
  → 1 preprint citation flagged
  → 0 uncited claims

verify_entity_consistency(query, full_text, entity_profile, groq)
  → entity_correct=True, confidence=0.97

quality_score = evaluate_answer(query, full_text, 25, groq)
  → composite=0.89, overall="excellent", needs_regeneration=False

── Stage K: Delivery ─────────────────────────────────────────────────────────
yield SSE("answer_complete", {answer_id, markdown_text, citations_json})
Store in answers table + evidence_tables table
```

**Total latency breakdown (approximate):**
- Query understanding: 1.5-2s (2 LLM calls, fast model)
- Hybrid retrieval: 1.5-3s (parallel, depends on web APIs)
- Reranking: 0.5-2s (GPU vs CPU)
- Synthesis: 5-30s (streaming; visible immediately)
- Post-synthesis: 2-4s (verification passes)
- **Total to first token:** ~5-7 seconds
- **Total to answer complete:** ~15-40 seconds depending on answer length

---

## 19. Performance & Tuning Parameters

### Retrieval Funnel Configuration

The funnel parameters control the trade-off between recall (wider funnel = more candidates considered) and latency (wider funnel = slower).

| Parameter | Default | Effect of Increasing | Effect of Decreasing |
| --- | --- | --- | --- |
| `BM25_TOP_K` | 500 | Higher BM25 recall | Faster, may miss rare matches |
| `DENSE_TOP_K` | 500 | Higher semantic recall | Faster, may miss paraphrase matches |
| `FUSED_TOP_K` | 300 | More candidates for reranker | More reranker computation |
| `RERANKED_TOP_K` | 100 | Reranker sees more candidates | More cross-encoder inference |
| `FINAL_EVIDENCE_TOP_K` | 25 | More context for LLM | Longer prompts, more tokens, higher cost |
| `GRAPH_EXPANSION_LIMIT` | 25 | More citation neighbors | More DB queries |

### Hybrid Weighting

`HYBRID_ALPHA=0.7` (70% dense, 30% BM25). In practice:
- Dense retrieval dominates for conceptual queries
- BM25 is critical for exact technical term matches (model names, dataset names, metric names)
- Optimal alpha varies by domain; 0.7 is empirically strong for NLP/CS papers

### BM25 Tuning

`BM25_K1=1.2, BM25_B=0.85`:
- Reducing k1 (e.g., 0.8) reduces term saturation effects — good for short queries
- Reducing b (e.g., 0.5) reduces length penalty — better for corpora with very variable document lengths

### Reranker Cutoffs

`RERANKER_SCORE_THRESHOLD=0.30, RERANKER_ELBOW_DROP=0.25`:
- Increasing threshold filters more aggressively — may cause abstentions on thin corpora
- Decreasing threshold allows low-quality candidates through — may degrade synthesis quality

### Entity Grounding

`ENTITY_SPECIFICITY_THRESHOLD=0.60`: Queries below this specificity score don't trigger entity grounding (avoids false-positive grounding on general queries).

`ENTITY_GROUNDING_PENALTY=0.20`: Hard penalty for wrong-entity chunks. High enough to consistently exclude them but not so high that co-occurrence chunks (0.4 entity score) are unfairly excluded.

---

## 20. Security Considerations

### Python Sandbox Isolation

The math sandbox runs user-controlled code (LLM-generated Python) in a subprocess. Key isolation measures:

1. **No `exec()` or `eval()`**: Code runs as a standalone Python file via `subprocess.run([sys.executable, tempfile.name])`
2. **Timeout enforcement**: 8-second hard timeout via `subprocess.run(timeout=8)`
3. **Safe preamble only**: Sandbox imports only `math`, `fractions`, `decimal`, `statistics`, `numpy`, `sympy`
4. **No project code access**: Empty `PYTHONPATH` prevents importing backend modules
5. **Double-verification**: Code runs twice; divergent results are rejected
6. **Max code length**: `_MAX_CODE_CHARS=4000` prevents denial-of-service via enormous code blocks

### API Key Management

All API keys (`GROQ_API_KEY`, `EXA_API_KEY`, `TAVILY_API_KEY`, `S2_API_KEY`) are loaded from `.env` files, never hardcoded. The `Settings` frozen dataclass loads them at startup; they are never serialized to responses.

### Input Validation

FastAPI + Pydantic validate all request bodies before they reach business logic. SQL injection is prevented by SQLite parameterized queries (`?` placeholders) throughout the metadata store.

### CORS

Configured to allow only `FRONTEND_URL` (default: `http://localhost:5173`) — prevents unauthorized cross-origin access in production.

### Retracted Paper Disclosure

Papers flagged `is_retracted=True` by Semantic Scholar are never silently used. Every such citation triggers a visible warning in the response and is surfaced in the `/api/evidence/{answer_id}/audit` endpoint.

---

*This document covers the complete NexusScholar architecture as of 2026-04-07. Total system: ~9,100 lines of Python, 51 configuration parameters, 10 pipeline stages, 5+ retrieval lanes, 4 post-synthesis verification passes.*
