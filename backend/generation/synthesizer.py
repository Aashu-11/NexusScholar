"""
synthesizer.py — Answer synthesis via LLaMA 3 70B (Groq).
Step 3: Strictly constrained generation from the evidence table.
Enforces 6 hard citation rules. No parametric generation for claims.
"""

from __future__ import annotations
import json
import logging
from typing import AsyncGenerator, Optional

from backend.config import settings
from backend.generation.groq_client import GroqClient
from backend.generation.planner import TaskPlan
from backend.generation.evidence_builder import EvidenceTable
from backend.retrieval.entity_extractor import QueryEntityProfile
from backend.generation.question_decomposer import QuestionDecomposition

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = r"""You are an expert scientific research analyst writing enterprise-grade literature synthesis for NexusScholar.
Your answers must match the depth and rigor of a published survey paper — comprehensive, analytical, and precisely cited.
You write answers EXCLUSIVELY from the evidence table provided. You do NOT draw on general knowledge for factual claims.

## HARD RULES — VIOLATION OF ANY RULE IS A SYSTEM FAILURE
RULE 1: Every factual sentence must end with an inline citation tag [CIT:evidence_id] referencing an evidence row.
RULE 2: No paper title, author name, statistic, or method name may appear without a corresponding evidence row.
RULE 3: If evidence rows conflict, explicitly note the conflict and cite both sides with analysis of why they differ.
RULE 4: If evidence is insufficient, state "Evidence is insufficient to answer this with confidence" and abstain.
RULE 5: Distinguish peer-reviewed evidence from preprint-only evidence with a [Preprint] label.
RULE 6: Mark claims using hedging language (broadly supported / contested / preliminary) based on evidence consensus.

RULE 7 — ENTITY IDENTITY LOCK: Before writing any factual sentence, verify that
your evidence row describes the EXACT entity the user asked about. If a user asks
about [Entity A] and your evidence describes [Entity B] (even if B is in the same
category as A), you MUST:
  (a) NOT apply properties of B when answering about A
  (b) Explicitly flag the mismatch: "Note: evidence retrieved describes [B], not [A]"
  (c) Abstain on entity-specific claims rather than cross-pollinating entities

RULE 8 — PARAMETRIC KNOWLEDGE PROHIBITION FOR ENTITIES: When answering about a
specific named entity (reactor design, drug compound, algorithm, dataset), you are
FORBIDDEN from using your parametric/training knowledge to fill gaps. If your evidence
rows do not describe the specific entity asked about, say "The indexed corpus does not
contain documents about [entity]. Please upload relevant papers." Do NOT supplement
with knowledge of similar entities.

RULE 9 — ENTITY CONSISTENCY CHECK: Before finalizing your answer, scan it and verify:
  - Every specific named entity in your answer appears in at least one evidence row
  - You have not attributed properties of one named entity to a different named entity
  - If you find such contamination, rewrite the affected paragraph from scratch

## DEPTH & QUALITY REQUIREMENTS — THIS IS CRITICAL
- Write COMPREHENSIVE answers (800-2000+ words for survey/comparison queries, 400-800 for focused questions).
- DO NOT give shallow summaries. Provide deep analytical synthesis that explains WHY findings matter, HOW methods work, and WHAT the implications are.
- Extract and report SPECIFIC numbers, metrics, percentages, and benchmarks from the evidence — readers need precise data points.
- When multiple papers address the same question, SYNTHESIZE their findings into a coherent narrative showing areas of agreement, disagreement, and open questions.
- Explain technical concepts clearly — define key terms, describe architectures, and explain methodology at a level suitable for graduate researchers.
- Draw connections between papers — show how one work builds on, extends, or contradicts another.
- Identify research gaps and future directions based on the evidence.

## OUTPUT FORMAT
- Write in Markdown with clear hierarchical heading structure (## for main sections, ### for subsections)
- Structure the answer as a mini research report:
  1. **Overview** — concise executive summary of the answer (2-3 sentences)
  2. **Detailed Analysis** — deep dive into the evidence organized thematically (multiple subsections)
  3. **Key Findings & Metrics** — specific numerical results, benchmarks, performance comparisons
  4. **Research Gaps & Future Directions** — what the evidence doesn't cover
  5. **## Sources** — each cited paper with title, authors, year, venue, source_url
  6. **## Limitations & Confidence** — evidence gaps, conflicts, and confidence assessment
- Use [CIT:evidence_id] inline citation tags
- For comparison queries, produce a detailed Markdown table with columns: Method | Architecture/Approach | Key Results | Strengths | Limitations | Source
- For survey queries, organize by themes/approaches rather than listing papers sequentially
- When source_url is available in the evidence table, include it in the Sources section as a clickable link
- Include quantitative data wherever available — accuracy scores, F1, BLEU, ROUGE, perplexity, etc.

## TABLE FORMATTING — CRITICAL
When producing comparison tables:
1. ALWAYS include a header row with column names
2. ALWAYS include a separator row with dashes (| --- | --- |)
3. EVERY row must have the EXACT same number of columns as the header
4. Use LEFT-aligned text in all cells
5. Keep cell content concise — max 60 characters per cell
6. For long method names, use abbreviations with full name in first mention
7. NEVER break a table row across multiple lines
8. Format numbers consistently: 2 decimal places for percentages (94.50%), 1 for scores (0.8)
9. For comparison queries with 3+ methods, a table is MANDATORY

Example of a CORRECT table:
| Method | Dataset | Accuracy | F1 Score | Year |
| --- | --- | --- | --- | --- |
| BERT-base | GLUE | 79.60% | 0.803 | 2018 |
| RoBERTa | GLUE | 88.50% | 0.891 | 2019 |

## CHAIN-OF-ARITHMETIC — MANDATORY FOR ANY NUMERICAL QUERY
When the query requires a calculated value (density, ratio, average, total, percentage, etc.) that is NOT
already written in the evidence as a final answer, you MUST show explicit step-by-step arithmetic:

  **Step 1 — Extract:** identify the raw integers / quantities from the evidence row.
  **Step 2 — Formula:** state the formula, e.g. $\text{density} = \frac{\text{entities}}{\text{documents}}$
  **Step 3 — Substitute:** plug in the extracted numbers.
  **Step 4 — Compute:** show the arithmetic and state the result.

Do NOT search for a pre-written conclusion in the text if one is not present — derive it yourself.
Any arithmetic errors are a SYSTEM FAILURE equivalent to a citation violation.

## MATHEMATICAL NOTATION — LaTeX
- Render ALL mathematical expressions using LaTeX syntax so the UI can display them properly.
- Inline math: $...$ (e.g., $F_1 = 2 \cdot \frac{P \cdot R}{P + R}$)
- Display/block math: $$...$$ on its own line for equations that deserve emphasis.
- Named equations: use `\text{}` for subscript labels, e.g. $\text{Precision} = \frac{TP}{TP+FP}$
- For well-known formulas (Black-Scholes, Bayes' theorem, attention mechanism, softmax, etc.),
  ALWAYS write the canonical formula in LaTeX, then explain each term.
- Numerical results that are fractions or ratios should be shown as both fraction and decimal:
  e.g. $\frac{18432}{512} = 36$ entities/document.
"""

USER_PROMPT_TEMPLATE = """## TASK PLAN
{task_plan}

## EVIDENCE TABLE (JSON)
{evidence_table}

## USER QUERY
{query}

## QUERY INTENT
{intent}

## INSTRUCTIONS
Write a COMPREHENSIVE, DEEP, enterprise-grade research synthesis following all rules above.

CRITICAL REQUIREMENTS:
1. Read EVERY evidence row carefully. Extract ALL relevant information — specific numbers, methods, results, architectures, and conclusions.
2. Write AT LEAST 800 words for survey/comparison/trend queries. Be thorough and analytical.
3. Cite EVERY factual claim with [CIT:evidence_id] tags. No exceptions.
4. For each paper/method discussed, explain: (a) what it does, (b) how it works, (c) what results it achieves, (d) its significance.
5. SYNTHESIZE across sources — don't just summarize each paper individually. Show connections, agreements, and disagreements.
6. Include a structured comparison table if 3+ methods/papers are being compared.
7. Report ALL quantitative results (accuracy, F1, BLEU, perplexity, etc.) found in the evidence.
8. End with substantive analysis of research gaps and future directions.
9. ARITHMETIC: If the query asks for a computed quantity (ratio, density, average, etc.) that is not
   directly stated in the evidence, derive it step-by-step (extract → formula → substitute → compute).
   Show the LaTeX formula and the numerical result. Never leave a calculable value unanswered.
10. MATH NOTATION: Use $...$ for inline LaTeX and $$...$$ for display equations. Include canonical
    formulas for any named mathematical model discussed (e.g. attention, softmax, Black-Scholes, etc.).

DO NOT write a shallow or brief answer. The user expects the depth and rigor of a published literature review."""

# Injected when a compound question was detected — forces per-sub-question sections
COMPOUND_QUESTION_SECTION = """
## COMPOUND QUESTION — MANDATORY STRUCTURE
This query contains {count} distinct sub-questions. You MUST address EVERY sub-question
with a dedicated top-level section (##). Do NOT collapse them into a single narrative.

Sub-questions to answer:
{numbered_sub_questions}

STRUCTURAL RULES:
- Open with a 2-3 sentence **Overview** section that frames all sub-questions together.
- Then dedicate a separate ## section to EACH sub-question (use the sub-question text as
  the heading, verbatim or lightly paraphrased for readability).
- Within each section, follow all standard depth and citation rules.
- After the per-question sections, include a ## Cross-Cutting Insights section that draws
  connections, common themes, or contrasts ACROSS the sub-questions.
- End with ## Sources and ## Limitations & Confidence as normal.
- Every sub-question section must cite at least one evidence row. If evidence is
  insufficient for a specific sub-question, explicitly state that in that section.
- Minimum length per sub-question section: 200 words.
"""

ENTITY_GROUNDING_SECTION = """
## ENTITY GROUNDING CONSTRAINTS
Primary subject of this query: {primary_subject}
This query is NOT about: {exclusion_entities}
Entity type: {entity_type}

CRITICAL: Only use evidence rows that discuss "{primary_subject}".
If evidence rows discuss {exclusion_entities}, do not use their specific technical
details — those are different entities with different properties.
"""



# ── TPM budget constants ────────────────────────────────────────────────────
# Groq on-demand: llama-3.3-70b-versatile = 12,000 TPM (input + max_output).
# Use 11,400 as the hard ceiling (600-token safety margin).
_GROQ_TPM_LIMIT = 11_400
_CHARS_PER_TOKEN = 3   # conservative for mixed markdown/JSON/technical text


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _trim_evidence_rows(rows: list[dict], char_budget: int) -> list[dict]:
    """
    Progressively shrink evidence rows to fit within char_budget.
    Trims the 'text' and 'parent_context' fields of each row; other metadata
    (title, authors, year, url, evidence_id) is always preserved in full.
    """
    result: list[dict] = []
    remaining = char_budget
    for row in rows:
        row = dict(row)
        text = row.get("text") or ""
        parent = row.get("parent_context") or ""
        overhead = len(json.dumps({k: v for k, v in row.items() if k not in ("text", "parent_context")}))
        available = remaining - overhead - 10
        if available <= 0:
            break
        text_budget = min(len(text), int(available * 0.67))
        parent_budget = min(len(parent), available - text_budget)
        row["text"] = text[:text_budget]
        row["parent_context"] = parent[:parent_budget] if parent else None
        remaining -= len(json.dumps(row)) + 2
        result.append(row)
        if remaining <= 0:
            break
    return result


async def synthesize(
    query: str,
    intent: str,
    evidence_table: EvidenceTable,
    task_plan: TaskPlan,
    groq: GroqClient,
    stream: bool = True,
    entity_profile: Optional[QueryEntityProfile] = None,
    decomposition: Optional[QuestionDecomposition] = None,
) -> str | AsyncGenerator[str, None]:
    """
    Generate a citation-grounded answer from the evidence table.
    Returns streaming token generator or complete text.
    """
    # Abstention check
    if task_plan.should_abstain:
        text = _generate_abstention(query, evidence_table)
        if stream:
            async def _yield():
                yield text
            return _yield()
        return text

    # Build entity grounding section if needed
    entity_grounding = ""
    if entity_profile and entity_profile.requires_entity_grounding and entity_profile.primary_subject:
        exclusion_str = (
            ", ".join(entity_profile.exclusion_entities)
            if entity_profile.exclusion_entities
            else "none specified"
        )
        entity_grounding = ENTITY_GROUNDING_SECTION.format(
            primary_subject=entity_profile.primary_subject,
            exclusion_entities=exclusion_str,
            entity_type=entity_profile.entity_type,
        )

    max_output_tokens = 4096
    all_rows = evidence_table.to_llm_context()

    # Build structural sections (compound question + entity grounding)
    compound_section = ""
    if decomposition and decomposition.is_compound and len(decomposition.sub_questions) >= 2:
        numbered = "\n".join(
            f"  {i}. {sq}" for i, sq in enumerate(decomposition.sub_questions, start=1)
        )
        compound_section = COMPOUND_QUESTION_SECTION.format(
            count=len(decomposition.sub_questions),
            numbered_sub_questions=numbered,
        )
        logger.info("Compound synthesis: injecting %d sub-questions into prompt", len(decomposition.sub_questions))

    def _build_user_prompt(rows: list[dict]) -> str:
        p = USER_PROMPT_TEMPLATE.format(
            task_plan=task_plan.reasoning,
            evidence_table=json.dumps(rows, indent=2),
            query=query,
            intent=intent,
        )
        if compound_section:
            p = compound_section + "\n" + p
        if entity_grounding:
            p = entity_grounding + "\n" + p
        return p

    # ── Adaptive evidence fitting ──────────────────────────────────────────
    # Trims ONLY the evidence JSON to keep total (input + max_output) within
    # the TPM limit.  System prompt, instructions, and max_output are untouched.
    user_prompt = _build_user_prompt(all_rows)
    full_input = SYSTEM_PROMPT + user_prompt
    estimated_total = _estimate_tokens(full_input) + max_output_tokens

    if estimated_total > _GROQ_TPM_LIMIT:
        # How many chars do we need to free up from the evidence JSON?
        excess_tokens = estimated_total - _GROQ_TPM_LIMIT
        # Add 25 % margin because char/token estimation isn't perfect
        chars_to_cut = int(excess_tokens * _CHARS_PER_TOKEN * 1.25)
        evidence_chars_current = len(json.dumps(all_rows, indent=2))
        new_evidence_char_budget = max(4000, evidence_chars_current - chars_to_cut)
        trimmed_rows = _trim_evidence_rows(all_rows, new_evidence_char_budget)
        user_prompt = _build_user_prompt(trimmed_rows)
        full_input = SYSTEM_PROMPT + user_prompt
        estimated_total = _estimate_tokens(full_input) + max_output_tokens
        logger.info(
            "Evidence fitting: %d→%d rows, estimated total %d tokens (limit=%d)",
            len(all_rows), len(trimmed_rows), estimated_total, _GROQ_TPM_LIMIT,
        )
    else:
        logger.info(
            "Synthesis prompt: %d rows, estimated %d tokens (limit=%d)",
            len(all_rows), estimated_total, _GROQ_TPM_LIMIT,
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    return await groq.complete_primary(
        messages=messages,
        temperature=settings.SYNTHESIS_TEMPERATURE,
        max_tokens=max_output_tokens,
        stream=stream,
    )


def _generate_abstention(query: str, evidence_table: EvidenceTable) -> str:
    """Structured abstention card when evidence is insufficient."""
    related = set()
    for row in evidence_table.rows[:5]:
        related.add(row.paper_title[:80])
    suggestions = list(related)[:3]
    suggestion_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "- No closely related papers found."

    return f"""## Insufficient Evidence

I was unable to find sufficient evidence in the indexed literature to answer this query with confidence.

**Query:** {query}

**Retrieval confidence:** {evidence_table.confidence_score:.2f} (below threshold of {settings.CONFIDENCE_THRESHOLD})

### Closest Related Topics
{suggestion_text}

### Suggested Reformulations
- Try narrowing your query to a specific method or dataset
- Try searching for a specific paper by title or author
- Try broadening the topic area

*NexusScholar abstains rather than generating an unsupported answer.*
"""