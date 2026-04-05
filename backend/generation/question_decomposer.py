"""
question_decomposer.py — Compound question detection and decomposition.

Stage 0 of the pipeline. Analyzes the incoming query with an LLM to determine
whether it contains multiple embedded sub-questions (e.g. "What is X and how
does it compare to Y? Also explain Z."). If yes, splits them into atomic
sub-questions so the retrieval and synthesis stages can address each one
with proper coverage.

If the query is a single focused question, this stage is a no-op — it returns
the original query as the sole sub-question.
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field

from backend.generation.groq_client import GroqClient

logger = logging.getLogger(__name__)


DECOMPOSITION_PROMPT = """You are an expert at understanding research queries. Your task is to analyze a user's question and determine whether it contains MULTIPLE distinct sub-questions that each require their own answer, or whether it is a single focused question.

A question is COMPOUND if it:
- Contains multiple "?" marks asking different things
- Uses connectors like "also", "additionally", "furthermore", "as well as", "and also" to introduce a different question
- Asks about multiple distinct topics/concepts that each need separate answers
- Has clearly separable parts that could independently be asked as standalone questions

A question is NOT compound (it is atomic) if it:
- Asks about one main topic even if the answer requires covering multiple aspects
- Uses "and" to compare two things (e.g. "compare X and Y" is ONE question)
- Asks for a list or survey of multiple things under a single theme
- Has qualifications or conditions but fundamentally asks one thing

IMPORTANT RULES for decomposition:
- Maximum 5 sub-questions. If there are more, group related ones.
- Each sub-question must be self-contained and answerable independently.
- Preserve ALL specifics from the original (entity names, years, metrics, constraints).
- Do NOT rephrase or simplify — use the user's own language.
- If in doubt, treat as atomic (is_compound: false).

User query: "{query}"

Respond with ONLY a JSON object (no markdown fences, no prose):
{{
  "is_compound": <true|false>,
  "reasoning": "<1-2 sentences explaining your decision>",
  "sub_questions": [
    "<sub-question 1>",
    "<sub-question 2>"
  ]
}}

If is_compound is false, sub_questions must contain exactly one entry: the original query unchanged.

EXAMPLES:

Query: "What are the key differences between BERT and GPT architectures?"
Output: {{"is_compound": false, "reasoning": "Single comparison question about two architectures.", "sub_questions": ["What are the key differences between BERT and GPT architectures?"]}}

Query: "Explain how RAG works. Also, what are the main benchmarks used to evaluate RAG systems?"
Output: {{"is_compound": true, "reasoning": "Two distinct questions: one about RAG mechanics, one about evaluation benchmarks.", "sub_questions": ["How does Retrieval-Augmented Generation (RAG) work?", "What are the main benchmarks used to evaluate RAG systems?"]}}

Query: "What is LoRA? How does it compare to full fine-tuning and prefix tuning? Also, what datasets are commonly used to evaluate PEFT methods?"
Output: {{"is_compound": true, "reasoning": "Three distinct questions: LoRA explanation, comparison with other PEFT methods, and evaluation datasets.", "sub_questions": ["What is LoRA (Low-Rank Adaptation) and how does it work?", "How does LoRA compare to full fine-tuning and prefix tuning?", "What datasets are commonly used to evaluate parameter-efficient fine-tuning (PEFT) methods?"]}}

Query: "What are recent advances in diffusion models for image generation including architectures, training techniques, and evaluation metrics?"
Output: {{"is_compound": false, "reasoning": "Single survey question asking about multiple aspects of one topic (diffusion models). The aspects are facets of the same answer, not separate questions.", "sub_questions": ["What are recent advances in diffusion models for image generation including architectures, training techniques, and evaluation metrics?"]}}
"""


@dataclass
class QuestionDecomposition:
    """Result of compound question detection."""
    original_query: str
    is_compound: bool
    sub_questions: list[str] = field(default_factory=list)
    reasoning: str = ""

    @property
    def primary_query(self) -> str:
        """The main query to use for intent classification and external hydration."""
        return self.original_query

    @property
    def count(self) -> int:
        return len(self.sub_questions)


async def decompose_question(query: str, groq: GroqClient) -> QuestionDecomposition:
    """
    Analyze a query for compound sub-questions using an LLM.

    Returns a QuestionDecomposition. If the query is atomic, sub_questions
    will contain a single entry (the original query). If compound, it will
    contain 2-5 distinct sub-questions.
    """
    # Fast path: if query is short and has no structural compound signals,
    # skip the LLM call and treat as atomic.
    if _is_clearly_atomic(query):
        logger.info("Question decomposer: short/atomic query, skipping LLM call")
        return QuestionDecomposition(
            original_query=query,
            is_compound=False,
            sub_questions=[query],
            reasoning="Short query with no compound signals.",
        )

    try:
        prompt = DECOMPOSITION_PROMPT.format(query=query)
        response = await groq.complete_fast(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
        )
        raw = response.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)

        is_compound = bool(parsed.get("is_compound", False))
        reasoning = parsed.get("reasoning", "")
        sub_questions = parsed.get("sub_questions", [])

        # Validate: must have at least one sub-question
        if not sub_questions or not isinstance(sub_questions, list):
            raise ValueError("LLM returned empty sub_questions list")

        # Sanitize: strip empty strings, cap at 5
        sub_questions = [q.strip() for q in sub_questions if isinstance(q, str) and q.strip()][:5]

        if not sub_questions:
            raise ValueError("All sub-questions were empty after sanitization")

        # Safety check: if LLM says compound but only returned 1 sub-question, downgrade
        if is_compound and len(sub_questions) < 2:
            logger.warning("LLM said compound but returned <2 sub-questions; treating as atomic")
            is_compound = False
            sub_questions = [query]

        result = QuestionDecomposition(
            original_query=query,
            is_compound=is_compound,
            sub_questions=sub_questions,
            reasoning=reasoning,
        )
        logger.info(
            "Question decomposition: is_compound=%s sub_questions=%d reasoning=%r",
            is_compound, len(sub_questions), reasoning,
        )
        return result

    except Exception as e:
        logger.warning("Question decomposition failed: %s — treating as atomic", e)
        return QuestionDecomposition(
            original_query=query,
            is_compound=False,
            sub_questions=[query],
            reasoning="Decomposition unavailable.",
        )


def _is_clearly_atomic(query: str) -> bool:
    """
    Heuristic fast-path: skip LLM call for queries that are very unlikely to
    be compound. Returns True only when the query shows NO compound signals.

    IMPORTANT: compound signal patterns are checked FIRST so that short-but-compound
    queries like "What is X? Also explain Y?" are never wrongly fast-pathed as atomic.
    """
    q = query.strip()

    # Compound signal check runs before ANY length heuristic.
    compound_signals = [
        r"\?\s*[A-Z]",           # two sentences ending with "?" and starting with capital
        r"\?[^?]*\?",            # two or more "?" in the query
        r"\?\s*(also|and also|additionally|furthermore|moreover)\b",
        r"\.\s*(also|additionally|furthermore)\s+\w+\?",
        r"\b(also|additionally|furthermore|moreover)\b.{0,80}\?",
        r"(explain|describe|discuss).+\b(also|additionally)\b\s+(explain|describe|what|how|why)",
    ]
    for pattern in compound_signals:
        if re.search(pattern, q, re.IGNORECASE):
            return False  # send to LLM

    # Very short queries (< 6 words) with no compound signals are atomic
    if len(q.split()) < 6:
        return True

    # Single "?" with ≤ 20 words and no compound signals — very likely atomic
    if q.count("?") <= 1 and len(q.split()) <= 20:
        return True

    return False
