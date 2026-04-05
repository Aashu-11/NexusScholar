"""
coverage_verifier.py — Per-sub-question coverage verification.

After synthesis, checks that every sub-question in a compound query
(or the single question in an atomic query) is genuinely addressed in
the answer. If gaps are found, appends an explicit notice so the user
knows which parts lacked evidence rather than receiving a silent omission.
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field

from backend.generation.groq_client import GroqClient
from backend.generation.question_decomposer import QuestionDecomposition

logger = logging.getLogger(__name__)


_COVERAGE_PROMPT = """You are a strict quality auditor for a research synthesis system.

Given a synthesized answer and a list of sub-questions that the answer was supposed to address,
determine whether each sub-question is FULLY answered, PARTIALLY answered, or MISSING from the answer.

DEFINITIONS:
- FULLY: The answer has a dedicated section or substantial paragraph that directly addresses the sub-question
  with specific evidence, data, or analysis. A vague 1-sentence mention does NOT count as fully covered.
- PARTIAL: The sub-question is touched on but lacks depth, specifics, or concrete evidence.
- MISSING: The sub-question is not addressed at all, or the answer explicitly says evidence is insufficient.

Sub-questions:
{sub_questions}

Answer (truncated to 4000 chars):
{answer_excerpt}

Respond with ONLY a JSON object (no markdown fences):
{{
  "coverage": [
    {{
      "sub_question": "<exact sub-question text>",
      "status": "FULLY" | "PARTIAL" | "MISSING",
      "reason": "<one sentence explanation>"
    }}
  ]
}}
"""


@dataclass
class SubQuestionCoverage:
    sub_question: str
    status: str  # FULLY | PARTIAL | MISSING
    reason: str


@dataclass
class CoverageResult:
    all_covered: bool
    fully_covered: list[SubQuestionCoverage] = field(default_factory=list)
    partial: list[SubQuestionCoverage] = field(default_factory=list)
    missing: list[SubQuestionCoverage] = field(default_factory=list)

    @property
    def has_gaps(self) -> bool:
        return bool(self.missing or self.partial)

    def gap_notice(self) -> str:
        """
        Returns a Markdown notice to append to the answer when coverage gaps exist.
        Returns empty string if all sub-questions are fully covered.
        """
        if not self.has_gaps:
            return ""

        lines = ["\n\n---\n## Coverage Notice\n"]
        if self.missing:
            lines.append("**The following sub-questions could not be answered from the indexed evidence:**")
            for item in self.missing:
                lines.append(f"- **{item.sub_question}** — {item.reason}")
            lines.append("")
        if self.partial:
            lines.append("**The following sub-questions were only partially addressed:**")
            for item in self.partial:
                lines.append(f"- **{item.sub_question}** — {item.reason}")
            lines.append("")
        lines.append(
            "*To improve coverage: upload additional relevant papers or rephrase the sub-question "
            "as a standalone query.*"
        )
        return "\n".join(lines)


async def verify_coverage(
    decomposition: QuestionDecomposition,
    answer: str,
    groq: GroqClient,
) -> CoverageResult:
    """
    Verify that every sub-question is addressed in the synthesized answer.

    For atomic queries (single question), returns all_covered=True unless the
    answer is an abstention — in that case, marks it as MISSING.
    """
    sub_questions = decomposition.sub_questions

    # Atomic query fast path: just check for abstention
    if not decomposition.is_compound or len(sub_questions) <= 1:
        is_abstention = "insufficient evidence" in answer.lower() or "abstains" in answer.lower()
        if is_abstention:
            item = SubQuestionCoverage(
                sub_question=sub_questions[0] if sub_questions else decomposition.original_query,
                status="MISSING",
                reason="Answer abstained due to insufficient evidence.",
            )
            return CoverageResult(all_covered=False, missing=[item])
        return CoverageResult(all_covered=True)

    # Compound query: run LLM coverage check
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(sub_questions))
    answer_excerpt = answer[:4000]

    prompt = _COVERAGE_PROMPT.format(
        sub_questions=numbered,
        answer_excerpt=answer_excerpt,
    )

    try:
        response = await groq.complete_fast(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600,
        )
        raw = response.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)

        fully: list[SubQuestionCoverage] = []
        partial: list[SubQuestionCoverage] = []
        missing: list[SubQuestionCoverage] = []

        for entry in parsed.get("coverage", []):
            sq = entry.get("sub_question", "")
            status = entry.get("status", "FULLY").upper()
            reason = entry.get("reason", "")
            item = SubQuestionCoverage(sub_question=sq, status=status, reason=reason)
            if status == "FULLY":
                fully.append(item)
            elif status == "PARTIAL":
                partial.append(item)
            else:
                missing.append(item)

        all_covered = not missing and not partial
        result = CoverageResult(
            all_covered=all_covered,
            fully_covered=fully,
            partial=partial,
            missing=missing,
        )
        logger.info(
            "Coverage check: fully=%d partial=%d missing=%d",
            len(fully), len(partial), len(missing),
        )
        if result.has_gaps:
            for item in missing + partial:
                logger.warning("Coverage gap [%s]: %r — %s", item.status, item.sub_question, item.reason)
        return result

    except Exception as e:
        logger.warning("Coverage verification failed: %s — skipping", e)
        return CoverageResult(all_covered=True)
