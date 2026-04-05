"""
self_evaluator.py — Post-synthesis answer quality evaluation.
Runs a fast LLM pass to score the answer on multiple dimensions.
Now includes entity consistency and parametric contamination scoring.
"""

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

EVAL_PROMPT = """Rate this research synthesis answer on a 1-5 scale for each criterion.
Be STRICT — a 5 means publishable quality, a 1 means unusable.

QUERY: {query}

ANSWER: {answer}

EVIDENCE SOURCES USED: {source_count}

Rate each criterion (respond ONLY with JSON):
{{
  "completeness": <1-5: Does it address all aspects of the query?>,
  "citation_density": <1-5: Are claims properly cited?>,
  "specificity": <1-5: Does it include specific numbers, metrics, methods?>,
  "coherence": <1-5: Is it well-organized and logical?>,
  "table_quality": <1-5: If tables present, are they well-formatted? 3 if no tables needed>,
  "entity_consistency": <1-5: Does the answer discuss the exact entity asked about, not a similar substitute? 5=perfect match, 1=clearly wrong entity used>,
  "parametric_contamination": <1-5: 5=answer uses only provided evidence, 1=heavily uses training knowledge to fill gaps about entities not in evidence>,
  "overall": <1-5: Overall answer quality>,
  "issues": ["list of specific problems found"]
}}"""


@dataclass
class QualityScore:
    completeness: int = 3
    citation_density: int = 3
    specificity: int = 3
    coherence: int = 3
    table_quality: int = 3
    entity_consistency: int = 3
    parametric_contamination: int = 3
    overall: int = 3
    issues: list = field(default_factory=list)

    @property
    def composite(self) -> float:
        weights = {
            'completeness': 0.20,
            'citation_density': 0.20,
            'specificity': 0.15,
            'coherence': 0.10,
            'table_quality': 0.10,
            'entity_consistency': 0.15,
            'parametric_contamination': 0.10,
        }
        return sum(
            getattr(self, k) * v for k, v in weights.items()
        )

    @property
    def needs_regeneration(self) -> bool:
        return (
            self.overall <= 2
            or self.composite < 2.5
            or self.entity_consistency <= 1  # Hard fail on entity substitution
        )


async def evaluate_answer(
    query: str,
    answer: str,
    source_count: int,
    groq,
) -> QualityScore:
    try:
        prompt = EVAL_PROMPT.format(
            query=query,
            answer=answer[:3000],
            source_count=source_count,
        )
        response = await groq.complete_fast(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        # Extract JSON from response
        raw = response.strip()
        # Handle markdown code blocks
        if '```' in raw:
            import re
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
            if match:
                raw = match.group(1)
        parsed = json.loads(raw)
        return QualityScore(
            completeness=parsed.get("completeness", 3),
            citation_density=parsed.get("citation_density", 3),
            specificity=parsed.get("specificity", 3),
            coherence=parsed.get("coherence", 3),
            table_quality=parsed.get("table_quality", 3),
            entity_consistency=parsed.get("entity_consistency", 3),
            parametric_contamination=parsed.get("parametric_contamination", 3),
            overall=parsed.get("overall", 3),
            issues=parsed.get("issues", []),
        )
    except Exception as e:
        logger.warning("Self-evaluation failed: %s", e)
        return QualityScore()
