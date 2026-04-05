"""
entity_verifier.py — Post-synthesis entity consistency verification.

Runs after the LLM generates an answer to check whether entity substitution
occurred (e.g., the answer discusses TRIGA when user asked about RBMK).
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from backend.generation.groq_client import GroqClient
from backend.retrieval.entity_extractor import QueryEntityProfile

logger = logging.getLogger(__name__)

ENTITY_VERIFY_PROMPT = """You are checking whether a research answer correctly addresses the specific entity the user asked about.

USER QUERY: {query}
PRIMARY ENTITY REQUESTED: {primary_subject}
ENTITY TYPE: {entity_type}
THESE ARE SIMILAR BUT DIFFERENT ENTITIES (NOT what was asked): {exclusion_entities}

ANSWER TO VERIFY:
{answer}

CHECK:
1. Does the answer's primary technical subject match "{primary_subject}"?
2. Are there sentences that attribute properties of {exclusion_entities} to "{primary_subject}"?
3. Does the answer use parametric/training knowledge about similar-but-different entities to fill gaps?

Respond ONLY with JSON:
{{
  "entity_correct": <true if answer discusses the right entity, false if wrong entity used>,
  "substituted_entity": "<which wrong entity was used as the primary source, or null>",
  "confidence": <0.0-1.0 confidence in your assessment>,
  "issues": ["list of specific entity problems found, empty if none"]
}}"""


@dataclass
class EntityVerificationResult:
    entity_correct: bool = True
    substituted_entity: Optional[str] = None
    confidence: float = 0.0
    issues: list[str] = field(default_factory=list)
    requires_regeneration: bool = False


async def verify_entity_consistency(
    query: str,
    answer: str,
    entity_profile: QueryEntityProfile,
    groq: GroqClient,
) -> EntityVerificationResult:
    """
    Verify that the synthesized answer is about the correct entity.

    Never throws — returns entity_correct=True (safe default) on failure.

    Args:
        query: The original user query.
        answer: The synthesized answer text.
        entity_profile: The extracted entity profile from the query.
        groq: The GroqClient instance.

    Returns:
        EntityVerificationResult with verification details.
    """
    if not entity_profile.primary_subject or not entity_profile.requires_entity_grounding:
        return EntityVerificationResult(entity_correct=True, confidence=0.0)

    try:
        exclusion_str = (
            ", ".join(entity_profile.exclusion_entities)
            if entity_profile.exclusion_entities
            else "none specified"
        )
        prompt = ENTITY_VERIFY_PROMPT.format(
            query=query,
            primary_subject=entity_profile.primary_subject,
            entity_type=entity_profile.entity_type,
            exclusion_entities=exclusion_str,
            answer=answer[:3000],  # Truncate for token budget
        )

        response = await groq.complete_fast(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        raw = response.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        entity_correct = bool(parsed.get("entity_correct", True))
        substituted = parsed.get("substituted_entity") or None
        confidence = float(parsed.get("confidence", 0.0))
        issues = list(parsed.get("issues", []))

        # Only require regeneration for high-confidence severe contamination
        requires_regen = (
            not entity_correct
            and confidence > 0.8
            and substituted is not None
        )

        result = EntityVerificationResult(
            entity_correct=entity_correct,
            substituted_entity=substituted,
            confidence=confidence,
            issues=issues,
            requires_regeneration=requires_regen,
        )

        logger.info(
            "Entity verification: correct=%s substituted=%r confidence=%.2f issues=%s",
            result.entity_correct,
            result.substituted_entity,
            result.confidence,
            result.issues,
        )
        return result

    except Exception as exc:
        logger.warning("Entity verification failed: %s — assuming entity correct", exc)
        return EntityVerificationResult(entity_correct=True, confidence=0.0)
