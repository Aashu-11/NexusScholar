"""
entity_extractor.py — Named entity extraction for query grounding.

Extracts the primary subject, entity type, aliases, and critically,
EXCLUSION ENTITIES — things similar to but different from the query subject.

This profile is passed through the entire pipeline and used to:
1. Filter out wrong-entity candidates at reranking
2. Make the compressor reject wrong-entity chunks
3. Instruct the synthesizer to not cross-pollinate entities
4. Run post-synthesis entity consistency verification
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from backend.generation.groq_client import GroqClient

logger = logging.getLogger(__name__)

ENTITY_EXTRACTION_PROMPT = """You are a scientific query entity extractor. Your job is to identify the PRIMARY named entity in a research query and its closest look-alikes (exclusion entities) that the query is NOT about.

This is critical for preventing entity substitution hallucinations (e.g., answering about TRIGA reactors when the user asked about RBMK reactors).

User query: "{query}"

RULES:
- primary_subject: The single most specific named entity the query is about. null if query is general.
- entity_type: One of: reactor_type | chemical | drug | organism | technique | material | algorithm | dataset | model | general
- entity_aliases: Other names/abbreviations for the same entity (synonyms only).
- exclusion_entities: Entities in the SAME CATEGORY that are NOT what the user asked about. These are the dangerous look-alikes. Be generous — list all plausible confusables.
- domain: One of: nuclear_physics | chemistry | biology | machine_learning | medicine | materials_science | general
- requires_entity_grounding: true only when specificity_score >= 0.6
- specificity_score: 0.0-1.0 — how specific/named the entity is (0=totally generic, 1=uniquely named artifact)

FEW-SHOT EXAMPLES:

Query: "What is the relationship between the positive void coefficient and RBMK reactor design?"
Output: {{"primary_subject": "RBMK reactor", "entity_type": "reactor_type", "entity_aliases": ["RBMK-1000", "RBMK", "Chernobyl reactor"], "exclusion_entities": ["TRIGA", "PWR", "BWR", "CANDU", "VVER", "pressurized water reactor", "boiling water reactor"], "domain": "nuclear_physics", "requires_entity_grounding": true, "specificity_score": 0.92}}

Query: "How does BERT masked language modeling pretraining work?"
Output: {{"primary_subject": "BERT", "entity_type": "model", "entity_aliases": ["BERT-base", "BERT-large", "bidirectional encoder representations from transformers"], "exclusion_entities": ["RoBERTa", "GPT", "T5", "XLNet", "ALBERT", "DistilBERT"], "domain": "machine_learning", "requires_entity_grounding": true, "specificity_score": 0.88}}

Query: "What are the side effects of aspirin on platelet aggregation?"
Output: {{"primary_subject": "aspirin", "entity_type": "drug", "entity_aliases": ["acetylsalicylic acid", "ASA"], "exclusion_entities": ["ibuprofen", "acetaminophen", "naproxen", "warfarin", "clopidogrel", "paracetamol"], "domain": "medicine", "requires_entity_grounding": true, "specificity_score": 0.85}}

Query: "What are recent advances in transformer architectures?"
Output: {{"primary_subject": null, "entity_type": "general", "entity_aliases": [], "exclusion_entities": [], "domain": "machine_learning", "requires_entity_grounding": false, "specificity_score": 0.15}}

Query: "How does gradient descent optimization work?"
Output: {{"primary_subject": "gradient descent", "entity_type": "technique", "entity_aliases": ["SGD", "stochastic gradient descent", "batch gradient descent"], "exclusion_entities": [], "domain": "machine_learning", "requires_entity_grounding": false, "specificity_score": 0.45}}

Query: "Compare BERT vs RoBERTa on GLUE benchmark"
Output: {{"primary_subject": null, "entity_type": "general", "entity_aliases": [], "exclusion_entities": [], "domain": "machine_learning", "requires_entity_grounding": false, "specificity_score": 0.1}}

Query: "What are the properties of SARS-CoV-2 spike protein binding?"
Output: {{"primary_subject": "SARS-CoV-2", "entity_type": "organism", "entity_aliases": ["COVID-19 virus", "novel coronavirus", "2019-nCoV"], "exclusion_entities": ["SARS-CoV-1", "MERS-CoV", "influenza", "rhinovirus"], "domain": "biology", "requires_entity_grounding": true, "specificity_score": 0.93}}

Query: "Explain how the Superphénix LMFBR reactor design works"
Output: {{"primary_subject": "Superphénix LMFBR", "entity_type": "reactor_type", "entity_aliases": ["Superphénix", "liquid metal fast breeder reactor", "LMFBR"], "exclusion_entities": ["RBMK", "PWR", "BWR", "CANDU", "VVER", "SFR"], "domain": "nuclear_physics", "requires_entity_grounding": true, "specificity_score": 0.95}}

NOW PROCESS:
Query: "{query}"

Respond ONLY with valid JSON. No prose, no markdown fences.
"""


@dataclass
class QueryEntityProfile:
    primary_subject: Optional[str] = None
    entity_type: str = "general"
    entity_aliases: list[str] = field(default_factory=list)
    exclusion_entities: list[str] = field(default_factory=list)
    domain: str = "general"
    requires_entity_grounding: bool = False
    specificity_score: float = 0.0


def _safe_default() -> QueryEntityProfile:
    """Return a safe no-op entity profile."""
    return QueryEntityProfile(
        requires_entity_grounding=False,
        specificity_score=0.0,
    )


async def extract_query_entities(
    query: str,
    groq: GroqClient,
) -> QueryEntityProfile:
    """
    Extract named entity profile from a research query.

    Uses a fast Groq call to identify primary subject, aliases, and exclusion entities.
    Never throws — returns a safe default on any failure.

    Args:
        query: The original user query string.
        groq: The GroqClient instance.

    Returns:
        QueryEntityProfile with extracted entity information.
    """
    try:
        prompt = ENTITY_EXTRACTION_PROMPT.format(query=query)
        response = await groq.complete_fast(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        raw = response.strip()
        # Strip markdown fences if model wraps output
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)

        specificity = float(parsed.get("specificity_score", 0.0))
        requires_grounding = bool(parsed.get("requires_entity_grounding", False))
        # Enforce threshold — only ground when specific enough
        if specificity < 0.6:
            requires_grounding = False

        profile = QueryEntityProfile(
            primary_subject=parsed.get("primary_subject") or None,
            entity_type=parsed.get("entity_type", "general"),
            entity_aliases=list(parsed.get("entity_aliases", [])),
            exclusion_entities=list(parsed.get("exclusion_entities", [])),
            domain=parsed.get("domain", "general"),
            requires_entity_grounding=requires_grounding,
            specificity_score=specificity,
        )

        logger.info(
            "Entity profile extracted: subject=%r type=%s domain=%s "
            "aliases=%s exclusions=%s requires_grounding=%s specificity=%.2f",
            profile.primary_subject,
            profile.entity_type,
            profile.domain,
            profile.entity_aliases,
            profile.exclusion_entities,
            profile.requires_entity_grounding,
            profile.specificity_score,
        )
        return profile

    except Exception as exc:
        logger.warning("Entity extraction failed: %s — using safe default", exc)
        return _safe_default()
