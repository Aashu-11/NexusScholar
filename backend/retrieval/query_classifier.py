"""
query_classifier.py — Groq-powered intent classification.
Classifies each query into one of 10 intent types.
"""

from __future__ import annotations
import json
import logging

from backend.generation.groq_client import GroqClient

logger = logging.getLogger(__name__)

INTENT_TYPES = [
    "literature_survey",
    "benchmark_comparison",
    "method_explanation",
    "paper_lookup",
    "trend_analysis",
    "dataset_discovery",
    "author_search",
    "definition",
    "contradiction_check",
    "general",
]

CLASSIFICATION_PROMPT = """You are a query intent classifier for a scientific research platform.
Classify the user query into exactly ONE of these intent types:

{intents}

CLASSIFICATION GUIDELINES:
- "literature_survey": broad questions about a topic area, "what are...", "overview of...", research landscape questions
- "benchmark_comparison": comparing methods/models, "which is better", performance comparisons, leaderboards
- "method_explanation": "how does X work", mechanism questions, architectural details
- "paper_lookup": looking for a specific paper by name, DOI, or unique identifier
- "trend_analysis": questions about research trends over time, "recent advances", evolution of techniques
- "dataset_discovery": looking for datasets, corpora, data resources
- "author_search": looking for an author's work
- "definition": "what is X", terminology questions
- "contradiction_check": conflicting findings, disagreements between papers
- "general": anything else

Respond with ONLY a JSON object:
{{"intent": "<intent_type>", "reasoning": "<one sentence>"}}

User query: "{query}"
"""


async def classify_intent(query: str, groq: GroqClient) -> str:
    """Classify a user query into one of the 10 intent types."""
    try:
        logger.info("Intent classification started: %r", query)
        prompt = CLASSIFICATION_PROMPT.format(
            intents="\n".join(f"- {i}" for i in INTENT_TYPES),
            query=query,
        )
        response = await groq.complete_fast(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        parsed = json.loads(response.strip())
        intent = parsed.get("intent", "general")
        logger.info("Intent classification raw response: %s", response.strip())
        if intent in INTENT_TYPES:
            logger.info("Intent classification result: %s", intent)
            return intent
        return "general"
    except Exception as e:
        logger.warning("Intent classification failed: %s", e)
        return "general"
