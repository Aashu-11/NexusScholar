"""
claim_extractor.py — Extract claim-bearing sentences from scientific text.
Uses lexical indicators + structural heuristics to identify sentences
that make falsifiable, citeable claims (metrics, comparisons, findings).
"""

from __future__ import annotations
import re

# Lexical indicators that a sentence contains a claim
CLAIM_INDICATORS = re.compile(
    r"(?i)"
    r"(\d+\.?\d*\s*%)"            # percentages: "3.5%"
    r"|(outperform)"               # comparison claims
    r"|(achiev)"                   # "achieves", "achieved"
    r"|(state[\s-]*of[\s-]*the[\s-]*art)"
    r"|(significant(?:ly)?)"
    r"|(improv)"                   # "improves", "improvement"
    r"|(compar)"                   # "compared", "comparable"
    r"|(result(?:s|ed|ing)?)"
    r"|(demonstrate[sd]?)"
    r"|(\bshown?\b)"
    r"|(\bfound\b)"
    r"|(\bbetter\b)"
    r"|(\bworse\b)"
    r"|(increas)"
    r"|(decreas)"
    r"|(novel)"
    r"|(propos)"                   # "proposed", "proposes"
    r"|(efficien)"                 # "efficient", "efficiency"
    r"|(surpass)"
    r"|(exceed)"
    r"|(baseline)"
    r"|(benchmark)"
    r"|(F1[\s-]*score)"
    r"|(accuracy)"
    r"|(precision)"
    r"|(recall\b)"
    r"|(BLEU)"
    r"|(ROUGE)"
    r"|(perplexity)"
    r"|(\bp[\s<]=?\s*0\.\d+)"     # p-values: "p < 0.05"
    r"|(\bCI\b)"                   # confidence intervals
)

# Minimum sentence length to consider (avoids fragments)
MIN_CLAIM_LENGTH = 40

# Maximum number of claims to extract per section
MAX_CLAIMS_PER_SECTION = 50


def extract_claims(text: str) -> list[str]:
    """
    Extract claim-bearing sentences from a section of text.
    Returns sentences that contain scientific claim indicators.
    """
    sentences = _split_sentences(text)
    claims: list[str] = []

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < MIN_CLAIM_LENGTH:
            continue
        if CLAIM_INDICATORS.search(sent):
            claims.append(sent)
            if len(claims) >= MAX_CLAIMS_PER_SECTION:
                break

    return claims


def extract_claims_with_context(text: str) -> list[dict]:
    """
    Extract claims with their preceding setup sentence for context.
    Returns list of {"claim": str, "context": str, "section_offset": int}.
    """
    sentences = _split_sentences(text)
    results: list[dict] = []

    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if len(sent) < MIN_CLAIM_LENGTH:
            continue
        if CLAIM_INDICATORS.search(sent):
            # Include previous sentence as setup context
            context = sentences[i - 1].strip() if i > 0 else ""
            char_offset = text.find(sent)
            results.append({
                "claim": sent,
                "context": context,
                "section_offset": max(char_offset, 0),
            })
            if len(results) >= MAX_CLAIMS_PER_SECTION:
                break

    return results


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences. Handles common abbreviations
    and citation brackets to avoid false splits.
    """
    # Protect common abbreviations
    text = re.sub(r"(et al)\.", r"\1<DOT>", text)
    text = re.sub(r"(Fig|Eq|Sec|Tab|Ref|Vol|No|pp|vs)\.", r"\1<DOT>", text, flags=re.I)
    text = re.sub(r"(\d)\.", r"\1<DOT>", text)  # "3.5" → protect
    text = re.sub(r"(i\.e)\.", r"\1<DOT>", text)
    text = re.sub(r"(e\.g)\.", r"\1<DOT>", text)

    # Split on sentence-ending punctuation followed by space + uppercase
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\[])', text)

    # Restore dots
    return [s.replace("<DOT>", ".") for s in sentences]