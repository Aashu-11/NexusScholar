"""
normalizer.py — Unicode normalization, dehyphenation, ligature expansion, UTF-8 cleanup.
Applied to all text before chunking and indexing.
"""

from __future__ import annotations
import re
import unicodedata


# Common ligatures to expand
LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u0132": "IJ",
    "\u0133": "ij",
    "\u0152": "OE",
    "\u0153": "oe",
    "\u00c6": "AE",
    "\u00e6": "ae",
}

# Characters to strip
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def normalize_text(text: str) -> str:
    """Full normalization pipeline for extracted PDF text."""
    text = _unicode_normalize(text)
    text = _expand_ligatures(text)
    text = _dehyphenate(text)
    text = _strip_control_chars(text)
    text = _collapse_whitespace(text)
    return text.strip()


def _unicode_normalize(text: str) -> str:
    """NFKC normalization: compatibility decomposition + canonical composition."""
    return unicodedata.normalize("NFKC", text)


def _expand_ligatures(text: str) -> str:
    """Replace typographic ligatures with their ASCII equivalents."""
    for lig, expansion in LIGATURE_MAP.items():
        text = text.replace(lig, expansion)
    return text


def _dehyphenate(text: str) -> str:
    """
    Rejoin words broken across line boundaries.
    "multi-\\nline" → "multiline"
    But preserve intentional hyphens like "state-of-the-art"
    """
    # Pattern: word-char, hyphen, newline, lowercase word-char → join
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Also handle soft hyphens
    text = text.replace("\u00ad", "")
    return text


def _strip_control_chars(text: str) -> str:
    """Remove non-printable control characters."""
    return CONTROL_CHARS.sub("", text)


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces and excessive newlines."""
    text = re.sub(r"[ \t]{2,}", " ", text)       # multi-space → single
    text = re.sub(r"\n{3,}", "\n\n", text)        # 3+ newlines → 2
    text = re.sub(r" *\n *", "\n", text)           # spaces around newlines
    return text


def normalize_query(query: str) -> str:
    """Lighter normalization for user queries."""
    query = unicodedata.normalize("NFKC", query)
    query = CONTROL_CHARS.sub("", query)
    query = re.sub(r"\s+", " ", query)
    return query.strip()