"""
table_extractor.py — Extract and serialize structured tables from markdown text.
Tables from PDF parsers (Marker, Grobid) are preserved as searchable chunks.
"""
from __future__ import annotations
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_tables_from_markdown(
    markdown_text: str,
    section_tag: str = "results",
) -> list[dict]:
    """
    Find all markdown tables and return them as structured dicts.
    Each table becomes a separate indexable chunk with rich text serialization.
    """
    tables = []
    lines = markdown_text.split("\n")
    i = 0

    while i < len(lines):
        # Detect table start: line with pipes
        if _is_table_row(lines[i]):
            table_lines = []
            while i < len(lines) and (_is_table_row(lines[i]) or _is_separator_row(lines[i])):
                table_lines.append(lines[i])
                i += 1

            parsed = _parse_table(table_lines)
            if parsed and parsed.get("headers") and parsed.get("rows"):
                # Serialize to searchable text
                text = _serialize_table(parsed, section_tag)
                if len(text.strip()) > 20:
                    tables.append({
                        "headers": parsed["headers"],
                        "rows": parsed["rows"],
                        "text": text,
                        "section_tag": section_tag,
                        "row_count": len(parsed["rows"]),
                        "col_count": len(parsed["headers"]),
                    })
        else:
            i += 1

    logger.info("Extracted %s tables from section '%s'", len(tables), section_tag)
    return tables


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _is_separator_row(line: str) -> bool:
    return bool(re.match(r"^\|[\s\-:|]+\|$", line.strip()))


def _parse_table(lines: list[str]) -> Optional[dict]:
    rows = []
    headers = None
    for line in lines:
        if _is_separator_row(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if headers is None:
            headers = cells
        else:
            rows.append(cells)
    if not headers:
        return None
    return {"headers": headers, "rows": rows}


def _serialize_table(table: dict, section_tag: str) -> str:
    """
    Convert a parsed table into a retrieval-friendly text representation.
    Format: "Table from {section}: {headers}. Row 1: {col}: {val}, {col}: {val}. ..."
    This format allows the BM25/dense index to match both column names and values.
    """
    headers = table["headers"]
    rows = table["rows"]

    parts = [f"Table from {section_tag} section. Columns: {', '.join(headers)}."]

    for i, row in enumerate(rows[:20]):  # cap at 20 rows per table chunk
        pairs = []
        for h, v in zip(headers, row):
            if v and v.strip() and v.strip() != "-":
                pairs.append(f"{h}: {v.strip()}")
        if pairs:
            parts.append(f"Row {i+1}: {', '.join(pairs)}.")

    return " ".join(parts)
