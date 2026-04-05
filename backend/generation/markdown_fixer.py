"""
markdown_fixer.py — Post-synthesis markdown table repair and formatting.
Ensures all tables have proper alignment, consistent column counts,
and clean pipe-delimited structure.
"""

import re
from typing import Optional


def fix_markdown_tables(text: str) -> str:
    """Fix all markdown tables in synthesized output."""
    lines = text.split('\n')
    result = []
    table_buffer = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        is_table_line = stripped.startswith('|') and stripped.endswith('|')
        is_separator = bool(re.match(r'^\|[\s\-:|]+\|$', stripped))

        if is_table_line or is_separator:
            if not in_table:
                in_table = True
                table_buffer = []
            table_buffer.append(stripped)
        else:
            if in_table:
                fixed_table = _fix_single_table(table_buffer)
                result.extend(fixed_table)
                result.append('')  # blank line after table
                table_buffer = []
                in_table = False
            result.append(line)

    if in_table and table_buffer:
        fixed_table = _fix_single_table(table_buffer)
        result.extend(fixed_table)

    return '\n'.join(result)


def _fix_single_table(lines: list[str]) -> list[str]:
    """Fix a single markdown table block."""
    if len(lines) < 2:
        return lines

    # Parse cells
    rows = []
    separator_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^\|[\s\-:|]+\|$', line):
            separator_idx = i
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        rows.append(cells)

    if not rows:
        return lines

    # Normalize column count
    max_cols = max(len(row) for row in rows)
    for row in rows:
        while len(row) < max_cols:
            row.append('')

    # Calculate column widths (minimum 3 chars for separator)
    col_widths = []
    for col_idx in range(max_cols):
        max_width = max(
            max((len(row[col_idx]) for row in rows), default=3),
            3
        )
        col_widths.append(max_width)

    # Rebuild table
    result = []
    # Header row
    if rows:
        header = '| ' + ' | '.join(
            rows[0][i].ljust(col_widths[i]) for i in range(max_cols)
        ) + ' |'
        result.append(header)

    # Separator row
    separator = '| ' + ' | '.join(
        '-' * col_widths[i] for i in range(max_cols)
    ) + ' |'
    result.append(separator)

    # Data rows
    for row in rows[1:]:
        data = '| ' + ' | '.join(
            row[i].ljust(col_widths[i]) for i in range(max_cols)
        ) + ' |'
        result.append(data)

    return result


def ensure_table_in_synthesis(text: str) -> str:
    """
    Fix all markdown tables and repair common LLM table mistakes:
    - Missing separator row
    - Extra spaces breaking alignment
    - Inconsistent column counts
    """
    text = fix_markdown_tables(text)

    # Fix tables where LLM forgot the separator row
    lines = text.split('\n')
    fixed_lines = []
    for i, line in enumerate(lines):
        fixed_lines.append(line)
        # If this line looks like a header and next line is data (not separator)
        if (line.strip().startswith('|') and line.strip().endswith('|')
                and i + 1 < len(lines)):
            next_line = lines[i + 1].strip()
            if (next_line.startswith('|') and next_line.endswith('|')
                    and not re.match(r'^\|[\s\-:|]+\|$', next_line)
                    and i > 0
                    and not lines[i - 1].strip().startswith('|')):
                # Insert separator
                cols = line.count('|') - 1
                sep = '| ' + ' | '.join(['---'] * cols) + ' |'
                fixed_lines.append(sep)

    return '\n'.join(fixed_lines)
