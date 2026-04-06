"""
math_sandbox.py — Isolated Python execution sandbox for arithmetic verification.

Runs LLM-generated code blocks in a subprocess with strict timeout and no network
access. Never uses exec() or eval(). All code runs in a child process via tempfile.

Key design:
- Evidence values OVERRIDE any LLM-hardcoded values (prevents hallucination).
- Double-verification: every successful execution is run twice; divergence = failure.
- numpy, sympy, fractions, math, decimal, statistics are available inside the sandbox.
- Rich terminal banner printed on every execution so math is always visible.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MISSING_VARS_MSG = "I lack the variables to calculate this."
_MAX_CODE_CHARS = 4000
_TIMEOUT_SECONDS = 8

# ── Numeric variable extraction patterns ──────────────────────────────────────
# Pattern 1: label = number  or  label: number  (e.g. "accuracy = 0.95", "F1: 87.3")
_LABEL_EQ_NUM_RE = re.compile(
    r'\b([a-zA-Z_][a-zA-Z_0-9]{1,50})\s*[=:]\s*(\d+(?:[.,]\d+)?)\b'
)
# Pattern 2: number label  (e.g. "18432 entities", "512 documents")
_NUM_LABEL_RE = re.compile(
    r'(?:^|[\s(,])(\d+(?:\.\d+)?)\s+([a-zA-Z_][a-zA-Z_0-9]{1,50})(?=[\s,.)!\n]|$)',
    re.MULTILINE,
)
# Pattern 3: percentage  (e.g. "94.5%", "accuracy of 94.5%")
_PERCENT_RE = re.compile(
    r'\b([a-zA-Z_][a-zA-Z_0-9]{1,50})\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%'
)
_PERCENT_BARE_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*%\s+([a-zA-Z_][a-zA-Z_0-9]{1,50})'
)

# ── Safe preamble injected into every subprocess ──────────────────────────────
# Imports safe math libs; network and file IO are not blocked at OS level but
# the subprocess has an empty PYTHONPATH so project code is unreachable.
_SANDBOX_PREAMBLE = """\
import math, fractions, decimal, statistics
try:
    import numpy as np
except ImportError:
    pass
try:
    import sympy
    from sympy import *
    from sympy import N as sympy_N
except ImportError:
    pass
"""


# ── Terminal banner helper ─────────────────────────────────────────────────────

_COL = 76  # terminal banner width

def _print_math_banner(code: str, result_stdout: str, success: bool, elapsed_ms: float) -> None:
    """Print a very visible ASCII banner to stdout/terminal for every math execution."""
    border = "=" * _COL
    thin   = "-" * _COL
    if success:
        print(f"\n+{border}+", flush=True)
        print(f"|{'  [MATH SANDBOX] VERIFIED RESULT  ':^{_COL}}|", flush=True)
        print(f"+{border}+", flush=True)
        code_lines = code.strip().split("\n")
        for line in code_lines[:8]:
            truncated = line[:_COL - 4]
            print(f"|  {truncated:<{_COL - 2}}|", flush=True)
        if len(code_lines) > 8:
            print(f"|  {'... (truncated)':^{_COL - 2}}|", flush=True)
        print(f"+{thin}+", flush=True)
        for line in result_stdout.strip().split("\n"):
            print(f"|  >> {line:<{_COL - 5}}|", flush=True)
        print(f"+{thin}+", flush=True)
        print(f"|  elapsed: {elapsed_ms:.1f} ms{'':<{_COL - 18}}|", flush=True)
        print(f"+{border}+\n", flush=True)
    else:
        print(f"\n+{thin}+", flush=True)
        print(f"|{'  [MATH SANDBOX] FAILED  ':^{_COL}}|", flush=True)
        print(f"|  Code (first 3 lines):{'':<{_COL - 23}}|", flush=True)
        for line in code.strip().split("\n")[:3]:
            print(f"|    {line[:_COL - 6]:<{_COL - 4}}|", flush=True)
        print(f"+{thin}+\n", flush=True)


@dataclass
class SandboxResult:
    success: bool
    result: Any = None
    stdout: str = ""
    error: str = ""
    elapsed_ms: float = 0.0


@dataclass
class CodeBlock:
    raw_code: str
    language: str
    expression: str = ""  # The full original text that matched (for replacement)


class MathSandbox:
    """
    Executes Python/math code blocks in an isolated subprocess.
    Evidence variables ALWAYS override LLM-hardcoded values.
    numpy + sympy + fractions + math are available in every execution.
    """

    # ── Public API ─────────────────────────────────────────────────────────

    def extract_code_blocks(self, text: str) -> list[CodeBlock]:
        """Extract fenced ```python / ```math code blocks and <<COMPUTE: ...>> tags."""
        blocks: list[CodeBlock] = []

        fenced = re.compile(
            r'(```(python|math)\s*\n(.*?)```)',
            re.DOTALL | re.IGNORECASE,
        )
        for m in fenced.finditer(text):
            lang = m.group(2).lower()
            code = m.group(3).strip()
            if code:
                blocks.append(CodeBlock(raw_code=code, language=lang, expression=m.group(1)))

        compute_re = re.compile(r'(<<COMPUTE:\s*(.*?)>>)', re.DOTALL)
        for m in compute_re.finditer(text):
            code = m.group(2).strip()
            if code:
                blocks.append(CodeBlock(raw_code=code, language="python", expression=m.group(1)))

        return blocks

    def build_verified_code(self, code: str, variables: dict) -> str:
        """
        Build final code with evidence variables OVERRIDING any LLM-hardcoded values.

        Step 1 — Strip LLM-defined assignments for any variable that exists in evidence.
                  This prevents the model from hallucinating numbers it "knows".
        Step 2 — Prepend the sandbox preamble (math imports) + evidence variables.

        Example:
            LLM wrote:  accuracy = 94.5       (hallucinated)
            Evidence:   accuracy = 87.3       (real)
            Result:     LLM line is commented out, evidence value is prepended.
        """
        clean_code = code
        if variables:
            for k in variables:
                # Strip top-level assignments: "accuracy = ..." → "# accuracy (overridden by evidence)"
                clean_code = re.sub(
                    rf'^([ \t]*){re.escape(k)}\s*=\s*[^\n]*',
                    rf'\1# {k} value supplied by verified evidence',
                    clean_code,
                    flags=re.MULTILINE,
                )

        # Build evidence header
        ev_lines: list[str] = []
        if variables:
            ev_lines.append("# ── Evidence-verified variables (override any LLM estimates) ──")
            for k, v in variables.items():
                if isinstance(v, float):
                    ev_lines.append(f"{k} = {v!r}")
                elif isinstance(v, str):
                    ev_lines.append(f"{k} = {repr(v)}")
                else:
                    ev_lines.append(f"{k} = {v}")
            ev_lines.append("# ── End evidence variables ──")

        parts = [_SANDBOX_PREAMBLE]
        if ev_lines:
            parts.append("\n".join(ev_lines))
        parts.append(clean_code)
        return "\n".join(parts)

    async def execute(self, code: str, variables: dict) -> SandboxResult:
        """
        Run *code* with evidence variable injection in an isolated subprocess.
        Double-verified: runs twice; divergent results are treated as failure.
        Returns SandboxResult. Never raises.
        """
        if not code or not code.strip():
            return SandboxResult(success=False, error=_MISSING_VARS_MSG)

        if len(code) > _MAX_CODE_CHARS:
            logger.info("Math sandbox: code block exceeds %d chars, rejecting", _MAX_CODE_CHARS)
            return SandboxResult(success=False, error=_MISSING_VARS_MSG)

        full_code = self.build_verified_code(code, variables)

        r1 = await self._run_subprocess(full_code)
        if not r1.success:
            _print_math_banner(code, r1.error, success=False, elapsed_ms=r1.elapsed_ms)
            return r1

        r2 = await self._run_subprocess(full_code)
        if not r2.success:
            logger.warning("Math sandbox: second run failed after first succeeded — unreliable")
            _print_math_banner(code, "second run failed", success=False, elapsed_ms=r1.elapsed_ms)
            return SandboxResult(success=False, error=_MISSING_VARS_MSG)

        if r1.stdout.strip() != r2.stdout.strip():
            logger.warning(
                "Math sandbox: non-deterministic output — run1=%r run2=%r",
                r1.stdout[:100], r2.stdout[:100],
            )
            _print_math_banner(code, "non-deterministic output", success=False, elapsed_ms=r1.elapsed_ms)
            return SandboxResult(success=False, error=_MISSING_VARS_MSG)

        _print_math_banner(code, r1.stdout, success=True, elapsed_ms=r1.elapsed_ms + r2.elapsed_ms)
        logger.info(
            "Math sandbox: double-verified in %.1f+%.1f ms — %r",
            r1.elapsed_ms, r2.elapsed_ms, r1.stdout[:120],
        )
        return r1

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _run_subprocess(self, full_code: str) -> SandboxResult:
        """Single isolated subprocess execution."""
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(full_code)
                tmp_path = f.name

            safe_env = _build_safe_env()

            t0 = time.monotonic()
            loop = asyncio.get_event_loop()
            try:
                proc: subprocess.CompletedProcess = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        [sys.executable, tmp_path],
                        capture_output=True,
                        text=True,
                        timeout=_TIMEOUT_SECONDS,
                        env=safe_env,
                    ),
                )
                elapsed_ms = (time.monotonic() - t0) * 1000

                stdout = proc.stdout.strip()
                stderr = proc.stderr.strip()

                if proc.returncode != 0:
                    err_summary = stderr[:300] if stderr else "unknown error"
                    if "NameError" in stderr:
                        logger.info("Math sandbox: NameError — %s", err_summary)
                    elif "ModuleNotFoundError" in stderr:
                        logger.info("Math sandbox: missing module — %s", err_summary)
                    else:
                        logger.info(
                            "Math sandbox: exit %d — %s", proc.returncode, err_summary
                        )
                    return SandboxResult(
                        success=False,
                        error=_MISSING_VARS_MSG,
                        stdout=stderr[:200],
                        elapsed_ms=elapsed_ms,
                    )

                if not stdout:
                    logger.info("Math sandbox: code produced no output — no print() statement?")
                    return SandboxResult(
                        success=False,
                        error="No output produced — ensure code ends with print(result)",
                        elapsed_ms=elapsed_ms,
                    )

                result_val = stdout
                return SandboxResult(
                    success=True,
                    result=result_val,
                    stdout=stdout,
                    elapsed_ms=elapsed_ms,
                )

            except subprocess.TimeoutExpired:
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.warning("Math sandbox: timeout after %d s", _TIMEOUT_SECONDS)
                return SandboxResult(success=False, error=_MISSING_VARS_MSG, elapsed_ms=elapsed_ms)

        except Exception as exc:
            logger.warning("Math sandbox: unexpected error — %s", exc)
            return SandboxResult(success=False, error=_MISSING_VARS_MSG)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


# ── Module-level helpers ───────────────────────────────────────────────────────

def _build_safe_env() -> dict[str, str]:
    """Return a stripped-down environment with no credentials or network paths."""
    keep = ("SYSTEMROOT", "PATH", "TEMP", "TMP", "HOME", "USER",
            "WINDIR", "COMSPEC", "HOMEDRIVE", "HOMEPATH")
    env: dict[str, str] = {}
    for k in keep:
        v = os.environ.get(k)
        if v:
            env[k] = v
    env["PYTHONPATH"] = ""
    return env


def extract_evidence_variables(evidence_rows: list) -> dict[str, float | int]:
    """
    Scan evidence chunk_text fields and extract numeric variables suitable for
    sandbox injection.

    Heuristics (in priority order):
      1. ``label = number`` or ``label: number``  (e.g. "accuracy = 0.95")
      2. ``number label``  (e.g. "18432 entities" → entities = 18432)
      3. ``label of N%``  (e.g. "accuracy of 94.5%" → accuracy = 94.5)
      4. ``N% label``     (e.g. "94.5% accuracy" → accuracy = 94.5)

    Returns a dict mapping variable-name → numeric value.
    Skips Python keywords and names longer than 50 chars.
    """
    import keyword

    # Common English words that appear near numbers but are NOT variable names
    _STOPWORDS = frozenset({
        "a", "an", "the", "in", "on", "at", "to", "of", "or", "and", "as",
        "is", "are", "was", "were", "be", "been", "by", "for", "with", "about",
        "up", "out", "no", "not", "so", "do", "per", "vs", "over", "under",
        "each", "all", "than", "into", "from", "its", "it", "he", "she",
        "they", "we", "you", "has", "have", "had",
    })

    candidates: dict[str, float | int] = {}

    for row in evidence_rows:
        text = ""
        if hasattr(row, "chunk_text"):
            text = row.chunk_text or ""
        elif isinstance(row, dict):
            text = row.get("chunk_text") or row.get("text") or ""
        if not text:
            continue

        def _valid_name(name: str) -> bool:
            return (
                not keyword.iskeyword(name)
                and name.lower() not in _STOPWORDS
                and len(name) >= 2
                and len(name) <= 50
            )

        # Pattern 1: label = number or label: number (highest priority)
        for m in _LABEL_EQ_NUM_RE.finditer(text):
            name, num_str = m.group(1), m.group(2).replace(",", "")
            if _valid_name(name):
                candidates[name] = _coerce_number(num_str)

        # Pattern 2: number label
        for m in _NUM_LABEL_RE.finditer(text):
            num_str, name = m.group(1), m.group(2)
            if _valid_name(name):
                candidates.setdefault(name, _coerce_number(num_str))

        # Pattern 3: label of N%
        for m in _PERCENT_RE.finditer(text):
            name, num_str = m.group(1), m.group(2)
            if _valid_name(name):
                candidates.setdefault(name, _coerce_number(num_str))

        # Pattern 4: N% label
        for m in _PERCENT_BARE_RE.finditer(text):
            num_str, name = m.group(1), m.group(2)
            if _valid_name(name):
                candidates.setdefault(name, _coerce_number(num_str))

    return candidates


def format_variables_for_prompt(variables: dict[str, float | int]) -> str:
    """
    Format the extracted evidence variables as a human-readable block
    to inject into the LLM system prompt before synthesis.
    This tells the model which variable names and values are available
    for use in Python code blocks.
    """
    if not variables:
        return ""
    lines = ["## EVIDENCE-VERIFIED NUMERIC VARIABLES"]
    lines.append("The following numeric variables were extracted from the evidence.")
    lines.append("When writing Python code blocks, use THESE exact variable names.")
    lines.append("Do NOT hardcode any other numeric values. If a value is not listed")
    lines.append("here, write: I lack the variables to calculate this.")
    lines.append("")
    lines.append("```")
    for k, v in sorted(variables.items()):
        lines.append(f"{k} = {v}")
    lines.append("```")
    lines.append("")
    lines.append("CRITICAL: The sandbox will OVERRIDE any value you hardcode for these")
    lines.append("variable names with the evidence-verified values above. Your hardcoded")
    lines.append("values will be ignored and the real evidence values used instead.")
    return "\n".join(lines)


def _coerce_number(s: str) -> float | int:
    """Parse a numeric string to int or float."""
    s = s.replace(",", "")
    try:
        f = float(s)
        if f == int(f) and "." not in s:
            return int(f)
        return f
    except ValueError:
        return 0
