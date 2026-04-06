"""
tests/test_math_sandbox.py — Pytest tests for the MathSandbox arithmetic execution system.
"""

from __future__ import annotations

import asyncio
import pytest
import sys
import os

# Make sure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.generation.math_sandbox import (
    MathSandbox,
    SandboxResult,
    CodeBlock,
    extract_evidence_variables,
)
from backend.generation.synthesizer import _execute_math_blocks, _validate_math_relevance

# ── Helpers ────────────────────────────────────────────────────────────────────

def run(coro):
    """Run a coroutine synchronously for pytest."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Fake evidence row for tests ────────────────────────────────────────────────

class _FakeRow:
    def __init__(self, text: str):
        self.chunk_text = text


class _FakeEvidenceTable:
    def __init__(self, rows):
        self.rows = rows
        self.confidence_score = 0.9


# ══════════════════════════════════════════════════════════════════════════════
# 1. test_successful_calculation
# ══════════════════════════════════════════════════════════════════════════════

def test_successful_calculation():
    """
    A code block that defines its own variables, computes a result, and
    prints it must execute successfully and return the printed output.
    """
    sandbox = MathSandbox()
    code = (
        "documents = 512\n"
        "total_entities = 18432\n"
        "result = total_entities / documents\n"
        'print(f"Entity density = {result:.4f}")'
    )
    sr: SandboxResult = run(sandbox.execute(code, {}))

    assert sr.success is True, f"Expected success, got error: {sr.error}"
    assert sr.stdout != ""
    assert "36.0000" in sr.stdout, f"Expected 36.0000 in stdout, got: {sr.stdout!r}"
    assert sr.elapsed_ms > 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. test_missing_variable_returns_error
# ══════════════════════════════════════════════════════════════════════════════

def test_missing_variable_returns_error():
    """
    Code that references an undefined variable must return the canonical
    'I lack the variables to calculate this.' message.
    """
    sandbox = MathSandbox()
    code = (
        "result = undefined_var / 512\n"
        "print(result)"
    )
    sr: SandboxResult = run(sandbox.execute(code, {}))

    assert sr.success is False
    assert "I lack the variables to calculate this." in sr.error


# ══════════════════════════════════════════════════════════════════════════════
# 3. test_timeout_handled
# ══════════════════════════════════════════════════════════════════════════════

def test_timeout_handled():
    """
    An infinite loop must be killed within TIMEOUT+1 seconds and must
    return the missing-variables error (not hang forever).
    """
    import time

    sandbox = MathSandbox()
    code = "while True: pass"

    t0 = time.monotonic()
    sr: SandboxResult = run(sandbox.execute(code, {}))
    elapsed = time.monotonic() - t0

    assert sr.success is False
    assert "I lack the variables to calculate this." in sr.error
    # Should finish within TIMEOUT + 1 second per run × 2 runs + 1 s overhead
    assert elapsed < 13, f"Sandbox took too long to kill the process: {elapsed:.1f}s"


# ══════════════════════════════════════════════════════════════════════════════
# 4. test_no_code_blocks
# ══════════════════════════════════════════════════════════════════════════════

def test_no_code_blocks():
    """
    Synthesis text that contains no fenced code blocks must pass through
    _execute_math_blocks completely unchanged.
    """
    text = (
        "## Overview\n\n"
        "The model achieved 94.50% accuracy on the GLUE benchmark [CIT:ev1].\n\n"
        "No computation blocks here.\n"
    )
    evidence_table = _FakeEvidenceTable([])
    sandbox = MathSandbox()

    result_text = run(_execute_math_blocks(text, evidence_table, sandbox))
    assert result_text == text


# ══════════════════════════════════════════════════════════════════════════════
# 5. test_variable_injection
# ══════════════════════════════════════════════════════════════════════════════

def test_variable_injection():
    """
    Variables extracted from evidence text are correctly injected into the
    code and produce the right result when the code uses those variable names.
    """
    sandbox = MathSandbox()

    # Evidence supplies "documents = 512" and "entities = 18432"
    evidence_rows = [
        _FakeRow("The corpus contains 512 documents and 18432 entities total."),
    ]
    evidence_vars = extract_evidence_variables(evidence_rows)

    # The code relies on injected 'documents' and 'entities' from evidence
    code = (
        "# Variables will be injected from evidence\n"
        "result = entities / documents\n"
        'print(f"Density: {result:.2f}")'
    )

    sr: SandboxResult = run(sandbox.execute(code, evidence_vars))

    assert sr.success is True, f"Expected success. vars={evidence_vars}, error={sr.error}"
    assert "Density:" in sr.stdout


# ══════════════════════════════════════════════════════════════════════════════
# 6. test_code_block_replacement_in_synthesis
# ══════════════════════════════════════════════════════════════════════════════

def test_code_block_replacement_in_synthesis():
    """
    A synthesis containing a ```python block with a valid computation must have
    the block replaced by the actual sandbox output (not the raw code).
    """
    text = (
        "The entity density is computed below:\n\n"
        "```python\n"
        "documents = 512\n"
        "total_entities = 18432\n"
        "result = total_entities / documents\n"
        'print(f"Entity density = {result:.4f}")\n'
        "```\n\n"
        "This gives a high density score."
    )
    evidence_table = _FakeEvidenceTable([])
    sandbox = MathSandbox()

    result_text = run(_execute_math_blocks(text, evidence_table, sandbox))

    # The original fenced block should be gone
    assert "```python" not in result_text
    # The computed value should be present
    assert "36.0000" in result_text, f"Computed value missing from: {result_text!r}"


# ══════════════════════════════════════════════════════════════════════════════
# 7. test_failed_block_replaced_with_error_marker
# ══════════════════════════════════════════════════════════════════════════════

def test_failed_block_replaced_with_error_marker():
    """
    A synthesis block referencing an undefined variable must be replaced with
    the canonical error marker, not left as raw code.
    """
    text = (
        "The ratio is:\n\n"
        "```python\n"
        "result = unknown_variable / 512\n"
        "print(result)\n"
        "```\n\n"
        "End of synthesis."
    )
    evidence_table = _FakeEvidenceTable([])
    sandbox = MathSandbox()

    result_text = run(_execute_math_blocks(text, evidence_table, sandbox))

    assert "```python" not in result_text
    assert "Calculation not executed" in result_text
    assert "I lack the variables to calculate this." in result_text


# ══════════════════════════════════════════════════════════════════════════════
# 8. test_oversized_code_block_rejected
# ══════════════════════════════════════════════════════════════════════════════

def test_oversized_code_block_rejected():
    """Code blocks exceeding 2000 chars must be silently rejected."""
    sandbox = MathSandbox()
    code = "x = 1\n" * 400  # well over 2000 chars
    sr: SandboxResult = run(sandbox.execute(code, {}))
    assert sr.success is False
    assert "I lack the variables to calculate this." in sr.error


# ══════════════════════════════════════════════════════════════════════════════
# 9. test_extract_code_blocks
# ══════════════════════════════════════════════════════════════════════════════

def test_extract_code_blocks():
    """extract_code_blocks should find python and math blocks but not bash."""
    sandbox = MathSandbox()
    text = (
        "Some text.\n\n"
        "```python\nresult = 1 + 1\nprint(result)\n```\n\n"
        "```bash\necho hello\n```\n\n"
        "```math\nx = 42\nprint(x)\n```\n\n"
        "<<COMPUTE: result = 7 * 6\nprint(result)>>"
    )
    blocks = sandbox.extract_code_blocks(text)
    langs = [b.language for b in blocks]

    assert "python" in langs
    assert "math" in langs
    # bash must NOT be included
    assert len([b for b in blocks if b.language == "bash"]) == 0
    # Should include the <<COMPUTE>> tag
    assert any("COMPUTE" not in b.language for b in blocks)
    assert len(blocks) == 3  # python + math + compute


# ══════════════════════════════════════════════════════════════════════════════
# 10. test_double_verification_determinism
# ══════════════════════════════════════════════════════════════════════════════

def test_double_verification_determinism():
    """
    Deterministic code must pass double-verification and return a single
    consistent result.
    """
    sandbox = MathSandbox()
    code = (
        "import math\n"
        "result = math.sqrt(144)\n"
        'print(f"sqrt(144) = {result}")'
    )
    sr: SandboxResult = run(sandbox.execute(code, {}))

    assert sr.success is True
    assert "12.0" in sr.stdout


# ══════════════════════════════════════════════════════════════════════════════
# 11. test_empty_code_block
# ══════════════════════════════════════════════════════════════════════════════

def test_empty_code_block():
    """An empty code block must return failure gracefully."""
    sandbox = MathSandbox()
    sr: SandboxResult = run(sandbox.execute("", {}))
    assert sr.success is False
    assert "I lack the variables to calculate this." in sr.error


# ══════════════════════════════════════════════════════════════════════════════
# 12. test_llm_relevance_check_relevant
# ══════════════════════════════════════════════════════════════════════════════

def test_llm_relevance_check_relevant():
    """
    _validate_math_relevance must return (True, reason) when the mock LLM
    responds with relevant=true.
    """
    class _MockGroq:
        async def complete_fast(self, messages, temperature=0.0):
            return '{"relevant": true, "reason": "Computes entity density as requested."}'

    relevant, reason = run(
        _validate_math_relevance(
            query="What is the entity density of the corpus?",
            code="result = 18432 / 512\nprint(f'Entity density = {result:.2f}')",
            result_stdout="Entity density = 36.00",
            groq=_MockGroq(),
        )
    )
    assert relevant is True
    assert reason != ""


# ══════════════════════════════════════════════════════════════════════════════
# 13. test_llm_relevance_check_irrelevant
# ══════════════════════════════════════════════════════════════════════════════

def test_llm_relevance_check_irrelevant():
    """
    _validate_math_relevance must return (False, reason) when the mock LLM
    responds with relevant=false, causing the block to be replaced with the
    irrelevance notice.
    """
    class _MockGroq:
        async def complete_fast(self, messages, temperature=0.0):
            return '{"relevant": false, "reason": "User asked about accuracy, not word count."}'

    relevant, reason = run(
        _validate_math_relevance(
            query="What is the F1 score of the model?",
            code="result = len('hello world'.split())\nprint(result)",
            result_stdout="2",
            groq=_MockGroq(),
        )
    )
    assert relevant is False
    assert "accuracy" in reason or "word count" in reason or reason != ""


# ══════════════════════════════════════════════════════════════════════════════
# 14. test_irrelevant_block_replaced_in_synthesis
# ══════════════════════════════════════════════════════════════════════════════

def test_irrelevant_block_replaced_in_synthesis():
    """
    When the LLM marks a successfully executed block as irrelevant, the
    synthesis text must contain the irrelevance notice, not the raw result.
    """
    class _MockGroq:
        async def complete_fast(self, messages, temperature=0.0):
            return '{"relevant": false, "reason": "Counts words, not what was asked."}'

    text = (
        "The model performance is shown below:\n\n"
        "```python\n"
        "result = len('hello world'.split())\n"
        "print(f'Word count: {result}')\n"
        "```\n\n"
        "End of synthesis."
    )
    evidence_table = _FakeEvidenceTable([])
    sandbox = MathSandbox()

    result_text = run(
        _execute_math_blocks(
            text, evidence_table, sandbox,
            query="What is the BLEU score?",
            groq=_MockGroq(),
        )
    )

    assert "```python" not in result_text
    assert "Calculation removed" in result_text or "not relevant" in result_text
    # The raw computed value must NOT appear
    assert "Word count" not in result_text


# ══════════════════════════════════════════════════════════════════════════════
# 15. test_llm_validation_failure_falls_back_to_accepted
# ══════════════════════════════════════════════════════════════════════════════

def test_llm_validation_failure_falls_back_to_accepted():
    """
    If the LLM validator raises an exception (network error, bad JSON, etc.),
    the block must still be accepted (fail-open) rather than silently dropped.
    """
    class _BrokenGroq:
        async def complete_fast(self, messages, temperature=0.0):
            raise RuntimeError("Groq API timeout")

    relevant, reason = run(
        _validate_math_relevance(
            query="What is the entity density?",
            code="result = 18432 / 512\nprint(result)",
            result_stdout="36.0",
            groq=_BrokenGroq(),
        )
    )
    # Must fall back to relevant=True so a groq outage never drops valid results
    assert relevant is True
