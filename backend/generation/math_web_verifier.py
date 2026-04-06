"""
math_web_verifier.py — Web-search-grounded LLM verification of computed math results.

Pipeline per computed block:
  1. Build a targeted search query from code + result + user query context.
  2. Fetch top-N web snippets via Tavily (general web, not scholarly-only).
  3. Feed (query, code, result, web_snippets) to LLM with a strict audit prompt.
  4. LLM returns a structured verdict: correct / plausible / flagged / incorrect.
  5. Caller annotates or suppresses the block accordingly.

Design principles:
- Never blocks the answer on web-search latency: all failures fall through as "skipped".
- Only searches when the result is numeric and non-trivial (skips word counts, indices, etc.).
- Two verification axes: formula correctness AND result plausibility.
- Terminal banner printed for every verification so math audit is always visible.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_COL = 76


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class WebVerificationResult:
    verdict: str              # "verified" | "plausible" | "flagged" | "incorrect" | "skipped"
    correct: bool             # True → result is mathematically sound
    confidence: float         # 0.0–1.0
    issues: list[str] = field(default_factory=list)
    reference_snippet: str = ""   # web excerpt that supported/contradicted the result
    search_query: str = ""        # what was actually searched
    should_display: bool = True   # final gate: show in the answer?
    suppression_reason: str = ""  # set when should_display=False


# ── System prompt for the LLM verifier ───────────────────────────────────────

_WEB_VERIFY_SYSTEM = """You are a rigorous mathematical verification auditor for a scientific research assistant.

You receive:
  - USER QUERY: the original research question
  - PYTHON CODE: the computation that was run
  - SANDBOX RESULT: the actual output produced by executing the code
  - WEB CONTEXT: snippets from authoritative web sources about the topic

Your task: determine whether the sandbox result is mathematically correct and
contextually appropriate given the user query. Two axes:
  1. FORMULA AXIS: Is the formula/method used in the code correct for this type of problem?
  2. PLAUSIBILITY AXIS: Is the numerical result in a physically/statistically plausible range?

Respond with EXACTLY one JSON object on a single line — nothing else:
{
  "correct": true/false,
  "confidence": 0.0-1.0,
  "issues": ["<issue 1>", "<issue 2>"],
  "reference": "<one sentence quoting the most relevant web evidence>",
  "verdict": "verified|plausible|flagged|incorrect"
}

Verdict rules:
  "verified"   → formula is textbook-correct, result is plausible, confidence >= 0.75
  "plausible"  → no formula errors found but web context is insufficient to fully confirm
  "flagged"    → formula may be correct but result is outside expected range or units are wrong
  "incorrect"  → clear formula error or result is physically/statistically impossible

Be conservative: only return "incorrect" when you have strong web evidence of a mistake.
When web context is sparse, prefer "plausible" over "incorrect" to avoid false negatives.
DO NOT use general knowledge alone — ground every finding in the WEB CONTEXT provided."""


# ── Search query builder ──────────────────────────────────────────────────────

def _build_search_query(query: str, code: str, _result_stdout: str) -> str:
    """
    Build a focused web search query that will find authoritative pages
    about the formula/calculation type being verified.

    Strategy:
      1. Extract the computed variable name from a "result = ..." line.
      2. Look for known formula patterns (F1, entropy, energy, precision, etc.).
      3. Combine with key noun phrases from the user query.
    """
    # Extract result variable name or print label
    result_name = ""
    # Match: result = ... or print(f"Label: ...")
    m = re.search(r'(?:^|\n)\s*result\s*=\s*(.+)', code)
    if m:
        result_name = m.group(1).strip()[:60]
    print_label = ""
    m2 = re.search(r'print\s*\(\s*f?["\']([^"\']{3,60})["\']', code)
    if m2:
        print_label = re.sub(r'\{[^}]+\}', '', m2.group(1)).strip()

    # Known formula keywords to prioritise
    formula_hints: list[str] = []
    lower_code = code.lower()
    if any(k in lower_code for k in ("f1", "f_1", "f-score", "fscore")):
        formula_hints.append("F1 score formula")
    if any(k in lower_code for k in ("precision", "recall")):
        formula_hints.append("precision recall")
    if any(k in lower_code for k in ("entropy", "log2", "log(")):
        formula_hints.append("information entropy formula")
    if any(k in lower_code for k in ("energy", "power", "watt", "joule")):
        formula_hints.append("energy power calculation")
    if any(k in lower_code for k in ("perplexity", "cross_entropy")):
        formula_hints.append("perplexity language model formula")
    if any(k in lower_code for k in ("rouge", "bleu", "meteor")):
        formula_hints.append("NLP evaluation metric formula")
    if any(k in lower_code for k in ("accuracy", "acc =")):
        formula_hints.append("classification accuracy formula")
    if any(k in lower_code for k in ("mean", "average", "avg")):
        formula_hints.append("mean average calculation")
    if any(k in lower_code for k in ("stddev", "std(", "variance", "var(")):
        formula_hints.append("standard deviation variance formula")
    if any(k in lower_code for k in ("cosine", "dot(")):
        formula_hints.append("cosine similarity formula")
    if any(k in lower_code for k in ("softmax", "sigmoid", "relu")):
        formula_hints.append("neural network activation function")

    # Extract short noun phrases from user query (first 10 words)
    query_words = re.findall(r'\b[a-zA-Z]{3,}\b', query)
    query_context = " ".join(query_words[:6])

    parts: list[str] = []
    if formula_hints:
        parts.append(formula_hints[0])
    elif print_label:
        parts.append(print_label)
    elif result_name:
        parts.append(result_name[:40])
    parts.append(query_context)
    parts.append("calculation verification formula")

    search_query = " ".join(dict.fromkeys(parts))  # deduplicate, preserve order
    return search_query[:200]


# ── Numeric result detector ────────────────────────────────────────────────────

def _is_numeric_result(stdout: str) -> bool:
    """
    Return True if the sandbox output contains at least one meaningful number.
    Skips results that are purely word counts, indices, or boolean flags.
    """
    # Must have at least one digit
    nums = re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?', stdout)
    if not nums:
        return False
    # Skip trivially small integers (0, 1, 2 ...) unless they're floats
    meaningful = [n for n in nums if '.' in n or 'e' in n.lower() or abs(float(n)) > 2]
    return bool(meaningful)


# ── Terminal banner ────────────────────────────────────────────────────────────

_VERDICT_ICON = {
    "verified":  "[WEB-VERIFIED]",
    "plausible": "[WEB-PLAUSIBLE]",
    "flagged":   "[WEB-FLAGGED]  ",
    "incorrect": "[WEB-INCORRECT]",
    "skipped":   "[WEB-SKIPPED]  ",
}


def _print_web_verify_banner(
    verdict: str,
    confidence: float,
    search_query: str,
    issues: list[str],
    reference: str,
) -> None:
    thin   = "-" * _COL
    icon   = _VERDICT_ICON.get(verdict, f"[{verdict.upper()}]")
    print(f"\n+{thin}+", flush=True)
    print(f"|  {icon}  conf={confidence:.0%}  query: {search_query[:40]!r:<42}|", flush=True)
    if issues:
        for issue in issues[:3]:
            print(f"|  ISSUE: {issue[:_COL - 10]:<{_COL - 8}}|", flush=True)
    if reference:
        ref_lines = [reference[i:i+(_COL-10)] for i in range(0, min(len(reference), 200), _COL-10)]
        for line in ref_lines:
            print(f"|  REF:   {line:<{_COL - 8}}|", flush=True)
    print(f"+{thin}+\n", flush=True)


# ── Main verifier ─────────────────────────────────────────────────────────────

async def web_verify_math(
    query: str,
    code: str,
    result_stdout: str,
    groq,                         # GroqClient — typed loosely to avoid circular import
    max_web_results: int = 4,
) -> WebVerificationResult:
    """
    Verify a computed math result using web search + LLM audit.

    Steps:
      1. Check whether the result is numeric enough to warrant verification.
      2. Build a focused search query and fetch web snippets via Tavily.
      3. Ask the LLM to audit formula correctness + result plausibility.
      4. Return a structured WebVerificationResult.

    Never raises — all failures return verdict="skipped" so the caller can
    proceed to display the result without web verification.
    """
    # Skip non-numeric or trivial results
    if not _is_numeric_result(result_stdout):
        logger.info("Web math verify: skipping non-numeric result %r", result_stdout[:60])
        return WebVerificationResult(
            verdict="skipped", correct=True, confidence=0.5, should_display=True
        )

    search_query = _build_search_query(query, code, result_stdout)

    # ── Web search ────────────────────────────────────────────────────────────
    web_snippets: list[str] = []
    try:
        from backend.config import settings

        if not settings.TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY not set")

        # Use a broader search for formula verification (not scholarly-only)
        import httpx
        payload = {
            "query": search_query,
            "search_depth": "basic",   # fast, cheap — we only need snippets
            "topic": "general",
            "max_results": max_web_results,
            "include_raw_content": False,
            "include_answer": True,
            "include_images": False,
        }
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json=payload,
                headers={
                    "Authorization": f"Bearer {settings.TAVILY_API_KEY}",
                    "User-Agent": "NexusScholar/1.0",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # Collect answer + content snippets
        if data.get("answer"):
            web_snippets.append(f"[Tavily answer] {data['answer'][:400]}")
        for item in (data.get("results") or [])[:max_web_results]:
            content = (item.get("content") or "").strip()
            title   = (item.get("title")   or "").strip()
            url     = (item.get("url")     or "").strip()
            if content:
                web_snippets.append(f"[{title} | {url}]\n{content[:400]}")

        logger.info(
            "Web math verify: Tavily returned %d snippets for query %r",
            len(web_snippets), search_query[:80],
        )

    except Exception as exc:
        logger.warning("Web math verify: search failed (%s) — skipping", exc)
        # Print minimal banner so user sees what happened
        _print_web_verify_banner("skipped", 0.0, search_query, [str(exc)[:80]], "")
        return WebVerificationResult(
            verdict="skipped", correct=True, confidence=0.5,
            search_query=search_query, should_display=True,
        )

    if not web_snippets:
        logger.info("Web math verify: no snippets returned — skipping")
        return WebVerificationResult(
            verdict="skipped", correct=True, confidence=0.5,
            search_query=search_query, should_display=True,
        )

    # ── LLM audit ─────────────────────────────────────────────────────────────
    web_context_text = "\n\n---\n".join(web_snippets[:4])
    prompt = (
        f"USER QUERY:\n{query[:400]}\n\n"
        f"PYTHON CODE:\n```python\n{code[:800]}\n```\n\n"
        f"SANDBOX RESULT:\n{result_stdout[:300]}\n\n"
        f"WEB CONTEXT ({len(web_snippets)} snippet(s)):\n{web_context_text[:2000]}\n\n"
        "Based ONLY on the web context above, verify whether the computation is "
        "correct and the result is plausible. Reply with ONLY valid JSON (one line):\n"
        '{"correct": bool, "confidence": float, "issues": [str], '
        '"reference": str, "verdict": "verified|plausible|flagged|incorrect"}'
    )

    try:
        raw = await groq.complete_fast(
            messages=[
                {"role": "system", "content": _WEB_VERIFY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", raw).strip()
        # Strip any leading/trailing non-JSON characters
        cleaned = re.sub(r'^[^{]*', '', cleaned)
        cleaned = re.sub(r'[^}]*$', '', cleaned)
        obj = json.loads(cleaned)

        verdict    = str(obj.get("verdict", "plausible")).lower()
        correct    = bool(obj.get("correct", True))
        confidence = float(obj.get("confidence", 0.5))
        issues     = [str(i) for i in (obj.get("issues") or [])]
        reference  = str(obj.get("reference", ""))

        # Normalise verdict
        if verdict not in ("verified", "plausible", "flagged", "incorrect"):
            verdict = "plausible"

        # Decision: suppress result if LLM is confident it is incorrect
        # Threshold: incorrect + confidence >= 0.85
        should_display = not (verdict == "incorrect" and confidence >= 0.85)
        suppression_reason = ""
        if not should_display:
            suppression_reason = "; ".join(issues) if issues else "Web verification found this result to be incorrect."

        _print_web_verify_banner(verdict, confidence, search_query, issues, reference)
        logger.info(
            "Web math verify: verdict=%s correct=%s confidence=%.2f issues=%s",
            verdict, correct, confidence, issues,
        )

        return WebVerificationResult(
            verdict=verdict,
            correct=correct,
            confidence=confidence,
            issues=issues,
            reference_snippet=reference,
            search_query=search_query,
            should_display=should_display,
            suppression_reason=suppression_reason,
        )

    except Exception as exc:
        logger.warning("Web math verify: LLM audit failed (%s) — skipping", exc)
        return WebVerificationResult(
            verdict="skipped", correct=True, confidence=0.5,
            search_query=search_query, should_display=True,
        )


def format_verified_block(
    code: str,
    result_stdout: str,
    web_result: WebVerificationResult,
) -> str:
    """
    Format the final replacement string for a code block based on its
    web verification verdict.

    Verdicts → display format:
      verified   → green badge + result + reference snippet
      plausible  → result only (no badge)
      flagged    → result + orange warning with issues
      incorrect  → suppressed if confidence >= 0.85, else flagged display
      skipped    → result only (no badge)
    """
    if not web_result.should_display:
        # High-confidence incorrect — suppress and explain
        issue_text = web_result.suppression_reason or "Web verification found this result to be mathematically incorrect."
        return (
            f"> **[Math Suppressed]** The computed result was suppressed by web verification.\n"
            f"> **Reason:** {issue_text}\n"
            f"> *Search context: {web_result.search_query}*"
        )

    result_line = f"> **Result:** `{result_stdout}`"

    if web_result.verdict == "verified":
        badge = f"> **[Web-Verified]** Checked against web sources (confidence {web_result.confidence:.0%})"
        ref_line = (
            f"\n> **Reference:** _{web_result.reference_snippet}_"
            if web_result.reference_snippet else ""
        )
        return (
            f"**[Verified Calculation]**\n"
            f"```python\n{code}\n```\n"
            f"{result_line}\n"
            f"{badge}{ref_line}"
        )

    elif web_result.verdict == "flagged":
        issues_text = "\n".join(f"> - {i}" for i in web_result.issues[:3]) if web_result.issues else ""
        ref_line = f"\n> **Reference:** _{web_result.reference_snippet}_" if web_result.reference_snippet else ""
        return (
            f"**[Verified Calculation - Flagged]**\n"
            f"```python\n{code}\n```\n"
            f"{result_line}\n"
            f"> **[Web-Flagged]** Confidence {web_result.confidence:.0%} — review recommended\n"
            f"{issues_text}{ref_line}"
        )

    elif web_result.verdict == "incorrect" and web_result.confidence < 0.85:
        # Low-confidence incorrect — show result but warn
        issues_text = "\n".join(f"> - {i}" for i in web_result.issues[:3]) if web_result.issues else ""
        return (
            f"**[Verified Calculation - Unconfirmed]**\n"
            f"```python\n{code}\n```\n"
            f"{result_line}\n"
            f"> **[Web-Flagged]** Web sources suggest possible error (confidence {web_result.confidence:.0%})\n"
            f"{issues_text}"
        )

    else:
        # plausible or skipped — show cleanly
        return (
            f"**[Verified Calculation]**\n"
            f"```python\n{code}\n```\n"
            f"{result_line}"
        )
