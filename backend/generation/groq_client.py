from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import AsyncGenerator

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 2.0  # seconds
_MAX_RETRY_WAIT = 60.0   # seconds — if Retry-After exceeds this, don't retry (daily limit hit)
_MIN_MAX_TOKENS = 256    # never go below this on automatic reduction


def _reduce_max_tokens_for_413(body_text: str, current_max_tokens: int) -> int | None:
    """
    Parse a Groq 413 error body and return a reduced max_tokens value that
    should bring the request under the TPM limit.

    Groq error messages look like:
      "Limit 12000, Requested 13229, please reduce your message size…"

    Returns None if the body cannot be parsed or if the reduction would go
    below _MIN_MAX_TOKENS (meaning the prompt itself is too large; reducing
    output won't help).
    """
    match = re.search(r"Limit\s+(\d+),\s+Requested\s+(\d+)", body_text)
    if not match:
        return None
    limit = int(match.group(1))
    requested = int(match.group(2))
    overage = requested - limit
    # Reduce max_tokens by the overage plus a 10 % safety buffer
    new_max = current_max_tokens - overage - max(50, int(overage * 0.10))
    if new_max < _MIN_MAX_TOKENS:
        return None
    return new_max


class _RateLimitExhausted(Exception):
    """Raised when Retry-After exceeds _MAX_RETRY_WAIT (daily quota exhausted)."""


async def _wait_for_retry(response: httpx.Response, attempt: int) -> None:
    """Wait based on Retry-After header or exponential backoff.

    Raises _RateLimitExhausted if the server-requested wait exceeds
    _MAX_RETRY_WAIT, which signals a daily quota reset rather than a
    transient burst limit.
    """
    retry_after = response.headers.get("retry-after") or response.headers.get("x-ratelimit-reset-requests")
    if retry_after:
        try:
            wait = float(retry_after)
            if wait > _MAX_RETRY_WAIT:
                logger.warning(
                    "Groq 429: Retry-After=%.0fs exceeds max wait (%.0fs) — daily quota likely exhausted, falling back",
                    wait, _MAX_RETRY_WAIT,
                )
                raise _RateLimitExhausted(f"Retry-After={wait:.0f}s")
            logger.info("Groq 429: Retry-After=%.1fs (attempt %s)", wait, attempt)
            await asyncio.sleep(wait)
            return
        except ValueError:
            pass
    wait = _RETRY_BASE_DELAY * (2 ** attempt)
    logger.info("Groq 429: backoff %.1fs (attempt %s)", wait, attempt)
    await asyncio.sleep(wait)


class GroqClient:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.api_base = settings.GROQ_API_BASE.rstrip("/")
        self.primary_model = settings.GROQ_MODEL_PRIMARY
        self.fast_model = settings.GROQ_MODEL_FAST

    async def complete_fast(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        return await self._chat_completion(
            model=self.fast_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

    async def complete_primary(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        if not self.api_key:
            fallback_text = self._fallback_markdown(messages)
            if stream:
                return self._stream_text(fallback_text)
            return fallback_text

        try:
            if stream:
                return self._stream_completion(
                    model=self.primary_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return await self._chat_completion(
                model=self.primary_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
        except Exception as exc:
            logger.warning("Groq primary completion failed: %s", exc)
            fallback_text = self._fallback_markdown(messages)
            if stream:
                return self._stream_text(fallback_text)
            return fallback_text

    async def _chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        logger.info(
            "Groq request: model=%s temperature=%s max_tokens=%s stream=%s",
            model,
            temperature,
            max_tokens,
            stream,
        )
        last_exc: Exception | None = None
        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(_MAX_RETRIES):
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if response.status_code == 429:
                    logger.warning(
                        "Groq 429 Too Many Requests (model=%s, attempt=%s/%s)",
                        model, attempt + 1, _MAX_RETRIES,
                    )
                    last_exc = httpx.HTTPStatusError(
                        f"429 Too Many Requests after {_MAX_RETRIES} attempts",
                        request=response.request,
                        response=response,
                    )
                    if attempt < _MAX_RETRIES - 1:
                        try:
                            await _wait_for_retry(response, attempt)
                        except _RateLimitExhausted:
                            break
                        continue
                    break
                if response.status_code >= 400:
                    logger.warning(
                        "Groq error %s for model %s: %s",
                        response.status_code,
                        model,
                        response.text,
                    )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        raise last_exc or RuntimeError("Groq request failed after retries")

    async def _stream_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.api_base}/chat/completions"

        for attempt in range(_MAX_RETRIES):
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code == 429:
                        body = await response.aread()
                        logger.warning(
                            "Groq streaming 429 Too Many Requests (model=%s, attempt=%s/%s): %s",
                            model, attempt + 1, _MAX_RETRIES,
                            body.decode("utf-8", errors="ignore"),
                        )
                        if attempt < _MAX_RETRIES - 1:
                            try:
                                await _wait_for_retry(response, attempt)
                            except _RateLimitExhausted:
                                response.raise_for_status()
                            continue
                        response.raise_for_status()

                    if response.status_code == 413:
                        body = await response.aread()
                        body_text = body.decode("utf-8", errors="ignore")
                        logger.error(
                            "Groq 413 Payload Too Large (model=%s, attempt=%s/%s): %s",
                            model, attempt + 1, _MAX_RETRIES, body_text,
                        )
                        if attempt < _MAX_RETRIES - 1:
                            # Parse the overage from the error message and reduce max_tokens
                            new_max = _reduce_max_tokens_for_413(body_text, payload["max_tokens"])
                            if new_max is not None and new_max != payload["max_tokens"]:
                                logger.warning(
                                    "Groq 413: reducing max_tokens %d→%d and retrying (attempt %s/%s)",
                                    payload["max_tokens"], new_max, attempt + 1, _MAX_RETRIES,
                                )
                                payload = {**payload, "max_tokens": new_max}
                                continue
                        response.raise_for_status()

                    if response.status_code >= 400:
                        body = await response.aread()
                        logger.warning(
                            "Groq streaming error %s for model %s: %s",
                            response.status_code,
                            model,
                            body.decode("utf-8", errors="ignore"),
                        )
                        response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            return
                        try:
                            parsed = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        delta = parsed["choices"][0].get("delta", {}).get("content")
                        if delta:
                            yield delta
                    return  # successful stream completed

    async def _stream_text(self, text: str) -> AsyncGenerator[str, None]:
        for token in re.split(r"(\s+)", text):
            if token:
                yield token
                await asyncio.sleep(0)

    def _fallback_markdown(self, messages: list[dict]) -> str:
        logger.info("Using fallback answer synthesis")
        user_text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
        evidence = self._extract_evidence_rows(user_text)
        query = self._extract_section(user_text, "## USER QUERY")
        intent = self._extract_section(user_text, "## QUERY INTENT") or "general"

        if not evidence:
            return (
                "## Insufficient Evidence\n\n"
                "Evidence is insufficient to answer this with confidence.\n\n"
                "## Limitations & Confidence\n\n"
                "- No evidence rows were available to synthesize a grounded answer.\n"
            )

        lines = ["## Research Synthesis", ""]
        if intent == "benchmark_comparison":
            lines.append("### Comparison of Methods")
            lines.append("")
            lines.extend(
                [
                    "| Method | Key Finding | Year | Venue | Source |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for idx, row in enumerate(evidence[:8], start=1):
                finding = row.get("text", "").strip().replace("\n", " ")[:200] or "Relevant evidence excerpt"
                method = row.get("paper_title", "Unknown")[:60]
                year = row.get("year") or "n.d."
                venue = row.get("venue") or "Unknown"
                lines.append(f"| {method} | {finding} | {year} | {venue} | [{idx}] |")
        else:
            lines.append(
                f"### Overview\n\nFor the query '{query or 'research question'}', the following evidence was retrieved from the indexed literature.\n"
            )
            lines.append("### Detailed Evidence\n")
            for idx, row in enumerate(evidence[:8], start=1):
                preview = row.get("text", "").strip().replace("\n", " ")[:400]
                title = row.get("paper_title", "Unknown paper")
                venue = row.get("venue") or "Unknown venue"
                year = row.get("year") or "n.d."
                section = row.get("section") or "general"
                lines.append(f"**{idx}. {title}** ({venue}, {year}) — *{section}*")
                lines.append(f"   {preview} [{idx}]")
                lines.append("")

        lines.extend(
            [
                "",
                "## Limitations & Confidence",
                "",
                f"- Built with fallback synthesis because `GROQ_API_KEY` is missing or Groq was unavailable.",
                f"- Evidence rows used: {min(len(evidence), 5)} of {len(evidence)}.",
            ]
        )
        return "\n".join(lines)

    def _extract_evidence_rows(self, prompt: str) -> list[dict]:
        match = re.search(
            r"## EVIDENCE TABLE \(JSON\)\s*(.*?)\s*## USER QUERY",
            prompt,
            re.DOTALL,
        )
        if not match:
            return []
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return []

    def _extract_section(self, prompt: str, heading: str) -> str:
        match = re.search(rf"{re.escape(heading)}\s*(.*?)\s*(## |\Z)", prompt, re.DOTALL)
        if not match:
            return ""
        return match.group(1).strip()
