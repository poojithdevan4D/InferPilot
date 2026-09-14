"""Asynchronous streaming workload client + per-request measurement.

Sends requests to the OpenAI-compatible ``/v1/completions`` endpoint with
streaming enabled and produces one :class:`RequestMeasurement` per request.

Timing rules (per project constraints):

* All timing uses a **monotonic** clock, never wall-clock.
* ``start`` is stamped just before the request is sent, ``first_token`` at the
  first received output token, ``end`` at stream completion.
* ``start_time_s`` / ``end_time_s`` are expressed relative to a shared experiment
  origin ``t0`` (also monotonic) so measurements share one timeline.
* TTFT, e2e, and TPOT use the agreed formula
  ``tpot_ms = (e2e_latency_ms - ttft_ms) / (output_tokens - 1)``.

Failure handling: HTTP errors, malformed stream chunks, timeouts, and empty
outputs all yield a *failed* measurement (``success=False`` with an ``error``),
never an exception that aborts the run.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from time import monotonic
from typing import Optional

import httpx

from ..measurements import RequestMeasurement


@dataclass(frozen=True)
class GenerationParams:
    """Per-request generation parameters (deterministic, controlled-length)."""

    model: str
    max_tokens: int
    temperature: float = 0.0
    seed: Optional[int] = 0
    ignore_eos: bool = False
    min_tokens: Optional[int] = None
    request_timeout_s: float = 120.0


class _MalformedStream(Exception):
    pass


class _BadStatus(Exception):
    pass


def _build_payload(prompt: str, params: GenerationParams) -> dict:
    payload: dict = {
        "model": params.model,
        "prompt": prompt,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if params.seed is not None:
        payload["seed"] = params.seed
    if params.ignore_eos:
        payload["ignore_eos"] = True
        payload["min_tokens"] = params.min_tokens if params.min_tokens is not None else params.max_tokens
    return payload


def _compute_metrics(
    *, start: float, first_token: Optional[float], end: float, output_tokens: int
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (ttft_ms, tpot_ms, e2e_ms) from monotonic stamps and token count."""
    e2e_ms = (end - start) * 1000.0
    if first_token is None:
        return None, None, e2e_ms
    ttft_ms = (first_token - start) * 1000.0
    if output_tokens >= 2:
        tpot_ms = (e2e_ms - ttft_ms) / (output_tokens - 1)
    else:
        tpot_ms = None
    return ttft_ms, tpot_ms, e2e_ms


async def _measure_one(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    params: GenerationParams,
    t0: float,
    request_id: str,
) -> RequestMeasurement:
    payload = _build_payload(prompt, params)

    start = monotonic()
    first_token: Optional[float] = None
    usage: Optional[dict] = None
    text_chunks = 0
    error: Optional[str] = None

    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:500]
                raise _BadStatus(f"HTTP {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise _MalformedStream(f"invalid stream chunk: {exc}") from exc
                choices = chunk.get("choices") or []
                if choices:
                    text = choices[0].get("text", "")
                    if text:
                        if first_token is None:
                            first_token = monotonic()
                        text_chunks += 1
                if chunk.get("usage"):
                    usage = chunk["usage"]
    except httpx.TimeoutException as exc:
        error = f"timeout: {exc!r}"
    except httpx.HTTPError as exc:
        error = f"http_error: {exc!r}"
    except _BadStatus as exc:
        error = str(exc)
    except _MalformedStream as exc:
        error = f"malformed_stream: {exc}"

    end = monotonic()

    # Actual token counts from the server response where possible.
    if usage is not None:
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
    else:
        prompt_tokens = 0
        output_tokens = text_chunks

    if error is None and (first_token is None or output_tokens < 1):
        error = "empty_output: server returned no output tokens"

    start_time_s = max(0.0, start - t0)
    end_time_s = max(start_time_s, end - t0)

    if error is not None:
        return RequestMeasurement(
            request_id=request_id,
            prompt_tokens=max(0, prompt_tokens),
            output_tokens=0,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            success=False,
            error=error,
        )

    ttft_ms, tpot_ms, e2e_ms = _compute_metrics(
        start=start, first_token=first_token, end=end, output_tokens=output_tokens
    )
    return RequestMeasurement(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2e_latency_ms=e2e_ms,
        success=True,
    )


async def run_requests(
    base_url: str,
    prompts: list[str],
    params: GenerationParams,
    *,
    t0: float,
    max_concurrency: int,
    id_prefix: str = "req",
) -> list[RequestMeasurement]:
    """Issue ``prompts`` closed-loop with at most ``max_concurrency`` in flight.

    Returns measurements in the same order as ``prompts``.
    """
    url = base_url.rstrip("/") + "/v1/completions"
    sem = asyncio.Semaphore(max_concurrency)
    timeout = httpx.Timeout(params.request_timeout_s)

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def worker(index: int, prompt: str) -> RequestMeasurement:
            async with sem:
                return await _measure_one(
                    client, url, prompt, params, t0, f"{id_prefix}-{index}"
                )

        tasks = [asyncio.create_task(worker(i, p)) for i, p in enumerate(prompts)]
        return list(await asyncio.gather(*tasks))


async def run_requests_open_loop(
    base_url: str,
    prompts: list[str],
    params: GenerationParams,
    *,
    t0: float,
    offsets: list[float],
    id_prefix: str = "req",
) -> tuple[list[RequestMeasurement], list[float]]:
    """Dispatch ``prompts`` at their scheduled arrival ``offsets`` (open-loop).

    Every request is scheduled up front and fires at ``t0 + offset`` regardless of
    whether earlier requests have completed (no semaphore, no back-pressure).
    Returns ``(measurements, actual_dispatch_offsets)`` in prompt order; the
    dispatch offsets are the monotonic send times relative to ``t0``.
    """
    url = base_url.rstrip("/") + "/v1/completions"
    timeout = httpx.Timeout(params.request_timeout_s)
    dispatch_offsets: list[float] = [0.0] * len(prompts)

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def worker(index: int, prompt: str, scheduled: float) -> RequestMeasurement:
            delay = (t0 + scheduled) - monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            dispatch_offsets[index] = monotonic() - t0
            return await _measure_one(
                client, url, prompt, params, t0, f"{id_prefix}-{index}"
            )

        tasks = [
            asyncio.create_task(worker(i, p, offsets[i])) for i, p in enumerate(prompts)
        ]
        results = list(await asyncio.gather(*tasks))
    return results, dispatch_offsets
