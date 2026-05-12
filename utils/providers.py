"""Generation providers for QuanBench+.

The benchmark builds OpenRouter-shaped chat-completion payloads
(see ``utils/parse_prompt.py`` and ``utils/parse_prompt_with_feedback.py``)
and downstream consumers (``utils/parse_response.py`` and the pipeline
runners) expect OpenRouter-shaped responses back.

This module dispatches a single ``send_generation_request`` call to either
OpenRouter or the Coda agent endpoint, normalises Coda's SSE/JSON output to
the OpenRouter ``choices[0].message.content`` shape, and always returns
``(response_dict, chat_completion_prefill)`` so the existing parsers work
unchanged.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

OPENROUTER_PROVIDER = "openrouter"
CODA_PROVIDER = "coda"
BEDROCK_PROVIDER = "bedrock"
SUPPORTED_PROVIDERS = (OPENROUTER_PROVIDER, CODA_PROVIDER, BEDROCK_PROVIDER)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_CODA_BASE_URL = "https://api.conductorquantum.com/v0/coda"
DEFAULT_CODA_MODEL_LABEL = "coda/build"

# Default Bedrock model and region for the raw-LLM benchmark path.
# Pinned to the same model the Coda build agent uses so the comparison
# isolates the contribution of the Coda harness vs the underlying LLM.
DEFAULT_BEDROCK_MODEL = "global.anthropic.claude-opus-4-6-v1"
DEFAULT_BEDROCK_REGION = "us-west-1"
# Bedrock Converse caps thinking effort at 'low|medium|high|max' for Opus 4.6.
_BEDROCK_VALID_EFFORTS = frozenset({"low", "medium", "high", "max"})
_DEFAULT_BEDROCK_EFFORT = "high"

# Roles Coda's AgentsMessageRole enum allows. The OpenRouter prompt builders
# only ever emit ``user`` and ``assistant``, but the provider rejects anything
# else explicitly so a future bug doesn't silently elide a system instruction.
_CODA_ALLOWED_ROLES = frozenset({"user", "assistant"})

# HTTP statuses we treat as transient and retry with exponential backoff.
# ``303`` is included because the Coda gateway has been observed to return
# bare 303s with an error body (rather than a real redirect) under load.
_CODA_RETRYABLE_STATUS = frozenset({303, 408, 429, 500, 502, 503, 504})
_DEFAULT_CODA_MAX_RETRIES = 4
_DEFAULT_CODA_RETRY_BASE_DELAY = 1.0
_DEFAULT_CODA_RETRY_MAX_DELAY = 30.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def send_generation_request(
    request_payload: Dict[str, Any],
    provider: str,
    *,
    timeout: int = 180,
) -> Tuple[Dict[str, Any], str]:
    """Send a generation request through the chosen provider.

    Returns ``(normalized_response, chat_completion_prefill)`` so callers can
    pass the tuple straight to ``utils/parse_response.parse_response``.
    """
    provider = (provider or OPENROUTER_PROVIDER).lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Expected one of: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    chat_completion = _extract_chat_completion(request_payload)
    if provider == OPENROUTER_PROVIDER:
        response = _send_openrouter(request_payload, timeout=timeout)
    elif provider == BEDROCK_PROVIDER:
        response = _send_bedrock(request_payload, timeout=timeout)
    else:
        response = _send_coda(request_payload, timeout=_coda_request_timeout(timeout))
    return response, chat_completion


def send_generation_request_dict(
    request_payload: Dict[str, Any],
    provider: str,
    *,
    timeout: int = 180,
) -> Dict[str, Any]:
    """Same as ``send_generation_request`` but returns only the response dict."""
    response, _ = send_generation_request(
        request_payload, provider, timeout=timeout
    )
    return response


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------
def _send_openrouter(
    request_payload: Dict[str, Any], *, timeout: int
) -> Dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        return _error_response(
            status=0,
            message="Missing OPENROUTER_API_KEY (or legacy API_KEY) in environment for OpenRouter.",
            model=request_payload.get("model"),
        )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    print("Sending request to", request_payload.get("model"))
    try:
        response = requests.post(
            OPENROUTER_URL,
            json=request_payload,
            headers=headers,
            timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        return _error_response(
            status=0,
            message=f"OpenRouter request failed: {exc}",
            model=request_payload.get("model"),
        )

    if response.status_code != 200:
        return _error_response(
            status=response.status_code,
            message=response.text[:4000],
            model=request_payload.get("model"),
        )

    try:
        return response.json()
    except ValueError as exc:
        return _error_response(
            status=response.status_code,
            message=f"Could not decode OpenRouter response as JSON: {exc}; body={response.text[:1000]}",
            model=request_payload.get("model"),
        )


# ---------------------------------------------------------------------------
# Bedrock (raw LLM, no agent harness)
# ---------------------------------------------------------------------------
def _send_bedrock(request_payload: Dict[str, Any], *, timeout: int) -> Dict[str, Any]:
    """Call AWS Bedrock's Converse API directly with no system prompt and no tools.

    Used for the raw-LLM A/B baseline against the Coda agent harness. Only
    the user message is forwarded; assistant prefill is dropped (Bedrock
    Converse expects a final ``user`` turn). Adaptive thinking is enabled at
    ``effort=high`` by default to match the Coda build agent's settings, but
    no system prompt, tools, or langgraph nodes are involved.

    If the first attempt returns no extractable assistant text (which happens
    when adaptive thinking exhausts the token budget before the model emits
    any text blocks) or hits ``stopReason="max_tokens"``, the request is
    retried once with thinking disabled so the entire token budget is
    available for the answer.

    Returns an OpenRouter-shaped response dict so downstream parsers work
    unchanged.
    """
    try:
        import boto3
        from botocore.config import Config as _BotoConfig
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as exc:
        return _error_response(
            status=0,
            message=f"boto3 is required for the bedrock provider: {exc}",
            model=DEFAULT_BEDROCK_MODEL,
        )

    model = os.getenv("BEDROCK_MODEL", DEFAULT_BEDROCK_MODEL)
    region = os.getenv("BEDROCK_REGION", DEFAULT_BEDROCK_REGION)
    max_tokens = _env_int("BEDROCK_MAX_TOKENS", 16000)
    effort = os.getenv("BEDROCK_EFFORT", _DEFAULT_BEDROCK_EFFORT).lower()
    if effort not in _BEDROCK_VALID_EFFORTS:
        effort = _DEFAULT_BEDROCK_EFFORT
    enable_thinking = _env_truthy("BEDROCK_THINKING", default=True)
    fallback_no_thinking = _env_truthy("BEDROCK_FALLBACK_NO_THINKING", default=True)

    user_text = _bedrock_user_text(request_payload)
    if not user_text:
        return _error_response(
            status=0,
            message="bedrock provider received an empty user message; nothing to send.",
            model=model,
        )

    boto_config = _BotoConfig(retries={"max_attempts": 10, "mode": "adaptive"}, read_timeout=max(timeout, 600))
    try:
        client = boto3.client("bedrock-runtime", region_name=region, config=boto_config)
    except (BotoCoreError, ClientError) as exc:
        return _error_response(status=0, message=f"Could not init bedrock-runtime: {exc}", model=model)

    base_temperature = float(request_payload.get("temperature", 0.0))

    response, err = _bedrock_converse_once(
        client=client,
        model=model,
        user_text=user_text,
        max_tokens=max_tokens,
        thinking=enable_thinking,
        effort=effort,
        base_temperature=base_temperature,
    )
    if err is not None:
        return _error_response(status=0, message=err, model=model)

    text = _bedrock_extract_text(response)
    stop_reason = response.get("stopReason") or ""
    needs_retry = (not text) or stop_reason == "max_tokens"
    if needs_retry and enable_thinking and fallback_no_thinking:
        # Retry once with thinking disabled so the full token budget is
        # available for an actual text answer. This recovers most of the
        # cases where adaptive thinking ate the entire budget.
        print(
            f"bedrock no-text or max_tokens with thinking on (stopReason={stop_reason!r}); "
            "retrying once with thinking disabled.",
            file=sys.stderr,
        )
        response, err = _bedrock_converse_once(
            client=client,
            model=model,
            user_text=user_text,
            max_tokens=max_tokens,
            thinking=False,
            effort=effort,
            base_temperature=base_temperature,
        )
        if err is not None:
            return _error_response(status=0, message=err, model=model)
        text = _bedrock_extract_text(response)
        stop_reason = response.get("stopReason") or ""

    if not text:
        return _error_response(
            status=200,
            message=(
                f"Bedrock returned no extractable assistant text after retry. "
                f"stopReason={stop_reason!r}, output keys={sorted(response.get('output', {}).keys())}"
            ),
            model=model,
        )

    usage = response.get("usage") or {}
    return {
        "id": f"bedrock-{int(time.time() * 1000)}",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": stop_reason or "stop",
                "native_finish_reason": stop_reason or "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("inputTokens"),
            "completion_tokens": usage.get("outputTokens"),
            "total_tokens": usage.get("totalTokens"),
        },
    }


def _bedrock_converse_once(
    *,
    client: Any,
    model: str,
    user_text: str,
    max_tokens: int,
    thinking: bool,
    effort: str,
    base_temperature: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Single Bedrock Converse call. Returns (response_dict, error_message).

    On success returns ``(response, None)``. On botocore failure returns
    ``(None, "<message>")`` so the caller can build the right error envelope.
    """
    from botocore.exceptions import BotoCoreError, ClientError

    additional_fields: Dict[str, Any] = {}
    if thinking:
        additional_fields["thinking"] = {"type": "adaptive", "display": "summarized"}
        additional_fields["output_config"] = {"effort": effort}

    converse_kwargs: Dict[str, Any] = {
        "modelId": model,
        "messages": [{"role": "user", "content": [{"text": user_text}]}],
        "inferenceConfig": {
            "maxTokens": max_tokens,
            # Adaptive thinking requires temperature=1.0; otherwise honour the
            # request_payload's value (defaults to 0.0 for n=1, 0.8 for n>1).
            "temperature": 1.0 if thinking else base_temperature,
        },
    }
    if additional_fields:
        converse_kwargs["additionalModelRequestFields"] = additional_fields

    print(f"Sending request to bedrock ({model}, effort={effort if thinking else 'off'})")

    try:
        response = client.converse(**converse_kwargs)
    except (BotoCoreError, ClientError) as exc:
        return None, f"Bedrock converse failed: {exc}"
    return response, None


def _bedrock_user_text(request_payload: Dict[str, Any]) -> str:
    """Pick the first user message from the OpenRouter-shaped request_payload."""
    for msg in request_payload.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _bedrock_extract_text(response: Dict[str, Any]) -> str:
    """Concatenate the ``text`` blocks in a Bedrock Converse response, ignoring thinking."""
    output = response.get("output") or {}
    message = output.get("message") or {}
    blocks = message.get("content") or []
    parts: List[str] = []
    for block in blocks:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Coda
# ---------------------------------------------------------------------------
def _send_coda(request_payload: Dict[str, Any], *, timeout: int) -> Dict[str, Any]:
    api_key = os.getenv("CODA_API_KEY") or os.getenv("CONDUCTOR_API_KEY")
    model_label = _coda_model_label()
    if not api_key:
        return _error_response(
            status=0,
            message="Missing CODA_API_KEY (or CONDUCTOR_API_KEY) in environment for Coda.",
            model=model_label,
        )

    base_url = os.getenv("CODA_API_BASE_URL", DEFAULT_CODA_BASE_URL).rstrip("/")
    url = f"{base_url}/agents"

    body = _build_coda_body(request_payload)
    payload_json = _wrap_coda_payload(body)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream, application/json",
    }
    print(f"Sending request to coda ({body.get('mode')}, fast={body.get('fast', False)})")

    stream_retries = _env_int("CODA_STREAM_MAX_RETRIES", _env_int("CODA_MAX_RETRIES", _DEFAULT_CODA_MAX_RETRIES))
    last_stream_error = ""
    last_events: List[Dict[str, Any]] = []

    for attempt in range(stream_retries + 1):
        try:
            response = _post_coda_with_retry(
                url,
                json=payload_json,
                headers=headers,
                timeout=timeout,
                stream=True,
            )
        except requests.exceptions.RequestException as exc:
            return _error_response(
                status=0,
                message=f"Coda request failed after retries: {exc}",
                model=model_label,
            )

        if response.status_code != 200:
            try:
                body_text = response.text[:4000]
            except (requests.exceptions.RequestException, UnicodeDecodeError):
                body_text = "<unreadable body>"
            return _error_response(
                status=response.status_code,
                message=body_text,
                model=model_label,
            )

        text, raw_events = _read_coda_stream(response)
        last_events = raw_events
        entry_point = _entry_point_from_payload(request_payload)
        if entry_point and f"def {entry_point}" not in text:
            structured_text = _structured_code_from_events(raw_events, entry_point)
            if structured_text:
                text = structured_text
        stream_error = _coda_runtime_error(text, raw_events)
        if stream_error:
            last_stream_error = stream_error
            try:
                response.close()
            except (requests.exceptions.RequestException, AttributeError):
                pass
            if attempt < stream_retries:
                delay = min(
                    _env_float("CODA_RETRY_MAX_DELAY", _DEFAULT_CODA_RETRY_MAX_DELAY),
                    _env_float("CODA_RETRY_BASE_DELAY", _DEFAULT_CODA_RETRY_BASE_DELAY) * (2 ** attempt),
                )
                print(
                    f"Coda stream/runtime error: {stream_error[:200]}; "
                    f"retrying in {delay:.1f}s (attempt {attempt + 1}/{stream_retries})",
                    file=sys.stderr,
                )
                time.sleep(delay)
                continue
            return _error_response(
                status=0,
                message=f"Coda stream/runtime error after retries: {stream_error}",
                model=model_label,
            )

        if not text:
            # No assistant text could be extracted - surface the raw events so
            # the user can debug parser tuning instead of silently returning ""
            snippet = json.dumps(raw_events[:5], default=str)[:2000]
            return _error_response(
                status=200,
                message=f"Coda returned no extractable assistant text. First events: {snippet}",
                model=model_label,
            )

        return {
            "id": f"coda-{int(time.time() * 1000)}",
            "model": model_label,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "native_finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": {},
            "coda_events": raw_events,
        }

    if last_stream_error:
        return _error_response(
            status=0,
            message=f"Coda stream/runtime error after retries: {last_stream_error}",
            model=model_label,
        )
    snippet = json.dumps(last_events[:5], default=str)[:2000]
    return _error_response(
        status=200,
        message=f"Coda returned no extractable assistant text. First events: {snippet}",
        model=model_label,
    )


def _coda_model_label() -> str:
    mode = os.getenv("CODA_AGENT_MODE", "build")
    return f"coda/{mode}"


def _coda_request_timeout(default_timeout: int) -> int:
    """Coda agent runs can exceed raw chat-completion timeouts."""
    if os.getenv("CODA_REQUEST_TIMEOUT_SECONDS"):
        return _env_int("CODA_REQUEST_TIMEOUT_SECONDS", default_timeout)
    return max(default_timeout, 900)


def _post_coda_with_retry(url: str, **kwargs: Any) -> requests.Response:
    """POST to ``url`` with exponential backoff for transient errors.

    Retries on:

    * Any ``requests.exceptions.RequestException`` (timeouts, connection
      resets, DNS failures, etc.).
    * HTTP statuses in :data:`_CODA_RETRYABLE_STATUS` — gateway-side errors
      (502/503/504), throttling (408/429), generic 500s, and the bare 303s
      the Coda gateway has been observed to emit under load.

    On a successful (or non-retryable) response, the live ``Response`` is
    returned with its body still streamable. If every attempt is exhausted
    the last response is returned (so the caller can extract its body for
    the error envelope), or the final exception is re-raised if no response
    was ever obtained.

    Honours a ``Retry-After`` response header when present (used by Coda for
    429s); otherwise falls back to ``base_delay * 2**attempt`` capped by
    ``max_delay``.

    Tunable via env vars:

    * ``CODA_MAX_RETRIES`` (default 4)
    * ``CODA_RETRY_BASE_DELAY`` seconds (default 1.0)
    * ``CODA_RETRY_MAX_DELAY`` seconds (default 30.0)
    """
    max_retries = _env_int("CODA_MAX_RETRIES", _DEFAULT_CODA_MAX_RETRIES)
    base_delay = _env_float("CODA_RETRY_BASE_DELAY", _DEFAULT_CODA_RETRY_BASE_DELAY)
    max_delay = _env_float("CODA_RETRY_MAX_DELAY", _DEFAULT_CODA_RETRY_MAX_DELAY)

    last_response: Optional[requests.Response] = None
    last_exception: Optional[BaseException] = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, **kwargs)
        except requests.exceptions.RequestException as exc:
            last_exception = exc
            last_response = None
            if attempt >= max_retries:
                raise
            delay = min(max_delay, base_delay * (2 ** attempt))
            print(
                f"Coda request raised {type(exc).__name__}: {exc}; "
                f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})",
                file=sys.stderr,
            )
            time.sleep(delay)
            continue

        if response.status_code in _CODA_RETRYABLE_STATUS:
            last_response = response
            last_exception = None
            if attempt >= max_retries:
                return response
            delay = _retry_delay(response, attempt, base_delay, max_delay)
            print(
                f"Coda returned HTTP {response.status_code}; "
                f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})",
                file=sys.stderr,
            )
            try:
                response.close()
            except (requests.exceptions.RequestException, AttributeError):
                pass
            time.sleep(delay)
            continue

        return response

    # Defensive: the loop above always returns or raises, but if a future
    # refactor breaks that invariant we still want a sensible fallback.
    if last_response is not None:
        return last_response
    raise last_exception or requests.exceptions.RequestException(
        "Coda retry loop exhausted without obtaining a response"
    )


def _retry_delay(
    response: requests.Response,
    attempt: int,
    base_delay: float,
    max_delay: float,
) -> float:
    """Compute the backoff delay, honouring ``Retry-After`` when supplied."""
    retry_after = response.headers.get("Retry-After") if response.headers else None
    if retry_after:
        try:
            return min(max_delay, max(0.0, float(retry_after)))
        except ValueError:
            pass
    return min(max_delay, base_delay * (2 ** attempt))


def _build_coda_body(request_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Project an OpenRouter-shaped request onto Coda's ``AgentsRequest`` schema.

    The body is built by allowlisting the three fields Coda accepts
    (``messages``, ``mode``, ``fast``) rather than by stripping the
    OpenRouter-only ones. This keeps the projection robust if the prompt
    builders ever start emitting additional fields.
    """
    raw_messages: List[Dict[str, Any]] = list(request_payload.get("messages") or [])
    messages: List[Dict[str, str]] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if role not in _CODA_ALLOWED_ROLES:
            raise ValueError(
                f"Coda's AgentsMessageRole only accepts user|assistant; got '{role}'"
            )
        if not isinstance(content, str):
            content = str(content)
        messages.append({"role": role, "content": content})

    # Drop a trailing empty assistant message - parse_prompt always adds one
    # for OpenRouter prefill, but Coda treats it as a malformed turn.
    while messages and messages[-1]["role"] == "assistant" and not messages[-1]["content"].strip():
        messages.pop()

    if not messages:
        raise ValueError("Cannot send an empty message list to Coda.")

    body: Dict[str, Any] = {
        "messages": messages,
        "mode": os.getenv("CODA_AGENT_MODE", "build"),
        "fast": _env_truthy("CODA_AGENT_FAST", default=False),
    }
    return body


def _entry_point_from_payload(request_payload: Dict[str, Any]) -> str:
    for message in reversed(request_payload.get("messages") or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", content)
        if match:
            return match.group(1)
    return ""


def _structured_code_from_events(raw_events: List[Dict[str, Any]], entry_point: str) -> str:
    target = (os.getenv("CODA_TARGET_FRAMEWORK") or "").strip().lower()
    keys = [key for key in (target, "code", "qiskit", "cirq", "pennylane") if key]
    seen: set[str] = set()
    ordered_keys = [key for key in keys if not (key in seen or seen.add(key))]

    for event in reversed(raw_events):
        payloads = []
        if isinstance(event.get("structured_response"), dict):
            payloads.append(event["structured_response"])
        if isinstance(event.get("data"), dict):
            payloads.append(event["data"])
        if str(event.get("type") or "").lower() == "structured_response":
            payloads.append(event)

        for payload in payloads:
            for key in ordered_keys:
                value = payload.get(key)
                if isinstance(value, str) and f"def {entry_point}" in value:
                    return value.strip()
    return ""


def _wrap_coda_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the OpenAPI-direct vs Fern-``{"body": ...}`` escape hatch."""
    wrapper = os.getenv("CODA_AGENT_PAYLOAD_WRAPPER", "direct").strip().lower()
    if wrapper == "body":
        return {"body": body}
    if wrapper not in ("direct", ""):
        print(
            f"WARNING: Unknown CODA_AGENT_PAYLOAD_WRAPPER='{wrapper}'. Falling back to 'direct'.",
            file=sys.stderr,
        )
    return body


def _read_coda_stream(response: requests.Response) -> Tuple[str, List[Dict[str, Any]]]:
    """Read the Coda response, which may be SSE-framed or a single JSON object.

    Returns ``(assistant_text, raw_events)``. ``assistant_text`` is the best-
    effort concatenation of streamed deltas (or the full message from a
    terminal event, if one supplies it). ``raw_events`` is the parsed event
    list, retained for debugging.
    """
    raw_events: List[Dict[str, Any]] = []
    accumulated: List[str] = []
    final_text: Optional[str] = None
    last_event_type: Optional[str] = None

    saw_any_line = False
    try:
        for line in response.iter_lines(decode_unicode=True):
            saw_any_line = True
            if line is None:
                continue
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:"):
                last_event_type = line.split(":", 1)[1].strip()
                continue
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
            else:
                payload = line
            if not payload:
                continue
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except ValueError:
                # Not JSON - treat the line itself as plain text.
                accumulated.append(payload)
                continue
            if isinstance(event, dict) and last_event_type and "type" not in event:
                event = {**event, "type": last_event_type}
            raw_events.append(event)
            event_type = str(event.get("type") or "").lower()
            if (
                event_type == "token"
                and _coerce_text(event.get("content")).strip() == _CODA_DONE_SENTINEL
            ):
                break
            text, terminal = _extract_text_from_coda_event(event)
            if not text and event_type == "error":
                text = _coerce_text(event.get("message") or event.get("error"))
            if terminal and text:
                final_text = text
            elif text:
                accumulated.append(text)
            if terminal or event_type in {"completed", "structured_response", "stream_end", "error", "cancelled"}:
                break
    except requests.exceptions.ChunkedEncodingError as exc:
        raw_events.append({"type": "stream_error", "error": f"stream interrupted: {exc}"})

    if not saw_any_line:
        # Server returned a single body, not a stream. Fall back to JSON.
        try:
            body = response.json()
        except ValueError:
            body_text = response.text or ""
            return body_text.strip(), [{"raw": body_text[:4000]}]
        if isinstance(body, list):
            for event in body:
                if isinstance(event, dict):
                    raw_events.append(event)
                    text, terminal = _extract_text_from_coda_event(event)
                    if terminal and text:
                        final_text = text
                    elif text:
                        accumulated.append(text)
        elif isinstance(body, dict):
            raw_events.append(body)
            text, terminal = _extract_text_from_coda_event(body)
            if terminal and text:
                final_text = text
            elif text:
                accumulated.append(text)
        else:
            return str(body), [{"raw": body}]

    if final_text is not None:
        return final_text, raw_events
    return "".join(accumulated).strip(), raw_events


def _coda_runtime_error(text: str, raw_events: List[Dict[str, Any]]) -> str:
    """Detect Coda transport/runtime failures surfaced inside a 200 SSE stream."""
    for event in raw_events:
        if str(event.get("type") or "").lower() == "stream_error":
            error = event.get("error")
            return str(error or "Coda stream interrupted")

    stripped = (text or "").strip()
    lower = stripped.lower()
    if lower.startswith("agent call failed:"):
        return stripped
    if lower.startswith("agent timed out"):
        return stripped
    if lower.startswith("[stream interrupted:"):
        return stripped
    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Coda agent emits these event types as pipeline plumbing. They are NOT part
# of the assistant's response and would only pollute the extracted code.
_IGNORED_EVENT_TYPES = frozenset(
    {
        "run_received",
        "node_start",
        "node_end",
        "decision_router",
        "tool_call",
        "tool_result",
        "thinking_token",
        "heartbeat",
        "completed",
    }
)

# Event types that carry the complete final assistant message. When one of
# these arrives, the caller should treat its text as the canonical answer
# and discard any accumulated streaming tokens.
_TERMINAL_EVENT_TYPES = frozenset(
    {
        "structured_response",
        "done",
        "complete",
        "final",
        "final_message",
        "message_complete",
        "assistant_message",
        "message",
        "response",
        "finish",
    }
)

# Coda emits this string as a regular `token` event right before `completed`.
_CODA_DONE_SENTINEL = "<DONE>"

_TEXT_KEYS = ("content", "text", "delta", "message", "data", "output", "value")


def _extract_text_from_coda_event(event: Any) -> Tuple[str, bool]:
    """Best-effort extraction of assistant text from a Coda event payload.

    Returns ``(text, is_terminal_full_message)``. When ``is_terminal_full_message``
    is True the caller should treat ``text`` as the complete assistant message
    and discard accumulated deltas. When ``text`` is empty the caller should
    skip the event entirely.
    """
    if not isinstance(event, dict):
        return ("", False)
    event_type = str(event.get("type") or "").lower()

    # Skip pure-metadata events. These are pipeline plumbing, not response.
    if event_type in _IGNORED_EVENT_TYPES:
        return ("", False)

    is_terminal = event_type in _TERMINAL_EVENT_TYPES

    # Coda's `structured_response` event carries the *post-pipeline* output:
    # the agent runs validation, transpilation, and a QASM round-trip, then
    # emits a `data.code` field that is the transpiled flat circuit. For the
    # QuanBench+ benchmark we want the LLM's *original* function-form output
    # (the streamed `token` events), so we ignore `structured_response` by
    # default. Set `CODA_PREFER_STRUCTURED_RESPONSE=1` to opt back in (e.g. to
    # benchmark Coda's full pipeline rather than just its LLM step).
    if event_type == "structured_response":
        if not _env_truthy("CODA_PREFER_STRUCTURED_RESPONSE", default=False):
            return ("", False)
        data = event.get("data")
        if isinstance(data, dict):
            target = (os.getenv("CODA_TARGET_FRAMEWORK") or "").strip().lower()
            if target:
                value = data.get(target)
                if isinstance(value, str) and value.strip():
                    return (value, True)
            value = data.get("code")
            if isinstance(value, str) and value.strip():
                return (value, True)
        # Fall through to generic shape walker below.

    # Streaming response tokens (Coda's main response surface).
    if event_type == "token":
        text = _coerce_text(event.get("content"))
        if text.strip() == _CODA_DONE_SENTINEL:
            return ("", False)
        return (text, False)

    # Prefer the most specific known shapes first.
    for key in ("delta", "content", "text"):
        value = event.get(key)
        text = _coerce_text(value)
        if text:
            return (text, is_terminal)

    # Nested message.content (OpenAI-style) and choices[].message.content.
    msg = event.get("message")
    if isinstance(msg, dict):
        text = _coerce_text(msg.get("content"))
        if text:
            return (
                text,
                True
                if event_type in _TERMINAL_EVENT_TYPES or "message" in event_type
                else is_terminal,
            )

    choices = event.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            inner_msg = first.get("message")
            if isinstance(inner_msg, dict):
                text = _coerce_text(inner_msg.get("content"))
                if text:
                    return (text, True)
            inner_delta = first.get("delta")
            if isinstance(inner_delta, dict):
                text = _coerce_text(inner_delta.get("content"))
                if text:
                    return (text, is_terminal)

    # Final fallback: scan a small set of common text keys.
    for key in _TEXT_KEYS:
        if key in ("delta", "content", "text", "message"):
            continue
        text = _coerce_text(event.get(key))
        if text:
            return (text, is_terminal)

    return ("", is_terminal)


def _coerce_text(value: Any) -> str:
    """Pull a text string out of common LLM event payloads."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # OpenAI-style {"type": "text", "text": "..."}
        for key in ("text", "content", "value"):
            inner = value.get(key)
            if isinstance(inner, str) and inner:
                return inner
        return ""
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            text = _coerce_text(item)
            if text:
                parts.append(text)
        return "".join(parts)
    return ""


def _extract_chat_completion(request_payload: Dict[str, Any]) -> str:
    """Replicate the messages[1].content extraction the legacy code did."""
    messages = request_payload.get("messages") or []
    if len(messages) > 1 and isinstance(messages[1], dict):
        content = messages[1].get("content")
        if isinstance(content, str):
            return content
    return ""


def _env_truthy(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(0, int(raw.strip()))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return default


def _error_response(*, status: int, message: str, model: Optional[str]) -> Dict[str, Any]:
    """Build an OpenRouter-shaped error envelope the existing parsers tolerate."""
    return {
        "id": f"error-{int(time.time() * 1000)}",
        "model": model or "unknown",
        "error": {"status_code": status, "body": message},
        "choices": [
            {
                "index": 0,
                "finish_reason": "error",
                "native_finish_reason": "error",
                "message": {"role": "assistant", "content": f"# Error: {message[:1000]}"},
            }
        ],
        "usage": {},
    }


# ---------------------------------------------------------------------------
# Smoke test entry point
# ---------------------------------------------------------------------------
def _smoke_main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Smoke-test a generation provider with one prompt."
    )
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default=CODA_PROVIDER,
        help="Provider to call (default: coda)",
    )
    parser.add_argument(
        "--prompt",
        default="Write a Python function that returns a Qiskit Bell state circuit. Reply with code only.",
        help="Single user prompt to send.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenRouter model id (only used for --provider openrouter).",
    )
    parser.add_argument(
        "--show-events",
        action="store_true",
        help="Print the raw Coda events alongside the normalized response.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    request_payload: Dict[str, Any] = {
        "messages": [{"role": "user", "content": args.prompt}],
    }
    if args.provider == OPENROUTER_PROVIDER:
        if not args.model:
            parser.error("--model is required when --provider=openrouter")
        request_payload["model"] = args.model
        request_payload["temperature"] = 0.0
        request_payload["stream"] = False

    response, prefill = send_generation_request(request_payload, args.provider)
    if not args.show_events and isinstance(response, dict):
        response = {k: v for k, v in response.items() if k != "coda_events"}
    print(json.dumps({"prefill": prefill, "response": response}, indent=2, default=str))
    # Surface error envelopes as non-zero exits so wrapper scripts can rely
    # on the process status instead of grepping stdout.
    if isinstance(response, dict) and "error" in response:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke_main())
