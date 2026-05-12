"""Unit tests for ``utils.providers``.

These cover the novel logic introduced for the Coda integration:

* Projection of an OpenRouter-shaped chat-completion request onto Coda's
  ``AgentsRequest`` schema (field stripping, role validation, trailing-empty
  assistant cleanup).
* SSE response parsing, including filtering of pipeline-plumbing events
  (``thinking_token``, ``tool_call``, etc.) and selection between streamed
  ``token`` events and ``structured_response`` events.
* The ``CODA_PREFER_STRUCTURED_RESPONSE`` / ``CODA_TARGET_FRAMEWORK``
  opt-in for using Coda's post-pipeline transpiled output.
* Plain-JSON fallback when the server returns a single body instead of
  an SSE stream.
* OpenRouter dispatch: missing-key envelope and request shape preserved.

The tests construct a minimal stub for ``requests.Response`` so nothing in
this file performs real network I/O.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Iterable, List, Optional

import pytest

from utils import providers


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
class FakeResponse:
    """Minimal stand-in for ``requests.Response`` used by the provider.

    Supports either an SSE-framed iter_lines stream or a single JSON body
    accessed via ``.json()``/``.text``.
    """

    def __init__(
        self,
        *,
        status_code: int = 200,
        sse_lines: Optional[Iterable[str]] = None,
        json_body: Any = None,
        text: str = "",
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.status_code = status_code
        self._sse_lines = list(sse_lines) if sse_lines is not None else None
        self._json_body = json_body
        self.text = text or (json.dumps(json_body) if json_body is not None else "")
        self.headers: Dict[str, str] = dict(headers) if headers else {}

    def iter_lines(self, decode_unicode: bool = True) -> Iterable[str]:
        if self._sse_lines is None:
            return iter([])
        return iter(self._sse_lines)

    def json(self) -> Any:
        if self._json_body is None:
            raise ValueError("no json body")
        return self._json_body

    def close(self) -> None:
        pass


def _sse(events: Iterable[Dict[str, Any]]) -> List[str]:
    """Render a list of event dicts as SSE ``data: {...}`` lines."""
    out: List[str] = []
    for event in events:
        out.append(f"data: {json.dumps(event)}")
        out.append("")
    out.append("data: [DONE]")
    return out


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Each test starts with a clean Coda env so order can't matter."""
    for var in (
        "CODA_API_KEY",
        "CONDUCTOR_API_KEY",
        "CODA_API_BASE_URL",
        "CODA_AGENT_MODE",
        "CODA_AGENT_FAST",
        "CODA_AGENT_PAYLOAD_WRAPPER",
        "CODA_PREFER_STRUCTURED_RESPONSE",
        "CODA_TARGET_FRAMEWORK",
        "CODA_STREAM_MAX_RETRIES",
        "API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------
def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        providers.send_generation_request({"messages": []}, "anthropic")


def test_send_generation_request_returns_prefill_tuple(monkeypatch):
    monkeypatch.setenv("CODA_API_KEY", "test-key")
    monkeypatch.setattr(
        providers.requests,
        "post",
        lambda *a, **kw: FakeResponse(sse_lines=_sse([{"type": "token", "content": "ok"}])),
    )
    payload = {
        "messages": [
            {"role": "user", "content": "first user"},
            {"role": "user", "content": "second user (used as prefill)"},
        ]
    }
    response, prefill = providers.send_generation_request(payload, "coda")
    assert response["choices"][0]["message"]["content"] == "ok"
    assert prefill == "second user (used as prefill)"


def test_send_generation_request_dict_strips_prefill(monkeypatch):
    monkeypatch.setenv("CODA_API_KEY", "test-key")
    monkeypatch.setattr(
        providers.requests,
        "post",
        lambda *a, **kw: FakeResponse(sse_lines=_sse([{"type": "token", "content": "x"}])),
    )
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert response["choices"][0]["message"]["content"] == "x"


# ---------------------------------------------------------------------------
# OpenRouter dispatch
# ---------------------------------------------------------------------------
def test_openrouter_missing_key_returns_error_envelope(monkeypatch):
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}], "model": "openai/gpt-4.1"},
        "openrouter",
    )
    assert response["error"]["status_code"] == 0
    assert "API_KEY" in response["error"]["body"]
    assert response["choices"][0]["finish_reason"] == "error"


def test_openrouter_passes_request_through_unmodified(monkeypatch):
    captured: Dict[str, Any] = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return FakeResponse(
            status_code=200,
            json_body={
                "id": "x",
                "model": "openai/gpt-4.1",
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            },
        )

    monkeypatch.setenv("API_KEY", "or-key")
    monkeypatch.setattr(providers.requests, "post", fake_post)
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "openai/gpt-4.1",
        "temperature": 0.7,
    }
    response = providers.send_generation_request_dict(payload, "openrouter")
    assert captured["url"] == providers.OPENROUTER_URL
    assert captured["json"] == payload  # OpenRouter path must NOT mutate the request.
    assert response["choices"][0]["message"]["content"] == "hi"


# ---------------------------------------------------------------------------
# Coda body projection
# ---------------------------------------------------------------------------
def test_coda_body_strips_openrouter_only_fields(monkeypatch):
    monkeypatch.setenv("CODA_API_KEY", "k")
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "ignored/by/coda",
        "stream": False,
        "temperature": 0.8,
        "top_p": 0.9,
        "n": 1,
        "reasoning": {"effort": "high"},
        "provider": {"order": ["x"]},
    }
    body = providers._build_coda_body(payload)
    assert set(body.keys()) == {"messages", "mode", "fast"}
    assert body["mode"] == "build"
    assert body["fast"] is False


def test_coda_body_drops_trailing_empty_assistant():
    body = providers._build_coda_body(
        {
            "messages": [
                {"role": "user", "content": "write a function"},
                {"role": "assistant", "content": "   "},
                {"role": "assistant", "content": ""},
            ]
        }
    )
    assert body["messages"] == [{"role": "user", "content": "write a function"}]


def test_coda_body_preserves_multiturn_history():
    body = providers._build_coda_body(
        {
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "first attempt"},
                {"role": "user", "content": "feedback"},
            ]
        }
    )
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][-1]["content"] == "feedback"


def test_coda_body_rejects_system_role():
    with pytest.raises(ValueError, match="AgentsMessageRole"):
        providers._build_coda_body(
            {"messages": [{"role": "system", "content": "you are helpful"}]}
        )


def test_coda_body_rejects_empty_messages():
    with pytest.raises(ValueError, match="empty message list"):
        providers._build_coda_body(
            {"messages": [{"role": "assistant", "content": ""}]}
        )


def test_coda_body_coerces_non_string_content():
    body = providers._build_coda_body(
        {"messages": [{"role": "user", "content": 42}]}
    )
    assert body["messages"][0]["content"] == "42"


def test_coda_payload_wrapper_direct(monkeypatch):
    body = {"messages": [], "mode": "build", "fast": False}
    assert providers._wrap_coda_payload(body) is body


def test_coda_payload_wrapper_body(monkeypatch):
    monkeypatch.setenv("CODA_AGENT_PAYLOAD_WRAPPER", "body")
    body = {"messages": [], "mode": "build", "fast": False}
    wrapped = providers._wrap_coda_payload(body)
    assert wrapped == {"body": body}


def test_coda_mode_and_fast_overrides(monkeypatch):
    monkeypatch.setenv("CODA_AGENT_MODE", "learn")
    monkeypatch.setenv("CODA_AGENT_FAST", "true")
    body = providers._build_coda_body(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    assert body["mode"] == "learn"
    assert body["fast"] is True


def test_coda_body_preserves_user_prompt_without_benchmark_instructions():
    body = providers._build_coda_body(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "I need you to complete the following code.\ndef qpe_x_gate():",
                }
            ]
        }
    )
    assert body["messages"][0]["content"] == (
        "I need you to complete the following code.\ndef qpe_x_gate():"
    )


# ---------------------------------------------------------------------------
# Coda SSE parser
# ---------------------------------------------------------------------------
def test_token_events_concatenate_in_order():
    text, events = providers._read_coda_stream(
        FakeResponse(
            sse_lines=_sse(
                [
                    {"type": "token", "content": "def foo():\n"},
                    {"type": "token", "content": "    return 1\n"},
                ]
            )
        )
    )
    assert text == "def foo():\n    return 1"
    assert len(events) == 2


def test_pipeline_events_are_ignored():
    text, _ = providers._read_coda_stream(
        FakeResponse(
            sse_lines=_sse(
                [
                    {"type": "run_received"},
                    {"type": "node_start", "name": "router"},
                    {"type": "thinking_token", "content": "let me think..."},
                    {"type": "tool_call", "name": "simulate"},
                    {"type": "tool_result", "ok": True},
                    {"type": "decision_router", "branch": "code"},
                    {"type": "node_end", "name": "router"},
                    {"type": "heartbeat"},
                    {"type": "token", "content": "the_real_code"},
                    {"type": "completed"},
                ]
            )
        )
    )
    assert text == "the_real_code"


def test_done_sentinel_is_filtered():
    text, _ = providers._read_coda_stream(
        FakeResponse(
            sse_lines=_sse(
                [
                    {"type": "token", "content": "hello"},
                    {"type": "token", "content": "<DONE>"},
                ]
            )
        )
    )
    assert text == "hello"


def test_done_sentinel_stops_stream_read():
    class DoneThenBroken(FakeResponse):
        def iter_lines(self, decode_unicode: bool = True):
            yield 'data: {"type": "token", "content": "hello"}'
            yield ""
            yield "data: [DONE]"
            yield ""
            raise AssertionError("read past done")

    text, _ = providers._read_coda_stream(DoneThenBroken())
    assert text == "hello"


def test_completed_event_stops_stream_read():
    class CompletedThenBroken(FakeResponse):
        def iter_lines(self, decode_unicode: bool = True):
            yield 'data: {"type": "token", "content": "hello"}'
            yield ""
            yield 'data: {"type": "completed"}'
            yield ""
            raise AssertionError("read past completed")

    text, _ = providers._read_coda_stream(CompletedThenBroken())
    assert text == "hello"


def test_stream_end_event_stops_stream_read():
    class StreamEndThenBroken(FakeResponse):
        def iter_lines(self, decode_unicode: bool = True):
            yield 'data: {"type": "token", "content": "hello"}'
            yield ""
            yield 'data: {"type": "stream_end"}'
            yield ""
            raise AssertionError("read past stream_end")

    text, _ = providers._read_coda_stream(StreamEndThenBroken())
    assert text == "hello"


def test_structured_response_event_stops_stream_read():
    class StructuredThenBroken(FakeResponse):
        def iter_lines(self, decode_unicode: bool = True):
            yield 'data: {"type": "token", "content": "hello"}'
            yield ""
            yield 'data: {"type": "structured_response", "data": {"code": "ignored"}}'
            yield ""
            raise AssertionError("read past structured_response")

    text, _ = providers._read_coda_stream(StructuredThenBroken())
    assert text == "hello"


def test_error_event_stops_stream_read():
    class ErrorThenBroken(FakeResponse):
        def iter_lines(self, decode_unicode: bool = True):
            yield 'event: error'
            yield 'data: {"message": "boom"}'
            yield ""
            raise AssertionError("read past error")

    text, _ = providers._read_coda_stream(ErrorThenBroken())
    assert text == "boom"


def test_structured_response_ignored_by_default():
    text, _ = providers._read_coda_stream(
        FakeResponse(
            sse_lines=_sse(
                [
                    {"type": "token", "content": "function-form code"},
                    {
                        "type": "structured_response",
                        "data": {"code": "transpiled-flat-circuit"},
                    },
                ]
            )
        )
    )
    assert text == "function-form code"


def test_structured_response_used_when_opted_in(monkeypatch):
    monkeypatch.setenv("CODA_PREFER_STRUCTURED_RESPONSE", "1")
    text, _ = providers._read_coda_stream(
        FakeResponse(
            sse_lines=_sse(
                [
                    {"type": "token", "content": "function-form code"},
                    {
                        "type": "structured_response",
                        "data": {"code": "transpiled"},
                    },
                ]
            )
        )
    )
    assert text == "transpiled"


def test_structured_response_picks_target_framework(monkeypatch):
    monkeypatch.setenv("CODA_PREFER_STRUCTURED_RESPONSE", "1")
    monkeypatch.setenv("CODA_TARGET_FRAMEWORK", "qiskit")
    text, _ = providers._read_coda_stream(
        FakeResponse(
            sse_lines=_sse(
                [
                    {
                        "type": "structured_response",
                        "data": {
                            "qiskit": "qiskit-code",
                            "cirq": "cirq-code",
                            "code": "fallback",
                        },
                    },
                ]
            )
        )
    )
    assert text == "qiskit-code"


def test_event_marker_is_attached_to_data_payload():
    text, events = providers._read_coda_stream(
        FakeResponse(
            sse_lines=[
                "event: token",
                'data: {"content": "via marker"}',
                "",
                "data: [DONE]",
            ]
        )
    )
    assert text == "via marker"
    assert events[0]["type"] == "token"


def test_chunked_encoding_error_marks_stream_error():
    class FlakyResponse(FakeResponse):
        def iter_lines(self, decode_unicode: bool = True):
            from requests.exceptions import ChunkedEncodingError

            yield 'data: {"type": "token", "content": "partial"}'
            yield ""
            raise ChunkedEncodingError("conn reset")

    text, events = providers._read_coda_stream(FlakyResponse())
    assert "partial" in text
    assert events[-1]["type"] == "stream_error"
    assert "stream interrupted" in events[-1]["error"]


def test_plain_json_fallback_when_no_stream_lines():
    text, events = providers._read_coda_stream(
        FakeResponse(
            json_body={
                "type": "assistant_message",
                "message": {"role": "assistant", "content": "single-shot reply"},
            }
        )
    )
    assert text == "single-shot reply"
    assert events  # the body became a one-element event list.


def test_openai_style_choices_message_content():
    text, _ = providers._read_coda_stream(
        FakeResponse(
            sse_lines=_sse(
                [
                    {
                        "type": "final",
                        "choices": [
                            {"message": {"role": "assistant", "content": "from-choices"}}
                        ],
                    }
                ]
            )
        )
    )
    assert text == "from-choices"


# ---------------------------------------------------------------------------
# Coda end-to-end (with mocked transport)
# ---------------------------------------------------------------------------
def test_coda_missing_key_returns_error_envelope():
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert response["error"]["status_code"] == 0
    assert "CODA_API_KEY" in response["error"]["body"]
    assert response["choices"][0]["finish_reason"] == "error"


def test_coda_uses_conductor_alias(monkeypatch):
    monkeypatch.setenv("CONDUCTOR_API_KEY", "alias-key")
    captured: Dict[str, Any] = {}

    def fake_post(url, **kwargs):
        captured["headers"] = kwargs.get("headers")
        return FakeResponse(sse_lines=_sse([{"type": "token", "content": "ok"}]))

    monkeypatch.setattr(providers.requests, "post", fake_post)
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert captured["headers"]["Authorization"] == "Bearer alias-key"
    assert response["choices"][0]["message"]["content"] == "ok"


def test_coda_non_200_surfaces_status_and_body(monkeypatch):
    monkeypatch.setenv("CODA_API_KEY", "k")
    monkeypatch.setattr(
        providers.requests,
        "post",
        lambda *a, **kw: FakeResponse(
            status_code=422,
            text=json.dumps({"detail": [{"loc": ["messages"], "msg": "bad"}]}),
        ),
    )
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert response["error"]["status_code"] == 422
    assert "messages" in response["error"]["body"]


def test_coda_empty_extracted_text_returns_error(monkeypatch):
    monkeypatch.setenv("CODA_API_KEY", "k")
    monkeypatch.setattr(
        providers.requests,
        "post",
        lambda *a, **kw: FakeResponse(
            sse_lines=_sse([{"type": "thinking_token", "content": "thinking only"}])
        ),
    )
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert response["error"]["status_code"] == 200
    assert "no extractable assistant text" in response["error"]["body"]


def test_coda_falls_back_to_completed_structured_code(monkeypatch):
    monkeypatch.setenv("CODA_API_KEY", "k")
    monkeypatch.setattr(
        providers.requests,
        "post",
        lambda *a, **kw: FakeResponse(
            sse_lines=_sse(
                [
                    {"type": "token", "content": "Here is what I would do."},
                    {
                        "type": "completed",
                        "structured_response": {"code": "def foo():\n    return 1"},
                    },
                    {"type": "stream_end"},
                ]
            )
        ),
    )
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "def foo():\n    pass"}]}, "coda"
    )
    assert response["choices"][0]["message"]["content"] == "def foo():\n    return 1"


def test_coda_request_uses_configured_base_url(monkeypatch):
    monkeypatch.setenv("CODA_API_KEY", "k")
    monkeypatch.setenv("CODA_API_BASE_URL", "https://custom.example/v9/coda/")
    captured: Dict[str, Any] = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        return FakeResponse(sse_lines=_sse([{"type": "token", "content": "ok"}]))

    monkeypatch.setattr(providers.requests, "post", fake_post)
    providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert captured["url"] == "https://custom.example/v9/coda/agents"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_coerce_text_handles_nested_content_blocks():
    value = [
        {"type": "text", "text": "alpha "},
        {"type": "text", "text": "beta"},
    ]
    assert providers._coerce_text(value) == "alpha beta"


def test_coerce_text_returns_empty_for_unknown_shape():
    assert providers._coerce_text({"unrecognised": True}) == ""
    assert providers._coerce_text(None) == ""
    assert providers._coerce_text(7) == ""


def test_env_truthy_recognises_common_values(monkeypatch):
    monkeypatch.setenv("X", "true")
    assert providers._env_truthy("X", default=False) is True
    monkeypatch.setenv("X", "no")
    assert providers._env_truthy("X", default=True) is False
    monkeypatch.delenv("X")
    assert providers._env_truthy("X", default=True) is True


def test_env_int_handles_bad_values(monkeypatch):
    monkeypatch.setenv("N", "7")
    assert providers._env_int("N", 0) == 7
    monkeypatch.setenv("N", "not-an-int")
    assert providers._env_int("N", 3) == 3
    monkeypatch.setenv("N", "")
    assert providers._env_int("N", 5) == 5


def test_env_float_handles_bad_values(monkeypatch):
    monkeypatch.setenv("F", "1.5")
    assert providers._env_float("F", 0.0) == 1.5
    monkeypatch.setenv("F", "garbage")
    assert providers._env_float("F", 9.0) == 9.0


# ---------------------------------------------------------------------------
# Coda retry behaviour
# ---------------------------------------------------------------------------
class _CountingResponse(FakeResponse):
    """Like FakeResponse but tracks ``close()`` calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def no_sleep(monkeypatch):
    """Replace ``time.sleep`` so retry tests don't actually wait."""
    delays: List[float] = []
    monkeypatch.setattr(providers.time, "sleep", lambda d: delays.append(d))
    return delays


def _scripted_post(responses):
    """Return a fake ``requests.post`` that returns the next scripted item.

    Items can be either ``FakeResponse``-like objects (returned) or
    ``Exception`` instances (raised).
    """
    iterator = iter(responses)

    def fake_post(url, **kwargs):
        item = next(iterator)
        if isinstance(item, BaseException):
            raise item
        return item

    return fake_post


def test_retry_succeeds_after_transient_502(monkeypatch, no_sleep):
    monkeypatch.setenv("CODA_API_KEY", "k")
    monkeypatch.setattr(
        providers.requests,
        "post",
        _scripted_post(
            [
                _CountingResponse(status_code=502, text="bad gateway"),
                _CountingResponse(status_code=502, text="bad gateway"),
                _CountingResponse(
                    sse_lines=_sse([{"type": "token", "content": "ok"}]),
                ),
            ]
        ),
    )
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert response["choices"][0]["message"]["content"] == "ok"
    # Two retries before success → two backoff sleeps.
    assert len(no_sleep) == 2
    # Default backoff: 1.0 then 2.0.
    assert no_sleep == [1.0, 2.0]


def test_retry_recovers_from_connection_error(monkeypatch, no_sleep):
    monkeypatch.setenv("CODA_API_KEY", "k")
    monkeypatch.setattr(
        providers.requests,
        "post",
        _scripted_post(
            [
                providers.requests.exceptions.ConnectionError("reset by peer"),
                _CountingResponse(
                    sse_lines=_sse([{"type": "token", "content": "ok"}]),
                ),
            ]
        ),
    )
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert response["choices"][0]["message"]["content"] == "ok"
    assert len(no_sleep) == 1


def test_retry_exhausted_returns_last_response(monkeypatch, no_sleep):
    monkeypatch.setenv("CODA_API_KEY", "k")
    monkeypatch.setenv("CODA_MAX_RETRIES", "2")
    monkeypatch.setattr(
        providers.requests,
        "post",
        _scripted_post(
            [
                _CountingResponse(status_code=502, text="bad gateway"),
                _CountingResponse(status_code=502, text="bad gateway"),
                _CountingResponse(status_code=502, text="bad gateway final"),
            ]
        ),
    )
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert response["error"]["status_code"] == 502
    assert "bad gateway final" in response["error"]["body"]
    assert len(no_sleep) == 2  # max_retries=2 → 2 sleeps before final attempt.


def test_retry_exhausted_after_persistent_connection_error(monkeypatch, no_sleep):
    monkeypatch.setenv("CODA_API_KEY", "k")
    monkeypatch.setenv("CODA_MAX_RETRIES", "1")
    monkeypatch.setattr(
        providers.requests,
        "post",
        _scripted_post(
            [
                providers.requests.exceptions.ConnectionError("reset 1"),
                providers.requests.exceptions.ConnectionError("reset 2"),
            ]
        ),
    )
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert response["error"]["status_code"] == 0
    assert "reset 2" in response["error"]["body"]


def test_no_retry_on_4xx_client_errors(monkeypatch, no_sleep):
    monkeypatch.setenv("CODA_API_KEY", "k")
    calls: List[str] = []

    def fake_post(url, **kwargs):
        calls.append(url)
        return _CountingResponse(status_code=401, text="unauthorized")

    monkeypatch.setattr(providers.requests, "post", fake_post)
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert response["error"]["status_code"] == 401
    assert len(calls) == 1  # No retries for 401.
    assert no_sleep == []


def test_no_retry_on_422_validation_error(monkeypatch, no_sleep):
    """422s come from the request being malformed; retrying won't help."""
    monkeypatch.setenv("CODA_API_KEY", "k")
    calls: List[str] = []

    def fake_post(url, **kwargs):
        calls.append(url)
        return _CountingResponse(
            status_code=422,
            text=json.dumps({"detail": [{"msg": "messages required"}]}),
        )

    monkeypatch.setattr(providers.requests, "post", fake_post)
    providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert len(calls) == 1


def test_retry_after_header_is_honoured(monkeypatch, no_sleep):
    monkeypatch.setenv("CODA_API_KEY", "k")
    monkeypatch.setattr(
        providers.requests,
        "post",
        _scripted_post(
            [
                _CountingResponse(
                    status_code=429, headers={"Retry-After": "5"}, text="slow down"
                ),
                _CountingResponse(
                    sse_lines=_sse([{"type": "token", "content": "ok"}])
                ),
            ]
        ),
    )
    providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert no_sleep == [5.0]


def test_retry_after_header_capped_by_max_delay(monkeypatch, no_sleep):
    monkeypatch.setenv("CODA_API_KEY", "k")
    monkeypatch.setenv("CODA_RETRY_MAX_DELAY", "3")
    monkeypatch.setattr(
        providers.requests,
        "post",
        _scripted_post(
            [
                _CountingResponse(
                    status_code=429, headers={"Retry-After": "120"}, text=""
                ),
                _CountingResponse(
                    sse_lines=_sse([{"type": "token", "content": "ok"}])
                ),
            ]
        ),
    )
    providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert no_sleep == [3.0]


def test_max_retries_zero_disables_retries(monkeypatch, no_sleep):
    monkeypatch.setenv("CODA_API_KEY", "k")
    monkeypatch.setenv("CODA_MAX_RETRIES", "0")
    calls: List[str] = []

    def fake_post(url, **kwargs):
        calls.append(url)
        return _CountingResponse(status_code=502, text="bad gateway")

    monkeypatch.setattr(providers.requests, "post", fake_post)
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert response["error"]["status_code"] == 502
    assert len(calls) == 1
    assert no_sleep == []


def test_303_is_treated_as_retryable(monkeypatch, no_sleep):
    """The Coda gateway has been observed to return bare 303s under load."""
    monkeypatch.setenv("CODA_API_KEY", "k")
    monkeypatch.setattr(
        providers.requests,
        "post",
        _scripted_post(
            [
                _CountingResponse(status_code=303, text="see other"),
                _CountingResponse(
                    sse_lines=_sse([{"type": "token", "content": "ok"}])
                ),
            ]
        ),
    )
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert response["choices"][0]["message"]["content"] == "ok"
    assert len(no_sleep) == 1


def test_in_band_agent_failure_is_retried(monkeypatch, no_sleep):
    monkeypatch.setenv("CODA_API_KEY", "k")
    monkeypatch.setattr(
        providers.requests,
        "post",
        _scripted_post(
            [
                _CountingResponse(
                    sse_lines=_sse([{"type": "token", "content": "Agent call failed: HTTP 303"}])
                ),
                _CountingResponse(
                    sse_lines=_sse([{"type": "token", "content": "ok"}])
                ),
            ]
        ),
    )
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert response["choices"][0]["message"]["content"] == "ok"
    assert no_sleep == [1.0]


def test_in_band_agent_timeout_is_retried(monkeypatch, no_sleep):
    monkeypatch.setenv("CODA_API_KEY", "k")
    monkeypatch.setattr(
        providers.requests,
        "post",
        _scripted_post(
            [
                _CountingResponse(
                    sse_lines=_sse([{"type": "token", "content": "Agent timed out"}])
                ),
                _CountingResponse(
                    sse_lines=_sse([{"type": "token", "content": "ok"}])
                ),
            ]
        ),
    )
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert response["choices"][0]["message"]["content"] == "ok"
    assert no_sleep == [1.0]


def test_stream_error_becomes_error_envelope_after_retries(monkeypatch, no_sleep):
    monkeypatch.setenv("CODA_API_KEY", "k")
    monkeypatch.setenv("CODA_STREAM_MAX_RETRIES", "0")

    class FlakyResponse(_CountingResponse):
        def iter_lines(self, decode_unicode: bool = True):
            from requests.exceptions import ChunkedEncodingError

            yield 'data: {"type": "token", "content": "partial"}'
            raise ChunkedEncodingError("Response ended prematurely")

    monkeypatch.setattr(providers.requests, "post", lambda *a, **kw: FlakyResponse())
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    assert response["choices"][0]["finish_reason"] == "error"
    assert "stream/runtime error" in response["error"]["body"]


def test_retry_logs_to_stderr(monkeypatch, no_sleep, capsys):
    monkeypatch.setenv("CODA_API_KEY", "k")
    monkeypatch.setattr(
        providers.requests,
        "post",
        _scripted_post(
            [
                _CountingResponse(status_code=502, text="bad gateway"),
                _CountingResponse(
                    sse_lines=_sse([{"type": "token", "content": "ok"}])
                ),
            ]
        ),
    )
    providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": "hi"}]}, "coda"
    )
    err = capsys.readouterr().err
    assert "HTTP 502" in err
    assert "retrying" in err


# ---------------------------------------------------------------------------
# Bedrock (raw LLM, no agent harness)
# ---------------------------------------------------------------------------
class _StubBedrockClient:
    """Minimal stand-in for a boto3 ``bedrock-runtime`` client used in tests.

    Records every ``converse`` call and returns a configurable response so we
    can assert both wire shape (modelId, messages, additional fields) and
    response normalisation independently.
    """

    def __init__(self, response: Optional[Dict[str, Any]] = None, raise_with: Optional[Exception] = None) -> None:
        self.response = response or {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "```python\nqc = 1\n```"}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }
        self.raise_with = raise_with
        self.calls: List[Dict[str, Any]] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_with is not None:
            raise self.raise_with
        return self.response


@pytest.fixture
def stub_bedrock(monkeypatch):
    """Yield a ``_StubBedrockClient`` that the provider's ``boto3.client`` returns."""

    stub = _StubBedrockClient()

    class _StubBoto3Module:
        @staticmethod
        def client(service, region_name=None, config=None):  # noqa: ARG004
            assert service == "bedrock-runtime"
            stub.last_region = region_name
            return stub

    class _StubBotocoreConfigModule:
        @staticmethod
        def Config(**kwargs):  # noqa: N802 — mirrors botocore's class name
            return kwargs

    class _StubBotocoreExceptionsModule:
        BotoCoreError = type("BotoCoreError", (Exception,), {})
        ClientError = type("ClientError", (Exception,), {})

    import sys as _sys

    monkeypatch.setitem(_sys.modules, "boto3", _StubBoto3Module())
    monkeypatch.setitem(_sys.modules, "botocore", type("M", (), {}))
    monkeypatch.setitem(_sys.modules, "botocore.config", _StubBotocoreConfigModule())
    monkeypatch.setitem(_sys.modules, "botocore.exceptions", _StubBotocoreExceptionsModule())
    return stub


def _bedrock_payload(text="Complete this circuit", **extra):
    return {"messages": [{"role": "user", "content": text}], **extra}


def test_bedrock_sends_only_user_message_no_system_prompt(stub_bedrock, monkeypatch):
    monkeypatch.delenv("BEDROCK_MODEL", raising=False)
    monkeypatch.delenv("BEDROCK_REGION", raising=False)
    providers.send_generation_request_dict(
        _bedrock_payload("Build a Bell state in Qiskit"), "bedrock"
    )
    assert len(stub_bedrock.calls) == 1
    call = stub_bedrock.calls[0]
    assert call["modelId"] == providers.DEFAULT_BEDROCK_MODEL
    assert call["messages"] == [{"role": "user", "content": [{"text": "Build a Bell state in Qiskit"}]}]
    assert "system" not in call, "no system prompt should be sent in raw-LLM mode"
    assert "tools" not in call, "no tools should be sent in raw-LLM mode"
    assert "toolConfig" not in call, "no tool config should be sent in raw-LLM mode"


def test_bedrock_includes_adaptive_thinking_by_default(stub_bedrock):
    providers.send_generation_request_dict(_bedrock_payload(), "bedrock")
    fields = stub_bedrock.calls[0]["additionalModelRequestFields"]
    assert fields["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert fields["output_config"]["effort"] == "high"
    # Adaptive thinking requires temperature=1.0.
    assert stub_bedrock.calls[0]["inferenceConfig"]["temperature"] == 1.0


def test_bedrock_thinking_can_be_disabled(stub_bedrock, monkeypatch):
    monkeypatch.setenv("BEDROCK_THINKING", "false")
    providers.send_generation_request_dict(
        _bedrock_payload(), "bedrock"
    )
    call = stub_bedrock.calls[0]
    assert "additionalModelRequestFields" not in call
    # Without thinking the request honours the request_payload's temperature.
    assert call["inferenceConfig"]["temperature"] == 0.0


def test_bedrock_clamps_unknown_effort_to_high(stub_bedrock, monkeypatch):
    monkeypatch.setenv("BEDROCK_EFFORT", "xhigh")  # 4.7-only value, invalid for 4.6
    providers.send_generation_request_dict(_bedrock_payload(), "bedrock")
    fields = stub_bedrock.calls[0]["additionalModelRequestFields"]
    assert fields["output_config"]["effort"] == "high"


def test_bedrock_normalises_response_to_openrouter_shape(stub_bedrock):
    response, _ = providers.send_generation_request(
        _bedrock_payload(), "bedrock"
    )
    assert response["model"] == providers.DEFAULT_BEDROCK_MODEL
    assert response["choices"][0]["message"]["content"] == "```python\nqc = 1\n```"
    assert response["choices"][0]["finish_reason"] == "end_turn"
    assert response["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_bedrock_concatenates_multiple_text_blocks(stub_bedrock):
    stub_bedrock.response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "first "},
                    {"reasoningContent": {"reasoningText": {"text": "thinking"}}},
                    {"text": "second"},
                ],
            }
        },
        "stopReason": "end_turn",
        "usage": {},
    }
    response, _ = providers.send_generation_request(_bedrock_payload(), "bedrock")
    assert response["choices"][0]["message"]["content"] == "first second"


def test_bedrock_empty_user_message_is_an_error(stub_bedrock):
    response = providers.send_generation_request_dict(
        {"messages": [{"role": "user", "content": ""}]}, "bedrock"
    )
    assert response["error"]["status_code"] == 0
    assert "empty user message" in response["error"]["body"].lower()
    assert stub_bedrock.calls == []


def test_bedrock_propagates_client_error_into_envelope(stub_bedrock):
    err_cls = sys.modules["botocore.exceptions"].ClientError
    stub_bedrock.raise_with = err_cls("ThrottlingException")
    response = providers.send_generation_request_dict(_bedrock_payload(), "bedrock")
    assert response["error"]["status_code"] == 0
    assert "Bedrock converse failed" in response["error"]["body"]


def test_bedrock_no_extractable_text_is_a_200_error(stub_bedrock):
    stub_bedrock.response = {
        "output": {"message": {"role": "assistant", "content": [{"reasoningContent": {}}]}},
        "stopReason": "end_turn",
        "usage": {},
    }
    response = providers.send_generation_request_dict(_bedrock_payload(), "bedrock")
    assert response["error"]["status_code"] == 200
    assert "no extractable assistant text" in response["error"]["body"].lower()


def test_bedrock_honours_model_region_and_max_tokens_env(stub_bedrock, monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL", "global.anthropic.claude-opus-4-7")
    monkeypatch.setenv("BEDROCK_REGION", "us-east-1")
    monkeypatch.setenv("BEDROCK_MAX_TOKENS", "1234")
    providers.send_generation_request_dict(_bedrock_payload(), "bedrock")
    call = stub_bedrock.calls[0]
    assert call["modelId"] == "global.anthropic.claude-opus-4-7"
    assert stub_bedrock.last_region == "us-east-1"
    assert call["inferenceConfig"]["maxTokens"] == 1234


def test_bedrock_provider_in_supported_set():
    assert "bedrock" in providers.SUPPORTED_PROVIDERS
