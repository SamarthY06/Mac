from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytest
from claude_agent_sdk import CLIConnectionError, CLINotFoundError, ProcessError

from macd.brain import errors
from macd.contracts import BrainErrorKind as K


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Invalid API key · Please run /login", K.AUTH),
        ("OAuth token has expired. Please obtain a new token", K.AUTH),
        ("API Error: 401 authentication_error", K.AUTH),
        ("Claude AI usage limit reached|1767225600", K.RATE_LIMIT),
        ("5-hour limit reached ∙ resets 3pm", K.RATE_LIMIT),
        ("API Error: 429 rate_limit_error", K.RATE_LIMIT),
        ("Credit balance is too low", K.RATE_LIMIT),
        ("getaddrinfo ENOTFOUND api.anthropic.com", K.NETWORK),
        ("Connection error.", K.NETWORK),
        ("API Error: 529 overloaded_error", K.NETWORK),
        ("Request timed out", K.TIMEOUT),
        ("something odd", K.INTERNAL),
    ],
)
def test_classify_text(text: str, kind: K) -> None:
    assert errors.classify_text(text).kind == kind


def test_auth_message() -> None:
    assert errors.classify_text("not logged in").message == errors.MESSAGES[K.AUTH]
    assert "sign in" in errors.MESSAGES[K.AUTH] and "run claude" in errors.MESSAGES[K.AUTH]


def test_rate_limit_reset_time_epoch() -> None:
    ts = int(dt.datetime(2026, 11, 7, 15, 0).timestamp())
    msg = errors.classify_text(f"Claude AI usage limit reached|{ts}").message
    assert "3 PM" in msg


def test_rate_limit_reset_time_text() -> None:
    assert "3pm" in errors.classify_text("5-hour limit reached ∙ resets 3pm (Europe/London)").message


def test_rate_limit_resets_at_param() -> None:
    ts = dt.datetime(2026, 1, 1, 9, 30).timestamp()
    assert "9:30 AM" in errors.classify_text("rate limit", resets_at=ts).message


def test_exceptions() -> None:
    assert errors.classify_exception(CLINotFoundError("nope")).kind == K.CLI_MISSING
    assert errors.classify_exception(TimeoutError()).kind == K.TIMEOUT
    assert errors.classify_exception(ConnectionResetError()).kind == K.NETWORK
    assert errors.classify_exception(ProcessError("failed", 1, "Invalid API key")).kind == K.AUTH
    assert errors.classify_exception(CLIConnectionError("x"), "ECONNREFUSED").kind == K.NETWORK
    assert errors.classify_exception(RuntimeError("weird")).kind == K.INTERNAL


@dataclass
class FakeResult:
    is_error: bool = True
    subtype: str = "success"
    result: str | None = None
    errors: list[str] | None = None
    api_error_status: int | None = None
    terminal_reason: str | None = None


def test_classify_result() -> None:
    assert errors.classify_result(FakeResult(is_error=False)) is None
    assert errors.classify_result(FakeResult(terminal_reason="aborted_streaming")) is None
    assert errors.classify_result(FakeResult(api_error_status=401)).kind == K.AUTH
    assert errors.classify_result(FakeResult(api_error_status=429)).kind == K.RATE_LIMIT
    assert errors.classify_result(FakeResult(api_error_status=529)).kind == K.NETWORK
    assert errors.classify_result(FakeResult(result="Invalid API key")).kind == K.AUTH
    assert errors.classify_result(FakeResult(subtype="error_max_turns")).kind == K.INTERNAL


def test_assistant_error_codes() -> None:
    assert errors.classify_assistant_error("authentication_failed").kind == K.AUTH
    assert errors.classify_assistant_error("rate_limit").kind == K.RATE_LIMIT
    assert errors.classify_assistant_error("server_error").kind == K.NETWORK
    assert errors.classify_assistant_error("unknown").kind == K.INTERNAL
