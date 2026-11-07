from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from macd import config, connect
from macd.connect import (
    EXTENSION_ID,
    FIX_API_KEY,
    FIX_BROWSER,
    FIX_CONNECT,
    FIX_EXTENSION,
    FIX_INSTALL_CLAUDE,
    FIX_SIGN_IN,
    FIX_UPDATE_CLAUDE,
    NATIVE_HOST_NAME,
    ConnectChecker,
    check_all,
    check_chrome,
    check_claude,
    parse_version,
)


def make_claude(tmp_path: Path, version: str = "2.1.319 (Claude Code)", logged_in: bool | None = True,
                method: str = "claude.ai") -> tuple[config.Config, Any]:
    cli = tmp_path / "claude"
    cli.write_text("#!/bin/sh\n")
    cfg = config.Config()
    cfg.brain.cli_path = str(cli)

    def run(argv: list[str]) -> tuple[int, str]:
        if argv[1:] == ["--version"]:
            return 0, version + "\n"
        if argv[1:] == ["auth", "status"]:
            if logged_in is None:
                return 1, ""
            return 0, json.dumps({"loggedIn": logged_in, "authMethod": method, "email": "x@y"})
        return 1, ""

    return cfg, run


def make_chrome(tmp_path: Path, *, extension: bool = True, host: bool = True, origins_ok: bool = True,
                enabled: bool | None = True, browser: str = "Google/Chrome") -> tuple[Path, Path]:
    app = tmp_path / "Application Support"
    base = app / browser
    if extension:
        (base / "Default" / "Extensions" / EXTENSION_ID).mkdir(parents=True)
    if host:
        (base / "NativeMessagingHosts").mkdir(parents=True, exist_ok=True)
        origins = [f"chrome-extension://{EXTENSION_ID}/"] if origins_ok else ["chrome-extension://other/"]
        (base / "NativeMessagingHosts" / NATIVE_HOST_NAME).write_text(json.dumps(
            {"name": "com.anthropic.claude_code_browser_extension", "allowed_origins": origins}))
    cj = tmp_path / ".claude.json"
    data: dict[str, Any] = {"hasCompletedClaudeInChromeOnboarding": True, "oauthAccount": {"secret": "never-read"}}
    if enabled is not None:
        data[connect.ENABLED_BY_DEFAULT_KEY] = enabled
    cj.write_text(json.dumps(data))
    return app, cj


def test_parse_version() -> None:
    assert parse_version("2.1.319 (Claude Code)") == (2, 1, 319) and parse_version("garbage") is None


def test_claude_ok(tmp_path: Path) -> None:
    cfg, run = make_claude(tmp_path)
    c = check_claude(cfg, run=run, env={})
    assert c == {"installed": True, "path": str(tmp_path / "claude"), "version": "2.1.319", "min_ok": True,
                 "signed_in": "yes", "auth_method": "claude_ai", "fix": ""}


@pytest.mark.parametrize(
    ("kw", "env", "expect"),
    [({"version": "2.0.9"}, {}, {"min_ok": False, "fix": FIX_UPDATE_CLAUDE}),
     ({"logged_in": False}, {}, {"signed_in": "no", "fix": FIX_SIGN_IN}),
     ({"logged_in": None}, {}, {"signed_in": "no", "fix": FIX_SIGN_IN}),
     ({"method": "apiKey"}, {}, {"auth_method": "api_key", "fix": FIX_API_KEY}),
     ({}, {"ANTHROPIC_API_KEY": "sk-x"}, {"auth_method": "api_key", "fix": FIX_API_KEY})],
)
def test_claude_fixes(tmp_path: Path, kw: dict, env: dict, expect: dict) -> None:
    cfg, run = make_claude(tmp_path, **kw)
    c = check_claude(cfg, run=run, env=env)
    for k, v in expect.items():
        assert c[k] == v, k


def test_claude_missing(tmp_path: Path) -> None:
    cfg = config.Config()
    cfg.brain.cli_path = str(tmp_path / "nope")
    c = check_claude(cfg, run=lambda argv: (1, ""), env={})
    assert c["installed"] is False and c["path"] is None and c["fix"] == FIX_INSTALL_CLAUDE


def test_chrome_all_good(tmp_path: Path) -> None:
    app, cj = make_chrome(tmp_path)
    c = check_chrome(app_support=app, claude_json=cj, running=lambda: ["Google Chrome"], connected="connected")
    assert c == {"browser_running": True, "browsers": ["Google Chrome"], "extension_installed": True, "native_host": True,
                 "enabled_by_default": True, "onboarding_done": True, "connected": "connected", "fix": ""}


@pytest.mark.parametrize(
    ("chrome_kw", "running", "auth", "connected", "fix"),
    [({}, [], "claude_ai", "unknown", FIX_BROWSER),
     ({"extension": False}, ["Google Chrome"], "claude_ai", "unknown", FIX_EXTENSION),
     ({"host": False}, ["Google Chrome"], "claude_ai", "unknown", FIX_CONNECT),
     ({"origins_ok": False}, ["Google Chrome"], "claude_ai", "unknown", FIX_CONNECT),
     ({"enabled": False}, ["Google Chrome"], "claude_ai", "unknown", ""),
     ({"enabled": None}, ["Google Chrome"], "claude_ai", "unknown", ""),
     ({}, ["Google Chrome"], "claude_ai", "not_connected", FIX_CONNECT),
     ({}, ["Google Chrome"], "api_key", "unknown", FIX_API_KEY),
     ({}, [], "claude_ai", "unavailable_api_key", FIX_API_KEY)],
)
def test_chrome_fixes(tmp_path: Path, chrome_kw: dict, running: list[str], auth: str, connected: str, fix: str) -> None:
    app, cj = make_chrome(tmp_path, **chrome_kw)
    c = check_chrome(app_support=app, claude_json=cj, running=lambda: running, auth_method=auth, connected=connected)
    assert c["fix"] == fix
    if "enabled" in chrome_kw:
        assert c["enabled_by_default"] is chrome_kw["enabled"]


def test_other_browsers_and_missing_dirs(tmp_path: Path) -> None:
    app, cj = make_chrome(tmp_path, browser="BraveSoftware/Brave-Browser")
    c = check_chrome(app_support=app, claude_json=cj, running=lambda: ["Brave Browser"])
    assert c["extension_installed"] and c["native_host"] and c["fix"] == ""
    c2 = check_chrome(app_support=tmp_path / "none", claude_json=tmp_path / "none.json", running=list)
    assert c2["extension_installed"] is False and c2["native_host"] is False and c2["enabled_by_default"] is None


def test_check_all_ready(tmp_path: Path) -> None:
    cfg, run = make_claude(tmp_path)
    app, cj = make_chrome(tmp_path)
    d = check_all(cfg, run=run, env={}, app_support=app, claude_json=cj, running=lambda: ["Google Chrome"])
    assert d["ready"] is True and set(d) == {"claude", "chrome", "ready", "checked_at"}
    app_off, cj_off = make_chrome(tmp_path / "off", enabled=False)
    d_off = check_all(cfg, run=run, env={}, app_support=app_off, claude_json=cj_off, running=lambda: ["Google Chrome"])
    assert d_off["ready"] is True and d_off["chrome"]["enabled_by_default"] is False  # informational only
    d2 = check_all(cfg, run=run, env={}, app_support=app, claude_json=cj, running=list)
    assert d2["ready"] is False and d2["chrome"]["fix"] == FIX_BROWSER
    cfg_api, run_api = make_claude(tmp_path, method="apiKey")
    d3 = check_all(cfg_api, run=run_api, env={}, app_support=app, claude_json=cj, running=lambda: ["Google Chrome"])
    assert d3["ready"] is False and d3["chrome"]["fix"] == FIX_API_KEY


async def test_checker_cache_and_deep(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_all(cfg: config.Config, *, connected: str = "unknown", **kw: Any) -> dict[str, Any]:
        calls.append(connected)
        return {"claude": {}, "chrome": {"connected": connected}, "ready": False}

    monkeypatch.setattr(connect, "check_all", fake_all)

    class Brain:
        probes = 0

        async def connect_check(self) -> dict[str, Any]:
            Brain.probes += 1
            return {"auth": "ok", "auth_detail": "", "chrome": "connected", "chrome_detail": "Claude in Chrome extension connected.",
                    "cli_version": "2.1.319", "session_ok": True}

    hint = {"v": "disconnected"}
    ck = ConnectChecker(config.Config(), brain=Brain(), brain_chrome=lambda: hint["v"], ttl_s=0.2)
    first = ck.snapshot()
    assert first == {"claude": None, "chrome": None, "ready": False, "pending": True}
    await asyncio.sleep(0.02)
    assert calls == ["not_connected"] and ck.snapshot()["chrome"]["connected"] == "not_connected"
    assert await ck.command(False) is ck.snapshot() and calls == ["not_connected"]  # cached, polled cheaply
    deep = await ck.command(True)
    assert Brain.probes == 1 and deep["chrome"]["connected"] == "connected" and calls[-1] == "connected"
    assert deep["chrome"]["probe"]["chrome_detail"] == "Claude in Chrome extension connected."
    await asyncio.sleep(0.25)
    ck.snapshot()  # stale → background refresh
    await asyncio.sleep(0.02)
    assert len(calls) == 3 and Brain.probes == 1


async def test_checker_probe_failure_and_not_signed_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(connect, "check_all", lambda cfg, *, connected="unknown", **kw: {
        "claude": {"signed_in": "yes", "fix": ""}, "chrome": {"connected": connected}, "ready": True})

    class Broken:
        async def connect_check(self) -> dict[str, Any]:
            raise RuntimeError("boom")

    d = await ConnectChecker(config.Config(), brain=Broken(), brain_chrome=lambda: "unknown").command(True)
    assert d["chrome"]["connected"] == "unknown" and "probe" not in d["chrome"]

    class SignedOut:
        async def connect_check(self) -> dict[str, Any]:
            return {"auth": "not_signed_in", "chrome": "unknown"}

    d = await ConnectChecker(config.Config(), brain=SignedOut(), brain_chrome=lambda: "unknown").command(True)
    assert d["claude"]["signed_in"] == "no" and d["claude"]["fix"] == FIX_SIGN_IN and d["ready"] is False


def test_no_dashes_in_fix_strings() -> None:
    for name in dir(connect):
        if name.startswith("FIX_"):
            assert "—" not in getattr(connect, name) and "–" not in getattr(connect, name), name
