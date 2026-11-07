"""Release-readiness checks: can this Mac talk to Claude Code and Claude in Chrome?

`check_claude()` / `check_chrome()` are synchronous, read-only and fast on the file-system
side; the two `claude` subprocess probes (~0.5 s each) are the slow part, so the daemon runs
everything through `ConnectChecker` which caches for 10 s and refreshes in the background
(`status` never blocks on it). `doctor` and `macctl connect` print the same data.

Nothing here writes to ~/.claude.json or any Claude Code file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from macd import config

log = logging.getLogger(__name__)

EXTENSION_ID = "fcoeoabgfenejglbffodgkkbkcdhcgfn"
EXTENSION_URL = f"https://chromewebstore.google.com/detail/claude/{EXTENSION_ID}"
NATIVE_HOST_NAME = "com.anthropic.claude_code_browser_extension.json"
MIN_CLAUDE_VERSION = (2, 1, 0)
CACHE_TTL_S = 10.0
#: Keys Claude Code writes to ~/.claude.json (read-only here); the first is what "/chrome →
#: Enabled by default" sets.
ENABLED_BY_DEFAULT_KEY = "claudeInChromeDefaultEnabled"
ONBOARDING_KEY = "hasCompletedClaudeInChromeOnboarding"
CACHED_INSTALLED_KEY = "cachedChromeExtensionInstalled"
#: Chromium browsers that can host the extension: profile root under ~/Library/Application Support.
BROWSERS: dict[str, tuple[str, str]] = {  # name -> (profile root, bundle id)
    "Google Chrome": ("Google/Chrome", "com.google.Chrome"),
    "Microsoft Edge": ("Microsoft Edge", "com.microsoft.edgemac"),
    "Brave Browser": ("BraveSoftware/Brave-Browser", "com.brave.Browser"),
    "Arc": ("Arc/User Data", "company.thebrowser.Browser"),
}

FIX_INSTALL_CLAUDE = "Install Claude Code: https://claude.com/claude-code"
FIX_UPDATE_CLAUDE = "Update Claude Code (needs 2.1.0 or newer): run `claude update`"
FIX_SIGN_IN = "Open Terminal, run `claude` and sign in."
FIX_API_KEY = "API-key login can't use Chrome — sign in with `claude` instead"
FIX_BROWSER = "Chrome isn't running"
FIX_EXTENSION = f"Install the Claude in Chrome extension: {EXTENSION_URL}"
FIX_ENABLE = ("Open Terminal, run `claude --chrome`, press Enter at the dialog, then type /chrome and choose "
              "Enabled by default")

Runner = Callable[[list[str]], tuple[int, str]]  # argv -> (returncode, stdout)


def run_cli(argv: list[str], timeout: float = 15.0) -> tuple[int, str]:
    try:
        res = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return res.returncode, res.stdout
    except (OSError, subprocess.TimeoutExpired):
        return -1, ""


def parse_version(text: str) -> tuple[int, ...] | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(x) for x in m.groups()) if m else None


# ----------------------------------------------------------------------------- claude


def check_claude(cfg: config.Config, *, run: Runner = run_cli, env: dict[str, str] | None = None) -> dict[str, Any]:
    from macd.brain.claude_brain import default_cli_path

    env = os.environ if env is None else env
    path = default_cli_path(cfg)
    out: dict[str, Any] = {"installed": False, "path": path, "version": None, "min_ok": False, "signed_in": "unknown",
                           "auth_method": "unknown", "fix": ""}
    if not path or not Path(path).exists():
        out["path"] = None
        out["fix"] = FIX_INSTALL_CLAUDE
        return out
    out["installed"] = True
    rc, text = run([path, "--version"])
    ver = parse_version(text) if rc == 0 else None
    out["version"] = ".".join(map(str, ver)) if ver else None
    out["min_ok"] = bool(ver and ver >= MIN_CLAUDE_VERSION)
    if cfg.brain.auth == "api_key" or env.get("ANTHROPIC_API_KEY"):
        out["auth_method"] = "api_key"
    rc, text = run([path, "auth", "status"])
    try:
        data = json.loads(text) if rc == 0 and text.strip() else {}
    except json.JSONDecodeError:
        data = {}
    if isinstance(data, dict) and "loggedIn" in data:
        out["signed_in"] = "yes" if data.get("loggedIn") else "no"
        if out["auth_method"] == "unknown":
            method = str(data.get("authMethod") or "").lower().replace(".", "_").replace("-", "_")
            out["auth_method"] = {"claude_ai": "claude_ai", "claudeai": "claude_ai", "api_key": "api_key",
                                  "apikey": "api_key", "console": "api_key"}.get(method, method or "unknown")
    elif rc != 0 and out["auth_method"] != "api_key":
        out["signed_in"] = "no"
    if not out["min_ok"]:
        out["fix"] = FIX_UPDATE_CLAUDE
    elif out["auth_method"] == "api_key":
        out["fix"] = FIX_API_KEY
    elif out["signed_in"] == "no":
        out["fix"] = FIX_SIGN_IN
    return out


# ----------------------------------------------------------------------------- chrome


def browsers_running() -> list[str]:
    """Names of running Chromium browsers (NSWorkspace when available, else pgrep)."""
    try:
        from AppKit import NSWorkspace

        ids = {app.bundleIdentifier() for app in NSWorkspace.sharedWorkspace().runningApplications()}
        return [name for name, (_root, bundle) in BROWSERS.items() if bundle in ids]
    except Exception:  # noqa: BLE001 - headless / no pyobjc: fall back to pgrep
        running = []
        for name in BROWSERS:
            rc, _ = run_cli(["pgrep", "-x", name], timeout=3)
            if rc == 0:
                running.append(name)
        return running


def read_claude_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def check_chrome(
    *,
    app_support: Path | None = None,
    claude_json: Path | None = None,
    running: Callable[[], list[str]] = browsers_running,
    auth_method: str = "unknown",
    connected: str = "unknown",
) -> dict[str, Any]:
    app_support = app_support or Path.home() / "Library/Application Support"
    claude_json = claude_json or Path.home() / ".claude.json"
    running_names = running()
    ext_dirs: list[str] = []
    hosts_ok: list[str] = []
    for name, (root, _bundle) in BROWSERS.items():
        base = app_support / root
        if not base.is_dir():
            continue
        if any((p / "Extensions" / EXTENSION_ID).is_dir() for p in base.iterdir() if p.is_dir()):
            ext_dirs.append(name)
        manifest = base / "NativeMessagingHosts" / NATIVE_HOST_NAME
        try:
            data = json.loads(manifest.read_text())
            origins = data.get("allowed_origins", []) if isinstance(data, dict) else []
            if any(EXTENSION_ID in str(o) for o in origins):
                hosts_ok.append(name)
        except (OSError, ValueError):
            pass
    cj = read_claude_json(claude_json)
    enabled = cj.get(ENABLED_BY_DEFAULT_KEY)
    out: dict[str, Any] = {
        "browser_running": bool(running_names),
        "browsers": running_names,
        "extension_installed": bool(ext_dirs),
        "native_host": bool(hosts_ok),
        "enabled_by_default": enabled if isinstance(enabled, bool) else None,
        "onboarding_done": cj.get(ONBOARDING_KEY) if isinstance(cj.get(ONBOARDING_KEY), bool) else None,
        "connected": connected,
        "fix": "",
    }
    if auth_method == "api_key" or connected == "unavailable_api_key":
        out["fix"] = FIX_API_KEY
    elif not out["browser_running"]:
        out["fix"] = FIX_BROWSER
    elif not out["extension_installed"]:
        out["fix"] = FIX_EXTENSION
    elif not out["native_host"] or out["enabled_by_default"] is not True or connected == "not_connected":
        out["fix"] = FIX_ENABLE
    return out


def is_ready(claude: dict[str, Any], chrome: dict[str, Any]) -> bool:
    return bool(
        claude["installed"] and claude["min_ok"] and claude["signed_in"] == "yes" and claude["auth_method"] != "api_key"
        and chrome["browser_running"] and chrome["extension_installed"] and chrome["native_host"]
        and chrome["enabled_by_default"] is True and chrome["connected"] not in ("not_connected", "unavailable_api_key")
    )


def check_all(cfg: config.Config, *, connected: str = "unknown", **kw: Any) -> dict[str, Any]:
    claude = check_claude(cfg, **{k: v for k, v in kw.items() if k in ("run", "env")})
    chrome = check_chrome(auth_method=claude["auth_method"], connected=connected,
                          **{k: v for k, v in kw.items() if k in ("app_support", "claude_json", "running")})
    return {"claude": claude, "chrome": chrome, "ready": is_ready(claude, chrome), "checked_at": time.time()}


# ----------------------------------------------------------------------------- daemon-side cache


class ConnectChecker:
    """Cached, non-blocking view for `status` + on-demand `connect_check` (deep = ask the brain)."""

    def __init__(self, cfg: config.Config, *, brain: Any | None = None,
                 brain_chrome: Callable[[], str] | None = None, ttl_s: float = CACHE_TTL_S) -> None:
        self.cfg = cfg
        self.brain = brain
        self.brain_chrome = brain_chrome  # () -> "connected"|"disconnected"|"unknown" (last known)
        self.ttl_s = ttl_s
        self._cache: dict[str, Any] | None = None
        self._cached_at = 0.0
        self._refresh: asyncio.Task[Any] | None = None

    def _connected_hint(self) -> str:
        last = self.brain_chrome() if self.brain_chrome is not None else "unknown"
        return {"connected": "connected", "disconnected": "not_connected"}.get(last, "unknown")

    def snapshot(self) -> dict[str, Any]:
        """Last result (never blocks); kicks off a background refresh when stale."""
        if (self._cache is None or time.monotonic() - self._cached_at > self.ttl_s) and (
            self._refresh is None or self._refresh.done()
        ):
            self._refresh = asyncio.get_running_loop().create_task(self.check(), name="connect-refresh")
        return self._cache or {"claude": None, "chrome": None, "ready": False, "pending": True}

    async def command(self, deep: bool = False) -> dict[str, Any]:
        """`connect_check`: deep → probe the brain (starts Claude); shallow → cached when fresh
        (the app polls this every 2 s while onboarding), else one refresh."""
        if not deep and self._cache is not None and time.monotonic() - self._cached_at <= self.ttl_s:
            return self._cache
        if self._refresh is not None and not self._refresh.done() and not deep:
            return await self._refresh
        return await self.check(deep=deep)

    async def check(self, deep: bool = False) -> dict[str, Any]:
        connected = self._connected_hint()
        probe_result: dict[str, Any] | None = None
        if deep and self.brain is not None:
            try:
                # ClaudeBrain.connect_check() → {"auth", "auth_detail", "chrome", "chrome_detail",
                # "cli_version", "session_ok"}; chrome ∈ connected|not_connected|unavailable_api_key|unknown
                raw = await asyncio.wait_for(self.brain.connect_check(), 90)
                probe_result = dict(raw) if isinstance(raw, dict) else {"chrome": str(raw)}
                connected = str(probe_result.get("chrome") or "unknown")
            except Exception:
                log.exception("brain.connect_check failed")
                connected = "unknown"
        result = await asyncio.to_thread(check_all, self.cfg, connected=connected)
        if probe_result is not None:
            result["chrome"]["probe"] = probe_result
            if probe_result.get("auth") == "not_signed_in":
                result["claude"]["signed_in"] = "no"
                result["claude"]["fix"] = result["claude"]["fix"] or FIX_SIGN_IN
                result["ready"] = False
        self._cache, self._cached_at = result, time.monotonic()
        return result
