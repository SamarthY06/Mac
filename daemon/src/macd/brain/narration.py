"""Short spoken status lines for tool calls (ARCHITECTURE §8); the orchestrator rate-limits them."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

_SILENT = {"mcp__mac__say", "mcp__mac__request_confirmation", "mcp__mac__remember",
           "TodoWrite", "mcp__claude-in-chrome__tabs_context_mcp"}
_TELL_APP = re.compile(r"""tell\s+application\s+(?:id\s+)?["']([^"']+)["']""", re.IGNORECASE)
_CHROME = "mcp__claude-in-chrome__"


def _host(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.hostname or "").removeprefix("www.")
    return host


def _clip(text: str, n: int = 60) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def _chrome(short: str, inp: dict[str, Any]) -> str | None:
    if short == "navigate":
        url = str(inp.get("url", ""))
        if url in ("back", "forward"):
            return "Going " + url
        host = _host(url)
        return f"Opening {host}" if host else "Opening a page"
    if short.startswith("tabs_create"):
        return "Opening a new tab"
    if short in ("get_page_text", "read_page"):
        return "Reading the page"
    if short == "find":
        q = _clip(str(inp.get("query", "")), 40)
        return f"Looking for {q}" if q else "Looking for something on the page"
    if short == "form_input":
        return "Filling in the form"
    if short == "computer":
        action = str(inp.get("action", ""))
        if action in ("screenshot", "zoom"):
            return "Looking at the screen"
        if action == "scroll" or action == "scroll_to":
            return "Scrolling"
        if action == "type":
            return "Typing"
        if action == "wait":
            return None
        return "Clicking around the page"
    if short == "javascript_tool":
        return "Working on the page"
    return "Working in the browser"


def describe_tool(name: str, tool_input: dict[str, Any] | None) -> str | None:
    """A short, speakable status for a tool call, or None if it should stay silent."""
    inp = tool_input or {}
    if name in _SILENT:
        return None
    if name.startswith(_CHROME):
        return _chrome(name.removeprefix(_CHROME), inp)
    if name == "Bash":
        cmd = str(inp.get("command", ""))
        m = _TELL_APP.search(cmd)
        if m:
            return f"Working in {m.group(1)}"
        if re.search(r"\bmdfind\b|\bfind\b", cmd):
            return "Searching your files"
        return "Running a command"
    if name == "WebSearch":
        q = _clip(str(inp.get("query", "")), 50)
        return f"Searching the web for {q}" if q else "Searching the web"
    if name == "WebFetch":
        host = _host(str(inp.get("url", "")))
        return f"Reading {host}" if host else "Reading a web page"
    if name in ("Read", "Grep", "Glob", "LS", "NotebookRead"):
        return "Looking through files"
    if name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        return "Writing a file"
    if name in ("Task", "Agent"):
        return "Working on it"
    if name.startswith("mcp__mac__"):
        short = name.removeprefix("mcp__mac__")
        if short.startswith("ui_"):
            app = str(inp.get("app", "") or inp.get("_app", ""))
            return f"Working in {app}" if app else "Working on the screen"
        if short == "screenshot":
            return "Looking at the screen"
        if short == "open_app":
            app = str(inp.get("name", ""))
            return f"Opening {app}" if app else "Opening an app"
        if short == "notify":
            return None
        return None
    return None
