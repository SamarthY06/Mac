from __future__ import annotations

import pytest

from macd.brain.narration import describe_tool

C = "mcp__claude-in-chrome__"


@pytest.mark.parametrize(
    ("name", "inp", "expected"),
    [
        (f"{C}navigate", {"url": "https://www.linkedin.com/search?q=x"}, "Opening linkedin.com"),
        (f"{C}navigate", {"url": "github.com/foo"}, "Opening github.com"),
        (f"{C}tabs_create_mcp", {}, "Opening a new tab"),
        (f"{C}get_page_text", {"tabId": 1}, "Reading the page"),
        (f"{C}read_page", {"tabId": 1}, "Reading the page"),
        (f"{C}find", {"query": "search bar"}, "Looking for search bar"),
        (f"{C}computer", {"action": "screenshot"}, "Looking at the screen"),
        ("Bash", {"command": "osascript -e 'tell application \"Mail\" to get subject of messages'"},
         "Working in Mail"),
        ("Bash", {"command": "ls -la"}, "Running a command"),
        ("WebSearch", {"query": "weather in Paris"}, "Searching the web for weather in Paris"),
        ("WebFetch", {"url": "https://docs.python.org/3/"}, "Reading docs.python.org"),
        ("Read", {"file_path": "/x"}, "Looking through files"),
        ("Grep", {"pattern": "x"}, "Looking through files"),
        ("Glob", {"pattern": "*"}, "Looking through files"),
        ("mcp__mac__ui_tree", {"app": "WhatsApp"}, "Working in WhatsApp"),
        ("mcp__mac__ui_click", {"element_id": "3"}, "Working on the screen"),
        ("mcp__mac__screenshot", {}, "Looking at the screen"),
        ("mcp__mac__say", {"text": "hi"}, None),
        ("mcp__mac__request_confirmation", {"summary": "x"}, None),
    ],
)
def test_describe_tool(name: str, inp: dict, expected: str | None) -> None:
    assert describe_tool(name, inp) == expected


def test_long_query_is_clipped() -> None:
    out = describe_tool("WebSearch", {"query": "word " * 40})
    assert out is not None and len(out) < 90
