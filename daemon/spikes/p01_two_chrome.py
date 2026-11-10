"""Spike P0.1: can two `claude --chrome` processes from the same parent share one Claude in
Chrome extension?

Run: cd daemon && uv run python spikes/p01_two_chrome.py [solo|pair|both]
Each worker: new tab -> https://example.com -> read the title -> close that tab. Read only, no
logged-in sites. Records the MCP server status at init, every chrome tool call and its result,
and wall time. "pair" starts both workers at the same moment from this one parent process.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

from claude_agent_sdk import ClaudeAgentOptions, query
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    UserMessage,
)

from macd import config
from macd.brain.claude_brain import default_cli_path

PROMPT = ("Use only the Claude in Chrome browser tools. Steps: 1) create a NEW tab, 2) navigate that new tab "
          "to https://example.com, 3) read that page's title, 4) close the tab you created and any other tab in your own tab group (e.g. an empty New Tab it made). Never touch, "
          "read or close any other tab. Reply with exactly: TITLE=<the title> TAB=<tab id>. If the browser "
          "tools are missing or fail, reply BROWSER_UNAVAILABLE: <the error>.")


async def worker(name: str, cli: str | None, t_start: float) -> dict:
    opts = ClaudeAgentOptions(
        permission_mode="bypassPermissions", tools=[], setting_sources=[], max_turns=12, model="sonnet",
        extra_args={"chrome": None}, cli_path=cli)
    rec: dict = {"worker": name, "chrome_status": None, "calls": [], "final": "", "error": ""}
    t0 = time.monotonic()
    try:
        async for m in query(prompt=PROMPT, options=opts):
            if isinstance(m, SystemMessage) and m.subtype == "init":
                for s in m.data.get("mcp_servers", []):
                    if "chrome" in s.get("name", ""):
                        rec["chrome_status"] = s.get("status")
                rec["init_s"] = round(time.monotonic() - t_start, 1)
            elif isinstance(m, AssistantMessage):
                for b in m.content:
                    if getattr(b, "name", None):
                        rec["calls"].append({"t": round(time.monotonic() - t_start, 1),
                                             "tool": b.name.replace("mcp__claude-in-chrome__", ""),
                                             "input": json.dumps(b.input)[:120]})
                    elif getattr(b, "text", None):
                        rec["final"] = b.text.strip()[:200]
            elif isinstance(m, UserMessage) and isinstance(m.content, list):
                for b in m.content:
                    if getattr(b, "tool_use_id", None):
                        c = b.content
                        txt = json.dumps(c)[:160] if not isinstance(c, str) else c[:160]
                        rec["calls"].append({"t": round(time.monotonic() - t_start, 1),
                                             "result_error": bool(getattr(b, "is_error", False)),
                                             "result": txt})
            elif isinstance(m, ResultMessage):
                rec["cost_usd"] = m.total_cost_usd
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    rec["wall_s"] = round(time.monotonic() - t0, 1)
    return rec


async def main() -> None:
    cli = default_cli_path(config.load())
    mode = (sys.argv[1:] or ["both"])[0]
    if mode in ("solo", "both"):
        t = time.monotonic()
        print(json.dumps({"mode": "solo", **await worker("solo", cli, t)}, indent=1), flush=True)
    if mode in ("pair", "both"):
        t = time.monotonic()
        a, b = await asyncio.gather(worker("A", cli, t), worker("B", cli, t))
        print(json.dumps({"mode": "pair", "A": a, "B": b}, indent=1), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
