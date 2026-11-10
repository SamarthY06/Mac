"""Spike P0.3: does HookMatcher(on_failure="block", timeout=N) deny a tool call whose hook
times out (or raises)? Harmless tool: Bash `echo` writing a marker file in a temp dir.

Run: cd daemon && uv run python spikes/p03_hook_timeout.py
Cases:
  control   hook returns {} at once                    -> expect the command RUNS
  timeout   hook sleeps 3x the timeout, on_failure=block -> expect DENIED (marker absent)
  raises    hook raises, on_failure=block              -> expect DENIED
  timeout_continue  same as timeout, on_failure unset  -> shows the default
  raises_continue   same as raises, on_failure unset   -> shows the default
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query
from claude_agent_sdk.types import AssistantMessage, ResultMessage, UserMessage

from macd import config
from macd.brain.claude_brain import default_cli_path

TIMEOUT_S = 2


async def run_case(name: str, behaviour: str, on_failure: str | None) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"p03-{name}-"))
    marker = tmp / "marker.txt"
    hook_calls: list[float] = []

    async def hook(inp, tool_use_id, ctx):
        hook_calls.append(time.monotonic())
        if behaviour == "sleep":
            await asyncio.sleep(TIMEOUT_S * 3)
            return {}
        if behaviour == "raise":
            raise RuntimeError("spike: hook failure")
        return {}

    opts = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        allowed_tools=["Bash"],
        hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[hook], timeout=TIMEOUT_S,
                                          **({"on_failure": on_failure} if on_failure else {}))]},
        cwd=str(tmp), setting_sources=[], max_turns=3, model="haiku",
        cli_path=default_cli_path(config.load()),
    )
    prompt = (f"Run exactly this one Bash command and nothing else: echo spike > {marker}\n"
              "Then reply with one word: OK if it ran, BLOCKED if the tool call was refused.")
    t0 = time.monotonic()
    tool_results: list[str] = []
    final = ""
    err = ""
    try:
        async for m in query(prompt=prompt, options=opts):
            if isinstance(m, UserMessage) and isinstance(m.content, list):
                for b in m.content:
                    c = getattr(b, "content", None)
                    if c is not None:
                        tool_results.append(str(c)[:200])
            elif isinstance(m, AssistantMessage):
                for b in m.content:
                    if getattr(b, "text", None):
                        final = b.text
            elif isinstance(m, ResultMessage):
                pass
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:300]
    return {"case": name, "on_failure": on_failure, "hook": behaviour,
            "ran": marker.exists(), "hook_calls": len(hook_calls),
            "elapsed_s": round(time.monotonic() - t0, 1), "tool_results": tool_results,
            "final": final[:120], "error": err}


async def main() -> None:
    cases = [("control", "ok", "block"), ("timeout", "sleep", "block"),
             ("raises", "raise", "block"), ("timeout_continue", "sleep", None), ("raises_continue", "raise", None)]
    only = sys.argv[1:]
    for name, beh, onf in cases:
        if only and name not in only:
            continue
        print(json.dumps(await run_case(name, beh, onf)), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
