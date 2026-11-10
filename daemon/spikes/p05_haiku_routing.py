"""Spike P0.5: latency of one Haiku routing call (folder + continue/new + confidence).

Run: cd daemon && uv run python spikes/p05_haiku_routing.py
Mode "cold": one `query()` per call (spawns the CLI each time), as W3.5 step 2 is written.
Mode "warm": one long-lived ClaudeSDKClient, one query per call (measures what a warm router
process would buy). Candidates are fictional but shaped like the real list (W3.5).
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, query
from claude_agent_sdk.types import AssistantMessage, ResultMessage

from macd import config
from macd.brain.claude_brain import default_cli_path

CANDIDATES = [
    ("Resume", "Job search, resume and cover letters. Sessions: 2026-11-09 'AI job hunt' (id s-r1), 2026-11-02 'Resume rewrite' (id s-r2)"),
    ("Mac", "The Mac voice agent source code (Python daemon, Swift app). Sessions: 2026-11-10 'WhatsApp design' (id s-m1)"),
    ("Taxes2026", "Tax documents and receipts for the 2026 return. Sessions: none"),
    ("Thesis", "Master's thesis LaTeX sources and experiments. Sessions: 2026-10-20 'Chapter 3 edits' (id s-t1)"),
    ("Recipes", "Personal recipe collection in markdown. Sessions: none"),
    ("Photos-Export", "Exported holiday photos to sort. Sessions: none"),
    ("Kaggle", "Kaggle competition notebooks. Sessions: 2026-10-30 'House prices model' (id s-k1)"),
    ("Blog", "Personal blog (Hugo site). Sessions: 2026-11-05 'Post about agents' (id s-b1)"),
    ("Finance", "Budget spreadsheets and bank exports. Sessions: none"),
    ("Interview-Prep", "Leetcode and system design notes. Sessions: 2026-11-08 'Graph problems' (id s-i1)"),
    ("Travel", "Trip plans and bookings. Sessions: 2026-09-12 'Japan itinerary' (id s-tr1)"),
    ("dotfiles", "Shell and editor configuration. Sessions: none"),
]

SYSTEM = ("You route a user's request to a folder on their Mac. Answer with ONE line of JSON and nothing "
          'else: {"folder": "<name from the list or general>", "session": "continue:<id>" or "new", '
          '"confidence": <0..1>}. Continue a session only for a follow-up or repeat of that work; '
          "a totally new task in the same folder is new. Use general when no folder fits.")

REQUESTS = [
    "find me 20 more AI engineer jobs like last time",
    "add a cover letter for the DeepMind role",
    "how much did I spend on groceries in October?",
    "fix the typo in my latest blog post",
    "write a new blog post about WhatsApp bots",
    "what's the weather tomorrow in London?",
    "continue the japan trip plan, add 2 days in Kyoto",
    "make the house prices model use gradient boosting",
    "practice a dynamic programming problem with me",
    "sort the holiday photos by date",
]


def prompt_for(req: str) -> str:
    lines = "\n".join(f"- {n}: {d}" for n, d in CANDIDATES)
    return f"Folders:\n{lines}\n\nToday: 2026-11-10.\nRequest: {req}"


def opts(cli: str | None) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(model="haiku", tools=[], max_turns=1, setting_sources=[],
                              system_prompt=SYSTEM, cli_path=cli)


def pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return s[k]


async def cold(cli: str | None) -> list[tuple[float, str]]:
    out = []
    for req in REQUESTS:
        t0 = time.monotonic()
        ans = ""
        async for m in query(prompt=prompt_for(req), options=opts(cli)):
            if isinstance(m, AssistantMessage):
                ans += "".join(getattr(b, "text", "") for b in m.content)
        out.append((time.monotonic() - t0, ans.strip()))
    return out


async def warm(cli: str | None) -> list[tuple[float, str]]:
    out = []
    async with ClaudeSDKClient(opts(cli)) as c:
        for req in REQUESTS:
            t0 = time.monotonic()
            await c.query(prompt_for(req))
            ans = ""
            async for m in c.receive_response():
                if isinstance(m, AssistantMessage):
                    ans += "".join(getattr(b, "text", "") for b in m.content)
                if isinstance(m, ResultMessage):
                    break
            out.append((time.monotonic() - t0, ans.strip()))
    return out


async def main() -> None:
    cli = default_cli_path(config.load())
    for name, fn in (("cold", cold), ("warm", warm)):
        res = await fn(cli)
        ts = [t * 1000 for t, _ in res]
        for (t, a), r in zip(res, REQUESTS):
            print(f"{name} {t*1000:7.0f} ms  {r!r} -> {a}")
        print(json.dumps({"mode": name, "n": len(ts), "p50_ms": round(statistics.median(ts)),
                          "p95_ms": round(pct(ts, 95)), "min_ms": round(min(ts)), "max_ms": round(max(ts))}),
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
