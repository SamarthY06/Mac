"""W6 gate 2 (V7): RemoteBrain's added latency and agentd's loop lag, under a scripted chat load.

agentd runs in its own process: the real Service and Server, with FakeBrain-backed conversations
(no `claude`). While a chat client process runs back-to-back WhatsApp turns through agentd's real
chat path (routing with a 0.3 s model wait, a settings hash of an approved folder's 2 MB `.claude/`
tree, conversation-record writes with fsync, one event every 50 ms of which 10 are `attach` events
with path checks), this test drives voice turns through `RemoteBrain` and measures ask -> first
TEXT, SAY and TOOL_START. The baseline is the same script on an in-process FakeBrain.

Gate: added p95 under 20 ms for each, and agentd's loop lag p99 under 10 ms during the run.
Run it on its own: `uv run pytest -q -s -m perf`. `MAC_PERF_TURNS` (default 300) sets the number of voice turns; the numbers are printed (`-s`).

This file is also the agentd/chat-client entry point: `python <this file> serve <root>` and
`python <this file> chat <socket>`.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "agentd"))

from agentd import schema as s
from macd.brain import fake_brain as fb
from macd.contracts import BrainEventKind as K

V = s.VOICE_CONV
CHAT = "whatsapp:123@g.us"
TURNS = int(os.environ.get("MAC_PERF_TURNS", "300"))
BUDGET_MS = 20.0
LAG_BUDGET_MS = 10.0
KINDS = (K.TEXT, K.SAY, K.TOOL_START)

pytestmark = pytest.mark.perf  # a benchmark: run alone, serially (`-m perf`), not in the fast suite


def voice_script(_text: str) -> list[fb.Step]:
    return [fb.text("Sure, checking."), *fb.tool("mcp__mac__calendar", duration=0.01),
            fb.say("Looking at your calendar"), fb.text("You have two meetings today.")]


def chat_script(attach_dir: Path) -> fb.Script:
    def script(_text: str) -> list[fb.Step]:
        steps: list[fb.Step] = [fb.say("Finding jobs in Proj")]
        for i in range(40):  # 2 s at one event per 50 ms, 10 of them attach events
            if i % 4 == 0:
                p = attach_dir / f"result-{i}.csv"
                steps.append(fb.Step(K.ATTACH, text=str(p), delay=0.05,
                                     data={"attach_id": f"a{i}-{random.random()}", "path": str(p), "caption": ""}))
            else:
                steps.append(fb.say(f"Searching, {i} so far", delay=0.05))
        steps.append(fb.text("Found 20 roles."))
        return steps

    return script


# ----------------------------------------------------------------------------- agentd process


class SlowModel:
    """The router's model: a 0.3 s wait (warm Haiku), then the approved folder."""

    async def complete(self, prompt: str) -> str:
        await asyncio.sleep(0.3)
        return '{"folder": "Proj", "session": "new", "confidence": 0.9}'


def make_tree(root: Path) -> None:
    home = root / "home"
    proj = home / "Proj"
    skills = proj / ".claude" / "skills"
    skills.mkdir(parents=True)
    (proj / ".claude" / "settings.json").write_text('{"permissions": {"allow": []}}')
    for i in range(200):  # 2 MB
        (skills / f"s{i:03}.md").write_bytes(os.urandom(5120).hex().encode())
    attach = home / "out"
    attach.mkdir()
    for i in range(0, 40, 4):
        (attach / f"result-{i}.csv").write_text("role,company\n")


async def serve(root: Path) -> None:
    import agentd_kit as kit

    from agentd.router import Router
    from agentd.server import Server
    from agentd.service import AgentPaths, Service
    from agentd.stream import LagMonitor
    from agentd.workspaces import WorkspaceStore

    home, attach = root / "home", root / "home" / "out"
    paths = AgentPaths(home=home, mac_home=root / "mac", log_dir=root / "logs")
    kit.write_chat_config(paths.mac_home, [CHAT], [])
    ws = WorkspaceStore(paths.state_dir / "workspaces.json", home, seeds=())
    await ws.load()
    await ws.approve(home / "Proj")
    router = Router(ws, SlowModel(), list_sessions=lambda **k: [], mdfind=lambda args: "")

    def factory(spec: Any) -> Any:
        return fb.FakeBrain(voice_script if spec.conv == V else chat_script(attach))

    svc = Service(paths, factory, router=router, workspaces=ws, lag_interval_s=0.01)
    server = Server(svc, kit.TOKENS, root / "a.sock")
    await svc.start()
    await server.start()
    lag = LagMonitor(0.01, window=1_000_000)  # the whole run, not just the status window
    lag_task = asyncio.create_task(lag.run())
    print("READY", flush=True)
    await asyncio.to_thread(sys.stdin.read)  # until the test closes our stdin
    lag_task.cancel()
    xs = sorted(lag.samples)
    p99 = xs[min(len(xs) - 1, int(0.99 * (len(xs) - 1) + 0.5))] if xs else 0.0
    status = svc.conversation(V).status()
    print(json.dumps({"lag_p99_ms": round(p99, 2), "lag_max_ms": round(lag.max_ms, 2), "lag_samples": len(xs),
                      "status_lag_p99_ms": status.loop_lag_ms_p99}), flush=True)
    await server.close()
    await svc.close()


async def chat(sock: Path) -> None:
    import agentd_kit as kit

    stop = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.read(), stop.set()), daemon=True).start()
    c = await kit.Client.open(sock)
    await c.hello(kit.CHAT_TOKEN, [CHAT], "chat-perf")
    await c.call({"op": "subscribe", "conv": CHAT})
    turns = attaches = events = 0
    folders: set[str] = set()
    while not stop.is_set():
        await c.call({"op": "new_session", "conv": CHAT})  # route (and hash) every turn
        r = await c.call({"op": "ask", "conv": CHAT, "turn": turns + 1, "text": "find me AI engineer jobs"})
        assert r["ok"], r
        turns += 1
        if turns == 1:
            print("STARTED", flush=True)
        while True:
            e = await c.event(timeout=30)
            events += 1
            if e["ev"] == "attach":
                attaches += 1
            elif e["ev"] == "status" and "folder" in e.get("data", {}):
                folders.add(e["data"]["folder"])
            elif e["ev"] in ("done", "error"):
                break
        await c.call({"op": "ack", "conv": CHAT, "turn_uid": e["turn_uid"]})
    print(json.dumps({"chat_turns": turns, "chat_events": events, "attaches": attaches,
                      "folders": sorted(folders)}), flush=True)
    await c.close()


# ----------------------------------------------------------------------------- measurement


async def run_turns(brain: Any, n: int, pace: tuple[float, float]) -> dict[K, list[float]]:
    """ask -> first event of each kind, in ms, over `n` sequential turns."""
    firsts: dict[int, dict[K, float]] = {}
    done: dict[int, asyncio.Event] = {}

    async def consume() -> None:
        async for ev in brain.events():
            now = time.perf_counter()
            seen = firsts.setdefault(ev.turn_id, {})
            seen.setdefault(ev.kind, now)
            if ev.kind in (K.DONE, K.ERROR) and ev.turn_id in done:
                done[ev.turn_id].set()

    task = asyncio.create_task(consume())
    out: dict[K, list[float]] = {k: [] for k in KINDS}
    try:
        for turn in range(1, n + 1):
            done[turn] = asyncio.Event()
            t0 = time.perf_counter()
            await brain.ask(turn, "what's on my calendar")
            await asyncio.wait_for(done[turn].wait(), 5)
            seen = firsts[turn]
            assert K.ERROR not in seen, "voice turn failed"
            for k in KINDS:
                out[k].append((seen[k] - t0) * 1000)
            if pace[1]:
                await asyncio.sleep(random.uniform(*pace))
    finally:
        task.cancel()
    return out


def p95(xs: list[float]) -> float:
    return statistics.quantiles(xs, n=100)[94]


async def test_remote_brain_latency_under_chat_load() -> None:
    from macd.brain.remote import RemoteBrain
    from macd.config import Config

    root = Path(tempfile.mkdtemp(prefix="rp", dir="/tmp"))
    procs: list[subprocess.Popen[str]] = []
    try:
        make_tree(root)
        env = {**os.environ, "MAC_HOME": str(root / "mac"), "MAC_LOG_DIR": str(root / "logs")}

        def spawn(*args: str) -> subprocess.Popen[str]:
            p = subprocess.Popen([sys.executable, str(HERE), *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 text=True, env=env)
            procs.append(p)
            return p

        agentd = spawn("serve", str(root))
        assert agentd.stdout is not None and agentd.stdout.readline().strip() == "READY"

        # baseline: the same script on an in-process FakeBrain
        local = fb.FakeBrain(voice_script)
        await local.start()
        base = await run_turns(local, TURNS, (0.0, 0.0))
        await local.close()

        chat_proc = spawn("chat", str(root / "a.sock"))
        assert chat_proc.stdout is not None and chat_proc.stdout.readline().strip() == "STARTED"

        brain = RemoteBrain(Config(), token=lambda: "v" * 64, sock_path=root / "a.sock")
        await brain.start()
        await asyncio.wait_for(brain.connected.wait(), 5)
        try:
            remote = await run_turns(brain, TURNS, (0.020, 0.048))
        finally:
            await brain.close()

        chat_out, _ = chat_proc.communicate("", timeout=30)
        chat_stats = json.loads(chat_out.strip().splitlines()[-1])
        agentd_out, _ = agentd.communicate("", timeout=30)
        lag = json.loads(agentd_out.strip().splitlines()[-1])
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        shutil.rmtree(root, ignore_errors=True)

    report = {k.value: {"local_p50": round(statistics.median(base[k]), 3), "local_p95": round(p95(base[k]), 3),
                        "remote_p50": round(statistics.median(remote[k]), 3), "remote_p95": round(p95(remote[k]), 3),
                        "added_p95": round(p95(remote[k]) - p95(base[k]), 3)} for k in KINDS}
    print("\nRemoteBrain latency (ms), turns =", TURNS)
    print(json.dumps({"latency": report, "agentd": lag, "chat": chat_stats}, indent=1))

    assert chat_stats["chat_turns"] >= 2 and chat_stats["attaches"] >= 10, chat_stats  # the load really ran
    assert chat_stats["folders"] == ["~/Proj"], chat_stats  # routed into the approved folder, hash included
    for k in KINDS:
        assert report[k.value]["added_p95"] < BUDGET_MS, report
    assert lag["lag_p99_ms"] < LAG_BUDGET_MS, lag


if __name__ == "__main__":
    mode, arg = sys.argv[1], Path(sys.argv[2])
    asyncio.run(serve(arg) if mode == "serve" else chat(arg))
