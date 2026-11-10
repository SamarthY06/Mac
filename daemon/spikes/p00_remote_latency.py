"""Spike P0.0: added latency of a RemoteBrain-style relay over agent.sock, and agentd loop lag.

Throwaway code (not the phase 1 server). A minimal "agentd" runs FakeBrain conversations and
relays their events as NDJSON over a unix socket, one write per event, encoded with
agentd.schema. A voice client in the bench process measures:

- e2e: ask sent -> first TEXT received (and the same for an in-process FakeBrain, as baseline);
- hop: brain emitted the event (stamp in agentd) -> client decoded it, for TEXT, SAY, TOOL_START;
- agentd event-loop lag: a 10 ms sleeper's overshoot, sampled the whole run.

Scenarios:
  idle            voice only
  chat            plus a scripted chat client in its own process (W6 gate 2): per chat turn a
                  routing wait, a settings hash of a 2 MB .claude/ tree, a store write with fsync,
                  10 attach events with path checks, and one event every 50 ms; blocking work in
                  asyncio.to_thread (V7)
  chat-blocking   the same, but the hash and fsync run ON the loop (what V7 forbids), for contrast

Run: cd daemon && uv run python spikes/p00_remote_latency.py bench [--turns 300]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from agentd import schema as s
from macd.brain import fake_brain as fb
from macd.contracts import BrainEventKind

VOICE = "voice:local"
CHAT = "whatsapp:spike@g.us"


def voice_script(_text: str) -> list[fb.Step]:
    return [fb.text("Sure, checking."), *fb.tool("mcp__mac__calendar", duration=0.01),
            fb.say("Looking at your calendar"), fb.text("You have two meetings today.")]


def chat_script(_text: str) -> list[fb.Step]:
    steps: list[fb.Step] = [fb.say("Finding jobs in Resume")]
    for i in range(40):  # 2 s at one event per 50 ms
        if i % 4 == 0:
            steps.append(fb.Step(BrainEventKind.ATTACH, text=f"/tmp/spike-attach-{i}.csv", delay=0.05,
                                 data={"attach_id": f"a{i}", "caption": ""}))
        else:
            steps.append(fb.say(f"Searching, {i} so far", delay=0.05))
    steps.append(fb.text("Found 20 roles."))
    return steps


# ----------------------------------------------------------------------------- server


class Server:
    def __init__(self, blocking: bool, tree: Path) -> None:
        self.blocking = blocking
        self.tree = tree
        self.boot = os.urandom(8).hex()
        self.brains = {VOICE: fb.FakeBrain(voice_script), CHAT: fb.FakeBrain(chat_script)}
        self.seq = {c: 0 for c in self.brains}
        self.subs: dict[str, list[asyncio.StreamWriter]] = {c: [] for c in self.brains}
        self.turn_of: dict[tuple[str, int], int] = {}
        self.lags: list[float] = []
        self.next_internal = 0
        self.hashes = 0

    async def start(self, path: str) -> None:
        for b in self.brains.values():
            await b.start()
        for conv in self.brains:
            asyncio.create_task(self.relay(conv))
        asyncio.create_task(self.lag_sampler())
        await asyncio.start_unix_server(self.handle, path=path)

    async def lag_sampler(self) -> None:
        while True:
            t = time.perf_counter()
            await asyncio.sleep(0.01)
            self.lags.append((time.perf_counter() - t - 0.01) * 1000)

    async def relay(self, conv: str) -> None:
        async for bev in self.brains[conv].events():
            t_emit = time.monotonic_ns()
            self.seq[conv] += 1
            turn, uid = bev.turn_id, s.turn_uid(self.boot, conv, "e", bev.turn_id)
            base = {"conv": conv, "boot": self.boot, "seq": self.seq[conv], "turn": turn, "turn_uid": uid}
            k = bev.kind
            if k is BrainEventKind.TEXT:
                ev = s.TextEvent(**base, text=bev.text)
            elif k is BrainEventKind.SAY:
                ev = s.SayEvent(**base, text=bev.text)
            elif k is BrainEventKind.TOOL_START:
                ev = s.ToolStartEvent(**base, tool=bev.tool_name, tool_use_id=bev.tool_use_id or "t")
            elif k is BrainEventKind.TOOL_END:
                ev = s.ToolEndEvent(**base, tool=bev.tool_name, tool_use_id=bev.tool_use_id or "t")
            elif k is BrainEventKind.ATTACH:
                await self.off_loop(os.path.exists, bev.text)  # attach path check (V7)
                ev = s.AttachEvent(**base, attach_id=bev.data["attach_id"], path=bev.text)
            elif k is BrainEventKind.DONE:
                ev = s.DoneEvent(**base, data=s.DoneData(duration_s=0.0, interrupted=bool(bev.data.get("interrupted"))))
                if conv == CHAT:
                    await self.off_loop(self.write_record)  # record write with fsync at turn end
            else:
                continue
            obj = s.encode_event(ev)
            obj["t_emit_ns"] = t_emit  # unknown field: ignored by schema decoders
            line = s.encode_line(obj)
            for w in self.subs[conv]:
                w.write(line)  # one write per event, no batching

    async def off_loop(self, fn, *a):
        return fn(*a) if self.blocking else await asyncio.to_thread(fn, *a)

    def settings_hash(self) -> str:
        self.hashes += 1
        h = hashlib.sha256()
        for p in sorted(self.tree.rglob("*")):
            if p.is_file():
                h.update(str(p.relative_to(self.tree)).encode())
                h.update(p.read_bytes())
        return h.hexdigest()

    def write_record(self) -> None:
        d = self.tree.parent
        fd, tmp = tempfile.mkstemp(dir=d)
        with os.fdopen(fd, "w") as f:
            json.dump({"conv": CHAT, "seq": self.seq[CHAT], "pad": "x" * 4000}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, d / "conversations.json")

    async def chat_turn_prelude(self) -> None:
        await asyncio.sleep(0.3)  # routing: waiting on a Haiku call is I/O, not loop work
        await self.off_loop(self.settings_hash)  # hash re-check at every turn start (W3.5)
        await self.off_loop(self.write_record)

    async def handle(self, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            while line := await r.readline():
                obj = s.parse_line(line)
                rid, op = obj.get("id"), obj.get("op")
                if op == "spike_stats":
                    lags = sorted(self.lags)
                    w.write(s.encode_line({"id": rid, "ok": True, "data": {
                        "n": len(lags), "p50": statistics.median(lags), "p99": lags[int(0.99 * (len(lags) - 1))],
                        "max": lags[-1], "chat_events": self.seq[CHAT], "chat_hashes": self.hashes}}))
                    self.lags.clear()
                    continue
                rid, req = s.decode_request(obj)
                if isinstance(req, s.Hello):
                    data = s.HelloData(api=s.API_VERSION, server="spike", kind=s.ClientKind.TEST,
                                       boot=self.boot, conversations=req.conversations)
                elif isinstance(req, s.Subscribe):
                    self.subs[req.conv].append(w)
                    data = s.SubscribeData(boot=self.boot, seq=self.seq[req.conv])
                elif isinstance(req, s.Ask):
                    self.next_internal += 1
                    internal = self.next_internal
                    if req.conv == CHAT:
                        asyncio.create_task(self._chat_ask(internal, req.text))
                    else:
                        await self.brains[req.conv].ask(internal, req.text)
                    data = s.AskData(turn_uid=s.turn_uid(self.boot, req.conv, "e", internal))
                else:
                    data = s.Empty()
                w.write(s.encode_line(s.encode_ok(rid, data)))
        except (ConnectionError, s.SchemaError):
            pass

    async def _chat_ask(self, internal: int, text: str) -> None:
        await self.chat_turn_prelude()
        await self.brains[CHAT].ask(internal, text)


def make_tree(root: Path) -> Path:
    tree = root / "Resume" / ".claude"
    (tree / "skills").mkdir(parents=True)
    for i in range(200):  # 200 x 10 KiB = 2 MB
        (tree / "skills" / f"f{i:03}.md").write_bytes(os.urandom(5120).hex().encode())
    return tree


async def serve(sock: str, blocking: bool) -> None:
    root = Path(tempfile.mkdtemp(prefix="p00-"))
    srv = Server(blocking, make_tree(root))
    await srv.start(sock)
    print("READY", flush=True)
    await asyncio.Event().wait()


# ----------------------------------------------------------------------------- clients


class Conn:
    def __init__(self, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        self.r, self.w, self.id = r, w, 0
        self.events: asyncio.Queue[tuple[int, dict, s.Event]] = asyncio.Queue()
        self.replies: dict[int, asyncio.Future] = {}
        asyncio.create_task(self._read())

    @classmethod
    async def open(cls, sock: str, convs: list[str]) -> Conn:
        c = cls(*await asyncio.open_unix_connection(sock))
        await c.call(s.Hello(api="1.0", token="t", client="spike", client_epoch="e", conversations=convs))
        for conv in convs:
            await c.call(s.Subscribe(conv=conv))
        return c

    async def _read(self) -> None:
        while line := await self.r.readline():
            obj = json.loads(line)
            t = time.monotonic_ns()
            if "ev" in obj:
                ev = s.decode_event(obj)  # what RemoteBrain will do
                self.events.put_nowait((t, obj, ev))
            elif (f := self.replies.pop(obj.get("id"), None)) is not None:
                f.set_result(obj)

    async def call(self, op: s.Op | None = None, raw: dict | None = None) -> dict:
        self.id += 1
        f = asyncio.get_running_loop().create_future()
        self.replies[self.id] = f
        self.w.write(s.encode_line(s.encode_request(self.id, op) if op else {"id": self.id, **raw}))
        return await f


async def voice_turns(sock: str, n: int) -> dict[str, list[float]]:
    c = await Conn.open(sock, [VOICE])
    out: dict[str, list[float]] = {"e2e": [], "hop_text": [], "hop_say": [], "hop_tool_start": []}
    for i in range(1, n + 1):
        t0 = time.monotonic_ns()
        await c.call(s.Ask(conv=VOICE, turn=i, text="what's on my calendar"))
        first = True
        while True:
            t, obj, ev = await c.events.get()
            hop = (t - obj["t_emit_ns"]) / 1e6
            if isinstance(ev, s.TextEvent):
                out["hop_text"].append(hop)
                if first:
                    out["e2e"].append((t - t0) / 1e6)
                    first = False
            elif isinstance(ev, s.SayEvent):
                out["hop_say"].append(hop)
            elif isinstance(ev, s.ToolStartEvent):
                out["hop_tool_start"].append(hop)
            elif isinstance(ev, s.DoneEvent):
                break
        await asyncio.sleep(0.02 + (i % 5) * 0.007)  # desynchronise from the chat event clock
    stats = await c.call(raw={"op": "spike_stats"})
    d = stats["data"]
    out["lag"] = [d["p50"], d["p99"], d["max"], d["n"], d["chat_events"], d["chat_hashes"]]
    c.w.close()
    return out


async def local_turns(n: int) -> list[float]:
    b = fb.FakeBrain(voice_script)
    await b.start()
    it = b.events().__aiter__()
    res = []
    for i in range(1, n + 1):
        t0 = time.monotonic_ns()
        await b.ask(i, "what's on my calendar")
        first = True
        while True:
            ev = await it.__anext__()
            if ev.kind is BrainEventKind.TEXT and first:
                res.append((time.monotonic_ns() - t0) / 1e6)
                first = False
            if ev.kind is BrainEventKind.DONE:
                break
        await asyncio.sleep(0.02 + (i % 5) * 0.007)
    await b.close()
    return res


async def chat_load(sock: str) -> None:
    c = await Conn.open(sock, [CHAT])
    print("READY", flush=True)
    turn = 0
    while True:
        turn += 1
        await c.call(s.Ask(conv=CHAT, turn=turn, text="find me 20 jobs"))
        while True:
            _, _, ev = await c.events.get()
            if isinstance(ev, s.DoneEvent):
                break


# ----------------------------------------------------------------------------- bench


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def spawn(*args: str) -> subprocess.Popen:
    p = subprocess.Popen([sys.executable, __file__, *args], stdout=subprocess.PIPE, text=True)
    assert p.stdout and p.stdout.readline().strip() == "READY"
    return p


async def bench(turns: int) -> None:
    local = await local_turns(turns)
    print(json.dumps({"scenario": "in-process FakeBrain", "e2e_p50_ms": round(statistics.median(local), 3),
                      "e2e_p95_ms": round(pct(local, 95), 3)}), flush=True)
    for scen in ("idle", "chat", "chat-blocking"):
        sock = tempfile.mktemp(prefix="p00-", suffix=".sock", dir="/tmp")
        srv = spawn("server", "--sock", sock, *(["--blocking"] if scen == "chat-blocking" else []))
        load = spawn("chatload", "--sock", sock) if scen != "idle" else None
        try:
            if load:
                await asyncio.sleep(1.0)
            r = await voice_turns(sock, turns)
        finally:
            for p in (load, srv):
                if p:
                    p.terminate()
                    p.wait()
            Path(sock).unlink(missing_ok=True)
        print(json.dumps({
            "scenario": scen, "turns": turns,
            "e2e_p50_ms": round(statistics.median(r["e2e"]), 3), "e2e_p95_ms": round(pct(r["e2e"], 95), 3),
            "added_p95_ms": round(pct(r["e2e"], 95) - pct(local, 95), 3),
            "hop_text_p95_ms": round(pct(r["hop_text"], 95), 3), "hop_text_max_ms": round(max(r["hop_text"]), 3),
            "hop_say_p95_ms": round(pct(r["hop_say"], 95), 3),
            "hop_tool_start_p95_ms": round(pct(r["hop_tool_start"], 95), 3),
            "loop_lag_p50_ms": round(r["lag"][0], 3), "loop_lag_p99_ms": round(r["lag"][1], 3),
            "loop_lag_max_ms": round(r["lag"][2], 3), "lag_samples": r["lag"][3],
            "chat_events": r["lag"][4], "chat_hashes": r["lag"][5]}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "server", "chatload"])
    ap.add_argument("--sock")
    ap.add_argument("--blocking", action="store_true")
    ap.add_argument("--turns", type=int, default=300)
    a = ap.parse_args()
    if a.mode == "server":
        asyncio.run(serve(a.sock, a.blocking))
    elif a.mode == "chatload":
        asyncio.run(chat_load(a.sock))
    else:
        asyncio.run(bench(a.turns))


if __name__ == "__main__":
    main()
