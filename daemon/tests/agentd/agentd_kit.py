"""Test kit for agentd: a real Service + Server on a short unix socket, FakeBrain conversations
whose CONFIRM steps really wait on the conversation's ConfirmationGate, a mocked router model, and a
small NDJSON client."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentd import schema as s
from agentd.brains import BrainSpec
from agentd.router import Router
from agentd.server import Server
from agentd.service import AgentPaths, Service
from agentd.workspaces import WorkspaceStore
from macd.brain import fake_brain as fb
from macd.contracts import BrainEvent, BrainEventKind

CHAT = "whatsapp:123@g.us"
OTHER_CHAT = "whatsapp:999@g.us"
NEVER_CHAT = "whatsapp:777@g.us"
TOKENS = {"v" * 64: s.ClientKind.VOICE, "c" * 64: s.ClientKind.CHAT, "a" * 64: s.ClientKind.APP,
          "t" * 64: s.ClientKind.TEST}
VOICE_TOKEN, CHAT_TOKEN, APP_TOKEN, TEST_TOKEN = "v" * 64, "c" * 64, "a" * 64, "t" * 64


def confirm_step(summary: str = "Send the email?", target: str = "bob@example.com") -> fb.Step:
    return fb.Step(BrainEventKind.CONFIRM, text=summary, data={"target": target})


def slow(text: str = "working", n: int = 200, delay: float = 0.02) -> list[fb.Step]:
    return [fb.say(f"{text} {i}", delay=delay) for i in range(n)]


class _GateOut(asyncio.Queue):  # type: ignore[type-arg]
    def __init__(self, brain: GateBrain) -> None:
        super().__init__()
        self.brain = brain

    async def put(self, ev: BrainEvent | None) -> None:
        await super().put(ev)
        if ev is not None and ev.kind is BrainEventKind.CONFIRM:
            ok = await self.brain.gate.request(ev.turn_id, ev.text, self.brain.confirm_timeout, "act")
            self.brain.confirm_results.append(ok)
            await super().put(BrainEvent(BrainEventKind.TEXT, ev.turn_id, text="sent" if ok else "not sent"))


class GateBrain(fb.FakeBrain):
    """FakeBrain whose CONFIRM step waits on the real ConfirmationGate (like ClaudeBrain's tool)."""

    def __init__(self, spec: BrainSpec, script: fb.Script | None, **kw: Any) -> None:
        super().__init__(script, **kw)
        self.spec = spec
        self.gate = spec.gate
        self.confirm_timeout = 5.0
        self.confirm_results: list[bool] = []
        self.connect_delay = 0.0
        self.handoff_delay = 0.0
        self._out = _GateOut(self)

    async def connect_check(self) -> dict[str, Any]:
        await asyncio.sleep(self.connect_delay)
        return await super().connect_check()

    async def handoff(self) -> bool:
        await asyncio.sleep(self.handoff_delay)
        return await super().handoff()


class MockModel:
    def __init__(self) -> None:
        self.reply: str | Exception = '{"folder": "general", "session": "new", "confidence": 0.9}'
        self.calls: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@dataclass
class Agent:
    root: Path
    paths: AgentPaths
    svc: Service
    server: Server
    model: MockModel
    scripts: dict[str, fb.Script]
    brains: dict[str, list[GateBrain]] = field(default_factory=dict)
    mdfind_out: str = ""
    sessions: list[Any] = field(default_factory=list)

    @property
    def sock(self) -> Path:
        return self.server.sock_path

    def brain(self, conv: str) -> GateBrain:
        return self.brains[conv][-1]

    async def client(self, token: str = VOICE_TOKEN, convs: list[str] | None = None, epoch: str = "e1",
                     subscribe: bool = True) -> Client:
        c = await Client.open(self.sock)
        if convs is None:
            convs = {VOICE_TOKEN: [s.VOICE_CONV], CHAT_TOKEN: [CHAT], APP_TOKEN: []}.get(token, [])
        r = await c.hello(token, convs, epoch)
        assert r["ok"], r
        if subscribe:
            for conv in convs:
                assert (await c.call({"op": "subscribe", "conv": conv}))["ok"]
        return c


def write_chat_config(mac_home: Path, chats: list[str], never: list[str], extra: str = "") -> None:
    p = mac_home / "chat" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    ids = [c.split(":", 1)[1] for c in chats]
    nids = [c.split(":", 1)[1] for c in never]
    p.write_text("[channels.whatsapp]\nenabled = true\n"
                 f"chats = {json.dumps(ids)}\nnever_chats = {json.dumps(nids)}\n{extra}")


def write_pairing(mac_home: Path, entries: dict[str, Any]) -> None:
    p = mac_home / "chat" / "pairing.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entries))


async def make_agent(root: Path, *, scripts: dict[str, fb.Script] | None = None, boot: str | None = None,
                     voice_grace_s: float = 0.2, queue_events: int = s.CLIENT_QUEUE_EVENTS,
                     uid: int | None = None, clock: Callable[[], float] | None = None,
                     lag_interval_s: float = 0.1, model: MockModel | None = None,
                     full_caps: bool = True, config: bool = True,
                     pairing: dict[str, Any] | None = None) -> Agent:
    """A real agentd on a tmp HOME. The seed gives CHAT and OTHER_CHAT profiles from the old
    `chats` key; `full_caps` makes sure they have every capability, so the pre-profile tests
    keep their folders and router."""
    home = root / "home"
    home.mkdir(exist_ok=True)
    paths = AgentPaths(home=home, mac_home=root / "mac", log_dir=root / "logs")
    if config:
        write_chat_config(paths.mac_home, [CHAT, OTHER_CHAT, NEVER_CHAT], [NEVER_CHAT])
    if pairing is not None:
        write_pairing(paths.mac_home, pairing)
    model = model or MockModel()
    agent_box: list[Agent] = []

    def factory(spec: BrainSpec) -> GateBrain:
        a = agent_box[0]
        script = a.scripts.get(spec.conv) or a.scripts.get("*")
        b = GateBrain(spec, script)
        a.brains.setdefault(spec.conv, []).append(b)
        return b

    kw: dict[str, Any] = {} if clock is None else {"clock": clock}
    ws = WorkspaceStore(paths.state_dir / "workspaces.json", home, seeds=())
    router = Router(ws, model, list_sessions=lambda **k: list(agent_box[0].sessions),
                    mdfind=lambda args: agent_box[0].mdfind_out, **kw)
    svc = Service(paths, factory, router=router, workspaces=ws, boot=boot, voice_grace_s=voice_grace_s,
                  lag_interval_s=lag_interval_s, **kw)
    server = Server(svc, TOKENS, root / "s.sock", queue_events=queue_events, uid=uid)
    agent = Agent(root, paths, svc, server, model, dict(scripts or {}))
    agent_box.append(agent)
    await svc.start()
    if full_caps:
        for conv in (CHAT, OTHER_CHAT):
            channel, _, cid = conv.partition(":")
            if svc.profiles.get(channel, cid) is not None:
                await svc.profiles.update(channel, cid, {"capabilities": dict.fromkeys(s.CAPABILITIES, True)})
    await server.start()
    return agent


async def stop_agent(a: Agent) -> None:
    await a.server.close()
    await a.svc.close()


class Client:
    def __init__(self, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        self.r, self.w = r, w
        self.id = 0
        self.replies: dict[int | None, asyncio.Future[dict[str, Any]]] = {}
        self.frames: list[dict[str, Any]] = []  # everything received, in order
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.errors: list[dict[str, Any]] = []  # id-less errors
        self.notices: list[Any] = []  # profile_changed
        self.closed = asyncio.Event()
        self._task = asyncio.create_task(self._read())

    @classmethod
    async def open(cls, sock: Path) -> Client:
        r, w = await asyncio.open_unix_connection(str(sock), limit=4 * s.MAX_LINE_BYTES)
        return cls(r, w)

    async def _read(self) -> None:
        try:
            while line := await self.r.readline():
                obj = json.loads(line)
                self.frames.append(obj)
                if obj.get("ev") in s.NOTICES:
                    self.notices.append(s.decode_notice(obj))  # connection-level, no conv or seq
                elif "ev" in obj:
                    s.decode_event(obj)  # every event matches the schema
                    self.events.put_nowait(obj)
                elif (f := self.replies.pop(obj.get("id"), None)) is not None:
                    f.set_result(obj)
                else:
                    self.errors.append(obj)
        except (ConnectionError, OSError):
            pass
        finally:
            self.closed.set()

    def send(self, obj: dict[str, Any]) -> asyncio.Future[dict[str, Any]]:
        self.id += 1
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self.replies[self.id] = fut
        self.w.write(s.encode_line({"id": self.id, **obj}))
        return fut

    async def call(self, obj: dict[str, Any], timeout: float = 3.0) -> dict[str, Any]:
        return await asyncio.wait_for(self.send(obj), timeout)

    async def hello(self, token: str, convs: list[str], epoch: str = "e1", api: str = s.API_VERSION,
                    **extra: Any) -> dict[str, Any]:
        return await self.call({"op": "hello", "api": api, "token": token, "client": "test/1",
                                "client_epoch": epoch, "conversations": convs, **extra})

    async def event(self, ev: str | None = None, timeout: float = 3.0,
                    pred: Callable[[dict[str, Any]], bool] | None = None) -> dict[str, Any]:
        async def get() -> dict[str, Any]:
            while True:
                e = await self.events.get()
                if (ev is None or e["ev"] == ev) and (pred is None or pred(e)):
                    return e

        return await asyncio.wait_for(get(), timeout)

    def drain(self) -> list[dict[str, Any]]:
        out = []
        while not self.events.empty():
            out.append(self.events.get_nowait())
        return out

    async def close(self) -> None:
        self.w.close()
        with contextlib.suppress(Exception):
            await self.w.wait_closed()
        self._task.cancel()
