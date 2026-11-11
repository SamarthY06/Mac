"""RemoteBrain parity and contract (ARCHITECTURE W6, W7.1 "Brain-protocol parity", builder C gate).

- **Parity:** every orchestrator behaviour test in test_orchestrator.py runs again here with the
  orchestrator driving a `RemoteBrain` + `RemoteGate` over a real agentd (its Service and Server on
  a unix socket), whose voice:local conversation is backed by the same scripted FakeBrain. The
  assertions on `rig.brain` (asks, steers, interrupts, resets, ...) are made on that agentd-side
  brain, so they prove the calls really crossed the socket.
- **Event for event:** the same scripts produce the same BrainEvents locally and remotely
  (`tool_follows` included).
- **V1** a late "yes" for a closed CONFIRM never approves the newer one; **V2** a restarted macd
  (new epoch) never sees the old turn, which is interrupted, and turn ids never collide; **B2**
  handoff and connect_check are awaited via status events, and an interrupt right behind them
  lands within 100 ms.
- A smoke run against `python -m agentd --fake-brain --dev` (temp MAC_HOME) covers the real process.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent / "agentd"))

import agentd_kit as kit
import test_orchestrator as T
from orch_fakes import FakeListener, FakeOutput, FakeTTS, wait_until

from agentd import schema as s
from agentd.brains import BrainSpec
from agentd.server import Server
from agentd.service import AgentPaths, Service
from macd import config
from macd import phrases as P
from macd.brain import fake_brain as fb
from macd.brain import remote as rb
from macd.brain.remote import RemoteBrain
from macd.config import Config
from macd.contracts import (
    BrainErrorKind,
    BrainEvent,
    BrainEventKind,
    State,
    WindowReason,
)
from macd.orchestrator import Orchestrator
from macd.speech.segmenter import SentenceSegmenter
from macd.speech_queue import SpeechQueue

V = s.VOICE_CONV
K = BrainEventKind
VOICE_TOKEN = kit.VOICE_TOKEN


# ----------------------------------------------------------------------------- an agentd in-process


class ServerBrain(fb.FakeBrain):
    """agentd's voice brain in the parity rig. The orchestrator tests set `rig.brain.alive` to model
    the claude process; `mirror` hands that straight to the RemoteBrain's cache (in production the
    cache follows agentd's status after every turn, warm and sleep: tested separately below)."""

    mirror: Callable[[bool], None] | None = None

    @property
    def alive(self) -> bool:  # type: ignore[override]
        return self._alive_v

    @alive.setter
    def alive(self, value: bool) -> None:
        self._alive_v = value
        if self.mirror is not None:
            self.mirror(value)


class GatedBrain(kit.GateBrain):
    """agentd_kit's GateBrain (CONFIRM waits on the real ConfirmationGate), one timeout per request."""

    def __init__(self, spec: BrainSpec, script: fb.Script | None, timeouts: list[float]) -> None:
        super().__init__(spec, script)
        self.timeouts = list(timeouts)

    @property
    def confirm_timeout(self) -> float:  # read once per CONFIRM step
        return self.timeouts.pop(0) if self.timeouts else 5.0

    @confirm_timeout.setter
    def confirm_timeout(self, value: float) -> None:
        pass


@dataclass
class Agentd:
    root: Path
    svc: Service
    server: Server
    brains: list[Any]

    @property
    def sock(self) -> Path:
        return self.server.sock_path

    @property
    def brain(self) -> Any:
        return self.brains[-1]

    async def stop(self) -> None:
        for b in self.brains:
            for tid, _ in list(b.asks):
                await b.interrupt(tid)  # unblock scripted turns parked on a long step
        await self.server.close()
        await self.svc.close()


async def start_agentd(root: Path, make: Callable[[BrainSpec], Any], *, voice_grace_s: float = 0.2,
                       sock: Path | None = None) -> Agentd:
    home = root / "home"
    home.mkdir(parents=True, exist_ok=True)
    paths = AgentPaths(home=home, mac_home=root / "mac", log_dir=root / "logs")
    brains: list[Any] = []

    def factory(spec: BrainSpec) -> Any:
        b = make(spec)
        brains.append(b)
        return b

    svc = Service(paths, factory, voice_grace_s=voice_grace_s, lag_interval_s=0.01)
    server = Server(svc, kit.TOKENS, sock or root / "s.sock")
    await svc.start()
    await server.start()
    return Agentd(root, svc, server, brains)


@pytest.fixture
def short_root() -> AsyncIterator[Path]:
    d = Path(tempfile.mkdtemp(prefix="rb", dir="/tmp"))  # unix socket paths must stay short
    yield d
    shutil.rmtree(d, ignore_errors=True)


async def connect(agentd: Agentd, cfg: Config | None = None, **kw: Any) -> RemoteBrain:
    brain = RemoteBrain(cfg or Config(), token=lambda: VOICE_TOKEN, sock_path=agentd.sock, **kw)
    await brain.start()
    await asyncio.wait_for(brain.connected.wait(), 3)
    return brain


async def collect(brain: Any, until_done: int = 1, timeout: float = 3.0) -> list[BrainEvent]:
    out: list[BrainEvent] = []

    async def run() -> None:
        done = 0
        async for ev in brain.events():
            out.append(ev)
            if ev.kind in (K.DONE, K.ERROR):
                done += 1
                if done >= until_done:
                    return

    await asyncio.wait_for(run(), timeout)
    return out


# ----------------------------------------------------------------------------- the parity rig


class RecordingGate:
    """FakeGate's record (resolved, revokes) over the real RemoteGate."""

    def __init__(self, inner: rb.RemoteGate) -> None:
        self.inner = inner
        self.resolved: list[bool] = []
        self.revokes = 0

    def resolve(self, approved: bool) -> None:
        self.resolved.append(approved)
        self.inner.resolve(approved)

    def revoke_all(self) -> None:
        self.revokes += 1
        self.inner.revoke_all()


@dataclass
class RemoteRig(T.Rig):
    remote: RemoteBrain | None = None
    agentd: Agentd | None = None

    def inject(self, kind: BrainEventKind, turn_id: int, **kw: object) -> None:
        """A raw event straight into the stream the orchestrator consumes (RemoteBrain's)."""
        assert self.remote is not None
        self.remote._out.put_nowait(BrainEvent(kind, turn_id, **kw))  # type: ignore[arg-type]


async def make_remote_rig(cfg: Config | None = None, outputs: dict[str, FakeOutput] | None = None,
                          segmenter: Callable[[], SentenceSegmenter] | None = None, *,
                          make_brain: Callable[[BrainSpec, dict[str, list[fb.Step]]], Any] | None = None,
                          ) -> tuple[RemoteRig, asyncio.Task]:
    cfg = cfg or T.make_cfg()
    root = Path(tempfile.mkdtemp(prefix="rb", dir="/tmp"))
    scripts: dict[str, list[fb.Step]] = {}
    box: list[RemoteBrain] = []

    def default_brain(spec: BrainSpec, sc: dict[str, list[fb.Step]]) -> Any:
        b = ServerBrain(lambda t: sc.get(t, [fb.text("Okay. ")]))
        b.mirror = lambda v: box[0]._set_alive(v) if box else None
        return b

    make = make_brain or default_brain
    agentd = await start_agentd(root, lambda spec: make(spec, scripts))
    remote = await connect(agentd, cfg)
    box.append(remote)
    server_brain = await agentd.svc.ensure_voice_brain(agentd.svc.conversation(V))

    tts = next(iter(outputs.values())).tts if outputs else FakeTTS()
    out = outputs["local"] if outputs else FakeOutput(tts)
    speech = SpeechQueue(tts, outputs or {"local": out})
    gate, mic = RecordingGate(remote.gate), FakeListener("local")
    orch = Orchestrator(cfg, brain=remote, gate=gate, speech=speech, earcon=T.fake_earcon,
                        segmenter_factory=segmenter or (lambda: SentenceSegmenter(min_chars=0)),
                        describe_tool=lambda n, i: f"Using {n}")
    events: list[dict] = []
    orch.subscribe(events.append)
    orch.add_listener(mic)
    task = asyncio.create_task(orch.run())
    await asyncio.sleep(0.01)
    rig = RemoteRig(orch, server_brain, gate, tts, out, mic, speech, events, scripts,  # type: ignore[arg-type]
                    remote=remote, agentd=agentd)
    return rig, task


async def teardown_remote(r: RemoteRig, task: asyncio.Task) -> None:
    r.orch.close()
    await asyncio.wait_for(task, 2)
    assert r.remote is not None and r.agentd is not None
    await r.agentd.stop()
    await r.remote.close()
    shutil.rmtree(r.agentd.root, ignore_errors=True)


@pytest.fixture(autouse=True)
def _remote_rigs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parity tests below call test_orchestrator's make_rig/teardown: point them at agentd."""
    monkeypatch.setattr(T, "make_rig", make_remote_rig)
    monkeypatch.setattr(T, "teardown", teardown_remote)


@pytest.fixture
async def rig() -> AsyncIterator[RemoteRig]:
    r, task = await make_remote_rig()
    r.brain.alive = True
    try:
        yield r
    finally:
        await teardown_remote(r, task)


# Local-brain shape only: they delete a method from the in-process brain's class, or patch it with a
# keyword-only signature. RemoteBrain always implements both; see test_screen_lock_reaches_agentd.
LOCAL_ONLY = {"test_brain_without_handoff_is_fine", "test_screen_lock_without_brain_support",
              "test_screen_lock_events"}


def _uses_rig(fn: Any) -> bool:
    return "rig" in inspect.signature(fn).parameters or "make_rig(" in inspect.getsource(fn)


PARITY = sorted(n for n, f in vars(T).items() if n.startswith("test_") and inspect.iscoroutinefunction(f)
                and _uses_rig(f) and n not in LOCAL_ONLY)
for _name in PARITY:
    globals()[f"test_parity_{_name[5:]}"] = getattr(T, _name)


def test_parity_covers_the_orchestrator_suite() -> None:
    rig_tests = [n for n, f in vars(T).items() if n.startswith("test_") and inspect.iscoroutinefunction(f)
                 and _uses_rig(f)]
    assert len(PARITY) >= 50 and set(rig_tests) - set(PARITY) == LOCAL_ONLY


async def test_screen_lock_reaches_agentd(rig: RemoteRig) -> None:
    await rig.orch.power("screen_locked")
    await wait_until(lambda: rig.brain.locked is True)
    assert rig.orch.locked is True and rig.orch.status()["locked"] is True
    tid = await T.start_turn(rig, "what time is it")
    assert rig.orch.turn is not None and rig.orch.turn.id == tid
    await rig.orch.power("screen_unlocked")
    await wait_until(lambda: rig.brain.locked is False)


# ----------------------------------------------------------------------------- event for event

SCRIPT = [
    fb.text("Let me check your calendar."),
    fb.Step(K.TEXT, text="", data={"tool_follows": True}),
    fb.Step(K.TOOL_START, tool_name="mcp__mac__calendar", tool_input={"day": "today"},
            data={"narration": "Checking your calendar"}),
    fb.Step(K.TOOL_END, tool_name="mcp__mac__calendar"),
    fb.say("Which one, the first or the second?"),
    fb.Step(K.STATUS, data={"chrome_connected": True, "session_id": "abc"}),
    fb.Step(K.CONFIRM, text="I'll email Bob the agenda", data={"target": "bob@example.com"}),
    fb.text("You have two meetings."),
]


def _norm(ev: BrainEvent) -> tuple[Any, ...]:
    data = {k: v for k, v in ev.data.items() if k in ("tool_follows", "narration", "target", "interrupted",
                                                      "chrome_connected", "session_id")}
    return (ev.kind, ev.turn_id, ev.text, ev.tool_name, ev.is_error, ev.error_kind, data)


@pytest.mark.parametrize("script", [SCRIPT, [fb.text("Hi."), fb.error(BrainErrorKind.NETWORK, "Offline.")]],
                         ids=["full", "error"])
async def test_events_match_local_event_for_event(short_root: Path, script: list[fb.Step]) -> None:
    local = fb.FakeBrain(lambda t: list(script))
    await local.start()
    await local.ask(1, "go")
    want = await collect(local)
    await local.close()

    agentd = await start_agentd(short_root, lambda spec: fb.FakeBrain(lambda t: list(script)))
    brain = await connect(agentd)
    try:
        await brain.ask(1, "go")
        got = await collect(brain)
    finally:
        await brain.close()
        await agentd.stop()
    assert [_norm(e) for e in got] == [_norm(e) for e in want]
    starts = [e for e in got if e.kind is K.TOOL_START]
    ends = [e for e in got if e.kind is K.TOOL_END]
    assert all(e.tool_use_id for e in starts) and [e.tool_use_id for e in starts] == [e.tool_use_id for e in ends]
    confirms = [e for e in got if e.kind is K.CONFIRM]
    assert all(e.data["question_id"] for e in confirms)


async def test_interrupt_drains_and_next_ask_waits(short_root: Path) -> None:
    long = [fb.text(f"{i} ", delay=0.05) for i in range(50)]
    agentd = await start_agentd(short_root, lambda spec: fb.FakeBrain({"long": long, "short": [fb.text("51")]}.get,
                                                                      drain_s=0.1))
    brain = await connect(agentd)
    try:
        await brain.ask(1, "long")
        await asyncio.sleep(0.12)
        await brain.interrupt(1)
        await brain.ask(2, "short")
        evs = await collect(brain, until_done=2)
    finally:
        await brain.close()
        await agentd.stop()
    t1 = [e for e in evs if e.turn_id == 1]
    assert t1[-1].kind is K.DONE and t1[-1].data["interrupted"] is True
    assert len([e for e in t1 if e.kind is K.TEXT]) < 10
    first2 = next(i for i, e in enumerate(evs) if e.turn_id == 2)
    assert all(e.turn_id == 2 for e in evs[first2:])
    assert [e.text for e in evs if e.turn_id == 2 and e.kind is K.TEXT] == ["51"]


async def test_steer_reaches_the_running_turn_and_finished_interrupt_is_noop(short_root: Path) -> None:
    agentd = await start_agentd(short_root, lambda spec: fb.FakeBrain(
        {"a": [fb.text("x", delay=0.2)]}.get, steer_script=lambda t: [fb.text(f"steered {t}")]))
    brain = await connect(agentd)
    try:
        await brain.ask(1, "a")
        await asyncio.sleep(0.05)
        await brain.steer(1, "more")
        evs = await collect(brain)
        await brain.interrupt(1)  # finished: no-op, nothing more
        await asyncio.sleep(0.05)
        assert brain._out.empty()
    finally:
        await brain.close()
        await agentd.stop()
    assert [e.text for e in evs if e.kind is K.TEXT] == ["x", "steered more"]
    assert agentd.brain.steers == [(1, "more")]


# ----------------------------------------------------------------------------- V1: confirm ids


async def _wait_ev(brain: RemoteBrain, kind: BrainEventKind, timeout: float = 3.0) -> BrainEvent:
    async def get() -> BrainEvent:
        while True:
            ev = await brain._out.get()
            if ev is not None and ev.kind is kind:
                return ev

    return await asyncio.wait_for(get(), timeout)


async def test_v1_late_yes_never_approves_a_newer_confirm(short_root: Path) -> None:
    """Q1 times out in the brain, the model asks again (Q2), and only then does the "yes" for Q1
    arrive. It must not approve Q2."""
    script = [kit.confirm_step("Send the report to Bob?"), kit.confirm_step("Send the report to Bob now?")]
    agentd = await start_agentd(short_root, lambda spec: GatedBrain(spec, lambda t: list(script), [0.2, 1.0]))
    brain = await connect(agentd)
    try:
        await brain.ask(1, "email Bob the report")
        q1 = await _wait_ev(brain, K.CONFIRM)
        assert brain.gate.pending == (1, "Send the report to Bob?")
        q2 = await _wait_ev(brain, K.CONFIRM)  # Q1 timed out in the brain; Q2 replaced it
        assert q2.data["question_id"] != q1.data["question_id"]
        assert brain.gate.question_id == q1.data["question_id"]  # still bound to the one being answered
        brain.gate.resolve(True)  # the user's late "yes" to Q1
        await _wait_ev(brain, K.DONE)
        assert agentd.brain.confirm_results == [False, False]  # Q2 timed out unanswered
        closed = [json.loads(x) for x in agentd.svc.conversation(V).stream.lines()]
        outcomes = {e["question_id"]: e["outcome"] for e in closed if e["ev"] == "question_closed"}
        assert outcomes[q1.data["question_id"]] == "skipped" and outcomes[q2.data["question_id"]] != "approved"
    finally:
        await brain.close()
        await agentd.stop()


async def test_v1_yes_in_time_approves_its_own_confirm(short_root: Path) -> None:
    agentd = await start_agentd(short_root, lambda spec: GatedBrain(spec, lambda t: [kit.confirm_step()], [3.0]))
    brain = await connect(agentd)
    try:
        await brain.ask(1, "email Bob")
        await _wait_ev(brain, K.CONFIRM)
        brain.gate.resolve(True)
        assert (await _wait_ev(brain, K.TEXT)).text == "sent"
        assert agentd.brain.confirm_results == [True] and brain.gate.pending is None
    finally:
        await brain.close()
        await agentd.stop()


async def test_v1_orchestrator_race_never_approves_the_new_confirm() -> None:
    """The same race end to end: the orchestrator is still CONFIRMING Q1 (its window is longer than
    the brain's) when Q2 arrives; it ignores Q2 (N16), and the spoken "yes" goes to Q1 only."""
    cfg = T.make_cfg()
    cfg.listen.confirm_window_s = 3.0

    def gated(spec: BrainSpec, sc: dict[str, list[fb.Step]]) -> Any:
        return GatedBrain(spec, lambda t: sc.get(t, [fb.text("Okay. ")]), [0.2, 1.0])

    r, task = await make_remote_rig(cfg, make_brain=gated)
    try:
        r.scripts["email Bob the report"] = [kit.confirm_step("I'll send the report to Bob"),
                                             kit.confirm_step("I'll send the report to Bob now")]
        r.mic.say("email Bob the report", woke=True)
        await wait_until(lambda: r.orch.state == State.CONFIRMING)
        await wait_until(lambda: len(r.brain.confirm_results) == 1, timeout=2)  # Q1 expired in the brain
        await asyncio.sleep(0.05)  # Q2 is open now, and ignored by the orchestrator
        assert r.orch.state == State.CONFIRMING and r.orch.metrics.counters["confirms"] == 1
        r.mic.say("yes", window=WindowReason.CONFIRM)
        await wait_until(lambda: r.gate.resolved == [True])
        await wait_until(lambda: len(r.brain.confirm_results) == 2, timeout=3)
        assert r.brain.confirm_results == [False, False]
    finally:
        await teardown_remote(r, task)


async def test_v1_cancel_order_resolve_revoke_interrupt(short_root: Path) -> None:
    """resolve(False), revoke_all(), interrupt arrive in that order on one connection."""
    agentd = await start_agentd(short_root, lambda spec: GatedBrain(spec, lambda t: [kit.confirm_step()], [3.0]))
    brain = await connect(agentd)
    try:
        await brain.ask(1, "email Bob")
        await _wait_ev(brain, K.CONFIRM)
        brain.gate.resolve(False)
        brain.gate.revoke_all()
        await brain.interrupt(1)
        await _wait_ev(brain, K.DONE)
        assert agentd.brain.confirm_results == [False] and agentd.brain.interrupts == [1]
        assert not agentd.svc.conversation(V).gate.has_token(1)
    finally:
        await brain.close()
        await agentd.stop()


# ----------------------------------------------------------------------------- V2: client epochs


async def test_v2_restarted_macd_never_sees_the_old_turn(short_root: Path) -> None:
    script = {"long task": kit.slow("old", n=300, delay=0.01), "hello": [fb.text("Hi again.")]}
    agentd = await start_agentd(short_root, lambda spec: fb.FakeBrain(script.get), voice_grace_s=30.0)
    old = await connect(agentd)
    await old.ask(1, "long task")
    await _wait_ev(old, K.SAY)
    await old.close()  # macd died mid turn (the grace rule is far away: the epoch rule acts)

    new = await connect(agentd)
    try:
        assert new.epoch != old.epoch
        await wait_until(lambda: agentd.brain.interrupts == [1], msg="old turn interrupted on the new hello")
        await new.ask(1, "hello")  # the orchestrator's counter restarted at 1
        evs = await collect(new)
        assert [(e.kind, e.turn_id, e.text) for e in evs] == [(K.TEXT, 1, "Hi again."), (K.DONE, 1, "")]
        assert evs[-1].data["interrupted"] is False
        assert agentd.brain.asks == [(1, "long task"), (2, "hello")]  # agentd's own turn ids differ
        uids = agentd.svc.conversation(V).uids
        assert len(uids) == 2 and {u.split(":")[-2] for u in uids} == {old.epoch, new.epoch}
    finally:
        await new.close()
        await agentd.stop()


async def test_v2_reconnect_keeps_the_epoch_and_ends_the_turn_plainly(short_root: Path) -> None:
    sock = short_root / "a.sock"
    agentd = await start_agentd(short_root / "one", lambda spec: fb.FakeBrain({"long": kit.slow(n=300)}.get),
                                sock=sock)
    brain = await connect(agentd)
    epoch = brain.epoch
    try:
        await brain.ask(1, "long")
        await _wait_ev(brain, K.SAY)
        await agentd.stop()  # agentd crashed
        err = await _wait_ev(brain, K.ERROR)
        assert err.turn_id == 1 and err.text == rb.UNREACHABLE and err.error_kind is BrainErrorKind.INTERNAL
        assert not brain.connected.is_set() and brain.alive is False
        agentd = await start_agentd(short_root / "two", lambda spec: fb.FakeBrain(), sock=sock)
        await asyncio.wait_for(brain.connected.wait(), 3)
        assert brain.epoch == epoch
        await brain.ask(2, "hi")
        evs = await collect(brain)
        assert [e.kind for e in evs] == [K.TEXT, K.DONE] and evs[0].text == "Okay."
    finally:
        await brain.close()
        await agentd.stop()


# ----------------------------------------------------------------------------- B2: long ops


async def test_b2_handoff_and_connect_check_await_their_status_events(short_root: Path) -> None:
    def make(spec: BrainSpec) -> Any:
        b = kit.GateBrain(spec, lambda t: kit.slow(n=300))
        b.connect_delay = b.handoff_delay = 0.4
        return b

    agentd = await start_agentd(short_root, make)
    brain = await connect(agentd)
    try:
        await brain.ask(1, "long")
        await _wait_ev(brain, K.SAY)
        check = asyncio.create_task(brain.connect_check())
        handoff = asyncio.create_task(brain.handoff())
        await asyncio.sleep(0.01)
        t0 = time.monotonic()
        await brain.interrupt(1)
        await wait_until(lambda: agentd.brain.interrupts == [1], timeout=0.1)
        assert time.monotonic() - t0 < 0.1
        assert not check.done() and not handoff.done()
        res = await asyncio.wait_for(check, 3)
        assert res["auth"] == "ok" and res["chrome"] == "connected" and "check_id" not in res
        assert await asyncio.wait_for(handoff, 3) is True
        assert agentd.brain.connect_checks == 1 and agentd.brain.handoffs == 1
    finally:
        await brain.close()
        await agentd.stop()


# ----------------------------------------------------------------------------- RemoteBrain details


async def test_unreachable_agentd_fails_plainly_and_never_raises(short_root: Path) -> None:
    brain = RemoteBrain(Config(), token=lambda: VOICE_TOKEN, sock_path=short_root / "none.sock",
                        backoff_s=(0.05,))
    await brain.start()
    try:
        t0 = time.monotonic()
        await brain.ask(1, "hello")
        ev = await _wait_ev(brain, K.ERROR)
        assert ev.turn_id == 1 and ev.text == rb.UNREACHABLE and time.monotonic() - t0 < 2
        await brain.warm()
        await brain.interrupt(1)
        await brain.steer(1, "x")
        await brain.reset()
        await brain.sleep()
        assert await brain.handoff() is False
        assert (await brain.connect_check())["auth"] == "error"
        brain.gate.resolve(True)
        brain.gate.revoke_all()
        with pytest.raises(ConnectionError):
            await brain.stop_other()
    finally:
        await brain.close()


async def test_no_token_means_not_connected(short_root: Path) -> None:
    agentd = await start_agentd(short_root, lambda spec: fb.FakeBrain())
    token: list[str] = []
    brain = RemoteBrain(Config(), token=lambda: token[0] if token else None, sock_path=agentd.sock,
                        backoff_s=(0.05,))
    await brain.start()
    try:
        await asyncio.sleep(0.15)
        assert not brain.connected.is_set()
        token.append(VOICE_TOKEN)
        await asyncio.wait_for(brain.connected.wait(), 2)
    finally:
        await brain.close()
        await agentd.stop()


async def test_voice_settings_follow_macd_config(short_root: Path) -> None:
    agentd = await start_agentd(short_root, lambda spec: kit.GateBrain(spec, None))
    cfg = Config()
    cfg.brain.model, cfg.brain.chrome, cfg.brain.setting_sources = "opus", False, ["user", "project"]
    cfg.name, cfg.tts.lang = "Jarvis", "fr-fr"
    brain = await connect(agentd, cfg)
    try:
        await brain.ask(1, "hi")
        await collect(brain)
        vs = agentd.svc.voice_settings
        assert (vs.model, vs.chrome, vs.setting_sources, vs.name, vs.language) == \
            ("opus", False, ["user", "project"], "Jarvis", "fr-fr")
        assert agentd.brain.spec.model == "opus" and agentd.brain.name == "Jarvis"
        cfg.name = "Friday"  # the Settings window changed it: pushed before the next ask
        await brain.ask(2, "hi")
        await collect(brain)
        assert agentd.svc.voice_settings.name == "Friday"
    finally:
        await brain.close()
        await agentd.stop()


async def test_alive_follows_agentd_status(short_root: Path) -> None:
    agentd = await start_agentd(short_root, lambda spec: fb.FakeBrain())
    brain = await connect(agentd)
    try:
        assert brain.alive is False
        await brain.warm()
        assert brain.alive is True and agentd.brain.warms == 1
        await brain.sleep()
        assert brain.alive is False and agentd.brain.sleeps == 1
    finally:
        await brain.close()
        await agentd.stop()


async def test_status_notify_and_chrome_reach_macd(short_root: Path) -> None:
    notes: list[tuple[str, str, str | None]] = []
    chrome: list[bool] = []
    script = [fb.Step(K.STATUS, data={"chrome_connected": True}), fb.Step(K.STATUS, data={"chrome_connected": True}),
              fb.Step(K.STATUS, data={"notify": {"title": "Mac", "body": "Saved.", "path": "/tmp/x"}}),
              fb.text("Done.")]
    agentd = await start_agentd(short_root, lambda spec: fb.FakeBrain(lambda t: list(script)))
    brain = await connect(agentd, on_notify=lambda *a: notes.append(a), chrome_status_cb=chrome.append)
    try:
        await brain.ask(1, "save it")
        evs = await collect(brain)
    finally:
        await brain.close()
        await agentd.stop()
    assert notes == [("Mac", "Saved.", "/tmp/x")] and chrome == [True]
    assert [e.kind for e in evs] == [K.STATUS, K.STATUS, K.TEXT, K.DONE]  # notify is not a STATUS


def test_voice_token_sources(tmp_path: Path) -> None:
    assert rb.voice_token('{"token": "abc"}') == "abc"
    assert rb.voice_token(None) is None and rb.voice_token("not json") is None
    config.RUN_DIR.mkdir(parents=True, exist_ok=True)
    (config.RUN_DIR / "dev-tokens.json").write_text(json.dumps({"voice": "v1", "chat": "c1"}))
    assert rb.voice_token(None) == "v1" and rb.voice_token('{"token": "abc"}') == "abc"


# ----------------------------------------------------------------------------- stop the other task


async def test_stop_other_both_modes(short_root: Path) -> None:
    agent = await kit.make_agent(short_root)
    try:
        agent.scripts[kit.CHAT] = lambda t: kit.slow(n=300, delay=0.01)
        chat = await agent.client(kit.CHAT_TOKEN, [kit.CHAT])
        await chat.call({"op": "ask", "conv": kit.CHAT, "turn": 1, "text": "long"})
        await chat.event("say")
        # local mode: a short voice-token connection just for this op
        assert await rb.stop_other_once(VOICE_TOKEN, "e-local", agent.sock) == [kit.CHAT]
        assert (await chat.event("done"))["data"]["interrupted"] is True
        assert await rb.stop_other_once(VOICE_TOKEN, "e-local", agent.sock) == []
        # remote mode: on the brain's own connection
        await chat.call({"op": "ask", "conv": kit.CHAT, "turn": 2, "text": "long"})
        await chat.event("say")
        brain = RemoteBrain(Config(), token=lambda: VOICE_TOKEN, sock_path=agent.sock)
        await brain.start()
        await asyncio.wait_for(brain.connected.wait(), 3)
        try:
            assert await brain.stop_other() == [kit.CHAT]
        finally:
            await brain.close()
        with pytest.raises(ConnectionError):
            await rb.stop_other_once(None, "e", agent.sock)
        await chat.close()
    finally:
        await kit.stop_agent(agent)


@pytest.mark.parametrize(("t", "yes"), [("stop the other task", True), ("Stop WhatsApp.", True),
                                        ("okay stop the other task please", True), ("stop", False),
                                        ("stop the timer", False), ("cancel the other task", True)])
def test_is_stop_other(t: str, yes: bool) -> None:
    assert P.is_stop_other(t) is yes and not (yes and P.is_stop(t))


async def _local_rig_with_stop_other(result: Any) -> tuple[T.Rig, asyncio.Task, list[int]]:
    calls: list[int] = []
    tts = FakeTTS()
    out = FakeOutput(tts)
    speech = SpeechQueue(tts, {"local": out})
    scripts: dict[str, list[fb.Step]] = {}
    brain = fb.FakeBrain(lambda t: scripts.get(t, [fb.text("Okay. ")]))
    await brain.start()
    brain.alive = True

    async def stop_other() -> Any:
        calls.append(1)
        if isinstance(result, Exception):
            raise result
        return result

    from orch_fakes import FakeGate

    gate, mic = FakeGate(), FakeListener("local")
    orch = Orchestrator(T.make_cfg(), brain=brain, gate=gate, speech=speech, earcon=T.fake_earcon,
                        segmenter_factory=lambda: SentenceSegmenter(min_chars=0),
                        describe_tool=lambda n, i: f"Using {n}", stop_other=stop_other)
    events: list[dict] = []
    orch.subscribe(events.append)
    orch.add_listener(mic)
    task = asyncio.create_task(orch.run())
    await asyncio.sleep(0.01)
    return T.Rig(orch, brain, gate, tts, out, mic, speech, events, scripts), task, calls


@pytest.mark.parametrize(("result", "line"), [(["whatsapp:1@g.us"], P.STOPPED_OTHER), ([], P.NO_OTHER_TASK),
                                              (ConnectionError("down"), P.STOP_OTHER_FAILED)])
async def test_spoken_stop_other_when_idle(result: Any, line: str) -> None:
    r, task, calls = await _local_rig_with_stop_other(result)
    try:
        r.mic.say("stop the other task", woke=True)
        await wait_until(lambda: line in r.spoken)
        assert calls == [1] and r.brain.asks == [] and r.orch.state == State.IDLE
    finally:
        await _local_td(r, task)


async def test_spoken_stop_other_mid_turn_keeps_the_voice_turn() -> None:
    r, task, calls = await _local_rig_with_stop_other(["whatsapp:1@g.us"])
    try:
        r.scripts["read my email"] = [T.pause()]
        r.mic.say("read my email", woke=True)
        await wait_until(lambda: r.brain.asks)
        r.mic.say("stop the other task", woke=True)
        await wait_until(lambda: P.STOPPED_OTHER in r.spoken)
        assert calls == [1] and r.brain.interrupts == [] and r.brain.steers == []
        assert r.orch.turn is not None and r.orch.turn.id == 1
        other = T.FakeListener("other")
        r.orch.add_listener(other)
        other.say("stop the other task")  # no wake word, not the turn's speaker: ignored
        await asyncio.sleep(0.05)
        assert calls == [1]
        r.mic.say("stop the other task")  # the turn's speaker needs no wake word mid turn (2026-11-11)
        await wait_until(lambda: calls == [1, 1])
    finally:
        await _local_td(r, task)


async def _local_td(r: T.Rig, task: asyncio.Task) -> None:
    r.orch.close()
    await asyncio.wait_for(task, 2)
    for tid, _ in r.brain.asks:
        await r.brain.interrupt(tid)
    await r.brain.close()


# ----------------------------------------------------------------------------- __main__ wiring


def _daemon(mode: str | None = None, fake: bool = False) -> Any:
    from macd.__main__ import Daemon, build_parser

    cfg = Config()
    if mode is not None:
        cfg.brain.mode = mode
    return Daemon(cfg, build_parser().parse_args(["--fake-brain"] if fake else []))


def test_default_mode_is_local() -> None:
    from macd.brain.claude_brain import ClaudeBrain
    from macd.brain.gate import ConfirmationGate

    assert Config().brain.mode == "local"
    config.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CONFIG_PATH.write_text('[brain]\nmodel = ""\n')
    assert config.load().brain.mode == "local"
    for mode in (None, "local", "bogus"):
        d = _daemon(mode)
        gate = ConfirmationGate()
        brain = d._make_brain(gate)
        assert isinstance(brain, ClaudeBrain) and d.remote_gate is None and brain.gate is gate


def test_remote_mode_makes_a_remote_brain(monkeypatch: pytest.MonkeyPatch) -> None:
    from macd.brain.gate import ConfirmationGate

    monkeypatch.setenv("MAC_BRAIN_MODE", "remote")
    cfg = config.load()
    assert cfg.brain.mode == "remote"
    d = _daemon("remote")
    brain = d._make_brain(ConfirmationGate())
    assert isinstance(brain, RemoteBrain) and d.remote_gate is brain.gate and brain.epoch == d.client_epoch
    assert isinstance(_daemon("remote", fake=True)._make_brain(ConfirmationGate()), fb.FakeBrain)


async def test_daemon_stop_other_local_mode_uses_a_side_connection(short_root: Path) -> None:
    agent = await kit.make_agent(short_root)
    try:
        d = _daemon()
        d.brain = fb.FakeBrain()
        d.lifeline_line = json.dumps({"token": VOICE_TOKEN})
        import macd.brain.remote as remote_mod

        orig = remote_mod.agent_sock_path
        remote_mod.agent_sock_path = lambda: agent.sock  # type: ignore[assignment]
        try:
            assert await d._stop_other() == []
        finally:
            remote_mod.agent_sock_path = orig  # type: ignore[assignment]
    finally:
        await kit.stop_agent(agent)


# ----------------------------------------------------------------------------- the real agentd process


@pytest.fixture
def agentd_process(_isolated_mac_home: Path) -> Any:
    sock = Path(tempfile.mkdtemp(prefix="rp", dir="/tmp")) / "a.sock"
    env = {**os.environ, "MAC_HOME": str(_isolated_mac_home), "MAC_LOG_DIR": str(_isolated_mac_home / "logs")}
    proc = subprocess.Popen([sys.executable, "-m", "agentd", "--fake-brain", "--dev", "--socket", str(sock)],
                            env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 15
        while not sock.exists():
            if proc.poll() is not None or time.monotonic() > deadline:
                raise AssertionError(f"agentd did not start: {proc.stderr.read().decode() if proc.stderr else ''}")
            time.sleep(0.05)
        yield sock
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(5)
        if proc.poll() is None:
            proc.kill()
        shutil.rmtree(sock.parent, ignore_errors=True)


async def test_real_agentd_process_two_macd_lifetimes(agentd_process: Path) -> None:
    """`python -m agentd --fake-brain --dev`: the dev token is found, and two macd lifetimes both
    start at turn 1 without colliding."""
    for _ in range(2):
        brain = RemoteBrain(Config(), token=rb.voice_token, sock_path=agentd_process)
        await brain.start()
        try:
            await asyncio.wait_for(brain.connected.wait(), 5)
            await brain.ask(1, "hello")
            evs = await collect(brain, timeout=5)
            assert [(e.kind, e.text) for e in evs] == [(K.TEXT, "Okay."), (K.DONE, "")]
            assert (await brain.connect_check())["auth"] == "ok"
            assert await brain.handoff() is True
            assert await brain.stop_other() == []
        finally:
            await brain.close()


# ----------------------------------------------------------------------------- AG15/AG16/AG18 wiring


async def test_rename_and_reset_voice_tools_reach_settings() -> None:
    """Local mode: ClaudeBrain's mac.rename_assistant / mac.reset_voice call macd's callbacks, which
    go through the one settings path (name + wake phrases) and report a refusal in plain words."""
    from macd.brain.gate import ConfirmationGate
    from macd.settings import Settings

    d = _daemon()
    brain = d._make_brain(ConfirmationGate())
    assert brain.tools.on_rename == d._rename and brain.tools.on_reset_voice == d._reset_voice
    assert brain.tools.on_go_to_sleep == d._go_to_sleep
    assert await d._go_to_sleep() == "Mac is still starting up"
    assert await d._rename("Jarvis") == "settings are not ready yet"
    d.settings = Settings(d.cfg, save=lambda cfg: None)
    assert await d._rename("Jarvis") is None
    assert d.cfg.name == "Jarvis" and d.cfg.wake.phrases == ["hey jarvis", "ok jarvis"]
    assert "wake phrase" in (await d._rename("Hal 9000") or "")  # a wake phrase must be letters
    assert d.cfg.name == "Jarvis"
    assert "voice is unavailable" in (await d._reset_voice() or "")  # no speech stack in tests


async def test_remote_rename_and_reset_voice_status_events(short_root: Path) -> None:
    calls: list[Any] = []

    async def rename(name: str) -> None:
        calls.append(("rename", name))

    async def reset() -> None:
        calls.append(("reset",))

    async def sleep() -> None:
        calls.append(("sleep",))

    script = [fb.Step(K.STATUS, data={"rename": {"name": "Jarvis"}}), fb.Step(K.STATUS, data={"reset_voice": {}}),
              fb.Step(K.STATUS, data={"go_to_sleep": {}}), fb.text("Done.")]
    agentd = await start_agentd(short_root, lambda spec: fb.FakeBrain(lambda t: list(script)))
    brain = await connect(agentd, on_rename=rename, on_reset_voice=reset, on_go_to_sleep=sleep)
    try:
        await brain.ask(1, "call yourself Jarvis, use your normal voice, then go to sleep")
        evs = await collect(brain)
        await wait_until(lambda: len(calls) == 3)
    finally:
        await brain.close()
        await agentd.stop()
    assert calls == [("rename", "Jarvis"), ("reset",), ("sleep",)]
    assert [e.kind for e in evs] == [K.TEXT, K.DONE]  # handled in macd, not HUD statuses


async def test_remote_rename_refusal_reaches_the_model(short_root: Path) -> None:
    """N11: agentd's rename tool callback runs macd's pure check first, so an invalid name comes
    back to the model as the reason (never "Renamed"), and macd is only told about a valid one."""
    calls: list[str] = []
    specs: list[BrainSpec] = []

    async def rename(name: str) -> None:
        calls.append(name)

    def make(spec: BrainSpec) -> Any:
        specs.append(spec)
        return fb.FakeBrain()

    agentd = await start_agentd(short_root, make)
    brain = await connect(agentd, on_rename=rename)
    try:
        await brain.ask(1, "hello")
        await collect(brain)
        on_rename = next(sp.on_rename for sp in specs if sp.on_rename is not None)
        assert "wake phrase" in (on_rename("Hal 9000") or "")
        assert on_rename("Jarvis") is None
        await wait_until(lambda: calls == ["Jarvis"])
    finally:
        await brain.close()
        await agentd.stop()
    assert P.confirmation_line("x", "", None).endswith(P.CONFIRM_FALLBACK_QUESTION)


def test_remote_confirm_carries_the_models_question() -> None:
    brain = RemoteBrain(Config(), token=lambda: None)
    base = {"ev": "confirm", "conv": V, "boot": "b", "seq": 1, "turn": 3, "turn_uid": "b:v:e:3",
            "question_id": "q1", "summary": "I'll delete the draft to Bob", "target": "Mail: Delete"}
    brain._on_event({**base, "question": "Delete it?"})
    brain._on_event({**base, "seq": 2, "question_id": "q2"})
    a, b = brain._out.get_nowait(), brain._out.get_nowait()
    assert a.kind is K.CONFIRM and a.data["question"] == "Delete it?" and "question" not in b.data
    assert P.confirmation_line(a.text, a.data["target"], a.data.get("question")) == \
        "I'll delete the draft to Bob in Mail. Delete it?"
    assert P.confirmation_line(b.text, b.data["target"], b.data.get("question")).endswith(P.CONFIRM_FALLBACK_QUESTION)


async def test_alive_is_pushed_by_agentd_without_polling(short_root: Path) -> None:
    agentd = await start_agentd(short_root, lambda spec: fb.FakeBrain())
    brain = await connect(agentd)
    try:
        assert not hasattr(rb, "ALIVE_POLL_S")
        await agentd.svc.warm(V)  # agentd-side change, no op from this client
        await wait_until(lambda: brain.alive is True)
        await agentd.svc.sleep(V)
        await wait_until(lambda: brain.alive is False)
    finally:
        await brain.close()
        await agentd.stop()
