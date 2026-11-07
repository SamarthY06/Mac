"""ClaudeBrain with a mocked SDK client: turn tagging, drain, steer, sessions, errors."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    CLIConnectionError,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from macd import config
from macd.brain import claude_brain as cb
from macd.brain.gate import ConfirmationGate
from macd.contracts import BrainErrorKind, BrainEvent
from macd.contracts import BrainEventKind as K

EOF = object()


def result(session: str = "sess-1", is_error: bool = False, **kw: Any) -> ResultMessage:
    return ResultMessage(subtype="success", duration_ms=10, duration_api_ms=5, is_error=is_error,
                         num_turns=1, session_id=session, total_cost_usd=0.01, **kw)


def assistant(*blocks: Any, mid: str | None = None) -> AssistantMessage:
    return AssistantMessage(content=list(blocks), model="m", message_id=mid)


def delta(text: str) -> StreamEvent:
    return StreamEvent(uuid="u", session_id="s",
                       event={"type": "content_block_delta", "index": 0,
                              "delta": {"type": "text_delta", "text": text}})


def init(chrome: str = "connected", version: str = "2.1.319") -> SystemMessage:
    return SystemMessage(subtype="init", data={"session_id": "sess-1", "claude_code_version": version,
                                              "mcp_servers": [{"name": "claude-in-chrome", "status": chrome},
                                                              {"name": "mac", "status": "connected"}]})


Responder = Callable[["MockClient", str], Any]


class MockClient:
    instances: ClassVar[list[MockClient]] = []

    def __init__(self, options: Any, responder: Responder, fail_connect: int = 0) -> None:
        self.options = options
        self.responder = responder
        self.queries: list[str] = []
        self.interrupts = 0
        self.connected = False
        self.disconnected = False
        self.q: asyncio.Queue[Any] = asyncio.Queue()
        self.fail_connect = fail_connect
        self.on_interrupt: Callable[[MockClient], None] | None = None
        self.tasks: list[asyncio.Task[Any]] = []
        MockClient.instances.append(self)

    async def connect(self) -> None:
        if self.fail_connect:
            raise CLIConnectionError("Failed to start Claude Code: ECONNREFUSED")
        self.connected = True

    async def query(self, text: str) -> None:
        self.queries.append(text)
        res = self.responder(self, text)
        if asyncio.iscoroutine(res):
            self.tasks.append(asyncio.create_task(res))

    async def interrupt(self) -> None:
        self.interrupts += 1
        if self.on_interrupt:
            self.on_interrupt(self)

    async def receive_messages(self):
        while True:
            m = await self.q.get()
            if m is EOF:
                return
            if isinstance(m, BaseException):
                raise m
            yield m

    async def disconnect(self) -> None:
        self.disconnected = True
        for t in self.tasks:
            t.cancel()

    def push(self, *msgs: Any) -> None:
        for m in msgs:
            self.q.put_nowait(m)


def simple(text: str) -> Responder:
    def r(c: MockClient, prompt: str) -> None:
        c.push(init(), delta(text), assistant(TextBlock(text)), result())
    return r


@pytest.fixture
def make_brain(tmp_path: Path, monkeypatch):
    MockClient.instances = []
    monkeypatch.setattr(config, "STATE_PATH", tmp_path / "state.json")

    def factory(responder: Responder, fail_connect: int = 0, **kw: Any) -> cb.ClaudeBrain:
        cfg = config.Config()
        cfg.brain.cli_path = "/bin/echo"
        for k, v in kw.pop("brain_cfg", {}).items():
            setattr(cfg.brain, k, v)
        counter = {"n": 0}

        def client_factory(opts: Any) -> MockClient:
            counter["n"] += 1
            return MockClient(opts, responder, fail_connect=max(0, fail_connect - counter["n"] + 1))

        kw.setdefault("chrome_running", lambda: False)
        kw.setdefault("frontmost", lambda: "")
        return cb.ClaudeBrain(cfg, kw.pop("gate", ConfirmationGate()), cwd=tmp_path / "ws",
                              client_factory=client_factory, **kw)

    return factory


async def collect(brain: cb.ClaudeBrain, n_terminal: int = 1, timeout: float = 3) -> list[BrainEvent]:
    out: list[BrainEvent] = []
    done = 0

    async def run() -> None:
        nonlocal done
        async for ev in brain.events():
            out.append(ev)
            if ev.kind in (K.DONE, K.ERROR):
                done += 1
                if done >= n_terminal:
                    return

    await asyncio.wait_for(run(), timeout)
    return out


async def test_basic_turn_with_partial_text_dedup(make_brain) -> None:
    brain = make_brain(simple("Fifty one."))
    await brain.start()
    await brain.ask(1, "17*3?")
    evs = await collect(brain)
    await brain.close()
    texts = [e.text for e in evs if e.kind == K.TEXT]
    assert texts == ["Fifty one."]  # delta emitted, TextBlock not duplicated
    assert all(e.turn_id == 1 for e in evs)
    status = next(e for e in evs if e.kind == K.STATUS)
    assert status.data["chrome_connected"] is True
    done = evs[-1]
    assert done.kind == K.DONE and done.data["session_id"] == "sess-1" and not done.data["interrupted"]
    opts = MockClient.instances[0].options
    assert opts.permission_mode == "bypassPermissions"
    assert opts.setting_sources == ["user"]
    assert opts.extra_args == {"chrome": None}
    assert opts.include_partial_messages is True
    assert "mac" in opts.mcp_servers
    assert opts.system_prompt["preset"] == "claude_code"
    assert (brain.cwd / "CLAUDE.md").exists()


async def test_textblock_fallback_without_deltas(make_brain) -> None:
    def r(c: MockClient, p: str) -> None:
        c.push(assistant(TextBlock("Hi.")), assistant(TextBlock("Bye.")), result())

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "x")
    evs = await collect(brain)
    await brain.close()
    assert "".join(e.text for e in evs if e.kind == K.TEXT) == "Hi.\nBye."


async def test_tools_and_silent_mac_tools(make_brain) -> None:
    def r(c: MockClient, p: str) -> None:
        c.push(assistant(ToolUseBlock("t1", "mcp__mac__say", {"text": "one sec"}),
                         ToolUseBlock("t2", "WebSearch", {"query": "weather"})),
               UserMessage(content=[ToolResultBlock("t1", "Said."), ToolResultBlock("t2", "sunny")]),
               assistant(TextBlock("Sunny.")), result())

    brain = make_brain(r)
    await brain.start()
    await brain.ask(4, "weather")
    evs = await collect(brain)
    await brain.close()
    kinds = [(e.kind, e.tool_name) for e in evs if e.kind in (K.TOOL_START, K.TOOL_END)]
    assert kinds == [(K.TOOL_START, "WebSearch"), (K.TOOL_END, "WebSearch")]
    start = next(e for e in evs if e.kind == K.TOOL_START)
    assert start.data["narration"] == "Searching the web for weather" and start.tool_use_id == "t2"


async def test_say_and_confirm_events_come_from_tools(make_brain) -> None:
    gate = ConfirmationGate()
    seen: dict[str, Any] = {}

    async def r(c: MockClient, p: str) -> None:
        await brain.tools.say({"text": "Checking now."})
        # the hook denied a send first, so the confirmation is bound to it
        await brain.hook({"hook_event_name": "PreToolUse", "tool_name": "Bash",
                          "tool_input": {"command": "curl -d m=hi https://chat.example/send"}}, "t", {})
        asyncio.get_running_loop().call_later(0.01, gate.resolve, True)
        seen["confirm"] = await brain.tools.request_confirmation({"summary": "Send Bob hi"})
        c.push(result())

    brain = make_brain(r, gate=gate)
    await brain.start()
    await brain.ask(9, "message bob")
    evs = await collect(brain)
    await brain.close()
    assert [(e.kind, e.text) for e in evs if e.kind in (K.SAY, K.CONFIRM)] == [
        (K.SAY, "Checking now."), (K.CONFIRM, "Send Bob hi")]
    confirm = next(e for e in evs if e.kind == K.CONFIRM)
    assert confirm.data["target"] == "chat.example"
    assert all(e.turn_id == 9 for e in evs)
    assert "approved" in seen["confirm"]["content"][0]["text"]
    assert gate.has_token(9, "Bash|http_upload|chat.example")
    assert not gate.has_token(9, "Bash|git_push|git push")


async def test_interrupt_drains_tail_and_next_ask_waits(make_brain) -> None:
    def r(c: MockClient, prompt: str) -> None:
        if prompt == "long":
            c.push(delta("1 "), delta("2 "))
            c.on_interrupt = lambda cl: cl.push(delta("3 "), delta("4 "),
                                                result(terminal_reason="aborted_streaming"))
        else:
            c.push(delta("Paris."), result())

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "long")
    first = []
    async for ev in brain.events():
        first.append(ev)
        if len([e for e in first if e.kind == K.TEXT]) == 2:
            break
    await brain.interrupt(1)
    await brain.ask(2, "capital of France?")
    evs = await collect(brain, n_terminal=2)
    await brain.close()
    t1 = [e for e in evs if e.turn_id == 1]
    assert [e.kind for e in t1] == [K.DONE] and t1[0].data["interrupted"] is True
    assert [e.text for e in evs if e.turn_id == 2 and e.kind == K.TEXT] == ["Paris."]
    client = MockClient.instances[0]
    assert client.interrupts == 1
    assert client.queries == ["long", "capital of France?"]  # second sent after drain


async def test_drain_timeout_reconnects(make_brain, monkeypatch) -> None:
    monkeypatch.setattr(cb, "DRAIN_TIMEOUT_S", 0.1)

    def r(c: MockClient, prompt: str) -> None:
        if prompt == "hang":
            c.push(delta("..."))  # never sends a ResultMessage, ignores interrupt
        else:
            c.push(delta("ok"), result())

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "hang")
    await asyncio.sleep(0.05)
    await brain.interrupt(1)
    await brain.ask(2, "next")
    evs = await collect(brain, n_terminal=2)
    await brain.close()
    assert [e.kind for e in evs if e.turn_id == 1 and e.kind != K.TEXT] == [K.DONE]
    assert [e.text for e in evs if e.turn_id == 2 and e.kind == K.TEXT] == ["ok"]
    assert len(MockClient.instances) == 2 and MockClient.instances[0].disconnected


async def test_steer_sends_query_during_turn(make_brain) -> None:
    async def r(c: MockClient, prompt: str) -> None:
        if prompt == "find people":
            c.push(delta("Searching. "))
            await asyncio.sleep(0.1)
            c.push(result(session="sess-a"))
        else:
            c.push(delta("Adding titles."), result(session="sess-a"))

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "find people")
    await asyncio.sleep(0.02)
    await brain.steer(1, "also add their titles")
    evs = await collect(brain)
    await brain.close()
    assert MockClient.instances[0].queries == ["find people", "also add their titles"]
    assert [e.kind for e in evs].count(K.DONE) == 1
    assert "".join(e.text for e in evs if e.kind == K.TEXT) == "Searching. Adding titles."


async def test_session_persist_and_resume(make_brain, tmp_path: Path) -> None:
    brain = make_brain(simple("hi"))
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    await brain.close()
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["brain"]["session_id"] == "sess-1"

    brain2 = make_brain(simple("hi"))
    await brain2.start()
    await brain2.ask(1, "x")
    await collect(brain2)
    await brain2.close()
    assert MockClient.instances[-1].options.resume == "sess-1"


async def test_stale_session_not_resumed(make_brain, tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text(json.dumps(
        {"other": 1, "brain": {"session_id": "old", "updated_at": 0}}))
    brain = make_brain(simple("hi"))
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    await brain.close()
    assert MockClient.instances[-1].options.resume is None
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["other"] == 1  # other subsystems' state is preserved


async def test_reset_starts_fresh(make_brain, tmp_path: Path) -> None:
    brain = make_brain(simple("hi"))
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    await brain.reset()
    await brain.ask(2, "y")
    await collect(brain)
    await brain.close()
    assert len(MockClient.instances) == 2
    assert MockClient.instances[1].options.resume is None


async def test_result_error_maps_to_error_event(make_brain) -> None:
    def r(c: MockClient, p: str) -> None:
        c.push(assistant(TextBlock("Invalid API key · Please run /login")),
               result(is_error=True, result="Invalid API key · Please run /login"))

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "x")
    evs = await collect(brain)
    await brain.close()
    err = evs[-1]
    assert err.kind == K.ERROR and err.error_kind == BrainErrorKind.AUTH
    assert "sign in" in err.text
    assert K.DONE not in [e.kind for e in evs]


async def test_cli_crash_emits_error_and_reconnects(make_brain) -> None:
    calls = {"n": 0}

    def r(c: MockClient, p: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            c.push(delta("par"), RuntimeError("Command failed: getaddrinfo ENOTFOUND"))
        else:
            c.push(delta("fine"), result())

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "a")
    evs = await collect(brain)
    assert evs[-1].kind == K.ERROR and evs[-1].error_kind == BrainErrorKind.NETWORK
    await brain.ask(2, "b")
    evs = await collect(brain)
    await brain.close()
    assert evs[-1].kind == K.DONE and evs[0].turn_id == 2
    assert len(MockClient.instances) == 2


async def test_circuit_breaker(make_brain) -> None:
    brain = make_brain(simple("x"), fail_connect=99)
    await brain.start()
    for t in (1, 2, 3):
        await brain.ask(t, "q")
        evs = await collect(brain)
        assert evs[-1].kind == K.ERROR and evs[-1].error_kind == BrainErrorKind.NETWORK
    n_clients = len(MockClient.instances)
    await brain.ask(4, "q")  # breaker open: immediate error, no connect attempt
    evs = await collect(brain)
    await brain.close()
    assert evs[-1].kind == K.ERROR and evs[-1].data.get("circuit_open")
    assert len(MockClient.instances) == n_clients


async def test_turn_timeout(make_brain) -> None:
    def r(c: MockClient, p: str) -> None:
        c.push(delta("thinking"))
        c.on_interrupt = lambda cl: cl.push(result(terminal_reason="aborted_streaming"))

    brain = make_brain(r, brain_cfg={"max_turn_s": 0.1})
    await brain.start()
    await brain.ask(1, "slow")
    evs = await collect(brain)
    await asyncio.sleep(0.05)
    await brain.close()
    assert evs[-1].kind == K.ERROR and evs[-1].error_kind == BrainErrorKind.TIMEOUT
    assert MockClient.instances[0].interrupts == 1


async def test_chrome_down_note_and_callback(make_brain) -> None:
    seen: list[bool] = []

    def r(c: MockClient, p: str) -> None:
        c.push(init(chrome="failed"), delta("ok"), result())

    brain = make_brain(r, chrome_status_cb=seen.append)
    await brain.start()
    await brain.ask(1, "a")
    await collect(brain)
    await brain.ask(2, "b")
    await collect(brain)
    await brain.close()
    assert seen == [False]
    assert MockClient.instances[0].queries[1].startswith(cb.CHROME_DOWN_NOTE)


async def test_hook_sees_cancelled_turn(make_brain) -> None:
    hold = asyncio.Event()

    async def r(c: MockClient, p: str) -> None:
        await hold.wait()
        c.push(result())

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "x")
    await asyncio.sleep(0.02)
    ok = await brain.hook({"hook_event_name": "PreToolUse", "tool_name": "Bash",
                           "tool_input": {"command": "ls"}}, "t", {})
    assert ok == {}
    await brain.interrupt(1)
    denied = await brain.hook({"hook_event_name": "PreToolUse", "tool_name": "Bash",
                               "tool_input": {"command": "ls"}}, "t", {})
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    hold.set()
    await collect(brain)
    await brain.close()


async def test_memory_allowed_only_when_asked(make_brain, tmp_path: Path) -> None:
    out: dict[str, Any] = {}

    async def r(c: MockClient, p: str) -> None:
        out[p] = await brain.tools.remember({"fact": "likes tea"})
        c.push(result())

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "what's the time")
    await collect(brain)
    await brain.ask(2, "remember that I like tea")
    await collect(brain)
    await brain.close()
    assert out["what's the time"].get("is_error")
    assert not out["remember that I like tea"].get("is_error")
    assert "- likes tea [kept]" in (tmp_path / "ws" / "memory.md").read_text()


async def settle(brain: cb.ClaudeBrain) -> None:
    """Let the owner task process everything queued so far."""
    for _ in range(5):
        await asyncio.sleep(0)
    while not brain._inbox.empty():
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.01)


async def test_start_is_lazy_and_warm_connects(make_brain) -> None:
    brain = make_brain(simple("hi"))
    await brain.start()
    await settle(brain)
    assert MockClient.instances == [] and not brain.alive
    await brain.warm()
    await brain.warm()
    await settle(brain)
    assert len(MockClient.instances) == 1 and brain.alive
    await brain.ask(1, "x")
    await collect(brain)
    await brain.close()
    assert len(MockClient.instances) == 1  # ask reused the warmed client


async def test_warm_failure_never_raises_and_surfaces_on_ask(make_brain) -> None:
    brain = make_brain(simple("hi"), fail_connect=1)
    await brain.start()
    await brain.warm()
    await settle(brain)
    assert not brain.alive
    await brain.ask(1, "x")
    evs = await collect(brain)
    await brain.close()
    assert evs[-1].kind == K.DONE  # second connect attempt succeeded


async def test_sleep_persists_session_and_next_ask_resumes(make_brain, tmp_path: Path) -> None:
    brain = make_brain(simple("hi"))
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    await brain.sleep()
    await brain.sleep()
    await settle(brain)
    assert not brain.alive and MockClient.instances[0].disconnected
    assert json.loads((tmp_path / "state.json").read_text())["brain"]["session_id"] == "sess-1"
    await brain.ask(2, "y")
    evs = await collect(brain)
    await brain.close()
    assert evs[-1].kind == K.DONE and evs[-1].turn_id == 2
    assert len(MockClient.instances) == 2 and MockClient.instances[1].options.resume == "sess-1"


async def test_sleep_mid_turn_is_ignored(make_brain) -> None:
    hold = asyncio.Event()

    async def r(c: MockClient, p: str) -> None:
        await hold.wait()
        c.push(delta("done"), result())

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "x")
    await settle(brain)
    await brain.sleep()
    await settle(brain)
    assert brain.alive
    hold.set()
    evs = await collect(brain)
    await brain.close()
    assert evs[-1].kind == K.DONE and "done" in "".join(e.text for e in evs)


async def test_idle_shutdown_fires_only_when_idle(make_brain) -> None:
    clock = {"t": 1000.0}
    hold = asyncio.Event()

    async def r(c: MockClient, p: str) -> None:
        await hold.wait()
        c.push(result())

    brain = make_brain(r, clock=lambda: clock["t"], brain_cfg={"idle_shutdown_min": 10})
    await brain.start()
    await brain.ask(1, "x")
    await settle(brain)
    clock["t"] += 11 * 60
    await brain._on_tick()
    assert brain.alive  # mid-turn: the idle timer must not fire
    hold.set()
    await collect(brain)
    clock["t"] += 5 * 60
    await brain._on_tick()
    assert brain.alive  # only 5 idle minutes
    clock["t"] += 6 * 60
    await brain._on_tick()
    assert not brain.alive and MockClient.instances[0].disconnected
    await brain.warm()
    await settle(brain)
    assert brain.alive and MockClient.instances[1].options.resume == "sess-1"
    await brain.close()


async def test_idle_shutdown_wakeup_is_scheduled(make_brain) -> None:
    brain = make_brain(simple("hi"), brain_cfg={"idle_shutdown_min": 10})
    await brain.start()
    assert brain._next_wakeup() is None
    await brain.warm()
    await settle(brain)
    wake = brain._next_wakeup()
    assert wake is not None and 590 < wake <= 600
    await brain.close()


async def test_sleep_then_stale_session_rotates(make_brain) -> None:
    wall = {"t": 1_000_000.0}
    brain = make_brain(simple("hi"), wall_clock=lambda: wall["t"])
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    await brain.sleep()
    await settle(brain)
    wall["t"] += (brain.cfg.brain.session_idle_rotate_min + 1) * 60
    await brain.ask(2, "y")
    await collect(brain)
    await brain.close()
    assert MockClient.instances[1].options.resume is None


async def test_setting_sources_from_config(make_brain) -> None:
    brain = make_brain(simple("hi"), brain_cfg={"setting_sources": ["user", "project", "bogus"]})
    assert brain.build_options(None).setting_sources == ["user", "project"]
    brain = make_brain(simple("hi"), brain_cfg={"setting_sources": []})
    opts = brain.build_options(None)
    assert opts.setting_sources == [] and opts.allowed_tools == [] and opts.disallowed_tools == []


async def test_owner_survives_handler_bug_and_speaks_connection_lost(make_brain, monkeypatch) -> None:
    brain = make_brain(simple("hi"))
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)

    async def boom(cmd: Any) -> None:
        raise RuntimeError("bug in handler")

    monkeypatch.setitem(brain._handlers, "warm", boom)
    await brain.warm()
    await settle(brain)
    assert not brain._task.done()  # owner still alive after the exception
    await brain.ask(2, "y")
    evs = await collect(brain)
    await brain.close()
    assert evs[-1].kind == K.DONE and evs[-1].turn_id == 2


async def test_handler_bug_mid_turn_emits_error_and_reconnects(make_brain, monkeypatch) -> None:
    async def r(c: MockClient, p: str) -> None:
        c.push(delta("partial"))
        if p == "first":
            c.push(RuntimeError("stream broke"))
        else:
            c.push(result())

    brain = make_brain(r)
    monkeypatch.setattr(cb.errors, "classify_exception", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    await brain.start()
    await brain.ask(1, "first")
    evs = await collect(brain)
    assert evs[-1].kind == K.ERROR and evs[-1].text == cb.errors.CONNECTION_LOST
    await brain.ask(2, "second")
    evs = await collect(brain)
    await brain.close()
    assert evs[-1].kind == K.DONE and len(MockClient.instances) == 2


async def test_dead_owner_task_is_restarted_on_next_command(make_brain) -> None:
    brain = make_brain(simple("hi"))
    await brain.start()
    brain._task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await brain._task
    await brain.ask(1, "x")
    evs = await collect(brain)
    await brain.close()
    assert evs[-1].kind == K.DONE


async def test_steer_folded_into_one_result_finishes_by_quiescence(make_brain, monkeypatch) -> None:
    monkeypatch.setattr(cb, "QUIET_AFTER_STEER_S", 0.1)

    async def r(c: MockClient, prompt: str) -> None:
        if prompt == "go":
            c.push(delta("a "))
            await asyncio.sleep(0.05)
            c.push(delta("b"), result())  # ONE result for both queries, nothing after it
        # the steer produces no messages of its own

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "go")
    await asyncio.sleep(0.01)
    await brain.steer(1, "also")
    evs = await collect(brain, timeout=2)
    await brain.close()
    assert evs[-1].kind == K.DONE and evs[-1].turn_id == 1
    assert MockClient.instances[0].queries == ["go", "also"]


async def test_steer_quiescence_is_cancelled_by_more_output(make_brain, monkeypatch) -> None:
    monkeypatch.setattr(cb, "QUIET_AFTER_STEER_S", 0.1)

    async def r(c: MockClient, prompt: str) -> None:
        if prompt == "go":
            c.push(delta("a "), result())
        else:
            await asyncio.sleep(0.05)
            c.push(delta("steer-text"))
            await asyncio.sleep(0.15)  # longer than the quiet window: still one turn
            c.push(result())

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "go")
    await brain.steer(1, "also")
    evs = await collect(brain, timeout=2)
    await brain.close()
    assert [e.kind for e in evs].count(K.DONE) == 1
    assert "steer-text" in "".join(e.text for e in evs if e.kind == K.TEXT)


async def test_stop_during_connect_is_honoured(make_brain) -> None:
    gate_evt = asyncio.Event()

    class SlowClient(MockClient):
        async def connect(self) -> None:
            await gate_evt.wait()
            await super().connect()

    cfg = config.Config()
    cfg.brain.cli_path = "/bin/echo"
    brain = cb.ClaudeBrain(cfg, ConfirmationGate(), cwd=Path("/tmp/unused"),
                           client_factory=lambda o: SlowClient(o, simple("hi")),
                           chrome_running=lambda: False, frontmost=lambda: "")
    brain.cwd = brain.cwd  # keep type checkers quiet
    await brain.start()
    await brain.ask(1, "x")
    await asyncio.sleep(0.02)
    await brain.interrupt(1)
    gate_evt.set()
    evs = await collect(brain)
    await brain.close()
    assert [e.kind for e in evs] == [K.DONE] and evs[0].data["interrupted"]
    assert MockClient.instances[-1].queries == []  # the query was never sent


async def test_connect_retries_with_resume_before_dropping_it(make_brain, tmp_path: Path) -> None:
    config.update_state(brain={"session_id": "keep-me", "updated_at": 10**12})
    brain = make_brain(simple("hi"), fail_connect=1, wall_clock=lambda: 10**12 + 1)
    await brain.start()
    await brain.ask(1, "x")
    evs = await collect(brain)
    await brain.close()
    assert evs[-1].kind == K.DONE
    assert [c.options.resume for c in MockClient.instances] == ["keep-me", "keep-me"]


async def test_chrome_note_dropped_when_chrome_appears(make_brain) -> None:
    clock = {"t": 1000.0}
    chrome = {"running": False}

    def r(c: MockClient, p: str) -> None:
        if len(c.queries) == 1:
            c.push(init(chrome="failed"))  # one init per connection, like the real CLI
        c.push(delta("ok"), result())

    brain = make_brain(r, clock=lambda: clock["t"], chrome_running=lambda: chrome["running"])
    await brain.start()
    await brain.ask(1, "a")
    await collect(brain)
    await brain.ask(2, "b")
    await collect(brain)
    assert MockClient.instances[0].queries[1].startswith(cb.CHROME_DOWN_NOTE)
    chrome["running"] = True
    clock["t"] += 61
    await brain.ask(3, "c")
    await collect(brain)
    # Chrome appeared: the note is dropped at once, and the CLI is reconnected at the next idle
    # moment so the SDK re-probes the extension on the following ask.
    assert not MockClient.instances[0].queries[2].startswith(cb.CHROME_DOWN_NOTE)
    await brain.ask(4, "d")
    await collect(brain)
    await brain.close()
    assert len(MockClient.instances) == 2 and MockClient.instances[1].options.resume == "sess-1"


async def test_notify_routes_through_callback_with_path(make_brain, tmp_path: Path) -> None:
    seen: list[tuple[str, str, str | None]] = []

    async def on_notify(title: str, body: str, path: str | None) -> bool:
        seen.append((title, body, path))
        return True

    brain = make_brain(simple("hi"), on_notify=on_notify)
    out = await brain.tools.notify({"title": "Done", "body": "20 people", "path": "~/Documents/Mac/people.csv"})
    assert out["content"][0]["text"] == "Notification shown."
    assert seen == [("Done", "20 people", str(Path("~/Documents/Mac/people.csv").expanduser()))]


async def test_second_confirmation_while_pending_is_distinct(make_brain) -> None:
    gate = ConfirmationGate()
    brain = make_brain(simple("hi"), gate=gate)
    first = asyncio.create_task(gate.request(1, "a", timeout=1, action="x"))
    await asyncio.sleep(0)
    out = await brain.tools.request_confirmation({"summary": "b"})
    assert out.get("is_error") and "already" in out["content"][0]["text"]
    gate.resolve(False)
    await first


async def test_interrupt_after_steer_swallows_every_result(make_brain) -> None:
    """MAJOR-1: both queries of an interrupted turn produce a result; neither may end turn 2."""
    def r(c: MockClient, prompt: str) -> None:
        if prompt == "long":
            c.push(delta("1 "))
            c.on_interrupt = lambda cl: cl.push(result(terminal_reason="aborted_streaming"),
                                                result(terminal_reason="aborted_streaming"))
        elif prompt == "steer":
            pass
        else:
            c.push(delta("Paris."), result())

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "long")
    await settle(brain)
    await brain.steer(1, "steer")
    await settle(brain)
    await brain.interrupt(1)
    await brain.ask(2, "capital?")
    evs = await collect(brain, n_terminal=2)
    await brain.close()
    t1 = [e for e in evs if e.turn_id == 1 and e.kind != K.TEXT]
    assert [e.kind for e in t1] == [K.DONE] and t1[0].data["interrupted"]
    t2 = [e for e in evs if e.turn_id == 2]
    assert [e.kind for e in t2] == [K.TEXT, K.DONE] and t2[0].text == "Paris."
    assert MockClient.instances[0].queries == ["long", "steer", "capital?"]


async def test_interrupt_after_folded_steer_finishes_by_quiescence(make_brain, monkeypatch) -> None:
    monkeypatch.setattr(cb, "QUIET_AFTER_STEER_S", 0.1)

    def r(c: MockClient, prompt: str) -> None:
        if prompt == "long":
            c.push(delta("1 "))
            c.on_interrupt = lambda cl: cl.push(result(terminal_reason="aborted_streaming"))  # one result only
        elif prompt != "steer":
            c.push(delta("ok"), result())

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "long")
    await settle(brain)
    await brain.steer(1, "steer")
    await settle(brain)
    await brain.interrupt(1)
    await brain.ask(2, "next")
    evs = await collect(brain, n_terminal=2, timeout=3)
    await brain.close()
    assert [e.kind for e in evs if e.turn_id == 1 and e.kind != K.TEXT] == [K.DONE]
    assert [e.text for e in evs if e.turn_id == 2 and e.kind == K.TEXT] == ["ok"]
    assert len(MockClient.instances) == 1  # no drain-deadline reconnect was needed


async def test_request_confirmation_without_denial_is_refused(make_brain) -> None:
    gate = ConfirmationGate()
    out: dict[str, Any] = {}

    async def r(c: MockClient, p: str) -> None:
        out["res"] = await brain.tools.request_confirmation({"summary": "Send Bob hi"})
        c.push(result())

    brain = make_brain(r, gate=gate)
    await brain.start()
    await brain.ask(1, "x")
    evs = await collect(brain)
    await brain.close()
    assert out["res"].get("is_error") and "Nothing to confirm" in out["res"]["content"][0]["text"]
    assert not gate.has_token(1)
    assert K.CONFIRM not in [e.kind for e in evs]  # nothing was spoken to the user


async def test_dead_owner_restart_reaps_the_old_client(make_brain) -> None:
    brain = make_brain(simple("hi"))
    await brain.start()
    await brain.warm()
    await settle(brain)
    old = MockClient.instances[0]
    assert brain.alive and not old.disconnected
    brain._task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await brain._task
    await brain.ask(1, "x")  # F2: restarts the owner, which disconnects the orphaned client first
    evs = await collect(brain)
    await brain.close()
    assert evs[-1].kind == K.DONE
    assert old.disconnected and len(MockClient.instances) == 2


async def test_sleep_mid_turn_is_honoured_when_the_turn_ends(make_brain) -> None:
    hold = asyncio.Event()

    async def r(c: MockClient, p: str) -> None:
        await hold.wait()
        c.push(result())

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "x")
    await settle(brain)
    await brain.sleep()
    await settle(brain)
    assert brain.alive  # never killed mid-turn
    hold.set()
    await collect(brain)
    await settle(brain)
    assert not brain.alive  # F3: latched sleep acted on at idle
    await brain.ask(2, "y")  # resumes the parked session
    await collect(brain)
    await brain.close()
    assert MockClient.instances[1].options.resume == "sess-1"


async def test_new_ask_cancels_a_latched_sleep(make_brain) -> None:
    hold = asyncio.Event()

    async def r(c: MockClient, p: str) -> None:
        await hold.wait()
        c.push(result())

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "x")
    await settle(brain)
    await brain.sleep()
    await brain.ask(2, "y")  # supersedes turn 1 and cancels the sleep request
    hold.set()
    await collect(brain, n_terminal=2)
    await settle(brain)
    assert brain.alive
    await brain.close()


def test_persona_language_instruction(tmp_path: Path) -> None:
    from macd.brain import prompts

    cfg = config.Config()
    cfg.tts.lang = "en-us"
    assert "## Language" not in prompts.persona_prompt(cfg, memory_path=tmp_path / "m.md")
    cfg.tts.lang = "hi"
    text = prompts.persona_prompt(cfg, memory_path=tmp_path / "m.md")
    assert "Speak Hindi" in text and "talks to you in English" in text
    cfg.tts.lang = "pt-br"
    assert "Brazilian Portuguese" in prompts.persona_prompt(cfg, memory_path=tmp_path / "m.md")
    assert prompts.language_name("cmn") == "Mandarin Chinese" and prompts.language_name("xx") == ""


async def test_set_name_and_language_apply_on_next_connect(make_brain) -> None:
    brain = make_brain(simple("hi"))
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    assert "# You are Mac" in MockClient.instances[0].options.system_prompt["append"]
    await brain.set_name("Jarvis")
    await settle(brain)
    assert not brain.alive  # parked, not reset
    await brain.set_language("fr-fr")
    await brain.ask(2, "y")
    await collect(brain)
    await brain.close()
    opts = MockClient.instances[1].options
    assert opts.resume == "sess-1"  # context survives the rename
    assert "# You are Jarvis" in opts.system_prompt["append"]
    assert "Speak French" in opts.system_prompt["append"]


async def test_proactive_memory_is_grounded_in_user_text(make_brain, tmp_path: Path) -> None:
    out: dict[str, Any] = {}

    async def r(c: MockClient, p: str) -> None:
        out["user"] = await brain.tools.remember({"fact": "Prefers short answers", "kind": "preference"})
        out["page"] = await brain.tools.remember({"fact": "Lives at 42 Elm Street, phone 555-0199",
                                                  "kind": "about"})
        out["forget"] = await brain.tools.forget({"text": "short answers"})
        c.push(result())

    recent = ["Hey Mac, what's the weather?", "I prefer short answers, by the way."]
    brain = make_brain(r, recent_user_text=lambda: recent)
    await brain.start()
    await brain.ask(1, "and what time is it")
    await collect(brain)
    await brain.close()
    assert not out["user"].get("is_error") and "added" in out["user"]["content"][0]["text"]
    assert out["page"].get("is_error") and "come from the user" in out["page"]["content"][0]["text"]
    assert not out["forget"].get("is_error") and "Removed 1" in out["forget"]["content"][0]["text"]
    text = (tmp_path / "ws" / "memory.md").read_text()
    assert "Elm Street" not in text and "short answers" not in text


async def test_handoff_writes_recent_context_silently(make_brain, tmp_path: Path) -> None:
    async def r(c: MockClient, prompt: str) -> None:
        if prompt.startswith("[Session handoff"):
            # tools other than memory are refused; context lines land in "Recent context"
            denied = await brain.hook({"hook_event_name": "PreToolUse", "tool_name": "Bash",
                                       "tool_input": {"command": "ls"}}, "t", {})
            assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
            for line in ("Was drafting the weekly timetable update", "Decided to send it Monday"):
                await brain.tools.remember({"fact": line, "kind": "context"})
            c.push(delta("(ignored)"), result(session="sess-1"))
        else:
            c.push(delta("ok"), result(session="sess-1"))

    brain = make_brain(r)
    await brain.start()
    assert await brain.handoff() is False  # nothing happened yet: free
    await brain.ask(1, "x")
    evs = await collect(brain)
    n_events = len(brain._out._queue)
    assert await brain.handoff() is True
    assert len(brain._out._queue) == n_events and evs[-1].kind == K.DONE
    assert await brain.handoff() is False  # idempotent
    await brain.close()
    text = (tmp_path / "ws" / "memory.md").read_text()
    assert "## Recent context\n- Was drafting the weekly timetable update\n- Decided to send it Monday" in text
    assert "(ignored)" not in text
    assert MockClient.instances[0].queries[1].startswith(cb.HANDOFF_PROMPT[:20])


async def test_reset_runs_handoff_first_and_resume_prompt_has_context(make_brain, tmp_path: Path) -> None:
    async def r(c: MockClient, prompt: str) -> None:
        if prompt.startswith("[Session handoff"):
            await brain.tools.remember({"fact": "We were comparing laptops", "kind": "context"})
        c.push(result(session="sess-1"))

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    await brain.reset()
    for _ in range(200):  # the handoff runs first; reset completes (CLI parked) when it ends
        await asyncio.sleep(0.01)
        if not brain.alive:
            break
    assert not brain.alive
    await brain.ask(2, "y")
    await collect(brain)
    await brain.close()
    assert len(MockClient.instances) == 2 and MockClient.instances[1].options.resume is None
    assert "We were comparing laptops" in MockClient.instances[1].options.system_prompt["append"]


async def test_handoff_consolidates_and_keeps_explicit_lines(make_brain, tmp_path: Path) -> None:
    mem = (tmp_path / "ws" / "memory.md")
    mem.parent.mkdir(parents=True, exist_ok=True)
    filler = "\n".join(f"- Old project note number {i} with plenty of filler text to grow the file"
                        for i in range(120))
    mem.write_text(f"# Memory\n## About you\n- Name is Nikhil [kept]\n## Projects & work\n{filler}\n")
    assert mem.stat().st_size > 6 * 1024

    async def r(c: MockClient, prompt: str) -> None:
        if prompt.startswith("[Session handoff"):
            assert "too long" in prompt and "[kept]" in prompt
            await brain.tools.rewrite_memory({"text": "# Memory\n## Projects & work\n- Works on Mac\n"})
        c.push(result(session="sess-1"))

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    assert (await brain.tools.rewrite_memory({"text": "x"})).get("is_error")  # outside handoff: refused
    assert await brain.handoff() is True
    await brain.close()
    text = mem.read_text()
    assert "- Name is Nikhil [kept]" in text and "Works on Mac" in text and "number 7" not in text
    assert mem.stat().st_size < 4 * 1024


async def test_idle_shutdown_hands_off_old_sessions(make_brain) -> None:
    clock = {"t": 1000.0}
    wall = {"t": 10**9}
    seen: list[str] = []

    async def r(c: MockClient, prompt: str) -> None:
        seen.append(prompt[:16])
        c.push(result(session="sess-1"))

    brain = make_brain(r, clock=lambda: clock["t"], wall_clock=lambda: wall["t"],
                       brain_cfg={"idle_shutdown_min": 10, "session_idle_rotate_min": 180})
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    wall["t"] += 181 * 60  # the session itself is old now
    clock["t"] += 11 * 60
    await brain._on_tick()
    await settle(brain)
    await brain.close()
    assert seen == ["x", "[Session handoff"] and not brain.alive


async def test_ask_during_handoff_interrupts_it(make_brain) -> None:
    hold = asyncio.Event()

    async def r(c: MockClient, prompt: str) -> None:
        if prompt.startswith("[Session handoff"):
            c.on_interrupt = lambda cl: cl.push(result(terminal_reason="aborted_streaming"))
            await hold.wait()
        else:
            c.push(delta("ok"), result())

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    task = asyncio.create_task(brain.handoff())
    await settle(brain)
    await brain.ask(2, "y")  # wake during handoff: it is cut short, the turn runs
    evs = await collect(brain)
    assert await task is False
    await brain.close()
    assert [e.kind for e in evs] == [K.TEXT, K.DONE] and evs[0].turn_id == 2


CHECK_KEYS = {"auth", "auth_detail", "chrome", "chrome_detail", "cli_version", "session_ok"}


async def test_connect_check_connected(make_brain) -> None:
    seen: list[str] = []

    async def r(c: MockClient, p: str) -> None:
        seen.append(p)
        denied = await brain.hook({"hook_event_name": "PreToolUse", "tool_name": "Bash",
                                   "tool_input": {"command": "ls"}}, "t", {})
        assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"  # probe: no tools
        c.push(init(), delta("OK"), result())

    brain = make_brain(r)
    await brain.start()
    res = await brain.connect_check()
    assert set(res) == CHECK_KEYS
    assert res["auth"] == "ok" and res["chrome"] == "connected" and res["session_ok"] is True
    assert res["cli_version"] == "2.1.319" and res["auth_detail"] == ""
    assert seen == [cb.PROBE_PROMPT] and brain._out.empty()  # nothing spoken
    assert brain.alive  # warmed as a side effect
    await brain.close()


async def test_connect_check_chrome_not_connected(make_brain) -> None:
    def r(c: MockClient, p: str) -> None:
        c.push(init(chrome="failed"), delta("OK"), result())

    brain = make_brain(r)
    await brain.start()
    res = await brain.connect_check()
    await brain.close()
    assert res["auth"] == "ok" and res["chrome"] == "not_connected"
    assert "extension" in res["chrome_detail"] and "failed" in res["chrome_detail"]


async def test_connect_check_api_key_mode(make_brain, monkeypatch) -> None:
    monkeypatch.setattr(config, "get_secret", lambda name: "sk-test-key")
    brain = make_brain(simple("OK"), brain_cfg={"auth": "api_key"})
    await brain.start()
    res = await brain.connect_check()
    await brain.close()
    assert res["auth"] == "api_key" and res["chrome"] == "unavailable_api_key" and res["session_ok"]
    assert "claude.ai" in res["chrome_detail"]
    assert MockClient.instances[0].options.env["ANTHROPIC_API_KEY"] == "sk-test-key"


async def test_connect_check_not_signed_in(make_brain) -> None:
    def r(c: MockClient, p: str) -> None:
        c.push(init(), result(is_error=True, result="Invalid API key · Please run /login"))

    brain = make_brain(r)
    await brain.start()
    res = await brain.connect_check()
    await brain.close()
    assert res["auth"] == "not_signed_in" and res["session_ok"] is False
    assert "sign in" in res["auth_detail"]


async def test_connect_check_cli_missing(make_brain, monkeypatch) -> None:
    monkeypatch.setattr(cb, "default_cli_path", lambda cfg: None)
    brain = make_brain(simple("OK"))
    await brain.start()
    res = await brain.connect_check()
    await brain.close()
    assert res["auth"] == "cli_missing" and res["session_ok"] is False
    assert MockClient.instances == []  # never tried to connect


async def test_connect_check_while_busy_does_not_interrupt(make_brain) -> None:
    hold = asyncio.Event()

    async def r(c: MockClient, p: str) -> None:
        await hold.wait()
        c.push(result())

    brain = make_brain(r)
    await brain.start()
    await brain.ask(1, "x")
    await settle(brain)
    res = await brain.connect_check()
    assert res["auth"] == "error" and "busy" in res["auth_detail"]
    hold.set()
    evs = await collect(brain)
    await brain.close()
    assert evs[-1].kind == K.DONE and MockClient.instances[0].queries == ["x"]


async def test_locked_context_note_on_and_off(make_brain) -> None:
    brain = make_brain(simple("hi"))
    await brain.start()
    await brain.ask(1, "a")
    await collect(brain)
    await brain.set_context(locked=True)
    await brain.ask(2, "b")
    await collect(brain)
    await brain.set_context(locked=False)
    await brain.ask(3, "c")
    await collect(brain)
    await brain.close()
    q = MockClient.instances[0].queries
    assert q[0] == "a"
    assert q[1].startswith(cb.LOCKED_NOTE) and q[1].endswith("b") and "Never try to unlock" in q[1]
    assert q[2] == "c"
