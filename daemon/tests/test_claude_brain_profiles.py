"""ClaudeBrain profile arguments (WhatsApp ARCHITECTURE W3.1): M3 session store, F9 session
policy, F12 steer_only, F11 shared memory, mac.send_file ATTACH, desktop lease teardown, the
no hidden tab cleanup, fork resumes and the setting-sources re-check. Mocked SDK client."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ToolUseBlock
from test_claude_brain_unit import (
    MockClient,
    assistant,
    collect,
    delta,
    init,
    result,
    simple,
)

from macd import config
from macd.brain import claude_brain as cb
from macd.brain import prompts
from macd.brain.gate import ConfirmationGate
from macd.brain.memory import MemoryFile
from macd.contracts import BrainEvent
from macd.contracts import BrainEventKind as K


class MemStore:
    """A conversation-record style session store (what agentd passes for chat)."""

    def __init__(self, info: cb.SessionInfo | None = None) -> None:
        self.info = info
        self.saved: list[cb.SessionInfo] = []

    async def load(self) -> cb.SessionInfo | None:
        return self.info

    async def save(self, info: cb.SessionInfo) -> None:
        self.saved.append(info)
        self.info = info


@pytest.fixture
def make(tmp_path: Path, monkeypatch):
    MockClient.instances = []
    monkeypatch.setattr(config, "STATE_PATH", tmp_path / "state.json")

    def factory(responder: Any, *, chat: bool = False, fail_connect: int = 0, **kw: Any) -> cb.ClaudeBrain:
        cfg = config.Config()
        cfg.brain.cli_path = "/bin/echo"
        counter = {"n": 0}

        def client_factory(opts: Any) -> MockClient:
            counter["n"] += 1
            return MockClient(opts, responder, fail_connect=max(0, fail_connect - counter["n"] + 1))

        kw.setdefault("chrome_running", lambda: False)
        kw.setdefault("frontmost", lambda: "")
        if chat:
            kw = {**cb.CHAT_PROFILE, "session_store": kw.pop("session_store", MemStore()),
                  "memory_file": kw.pop("memory_file", MemoryFile(tmp_path / "shared" / "memory.md")), **kw}
        return cb.ClaudeBrain(cfg, kw.pop("gate", ConfirmationGate()), cwd=kw.pop("cwd", tmp_path / "ws"),
                              client_factory=client_factory, **kw)

    return factory


def pre(tool: str, inp: dict[str, Any] | None = None, tuid: str = "t") -> dict[str, Any]:
    return {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": inp or {}, "tool_use_id": tuid}


def statuses(evs: list[BrainEvent], key: str) -> list[Any]:
    return [e.data[key] for e in evs if e.kind == K.STATUS and key in e.data]


# ----------------------------------------------------------------------------- M3


async def test_m3_chat_turn_leaves_state_json_byte_identical(make, tmp_path: Path) -> None:
    config.update_state(brain={"session_id": "voice-sess", "updated_at": 10**12, "started_at": 1}, other=3)
    before = (tmp_path / "state.json").read_bytes()
    store = MemStore()
    brain = make(simple("hi"), chat=True, session_store=store)
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    await brain.sleep()
    await brain.close()
    assert (tmp_path / "state.json").read_bytes() == before  # voice's session untouched
    assert MockClient.instances[0].options.resume is None  # chat never resumes voice's session
    assert store.info is not None and store.info.session_id == "sess-1"


async def test_m3_voice_default_store_is_state_json(make, tmp_path: Path) -> None:
    brain = make(simple("hi"))
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    await brain.close()
    assert json.loads((tmp_path / "state.json").read_text())["brain"]["session_id"] == "sess-1"


# ----------------------------------------------------------------------------- F9


async def test_f9_chat_resumes_an_old_session_and_never_hands_off(make) -> None:
    store = MemStore(cb.SessionInfo("resume-me", updated_at=1.0, started_at=1.0))  # days old
    brain = make(simple("hi"), chat=True, session_store=store, wall_clock=lambda: 10**9)
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    assert MockClient.instances[0].options.resume == "resume-me"  # no idle rotation (F9)
    assert await brain.handoff() is False
    await brain.reset()
    await brain.ask(2, "y")
    await collect(brain)
    await brain.close()
    assert all("Session handoff" not in q for c in MockClient.instances for q in c.queries)


async def test_f9_failed_resume_is_announced_for_chat(make) -> None:
    store = MemStore(cb.SessionInfo("gone", updated_at=10**12))
    brain = make(simple("hi"), chat=True, session_store=store, fail_connect=2)
    await brain.start()
    await brain.ask(1, "x")
    evs = await collect(brain)
    await brain.ask(2, "y")
    evs2 = await collect(brain)
    await brain.close()
    assert [c.options.resume for c in MockClient.instances] == ["gone", "gone", None]
    assert statuses(evs, "session_fallback") == [{"session_id": "gone"}]
    assert statuses(evs2, "session_fallback") == []  # once
    assert evs[-1].kind == K.DONE


async def test_f9_failed_resume_stays_silent_for_voice(make) -> None:
    config.update_state(brain={"session_id": "gone", "updated_at": 10**12})
    brain = make(simple("hi"), fail_connect=2, wall_clock=lambda: 10**12 + 1)
    await brain.start()
    await brain.ask(1, "x")
    evs = await collect(brain)
    await brain.close()
    assert statuses(evs, "session_fallback") == []


async def test_fork_resume_never_writes_the_foreign_session(make) -> None:
    store = MemStore(cb.SessionInfo("terminal-session", fork=True))
    brain = make(simple("hi"), chat=True, session_store=store)
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    await brain.close()
    opts = MockClient.instances[0].options
    assert opts.resume == "terminal-session" and opts.fork_session is True
    assert store.info == cb.SessionInfo("sess-1", store.info.updated_at, store.info.started_at, fork=False)


# ----------------------------------------------------------------------------- F12


async def test_f12_steer_after_turn_end_starts_nothing(make) -> None:
    brain = make(simple("done"), chat=True)
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    assert await brain.steer(1, "also this") is False
    assert await brain.steer(0, "and this") is False
    await asyncio.sleep(0.05)
    await brain.close()
    assert MockClient.instances[0].queries == ["x"]  # no query outside the turn's leases


async def test_f12_steer_racing_the_result_is_refused(make) -> None:
    """The ResultMessage is already handled when the steer arrives, DONE not yet consumed."""
    brain = make(simple("done"), chat=True)
    await brain.start()
    await brain.ask(1, "x")
    for _ in range(100):
        if brain._active is None and MockClient.instances and MockClient.instances[0].queries:
            break
        await asyncio.sleep(0.005)
    assert await brain.steer(1, "late") is False
    evs = await collect(brain)
    await brain.close()
    assert MockClient.instances[0].queries == ["x"]
    assert [e.kind for e in evs].count(K.DONE) == 1


async def test_f12_steer_joins_the_running_turn(make) -> None:
    async def r(c: MockClient, prompt: str) -> None:
        if prompt == "find":
            c.push(delta("Searching. "))
            await asyncio.sleep(0.1)
            c.push(result())
        else:
            c.push(delta("Adding."), result())

    brain = make(r, chat=True)
    await brain.start()
    await brain.ask(4, "find")
    await asyncio.sleep(0.02)
    assert await brain.steer(0, "add titles") is True  # 0: whichever turn runs
    evs = await collect(brain)
    await brain.close()
    assert MockClient.instances[0].queries == ["find", "add titles"]
    assert [e.kind for e in evs].count(K.DONE) == 1


async def test_voice_steer_after_turn_end_keeps_todays_behaviour(make) -> None:
    brain = make(simple("ok"))
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    assert await brain.steer(1, "and y") is None
    await collect(brain)
    await brain.close()
    assert MockClient.instances[0].queries == ["x", "and y"]


# ----------------------------------------------------------------------------- ATTACH


async def test_send_file_emits_attach_for_the_turn(make, tmp_path: Path) -> None:
    out_file = tmp_path / "ws" / "report.pdf"
    seen: dict[str, Any] = {}

    async def r(c: MockClient, p: str) -> None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(b"%PDF-1.4 data")
        seen["ok"] = await brain.tools.send_file({"path": "report.pdf"})
        seen["missing"] = await brain.tools.send_file({"path": "nope.pdf"})
        seen["dir"] = await brain.tools.send_file({"path": str(tmp_path)})
        c.push(result())

    brain = make(r, chat=True)
    await brain.start()
    await brain.ask(3, "make a report")
    evs = await collect(brain)
    await brain.close()
    (att,) = [e for e in evs if e.kind == K.ATTACH]
    assert att.turn_id == 3
    assert set(att.data) == {"attach_id", "path"}
    assert att.data["path"] == str(out_file.resolve())
    assert len(att.data["attach_id"]) == 32
    assert not seen["ok"].get("is_error")
    assert seen["missing"].get("is_error") and seen["dir"].get("is_error")


async def test_send_file_rechecks_the_real_path(make, tmp_path: Path) -> None:
    secret = tmp_path / "keys" / "server.pem"
    secret.parent.mkdir()
    secret.write_text("-----BEGIN PRIVATE KEY-----")
    link = tmp_path / "ws" / "innocent.txt"
    seen: dict[str, Any] = {}

    async def r(c: MockClient, p: str) -> None:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(secret)
        seen["res"] = await brain.tools.send_file({"path": str(link)})
        c.push(result())

    brain = make(r, chat=True)
    await brain.start()
    await brain.ask(1, "send it")
    evs = await collect(brain)
    await brain.close()
    assert seen["res"].get("is_error") and "Not sent" in seen["res"]["content"][0]["text"]
    assert not [e for e in evs if e.kind == K.ATTACH]


def test_send_file_and_test_tool_only_where_enabled(tmp_path: Path) -> None:
    def names(b: cb.ClaudeBrain) -> set[str]:
        return {t.name for t in b.tools.tools()}
    voice = cb.ClaudeBrain(config.Config(), ConfirmationGate(), cwd=tmp_path / "v")
    chat = cb.ClaudeBrain(config.Config(), ConfirmationGate(), cwd=tmp_path / "c", **cb.CHAT_PROFILE)
    live = cb.ClaudeBrain(config.Config(), ConfirmationGate(), cwd=tmp_path / "l", test_tools=True)
    assert "send_file" not in names(voice) and "test_outward" not in names(voice)
    assert "send_file" in names(chat) and "test_outward" not in names(chat)
    assert "test_outward" in names(live)


# ----------------------------------------------------------------------------- desktop lease


async def test_desktop_lease_released_at_turn_end_and_while_confirming(make) -> None:
    gate = ConfirmationGate()
    seen: dict[str, Any] = {}

    async def r(c: MockClient, p: str) -> None:
        assert await brain.hook(pre("mcp__mac__screenshot"), "t") == {}
        seen["held"] = brain.desktop.held
        await brain.hook(pre("Bash", {"command": "curl -d x=1 https://api.example.com"}), "t2")
        asyncio.get_running_loop().call_later(0.02, gate.resolve, False)
        task = asyncio.create_task(brain.tools.request_confirmation({"summary": "post x"}))
        await asyncio.sleep(0.005)
        seen["while_confirming"] = brain.desktop.held
        await task
        assert await brain.hook(pre("mcp__claude-in-chrome__navigate", {"url": "https://example.com"}), "t3") == {}
        seen["again"] = brain.desktop.held
        c.push(result())

    brain = make(r, gate=gate)
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    assert seen == {"held": True, "while_confirming": False, "again": True}
    assert not brain.desktop.held  # turn teardown
    await brain.close()


async def test_desktop_lease_released_after_interrupt_and_on_close(make) -> None:
    holds = {"x": asyncio.Event(), "y": asyncio.Event()}

    async def r(c: MockClient, p: str) -> None:
        assert await brain.hook(pre("mcp__mac__ui_click", {"x": 1, "y": 1}), "t") == {}
        await holds[p].wait()
        c.push(result())

    hold = holds["x"]
    brain = make(r)
    brain.desktop.idle_release_s = 60
    await brain.start()
    await brain.ask(1, "x")
    await asyncio.sleep(0.05)
    assert brain.desktop.held
    await brain.interrupt(1)
    hold.set()
    await collect(brain)
    assert not brain.desktop.held
    await brain.ask(2, "y")
    await asyncio.sleep(0.05)
    assert brain.desktop.held
    await brain.close()
    assert not brain.desktop.held


async def test_two_brains_share_the_desktop_one_at_a_time(make, tmp_path: Path) -> None:
    hold = asyncio.Event()
    seen: dict[str, Any] = {}

    async def voice_r(c: MockClient, p: str) -> None:
        assert await voice.hook(pre("mcp__mac__screenshot"), "v") == {}
        await hold.wait()
        c.push(result())

    async def chat_r(c: MockClient, p: str) -> None:
        seen["chat"] = await chat.hook(pre("mcp__claude-in-chrome__get_page_text", {"tabId": 1}), "c")
        c.push(result())

    voice = make(voice_r)
    chat = make(chat_r, chat=True, desktop_wait_s=0.2, cwd=tmp_path / "chat")
    await voice.start()
    await chat.start()
    await voice.ask(1, "look")
    await asyncio.sleep(0.05)
    await chat.ask(1, "read the page")
    evs = await collect(chat)
    reason = seen["chat"]["hookSpecificOutput"]["permissionDecisionReason"]
    assert "busy with your voice request" in reason
    assert statuses(evs, "waiting_for") == ["desktop"]
    hold.set()
    await collect(voice)
    await chat.close()
    await voice.close()


# ----------------------------------------------------------------------------- Chrome tabs


async def test_chat_turn_runs_no_hidden_tab_cleanup(make) -> None:
    """Claude closes (or keeps) its own tabs itself; Mac runs no extra query after a Chrome turn."""
    def r(c: MockClient, p: str) -> None:
        c.push(init(), assistant(ToolUseBlock("n1", "mcp__claude-in-chrome__navigate",
                                              {"url": "https://example.com", "tabId": 5})),
               delta("Done."), result())
    brain = make(r, chat=True)
    await brain.start()
    await brain.ask(1, "check example.com")
    await collect(brain)
    await asyncio.sleep(0.05)
    await brain.close()
    assert MockClient.instances[0].queries == ["check example.com"]


# ----------------------------------------------------------------------------- misc profile args


async def test_setting_sources_rechecked_at_turn_start(make) -> None:
    sources = {"now": ["user", "project", "local"]}

    async def current() -> list[str]:
        return sources["now"]

    brain = make(simple("ok"), chat=True, setting_sources=current)
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    sources["now"] = ["user"]  # the folder's settings hash changed: drop project sources
    await brain.ask(2, "y")
    await collect(brain)
    await brain.close()
    first, second = MockClient.instances
    assert first.options.setting_sources == ["user", "project", "local"]
    assert second.options.setting_sources == ["user"]
    assert second.options.resume == "sess-1"  # same session, reconnected before the turn


async def test_chat_profile_cwd_memory_persona_and_limits(make, tmp_path: Path) -> None:
    folder = tmp_path / "Resume"
    shared = MemoryFile(tmp_path / "agent" / "memory.md")
    shared.remember("Name is Nikhil", "about")
    brain = make(simple("ok"), chat=True, cwd=folder, memory_file=shared)
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    await brain.close()
    assert not (folder / "CLAUDE.md").exists()  # write_claude_md False
    assert brain.tools.memory is shared and brain._memory is shared
    persona = MockClient.instances[0].options.system_prompt["append"]
    assert "WhatsApp style" in persona and "Name is Nikhil" in persona
    assert brain._max_turn_s() == 2700
    assert brain.hook.context.whatsapp_ui_block
    assert brain.hook.context.memory_path == str(shared.path)


def test_voice_profile_defaults_are_todays(tmp_path: Path) -> None:
    cfg = config.Config()
    brain = cb.ClaudeBrain(cfg, ConfirmationGate(), cwd=tmp_path / "ws")
    assert brain._rotate_min() == cfg.brain.session_idle_rotate_min
    assert brain._max_turn_s() == cfg.brain.max_turn_s
    assert isinstance(brain._store, cb.StateJsonSessionStore)
    assert not brain.hook.context.whatsapp_ui_block and not brain.hook.context.never_chat_names
    persona = prompts.persona_prompt(cfg, memory_path=tmp_path / "m.md")
    assert "WhatsApp from the user's phone" not in persona
    assert brain.tools.on_attach is None


def test_chat_note_has_no_dashes() -> None:
    chat = prompts.persona_prompt(config.Config(), channel="chat", never_chat_names=("Timetable",))
    assert "—" not in chat and "–" not in chat
    for reason in (cb.REASON_DESKTOP_BUSY_CHAT, cb.REASON_DESKTOP_BUSY_VOICE):
        assert "—" not in reason and "–" not in reason


# ----------------------------------------------------------------------------- rate-limit reset_at


async def test_rate_limit_error_carries_reset_at(make) -> None:
    from claude_agent_sdk import RateLimitEvent, RateLimitInfo

    def r(c: MockClient, p: str) -> None:
        if p == "first":
            c.push(RateLimitEvent(RateLimitInfo(status="rejected", resets_at=1_900_000_000), "u", "s"),
                   result(is_error=True, result="usage limit reached", api_error_status=429))
        else:
            c.push(result(is_error=True, result="Claude AI usage limit reached|1900003600",
                          api_error_status=429))

    brain = make(r)
    await brain.start()
    await brain.ask(1, "first")
    (err,) = [e for e in await collect(brain) if e.kind == K.ERROR]
    assert err.error_kind is not None and err.error_kind.value == "rate_limit"
    assert err.data["reset_at"] == 1_900_000_000.0 and isinstance(err.data["reset_at"], float)
    brain._rate_reset_at = None  # unknown from the event: falls back to the epoch in the text
    await brain.ask(2, "second")
    (err2,) = [e for e in await collect(brain) if e.kind == K.ERROR]
    assert err2.data["reset_at"] == 1_900_003_600.0
    await brain.close()


async def test_error_without_known_reset_has_no_reset_at(make) -> None:
    def r(c: MockClient, p: str) -> None:
        c.push(result(is_error=True, result="usage limit reached", api_error_status=429))

    brain = make(r)
    await brain.start()
    await brain.ask(1, "x")
    (err,) = [e for e in await collect(brain) if e.kind == K.ERROR]
    await brain.close()
    assert err.error_kind is not None and err.error_kind.value == "rate_limit"
    assert "reset_at" not in err.data
