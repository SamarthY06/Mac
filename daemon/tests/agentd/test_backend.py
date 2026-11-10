"""Records and atomic stores, the shared breaker, voice session migration (M3), the session store,
the brain factory seam and loop lag (V7)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import time

from agentd_kit import CHAT, CHAT_TOKEN, OTHER_CHAT, make_agent, stop_agent

from agentd import brains
from agentd.breaker import PlanBreaker
from agentd.records import ConversationRecord, OpenQuestion, RunningTurn
from agentd.stores import JsonStore, atomic_write_json, read_json
from agentd.stream import LagMonitor
from agentd.workspaces import settings_hash
from macd import config
from macd.brain import fake_brain as fb
from macd.contracts import BrainErrorKind

V = "voice:local"


def test_atomic_write_is_private_and_leaves_no_temp(tmp_path):
    p = tmp_path / "state" / "x.json"
    atomic_write_json(p, {"a": 1})
    atomic_write_json(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 2}
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    assert os.listdir(p.parent) == ["x.json"]


def test_corrupt_store_is_kept_aside(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{nope")
    assert read_json(p, {}) == {}
    assert (tmp_path / "x.json.corrupt").exists() and not p.exists()


async def test_concurrent_saves_keep_the_latest(tmp_path):
    store = JsonStore(tmp_path / "x.json")
    state = {"n": 0}

    async def bump() -> None:
        state["n"] += 1
        await store.save(lambda: dict(state))

    await asyncio.gather(*(bump() for _ in range(20)))
    assert json.loads((tmp_path / "x.json").read_text()) == {"n": 20}


def test_record_round_trip_ignores_unknown_fields():
    r = ConversationRecord(conv=CHAT, folder="/x", running=RunningTurn("u", "t", 1.0, "m"),
                           question=OpenQuestion("q", "route", "?", 5.0, "p", {"options": []}))
    d = r.to_dict() | {"future_field": 1}
    assert ConversationRecord.from_dict(d) == r
    for i in range(30):
        r.add_unacked({"turn_uid": str(i)})
    assert len(r.unacked) == 20 and r.unacked[0]["turn_uid"] == "10"


def test_breaker_trip_message_and_close():
    now = [1000.0]
    b = PlanBreaker(clock=lambda: now[0])
    assert not b.is_open()
    b.trip(None)
    assert b.is_open() and b.reset_at == 1300.0
    b.trip(now[0] + 10 ** 9)  # an absurd reset time is capped
    assert b.reset_at <= now[0] + 6 * 3600
    assert b.message().startswith("I've hit the Claude usage limit. It resets at ")
    now[0] = b.reset_at + 1
    assert not b.is_open()


async def test_one_breaker_for_every_conversation(short_root):
    a = await make_agent(short_root)
    a.scripts[CHAT] = lambda t: [fb.error(BrainErrorKind.RATE_LIMIT, "limit")]
    try:
        c = await a.client(CHAT_TOKEN, [CHAT, OTHER_CHAT])
        await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "x"})
        assert (await c.event("error"))["kind"] == "rate_limit"
        await c.call({"op": "ask", "conv": OTHER_CHAT, "turn": 1, "text": "y"})
        err = await c.event("error", pred=lambda e: e["conv"] == OTHER_CHAT)
        assert err["kind"] == "rate_limit" and "resets at" in err["message"]
        assert OTHER_CHAT not in a.brains  # no claude started while the breaker is open
        v = await a.client()
        await v.call({"op": "ask", "conv": V, "turn": 1, "text": "hi"})
        assert (await v.event("error"))["kind"] == "rate_limit"
    finally:
        await stop_agent(a)


async def test_voice_session_copied_once_from_state_json(agent):
    config.update_state(brain={"session_id": "voice-s1", "updated_at": 50.0, "started_at": 40.0})
    before = config.STATE_PATH.read_bytes()
    await agent.client()
    r = agent.svc.convs[V].record
    assert (r.session_id, r.voice_migrated, r.session_fork) == ("voice-s1", True, False)
    config.update_state(brain={"session_id": "voice-s2"})
    await agent.client(epoch="e2")
    assert agent.svc.convs[V].record.session_id == "voice-s1"  # one-time copy
    assert config.read_state()["brain"]["session_id"] == "voice-s2"  # state.json never written by agentd
    assert before != config.STATE_PATH.read_bytes()


async def test_session_store_fork_for_foreign_sessions(agent):
    conv = agent.svc.conversation(CHAT)
    store = brains.RecordSessionStore(conv._session_get, conv._session_put)
    assert await store.load() is None
    conv.record.session_id, conv.record.session_fork = "terminal-1", True
    info = await store.load()
    assert info.session_id == "terminal-1" and info.fork is True
    await store.save(brains.SessionInfo("mac-2", 10.0, 5.0))
    assert conv.record.session_id == "mac-2" and not conv.record.session_fork
    assert conv.record.own_sessions == ["mac-2"]


def test_claude_factory_args_exist_in_claude_brain():
    """Builder B's ClaudeBrain takes every profile argument agentd passes (fails closed if not)."""
    import inspect

    from macd.brain.claude_brain import ClaudeBrain

    params = inspect.signature(ClaudeBrain.__init__).parameters
    assert [a for a in brains.CLAUDE_ARGS if a not in params] == []
    brains.claude_factory(config.Config())


async def test_loop_lag_stays_low_while_hashing_and_writing(short_root):
    a = await make_agent(short_root, lag_interval_s=0.005)
    try:
        proj = a.paths.home / "Big"
        skills = proj / ".claude" / "skills"
        skills.mkdir(parents=True)
        for i in range(200):  # 2 MB, as in spike P0.0
            (skills / f"f{i:03}.md").write_bytes(os.urandom(5120).hex().encode())
        t0 = time.perf_counter()
        settings_hash(proj, a.paths.home)
        on_loop_ms = (time.perf_counter() - t0) * 1000
        await asyncio.sleep(0.05)
        a.svc.lag.reset()
        for _ in range(5):
            await a.svc.workspaces.trust(proj)  # hash in a thread
            await a.svc.persist()  # record write with fsync in a thread
        await asyncio.sleep(0.05)
        assert a.svc.lag.max_ms < max(10.0, on_loop_ms / 2), (a.svc.lag.max_ms, on_loop_ms)
        assert a.svc.lag.p99() < 10.0
        assert a.svc.conversation(V).status().loop_lag_ms_p99 == a.svc.lag.p99()
    finally:
        await stop_agent(a)


async def test_lag_monitor_measures_a_blocked_loop():
    m = LagMonitor(0.005)
    task = asyncio.create_task(m.run())
    await asyncio.sleep(0.02)
    time.sleep(0.05)  # noqa: ASYNC251 - deliberately block the loop
    await asyncio.sleep(0.02)
    task.cancel()
    assert m.max_ms >= 30
    assert hashlib  # used above via settings_hash


def test_lifeline_tokens_and_dev_tokens(tmp_path):
    import pytest

    from agentd import __main__ as m
    from agentd import schema as s

    toks = m.parse_tokens(json.dumps({"tokens": {"a" * 64: "voice", "b" * 64: "chat"}}))
    assert toks == {"a" * 64: s.ClientKind.VOICE, "b" * 64: s.ClientKind.CHAT}
    for bad in ('{"tokens": {"short": "voice"}}', '{"tokens": {"' + "c" * 64 + '": "test"}}', "{}", "[]"):
        with pytest.raises(ValueError):
            m.parse_tokens(bad)
    dev = m.dev_tokens(tmp_path / "run")
    f = tmp_path / "run" / "dev-tokens.json"
    assert stat.S_IMODE(os.stat(f).st_mode) == 0o600
    assert {v.value for v in dev.values()} == {"voice", "chat", "app"}
    assert set(json.loads(f.read_text()).values()) == set(dev)
    fd = m.take_pid_lock(tmp_path / "run" / "agent.pid.lock")
    assert fd is not None and m.take_pid_lock(tmp_path / "run" / "agent.pid.lock") is None
    os.close(fd)
