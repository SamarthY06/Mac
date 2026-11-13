"""Scheduled tasks: the usage limit (DESIGN 8), continue and stop, restart recovery (5.3), the
durability must-fixes (a, b) and unacked eviction (F12)."""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest
from agentd_kit import APP_TOKEN, CHAT, CHAT_TOKEN, Agent, make_agent, slow
from tasks_kit import (
    IN_CHAT,
    VOICE,
    Clock,
    agent,
    app_notices,
    check,
    convs_json,
    create,
    finished,
    history,
    report,
    runs_file,
    settle,
    stop,
    tasks_json,
    until,
)

from agentd import schema as s
from agentd import tasks as tk
from agentd.service import Conversation
from macd.brain import fake_brain as fb
from macd.contracts import BrainErrorKind


def limit_error(reset_in: float, clock: Clock) -> list[fb.Step]:
    """A rate-limit ERROR like ClaudeBrain's, with `reset_at` when the API said when."""
    step = fb.error(BrainErrorKind.RATE_LIMIT, "limit", delay=0.01)
    if reset_in:
        step.data = {"reset_at": clock.now + reset_in}
    return [step]


def write_self_chat(a: Agent, chat_id: str) -> None:
    p = a.paths.mac_home / "chat" / "chats.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([{"id": "x@g.us", "subject": "Other", "is_self": False},
                             {"id": chat_id, "subject": "Me", "is_self": True}]))


# ----------------------------------------------------------------------------- the limit in a bound group


async def test_rate_limit_in_a_bound_run_is_one_message_and_waiting_limit(env):
    a, clock = env
    a.scripts[CHAT] = lambda text: limit_error(3600, clock)
    task = await create(a, IN_CHAT, schedule={"in_min": 1})
    c = await a.client(CHAT_TOKEN, [CHAT])
    clock.advance(61)
    await settle(a)
    await finished(a, task)
    assert task.state == "waiting_limit" and task.current.status == "limit" and task.current.reset_at
    evs = [e for e in c.drain() if e["ev"] in ("text", "done", "error")]
    assert [e["ev"] for e in evs] == ["text", "done"]  # exactly one message, no error event
    msg = evs[0]["text"]
    assert msg == a.tasks.limit_message(task, CHAT) and msg.startswith('"Job search" stopped at the Claude usage limit')
    assert "it resets at 11:01 AM. Reply " in msg and 'continue Job search" and I will pick it up then, or "' in msg
    assert msg.endswith("Continue in Mac Settings > Tasks works too.") and chr(0x2014) not in msg
    r = a.svc.convs[CHAT].record
    assert [u["answer"] for u in r.unacked] == [msg]
    assert a.svc.breaker.is_open() and task.rev == tasks_json(a)["tasks"][task.id]["rev"]
    assert runs_file(a, task.id)[0]["status"] == "limit" and runs_file(a, task.id)[0]["delivery"] == "posted"
    await c.close()


async def test_breaker_open_at_turn_start_ends_the_run_as_limit_through_the_first_check(env):
    a, clock = env
    task = await create(a, IN_CHAT, schedule={"in_min": 1})
    clock.advance(61)
    a.tasks.tick()  # dispatched: _run has not executed yet
    a.svc.breaker.trip(clock.now + 1800)
    await finished(a, task)
    assert task.state == "waiting_limit" and task.current.reset_at == clock.now + 1800
    assert CHAT not in a.brains  # the brain was never asked
    assert [u["answer"] for u in a.svc.convs[CHAT].record.unacked] == [a.tasks.limit_message(task, CHAT)]


async def test_mark_limit_is_on_disk_before_finish_so_a_crash_recovers_as_waiting_limit(short_root):
    clock = Clock()
    a = await agent(short_root, clock)
    a.scripts[CHAT] = lambda text: limit_error(3600, clock)
    task = await create(a, IN_CHAT, schedule={"in_min": 1})
    seen: list[str] = []
    orig = Conversation.finish

    async def crash(self, t, **kw):  # the process dies between mark_limit's save and finish
        if t.run_id:
            seen.append(tasks_json(a)["tasks"][task.id]["current"]["status"])
            raise RuntimeError("crash")
        await orig(self, t, **kw)

    Conversation.finish = crash  # type: ignore[method-assign]
    try:
        clock.advance(61)
        await settle(a)
        await until(lambda: seen)
    finally:
        Conversation.finish = orig  # type: ignore[method-assign]
    assert seen == ["limit"]
    await stop(a)
    b = await agent(short_root, clock)
    try:
        t2 = b.tasks.tasks[task.id]
        assert t2.state == "waiting_limit" and t2.current.status == "limit" and t2.runs == 1
        assert b.svc.convs[CHAT].record.running is None and b.svc.convs[CHAT].question is None
        check(b)
    finally:
        await stop(b)


# ----------------------------------------------------------------------------- unbound: owner chat and notice


async def test_unbound_limit_goes_to_the_owner_chat_when_enabled_else_notice_only(env):
    a, clock = env
    app = await a.client(APP_TOKEN)
    a.scripts["*"] = lambda text: limit_error(3600, clock)
    task = await create(a, VOICE, schedule={"in_min": 1})
    clock.advance(61)
    await settle(a)
    await finished(a, task)
    assert task.state == "waiting_limit" and task.limit_conv is None
    await until(lambda: app_notices(app, "task_report"))
    n = app_notices(app, "task_report")[0]
    assert n.status == "limit" and n.body == a.tasks.limit_message(task, None) and n.kind == "task"
    assert n.body.startswith("Stopped at the Claude usage limit; it resets at ") and "Say \"Hey Mac, continue Job search\", or use Continue in Mac Settings > Tasks." in n.body
    assert a.svc.convs[task.conv].record.unacked == []
    # with the self chat enabled: a canned turn there too, and that chat may continue or stop the task
    a.svc.breaker.close()
    write_self_chat(a, CHAT.partition(":")[2])
    r = await a.tasks.call(VOICE, "task_continue", {"task_id": task.id})
    assert r["starts_at"] == task.not_before >= clock.now
    clock.advance(r["starts_at"] - clock.now + 1)
    await settle(a)
    await finished(a, task)
    assert task.state == "waiting_limit" and task.limit_conv == CHAT
    await until(lambda: a.svc.convs[CHAT].record.unacked)
    (u,) = a.svc.convs[CHAT].record.unacked
    assert u["answer"] == a.tasks.limit_message(task, CHAT) and tk.is_task_uid(u["turn_uid"])
    listed = await a.tasks.call(IN_CHAT, "task_list", {})
    assert [t["id"] for t in listed["tasks"]] == [task.id]
    with pytest.raises(tk.TaskApiError) as e:
        await a.tasks.call(IN_CHAT, "task_update", {"task_id": task.id, "title": "x"})
    assert e.value.code == "task_not_found"  # the owner chat may continue and stop, not edit
    r = await a.tasks.call(IN_CHAT, "task_stop", {"task_id": task.id})
    assert r["stopped"] is True and task.state == "done"
    await app.close()


# ----------------------------------------------------------------------------- continue and stop


async def test_continue_resumes_the_same_session_and_is_idempotent(env):
    a, clock = env
    mode = ["ok"]
    a.scripts[CHAT] = lambda text: limit_error(0, clock) if mode[0] == "limit" else report("continued", session="s1")
    task = await create(a, IN_CHAT, schedule={"every_min": 15})
    c = await a.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "hello"})
    await c.event("done")
    mode[0] = "limit"
    clock.advance(901)
    await settle(a)
    await finished(a, task)
    assert task.state == "waiting_limit"
    reset_at = task.current.reset_at
    assert reset_at == a.svc.breaker.reset_at > clock.now
    mode[0] = "ok"
    r = await a.tasks.call(IN_CHAT, "task_continue", {})  # task_id omitted: the bound task
    assert r["starts_at"] == reset_at and r["task"]["state"] == "queued" and task.not_before == reset_at
    again = await a.tasks.call(VOICE, "task_continue", {"task_id": task.id})
    assert again["starts_at"] == reset_at and task.runs == 2  # repeated: the same run
    await settle(a, ticks=2)
    assert task.state == "queued"  # before the reset time
    with pytest.raises(tk.TaskApiError) as e:
        await a.tasks.call(VOICE, "task_run", {"task_id": task.id})
    assert e.value.code == "busy"
    clock.advance(reset_at - clock.now + 1)
    await settle(a)
    await finished(a, task)
    assert task.state == "active" and task.last_run["status"] == "ok" and task.runs == 2
    assert len(a.brains[CHAT]) == 1 and a.svc.convs[CHAT].record.session_id == "s1"
    prompt = a.brain(CHAT).asks[-1][1]
    assert "You stopped at the Claude usage limit. Continue where you stopped" in prompt
    assert [x["status"] for x in runs_file(a, task.id)] == ["limit", "ok"]
    with pytest.raises(tk.TaskApiError, match="not waiting"):
        await a.tasks.call(VOICE, "task_continue", {"task_id": task.id})
    await c.close()


async def test_stop_while_waiting_and_a_later_slot_supersedes(env):
    a, clock = env
    a.scripts["*"] = lambda text: limit_error(7200, clock)
    task = await create(a, VOICE, schedule={"every_min": 15})
    clock.advance(901)
    await settle(a)
    await finished(a, task)
    assert task.state == "waiting_limit"
    with pytest.raises(tk.TaskApiError, match="Continue it or stop it"):
        await a.tasks.call(VOICE, "task_run", {"task_id": task.id})
    r = await a.tasks.call(VOICE, "task_stop", {"task_id": task.id})
    assert r["stopped"] is True and task.state == "active" and task.current is None
    assert [x["status"] for x in await history(a, task.id)] == ["stopped"] and task.last_run["status"] == "stopped"
    # again, then a slot after the reset time supersedes the waiting run
    a.svc.breaker.close()
    clock.advance(900)
    await settle(a)
    await finished(a, task)
    assert task.state == "waiting_limit" and task.runs == 2
    clock.advance(900)  # still before the reset: the slot is skipped
    await settle(a)
    assert task.state == "waiting_limit" and [x["status"] for x in await history(a, task.id)][-1] == "skipped"
    a.svc.breaker.close()
    clock.advance(a.tasks.tasks[task.id].current.reset_at - clock.now + 901)
    await settle(a)
    await until(lambda: task.runs == 3)
    check(a)
    await finished(a, task)
    rows = runs_file(a, task.id)
    assert [x["status"] for x in rows if x["run_id"] == "r2"] == ["stopped"]
    assert "superseded" in next(x for x in rows if x["run_id"] == "r2")["note"]


# ----------------------------------------------------------------------------- restart recovery


async def test_restart_mid_run_resumes_exactly_once(short_root):
    clock = Clock()
    a = await agent(short_root, clock)
    a.scripts["*"] = lambda text: slow("w", n=500, delay=0.01)
    task = await create(a, VOICE, schedule={"every_min": 15})
    conv = task.conv
    clock.advance(901)
    await settle(a)
    await until(lambda: conv in a.brains and a.brain(conv).asks)
    uid = task.current.turn_uid
    await stop(a)  # the run stays `running` on disk, the record keeps `running`
    assert tasks_json(a)["tasks"][task.id]["current"]["status"] == "running"
    assert convs_json(a)[conv]["running"]["uid"] == uid
    b = await make_agent(short_root, clock=clock, tasks=tk.TaskSettings(max_concurrent=0))  # nothing dispatches
    try:
        t2 = b.tasks.tasks[task.id]
        check(b)
        assert t2.state == "queued" and t2.current.trigger == "resume" and t2.runs == 2
        rows = runs_file(b, task.id)
        assert [r["status"] for r in rows] == ["interrupted"] and rows[0]["turn_uid"] == uid
        assert b.svc.convs[conv].record.running is None and b.svc.take_notices([conv]) == []
        assert b.svc.convs[conv].question is None and conv not in b.brains  # no notice, no question, no owner resume
        await b.tasks.start()  # idempotent
        assert t2.runs == 2
    finally:
        await stop(b)
    c = await agent(short_root, clock, tasks=tk.TaskSettings(max_concurrent=0))
    try:
        t3 = c.tasks.tasks[task.id]
        check(c)
        assert t3.state == "queued" and t3.current.trigger == "resume" and t3.runs == 2  # still one resume
        assert [r["status"] for r in runs_file(c, task.id)] == ["interrupted"]
    finally:
        await stop(c)
    d = await agent(short_root, clock)
    d.scripts["*"] = lambda text: report("resumed")
    try:
        t4 = d.tasks.tasks[task.id]
        await finished(d, t4)
        assert t4.last_run["status"] == "ok" and "picking it up where it left off" in d.brain(conv).asks[0][1]
        assert [r["status"] for r in runs_file(d, task.id)] == ["interrupted", "ok"]
    finally:
        await stop(d)


async def test_restart_after_the_turn_finished_finalizes_from_unacked_and_never_resumes(short_root):
    clock = Clock()
    a = await agent(short_root, clock)
    a.scripts[CHAT] = lambda text: report("the report")
    task = await create(a, IN_CHAT, schedule={"every_min": 15})
    hang = asyncio.Event()

    async def never_finalize(*args, **kw):  # the crash sits between `finish` and `_finalize`
        await hang.wait()

    a.tasks._finalize = never_finalize  # type: ignore[method-assign]
    clock.advance(901)
    await settle(a)
    await until(lambda: a.svc.convs[CHAT].record.unacked)
    uid = a.svc.convs[CHAT].record.unacked[0]["turn_uid"]
    assert task.current.status == "running" and task.current.turn_uid == uid
    await stop(a)
    b = await agent(short_root, clock)
    try:
        t2 = b.tasks.tasks[task.id]
        check(b)
        assert t2.state == "active" and t2.current is None and t2.runs == 1
        assert t2.last_run["status"] == "ok" and t2.last_run["summary"] == "the report"
        rows = runs_file(b, task.id)
        assert rows[0]["report"] == "the report" and rows[0]["delivery"] == "posted" and rows[0]["turn_uid"] == uid
        assert [u["turn_uid"] for u in b.svc.convs[CHAT].record.unacked] == [uid]  # chatd still posts it
        assert CHAT not in b.brains
    finally:
        await stop(b)


async def test_restart_sweeps_stale_task_conv_mail_and_a_final_current_reenters_finalize(short_root):
    clock = Clock()
    a = await agent(short_root, clock)
    task = await create(a, VOICE, schedule={"in_min": 1})
    clock.advance(61)
    await settle(a)
    await finished(a, task)
    await stop(a)
    state = tasks_json(a)
    rec = state["tasks"][task.id]
    run = json.loads((a.tasks.task_dir(task.id) / "runs.jsonl").read_text().splitlines()[0])
    rec["current"], rec["state"] = {**run, "delivery": "pending"}, "running"  # crashed in step 2
    (a.paths.state_dir / "tasks.json").write_text(json.dumps(state))
    convs = json.loads((a.paths.state_dir / "conversations.json").read_text())
    convs["conversations"][task.conv]["unacked"] = [{"turn_uid": "old:task:x:task:1", "answer": "stale", "attaches": []}]
    (a.paths.state_dir / "conversations.json").write_text(json.dumps(convs))
    b = await agent(short_root, clock)
    app = await b.client(APP_TOKEN)
    try:
        t2 = b.tasks.tasks[task.id]
        check(b)
        assert t2.state == "done" and t2.current is None and t2.runs == 1
        assert [r["run_id"] for r in runs_file(b, task.id)] == ["r1"]  # not appended twice
        assert b.svc.convs[task.conv].record.unacked == []
        assert tasks_json(b)["tasks"][task.id]["current"] is None
        await history(b, task.id)  # every helper of the recovery has landed
        assert app_notices(app, "task_report") == []  # the notice was delivered (notified) before, not again
    finally:
        await app.close()
        await stop(b)


async def test_start_twice_on_the_same_files_changes_nothing(short_root):
    clock = Clock()
    a = await agent(short_root, clock)
    await create(a, VOICE, schedule={"at": ["08:00"]})
    await create(a, IN_CHAT, title="B", schedule={"every_min": 30})
    await stop(a)
    before = tasks_json(a)
    b = await agent(short_root, clock)
    await stop(b)
    assert tasks_json(b) == before


# ----------------------------------------------------------------------------- durability (must-fix a)


async def test_a_crash_right_after_finish_persisted_never_loses_a_bound_report(short_root):
    clock = Clock()
    a = await agent(short_root, clock)
    a.scripts[CHAT] = lambda text: [*slow("w", n=20, delay=0.01), *report("kept report")]
    task = await create(a, IN_CHAT, schedule={"in_min": 1})
    armed = [False]
    orig = a.svc._store.save

    async def save_then_die(snapshot):
        await orig(snapshot)
        if armed[0] and any(f.function == "finish" for f in inspect.stack()):
            armed[0] = False
            raise RuntimeError("the loop stopped right after the write")

    a.svc._store.save = save_then_die  # type: ignore[method-assign]
    clock.advance(61)
    await settle(a)
    await until(lambda: CHAT in a.brains and a.brain(CHAT).asks)
    armed[0] = True
    await until(lambda: convs_json(a)[CHAT]["unacked"])  # finish wrote the entry, then died
    assert task.current.status == "running" and not any(x.done.is_set() for x in [a.svc.convs[CHAT].last_finished] if x)
    await stop(a)
    b = await agent(short_root, clock)
    try:
        t2 = b.tasks.tasks[task.id]
        check(b)
        assert t2.state == "done" and t2.last_run["status"] == "ok" and t2.last_run["summary"] == "kept report"
        assert runs_file(b, task.id)[0]["delivery"] == "posted"
        assert [u["answer"] for u in b.svc.convs[CHAT].record.unacked] == ["kept report"]
    finally:
        await stop(b)


async def test_finish_persists_before_waiters_wake(env):
    a, _ = env
    a.scripts[CHAT] = lambda text: report("answer")
    c = await a.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "hello"})
    t = a.svc.convs[CHAT].running()
    assert t is not None
    await t.done.wait()
    assert [u["turn_uid"] for u in convs_json(a)[CHAT]["unacked"]] == [t.uid]
    await c.close()


# ----------------------------------------------------------------------------- eviction (F12)


async def test_evicted_bound_report_is_marked_dropped(env):
    a, clock = env
    a.scripts[CHAT] = lambda text: report("run report")
    task = await create(a, IN_CHAT, schedule={"in_min": 1})
    clock.advance(61)
    await settle(a)
    await finished(a, task)
    run_uid = runs_file(a, task.id)[0]["turn_uid"]
    c = a.svc.convs[CHAT]
    for i in range(20):  # chatd away: nothing is acked
        await a.svc.ask("e1", s.Ask(conv=CHAT, turn=i + 1, text=f"m{i}"))
        await until(lambda: c.running() is None)
    assert run_uid not in [u["turn_uid"] for u in c.record.unacked] and len(c.record.unacked) == 20
    rows = await history(a, task.id)
    assert rows[0]["delivery"] == "dropped" and "not posted" in rows[0]["note"]
    check(a)
