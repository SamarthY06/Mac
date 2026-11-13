"""Scheduled tasks: the code-review fixes (Mac-private/docs/tasks/CODE-REVIEW-*.md): the delete
race (S1), a missed once slot under a live run (S2), `fresh_next` without a session (S3), the
binary tail probe, a limit crash before finalise, the run uid on disk first, the exact
"continue <title>" / "stop <title>" replies while the limit is open (AT1), the clock in results
(AT3), plain error words (UX-11, UX-13) and `kind` on notices (UX-10)."""

from __future__ import annotations

import asyncio
import json

import pytest
from agentd_kit import APP_TOKEN, CHAT, CHAT_TOKEN, slow
from tasks_kit import (
    IN_CHAT,
    VOICE,
    agent,
    app_notices,
    check,
    create,
    finished,
    history,
    report,
    settle,
    stop,
    tasks_json,
    until,
)

from agentd import schema as s
from agentd import tasks as tk
from macd.brain import fake_brain as fb
from macd.contracts import BrainErrorKind


def limit_error(reset_in: float, now: float) -> list[fb.Step]:
    step = fb.error(BrainErrorKind.RATE_LIMIT, "limit", delay=0.01)
    step.data = {"reset_at": now + reset_in}
    return [step]


# ----------------------------------------------------------------------------- architecture


async def test_s1_delete_in_the_same_loop_step_as_dispatch_runs_nothing(env):
    a, clock = env
    app = await a.client(APP_TOKEN)
    task = await create(a, VOICE, schedule={"in_min": 1})
    conv = task.conv
    clock.advance(61)
    a.tasks.tick()  # dispatched: _run is spawned but has not executed yet
    assert task.state == "running" and task.current.turn_uid == ""
    r = await a.tasks.call(VOICE, "task_delete", {"task_id": task.id})  # lands before _run's first step
    assert r == {"deleted": True}
    await until(lambda: task.id not in a.tasks._runs)
    check(a)
    assert conv not in a.brains and conv not in a.svc.convs and task.id not in a.tasks.tasks
    assert not a.tasks.task_dir(task.id).exists() and task.busy_op == ""
    await until(lambda: app_notices(app, "tasks_changed") and app_notices(app, "tasks_changed")[-1].deleted)
    assert not app_notices(app, "task_report")
    await app.close()


async def test_s1_delete_finds_the_run_turn_by_run_id_before_its_uid_is_stored(env, monkeypatch):
    a, clock = env
    a.scripts["*"] = lambda text: slow("w", n=500, delay=0.01)
    task = await create(a, VOICE, schedule={"in_min": 1})
    conv = task.conv
    orig_save = a.tasks._save
    stall = asyncio.Event()

    async def slow_save() -> None:  # the uid save in _run takes a while: the turn exists, the record has no uid yet
        await orig_save()
        if task.current is not None and task.current.turn_uid and not stall.is_set():
            stall.set()
            await asyncio.sleep(0.2)

    a.tasks._save = slow_save  # type: ignore[method-assign]
    clock.advance(61)
    await settle(a)
    await until(lambda: stall.is_set())
    r = await asyncio.wait_for(a.tasks.call(VOICE, "task_delete", {"task_id": task.id}), 5)
    assert r == {"deleted": True}
    await until(lambda: task.id not in a.tasks._runs)
    check(a)
    assert conv not in a.svc.convs and task.id not in a.tasks.tasks
    assert conv not in a.brains or not a.brain(conv).asks  # the turn never reached a brain


async def test_an_ask_during_the_uid_save_sends_the_run_back_to_queued(env):
    a, clock = env
    n = [0]

    def script(text: str) -> list[fb.Step]:
        n[0] += 1
        return slow("owner", n=20, delay=0.01) if n[0] == 1 else report("run")

    a.scripts[CHAT] = script
    task = await create(a, IN_CHAT, schedule={"in_min": 1})
    orig_save = a.tasks._save
    asked = asyncio.Event()

    async def ask_during_save() -> None:
        await orig_save()
        if task.current is not None and task.current.turn_uid and not asked.is_set():
            asked.set()
            await a.svc.ask("e1", s.Ask(conv=CHAT, turn=1, text="hello"))  # reserves the conversation

    a.tasks._save = ask_during_save  # type: ignore[method-assign]
    c = await a.client(CHAT_TOKEN, [CHAT])
    clock.advance(61)
    await settle(a)
    await until(lambda: asked.is_set() and task.state == "queued")
    check(a)
    assert task.current.status == "queued" and task.current.turn_uid == "" and task.runs == 1
    assert task.id not in a.tasks._runs and not a.tasks._holds_slot(task)  # the slot is released (I3)
    await c.event("done")
    await finished(a, task)
    assert task.last_run["status"] == "ok" and task.runs == 1 and len(a.brain(CHAT).asks) == 2
    await c.close()


async def test_s2_missed_once_slot_under_a_live_run_ends_through_complete(env):
    a, clock = env
    a.scripts["*"] = lambda text: slow("w", n=500, delay=0.01)
    task = await create(a, VOICE, schedule={"once": tk.fmt_once(clock.now + 60)})
    await a.tasks.call(VOICE, "task_run", {"task_id": task.id})  # run_now just before the slot
    await settle(a)
    await until(lambda: task.conv in a.brains and a.brain(task.conv).asks)
    clock.advance(7 * 3600)  # the once slot is 7 h old while the run is still going
    await settle(a)
    check(a)
    assert task.state == "running" and task.current.status == "running"  # not `done` under a live current
    rows = await history(a, task.id)
    assert [r["status"] for r in rows] == ["missed"]
    await a.tasks.call(VOICE, "task_stop", {"task_id": task.id})
    await finished(a, task)
    assert task.state == "done" and task.runs == 1


async def test_s3_fresh_next_is_cleared_when_the_group_has_no_session_yet(env):
    a, clock = env
    a.scripts[CHAT] = lambda text: report("r", session="first")
    task = await create(a, IN_CHAT, schedule={"in_min": 1})
    r = a.svc.conversation(CHAT).record
    r.fresh_next = True  # `new chat` before any turn ran
    clock.advance(61)
    await settle(a)
    await finished(a, task)
    assert r.session_id == "first" and r.fresh_next is False and a.brain(CHAT).resets == 0
    c = await a.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "hello"})
    await c.event("done")
    assert r.session_id == "first" and a.brain(CHAT).resets == 0  # the real session was not discarded a turn later
    await c.close()


# ----------------------------------------------------------------------------- quality


async def test_history_append_survives_a_partial_multibyte_line(env):
    a, _ = env
    task = await create(a, VOICE, schedule={"at": ["08:00"]})
    path = a.tasks.task_dir(task.id) / "runs.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"run_id": "r0", "status": "ok"}\n{"run_id": "r1", "report": "caf\xc3')  # crash inside "é"
    await a.tasks._append_history(task, tk.RunRecord(run_id="r2", n=2, trigger="schedule", status="ok", report="café"))
    rows = tk._read_lines(path)
    assert [r["run_id"] for r in rows] == ["r0", "r2"] and rows[1]["report"] == "café"


async def test_limit_crash_before_finalize_recovers_with_history_and_delivery(short_root):
    from tasks_kit import Clock

    clock = Clock()
    a = await agent(short_root, clock)
    a.scripts["*"] = lambda text: limit_error(3600, clock.now)
    task = await create(a, VOICE, schedule={"in_min": 1})
    hang = asyncio.Event()

    async def never(*args, **kw):  # the crash sits between mark_limit's save and _finalize
        await hang.wait()

    a.tasks._finalize = never  # type: ignore[method-assign]
    clock.advance(61)
    await settle(a)
    await until(lambda: task.state == "waiting_limit")
    await stop(a)
    b = await agent(short_root, clock)
    app = await b.client(APP_TOKEN)
    try:
        t2 = b.tasks.tasks[task.id]
        check(b)
        assert t2.state == "waiting_limit" and t2.current.status == "limit" and t2.current.ended
        rows = await history(b, task.id)
        assert [r["status"] for r in rows] == ["limit"] and rows[0]["delivery"] == "notified"
        assert rows[0]["report"].startswith("Stopped at the Claude usage limit")
        assert b.svc.convs[task.conv].record.unacked == []
    finally:
        await app.close()
        await stop(b)


async def test_run_turn_uid_is_on_disk_before_the_turn_starts(env):
    a, clock = env
    a.scripts["*"] = lambda text: slow("w", n=200, delay=0.01)
    task = await create(a, VOICE, schedule={"in_min": 1})
    seen: list[tuple[str, str]] = []
    orig = a.svc._start_turn

    async def spy(c, t, meta):
        on_disk = json.loads((a.paths.state_dir / "tasks.json").read_text())["tasks"][task.id]["current"]["turn_uid"]
        seen.append((on_disk, t.uid))
        await orig(c, t, meta)

    a.svc._start_turn = spy  # type: ignore[method-assign]
    clock.advance(61)
    await settle(a)
    await until(lambda: seen)
    assert seen[0][0] == seen[0][1] != ""
    await a.tasks.call(VOICE, "task_stop", {"task_id": task.id})
    await finished(a, task)


# ----------------------------------------------------------------------------- the limit replies (AT1)


async def test_exact_continue_and_stop_replies_work_while_the_limit_is_open(env):
    a, clock = env
    mode = ["limit"]
    a.scripts[CHAT] = lambda text: limit_error(3600, clock.now) if mode[0] == "limit" else report("continued")
    task = await create(a, IN_CHAT, schedule={"every_min": 15})
    c = await a.client(CHAT_TOKEN, [CHAT])
    clock.advance(901)
    await settle(a)
    await finished(a, task)
    assert task.state == "waiting_limit" and a.svc.breaker.is_open()
    c.drain()
    reset_at = task.current.reset_at
    # anything else: the usual limit error, the model is not run
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "please continue the job search"})
    e = await c.event("error")
    assert e["kind"] == "rate_limit" and task.state == "waiting_limit"
    # the exact reply, case-insensitive, trailing punctuation ignored
    mode[0] = "ok"
    await c.call({"op": "ask", "conv": CHAT, "turn": 2, "text": "Continue Job Search!"})
    t = await c.event("text")
    assert t["text"] == f'I will pick up "Job search" at {tk.fmt_soon(reset_at, clock.now)}.'
    await c.event("done")
    assert task.state == "queued" and task.current.trigger == "continue" and task.not_before == reset_at
    assert len(a.brain(CHAT).asks) == 1  # no model call for the command
    clock.advance(3601)
    await settle(a)
    await finished(a, task)
    assert task.last_run["status"] == "ok"
    # stop, with the bare word when the chat waits on one task
    mode[0] = "limit"
    clock.advance(900)
    await settle(a)
    await finished(a, task)
    assert task.state == "waiting_limit"
    c.drain()
    await c.call({"op": "ask", "conv": CHAT, "turn": 3, "text": "stop."})
    t = await c.event("text")
    assert t["text"].startswith('Stopped "Job search". Its next run is ')
    await c.event("done")
    assert task.state == "active" and task.current is None
    # with the limit closed the words go to the model again
    a.svc.breaker.close()
    mode[0] = "ok"
    await c.call({"op": "ask", "conv": CHAT, "turn": 4, "text": "continue Job search"})
    await c.event("done")
    assert len(a.brain(CHAT).asks) == 4  # three runs and now the owner's words
    await c.close()


async def test_limit_reply_is_accepted_in_the_owner_chat_that_got_the_message(env):
    a, clock = env
    a.scripts["*"] = lambda text: limit_error(3600, clock.now)
    p = a.paths.mac_home / "chat" / "chats.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([{"id": CHAT.partition(":")[2], "subject": "Me", "is_self": True}]))
    task = await create(a, VOICE, schedule={"in_min": 1})
    c = await a.client(CHAT_TOKEN, [CHAT])
    clock.advance(61)
    await settle(a)
    await finished(a, task)
    await until(lambda: task.limit_conv == CHAT and a.svc.convs[CHAT].record.unacked)
    c.drain()
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "stop Job search"})
    t = await c.event("text")
    assert t["text"] == 'Stopped "Job search".'  # a once task: nothing further is scheduled
    assert task.state == "done"
    await c.close()


# ----------------------------------------------------------------------------- words and notices


async def test_create_results_carry_the_clock_and_errors_are_plain_words(env):
    a, clock = env
    app = await a.client(APP_TOKEN)
    r = await a.tasks.call(VOICE, "task_create", {"title": "Rem", "description": "call the bank", "kind": "reminder",
                                                   "schedule": {"in_min": 1}})
    assert r["now"] == clock.now and r["tz"]
    a.scripts["*"] = lambda text: [fb.error(BrainErrorKind.NETWORK, "offline")]
    task = await create(a, VOICE, schedule={"every_min": 15})
    clock.advance(901)
    await settle(a)
    await until(lambda: task.current is not None and task.current.trigger == "retry")
    await until(lambda: [n for n in app_notices(app, "task_report") if n.task_id == task.id])
    n = next(n for n in app_notices(app, "task_report") if n.task_id == task.id)
    assert n.body == "Could not finish: Mac could not reach Claude. Trying again in 2 minutes." and n.kind == "task"
    assert task.last_run["summary"] == n.body and "failed" not in task.last_run["summary"]
    rows = await history(a, task.id)
    assert rows[0]["note"] == n.body
    rem = a.tasks.tasks[r["task"]["id"]]
    await until(lambda: rem.state == "done")
    await finished(a, rem)
    rem_notice = next(x for x in app_notices(app, "task_report") if x.task_id == rem.id)
    assert rem_notice.kind == "reminder" and rem_notice.body == "call the bank"
    await app.close()


def test_fmt_soon_names_the_day_when_not_today():
    now = tk.once_instant("2026-11-13T10:00")
    assert tk.fmt_soon(tk.once_instant("2026-11-13T15:00"), now) == "3:00 PM"
    assert tk.fmt_soon(tk.once_instant("2026-11-14T15:00"), now) == "tomorrow at 3:00 PM"
    assert tk.fmt_soon(tk.once_instant("2026-11-16T09:30"), now) == "Mon 16 Nov 9:30 AM"
    assert tk._command("  Continue  Job Search!?. ") == "continue job search"


async def test_bound_run_rules_and_not_now_wording(env):
    a, clock = env
    task = await create(a, IN_CHAT, schedule={"in_min": 1})
    clock.advance(61)
    await settle(a)
    await finished(a, task)
    prompt = a.brain(CHAT).asks[0][1]
    assert "Put the top items in the message." in prompt and "give its path" not in prompt
    assert "If the owner asked you to stop, finish now with what you have" in tk.NOT_NOW
    assert s.TaskReport(task_id="t", run_id="r", title="x", body="b", status="ok").kind == "task"


# ----------------------------------------------------------------------------- model and effort per task


async def test_model_and_effort_round_trip_validate_and_old_records_load_empty(env):
    a, _ = env
    r = await a.tasks.call(VOICE, "task_create", {"title": "Jobs", "description": "find jobs", "schedule": {"every_min": 15},
                                                   "model": "Sonnet", "effort": "high"})
    task = a.tasks.tasks[r["task"]["id"]]
    assert (r["task"]["model"], r["task"]["effort"]) == ("sonnet", "high") == (task.model, task.effort)
    await until(lambda: tasks_json(a)["tasks"][task.id]["model"] == "sonnet")
    assert tk.TaskRecord.from_dict({**tasks_json(a)["tasks"][task.id], "id": task.id}) == task
    old = {k: v for k, v in task.to_dict().items() if k not in ("model", "effort")}  # a record from before
    loaded = tk.TaskRecord.from_dict({**old, "id": "told"})
    assert loaded.model == "" and loaded.effort == ""
    for args, words in [({"model": "Claude Opus 4"}, "model must be a name"), ({"model": "-x"}, "model must be a name"),
                        ({"model": "a" * 101}, "model must be a name"), ({"effort": "ultra"}, "effort must be"),
                        ({"effort": 3}, "effort must be"), ({"model": 1}, "model must be text")]:
        with pytest.raises(tk.TaskApiError, match=words):
            await a.tasks.call(VOICE, "task_update", {"task_id": task.id, **args})
    r = await a.tasks.call(VOICE, "task_update", {"task_id": task.id, "model": "claude-opus-4-1", "effort": ""})
    assert (r["task"]["model"], r["task"]["effort"]) == ("claude-opus-4-1", "") and task.rev == r["task"]["rev"] >= 2
    for ok in ("opus[1m]", "us.anthropic.claude-opus-4-1-20250805-v1:0"):  # Claude Code and gateway spellings
        assert (await a.tasks.call(VOICE, "task_update", {"task_id": task.id, "model": ok}))["task"]["model"] == ok


async def test_a_reminder_takes_no_model_and_a_model_change_shrinks_the_assumed_window(env):
    a, _ = env
    base = {"title": "Bank", "description": "Call the bank", "kind": "reminder", "schedule": {"every_min": 15}}
    with pytest.raises(tk.TaskApiError, match="a reminder does not run Claude"):
        await a.tasks.call(VOICE, "task_create", {**base, "model": "opus"})
    r = await a.tasks.call(VOICE, "task_create", {**base, "model": "", "effort": ""})
    with pytest.raises(tk.TaskApiError, match="a reminder does not run Claude"):
        await a.tasks.call(VOICE, "task_update", {"task_id": r["task"]["id"], "effort": "high"})
    await a.tasks.call(VOICE, "task_update", {"task_id": r["task"]["id"], "model": "", "title": "Bank call"})  # clearing is fine
    r = await a.tasks.call(VOICE, "task_create", {"title": "Jobs", "description": "find jobs", "schedule": {"every_min": 15}})
    task = a.tasks.tasks[r["task"]["id"]]
    task.context = {"session": "s1", "tokens": 400_000, "window": 1_000_000}
    await a.tasks.call(VOICE, "task_update", {"task_id": task.id, "effort": "high"})  # same model: the window stands
    assert task.context["window"] == 1_000_000
    await a.tasks.call(VOICE, "task_update", {"task_id": task.id, "model": "haiku"})  # measured on the old model
    assert task.context == {"session": "s1", "tokens": 400_000, "window": tk.MIN_MODEL_WINDOW}
    assert "model" in tk.ERROR_WORDS and "model" not in tk.RETRY_KINDS  # a missing model is said once, not retried


async def test_spec_carries_the_tasks_model_and_effort_and_an_update_applies_at_the_next_run(env):
    a, clock = env
    app = await a.client(APP_TOKEN)
    r = await a.tasks.call(VOICE, "task_create", {"title": "Jobs", "description": "find jobs", "schedule": {"every_min": 15},
                                                   "model": "haiku", "effort": "low"})
    task = a.tasks.tasks[r["task"]["id"]]
    clock.advance(901)
    await settle(a)
    await finished(a, task)
    spec = a.brain(task.conv).spec
    assert spec.model == "haiku" and spec.effort == "low" and len(a.brains[task.conv]) == 1
    c = a.svc.convs[task.conv]
    assert c.brain is not None and c.running() is None
    n = len(app_notices(app, "tasks_changed"))
    rev = task.rev
    await a.tasks.call(VOICE, "task_update", {"task_id": task.id, "model": "opus"})  # applies at the next run
    assert task.rev == rev + 1
    await until(lambda: len(app_notices(app, "tasks_changed")) == n + 1)
    clock.advance(900)
    await settle(a)
    await finished(a, task)
    assert len(a.brains[task.conv]) == 2 and a.brain(task.conv).spec.model == "opus" and a.brain(task.conv).spec.effort == "low"
    await a.tasks.call(VOICE, "task_update", {"task_id": task.id, "model": "", "effort": ""})  # back to the defaults
    clock.advance(900)
    await settle(a)
    await finished(a, task)
    assert a.brain(task.conv).spec.model == "" and a.brain(task.conv).spec.effort is None
    await app.close()


async def test_model_change_during_a_run_applies_after_it(env):
    a, clock = env
    a.scripts[CHAT] = lambda text: slow("w", n=200, delay=0.01)
    task = await create(a, IN_CHAT, schedule={"every_min": 15})
    clock.advance(901)
    await settle(a)
    await until(lambda: CHAT in a.brains and a.brain(CHAT).asks)
    c = a.svc.convs[CHAT]
    first = c.brain
    await a.tasks.call(IN_CHAT, "task_update", {"effort": "max"})
    assert c.brain is first and task.effort == "max"  # no mid-run swap
    await a.tasks.call(IN_CHAT, "task_stop", {})
    await finished(a, task)
    assert c.brain is first
    cl = await a.client(CHAT_TOKEN, [CHAT])
    await cl.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "hello"})
    await cl.event("done")
    assert c.brain is not first and c.brain.spec.effort == "max"  # the group's brain uses the task's values
    assert first.closed
    await cl.close()


def test_factory_passes_model_and_effort_only_when_set(monkeypatch):
    from test_agentic import _spec, _spy_factory

    from agentd import brains
    from macd import config
    from macd.brain import claude_brain as cb

    names = sorted({*brains.CLAUDE_ARGS, *brains.CHAT_SAFETY_ARGS, *cb.CHAT_PROFILE, "task_api",
                    "chat_persona", "display_name", "chat_shared", "on_set_persona", "snapshot"})
    make, seen = _spy_factory(monkeypatch, names)
    make(_spec(brains.TASK, never_chat_names=("Timetable",), capabilities=lambda: brains.TASK_CAPS,
               confirm_send_file=lambda: False, unattended=lambda: True, model="opus", effort="xhigh"))
    make(_spec(brains.TASK, never_chat_names=("Timetable",), capabilities=lambda: brains.TASK_CAPS,
               confirm_send_file=lambda: False, unattended=lambda: True))
    assert seen[0]["_model"] == "opus" and seen[0]["effort"] == "xhigh"  # the model rides in the brain's config
    assert seen[1]["_model"] == config.Config().brain.model and "effort" not in seen[1]  # defaults: the config decides
