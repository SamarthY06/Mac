"""Scheduled tasks: the run lifecycle on the task conversation, bound groups, the concurrency
cases C1 to C21 and summarise-then-fresh (DESIGN 5, 6, 12). Every step asserts the 12.4 invariants
through `tasks_kit.check`."""

from __future__ import annotations

import asyncio

import pytest
from agentd_kit import (
    APP_TOKEN,
    CHAT,
    CHAT_TOKEN,
    OTHER_CHAT,
    slow,
)
from tasks_kit import (
    APP,
    IN_CHAT,
    VOICE,
    app_notices,
    check,
    convs_json,
    create,
    finished,
    history,
    report,
    runs_file,
    settle,
    until,
)

from agentd import brains
from agentd import schema as s
from agentd import tasks as tk
from macd.brain import fake_brain as fb
from macd.contracts import BrainErrorKind, BrainEventKind, RateLimited

# ----------------------------------------------------------------------------- unbound runs


async def test_scheduled_run_fires_on_the_task_conv_with_the_task_spec(env):
    a, clock = env
    app = await a.client(APP_TOKEN)
    task = await create(a, VOICE, schedule={"in_min": 1})
    assert task.conv == f"task:{task.id}" and task.report_to == s.VOICE_CONV and not task.bound
    assert task.next_run() == clock.now + 60 and task.state == "active"
    await settle(a)
    assert task.state == "active"  # not due yet
    clock.advance(61)
    await settle(a)
    assert task.runs == 1
    await finished(a, task)
    b = a.brain(task.conv)
    spec = b.spec
    assert spec.profile is brains.TASK and spec.capabilities() == brains.TASK_CAPS and spec.chrome is True
    assert spec.cwd == a.paths.voice_cwd and spec.unattended() is True and spec.task_api is None
    assert spec.confirm_send_file() is False and spec.setting_sources == ["user"]
    prompt = b.asks[0][1]
    assert prompt.startswith('[Scheduled task "Job search", run 1. Last run: first run.') and "find 20 jobs" in prompt
    assert "none yet" in prompt and 'Scheduled run of "Job search".' in prompt
    c = a.svc.convs[task.conv]
    assert list(c.recent) == []  # the run text never reaches grounding
    assert c.record.unacked == []  # acked after finalize: the mailbox stays empty
    assert task.state == "done" and task.runs == 1 and task.last_run["status"] == "ok"
    assert task.last_run["summary"] == "Found 3 jobs."
    assert task.context == {"session": "s1", "tokens": 1000, "window": 200000}
    rows = runs_file(a, task.id)
    assert [r["status"] for r in rows] == ["ok"] and rows[0]["report"].startswith("Found 3 jobs.")
    assert rows[0]["delivery"] == "notified" and rows[0]["turn_uid"] and rows[0]["trigger"] == "schedule"
    await until(lambda: app_notices(app, "task_report"))
    n = app_notices(app, "task_report")[0]
    assert (n.task_id, n.run_id, n.title, n.body, n.status) == (task.id, "r1", "Job search", "Found 3 jobs.", "ok")
    assert app_notices(app, "tasks_changed")[-1].rev == a.tasks.rev
    assert a.svc.status(None).conversations == [cs for cs in a.svc.status(None).conversations if not cs.conv.startswith("task:")]
    await app.close()


async def test_osascript_fallback_when_no_app_client(env):
    a, clock = env
    calls: list[tuple[str, str]] = []

    async def fake_osascript(title: str, body: str) -> None:
        calls.append((title, body))

    a.tasks.osascript = fake_osascript
    task = await create(a, APP, schedule={"in_min": 1})
    assert task.report_to == "app"
    clock.advance(61)
    await settle(a)
    await finished(a, task)
    assert calls == [("Job search", "Found 3 jobs.")]
    assert runs_file(a, task.id)[0]["delivery"] == "notified"


async def test_unbound_reminder_delivers_its_text_without_a_brain(env):
    a, clock = env
    app = await a.client(APP_TOKEN)
    task = await create(a, VOICE, kind="reminder", description="call the bank", schedule={"in_min": 2})
    clock.advance(121)
    await settle(a)
    await finished(a, task)
    assert task.conv not in a.brains and task.state == "done"
    await until(lambda: app_notices(app, "task_report"))
    assert app_notices(app, "task_report")[0].body == "call the bank"
    await app.close()


async def test_every_min_catch_up_collapses_slots_and_a_late_at_slot_is_missed(env):
    a, clock = env
    task = await create(a, VOICE, schedule={"every_min": 30})
    assert task.next_run() == clock.now + 1800
    clock.advance(1800 + 10)
    await settle(a)
    await finished(a, task)
    assert task.runs == 1 and task.current is None and task.anchor == clock.now - 10
    # asleep for 2 h: four slots collapse into one late run
    clock.advance(2 * 3600)
    await settle(a)
    assert task.runs == 2
    await finished(a, task)
    assert runs_file(a, task.id)[-1]["trigger"] == "catch_up"
    assert "later than scheduled; it was not run on time" in a.brain(task.conv).asks[-1][1]
    assert task.anchor == clock.now - 10  # the last slot handled, not the first missed one
    # a daily slot more than 6 h old is missed: one history line, no run
    daily = await create(a, VOICE, title="Daily", schedule={"at": ["08:00"]})
    clock.advance(29 * 3600)  # Sat 15:00: the 08:00 slot is 7 h old
    await settle(a)
    assert daily.runs == 0 and daily.current is None and daily.state == "active"
    rows = await history(a, daily.id)
    assert [r["status"] for r in rows] == ["missed"] and rows[0]["note"] == "Not run on time: the 8:00 AM slot was missed."
    assert daily.last_run["status"] == "missed" and daily.next_run() == tk.next_slot({"at": ["08:00"]}, clock.now)


async def test_missed_once_task_is_done_and_missed_reminder_still_delivered(env):
    a, clock = env
    app = await a.client(APP_TOKEN)
    once = await create(a, VOICE, title="One", schedule={"once": tk.fmt_once(clock.now + 3600)})
    rem = await create(a, VOICE, title="Rem", kind="reminder", description="take the pills",
                       schedule={"once": tk.fmt_once(clock.now + 3600)})
    clock.advance(8 * 3600)
    await settle(a)
    await finished(a, rem)
    assert once.state == "done" and once.runs == 0 and once.last_run["status"] == "missed"
    assert rem.state == "done" and rem.runs == 1
    await until(lambda: app_notices(app, "task_report"))
    assert app_notices(app, "task_report")[0].body.startswith("Missed reminder from ") and "take the pills" in \
        app_notices(app, "task_report")[0].body
    await app.close()


async def test_clock_set_back_repeats_a_slot_at_most_once_with_the_note(env):
    a, clock = env
    task = await create(a, VOICE, schedule={"at": ["08:00"]})
    clock.advance(22 * 3600)  # Sat 08:00
    await settle(a)
    await finished(a, task)
    assert task.runs == 1
    clock.advance(-3 * 3600)  # the clock goes back to 05:00
    await settle(a)
    assert task.anchor == clock.now and task.runs == 1
    await settle(a)  # a second tick clamps nothing more and runs nothing
    assert task.runs == 1
    clock.advance(3 * 3600 + 5)  # 08:00 again
    await settle(a)
    await finished(a, task)
    assert task.runs == 2 and "clock was changed" in a.brain(task.conv).asks[-1][1]
    clock.advance(10)
    await settle(a)
    assert task.runs == 2


async def test_once_fires_within_one_tick_and_a_past_once_within_the_grace_runs_now(env):
    a, clock = env
    task = await create(a, VOICE, schedule={"once": tk.fmt_once(clock.now - 60)})
    await settle(a)
    assert task.runs == 1
    await finished(a, task)
    assert task.state == "done"
    with pytest.raises(tk.TaskApiError, match=r"That time has passed \(it is now Fri 13 Nov 10:00 AM\)\."):
        await create(a, VOICE, title="Old", schedule={"once": tk.fmt_once(clock.now - 600)})


# ----------------------------------------------------------------------------- bound groups


async def test_chat_create_binds_and_a_second_create_is_task_exists(env):
    a, _ = env
    task = await create(a, IN_CHAT)
    assert task.bound and task.conv == CHAT == task.report_to and a.tasks.by_conv[CHAT] == task.id
    assert a.tasks.view(task)["where"] == f"Posts in {a.svc.profile_of(CHAT).display_name}"
    with pytest.raises(tk.TaskApiError) as e:
        await create(a, IN_CHAT, title="Second")
    assert e.value.code == "task_exists" and e.value.message == \
        'This chat already runs "Job search". Change that one, delete it first, or create the new task in another chat.'
    other = await create(a, tk.Origin(conv=OTHER_CHAT, kind="chat"), title="Second")
    assert other.conv == OTHER_CHAT
    assert await a.svc.validate_switch(a.svc.conversation(CHAT), "Resume", "new") == tk.SWITCH_REFUSED


async def test_bound_run_is_a_turn_in_unacked_when_chatd_is_away(env):
    a, clock = env
    a.scripts[CHAT] = lambda text: report("Posted in the group.")
    task = await create(a, IN_CHAT, schedule={"in_min": 1})
    clock.advance(61)
    await settle(a)
    await finished(a, task)
    r = a.svc.convs[CHAT].record
    assert [u["answer"] for u in r.unacked] == ["Posted in the group."]
    assert r.unacked[0]["turn_uid"] == runs_file(a, task.id)[0]["turn_uid"] and tk.is_task_uid(r.unacked[0]["turn_uid"])
    assert runs_file(a, task.id)[0]["delivery"] == "posted" and task.state == "done"
    assert convs_json(a)[CHAT]["unacked"][0]["answer"] == "Posted in the group."
    b = a.brain(CHAT)
    assert b.spec.profile is brains.CHAT and b.spec.task_api is not None  # the group keeps its CHAT brain
    assert "outward_sends" in a.svc.live_caps(CHAT)


async def test_owner_turn_and_run_share_the_brain_and_session(env):
    a, clock = env
    a.scripts[CHAT] = lambda text: report("ok", session="shared-1")
    task = await create(a, IN_CHAT, schedule={"in_min": 1})
    c = await a.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "hello"})
    await c.event("done")
    assert a.svc.convs[CHAT].record.session_id == "shared-1"
    assert task.context == {"session": "shared-1", "tokens": 1000, "window": 200000}
    clock.advance(61)
    await settle(a)
    await finished(a, task)
    assert len(a.brains[CHAT]) == 1 and len(a.brain(CHAT).asks) == 2
    assert a.svc.convs[CHAT].record.session_id == "shared-1"
    assert a.svc.convs[CHAT].record.folder is None and not a.model.calls  # no router for a bound group
    await c.close()


async def test_bound_reminder_is_a_canned_turn_that_waits_behind_an_owner_turn(env):
    a, clock = env
    a.scripts[CHAT] = lambda text: slow("working", n=30, delay=0.01)
    task = await create(a, IN_CHAT, kind="reminder", description="stand up", schedule={"in_min": 1})
    c = await a.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "hello"})
    clock.advance(61)
    await settle(a)
    assert task.state == "queued"  # the conversation is busy: the reminder keeps its place
    await c.event("done")
    await finished(a, task)
    evs = [e for e in c.drain() if e["ev"] in ("text", "done") and e["turn_uid"] != c.frames[-1].get("x")]
    canned = [e for e in evs if e["ev"] == "text" and e["text"] == "stand up"]
    assert len(canned) == 1 and tk.is_task_uid(canned[0]["turn_uid"])
    r = a.svc.convs[CHAT].record
    assert [u["answer"] for u in r.unacked][-1] == "stand up"  # never acked by agentd: chatd posts it
    assert len(a.brain(CHAT).asks) == 1  # no brain for a canned turn
    await c.close()


# ----------------------------------------------------------------------------- concurrency


async def test_c1_two_ticks_one_run_and_c2_run_now_is_busy(env):
    a, clock = env
    a.scripts["*"] = lambda text: slow("w", n=20, delay=0.01)
    task = await create(a, VOICE, schedule={"in_min": 1})
    clock.advance(61)
    a.tasks.tick()
    a.tasks.tick()
    check(a)
    await until(lambda: task.state == "running" and task.conv in a.brains)
    assert task.runs == 1
    with pytest.raises(tk.TaskApiError) as e:
        await a.tasks.call(VOICE, "task_run", {"task_id": task.id})
    assert e.value.code == "busy"
    await finished(a, task)
    assert task.runs == 1 and len(a.brain(task.conv).asks) == 1


async def test_c3_stop_after_the_brain_finished_is_ok_and_mid_run_is_stopped(env):
    a, clock = env
    mode = ["fast"]
    a.scripts["*"] = lambda text: report() if mode[0] == "fast" else slow("w", n=200, delay=0.01)
    task = await create(a, VOICE, schedule={"every_min": 15})
    clock.advance(901)
    await settle(a)
    await finished(a, task)
    r = await a.tasks.call(VOICE, "task_stop", {"task_id": task.id})
    assert r["stopped"] is False and task.last_run["status"] == "ok"
    mode[0] = "slow"
    r = await a.tasks.call(VOICE, "task_run", {"task_id": task.id})
    assert r["task"]["state"] in ("queued", "running") and r["task"]["current"]["trigger"] == "run_now"
    await settle(a)
    await until(lambda: a.brain(task.conv).asks and len(a.brain(task.conv).asks) == 2)
    await until(lambda: a.svc.convs[task.conv].running() is not None and a.svc.convs[task.conv].running().in_brain)
    r = await a.tasks.call(VOICE, "task_stop", {"task_id": task.id})
    assert r["stopped"] is True
    await finished(a, task)
    assert [x["status"] for x in runs_file(a, task.id)] == ["ok", "stopped"] and task.state == "active"


async def test_c5_delete_mid_run_with_a_brain_that_never_ends(env, monkeypatch):
    a, clock = env
    monkeypatch.setattr(tk, "DELETE_WAIT_S", 0.2)
    a.scripts["*"] = lambda text: slow("w", n=500, delay=0.05)
    app = await a.client(APP_TOKEN)
    task = await create(a, VOICE, schedule={"in_min": 1})
    conv = task.conv
    clock.advance(61)
    await settle(a)
    await until(lambda: conv in a.brains and a.brain(conv).asks)
    b = a.brain(conv)

    async def never(turn_id: int) -> None:
        b.interrupts.append(turn_id)  # the brain hangs: no DONE follows

    b.interrupt = never

    async def fast_close() -> None:  # the real brain kills its process; FakeBrain would wait for the script
        b._task.cancel()
        b.closed = True
        await b._out.put(None)

    b.close = fast_close
    run_task = a.tasks._runs[task.id]
    r = await asyncio.wait_for(a.tasks.call(VOICE, "task_delete", {"task_id": task.id}), 3)
    assert r == {"deleted": True}
    check(a)
    assert task.id not in a.tasks.tasks and conv not in a.tasks.by_conv and conv not in a.svc.convs
    assert run_task.done() and not a.tasks.task_dir(task.id).exists()
    assert conv not in convs_json(a) and task.id not in (await a.tasks._store.load({}))["tasks"]
    await until(lambda: app_notices(app, "tasks_changed") and app_notices(app, "tasks_changed")[-1].deleted)
    assert not app_notices(app, "task_report")
    await app.close()


async def test_c7_slot_during_an_owner_turn_waits_then_runs(env):
    a, clock = env
    n = [0]

    def script(text: str) -> list[fb.Step]:
        n[0] += 1
        return slow("w", n=30, delay=0.01) if n[0] == 1 else report("run report")

    a.scripts[CHAT] = script
    task = await create(a, IN_CHAT, schedule={"in_min": 1})
    c = await a.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "hello"})
    clock.advance(61)
    await settle(a, ticks=3)
    assert task.state == "queued" and task.runs == 1
    await c.event("done")
    await finished(a, task)
    assert task.last_run["status"] == "ok" and [x["status"] for x in runs_file(a, task.id)] == ["ok"]
    await c.close()


async def test_c6_owner_steer_during_a_run_reaches_the_brain(env):
    a, clock = env
    a.scripts[CHAT] = lambda text: slow("w", n=40, delay=0.01)
    task = await create(a, IN_CHAT, schedule={"in_min": 1})
    c = await a.client(CHAT_TOKEN, [CHAT])
    clock.advance(61)
    await settle(a)
    await until(lambda: CHAT in a.brains and a.brain(CHAT).asks)
    r = await c.call({"op": "steer", "conv": CHAT, "text": "only London"})
    assert r["data"]["accepted"] is True
    r = await c.call({"op": "ask", "conv": CHAT, "turn": 2, "text": "and Paris"})
    assert r["code"] == "busy"
    await until(lambda: a.brain(CHAT).steers)
    assert a.brain(CHAT).steers[0][1] == "only London"
    assert list(a.svc.convs[CHAT].recent) == ["only London"]  # the owner's words, not the run text
    await finished(a, task)
    await c.close()


async def test_c10_max_two_concurrent_and_fifo(env):
    a, clock = env
    a.scripts["*"] = lambda text: slow("w", n=25, delay=0.01)
    t1 = await create(a, VOICE, title="A", schedule={"in_min": 1})
    clock.advance(1)
    t2 = await create(a, VOICE, title="B", schedule={"in_min": 1})
    clock.advance(1)
    t3 = await create(a, VOICE, title="C", schedule={"in_min": 1})
    clock.advance(61)
    await settle(a)
    assert (t1.state, t2.state, t3.state) == ("running", "running", "queued")
    await settle(a, ticks=2)
    assert t3.state == "queued"
    await finished(a, t1)
    await settle(a)
    await until(lambda: t3.state == "running")
    assert t3.current is not None and t3.current.queued_at > t2.current.queued_at if t2.current else True
    await finished(a, t2)
    await finished(a, t3)
    assert all(x.runs == 1 for x in (t1, t2, t3))


async def test_c11_breaker_open_holds_the_queue(env):
    a, clock = env
    task = await create(a, VOICE, schedule={"in_min": 1})
    a.svc.breaker.trip(clock.now + 600)
    clock.advance(61)
    await settle(a, ticks=2)
    assert task.state == "queued" and task.conv not in a.brains
    clock.advance(600)
    await settle(a)
    await finished(a, task)
    assert task.last_run["status"] == "ok"


async def test_f1_run_recheck_sends_the_task_back_to_queued_when_an_ask_won(env):
    a, clock = env
    n = [0]

    def script(text: str) -> list[fb.Step]:
        n[0] += 1
        return slow("owner", n=30, delay=0.01) if n[0] == 1 else report("run")

    a.scripts[CHAT] = script
    task = await create(a, IN_CHAT, schedule={"in_min": 1})
    c = await a.client(CHAT_TOKEN, [CHAT])
    clock.advance(61)
    a.tasks.tick()  # dispatch marked it running and spawned _run, which has not executed yet
    assert task.state == "running"
    await a.svc.ask("e1", s.Ask(conv=CHAT, turn=1, text="hello"))  # reserves the conversation first
    await until(lambda: task.state == "queued")
    check(a)
    assert task.state == "queued" and task.runs == 1 and a.svc.convs[CHAT].running() is not None
    await c.event("done")
    await finished(a, task)
    assert task.runs == 1 and task.last_run["status"] == "ok" and len(a.brain(CHAT).asks) == 2
    await c.close()


async def test_pause_lets_the_run_finish_then_holds_and_update_applies_at_once(env):
    a, clock = env
    a.scripts["*"] = lambda text: slow("w", n=20, delay=0.01)
    task = await create(a, VOICE, schedule={"every_min": 15})
    clock.advance(901)
    await settle(a)
    assert task.state == "running"
    r = await a.tasks.call(VOICE, "task_update", {"task_id": task.id, "paused": True, "title": "Jobs"})
    assert r["task"]["state"] == "running" and r["task"]["pause_reason"] == "user" and r["task"]["title"] == "Jobs"
    await finished(a, task)
    assert task.state == "paused" and task.runs == 1
    clock.advance(900)
    await settle(a)
    assert task.state == "paused" and task.runs == 1 and task.anchor == clock.now - 1  # advanced silently
    r = await a.tasks.call(VOICE, "task_update", {"task_id": task.id, "paused": False})
    assert r["task"]["state"] == "active"
    r = await a.tasks.call(VOICE, "task_update", {"task_id": task.id, "schedule": {"every_min": 30}})
    assert r["task"]["schedule_text"] == "Every 30 minutes" and r["task"]["next_run"] == clock.now + 1800


async def test_chat_off_pauses_a_bound_task_and_tears_a_running_run(env):
    a, clock = env
    a.scripts[CHAT] = lambda text: slow("w", n=200, delay=0.01)
    task = await create(a, IN_CHAT, schedule={"every_min": 15})
    clock.advance(901)
    await settle(a)
    await until(lambda: CHAT in a.brains and a.brain(CHAT).asks)
    channel, _, cid = CHAT.partition(":")
    p = a.svc.profile_of(CHAT)
    await a.svc.profile_set(s.ProfileSet(channel=channel, chat_id=cid, rev=p.rev, patch={"enabled": False}))
    await finished(a, task)
    assert task.last_run["status"] == "stopped" and runs_file(a, task.id)[0]["delivery"] == "none"
    assert a.svc.convs[CHAT].record.unacked == []  # torn: nothing of it is posted
    await settle(a)
    assert task.state == "paused" and task.pause_reason == "chat_off"


# ----------------------------------------------------------------------------- summarise then fresh


async def test_threshold_summarises_the_old_session_then_fresh(env):
    a, clock = env
    a.scripts["*"] = lambda text: report("r", session="big", tokens=120000)
    task = await create(a, VOICE, schedule={"every_min": 15})
    clock.advance(901)
    await settle(a)
    await finished(a, task)
    b = a.brain(task.conv)
    assert task.context["tokens"] == 120000 and b.summaries == [] and b.resets == 0
    c = a.svc.convs[task.conv]
    assert c.record.session_id == "big"
    clock.advance(900)
    await settle(a)
    await finished(a, task)
    assert b.summaries == [tk.SUMMARY_PROMPT] and b.resets == 1
    notes = (a.tasks.task_dir(task.id) / "notes.md").read_text()
    assert notes.startswith("Notes: done so far.") and notes in b.asks[1][1]
    assert c.record.session_id == "big"  # the run after the reset measured the (new) session again
    assert c.record.framed.get("") is None


async def test_fresh_next_summarises_and_a_foreign_measurement_does_not(env):
    a, clock = env
    a.scripts["*"] = lambda text: report("r", session="s-a", tokens=120000)
    task = await create(a, VOICE, schedule={"every_min": 15})
    clock.advance(901)
    await settle(a)
    await finished(a, task)
    c = a.svc.convs[task.conv]
    task.context = {"session": "other", "tokens": 190000, "window": 200000}  # a stale number (I10)
    clock.advance(900)
    await settle(a)
    await finished(a, task)
    assert a.brain(task.conv).summaries == [] and a.brain(task.conv).resets == 0
    c.record.fresh_next = True
    clock.advance(900)
    await settle(a)
    await finished(a, task)
    assert len(a.brain(task.conv).summaries) == 1 and a.brain(task.conv).resets == 1 and not c.record.fresh_next


@pytest.mark.parametrize("summary", [None, "too short", "x" * 7000, RuntimeError("boom")])
async def test_summary_failures_keep_old_notes_and_start_fresh_anyway(env, summary):
    a, clock = env
    a.scripts["*"] = lambda text: report("r", session="big", tokens=150000)
    task = await create(a, VOICE, schedule={"every_min": 15})
    (a.tasks.task_dir(task.id)).mkdir(parents=True)
    (a.tasks.task_dir(task.id) / "notes.md").write_text("old notes that stay")
    clock.advance(901)
    await settle(a)
    await finished(a, task)
    b = a.brain(task.conv)
    b.summary = summary
    clock.advance(900)
    await settle(a)
    await finished(a, task)
    assert len(b.summaries) == 1 and b.resets == 1
    assert (a.tasks.task_dir(task.id) / "notes.md").read_text() == "old notes that stay"
    assert "old notes that stay" in b.asks[-1][1]


async def test_summary_rate_limit_trips_the_breaker_and_the_run_waits_at_the_limit(env):
    a, clock = env
    a.scripts["*"] = lambda text: report("r", session="big", tokens=150000)
    task = await create(a, VOICE, schedule={"every_min": 15})
    clock.advance(901)
    await settle(a)
    await finished(a, task)
    b = a.brain(task.conv)
    b.summary = RateLimited(clock.now + 3600)
    clock.advance(900)
    await settle(a)
    await finished(a, task)
    assert a.svc.breaker.is_open() and task.state == "waiting_limit" and b.resets == 1
    assert len(b.asks) == 1  # the second run never reached the brain


async def test_c20_delete_while_the_summary_is_in_flight(env, monkeypatch):
    a, clock = env
    a.scripts["*"] = lambda text: report("r", session="big", tokens=150000)
    task = await create(a, VOICE, schedule={"every_min": 15})
    clock.advance(901)
    await settle(a)
    await finished(a, task)
    b = a.brain(task.conv)
    b.summary_delay = 5.0
    clock.advance(900)
    await settle(a)
    await until(lambda: b.summaries)
    r = await asyncio.wait_for(a.tasks.call(VOICE, "task_delete", {"task_id": task.id}), 3)
    assert r["deleted"] is True and b.summary_interrupted == 1
    check(a)
    assert task.conv not in a.svc.convs and task.id not in a.tasks.tasks


async def test_run_prompt_carries_notes_after_a_session_fallback(env):
    a, clock = env
    a.scripts["*"] = lambda text: [fb.Step(BrainEventKind.STATUS, data={"session_fallback": {"session_id": "lost"}}),
                                   *report("r", session="fresh")]
    task = await create(a, VOICE, schedule={"every_min": 15})
    (a.tasks.task_dir(task.id)).mkdir(parents=True)
    (a.tasks.task_dir(task.id) / "notes.md").write_text("Where results are: ~/Documents/Mac/jobs.md")
    clock.advance(901)
    await settle(a)
    await finished(a, task)
    prompt = a.brain(task.conv).asks[0][1]
    assert "find 20 jobs" in prompt and "Where results are" in prompt
    assert task.context["session"] == "fresh" and a.svc.convs[task.conv].record.session_id == "fresh"


async def test_network_error_retries_once_on_the_same_session(env):
    a, clock = env
    mode = ["fail"]
    a.scripts["*"] = lambda text: [fb.error(BrainErrorKind.NETWORK, "offline")] if mode[0] == "fail" else report("recovered")
    task = await create(a, VOICE, schedule={"every_min": 15})
    clock.advance(901)
    await settle(a)
    await until(lambda: task.current is not None and task.current.trigger == "retry")
    check(a)
    assert task.not_before == clock.now + 120 and task.runs == 2
    rows = runs_file(a, task.id)
    assert rows[0]["status"] == "error" and rows[0]["error_kind"] == "network" and rows[0]["note"] == "Could not finish: Mac could not reach Claude. Trying again in 2 minutes."
    await settle(a)
    assert task.state == "queued"  # not before 2 minutes
    mode[0] = "ok"
    clock.advance(121)
    await settle(a)
    await finished(a, task)
    assert task.last_run["status"] == "ok" and task.runs == 2 and len(a.brains[task.conv]) == 1
    assert "The previous attempt failed (network); try again." in a.brain(task.conv).asks[-1][1]
    mode[0] = "fail"
    clock.advance(900)
    await settle(a)
    await until(lambda: task.current is not None and task.current.trigger == "retry")
    clock.advance(121)
    await settle(a)
    await finished(a, task)
    assert task.state == "active" and [r["status"] for r in runs_file(a, task.id)][-2:] == ["error", "error"]
    assert task.runs == 4  # the retry failed too: no second retry


async def test_task_profile_reaches_the_claude_factory(monkeypatch):
    from test_agentic import _spec, _spy_factory

    from macd.brain import claude_brain as cb

    names = sorted({*brains.CLAUDE_ARGS, *brains.CHAT_SAFETY_ARGS, *cb.CHAT_PROFILE, "task_api", "on_alive",
                    "chat_persona", "display_name", "chat_shared", "on_set_persona", "snapshot"})
    make, seen = _spy_factory(monkeypatch, names)

    def caps():
        return brains.TASK_CAPS

    make(_spec(brains.TASK, never_chat_names=("Timetable",), capabilities=caps, confirm_send_file=lambda: False,
               unattended=lambda: True, chrome=True))
    (task,) = seen
    assert task["channel"] == "task" and task["send_file"] is False and task["switch_workspace"] is False
    assert task["capabilities"] is caps and task["whatsapp_ui_block"] is True and task["unattended"]() is True
    assert "task_api" not in task and task["never_chat_names"] == ["Timetable"] and task["handoff_enabled"] is False
    with pytest.raises(RuntimeError):
        make(_spec(brains.TASK, never_chat_names=("Timetable",), capabilities=caps, confirm_send_file=lambda: False))
