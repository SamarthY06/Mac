"""Scheduled tasks: the API (DESIGN 10) over the socket and in process, origin rules (9.2),
idempotency (I13), the store (3.1) and log hygiene (I12)."""

from __future__ import annotations

import logging

import pytest
from agentd_kit import (
    APP_TOKEN,
    CHAT,
    CHAT_TOKEN,
    OTHER_CHAT,
    VOICE_TOKEN,
    slow,
)
from tasks_kit import (
    APP,
    IN_CHAT,
    VOICE,
    Clock,
    agent,
    app_notices,
    check,
    create,
    finished,
    report,
    runs_file,
    settle,
    stop,
    tasks_json,
    until,
)

from agentd import schema as s
from agentd import tasks as tk

SCHEDULE = {"at": ["08:00"], "days": ["mon", "tue", "wed", "thu", "fri"]}


# ----------------------------------------------------------------------------- ops by token kind


@pytest.mark.parametrize("token", [VOICE_TOKEN, APP_TOKEN])
async def test_task_ops_for_voice_and_app_tokens(env, token):
    a, clock = env
    c = await a.client(token)
    r = await c.call({"op": "task_create", "title": "Job search", "description": "find 20 jobs", "schedule": SCHEDULE})
    assert r["ok"], r
    t = r["data"]["task"]
    assert t["schedule_text"] == "Weekdays at 8:00 AM" and t["state"] == "active" and t["rev"] == 1
    assert t["next_run"] == tk.next_slot(SCHEDULE, clock.now) and t["where"] == "Notifies on this Mac"
    assert t["report_to"] == ("voice:local" if token == VOICE_TOKEN else "app") and t["bound"] is False
    assert "anchor" not in t and "limit_conv" not in t and t["current"] is None
    r = await c.call({"op": "task_list"})
    assert r["data"]["rev"] == a.tasks.rev and [x["id"] for x in r["data"]["tasks"]] == [t["id"]]
    assert r["data"]["tz"] and r["data"]["now"] == clock.now  # an epoch the model dates a `once` with (AT3)
    r = await c.call({"op": "task_update", "task_id": t["id"], "rev": t["rev"], "title": "Jobs",
                      "schedule": {"every_min": 45}})
    assert r["ok"] and r["data"]["task"]["title"] == "Jobs" and r["data"]["task"]["rev"] == 2
    r = await c.call({"op": "task_update", "task_id": t["id"], "rev": 1, "title": "Stale"})
    assert r["code"] == "task_conflict" and r["error"] == "This task changed elsewhere. Your edit was not saved."
    r = await c.call({"op": "task_update", "task_id": t["id"], "schedule": {"every_min": 5}})
    assert r["code"] == "bad_args" and r["error"] == "every_min must be at least 15."
    r = await c.call({"op": "task_run", "task_id": t["id"]})
    assert r["ok"] and r["data"]["task"]["current"]["trigger"] == "run_now"
    task = a.tasks.tasks[t["id"]]
    await finished(a, task)
    r = await c.call({"op": "task_history", "task_id": t["id"], "limit": 3})
    assert [x["status"] for x in r["data"]["runs"]] == ["ok"] and r["data"]["runs"][0]["report"].startswith("Found 3 jobs.")
    r = await c.call({"op": "task_history", "task_id": t["id"], "limit": 21})
    assert r["code"] == "bad_args"
    r = await c.call({"op": "task_stop", "task_id": t["id"]})
    assert r["ok"] and r["data"] == {"task": r["data"]["task"], "stopped": False}
    r = await c.call({"op": "task_continue", "task_id": t["id"]})
    assert r["code"] == "bad_args"
    r = await c.call({"op": "task_run", "task_id": "t000000"})
    assert r["code"] == "task_not_found"
    r = await c.call({"op": "task_delete", "task_id": t["id"], "rev": 1})
    assert r["code"] == "task_conflict"
    r = await c.call({"op": "task_delete", "task_id": t["id"]})
    assert r["ok"] and r["data"] == {"deleted": True}
    r = await c.call({"op": "task_delete", "task_id": t["id"]})
    assert r["ok"] and r["data"] == {"deleted": False}
    check(a)
    await c.close()


async def test_chat_tokens_may_not_use_task_ops_and_notices_go_to_app_clients(env):
    a, _ = env
    chat = await a.client(CHAT_TOKEN, [CHAT])
    r = await chat.call({"op": "task_list"})
    assert r["code"] == "forbidden"
    app = await a.client(APP_TOKEN)
    voice = await a.client(VOICE_TOKEN)
    task = await create(a, VOICE, schedule=SCHEDULE)
    await until(lambda: app_notices(app, "tasks_changed"))
    n = app_notices(app, "tasks_changed")[0]
    assert isinstance(n, s.TasksChanged) and n.rev == a.tasks.rev and n.task_id == task.id and n.deleted is None
    await a.tasks.call(VOICE, "task_delete", {"task_id": task.id})
    await until(lambda: len(app_notices(app, "tasks_changed")) == 2)
    assert app_notices(app, "tasks_changed")[1].deleted is True
    assert not voice.notices and not chat.notices
    for c in (chat, app, voice):
        await c.close()


async def test_app_status_all_hides_task_conversations(env):
    a, clock = env
    task = await create(a, VOICE, schedule={"in_min": 1})
    clock.advance(61)
    await settle(a)
    await finished(a, task)
    assert task.conv in a.svc.convs
    app = await a.client(APP_TOKEN)
    r = await app.call({"op": "status"})
    assert task.conv not in [c["conv"] for c in r["data"]["conversations"]]
    r = await app.call({"op": "cancel", "conv": task.conv})  # addressable by name still (stop a run)
    assert r["ok"] and r["data"]["stopped"] is False
    await app.close()


# ----------------------------------------------------------------------------- origin rules


async def test_chat_origin_sees_only_its_task_and_a_run_turn_cannot_mutate(env):
    a, clock = env
    mine = await create(a, IN_CHAT, schedule={"in_min": 1})
    other = await create(a, tk.Origin(conv=OTHER_CHAT, kind="chat"), title="Other", schedule=SCHEDULE)
    voice_task = await create(a, VOICE, title="V", schedule=SCHEDULE)
    listed = await a.tasks.call(IN_CHAT, "task_list", {})
    assert [t["id"] for t in listed["tasks"]] == [mine.id]
    assert len((await a.tasks.call(VOICE, "task_list", {}))["tasks"]) == 3
    assert len((await a.tasks.call(APP, "task_list", {}))["tasks"]) == 3
    for tid in (other.id, voice_task.id):
        with pytest.raises(tk.TaskApiError) as e:
            await a.tasks.call(IN_CHAT, "task_history", {"task_id": tid})
        assert e.value.code == "task_not_found"
    r = await a.tasks.call(IN_CHAT, "task_update", {"paused": True})  # no task_id: the bound task
    assert r["task"]["id"] == mine.id and r["task"]["state"] == "paused"
    await a.tasks.call(IN_CHAT, "task_update", {"paused": False})
    # during a run the model's task tools go through the conversation's origin and cannot mutate
    a.scripts[CHAT] = lambda text: slow("w", n=100, delay=0.01)
    clock.advance(61)
    await settle(a)
    c = a.svc.convs[CHAT]
    await until(lambda: CHAT in a.brains and a.brain(CHAT).asks)
    assert c._unattended() is True and "outward_sends" not in c._capabilities()
    with pytest.raises(tk.TaskApiError) as e:
        await c._task_api("task_update", {"title": "x"})
    assert e.value.code == "forbidden" and e.value.message == tk.NOT_NOW
    assert [t["id"] for t in (await c._task_api("task_list", {}))["tasks"]] == [mine.id]
    await finished(a, mine)
    assert c._unattended() is False and "outward_sends" in c._capabilities()


async def test_idempotent_create_and_the_task_limit(env):
    a, clock = env
    one = await create(a, VOICE, schedule=SCHEDULE)
    two = await create(a, VOICE, schedule=SCHEDULE)
    assert one is two and len(a.tasks.tasks) == 1  # the same create over a lossy link
    three = await create(a, VOICE, title="Other", schedule=SCHEDULE)
    assert three is not one
    clock.advance(121)
    four = await create(a, VOICE, schedule=SCHEDULE)
    assert four is not one and len(a.tasks.tasks) == 3
    for i in range(tk.MAX_TASKS - 3):
        await create(a, VOICE, title=f"T{i}", schedule=SCHEDULE)
    with pytest.raises(tk.TaskApiError) as e:
        await create(a, VOICE, title="Too many", schedule=SCHEDULE)
    assert e.value.code == "task_limit"


async def test_validation_of_title_description_and_kind(env):
    a, _ = env
    for args, words in [({"title": "", "description": "x", "schedule": SCHEDULE}, "title must not be empty"),
                        ({"title": "x" * 61, "description": "x", "schedule": SCHEDULE}, "at most 60"),
                        ({"title": "t", "description": "", "schedule": SCHEDULE}, "description must not be empty"),
                        ({"title": "t", "description": "d", "schedule": SCHEDULE, "kind": "job"}, "kind must be"),
                        ({"title": 3, "description": "d", "schedule": SCHEDULE}, "title must be text")]:
        with pytest.raises(tk.TaskApiError, match=words):
            await a.tasks.call(VOICE, "task_create", args)
    with pytest.raises(tk.TaskApiError, match="Nothing to change"):
        await a.tasks.call(VOICE, "task_update", {"task_id": (await create(a, VOICE, schedule=SCHEDULE)).id})


# ----------------------------------------------------------------------------- store


async def test_store_round_trip_and_rev_bumps_only_on_visible_changes(env):
    a, clock = env
    task = await create(a, VOICE, schedule={"every_min": 15})
    rev0, store0 = task.rev, a.tasks.rev
    await until(lambda: tasks_json(a)["rev"] == store0)
    on_disk = tasks_json(a)
    assert on_disk["rev"] == store0 and tk.TaskRecord.from_dict({**on_disk["tasks"][task.id], "id": task.id}) == task
    await a.tasks.call(VOICE, "task_update", {"task_id": task.id, "paused": True})
    assert task.rev == rev0 + 1 and a.tasks.rev == store0 + 1
    clock.advance(901)  # a slot passes while paused: the anchor advances, nothing else
    await settle(a)
    await until(lambda: tasks_json(a)["tasks"][task.id]["anchor"] == clock.now - 1)
    assert task.rev == rev0 + 1 and a.tasks.rev == store0 + 1 and tasks_json(a)["rev"] == store0 + 1


async def test_corrupt_store_is_moved_aside_and_partial_history_line_skipped(short_root):
    clock = Clock()
    state = short_root / "mac" / "agent" / "state"
    state.mkdir(parents=True)
    (state / "tasks.json").write_text("{not json")
    a = await agent(short_root, clock)
    try:
        assert (state / "tasks.json.corrupt").exists() and a.tasks.tasks == {}
        task = await create(a, VOICE, schedule={"in_min": 1})
        clock.advance(61)
        await settle(a)
        await finished(a, task)
        path = a.tasks.task_dir(task.id) / "runs.jsonl"
        with path.open("a") as f:
            f.write('{"run_id": "r2", "stat')  # a crash mid-append
        r = await a.tasks.call(VOICE, "task_history", {"task_id": task.id})
        assert [x["run_id"] for x in r["runs"]] == ["r1"]
        await a.tasks.call(VOICE, "task_update", {"task_id": task.id, "schedule": {"in_min": 1}})
        clock.advance(61)
        await settle(a)
        await finished(a, task)
        assert [x["run_id"] for x in runs_file(a, task.id)] == ["r1", "r2"]
        assert (await a.tasks.call(VOICE, "task_history", {"task_id": task.id}))["runs"][0]["run_id"] == "r2"
    finally:
        await stop(a)


async def test_save_soon_coalesces(env):
    a, _ = env
    n = [0]
    orig = a.tasks._store.save

    async def counting(snapshot):
        n[0] += 1
        await orig(snapshot)

    a.tasks._store.save = counting  # type: ignore[method-assign]
    for _ in range(5):
        a.tasks.save_soon()
    await until(lambda: n[0] >= 1 and not a.tasks._save_pending and all(t.done() for t in a.tasks._helpers))
    assert n[0] == 1


async def test_history_rotation_keeps_the_last_200(env):
    a, _ = env
    task = await create(a, VOICE, schedule=SCHEDULE)
    for i in range(205):
        await a.tasks._append_history(task, tk.RunRecord(run_id=f"r{i}", n=i, trigger="schedule", status="ok"))
    rows = runs_file(a, task.id)
    assert len(rows) == 200 and rows[0]["run_id"] == "r5" and rows[-1]["run_id"] == "r204"
    assert (a.tasks.task_dir(task.id) / "runs.jsonl").stat().st_mode & 0o777 == 0o600


# ----------------------------------------------------------------------------- logs (I12)


async def test_no_task_text_reaches_the_logs(env, caplog):
    a, clock = env
    caplog.set_level(logging.DEBUG)
    a.scripts[CHAT] = lambda text: [*slow("w", n=10, delay=0.005), *report("SENTINEL-REPORT")]
    task = await create(a, IN_CHAT, description="SENTINEL-DESC", schedule={"in_min": 1})
    (a.tasks.task_dir(task.id)).mkdir(parents=True)
    (a.tasks.task_dir(task.id) / "notes.md").write_text("SENTINEL-NOTES")
    c = await a.client(CHAT_TOKEN, [CHAT])
    clock.advance(61)
    await settle(a)
    await until(lambda: CHAT in a.brains and a.brain(CHAT).asks)
    await c.call({"op": "steer", "conv": CHAT, "text": "SENTINEL-STEER"})
    await finished(a, task)
    await a.tasks.call(VOICE, "task_delete", {"task_id": task.id})
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert text  # something was logged (ids, states)
    for word in ("SENTINEL-REPORT", "SENTINEL-DESC", "SENTINEL-NOTES", "SENTINEL-STEER"):
        assert word not in text
    await c.close()
