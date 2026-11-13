"""Scheduled tasks: schedule validation, slot math (DST, zones) and `schedule_text` (DESIGN 4)."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

from agentd import tasks as tk

NOW = datetime(2026, 11, 13, 10, 0).timestamp()  # Fri 13 Nov 2026, 10:00 local


@pytest.fixture
def new_york(monkeypatch):
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.delenv("TZ")
    time.tzset()


def local(y, mo, d, h, mi) -> float:
    return datetime(y, mo, d, h, mi).timestamp()


def utc(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%d %H:%M")


# ----------------------------------------------------------------------------- validation


def test_exactly_one_form_and_in_min_becomes_once():
    assert tk.validate_schedule({"in_min": 2}, NOW) == {"once": tk.fmt_once(NOW + 120)}
    assert tk.validate_schedule({"every_min": 30}, NOW) == {"every_min": 30}
    assert tk.validate_schedule({"at": ["18:30", "08:00"], "days": ["fri", "mon"]}, NOW) == \
        {"at": ["08:00", "18:30"], "days": ["mon", "fri"]}
    assert tk.validate_schedule({"at": ["08:00"], "days": list(tk.DAYS)}, NOW) == {"at": ["08:00"]}
    assert tk.validate_schedule({"once": "2026-11-14T08:00"}, NOW) == {"once": "2026-11-14T08:00"}
    assert tk.validate_schedule({"once": "2026-11-13T09:59", "days": None}, NOW)["once"]  # 1 min ago: runs now


@pytest.mark.parametrize("raw,words", [
    ({}, "exactly one"), ({"in_min": 1, "every_min": 30}, "exactly one"), ("08:00", "object"),
    ({"every_min": 14}, "every_min must be at least 15"), ({"every_min": 10081}, "at most"),
    ({"every_min": "30"}, "whole number"), ({"every_min": True}, "whole number"),
    ({"in_min": 0}, "at least 1"), ({"in_min": 1441}, "at most"),
    ({"once": "tomorrow"}, "local time"), ({"once": "2026-11-13T09:00"}, "That time has passed (it is now Fri 13 Nov 10:00 AM)."),
    ({"once": "2028-01-01T08:00"}, "within a year"),
    ({"at": []}, "1 to 12"), ({"at": ["8:00"]}, "1 to 12"), ({"at": ["25:00"]}, "1 to 12"),
    ({"at": ["08:00"] * 13}, "1 to 12"), ({"at": ["08:00"], "days": ["monday"]}, "days must be"),
    ({"at": ["08:00"], "days": []}, "days must be"), ({"every_min": 30, "days": ["mon"]}, "days go with at"),
])
def test_bad_schedules_are_plain_words(raw, words):
    with pytest.raises(tk.TaskApiError) as e:
        tk.validate_schedule(raw, NOW)
    assert e.value.code == "bad_args" and words in e.value.message


# ----------------------------------------------------------------------------- math


def test_once_next_and_prev():
    s = {"once": "2026-11-14T08:00"}
    t = local(2026, 11, 14, 8, 0)
    assert tk.next_slot(s, t - 1) == t and tk.next_slot(s, t) is None
    assert tk.prev_slot(s, t) == t and tk.prev_slot(s, t - 1) is None


def test_every_min_is_elapsed_time_from_the_anchor():
    s = {"every_min": 30}
    anchor = NOW
    assert tk.next_slot(s, anchor, anchor) == anchor + 1800
    assert tk.next_slot(s, anchor + 1800, anchor) == anchor + 3600
    assert tk.prev_slot(s, anchor + 1799, anchor) is None
    assert tk.prev_slot(s, anchor + 1800, anchor) == anchor + 1800
    assert tk.prev_slot(s, anchor + 5000, anchor) == anchor + 3600  # many slots collapse into the last


def test_at_days_next_and_prev_skip_other_days():
    s = {"at": ["08:00", "18:30"], "days": ["mon", "fri"]}
    fri_10 = NOW
    assert tk.next_slot(s, fri_10) == local(2026, 11, 13, 18, 30)
    assert tk.next_slot(s, local(2026, 11, 13, 18, 30)) == local(2026, 11, 16, 8, 0)  # Monday
    assert tk.prev_slot(s, fri_10) == local(2026, 11, 13, 8, 0)
    assert tk.prev_slot(s, local(2026, 11, 15, 12, 0)) == local(2026, 11, 13, 18, 30)  # Sunday looks back
    assert tk.prev_slot(s, local(2026, 11, 13, 8, 0)) == local(2026, 11, 13, 8, 0)  # at the instant: due


def test_dst_gap_orders_by_instant_not_hhmm(new_york):
    s = {"at": ["02:30", "03:00"]}
    midnight = local(2026, 3, 8, 0, 0)
    first = tk.next_slot(s, midnight)
    second = tk.next_slot(s, first)
    assert utc(first) == "2026-03-08 07:00" and time.strftime("%H:%M", time.localtime(first)) == "03:00"
    assert utc(second) == "2026-03-08 07:30"  # 02:30 does not exist: it moved forward to 03:30 EDT
    assert tk.prev_slot(s, second) == second and tk.prev_slot(s, second - 1) == first


def test_dst_repeat_runs_once_and_a_second_hour_creation_waits_a_day(new_york):
    s = {"at": ["01:30"]}
    n = tk.next_slot(s, local(2026, 10, 31, 12, 0))
    assert utc(n) == "2026-11-01 05:30"  # the first 01:30 (EDT)
    assert utc(tk.next_slot(s, n)) == "2026-11-02 06:30"  # not the repeated 01:30 EST an hour later
    second_hour = datetime(2026, 11, 1, 6, 15, tzinfo=UTC).timestamp()  # 01:15 EST
    assert utc(tk.next_slot(s, second_hour)) == "2026-11-02 06:30"


def test_zone_change_applies_to_derived_times(monkeypatch):
    s = {"at": ["08:00"]}
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    ny = tk.next_slot(s, datetime(2026, 11, 13, 10, 0, tzinfo=UTC).timestamp())
    monkeypatch.setenv("TZ", "Europe/London")
    time.tzset()
    london = tk.next_slot(s, datetime(2026, 11, 13, 10, 0, tzinfo=UTC).timestamp())
    monkeypatch.delenv("TZ")
    time.tzset()
    assert utc(ny) == "2026-11-13 13:00" and utc(london) == "2026-11-14 08:00"  # 10:00 UTC is 05:00 in NY


# ----------------------------------------------------------------------------- words


def test_schedule_text():
    assert tk.schedule_text({"once": "2026-11-14T08:00"}) == "Once, Sat 14 Nov at 8:00 AM"
    assert tk.schedule_text({"at": ["08:00"]}) == "Every day at 8:00 AM"
    assert tk.schedule_text({"at": ["08:00"], "days": ["mon", "tue", "wed", "thu", "fri"]}) == "Weekdays at 8:00 AM"
    assert tk.schedule_text({"at": ["10:00"], "days": ["sat", "sun"]}) == "Weekends at 10:00 AM"
    assert tk.schedule_text({"at": ["09:00", "17:00"], "days": ["mon", "fri"]}) == "Mon and Fri at 9:00 AM and 5:00 PM"
    assert tk.schedule_text({"at": ["07:15", "12:00", "23:45"], "days": ["tue", "wed", "sat"]}) == \
        "Tue, Wed and Sat at 7:15 AM, 12:00 PM and 11:45 PM"
    assert tk.schedule_text({"every_min": 30}) == "Every 30 minutes"
    assert tk.schedule_text({"every_min": 60}) == "Every hour"
    assert tk.schedule_text({"every_min": 180}) == "Every 3 hours"
    assert tk.schedule_text({"every_min": 1440}) == "Every day" and tk.schedule_text({"every_min": 2880}) == "Every 2 days"


def test_run_prompt_is_plain_and_carries_task_and_notes():
    task = tk.TaskRecord(id="t1", title="Job search", description="find 20 jobs", kind="task",
                         schedule={"at": ["08:00"]}, conv="task:t1", report_to="voice:local", bound=False,
                         created_by="voice", created_at=NOW, updated_at=NOW, anchor=NOW, runs=2,
                         last_run={"run_id": "r1", "n": 1, "status": "ok", "ended_at": NOW - 86400, "summary": "x"})
    run = tk.RunRecord(run_id="r2", n=2, trigger="catch_up", slot=NOW - 3 * 3600, started_at=NOW)
    p = tk.run_prompt(task, run, "Saved to ~/Documents/Mac/jobs.md", clock_changed=True)
    assert p.startswith('[Scheduled task "Job search", run 2. Last run: ')
    assert "This run is 3 h later than scheduled; it was not run on time." in p and "clock was changed" in p
    assert "Save long results to a file and give its path." in p  # unbound: the owner sits at this Mac
    assert "Task: find 20 jobs\nNotes from earlier runs (yours):\nSaved to ~/Documents/Mac/jobs.md" in p
    assert chr(0x2014) not in p and chr(0x2014) not in tk.SUMMARY_PROMPT and chr(0x2014) not in tk._RUN_RULES_HEAD
    again = tk.run_prompt(task, tk.RunRecord(run_id="r3", n=3, trigger="retry", error_kind="network"), "", False)
    assert "The previous attempt failed (network); try again." in again and "none yet" in again
