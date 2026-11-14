"""Scheduled tasks: the table, the schedule math, the scheduler and the run lifecycle.

Design: Mac-private/docs/tasks/DESIGN.md. `TaskManager` is the single owner of task state: every
check and transition is one synchronous segment on the agentd loop (no await between a check and
the write it guards); a multi-await op (delete) marks the task `busy_op` first. A run is an
ordinary agentd turn on the task's conversation, so the one-turn-per-conversation rule, steering,
interrupt and restart recovery all apply. Deterministic code here only decides lifecycle, schedule
math, safety and protocol; the model turns words into the structured schedule and writes the
reports. Nothing the model or the owner wrote (description, notes, report, steers) is ever logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import re
import shutil
import time
from collections.abc import Awaitable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentd import schema as s
from agentd.records import _build
from agentd.stores import JsonStore, atomic_write_text, read_json
from macd.config import TasksConfig
from macd.contracts import TASK_OPS, RateLimited, TaskApiError

if TYPE_CHECKING:
    from agentd.service import Conversation, Service, Turn

log = logging.getLogger(__name__)

TICK_S = 30.0
MAX_CONCURRENT_RUNS = 2
SUMMARY_TIMEOUT_S = 120.0
DONE_KEEP_DAYS = 30
RUNS_KEEP = 200  # runs.jsonl lines kept
NOTES_MAX_CHARS = 6000
NOTES_MIN_CHARS = 10  # against an empty or "OK" answer only; short honest notes are kept (AT8)
REPORT_MAX_CHARS = 8000
MAX_TASKS = 50
RETRY_DELAY_S = 120.0
RESUME_WINDOW_S = 6 * 3600.0  # a run cut by a restart this recently is resumed
CREATE_DEDUP_S = 120.0  # the same create from the same origin within this window is one task (I13)
DELETE_WAIT_S = 15.0
RUN_MAX_TURN_S = 2700.0  # CHAT_PROFILE max_turn_s; the run waits this long plus RUN_WAIT_EXTRA_S
RUN_WAIT_EXTRA_S = 900.0
LATE_S = 120.0  # a slot served later than this is a catch_up run
CLOCK_BACK_S = 60.0  # an anchor this far in the future means the clock was set back
NOTIFY_TIMEOUT_S = 10.0
BODY_MAX_CHARS = 200
TITLE_MAX_CHARS = 60
DESCRIPTION_MAX_CHARS = 4000
ONCE_MAX_DAYS = 366
ONCE_GRACE_S = 120.0  # a once this little in the past runs now instead of being refused
HISTORY_DEFAULT = 5
HISTORY_MAX = 20
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
KINDS = ("task", "reminder")
TASK_EPOCH = "task"  # client_epoch of the turns TaskManager starts: `_recover` leaves them to it
RETRY_KINDS = frozenset({"network", "timeout", "internal"})
IDLE_STATES = frozenset({"active", "paused", "done"})
OPS = frozenset(TASK_OPS)
MUTATING_OPS = OPS - {"task_list", "task_history"}
READ_OPS = frozenset({"task_list", "task_history", "task_continue", "task_stop"})  # granted by `limit_conv`
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_MODEL = re.compile(r"^[a-z0-9][a-z0-9._:\[\]\-]{0,99}$")  # "sonnet", "opus[1m]" or a full id; no allowlist
MIN_MODEL_WINDOW = 200_000  # the smallest context window a model has: assumed until the new model is measured
EFFORTS = ("", "low", "medium", "high", "xhigh", "max")
NO_MODEL_FOR_REMINDER = "a reminder does not run Claude, so it has no model or effort."
_TRAILING_PUNCT = ".!?,;:"
# the words errors.py uses for the same kinds (UX-11): a notification never shows a code
ERROR_WORDS = {"network": "Mac could not reach Claude.", "timeout": "it took too long.",
               "internal": "something went wrong on the Mac.", "auth": "Mac is not signed in to Claude.",
               "cli_missing": "Claude Code is missing on this Mac.", "rate_limit": "the Claude usage limit was hit.",
               "model": "its model is not available. Pick another one in Settings > Tasks."}
TaskSettings = TasksConfig  # the `[tasks]` keys of config.toml (DESIGN 10)

SUMMARY_PROMPT = ("Before this task continues in a fresh session, write the notes your future self needs: "
                  "what this task has done so far, what to avoid repeating, where results are saved, open "
                  "threads, and anything the owner told you to change. Plain text, at most 60 lines. Reply "
                  "with the notes only; do not call tools.")
RESUMED_NOTE = ("Mac was restarted while working on this run and is picking it up where it left off, in the "
                "same conversation. Continue; do not start over or repeat what is already done.")
_RUN_RULES_HEAD = ("Nobody is watching. Do the work now. You cannot send, post, pay, delete or change anything "
                   "outside the Mac on your own: skip such steps and list them at the end under \"Waiting for your "
                   "OK:\".\nFinish everything in this run; do not leave background work running. ")
_RUN_RULES_UNBOUND = "Save long results to a file and give its path. "
_RUN_RULES_BOUND = ("Put the top items in the message. If the full list is in a file and it cannot be sent now, "
                    "name the file; the owner can ask for it. ")
_RUN_RULES_TAIL = ("Your reply is the report: first line = one-line summary, then what is new since the last run. "
                   "Plain text.]")
RUN_TURN_TEXT = 'Scheduled run of "{title}".'
NOT_NOW = ("Not now: a scheduled run cannot change tasks. If the owner asked you to stop, finish now with what "
           "you have; otherwise they can change the task in their next message.")
SWITCH_REFUSED = "this chat runs a task; it stays in its folder and session"


def _bad(message: str) -> TaskApiError:
    return TaskApiError(s.ErrorCode.BAD_ARGS.value, message)


@dataclass(frozen=True)
class Origin:
    """Who calls `TaskManager.call`: derived from the token kind or the conversation, never from
    a payload. `run_turn`: the caller is the model inside a scheduled run (never mutates)."""

    conv: str
    kind: str  # "voice" | "chat" | "app"
    run_turn: bool = False


# ----------------------------------------------------------------------------- schedule: validation


def validate_schedule(raw: Any, now: float) -> dict[str, Any]:
    """The tool / API input (4.1) -> the stored form: exactly one of once, at (+days), every_min.
    `in_min` becomes a `once`. Errors are plain words for the model (`bad_args`)."""
    if not isinstance(raw, dict):
        raise _bad("schedule must be an object.")
    keys = [k for k in ("in_min", "once", "at", "every_min") if raw.get(k) is not None]
    if len(keys) != 1:
        raise _bad("schedule needs exactly one of in_min, once, at (with days) or every_min.")
    key = keys[0]
    if raw.get("days") is not None and key != "at":
        raise _bad("days go with at.")
    if key == "in_min":
        n = _int(raw["in_min"], 1, 1440, "in_min")
        return {"once": fmt_once(now + n * 60)}
    if key == "once":
        text = raw["once"]
        instant = once_instant(text) if isinstance(text, str) else None
        if instant is None:
            raise _bad("once must be a local time like 2026-11-14T08:00.")
        if instant < now - ONCE_GRACE_S:
            raise _bad(f"That time has passed (it is now {fmt_when(now)}).")
        if instant > now + ONCE_MAX_DAYS * 86400:
            raise _bad("once must be within a year.")
        return {"once": fmt_once(instant)}
    if key == "every_min":
        return {"every_min": _int(raw["every_min"], 15, 10080, "every_min")}
    ats = raw["at"]
    if not isinstance(ats, list) or not ats or len(ats) > 12 or not all(isinstance(a, str) and _HHMM.match(a) for a in ats):
        raise _bad("at must list 1 to 12 times like 08:00 (24 h).")
    out: dict[str, Any] = {"at": sorted(set(ats))}
    days = raw.get("days")
    if days is not None:
        if not isinstance(days, list) or not days or not all(d in DAYS for d in days):
            raise _bad("days must be some of mon, tue, wed, thu, fri, sat, sun.")
        chosen = [d for d in DAYS if d in days]
        if len(chosen) < 7:
            out["days"] = chosen
    return out


def _int(v: Any, lo: int, hi: int, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise _bad(f"{name} must be a whole number.")
    if v < lo:
        raise _bad(f"{name} must be at least {lo}.")
    if v > hi:
        raise _bad(f"{name} must be at most {hi}.")
    return v


def once_instant(text: str) -> float | None:
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M").timestamp()  # naive local, fold=0
    except ValueError:
        return None


def fmt_once(instant: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M", time.localtime(instant))


# ----------------------------------------------------------------------------- schedule: math


def _local_instant(d: date, hhmm: str) -> float:
    h, m = hhmm.split(":")
    return datetime(d.year, d.month, d.day, int(h), int(m)).timestamp()  # fold=0: a DST repeat runs once


def _dates(around: float, forward: bool) -> list[date]:
    """Local dates to scan: a day either side, because a DST shift can move an instant across a
    midnight. Later dates always carry later instants, so the first date with a hit wins."""
    base = datetime.fromtimestamp(around).date()
    step = 1 if forward else -1
    return [base + timedelta(days=step * i - step) for i in range(10)]


def next_slot(schedule: dict[str, Any], after: float, anchor: float | None = None) -> float | None:
    """The first slot instant strictly after `after`; None when the schedule has none. Instants are
    compared, never HH:MM strings: on 2026-03-08 in New York 02:30 maps to 03:30 EDT, after 03:00."""
    if "once" in schedule:
        t = once_instant(str(schedule["once"]))
        return t if t is not None and t > after else None
    if "every_min" in schedule:
        base = after if anchor is None else anchor
        step = int(schedule["every_min"]) * 60.0
        k = max(1, math.floor((after - base) / step) + 1)
        return base + k * step
    days = schedule.get("days") or DAYS
    for d in _dates(after, True):
        if DAYS[d.weekday()] not in days:
            continue
        hits = [t for t in (_local_instant(d, a) for a in schedule["at"]) if t > after]
        if hits:
            return min(hits)
    return None


def prev_slot(schedule: dict[str, Any], before: float, anchor: float | None = None) -> float | None:
    """The last slot instant at or before `before`; None when there is none yet."""
    if "once" in schedule:
        t = once_instant(str(schedule["once"]))
        return t if t is not None and t <= before else None
    if "every_min" in schedule:
        base = before if anchor is None else anchor
        step = int(schedule["every_min"]) * 60.0
        k = math.floor((before - base) / step)
        return base + k * step if k >= 1 else None
    days = schedule.get("days") or DAYS
    for d in _dates(before, False):
        if DAYS[d.weekday()] not in days:
            continue
        hits = [t for t in (_local_instant(d, a) for a in schedule["at"]) if t <= before]
        if hits:
            return max(hits)
    return None


def fmt_clock(instant: float) -> str:
    return time.strftime("%I:%M %p", time.localtime(instant)).lstrip("0")


def fmt_day(instant: float) -> str:
    lt = time.localtime(instant)
    return f"{time.strftime('%a', lt)} {lt.tm_mday} {time.strftime('%b', lt)}"


def fmt_when(instant: float) -> str:
    return f"{fmt_day(instant)} {fmt_clock(instant)}"


def fmt_soon(instant: float, now: float) -> str:
    """A time the owner reads against today: "3:00 PM", "tomorrow at 3:00 PM", else day and time."""
    today = datetime.fromtimestamp(now).date()
    day = datetime.fromtimestamp(instant).date()
    if day == today:
        return fmt_clock(instant)
    if day == today + timedelta(days=1):
        return f"tomorrow at {fmt_clock(instant)}"
    return fmt_when(instant)


def _join(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def schedule_text(schedule: dict[str, Any]) -> str:
    """Deterministic plain words, shown everywhere the schedule is."""
    if "once" in schedule:
        t = once_instant(str(schedule["once"]))
        return f"Once, {fmt_day(t)} at {fmt_clock(t)}" if t is not None else "Once"
    if "every_min" in schedule:
        n = int(schedule["every_min"])
        if n == 1440:
            return "Every day"
        if n % 1440 == 0:
            return f"Every {n // 1440} days"
        if n == 60:
            return "Every hour"
        if n % 60 == 0:
            return f"Every {n // 60} hours"
        return f"Every {n} minutes"
    times = _join([fmt_clock(_local_instant(date(2026, 1, 5), a)) for a in schedule["at"]])
    days = schedule.get("days")
    if not days or len(days) == 7:
        head = "Every day"
    elif days == ["mon", "tue", "wed", "thu", "fri"]:
        head = "Weekdays"
    elif days == ["sat", "sun"]:
        head = "Weekends"
    else:
        head = _join([d.capitalize() for d in days])
    return f"{head} at {times}"


# ----------------------------------------------------------------------------- records


@dataclass
class RunRecord:
    run_id: str
    n: int
    trigger: str  # schedule | catch_up | run_now | continue | resume | retry
    slot: float | None = None
    queued_at: float = 0.0
    started_at: float | None = None
    ended_at: float | None = None
    status: str = "queued"  # queued | running | ok | error | stopped | interrupted | limit | skipped | missed
    ended: bool = False  # _finalize entered
    error_kind: str = ""
    turn_uid: str = ""
    report: str = ""  # never logged; runs.jsonl only
    note: str = ""  # plain words: why there is no report, or what happens next
    reset_at: float | None = None
    delivery: str = "none"  # none | pending | posted | notified | dropped

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunRecord:
        return _build(cls, d)


@dataclass
class TaskRecord:
    id: str
    title: str
    description: str
    kind: str
    schedule: dict[str, Any]
    conv: str
    report_to: str
    bound: bool
    created_by: str
    created_at: float
    updated_at: float
    anchor: float
    rev: int = 0  # the create's bump makes it 1
    state: str = "active"
    not_before: float | None = None
    runs: int = 0
    current: RunRecord | None = None
    last_run: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    limit_conv: str | None = None
    pause_reason: str = ""  # "" | "user" | "chat_off"
    model: str = ""  # "" = the brain config's model (older records load as "")
    effort: str = ""  # "" = Claude Code's default effort
    clock_changed: bool = False  # the next run prompt says so
    busy_op: str = ""  # memory only: "deleting"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("busy_op")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskRecord:
        rec = _build(cls, d)
        if isinstance(rec.current, dict):
            rec.current = RunRecord.from_dict(rec.current)
        rec.busy_op = ""
        return rec

    @property
    def once(self) -> bool:
        return "once" in self.schedule

    def next_run(self) -> float | None:
        if self.state == "done":
            return None
        return next_slot(self.schedule, self.anchor, self.anchor)

    def idle_state(self) -> str:
        """The state after a run ends (3.3): once schedules are done; a pause requested during the
        run applies now."""
        if self.once:
            return "done"
        return "paused" if self.pause_reason else "active"


def _rand_id(prefix: str) -> str:
    return prefix + os.urandom(3).hex()


def first_line(text: str, n: int = BODY_MAX_CHARS) -> str:
    for line in text.splitlines():
        line = " ".join(line.split())
        if line:
            return line if len(line) <= n else line[: n - 3].rstrip() + "..."
    return ""


def _plain(text: Any, limit: int, name: str) -> str:
    if not isinstance(text, str):
        raise _bad(f"{name} must be text.")
    out = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch >= " ").strip()
    if not out:
        raise _bad(f"{name} must not be empty.")
    if len(out) > limit:
        raise _bad(f"{name} must be at most {limit} characters.")
    return out


def _model(v: Any) -> str:
    if not isinstance(v, str):
        raise _bad("model must be text.")
    v = v.strip().lower()
    if v and not _MODEL.match(v):
        raise _bad("model must be a name like sonnet, opus or haiku, or a model id made of letters, digits, "
                   "dots and hyphens.")
    return v


def _effort(v: Any) -> str:
    if not isinstance(v, str) or v.strip().lower() not in EFFORTS:
        raise _bad("effort must be low, medium, high, xhigh or max (or empty for the default).")
    return v.strip().lower()


def _command(text: str) -> str:
    """The owner's whole message as a command key: one line, case-folded, trailing punctuation off."""
    return " ".join(text.split()).casefold().rstrip(_TRAILING_PUNCT).strip()


def run_prompt(task: TaskRecord, run: RunRecord, notes: str, clock_changed: bool) -> str:
    """The run turn's note (6.2): constant text, the task's own words and its notes. Plain. It says
    a run is late, not why (the slot may have waited behind an owner turn or a restart, AT4)."""
    last = task.last_run
    if last and last.get("ended_at"):
        last_text = f"{fmt_when(float(last['ended_at']))} ({last.get('status', '')})"
    else:
        last_text = "first run"
    head = f'[Scheduled task "{task.title}", run {run.n}. Last run: {last_text}.'
    extra = ""
    if run.trigger == "catch_up" and run.slot is not None and run.started_at is not None:
        hours = max(1, round((run.started_at - run.slot) / 3600.0))
        extra = f"This run is {hours} h later than scheduled; it was not run on time."
    elif run.trigger == "resume":
        extra = RESUMED_NOTE
    elif run.trigger == "retry":
        extra = f"The previous attempt failed ({run.error_kind or 'error'}); try again."
    elif run.trigger == "continue":
        extra = "You stopped at the Claude usage limit. Continue where you stopped; do not redo finished work."
    if clock_changed:
        extra = (extra + " " if extra else "") + "The Mac's clock was changed since the last run."
    lines = [head]
    if extra:
        lines.append(extra)
    lines.append(_RUN_RULES_HEAD + (_RUN_RULES_BOUND if task.bound else _RUN_RULES_UNBOUND) + _RUN_RULES_TAIL)
    return "\n".join(lines) + f"\nTask: {task.description}\nNotes from earlier runs (yours):\n{notes or 'none yet'}\n\n"


# ----------------------------------------------------------------------------- files (per task, in threads)


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "ab+") as f:  # binary: the tail probe must survive a crash inside a multi-byte char
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:  # a crash mid-append left a partial line: start a fresh one
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write((line + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def _read_lines(path: Path) -> list[dict[str, Any]]:
    """Every decodable line; an undecodable last line (crash mid-append) is skipped."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    out = []
    for ln in raw.splitlines():
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _rewrite_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _append_run(path: Path, row: dict[str, Any]) -> None:
    rows = _read_lines(path)
    if rows and rows[-1].get("run_id") == row.get("run_id"):
        return  # idempotent: recovery may re-enter
    if len(rows) + 1 > RUNS_KEEP:
        _rewrite_lines(path, [*rows, row][-RUNS_KEEP:])
    else:
        _append_line(path, json.dumps(row, ensure_ascii=False))


def _amend_run(path: Path, key: str, value: str, patch: dict[str, Any]) -> bool:
    """Patch the rows whose `key` ("run_id" | "turn_uid") equals `value`; atomic rewrite."""
    rows = _read_lines(path)
    hit = False
    for r in rows:
        if r.get(key) == value:
            r.update(patch)
            hit = True
    if hit:
        _rewrite_lines(path, rows)
    return hit


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _self_chat(path: Path) -> str | None:
    """The owner's own chat id from chatd's chats.json (read-only), or None."""
    rows = read_json(path, None)
    if not isinstance(rows, list):
        return None
    for r in rows:
        if isinstance(r, dict) and r.get("is_self") and isinstance(r.get("id"), str):
            return r["id"]
    return None


# ----------------------------------------------------------------------------- the manager


class TaskManager:
    def __init__(self, svc: Service, settings: TasksConfig | None = None) -> None:
        self.svc = svc
        self.settings = settings or TasksConfig()
        self.tasks: dict[str, TaskRecord] = {}
        self.by_conv: dict[str, str] = {}  # conv -> task id (at most one task per conv, I4)
        self.rev = 0
        self._store = JsonStore(svc.paths.state_dir / "tasks.json")
        self._dir = svc.paths.mac_home / "agent" / "tasks"
        self._locks: dict[str, asyncio.Lock] = {}
        self._loop_task: asyncio.Task[None] | None = None
        self._runs: dict[str, asyncio.Task[None]] = {}  # task id -> its _run coroutine
        self._helpers: set[asyncio.Task[Any]] = set()
        self._wake = asyncio.Event()
        self._save_pending = False
        self._finalizing: set[str] = set()
        self._started = False
        self.osascript: Any = self._osascript  # tests patch it

    @property
    def catch_up_s(self) -> float:
        return float(self.settings.catch_up_h) * 3600.0

    # -- lifecycle -----------------------------------------------------------------------
    async def start(self) -> None:
        """Load, recover (idempotent, 5.3) and start the scheduler loop. After `Service.start`
        loaded and recovered the conversations."""
        if self._started:
            return
        self._started = True
        raw = await self._store.load({})
        if isinstance(raw, dict):
            self.rev = int(raw.get("rev") or 0)
            for tid, d in (raw.get("tasks") or {}).items():
                if not isinstance(d, dict):
                    continue
                try:
                    rec = TaskRecord.from_dict({**d, "id": tid})
                except (TypeError, ValueError):
                    log.warning("tasks: dropping unreadable task %s", tid)  # ids only, never fields
                    continue
                if rec.conv in self.by_conv:
                    log.warning("tasks: dropping task %s: %s already has %s", tid, rec.conv, self.by_conv[rec.conv])
                    continue
                self.tasks[tid] = rec
                self.by_conv[rec.conv] = tid
        await self._recover()
        self._loop_task = asyncio.create_task(self._loop(), name="agentd:tasks")

    async def close(self) -> None:
        """Before the conversations close: the loop and every run stop; a run that was in flight
        keeps `current` running on disk and is recovered next boot (like `run_turn` on shutdown)."""
        pending = [t for t in (self._loop_task, *self._runs.values(), *self._helpers) if t is not None]
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._loop_task = None
        await self._save()

    def _spawn(self, coro: Awaitable[Any], name: str) -> asyncio.Task[Any]:
        t = asyncio.ensure_future(coro)
        t.set_name(name)
        self._helpers.add(t)
        t.add_done_callback(self._helpers.discard)
        return t

    def kick(self) -> None:
        """Wake the loop now (a conversation finished a turn, something was queued)."""
        self._wake.set()

    async def _loop(self) -> None:
        while True:
            try:
                self.tick()
            except Exception:
                log.exception("tasks: tick failed")
            delay = TICK_S
            now = self.svc.clock()
            for task in self.tasks.values():
                nxt = task.next_run()
                if nxt is not None:
                    delay = min(delay, max(0.5, nxt - now))
                if task.not_before is not None and task.state == "queued":
                    delay = min(delay, max(0.5, task.not_before - now))
            self._wake.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), delay)

    # -- persistence -----------------------------------------------------------------------
    def _snapshot(self) -> dict[str, Any]:
        return {"rev": self.rev, "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()}}

    async def _save(self) -> None:
        await self._store.save(self._snapshot)

    def save_soon(self) -> None:
        if self._save_pending:
            return
        self._save_pending = True

        async def run() -> None:
            self._save_pending = False
            await self._save()

        self._spawn(run(), "agentd:tasks:save")

    def _bump(self, task: TaskRecord, deleted: bool = False) -> None:
        """Every TaskView-visible mutation: task rev, store rev, exactly one `tasks_changed` (I6).
        The save is the caller's (awaited or `save_soon`)."""
        self.rev += 1
        if not deleted:
            task.rev += 1
            task.updated_at = self.svc.clock()
        notice: dict[str, Any] = {"ev": "tasks_changed", "rev": self.rev, "task_id": task.id}
        if deleted:
            notice["deleted"] = True
        self.svc.notify_app(notice)

    def _lock(self, task_id: str) -> asyncio.Lock:
        lock = self._locks.get(task_id)
        if lock is None:
            lock = self._locks[task_id] = asyncio.Lock()
        return lock

    def task_dir(self, task_id: str) -> Path:
        return self._dir / task_id

    def _live(self, task: TaskRecord) -> bool:
        """The record is still the table's and not being deleted (S1: a run never outlives it)."""
        return self.tasks.get(task.id) is task and not task.busy_op

    # -- views -----------------------------------------------------------------------------
    def view(self, task: TaskRecord) -> dict[str, Any]:
        d = task.to_dict()
        for k in ("anchor", "limit_conv", "clock_changed"):
            d.pop(k, None)
        if d.get("current"):
            d["current"] = {k: v for k, v in d["current"].items() if k != "report"}
        d["next_run"] = task.next_run()
        d["schedule_text"] = schedule_text(task.schedule)
        d["where"] = self._where(task)
        return d

    def _where(self, task: TaskRecord) -> str:
        """Plain words for the Tasks screen (shown verbatim, never empty)."""
        if task.bound:
            p = self.svc.profile_of(task.conv)
            return f"Posts in {p.display_name}" if p is not None and p.display_name else "Posts in WhatsApp"
        return "Notifies on this Mac"

    def _clock_data(self, now: float) -> dict[str, Any]:
        """The local clock the model needs for `once` (AT3): an epoch plus the zone name."""
        return {"now": now, "tz": time.strftime("%Z", time.localtime(now))}

    # -- recovery (5.3) --------------------------------------------------------------------
    async def _recover(self) -> None:
        svc, now = self.svc, self.svc.clock()
        uids = {t.current.turn_uid for t in self.tasks.values() if t.current is not None and t.current.turn_uid}
        for task in list(self.tasks.values()):
            try:
                await self._recover_task(task, now)
            except Exception:
                log.exception("tasks: recovery failed for %s", task.id)
        for c in list(svc.convs.values()):  # every task-epoch turn was decided above: none still runs
            r = c.record
            if r.running is not None and is_task_uid(r.running.uid):
                r.running = None
            if c.kind == "task":  # mailbox sweep: nobody reads a task conv's unacked but the manager
                keep = [u for u in r.unacked if u.get("turn_uid") in uids]
                if len(keep) != len(r.unacked):
                    r.unacked = keep
        await svc.persist()
        await self._save()

    async def _recover_task(self, task: TaskRecord, now: float) -> None:
        cur = task.current
        if cur is None:
            if task.state not in IDLE_STATES:
                task.state = task.idle_state()
            return
        if cur.status == "queued":
            task.state = "queued"
            return
        if cur.status == "limit":
            task.state = "waiting_limit"
            if not cur.ended:  # crashed between mark_limit's save and _finalize: finish the delivery now
                cur.ended, cur.ended_at = True, now
                cur.report = self.limit_message(task, task.conv if task.bound else None)
                cur.delivery = "pending" if not task.bound else "none"
                await self._complete(task, cur)
            return
        if cur.status == "running":
            c = self.svc.convs.get(task.conv)
            r = c.record if c is not None else None
            if r is not None and r.running is not None and r.running.uid == cur.turn_uid:
                r.running = None  # cut by the restart: the manager decides, then clears it (R1)
                cur.status, cur.ended, cur.ended_at = "interrupted", True, now
                cur.note = "cut by a restart"
                await self._append_history(task, cur)
                resumable = (cur.started_at is not None and now - cur.started_at <= RESUME_WINDOW_S
                             and not task.pause_reason)
                task.current, task.not_before = None, None
                if resumable:
                    self._queue(task, "resume", cur.slot)
                else:
                    task.last_run = self._summary(cur)
                    task.state = task.idle_state()
                self._bump(task)
                return
            entry = next((u for u in (r.unacked if r else []) if u.get("turn_uid") == cur.turn_uid), None)
            cur.ended = True
            if entry is not None:  # the turn ended before the crash: its answer is the report
                cur.status, cur.report = "ok", str(entry.get("answer", ""))[:REPORT_MAX_CHARS]
                cur.delivery = "posted" if task.bound else "pending"
            else:
                cur.status, cur.note = "interrupted", "ended during a restart"
            cur.ended_at = now
            task.state = "running"
            await self._complete(task, cur)
            return
        # any other final status: re-enter at step 2 (every step is idempotent)
        cur.ended = True
        task.state = "running"
        await self._complete(task, cur)

    # -- the tick (pure state transitions, no awaits) ---------------------------------------
    def tick(self) -> None:
        if hasattr(time, "tzset"):
            time.tzset()  # a zone change applies at the next tick
        now = self.svc.clock()
        dirty = False
        for task in list(self.tasks.values()):
            if task.busy_op:
                continue
            try:
                dirty = self._tick_task(task, now) or dirty
            except Exception:
                log.exception("tasks: tick failed for %s", task.id)
        dirty = self._dispatch(now) or dirty
        self._prune(now)
        if dirty:
            self.save_soon()

    def _tick_task(self, task: TaskRecord, now: float) -> bool:
        dirty = False
        if task.bound and task.state in ("active", "queued") and self.svc.profiles.available \
                and not self.svc.chat_ok(self.svc.profile_of(task.conv)):
            if task.current is not None and task.current.status == "queued":
                self._close_queued(task, "skipped", "the chat is turned off")
            task.state, task.pause_reason = "paused", "chat_off"
            self._bump(task)
            return True
        if task.anchor > now + CLOCK_BACK_S:  # the clock was set back: a slot may run again
            task.anchor, task.clock_changed = now, True
            dirty = True
        last = prev_slot(task.schedule, now, task.anchor)
        if last is None or last <= task.anchor or task.state == "done":
            return dirty
        old = task.anchor
        task.anchor = last  # many missed slots collapse into one
        first_after_old = next_slot(task.schedule, old, old)
        multiple = first_after_old is not None and first_after_old < last
        if now - last > self.catch_up_s:
            if task.kind == "reminder" and task.state in ("active", "paused"):
                if task.state == "active":
                    self._queue(task, "catch_up", last, note="missed")
                    self._bump(task)
                return True
            self._history_line(task, "missed", _missed_note(last), last)
            if task.once and task.state in ("active", "paused"):  # a live current ends through _complete (S2)
                task.state = "done"
            self._bump(task)
            return True
        trigger = "catch_up" if (now - last > LATE_S or multiple) else "schedule"
        if task.state == "active":
            self._queue(task, trigger, last)
            self._bump(task)
        elif task.state in ("running", "queued"):
            self._history_line(task, "skipped", "still running", last)
            self._bump(task)
        elif task.state == "waiting_limit":
            cur = task.current
            assert cur is not None
            if cur.reset_at is not None and now < cur.reset_at:
                self._history_line(task, "skipped", "waiting at the usage limit", last)
            else:
                self._spawn(self._amend_history(task, cur.run_id, {"status": "stopped",
                                                                   "note": "superseded by the next scheduled run"}),
                            f"agentd:tasks:{task.id}:amend")
                task.last_run = {**self._summary(cur), "status": "stopped"}
                task.current, task.not_before = None, None
                self._queue(task, trigger, last)
            self._bump(task)
        return True  # paused: the anchor advanced silently (no rev bump, no notice)

    def _queue(self, task: TaskRecord, trigger: str, slot: float | None, *, not_before: float | None = None,
               note: str = "") -> RunRecord:
        """One synchronous segment: the next run number, `current`, state `queued`."""
        task.runs += 1
        run = RunRecord(run_id=f"r{task.runs}", n=task.runs, trigger=trigger, slot=slot,
                        queued_at=self.svc.clock(), note=note)
        task.current, task.state, task.not_before = run, "queued", not_before
        self.kick()
        return run

    def _history_line(self, task: TaskRecord, status: str, note: str, slot: float | None) -> None:
        """A slot that produced no run (skipped, missed): one history line, `last_run` updated."""
        now = self.svc.clock()
        run = RunRecord(run_id=_rand_id("h"), n=task.runs, trigger="schedule", slot=slot, queued_at=now,
                        ended_at=now, status=status, ended=True, note=note)
        if status == "missed":
            task.last_run = self._summary(run)
        self._spawn(self._append_history(task, run), f"agentd:tasks:{task.id}:history")

    def _close_queued(self, task: TaskRecord, status: str, note: str) -> None:
        """A `current` that never ran (dequeued, skipped, missed): closed, history line, idle state."""
        cur = task.current
        assert cur is not None
        cur.status, cur.ended, cur.ended_at, cur.note = status, True, self.svc.clock(), note
        task.last_run = self._summary(cur)
        task.current, task.not_before = None, None
        task.state = task.idle_state()
        self._spawn(self._append_history(task, cur), f"agentd:tasks:{task.id}:history")

    @staticmethod
    def _summary(run: RunRecord) -> dict[str, Any]:
        """`last_run`: the report's first line, else the plain-words note (the app shows it verbatim
        after the result word, so the note never repeats that word, UX-13)."""
        return {"run_id": run.run_id, "n": run.n, "status": run.status, "ended_at": run.ended_at,
                "summary": first_line(run.report) or run.note}

    def _holds_slot(self, task: TaskRecord) -> bool:
        cur = task.current
        return cur is not None and (cur.status == "running" or task.id in self._finalizing)

    def _dispatch(self, now: float) -> bool:
        """FIFO by `queued_at`; a busy conversation is skipped without losing its place (I3)."""
        dirty = False
        used = sum(1 for t in self.tasks.values() if self._holds_slot(t))
        queued = sorted((t for t in self.tasks.values() if t.state == "queued" and not t.busy_op
                         and t.current is not None and t.current.status == "queued"),
                        key=lambda t: t.current.queued_at if t.current else 0.0)
        for task in queued:
            cur = task.current
            assert cur is not None
            if used >= self.settings.max_concurrent or self.svc.breaker.is_open():
                break
            if task.not_before is not None and now < task.not_before:
                continue
            if cur.slot is not None and now - cur.slot > self.catch_up_s and task.kind != "reminder":
                self._close_queued(task, "missed", _missed_note(cur.slot))
                self._bump(task)
                dirty = True
                continue
            if not self.svc.conv_idle(task.conv):
                continue
            cur.status, task.state = "running", "running"
            self._bump(task)
            dirty = True
            used += 1
            t = asyncio.create_task(self._run(task), name=f"agentd:tasks:{task.id}:{cur.run_id}")
            self._runs[task.id] = t
            t.add_done_callback(lambda _t, tid=task.id: self._runs.pop(tid, None) if self._runs.get(tid) is _t else None)
        return dirty

    def _prune(self, now: float) -> None:
        for task in list(self.tasks.values()):
            if task.state == "done" and not task.busy_op and task.last_run \
                    and now - float(task.last_run.get("ended_at") or task.updated_at) > DONE_KEEP_DAYS * 86400:
                task.busy_op = "deleting"
                self._spawn(self._delete(task), f"agentd:tasks:{task.id}:prune")

    # -- a run (6.2) --------------------------------------------------------------------------
    async def _run(self, task: TaskRecord) -> None:
        run = task.current
        if run is None or run.status != "running" or not self._live(task):
            return  # S1: a delete landed between dispatch and this first step
        svc = self.svc
        try:
            if task.kind == "reminder" and not task.bound:
                run.started_at = svc.clock()
                run.report = self._reminder_text(task, run)
                await self._finalize(task, run, "done", run.report, "")
                return
            c = svc.conversation(task.conv)
            if c.kind == "chat":
                if not svc.chat_ok(svc.profile_of(task.conv)):
                    if not svc.profiles.available:  # a store hiccup is not the owner turning the chat off
                        run.status, task.state = "queued", "queued"
                    else:
                        self._close_queued(task, "skipped", "the chat is turned off")
                        task.state, task.pause_reason = "paused", "chat_off"
                    self._bump(task)
                    await self._save()
                    return
                if c.record.paused:
                    self._close_queued(task, "skipped", "the chat is paused")
                    self._bump(task)
                    await self._save()
                    return
            if c.running() is not None:  # F1: an ask won the conversation since dispatch
                run.status, task.state = "queued", "queued"
                self._bump(task)
                self.save_soon()
                return
            text = self._reminder_text(task, run) if task.kind == "reminder" else RUN_TURN_TEXT.format(title=task.title)
            t = svc._new_turn(c, TASK_EPOCH, svc._next_internal + 1, text, "")
            t.run_id = run.run_id
            if task.kind == "reminder":
                t.canned = text
            run.started_at, run.turn_uid = svc.clock(), t.uid
            await self._save()  # the uid is on disk before the turn is (S4: a cut run is matched and resumed)
            if not self._live(task):
                return  # a delete landed during the save: the turn never starts (S1)
            if c.running() is not None:  # an ask won the conversation during the save: same as F1
                run.status, task.state, run.turn_uid = "queued", "queued", ""
                self._bump(task)
                self.save_soon()
                return
            await svc._start_turn(c, t, None)  # the prompt is built in prepare_turn, after any summary
            try:
                await asyncio.wait_for(t.done.wait(), RUN_MAX_TURN_S + RUN_WAIT_EXTRA_S)
            except TimeoutError:
                await c.finish(t, error=True, error_kind="timeout")
            await self._finalize(task, run, t.outcome or "error", "".join(t.answer), t.error_kind, torn=t.torn)
        except asyncio.CancelledError:
            raise  # shutdown or delete: the record keeps `running`; recovery decides next boot
        except Exception:
            log.exception("tasks: run failed for %s", task.id)
            with contextlib.suppress(Exception):
                await self._finalize(task, run, "error", "", "internal")

    def _reminder_text(self, task: TaskRecord, run: RunRecord) -> str:
        if run.note == "missed" and run.slot is not None:
            return f"Missed reminder from {fmt_clock(run.slot)}: {task.description}"
        return task.description

    # -- the turn hooks (6.2 step 4, 6.3, 6.4) ----------------------------------------------
    async def prepare_turn(self, c: Conversation, t: Turn, meta: s.AskMeta | None = None) -> None:
        """Replaces `_prepare_chat` for every turn on a conversation that has a task: no router,
        no route question, the task's session is never switched (R2)."""
        svc, r = self.svc, c.record
        tid = self.by_conv.get(c.conv)
        task = self.tasks.get(tid) if tid else None
        if c.kind == "task":
            await c.ensure_brain(svc.paths.voice_cwd, svc.voice_settings.setting_sources)
        else:
            c._late_note(t, meta)
            prof = svc.profile_of(c.conv)
            folder = Path(r.folder) if r.folder else svc.default_workspace(prof)
            r.fresh_folder = None  # a bound group ignores the folder of `new chat in <folder>` (R4)
            await c._enter(t, folder, "same")
        if task is not None:
            await self._maybe_summarise(c, task)
        run = task.current if task is not None else None
        if run is not None and t.run_id == run.run_id and not t.canned:
            notes = await self._read_notes(task)  # after the summary: the notes it just wrote
            t.note = run_prompt(task, run, notes, task.clock_changed)
            task.clock_changed = False

    async def _maybe_summarise(self, c: Conversation, task: TaskRecord) -> None:
        """Summarise then fresh (6.4), only between turns and only on the measured session (I10)."""
        svc, r = self.svc, c.record
        sid = r.session_id
        if not sid:  # nothing to summarise or drop: `new chat` before any session is simply done (S3)
            r.fresh_next, r.fresh_folder = False, None
            return
        ctx = task.context
        window = float(ctx.get("window") or 0) if ctx else 0.0
        over = bool(ctx and ctx.get("session") == sid and window > 0
                    and float(ctx.get("tokens") or 0) >= self.settings.summarise_ratio * window)
        if not (r.fresh_next or over):
            return
        brain = c.brain
        if not svc.breaker.is_open() and brain is not None:
            text: str | None = None
            try:
                text = await brain.summarize(SUMMARY_PROMPT, SUMMARY_TIMEOUT_S)
            except asyncio.CancelledError:
                raise
            except RateLimited as e:
                svc.breaker.trip(e.reset_at)
            except Exception as e:  # noqa: BLE001 - any failure keeps the old notes; the run never fails on it
                log.warning("tasks: summary of %s failed (%s)", task.id, type(e).__name__)
            if isinstance(text, str) and NOTES_MIN_CHARS <= len(text) <= NOTES_MAX_CHARS:
                await self._write_notes(task, text)
                log.info("tasks: %s summarised its session (notes=%d chars)", task.id, len(text))
            elif text is not None:
                log.info("tasks: %s summary of %d chars not kept; old notes stay", task.id, len(text))
        r.session_id, r.fresh_next = None, False
        r.framed.pop("", None)
        if brain is not None:
            await brain.reset()  # handoff_enabled is False: straight to a fresh session
        task.context = None
        self._bump(task)
        self.save_soon()
        svc.save_soon()

    def on_turn_done(self, conv: str, data: dict[str, Any]) -> None:
        """Every DONE on a conversation with a task: the context measurement, tied to the session."""
        tid = self.by_conv.get(conv)
        task = self.tasks.get(tid) if tid else None
        if task is None:
            return
        sid, tokens, window = data.get("session_id"), data.get("context_tokens"), data.get("context_window")
        if not sid or not _number(tokens) or not _number(window) or window <= 0:
            return
        task.context = {"session": str(sid), "tokens": int(tokens), "window": int(window)}
        self._bump(task)
        self.save_soon()

    def on_evicted(self, conv: str, uids: list[str]) -> None:
        """`add_unacked` dropped these finished turns (chatd away too long): their runs are
        `dropped`, with a history note (F12)."""
        tid = self.by_conv.get(conv)
        task = self.tasks.get(tid) if tid else None
        if task is None:
            return
        note = "not posted: too many unposted messages while WhatsApp was away"
        for uid in uids:
            cur = task.current
            if cur is not None and cur.turn_uid == uid:
                cur.delivery, cur.note = "dropped", note
                continue
            self._spawn(self._amend_history(task, uid, {"delivery": "dropped", "note": note}, key="turn_uid"),
                        f"agentd:tasks:{task.id}:evicted")

    # -- the usage limit (8) ---------------------------------------------------------------------
    async def mark_limit(self, run_id: str, reset_at: float) -> str:
        """The run hit the usage limit: state first, then the save is awaited so `limit` is on
        disk before `finish` persists the conversation (must-fix b). Returns the message text."""
        task = next((t for t in self.tasks.values() if t.current is not None and t.current.run_id == run_id), None)
        if task is None:
            return self.svc.breaker.message()
        cur = task.current
        assert cur is not None
        cur.status, cur.reset_at = "limit", reset_at
        task.state = "waiting_limit"
        self._bump(task)
        await self._save()
        return self.limit_message(task, task.conv if task.bound else None)

    def limit_message(self, task: TaskRecord, conv: str | None) -> str:
        """Plain words (8.2). In a chat (`conv`): the exact replies `limit_command` understands,
        with the chat's prefix when it answers to one. Without: the notification body (UX-9)."""
        now = self.svc.clock()
        at = fmt_soon(task.current.reset_at, now) if task.current and task.current.reset_at else "later"
        if conv is None:
            return (f'Stopped at the Claude usage limit; it resets at {at}. Say "Hey Mac, continue {task.title}", '
                    "or use Continue in Mac Settings > Tasks.")
        p = ""
        prof = self.svc.profile_of(conv)
        if prof is not None and (prof.trigger == "prefix" or prof.shared) and prof.prefix:
            p = f"{prof.prefix}, "
        return (f'"{task.title}" stopped at the Claude usage limit; it resets at {at}. Reply "{p}continue {task.title}" '
                f'and I will pick it up then, or "{p}stop {task.title}" to drop this run. Continue in Mac Settings > '
                "Tasks works too.")

    async def limit_command(self, conv: str, text: str) -> str | None:
        """While the breaker is open the model cannot read the owner's reply, so exactly the two
        replies the limit message taught are handled here (AT1): the whole message equals
        "continue <title>" or "stop <title>" (case-insensitive, trailing punctuation ignored), or
        "continue" / "stop" alone when this chat waits on one task. Anything else: None."""
        waiting = [t for t in self.tasks.values() if t.state == "waiting_limit" and conv in (t.conv, t.limit_conv)]
        if not waiting or not self.svc.breaker.is_open():
            return None
        cmd = _command(text)
        for task in waiting:
            title = _command(task.title)
            for verb in ("continue", "stop"):
                if cmd != f"{verb} {title}" and not (cmd == verb and len(waiting) == 1):
                    continue
                data = await self.call(Origin(conv=conv, kind="chat"), f"task_{verb}", {"task_id": task.id})
                if verb == "continue":
                    now = self.svc.clock()
                    when = "now" if data["starts_at"] <= now else f"at {fmt_soon(data['starts_at'], now)}"
                    return f'I will pick up "{task.title}" {when}.'
                nxt = task.next_run()
                return f'Stopped "{task.title}".' + (f" Its next run is {fmt_soon(nxt, self.svc.clock())}." if nxt else "")
        return None

    # -- finalising (6.6) ----------------------------------------------------------------------
    async def _finalize(self, task: TaskRecord, run: RunRecord, outcome: str, report: str, error_kind: str, *,
                        torn: bool = False) -> None:
        """Exactly once per run_id, only after `t.done` (I7). Each step is durable before the next."""
        if run.ended:
            return
        run.ended = True
        if not self._live(task):
            return  # C5: deliver nothing, write nothing
        self._finalizing.add(task.id)
        try:
            if run.status != "limit":
                run.status = {"done": "ok", "interrupted": "stopped"}.get(outcome, "error")
            run.ended_at = self.svc.clock()
            run.error_kind = error_kind
            run.report = report[:REPORT_MAX_CHARS]
            if run.status == "error":  # plain words, once: the notification body and the history row (UX-11, 13)
                run.note = f"Could not finish: {ERROR_WORDS.get(error_kind, error_kind or 'it failed.')}"
                if self._retry_due(task, run):
                    run.note += " Trying again in 2 minutes."
            if task.bound:
                run.delivery = "posted" if run.status != "stopped" and not torn else "none"
            else:
                run.delivery = "pending"
            await self._save()
            await self._complete(task, run)
        finally:
            self._finalizing.discard(task.id)

    @staticmethod
    def _retry_due(task: TaskRecord, run: RunRecord) -> bool:
        return run.status == "error" and run.error_kind in RETRY_KINDS and run.trigger != "retry" and not task.pause_reason

    async def _complete(self, task: TaskRecord, run: RunRecord) -> None:
        """Steps 2 to 4; recovery re-enters here before the loop exists (idempotent: `notified` is
        set before the send, the history append checks the tail)."""
        if not task.bound and run.delivery == "pending":
            run.delivery = "notified"  # at-most-once: set before sending
            await self._save()
            await self._deliver(task, run)
        await self._append_history(task, run)
        svc = self.svc
        if run.turn_uid and (not task.bound or run.status == "stopped"):
            await svc.ack(task.conv, run.turn_uid)  # never posted later by a `_resync`
        if run.status == "limit":
            task.state = "waiting_limit"
            await self._save()
            return
        task.last_run = self._summary(run)
        task.current, task.not_before = None, None
        if self._retry_due(task, run):
            r = self._queue(task, "retry", run.slot, not_before=svc.clock() + RETRY_DELAY_S)
            r.error_kind = run.error_kind
        else:
            task.state = task.idle_state()
        self._bump(task)
        await self._save()
        self.kick()

    async def _deliver(self, task: TaskRecord, run: RunRecord) -> None:
        """Unbound: an app notice, else a Mac notification (7.2); a limit message also goes to the
        owner's own chat when it has an enabled profile (8.2)."""
        if run.status == "limit":
            body = self.limit_message(task, None)
            owner = await self._owner_chat()
            if owner is not None:
                task.limit_conv = owner
                self._spawn(self._post_canned(owner, self.limit_message(task, owner), run.run_id),
                            f"agentd:tasks:{task.id}:limit")
        else:
            body = first_line(run.report) or run.note or {"ok": "Finished."}.get(run.status, "Stopped.")
        n = self.svc.notify_app({"ev": "task_report", "task_id": task.id, "run_id": run.run_id, "title": task.title,
                                 "body": body, "status": run.status, "kind": task.kind})
        if n == 0:
            await self.osascript(task.title, body)

    async def _osascript(self, title: str, body: str) -> None:
        from macd.brain.mac_tools import notify_argv

        try:
            proc = await asyncio.create_subprocess_exec(*notify_argv(title, body), stdin=asyncio.subprocess.DEVNULL,
                                                        stdout=asyncio.subprocess.DEVNULL,
                                                        stderr=asyncio.subprocess.DEVNULL)
            await asyncio.wait_for(proc.wait(), NOTIFY_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 - osascript missing or slow: history keeps the report
            log.warning("tasks: notification failed (%s)", type(e).__name__)

    async def _owner_chat(self) -> str | None:
        cid = await asyncio.to_thread(_self_chat, self.svc.paths.mac_home / "chat" / "chats.json")
        if cid is None:
            return None
        conv = f"whatsapp:{cid}"
        return conv if self.svc.chat_ok(self.svc.profile_of(conv)) else None

    async def _post_canned(self, conv: str, text: str, run_id: str) -> None:
        """A canned turn (7.1) on `conv`, after any running turn there has ended."""
        svc = self.svc
        c = svc.conversation(conv)
        while (cur := c.running()) is not None:
            await cur.done.wait()
        t = svc._new_turn(c, TASK_EPOCH, svc._next_internal + 1, text, "")
        t.canned, t.run_id = text, run_id
        await svc._start_turn(c, t, None)
        await t.done.wait()

    # -- files -------------------------------------------------------------------------------
    async def _append_history(self, task: TaskRecord, run: RunRecord) -> None:
        async with self._lock(task.id):
            await asyncio.to_thread(_append_run, self.task_dir(task.id) / "runs.jsonl", run.to_dict())

    async def _amend_history(self, task: TaskRecord, value: str, patch: dict[str, Any], *, key: str = "run_id") -> None:
        async with self._lock(task.id):
            await asyncio.to_thread(_amend_run, self.task_dir(task.id) / "runs.jsonl", key, value, patch)

    async def _read_history(self, task: TaskRecord, limit: int) -> list[dict[str, Any]]:
        async with self._lock(task.id):
            rows = await asyncio.to_thread(_read_lines, self.task_dir(task.id) / "runs.jsonl")
        return list(reversed(rows))[:limit]

    async def _read_notes(self, task: TaskRecord) -> str:
        async with self._lock(task.id):
            return await asyncio.to_thread(_read_text, self.task_dir(task.id) / "notes.md")

    async def _write_notes(self, task: TaskRecord, text: str) -> None:
        async with self._lock(task.id):
            await asyncio.to_thread(atomic_write_text, self.task_dir(task.id) / "notes.md", text)

    async def _remove_dir(self, task: TaskRecord) -> None:
        async with self._lock(task.id):
            await asyncio.to_thread(shutil.rmtree, self.task_dir(task.id), True)

    # -- the API (9.1, 10) ------------------------------------------------------------------
    async def call(self, origin: Origin, op: str, args: dict[str, Any]) -> dict[str, Any]:
        if op not in OPS:
            raise _bad(f"unknown task op {op}.")
        if origin.run_turn and op in MUTATING_OPS:
            raise TaskApiError(s.ErrorCode.FORBIDDEN.value, NOT_NOW)
        if not isinstance(args, dict):
            raise _bad("arguments must be an object.")
        handler = getattr(self, "_op_" + op[len("task_"):])
        return await handler(origin, args)

    def _visible(self, origin: Origin, task: TaskRecord, op: str) -> bool:
        if origin.kind != "chat":
            return True
        if task.conv == origin.conv:
            return True
        return task.limit_conv == origin.conv and op in READ_OPS

    def _find(self, origin: Origin, args: dict[str, Any], op: str) -> TaskRecord:
        tid = args.get("task_id")
        if not tid and origin.kind == "chat":
            tid = self.by_conv.get(origin.conv)
        task = self.tasks.get(tid) if isinstance(tid, str) else None
        if task is None or not self._visible(origin, task, op):
            raise TaskApiError(s.ErrorCode.TASK_NOT_FOUND.value, "There is no such task.")
        if task.busy_op and op in MUTATING_OPS:
            raise TaskApiError(s.ErrorCode.BUSY.value, "That task is being deleted.")
        rev = args.get("rev")
        if rev is not None and op in ("task_update", "task_delete") and rev != task.rev:
            raise TaskApiError(s.ErrorCode.TASK_CONFLICT.value, "This task changed elsewhere. Your edit was not saved.")
        return task

    async def _op_list(self, origin: Origin, args: dict[str, Any]) -> dict[str, Any]:
        tasks = [self.view(t) for t in self.tasks.values() if self._visible(origin, t, "task_list")]
        tasks.sort(key=lambda v: (v["next_run"] is None, v["next_run"] or 0.0, v["created_at"]))
        return {"rev": self.rev, **self._clock_data(self.svc.clock()), "tasks": tasks}

    async def _op_create(self, origin: Origin, args: dict[str, Any]) -> dict[str, Any]:
        now = self.svc.clock()
        title = _plain(args.get("title"), TITLE_MAX_CHARS, "title").replace("\n", " ")
        description = _plain(args.get("description"), DESCRIPTION_MAX_CHARS, "description")
        kind = args.get("kind") or "task"
        if kind not in KINDS:
            raise _bad("kind must be task or reminder.")
        schedule = validate_schedule(args.get("schedule"), now)
        model = _model(args["model"]) if args.get("model") is not None else ""
        effort = _effort(args["effort"]) if args.get("effort") is not None else ""
        if kind == "reminder" and (model or effort):
            raise _bad(NO_MODEL_FOR_REMINDER)
        bound = origin.kind == "chat"
        for t in self.tasks.values():  # I13: the same create over a lossy link is one task (the schedule may
            same_origin = t.created_by == origin.kind and t.bound == bound and (not bound or t.conv == origin.conv)
            if (same_origin and t.title == title and t.description == description and t.kind == kind  # differ by a
                    and now - t.created_at <= CREATE_DEDUP_S):  # minute when in_min crossed one, N2)
                return {"task": self.view(t), **self._clock_data(now)}
        if bound and (existing := self.tasks.get(self.by_conv.get(origin.conv, ""))) is not None:
            raise TaskApiError(s.ErrorCode.TASK_EXISTS.value,
                               f'This chat already runs "{existing.title}". Change that one, delete it first, or create '
                               "the new task in another chat.")
        if len(self.tasks) >= MAX_TASKS:
            raise TaskApiError(s.ErrorCode.TASK_LIMIT.value, f"Mac keeps at most {MAX_TASKS} tasks. Delete one first.")
        tid = _rand_id("t")
        while tid in self.tasks:
            tid = _rand_id("t")
        conv = origin.conv if bound else f"task:{tid}"
        report_to = origin.conv if bound else (s.VOICE_CONV if origin.kind == "voice" else "app")
        task = TaskRecord(id=tid, title=title, description=description, kind=kind, schedule=schedule, conv=conv,
                          report_to=report_to, bound=bound, created_by=origin.kind, created_at=now, updated_at=now,
                          anchor=_anchor_for(schedule, now), model=model, effort=effort)
        self.tasks[tid] = task
        self.by_conv[conv] = tid
        self._bump(task)
        await self._save()
        log.info("tasks: %s created by %s (%s, kind=%s)", tid, origin.kind, "bound" if bound else "unbound", kind)
        self.kick()
        return {"task": self.view(task), **self._clock_data(now)}

    async def _op_update(self, origin: Origin, args: dict[str, Any]) -> dict[str, Any]:
        task = self._find(origin, args, "task_update")
        now = self.svc.clock()
        patch: dict[str, Any] = {}
        if args.get("title") is not None:
            patch["title"] = _plain(args["title"], TITLE_MAX_CHARS, "title").replace("\n", " ")
        if args.get("description") is not None:
            patch["description"] = _plain(args["description"], DESCRIPTION_MAX_CHARS, "description")
        if args.get("schedule") is not None:
            patch["schedule"] = validate_schedule(args["schedule"], now)
        if args.get("model") is not None:
            patch["model"] = _model(args["model"])
        if args.get("effort") is not None:
            patch["effort"] = _effort(args["effort"])
        if task.kind == "reminder" and (patch.get("model") or patch.get("effort")):
            raise _bad(NO_MODEL_FOR_REMINDER)
        paused = args.get("paused")
        if paused is not None and not isinstance(paused, bool):
            raise _bad("paused must be true or false.")
        if not patch and paused is None:
            raise _bad("Nothing to change.")
        if patch.get("model", task.model) != task.model:
            if task.context:  # the saved window is the old model's: assume the smallest until the new one is measured
                task.context = {**task.context, "window": min(int(task.context.get("window") or 0), MIN_MODEL_WINDOW)}
            if not patch["model"]:
                self._fresh_for_default_model(task)
        for k, v in patch.items():
            setattr(task, k, v)
        if "schedule" in patch:
            task.anchor = _anchor_for(task.schedule, now)
            if task.state == "done":
                task.state = "active"
        if paused is True:
            task.pause_reason = task.pause_reason or "user"
            if task.state == "active":
                task.state = "paused"
        elif paused is False:
            task.pause_reason = ""
            if task.state == "paused":
                task.state = "active"
        self._bump(task)
        await self._save()
        self.kick()
        return {"task": self.view(task), **self._clock_data(now)}

    def _fresh_for_default_model(self, task: TaskRecord) -> None:
        """Claude Code resumes a session on the model it last used, so going back to the default
        model needs a new session: the conversation's next turn starts one (a task's own turn
        summarises the old one into the notes first, 6.4)."""
        c = self.svc.convs.get(task.conv)
        if c is not None and c.record.session_id:
            c.record.fresh_next = True
            self.svc.save_soon()

    async def _op_delete(self, origin: Origin, args: dict[str, Any]) -> dict[str, Any]:
        try:
            task = self._find(origin, args, "task_delete")
        except TaskApiError as e:
            if e.code == s.ErrorCode.TASK_NOT_FOUND.value:
                return {"deleted": False}  # I13: repeating a delete is benign
            raise
        task.busy_op = "deleting"  # before the first await: tick and every op leave it alone
        await self._delete(task)
        return {"deleted": True}

    async def _delete(self, task: TaskRecord) -> None:
        """C5, in this order: the run turn is torn and finished within a bound; the record goes
        first (a crash after that leaves only a harmless conversation); then the conversation
        (unbound only) and the task dir."""
        svc = self.svc
        try:
            cur = task.current
            c = svc.convs.get(task.conv)
            if cur is not None and cur.status == "running" and c is not None:
                t = next((x for x in c.turns.values() if x.run_id == cur.run_id), None)  # by run id, S1
                if t is not None and not t.finished:
                    t.torn = True  # the partial answer never enters unacked
                    await c.interrupt(t)
                    try:
                        await asyncio.wait_for(t.done.wait(), DELETE_WAIT_S)
                    except TimeoutError:
                        c.emit(s.DoneEvent, t, data=s.DoneData(duration_s=round(time.monotonic() - t.started, 3),
                                                               interrupted=True))
                        await c.finish(t, interrupted=True)
            run_task = self._runs.get(task.id)
            if run_task is not None and run_task is not asyncio.current_task():
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(asyncio.shield(run_task), 5.0)
            self.tasks.pop(task.id, None)
            if self.by_conv.get(task.conv) == task.id:
                self.by_conv.pop(task.conv, None)
            await self._save()
            self._bump(task, deleted=True)
            if task.bound and task.model:
                self._fresh_for_default_model(task)  # the group goes back to its own model
            if not task.bound:
                c = svc.convs.pop(task.conv, None)
                if c is not None:
                    for x in list(c.tasks):
                        x.cancel()
                    await c.close_brain()
                await svc.persist()
            await self._remove_dir(task)
            self._locks.pop(task.id, None)
            log.info("tasks: %s deleted", task.id)
        finally:
            task.busy_op = ""

    async def _op_run(self, origin: Origin, args: dict[str, Any]) -> dict[str, Any]:
        task = self._find(origin, args, "task_run")
        if task.state in ("running", "queued"):
            raise TaskApiError(s.ErrorCode.BUSY.value, "It is already running.")
        if task.state == "waiting_limit":
            raise _bad("It is waiting at the Claude usage limit. Continue it or stop it.")
        self._queue(task, "run_now", None)
        self._bump(task)
        await self._save()
        return {"task": self.view(task)}

    async def _op_continue(self, origin: Origin, args: dict[str, Any]) -> dict[str, Any]:
        task = self._find(origin, args, "task_continue")
        cur = task.current
        now = self.svc.clock()
        if cur is not None and cur.trigger == "continue" and task.state in ("queued", "running"):
            return {"task": self.view(task), "starts_at": task.not_before or cur.started_at or now}  # idempotent
        if task.state != "waiting_limit" or cur is None:
            raise _bad("Nothing to continue: this task is not waiting at the usage limit.")
        starts_at = max(now, cur.reset_at or 0.0)
        task.last_run = self._summary(cur)
        self._queue(task, "continue", None, not_before=starts_at)
        self._bump(task)
        await self._save()
        return {"task": self.view(task), "starts_at": starts_at}

    async def _op_stop(self, origin: Origin, args: dict[str, Any]) -> dict[str, Any]:
        task = self._find(origin, args, "task_stop")
        cur = task.current
        if cur is None:
            return {"task": self.view(task), "stopped": False}
        if task.state == "waiting_limit":
            await self._amend_history(task, cur.run_id, {"status": "stopped", "note": "stopped by the owner"})
            task.last_run = {**self._summary(cur), "status": "stopped"}
            task.current, task.not_before = None, None
            task.state = task.idle_state()
            self._bump(task)
            await self._save()
            return {"task": self.view(task), "stopped": True}
        if task.state == "queued":
            self._close_queued(task, "skipped", "stopped before it started")
            self._bump(task)
            await self._save()
            return {"task": self.view(task), "stopped": True}
        c = self.svc.convs.get(task.conv)
        t = next((x for x in c.turns.values() if x.run_id == cur.run_id), None) if c is not None else None
        stopped = await c.interrupt(t) if c is not None and t is not None else False
        return {"task": self.view(task), "stopped": bool(stopped)}

    async def _op_history(self, origin: Origin, args: dict[str, Any]) -> dict[str, Any]:
        task = self._find(origin, args, "task_history")
        limit = args.get("limit", HISTORY_DEFAULT)
        if limit is None:
            limit = HISTORY_DEFAULT
        limit = _int(limit, 1, HISTORY_MAX, "limit")
        return {"runs": await self._read_history(task, limit)}


def _missed_note(slot: float) -> str:
    return f"Not run on time: the {fmt_clock(slot)} slot was missed."


def _number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _anchor_for(schedule: dict[str, Any], now: float) -> float:
    """Creation or a schedule edit: `now`, except a `once` already due (within the grace) sits just
    behind its instant so the next tick fires it."""
    if "once" in schedule:
        t = once_instant(str(schedule["once"]))
        if t is not None and t <= now:
            return t - 1.0
    return now


def is_task_uid(uid: str) -> bool:
    parts = uid.rsplit(":", 2)
    return len(parts) == 3 and parts[1] == TASK_EPOCH


__all__ = ["Origin", "RunRecord", "TaskApiError", "TaskManager", "TaskRecord", "TaskSettings", "fmt_soon", "fmt_when",
           "next_slot", "prev_slot", "schedule_text", "validate_schedule"]
