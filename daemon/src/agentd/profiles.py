"""Per-chat profiles (ARCHITECTURE W9, agentd-api §4.6): the store, the never-set and the seed.

agentd is the only writer of `~/.mac-agent/agent/state/profiles.db`; no other package names that
file (a test greps for it). One sqlite connection, used only from one worker thread (V7), in WAL
mode with `synchronous=FULL`. Every write is one transaction that also bumps the row's `rev` and
the store-wide `profiles_rev`. The rows live in an in-memory cache after each commit: the gate,
the conversation check and the API read the cache, never the disk.

Fail closed: a database newer than this code, or a corrupt one, leaves the store unavailable
(`profiles_unavailable` for every chat op and conversation; voice is unaffected). It is never
recreated.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import os
import re
import sqlite3
import time
import tomllib
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from agentd import schema as s

log = logging.getLogger(__name__)
T = TypeVar("T")

DB_NAME = "profiles.db"
NEVER_SUBJECT = "timetable"  # trimmed and case-folded (P1)
FIXED_NEVER_NAMES = ("Timetable",)  # the gate's never_chat_names is never empty
PERSONA_MAX_CHARS = s.PERSONA_MAX_CHARS
PERSONA_MAX_LINES = s.PERSONA_MAX_LINES
ALL_ON = dict.fromkeys(s.CAPABILITIES, True)  # a new profile's default
_PREFIX = re.compile(r"^[^\W_]{1,20}$")

_SCHEMA_V1 = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE profiles (
  channel           TEXT NOT NULL,
  chat_id           TEXT NOT NULL,
  display_name      TEXT NOT NULL CHECK (length(display_name) BETWEEN 1 AND 80),
  enabled           INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0,1)),
  persona           TEXT NOT NULL DEFAULT '' CHECK (length(persona) <= 2000),
  trigger           TEXT NOT NULL CHECK (trigger IN ('all','prefix')),
  prefix            TEXT NOT NULL DEFAULT 'Mac',
  shared            INTEGER NOT NULL CHECK (shared IN (0,1)),
  default_workspace TEXT,
  capabilities      TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(capabilities)),
  confirm_files     INTEGER NOT NULL CHECK (confirm_files IN (0,1)),
  rev               INTEGER NOT NULL DEFAULT 1,
  created_at        REAL NOT NULL,
  updated_at        REAL NOT NULL,
  PRIMARY KEY (channel, chat_id)
)
"""

# ordered (version, sql) steps; a step's statements are separated by ";"
# v2 (v3.5, W10.6): `read_chat` is new and, like every capability, on by default (the user's
# all-on decision); rows whose capabilities are not a readable object are left as they are
_V2_READ_CHAT = """
UPDATE profiles SET capabilities = json_set(capabilities, '$.read_chat', json('true'))
 WHERE json_valid(capabilities) AND json_type(capabilities) = 'object'
   AND json_type(capabilities, '$.read_chat') IS NULL
"""
MIGRATIONS: list[tuple[int, str]] = [(1, _SCHEMA_V1), (2, _V2_READ_CHAT)]

_COLUMNS = ("channel", "chat_id", "display_name", "enabled", "persona", "trigger", "prefix", "shared",
            "default_workspace", "capabilities", "confirm_files", "rev", "created_at", "updated_at")
EDITABLE = ("display_name", "enabled", "persona", "trigger", "prefix", "shared", "default_workspace",
            "capabilities", "confirm_files")


class ProfileError(Exception):
    def __init__(self, code: s.ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ----------------------------------------------------------------------------- never-set (P1)


@dataclass(frozen=True)
class NeverSet:
    jids: frozenset[str] = frozenset()
    names: tuple[str, ...] = FIXED_NEVER_NAMES
    ok: bool = False  # False: neither source readable, or the config unparsable -> refuse every chat
    config: dict[str, Any] | None = None  # chatd's config, for the seed

    def blocks(self, chat_id: str) -> bool:
        return not self.ok or chat_id in self.jids


def normalize_subject(subject: str) -> str:
    """NFKC, format characters (zero-width and the like) dropped, trimmed, case-folded (N9)."""
    t = unicodedata.normalize("NFKC", subject)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")
    return " ".join(t.split()).casefold()


def is_never_subject(subject: str) -> bool:
    return normalize_subject(subject) == NEVER_SUBJECT


def read_pairing(path: Path) -> dict[str, dict[str, Any]] | None:
    """`{name: {id, members, owner_only}}`, or None when it cannot be read."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return {str(k): v for k, v in data.items() if isinstance(v, dict) and isinstance(v.get("id"), str)}


def load_never_set(config_path: Path, pairing_path: Path) -> NeverSet:
    """Blocking. chatd config `never_chats` ∪ the pairing.json JIDs named Timetable; the fixed
    name list plus config extras. Both files read-only from fixed paths."""
    cfg: dict[str, Any] | None
    try:
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
        cfg_state = "ok"
    except FileNotFoundError:
        cfg, cfg_state = None, "missing"
    except (OSError, tomllib.TOMLDecodeError):
        log.warning("profiles: chatd config unreadable; every chat is refused")
        return NeverSet()
    pairing = read_pairing(pairing_path)
    if cfg_state == "missing" and pairing is None:
        log.warning("profiles: neither chatd config nor pairing.json readable; every chat is refused")
        return NeverSet()
    jids: set[str] = set()
    names = list(FIXED_NEVER_NAMES)
    channels = (cfg or {}).get("channels")
    for sec in (channels.values() if isinstance(channels, dict) else ()):
        if not isinstance(sec, dict):
            continue
        jids.update(str(j) for j in sec.get("never_chats") or [] if isinstance(j, str))
        for n in sec.get("never_chat_names") or []:
            if isinstance(n, str) and n.strip() and n.strip().casefold() not in {x.casefold() for x in names}:
                names.append(n.strip())
    for name, entry in (pairing or {}).items():
        if is_never_subject(name):
            jids.add(entry["id"])
    return NeverSet(frozenset(jids), tuple(names), True, cfg)


# ----------------------------------------------------------------------------- validation


def _bool(name: str, v: Any) -> bool:
    if not isinstance(v, bool):
        raise ProfileError(s.ErrorCode.BAD_ARGS, f"{name} must be true or false.")
    return v


def validate(fields: dict[str, Any]) -> dict[str, Any]:
    """The W9.1 limits for editable fields. `capabilities` stays a delta of the keys given: a
    create merges it onto all on (the user's default), an update onto the row as it is inside
    the write transaction (two concurrent edits both survive). Unknown fields are bad_args."""
    out: dict[str, Any] = {}
    for k, v in fields.items():
        if k not in EDITABLE:
            raise ProfileError(s.ErrorCode.BAD_ARGS, f"{k} cannot be set.")
        if k == "display_name":
            if not isinstance(v, str) or not 1 <= len(v.strip()) <= 80 or "\x00" in v:
                raise ProfileError(s.ErrorCode.BAD_ARGS, "display_name must be 1 to 80 characters.")
            v = v.strip()
        elif k == "persona":
            if (not isinstance(v, str) or len(v) > PERSONA_MAX_CHARS or "\x00" in v
                    or len(v.splitlines()) > PERSONA_MAX_LINES):
                raise ProfileError(s.ErrorCode.BAD_ARGS, "persona must be at most 2,000 characters and 40 lines.")
        elif k == "trigger":
            if v not in ("all", "prefix"):
                raise ProfileError(s.ErrorCode.BAD_ARGS, "trigger must be all or prefix.")
        elif k == "prefix":
            if not isinstance(v, str) or not _PREFIX.match(v):
                raise ProfileError(s.ErrorCode.BAD_ARGS, "prefix must be one word of 1 to 20 letters or digits.")
        elif k == "default_workspace":
            if v is not None and (not isinstance(v, str) or not v.strip()):
                raise ProfileError(s.ErrorCode.BAD_ARGS, "default_workspace must be a path or null.")
        elif k == "capabilities":
            if not isinstance(v, dict):
                raise ProfileError(s.ErrorCode.BAD_ARGS, "capabilities must be an object.")
            caps: dict[str, bool] = {}
            for ck, cv in v.items():
                if ck not in s.CAPABILITIES:
                    raise ProfileError(s.ErrorCode.BAD_ARGS, f"unknown capability {ck}.")
                caps[ck] = _bool(ck, cv)
            v = caps
        else:
            v = _bool(k, v)
        out[k] = v
    return out


def caps_dict(p: s.Profile) -> dict[str, bool]:
    return {k: bool(getattr(p.capabilities, k)) for k in s.CAPABILITIES}


def granted(p: s.Profile | None) -> frozenset[str]:
    return frozenset(k for k, v in caps_dict(p).items() if v) if p is not None else frozenset()


def new_row(channel: str, chat_id: str, display_name: str, owner_only: bool, extra: dict[str, Any],
            now: float) -> dict[str, Any]:
    """Defaults for a new chat (api §4.6): shared = not owner_only; trigger prefix when shared;
    confirm_files = shared; capabilities all on (user decision 2026-11-10: the owner turns off
    what they want blocked). Only a fresh create defaults on; a stored row with a missing or
    corrupt capabilities value still grants nothing."""
    shared = extra.get("shared", not owner_only)
    row: dict[str, Any] = {
        "channel": channel, "chat_id": chat_id, "display_name": display_name, "enabled": True,
        "persona": "", "trigger": "prefix" if shared else "all", "prefix": "Mac", "shared": shared,
        "default_workspace": None,
        "confirm_files": shared, "rev": 1, "created_at": now, "updated_at": now}
    row.update(extra)
    row["capabilities"] = {**ALL_ON, **extra.get("capabilities", {})}
    return row


# ----------------------------------------------------------------------------- the store


def _to_profile(r: sqlite3.Row | tuple[Any, ...]) -> s.Profile:
    d = dict(zip(_COLUMNS, r, strict=True))
    try:
        raw = json.loads(d["capabilities"])
    except ValueError:
        raw = {}
    caps = s.Capabilities(**{k: raw.get(k) is True for k in s.CAPABILITIES}) if isinstance(raw, dict) \
        else s.Capabilities()
    return s.Profile(
        channel=d["channel"], chat_id=d["chat_id"], display_name=d["display_name"], enabled=bool(d["enabled"]),
        persona=d["persona"], trigger=d["trigger"], prefix=d["prefix"], shared=bool(d["shared"]),
        default_workspace=d["default_workspace"], capabilities=caps, confirm_files=bool(d["confirm_files"]),
        rev=int(d["rev"]), created_at=float(d["created_at"]), updated_at=float(d["updated_at"]))


def _row_values(row: dict[str, Any]) -> tuple[Any, ...]:
    out = []
    for c in _COLUMNS:
        v = row[c]
        if c == "capabilities":
            v = json.dumps({k: bool(v.get(k)) for k in s.CAPABILITIES}, sort_keys=True)
        elif isinstance(v, bool):
            v = int(v)
        out.append(v)
    return tuple(out)


class ProfileStore:
    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.clock = clock
        self.cache: dict[tuple[str, str], s.Profile] = {}
        self.rev = 0  # profiles_rev
        self.seeded_from: str | None = None
        self.never_jids: frozenset[str] = frozenset()  # sticky (I1): once never-listed, always
        self.error: str | None = "not opened"
        self._db: sqlite3.Connection | None = None
        self._ex = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="profiles")

    @property
    def available(self) -> bool:
        return self.error is None

    def get(self, channel: str, chat_id: str) -> s.Profile | None:
        return self.cache.get((channel, chat_id))

    def rows(self) -> list[s.Profile]:
        return sorted(self.cache.values(), key=lambda p: (p.channel, p.display_name.casefold(), p.chat_id))

    async def _run(self, fn: Callable[..., T], *args: Any) -> T:
        if self.error == "closed":
            raise ProfileError(s.ErrorCode.PROFILES_UNAVAILABLE, "Chat settings are closed.")
        return await asyncio.get_running_loop().run_in_executor(self._ex, fn, *args)

    def _need(self) -> sqlite3.Connection:
        if self.error is not None or self._db is None:
            raise ProfileError(s.ErrorCode.PROFILES_UNAVAILABLE,
                               "Chat settings can't be read right now, so WhatsApp is off. Voice still works.")
        return self._db

    # -- open and migrate ---------------------------------------------------------------
    async def open(self) -> None:
        await self._run(self._open)

    def _open(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not self.path.exists():
                os.close(os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600))
            db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        except (OSError, sqlite3.Error) as e:
            self._fail(f"cannot open: {e}")
            return
        try:
            for pragma in ("journal_mode=WAL", "synchronous=FULL", "foreign_keys=ON", "busy_timeout=2000"):
                db.execute(f"PRAGMA {pragma}")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            target = MIGRATIONS[-1][0]
            if version > target:
                db.close()
                self._fail(f"database version {version} is newer than this agentd ({target})")
                return
            if version < target:
                self._migrate(db, version, target)
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise sqlite3.DatabaseError("quick_check failed")
            self._db = db
            self._load()
            self.error = None
        except Exception as e:  # noqa: BLE001 - any bad row or file fails closed, never crashes start
            with contextlib.suppress(Exception):
                db.close()
            self._db = None
            self._fail(f"corrupt or unreadable: {e}")
            return
        for suffix in ("", "-wal", "-shm"):
            with contextlib.suppress(OSError):
                os.chmod(f"{self.path}{suffix}", 0o600)

    def _fail(self, why: str) -> None:
        self.error = why
        self._db = None
        log.error("profiles: %s; WhatsApp is off, voice is unaffected", why)

    def _migrate(self, db: sqlite3.Connection, version: int, target: int) -> None:
        if version > 0:
            bak = Path(f"{self.path}.bak-v{version}")
            if not bak.exists():
                dst = sqlite3.connect(bak)
                try:
                    db.backup(dst)
                finally:
                    dst.close()
                os.chmod(bak, 0o600)
        db.execute("BEGIN IMMEDIATE")
        try:
            for n, sql in MIGRATIONS:
                if version < n <= target:
                    for stmt in (x.strip() for x in sql.split(";")):
                        if stmt:
                            db.execute(stmt)
            db.execute(f"PRAGMA user_version = {int(target)}")
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        log.info("profiles: migrated %s from v%d to v%d", self.path.name, version, target)

    def _load(self) -> None:
        db = self._need_db()
        self.cache = {(p.channel, p.chat_id): p
                      for p in map(_to_profile, db.execute(f"SELECT {', '.join(_COLUMNS)} FROM profiles"))}
        meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
        self.rev = int(meta.get("profiles_rev", "0"))
        self.seeded_from = meta.get("seeded_from")
        raw = json.loads(meta.get("never_jids", "[]"))
        if not isinstance(raw, list) or not all(isinstance(j, str) for j in raw):
            raise ValueError("meta.never_jids is not a list of ids")
        self.never_jids = frozenset(raw)

    def _need_db(self) -> sqlite3.Connection:
        assert self._db is not None
        return self._db

    async def close(self) -> None:
        def shut() -> None:
            if self._db is not None:
                self._db.close()
                self._db = None
        if self._ex is not None:
            await self._run(shut)
            self._ex.shutdown(wait=False)
        self.error = "closed"

    # -- writes (one transaction each, bumping profiles_rev) ---------------------------------
    def _tx(self, fn: Callable[[sqlite3.Connection], T]) -> tuple[T, int]:
        db = self._need()
        db.execute("BEGIN IMMEDIATE")
        try:
            out = fn(db)
            rev = int((db.execute("SELECT value FROM meta WHERE key = 'profiles_rev'").fetchone() or ("0",))[0]) + 1
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('profiles_rev', ?)", (str(rev),))
            db.execute("COMMIT")
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                db.execute("ROLLBACK")
            raise
        self.rev = rev
        return out, rev

    async def create(self, row: dict[str, Any]) -> s.Profile:
        def go(db: sqlite3.Connection) -> s.Profile:
            if db.execute("SELECT 1 FROM profiles WHERE channel = ? AND chat_id = ?",
                          (row["channel"], row["chat_id"])).fetchone():
                raise ProfileError(s.ErrorCode.PROFILE_EXISTS, "This chat already has settings.")
            db.execute(f"INSERT INTO profiles ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})",
                       _row_values(row))
            return self._read(db, row["channel"], row["chat_id"])

        p, _ = await self._run(self._tx, go)
        self.cache[(p.channel, p.chat_id)] = p
        return p

    async def update(self, channel: str, chat_id: str, patch: dict[str, Any], *,
                     expect_rev: int | None = None) -> s.Profile:
        """`patch` (validated) applied to the row as it is inside the transaction; its
        `capabilities` is a delta merged there too (no lost update). `expect_rev`: profile_set."""
        now = self.clock()

        def go(db: sqlite3.Connection) -> s.Profile:
            cur = self._read(db, channel, chat_id)
            if expect_rev is not None and cur.rev != expect_rev:
                raise ProfileError(s.ErrorCode.PROFILE_CONFLICT,
                                   "This chat changed elsewhere. Your edit was not saved.")
            row = {**s.to_json(cur), "default_workspace": cur.default_workspace, "capabilities": caps_dict(cur)}
            row.update({k: v for k, v in patch.items() if k != "capabilities"})
            row["capabilities"] = {**caps_dict(cur), **patch.get("capabilities", {})}
            row.update(rev=cur.rev + 1, updated_at=now)
            sets = ", ".join(f"{c} = ?" for c in _COLUMNS[2:])
            db.execute(f"UPDATE profiles SET {sets} WHERE channel = ? AND chat_id = ?",
                       (*_row_values(row)[2:], channel, chat_id))
            return self._read(db, channel, chat_id)

        p, _ = await self._run(self._tx, go)
        self.cache[(p.channel, p.chat_id)] = p
        return p

    async def delete(self, channel: str, chat_id: str) -> None:
        def go(db: sqlite3.Connection) -> None:
            if db.execute("DELETE FROM profiles WHERE channel = ? AND chat_id = ?", (channel, chat_id)).rowcount == 0:
                raise ProfileError(s.ErrorCode.PROFILE_NOT_FOUND, "There are no settings for that chat.")

        await self._run(self._tx, go)
        self.cache.pop((channel, chat_id), None)

    async def add_never(self, jids: frozenset[str]) -> bool:
        """Make never-listed JIDs sticky (I1). True if any was new (a write, so rev moved)."""
        new = jids - self.never_jids
        if not new:
            return False
        merged = sorted(self.never_jids | new)

        def go(db: sqlite3.Connection) -> None:
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('never_jids', ?)", (json.dumps(merged),))

        await self._run(self._tx, go)
        self.never_jids = frozenset(merged)
        return True

    async def forget_never(self, jid: str) -> None:
        """App only (no chat path reaches this): take a JID off the sticky never-set."""
        left = sorted(self.never_jids - {jid})

        def go(db: sqlite3.Connection) -> None:
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('never_jids', ?)", (json.dumps(left),))

        await self._run(self._tx, go)
        self.never_jids = frozenset(left)

    async def bump(self) -> int:
        """A write with no row change (a row turned never-listed): the notice needs a new rev."""
        _, rev = await self._run(self._tx, lambda db: None)
        return rev

    async def mark_seeded(self, value: str) -> None:
        def go(db: sqlite3.Connection) -> None:
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('seeded_from', ?)", (value,))

        await self._run(self._tx, go)
        self.seeded_from = value

    @staticmethod
    def _read(db: sqlite3.Connection, channel: str, chat_id: str) -> s.Profile:
        r = db.execute(f"SELECT {', '.join(_COLUMNS)} FROM profiles WHERE channel = ? AND chat_id = ?",
                       (channel, chat_id)).fetchone()
        if r is None:
            raise ProfileError(s.ErrorCode.PROFILE_NOT_FOUND, "There are no settings for that chat.")
        return _to_profile(r)


# ----------------------------------------------------------------------------- the seed (W9.10)


def seed_rows(never: NeverSet, pairing: dict[str, dict[str, Any]] | None, resume_approved: str | None,
              now: float) -> list[dict[str, Any]]:
    """Interview prep from pairing.json (shared, prefix Mac, confirm_files, all capabilities,
    Resume if approved); any other JID of the old `chats` key with the defaults (all on). Never
    Timetable, a never-set JID or a subject that is timetable."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    names = {e["id"]: n.strip() for n, e in (pairing or {}).items()}
    for name, e in (pairing or {}).items():
        if normalize_subject(name) != "interview prep" or never.blocks(e["id"]) or is_never_subject(name):
            continue
        rows[("whatsapp", e["id"])] = new_row("whatsapp", e["id"], name.strip()[:80] or "Interview prep", False, {
            "shared": True, "trigger": "prefix", "prefix": "Mac", "confirm_files": True,
            "default_workspace": resume_approved}, now)
    channels = (never.config or {}).get("channels")
    for channel, sec in (channels.items() if isinstance(channels, dict) else ()):
        if not isinstance(sec, dict):
            continue
        for cid in sec.get("chats") or []:
            if not isinstance(cid, str) or (channel, cid) in rows or never.blocks(cid):
                continue
            name = names.get(cid, cid)
            if is_never_subject(name):
                continue
            owner_only = next((bool(e.get("owner_only")) for e in (pairing or {}).values() if e["id"] == cid), True)
            rows[(channel, cid)] = new_row(channel, cid, name[:80], owner_only, {}, now)
    return list(rows.values())
