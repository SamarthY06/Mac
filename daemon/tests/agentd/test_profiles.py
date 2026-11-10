"""Per-chat profiles (ARCHITECTURE W9, W7.6): store, never-set, seed, who may write, live grants,
notices, teardown, persona, capabilities and the chat's folder. Every test runs on a tmp HOME;
the real ~/.mac-agent is never opened."""

from __future__ import annotations

import asyncio
import sqlite3
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from agentd_kit import (
    APP_TOKEN,
    CHAT,
    CHAT_TOKEN,
    OTHER_CHAT,
    confirm_step,
    make_agent,
    slow,
    stop_agent,
    write_chat_config,
)

from agentd import profiles as pf
from agentd import schema as s
from macd.brain import fake_brain as fb

CH, CID = CHAT.split(":", 1)
PAIRING = {"Timetable ": {"id": "tt@g.us", "members": 3, "owner_only": False},
           "Interview prep": {"id": "ip@g.us", "members": 2, "owner_only": False}}
NOW = 1_762_790_000.0


def row(chat_id: str = "1@g.us", **kw) -> dict:
    return pf.new_row("whatsapp", chat_id, kw.pop("display_name", "Book club"), kw.pop("owner_only", True), kw, NOW)


async def opened(path: Path) -> pf.ProfileStore:
    st = pf.ProfileStore(path, clock=lambda: NOW)
    await st.open()
    return st


# ----------------------------------------------------------------------------- store


async def test_store_created_empty_at_v1_with_private_files(tmp_path):
    st = await opened(tmp_path / "state" / pf.DB_NAME)
    try:
        assert st.available and st.rows() == [] and st.rev == 0
        p = await st.create(row())
        assert p.rev == 1 and st.rev == 1 and pf.granted(p) == frozenset(s.CAPABILITIES)  # new: all on
    finally:
        await st.close()
    db = sqlite3.connect(tmp_path / "state" / pf.DB_NAME)
    assert db.execute("PRAGMA user_version").fetchone()[0] == 1
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    db.close()
    for f in (tmp_path / "state").iterdir():
        assert stat.S_IMODE(f.stat().st_mode) == 0o600, f
    assert stat.S_IMODE((tmp_path / "state").stat().st_mode) == 0o700


async def test_migration_v1_to_v2_keeps_rows_and_leaves_a_backup(tmp_path, monkeypatch):
    path = tmp_path / pf.DB_NAME
    st = await opened(path)
    await st.create(row())
    await st.close()
    monkeypatch.setattr(pf, "MIGRATIONS", [*pf.MIGRATIONS, (2, "ALTER TABLE profiles ADD COLUMN note TEXT")])
    st = await opened(path)
    try:
        assert st.available and st.get("whatsapp", "1@g.us").display_name == "Book club"
    finally:
        await st.close()
    bak = Path(f"{path}.bak-v1")
    assert bak.exists() and stat.S_IMODE(bak.stat().st_mode) == 0o600
    old = sqlite3.connect(bak)
    assert old.execute("PRAGMA user_version").fetchone()[0] == 1
    old.close()
    db = sqlite3.connect(path)
    assert db.execute("PRAGMA user_version").fetchone()[0] == 2
    db.close()


async def test_newer_database_fails_closed_and_voice_still_works(short_root):
    a = await make_agent(short_root)
    await stop_agent(a)
    db = sqlite3.connect(a.paths.state_dir / pf.DB_NAME)
    db.execute("PRAGMA user_version = 99")
    db.close()
    a = await make_agent(short_root, full_caps=False)
    try:
        assert not a.svc.profiles.available
        c = await a.client(CHAT_TOKEN, [], subscribe=False)  # an empty live set
        r = await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "hi"})
        assert r["code"] == "profiles_unavailable"
        app = await a.client(APP_TOKEN)
        assert (await app.call({"op": "profile_list"}))["code"] == "profiles_unavailable"
        v = await a.client()
        await v.call({"op": "ask", "conv": s.VOICE_CONV, "turn": 1, "text": "hi"})
        await v.event("done")
    finally:
        await stop_agent(a)


async def test_corrupt_file_is_never_recreated(tmp_path):
    path = tmp_path / pf.DB_NAME
    path.write_bytes(b"this is not a database" * 100)
    st = await opened(path)
    assert not st.available and "corrupt" in (st.error or "")
    with pytest.raises(pf.ProfileError) as e:
        await st.create(row())
    assert e.value.code is s.ErrorCode.PROFILES_UNAVAILABLE
    await st.close()
    assert path.read_bytes() == b"this is not a database" * 100


async def test_missing_or_corrupt_capabilities_grant_nothing(tmp_path):
    """Only a fresh create defaults to all on; a stored row that lost its value fails closed."""
    path = tmp_path / pf.DB_NAME
    st = await opened(path)
    await st.create(row("a@g.us"))
    await st.create(row("b@g.us"))
    await st.close()
    db = sqlite3.connect(path)
    db.execute("UPDATE profiles SET capabilities = '{}' WHERE chat_id = 'a@g.us'")
    db.execute("""UPDATE profiles SET capabilities = '{"web": "yes", "chrome": 1}' WHERE chat_id = 'b@g.us'""")
    db.commit()
    db.close()
    st = await opened(path)
    try:
        assert pf.granted(st.get("whatsapp", "a@g.us")) == frozenset()
        assert pf.granted(st.get("whatsapp", "b@g.us")) == frozenset()
    finally:
        await st.close()


def test_kill_9_inside_a_write_leaves_the_previous_row(tmp_path):
    path = tmp_path / pf.DB_NAME

    async def setup() -> None:
        st = await opened(path)
        await st.create(row(persona="before"))
        await st.close()

    asyncio.run(setup())
    code = (f"import os, signal, sqlite3\n"
            f"db = sqlite3.connect({str(path)!r}, isolation_level=None)\n"
            "db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')\n"
            "db.execute('BEGIN IMMEDIATE')\n"
            "db.execute(\"UPDATE profiles SET persona = 'after', rev = rev + 1\")\n"
            "os.kill(os.getpid(), signal.SIGKILL)\n")
    proc = subprocess.run([sys.executable, "-c", code], check=False)
    assert proc.returncode == -9

    async def check() -> None:
        st = await opened(path)
        try:
            p = st.get("whatsapp", "1@g.us")
            assert p.persona == "before" and p.rev == 1
        finally:
            await st.close()

    asyncio.run(check())


def test_no_other_package_names_the_profile_database():
    root = Path(__file__).resolve().parents[3]
    hits = []
    for base in (root / "daemon" / "src", root / "bridges", root / "app"):
        for f in base.rglob("*"):
            if f.suffix in (".py", ".go", ".swift") and f.is_file() and "profiles.db" in f.read_text(errors="ignore"):
                hits.append(f.relative_to(root).as_posix())
    assert hits == ["daemon/src/agentd/profiles.py"]


def test_validation_limits():
    for bad in ({"display_name": ""}, {"display_name": "x" * 81}, {"persona": "x" * 2001},
                {"persona": "\n".join(["l"] * 41)}, {"persona": "a\x00b"}, {"prefix": "Hey Mac"},
                {"prefix": "x" * 21}, {"trigger": "some"}, {"capabilities": {"mail": True}},
                {"capabilities": {"web": 1}}, {"enabled": "yes"}, {"rev": 3}):
        with pytest.raises(pf.ProfileError) as e:
            pf.validate(bad)
        assert e.value.code is s.ErrorCode.BAD_ARGS, bad
    ok = pf.validate({"persona": "x" * 2000, "prefix": "Jarvis2", "capabilities": {"web": False}})
    assert ok["capabilities"] == {"web": False}  # a delta: merged onto all on (create) or the row (update)
    assert pf.new_row("whatsapp", "1@g.us", "B", True, ok, NOW)["capabilities"] == {
        **dict.fromkeys(s.CAPABILITIES, True), "web": False}
    assert pf.validate({"persona": "\n".join(["l"] * 40)})


# ----------------------------------------------------------------------------- never-set (P1)


def test_never_set_from_pairing_without_config(tmp_path):
    (tmp_path / "pairing.json").write_text('{"Timetable ": {"id": "tt@g.us"}, "Interview prep": {"id": "ip@g.us"}}')
    n = pf.load_never_set(tmp_path / "missing.toml", tmp_path / "pairing.json")
    assert n.ok and n.jids == {"tt@g.us"} and n.names == ("Timetable",)
    assert n.blocks("tt@g.us") and not n.blocks("ip@g.us")


def test_never_set_fails_closed(tmp_path):
    n = pf.load_never_set(tmp_path / "missing.toml", tmp_path / "missing.json")
    assert not n.ok and n.blocks("anything@g.us") and n.names == ("Timetable",)
    (tmp_path / "bad.toml").write_text("[channels\n")
    (tmp_path / "pairing.json").write_text("{}")
    assert not pf.load_never_set(tmp_path / "bad.toml", tmp_path / "pairing.json").ok


async def test_add_group_timetable_is_never_listed_and_never_reaches_a_profile(short_root):
    a = await make_agent(short_root, config=False, pairing=PAIRING)
    try:
        assert a.svc.never.names == ("Timetable",) and a.svc.never.blocks("tt@g.us")
        c = await a.client(CHAT_TOKEN, [], subscribe=False)
        conv = "whatsapp:ip@g.us"
        assert (await c.call({"op": "hello", "api": s.API_VERSION, "token": CHAT_TOKEN, "client": "x",
                              "client_epoch": "e", "conversations": []}))["code"] == "bad_args"  # hello twice
        for subject, jid in (("Timetable", "tt@g.us"), ("timetable ", "brand-new@g.us"), ("Book", "tt@g.us")):
            r = await c.call({"op": "profile_command", "conv": conv, "command": "add", "subject": subject,
                              "chat_id": jid, "owner_only": True})
            assert r["code"] == "never_listed", subject
        app = await a.client(APP_TOKEN)
        r = await app.call({"op": "profile_create", "channel": "whatsapp", "chat_id": "x@g.us",
                            "display_name": " TimeTable", "owner_only": True})
        assert r["code"] == "never_listed"
        assert {p.chat_id for p in a.svc.profiles.rows()} == {"ip@g.us"}
    finally:
        await stop_agent(a)


async def test_neither_file_readable_refuses_every_chat(short_root):
    a = await make_agent(short_root)
    try:
        (a.paths.mac_home / "chat" / "config.toml").unlink()
        c = await a.client(CHAT_TOKEN, [], subscribe=False)
        assert (await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "hi"}))["code"] == \
            "forbidden_conversation"
        assert a.svc.live_chat_convs() == []
    finally:
        await stop_agent(a)


async def _notice(c, pred, timeout: float = 3.0):
    async def wait():
        while True:
            hit = [n for n in c.notices if pred(n)]
            if hit:
                return hit[-1]
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(wait(), timeout)


async def _no_event(c, ev: str, wait: float = 0.3) -> None:
    with pytest.raises(TimeoutError):
        await c.event(ev, timeout=wait)


async def test_a_row_later_never_listed_is_refused_and_torn_down(agent):
    """I3: the row reads disabled, with a new rev and a notice; the torn turn shows nothing more."""
    agent.scripts[CHAT] = lambda t: slow(n=200, delay=0.01)
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "long"})
    await c.event("say")
    rev = agent.svc.profiles.rev
    write_chat_config(agent.paths.mac_home, [CHAT], [CHAT])  # now in never_chats
    await agent.client(CHAT_TOKEN, [], subscribe=False)  # a hello re-reads the never-set
    n = await _notice(c, lambda n: n.chat_id == CID)
    assert n.profile.enabled is False and n.rev > rev
    await _no_event(c, "done")  # I2: not even its done
    assert agent.svc.convs[CHAT].record.unacked == []  # I2: a torn turn is never re-delivered
    assert (await c.call({"op": "ask", "conv": CHAT, "turn": 2, "text": "x"}))["code"] == "forbidden_conversation"
    app = await agent.client(APP_TOKEN)
    listed = (await app.call({"op": "profile_list"}))["data"]["profiles"]
    assert next(p for p in listed if p["chat_id"] == CID)["enabled"] is False


async def test_never_set_is_sticky_once_seen(agent):
    """I1: emptying config.toml and pairing.json unblocks nothing; only the app can forget a JID."""
    write_chat_config(agent.paths.mac_home, [CHAT], [CHAT])
    await agent.svc.refresh_never()
    assert CID in agent.svc.profiles.never_jids
    write_chat_config(agent.paths.mac_home, [CHAT], [])
    await agent.svc.refresh_never()
    assert agent.svc.never.blocks(CID) and agent.svc.live_caps(CHAT) == frozenset()
    await agent.svc.profiles.forget_never(CID)
    await agent.svc.refresh_never()
    assert not agent.svc.never.blocks(CID)


async def test_unreadable_never_set_shows_rows_disabled_with_a_notice(agent):
    c = await agent.client(CHAT_TOKEN, [CHAT])
    rev = agent.svc.profiles.rev
    (agent.paths.mac_home / "chat" / "config.toml").write_text("not = [toml")
    await agent.svc.refresh_never()
    n = await _notice(c, lambda n: n.chat_id == CID)
    assert n.profile.enabled is False and n.rev > rev


async def test_pause_never_listed_from_a_chat_token_is_sticky(agent):
    """N6: chatd saw the subject turn Timetable; a chat token may send it (it only narrows)."""
    agent.scripts[CHAT] = lambda t: slow(n=200, delay=0.01)
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "long"})
    await c.event("say")
    r = await c.call({"op": "pause", "conv": CHAT, "value": True, "reason": "never_listed"})
    assert r["ok"] is True, r
    n = await _notice(c, lambda n: n.chat_id == CID)
    assert n.profile.enabled is False
    assert CID in agent.svc.profiles.never_jids
    await _no_event(c, "done")
    again = await c.call({"op": "pause", "conv": CHAT, "value": True, "reason": "never_listed"})
    assert again["ok"] is True  # idempotent, even once the chat is refused


async def test_concurrent_capability_edits_both_survive(agent):
    """CQ race: two deltas to different capabilities, applied together, both stick."""
    cur = agent.svc.profile_of(CHAT)
    await asyncio.gather(agent.svc.profiles.update(CH, CID, {"capabilities": {"web": False}}),
                         agent.svc.profiles.update(CH, CID, {"capabilities": {"chrome": False}}))
    p = agent.svc.profile_of(CHAT)
    assert pf.granted(p) == frozenset(s.CAPABILITIES) - {"web", "chrome"} and p.rev == cur.rev + 2


async def test_teardown_order_with_a_slow_interrupt(agent):
    """I2: agentd state, then the notice, then brain.interrupt (slow here): chatd has the
    notice long before the interrupt returns, and never a done."""
    agent.scripts[CHAT] = lambda t: slow(n=300, delay=0.01)
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "long"})
    await c.event("say")
    brain = agent.svc.convs[CHAT].brain
    real = brain.interrupt
    entered = asyncio.Event()

    async def slow_interrupt(turn_id):
        entered.set()
        await asyncio.sleep(1.0)
        await real(turn_id)

    brain.interrupt = slow_interrupt
    app = await agent.client(APP_TOKEN)
    p = agent.svc.profile_of(CHAT)
    task = asyncio.create_task(app.call({"op": "profile_set", "channel": CH, "chat_id": CID, "rev": p.rev,
                                         "patch": {"enabled": False}}))
    await asyncio.wait_for(entered.wait(), 2)
    n = await _notice(c, lambda n: n.chat_id == CID, timeout=0.5)  # while the interrupt is still running
    assert n.profile.enabled is False and not task.done()
    await task
    await _no_event(c, "done")
    assert agent.svc.convs[CHAT].record.unacked == []


# ----------------------------------------------------------------------------- seed (W9.10)


async def test_seed_from_pairing_gives_interview_prep_only(short_root):
    a = await make_agent(short_root, config=False, pairing=PAIRING, full_caps=False)
    try:
        (p,) = a.svc.profiles.rows()
        assert (p.channel, p.chat_id, p.display_name) == ("whatsapp", "ip@g.us", "Interview prep")
        assert p.enabled and p.persona == "" and p.shared and p.trigger == "prefix" and p.prefix == "Mac"
        assert p.confirm_files and p.default_workspace is None  # Resume is not approved here
        assert pf.granted(p) == frozenset(s.CAPABILITIES)
        assert a.svc.profiles.seeded_from is not None
        await a.svc.profiles.delete("whatsapp", "ip@g.us")
    finally:
        await stop_agent(a)
    a = await make_agent(short_root, config=False, pairing=PAIRING, full_caps=False)
    try:
        assert a.svc.profiles.rows() == []  # the seed runs once, also after the table is emptied
    finally:
        await stop_agent(a)


async def test_seed_waits_for_a_readable_never_set_and_finds_resume_by_name(short_root):
    """N4: nothing seeds (and the seed is not marked done) while the never-set is unreadable.
    The chat's folder is the one approved workspace named Resume, wherever it lives."""
    from agentd_kit import write_pairing

    a = await make_agent(short_root, config=False, full_caps=False)
    try:
        assert a.svc.profiles.rows() == [] and a.svc.profiles.seeded_from is None
        resume = a.paths.home / "Work" / "Resume"
        resume.mkdir(parents=True)
        await a.svc.workspaces.approve(resume)
    finally:
        await stop_agent(a)
    write_pairing(short_root / "mac", PAIRING)
    a = await make_agent(short_root, config=False, full_caps=False)
    try:
        (p,) = a.svc.profiles.rows()
        assert p.display_name == "Interview prep" and p.default_workspace == "~/Work/Resume"
        assert a.svc.profiles.seeded_from is not None
    finally:
        await stop_agent(a)


@pytest.mark.parametrize("subject", ["Timetable", " timetable ", "TIME\u200bTABLE", "Ｔｉｍｅｔａｂｌｅ",
                                     "time\u2060table\ufeff"])
def test_never_subject_is_normalized(subject):
    assert pf.is_never_subject(subject)


async def test_store_closed_and_bad_meta_fail_closed(tmp_path):
    st = await opened(tmp_path / pf.DB_NAME)
    await st.close()
    assert not st.available and st.error == "closed"
    with pytest.raises(pf.ProfileError):
        await st.create(row())
    db = sqlite3.connect(tmp_path / pf.DB_NAME)
    db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('never_jids', '{bad')")
    db.commit()
    db.close()
    st = await opened(tmp_path / pf.DB_NAME)
    try:
        assert not st.available and "corrupt" in (st.error or "")
    finally:
        await st.close()


def test_seed_uses_resume_when_approved_and_old_chats_read_only():
    never = pf.NeverSet(frozenset({"tt@g.us"}), ("Timetable",), True,
                        {"channels": {"whatsapp": {"chats": ["ip@g.us", "old@g.us", "tt@g.us"]}}})
    rows = {r["chat_id"]: r for r in pf.seed_rows(never, {k: v for k, v in PAIRING.items()},
                                                  "~/Desktop/Personal/Resume", NOW)}
    assert set(rows) == {"ip@g.us", "old@g.us"}
    assert rows["ip@g.us"]["default_workspace"] == "~/Desktop/Personal/Resume"
    assert rows["old@g.us"]["capabilities"] == dict.fromkeys(s.CAPABILITIES, True)  # the default: all on


# ----------------------------------------------------------------------------- who may write


async def test_chat_token_cannot_write_and_lists_only_enabled_rows(agent):
    app = await agent.client(APP_TOKEN)
    other = agent.svc.profile_of(OTHER_CHAT)
    await app.call({"op": "profile_set", "channel": CH, "chat_id": other.chat_id, "rev": other.rev,
                    "patch": {"enabled": False}})
    c = await agent.client(CHAT_TOKEN, [], subscribe=False)
    for op in ({"op": "profile_create", "channel": CH, "chat_id": "n@g.us", "display_name": "N", "owner_only": True},
               {"op": "profile_set", "channel": CH, "chat_id": CID, "rev": 1, "patch": {}},
               {"op": "profile_delete", "channel": CH, "chat_id": CID}):
        assert (await c.call(op))["code"] == "forbidden"
    data = (await c.call({"op": "profile_list"}))["data"]
    assert [p["chat_id"] for p in data["profiles"]] == [CID]
    assert len((await app.call({"op": "profile_list"}))["data"]["profiles"]) == 2


async def _ask_add(c, jid="new@g.us"):
    r = await c.call({"op": "profile_command", "conv": CHAT, "command": "add", "subject": "Book club ",
                      "chat_id": jid, "owner_only": False})
    q = await c.event("question")
    assert q["kind"] == "profile" and q["question_id"] == r["data"]["question_id"]
    assert q["text"] == "Answer in “Book club” too? Reply yes to this message."
    await c.call({"op": "question_posted", "conv": CHAT, "question_id": q["question_id"], "msg_ref": "p1"})
    return q


async def test_add_group_needs_a_bound_yes(agent):
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await _ask_add(c)
    assert (await c.call({"op": "reply", "conv": CHAT, "text": "yes"}))["data"] == {"bound": False}
    assert (await c.call({"op": "reply", "conv": CHAT, "text": "yes", "quoted_ref": "other"}))["data"]["bound"] is False
    assert agent.svc.profiles.get(CH, "new@g.us") is None
    r = await c.call({"op": "reply", "conv": CHAT, "reaction": "\U0001F44D", "reaction_on": "p1"})
    assert r["data"] == {"bound": True, "outcome": "approved"}
    st = await c.event("status", pred=lambda e: "profile_command" in e["data"])
    assert st["data"]["profile_command"]["applied"] is True
    p = agent.svc.profiles.get(CH, "new@g.us")
    assert p.display_name == "Book club" and p.shared and p.trigger == "prefix" and p.confirm_files
    assert pf.granted(p) == frozenset(s.CAPABILITIES)  # a new profile: all on; the owner turns off
    assert [n.chat_id for n in c.notices] == ["new@g.us"]


async def test_add_group_question_expires_and_a_restart_never_approves(short_root):
    a = await make_agent(short_root)
    c = await a.client(CHAT_TOKEN, [CHAT])
    q = await _ask_add(c)
    a.svc.convs[CHAT]._expire(q["question_id"])
    r = await c.call({"op": "reply", "conv": CHAT, "text": "yes", "quoted_ref": "p1"})
    assert r["data"] == {"bound": False}
    await _ask_add(c)
    await stop_agent(a)
    a = await make_agent(short_root, boot="b2")
    try:
        c = await a.client(CHAT_TOKEN, [CHAT])
        r = await c.call({"op": "reply", "conv": CHAT, "text": "yes", "quoted_ref": "p1"})
        assert r["data"] == {"bound": False} and a.svc.profiles.get(CH, "new@g.us") is None
    finally:
        await stop_agent(a)


async def test_add_yes_after_the_chat_was_never_listed_is_refused(agent):
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await _ask_add(c, "late@g.us")
    write_chat_config(agent.paths.mac_home, [CHAT, OTHER_CHAT], ["whatsapp:late@g.us"])
    await c.call({"op": "reply", "conv": CHAT, "text": "yes", "quoted_ref": "p1"})
    st = await c.event("status", pred=lambda e: "profile_command" in e["data"])
    assert st["data"]["profile_command"]["applied"] is False
    assert agent.svc.profiles.get(CH, "late@g.us") is None


async def test_allow_applies_to_the_current_row_and_disallow_at_once(agent):
    channel = CH
    await agent.svc.profiles.update(channel, CID, {"capabilities": dict.fromkeys(s.CAPABILITIES, False)})
    c = await agent.client(CHAT_TOKEN, [CHAT])
    r = await c.call({"op": "profile_command", "conv": CHAT, "command": "allow", "capability": "web"})
    q = await c.event("question")
    assert q["text"] == "Turn on the web in this chat? Reply yes to this message."
    await c.call({"op": "question_posted", "conv": CHAT, "question_id": r["data"]["question_id"], "msg_ref": "p2"})
    # the row changes meanwhile (the app turns on chrome): the yes patches the row as it is then
    await agent.svc.profiles.update(channel, CID, {"capabilities": {**dict.fromkeys(s.CAPABILITIES, False),
                                                                      "chrome": True}})
    await c.call({"op": "reply", "conv": CHAT, "text": "yes", "quoted_ref": "p2"})
    await c.event("status", pred=lambda e: "profile_command" in e["data"])
    assert pf.granted(agent.svc.profile_of(CHAT)) == {"web", "chrome"}
    r = await c.call({"op": "profile_command", "conv": CHAT, "command": "disallow", "capability": "chrome"})
    assert r["data"] == {"applied": True} and pf.granted(agent.svc.profile_of(CHAT)) == {"web"}
    assert (await c.call({"op": "profile_command", "conv": CHAT, "command": "allow",
                          "capability": "mail"}))["code"] == "bad_args"


async def test_profile_set_conflict_not_found_and_workspace_rules(agent):
    app = await agent.client(APP_TOKEN)
    p = agent.svc.profile_of(CHAT)
    r = await app.call({"op": "profile_set", "channel": CH, "chat_id": CID, "rev": p.rev + 5, "patch": {}})
    assert r["code"] == "profile_conflict"
    r = await app.call({"op": "profile_set", "channel": CH, "chat_id": "nope", "rev": 1, "patch": {}})
    assert r["code"] == "profile_not_found"
    proj = agent.paths.home / "Proj"
    proj.mkdir()
    r = await app.call({"op": "profile_set", "channel": CH, "chat_id": CID, "rev": p.rev,
                        "patch": {"default_workspace": "~/Proj"}})
    assert r["code"] == "workspace_not_approved"
    await app.call({"op": "ws_approve", "path": "~/Proj"})
    r = await app.call({"op": "profile_set", "channel": CH, "chat_id": CID, "rev": p.rev,
                        "patch": {"default_workspace": "~/Proj"}})
    assert r["data"]["profile"]["default_workspace"] == "~/Proj" and r["data"]["profile"]["rev"] == p.rev + 1
    await app.call({"op": "ws_remove", "path": "~/Proj"})
    assert agent.svc.profile_of(CHAT).default_workspace is None
    r = await app.call({"op": "profile_create", "channel": CH, "chat_id": CID, "display_name": "x",
                        "owner_only": True})
    assert r["code"] == "profile_exists"


# ----------------------------------------------------------------------------- grants and notices


async def test_live_grants_and_one_notice_per_write(agent):
    c = await agent.client(CHAT_TOKEN, [], subscribe=False)
    hello = c.frames[0]["data"]
    assert sorted(hello["conversations"]) == sorted([CHAT, OTHER_CHAT])
    app = await agent.client(APP_TOKEN)
    r = await app.call({"op": "profile_create", "channel": CH, "chat_id": "late@g.us", "display_name": "Late",
                        "owner_only": True, "capabilities": {"web": True}})
    late = r["data"]["profile"]
    assert late["trigger"] == "all" and late["confirm_files"] is False and late["capabilities"]["web"] is True
    assert (await c.call({"op": "subscribe", "conv": "whatsapp:late@g.us"}))["ok"]  # no new hello
    await app.call({"op": "profile_set", "channel": CH, "chat_id": "late@g.us", "rev": 1,
                    "patch": {"persona": "Reply in Hindi"}})
    await agent.svc.set_persona("whatsapp:late@g.us", "Short answers.")
    await app.call({"op": "profile_set", "channel": CH, "chat_id": "late@g.us", "rev": 3, "patch": {"enabled": False}})
    r = await c.call({"op": "ask", "conv": "whatsapp:late@g.us", "turn": 1, "text": "x"})
    assert r["code"] == "forbidden_conversation"
    await app.call({"op": "profile_delete", "channel": CH, "chat_id": "late@g.us"})
    await asyncio.sleep(0.05)
    revs = [n.rev for n in c.notices]
    assert len(c.notices) == 5 and revs == sorted(revs) and len(set(revs)) == 5
    assert c.notices[1].profile.persona == "Reply in Hindi" and c.notices[2].profile.persona == "Short answers."
    assert c.notices[3].profile.enabled is False and c.notices[4].deleted is True
    assert not app.notices  # chat clients only


# ----------------------------------------------------------------------------- teardown (P3)


async def test_disable_mid_turn_tears_down_before_the_notice(agent):
    agent.scripts[CHAT] = lambda t: [confirm_step(), *slow(n=100, delay=0.01)]
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "send it"})
    ev = await c.event("confirm")
    app = await agent.client(APP_TOKEN)
    p = agent.svc.profile_of(CHAT)
    await app.call({"op": "profile_set", "channel": CH, "chat_id": CID, "rev": p.rev, "patch": {"enabled": False}})
    closed = await c.event("question_closed")
    assert closed["question_id"] == ev["question_id"] and closed["outcome"] == "skipped"
    await _no_event(c, "done")  # I2: a torn turn shows nothing more, not even done
    assert agent.brain(CHAT).confirm_results == [False]  # its token was revoked
    later = [e for e in c.drain() if e["ev"] in ("say", "text", "attach")]
    assert later == []
    assert c.notices[-1].profile.enabled is False


# ----------------------------------------------------------------------------- brain policy


async def test_capabilities_are_live_and_persona_rebuilds_with_the_same_session(agent):
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "hi"})
    await c.event("done")
    first = agent.brain(CHAT)
    assert first.spec.capabilities() == frozenset(s.CAPABILITIES) and first.spec.chrome is True
    await agent.svc.profiles.update(CH, CID, {"capabilities": {**dict.fromkeys(s.CAPABILITIES, True),
                                                                "web": False}})
    assert "web" not in first.spec.capabilities()  # at the next tool call, no rebuild needed
    agent.svc.convs[CHAT].record.session_id = "sess-1"
    await agent.svc.set_persona(CHAT, "Reply in Hindi")
    await c.call({"op": "ask", "conv": CHAT, "turn": 2, "text": "and again"})
    await c.event("done")
    second = agent.brain(CHAT)
    assert second is not first and second.spec.chat_persona == "Reply in Hindi"
    assert second.spec.display_name == CID and agent.svc.convs[CHAT].record.session_id == "sess-1"
    assert await second.spec.on_set_persona("x" * 2001) is not None  # refused, nothing written
    assert agent.svc.profile_of(CHAT).persona == "Reply in Hindi"


async def test_chrome_off_and_no_files_read_means_no_router_and_a_per_chat_dir(agent):
    await agent.svc.profiles.update(CH, CID, {"capabilities": dict.fromkeys(s.CAPABILITIES, False)})
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "find jobs in the Resume folder"})
    await c.event("done")
    spec = agent.brain(CHAT).spec
    assert spec.chrome is False and spec.capabilities() == frozenset()
    assert spec.cwd == agent.paths.chat_dir(CHAT) and agent.model.calls == []


async def test_default_workspace_replaces_the_route_question(agent):
    proj = agent.paths.home / "Proj"
    proj.mkdir()
    await agent.svc.workspaces.approve(proj)
    await agent.svc.profiles.update(CH, CID, {"default_workspace": "~/Proj"})
    agent.model.reply = '{"folder": "general", "session": "new", "confidence": 0.3, "ask": "Which one?"}'
    agent.sessions = []
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "tidy it up"})
    await c.event("done")
    assert [e for e in c.drain() if e["ev"] == "question"] == []
    assert agent.brain(CHAT).spec.cwd == proj
    await c.call({"op": "new_session", "conv": CHAT})
    calls = len(agent.model.calls)
    await c.call({"op": "ask", "conv": CHAT, "turn": 2, "text": "something new"})
    await c.event("done")
    assert len(agent.model.calls) == calls and agent.brain(CHAT).spec.cwd == proj  # a new session starts there


async def test_send_file_confirm_follows_confirm_files(agent):
    await agent.svc.profiles.update(CH, CID, {"shared": False, "confirm_files": True})
    agent.scripts[CHAT] = lambda t: [fb.text("ok")]
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "hi"})
    await c.event("done")
    assert agent.brain(CHAT).spec.confirm_send_file() is True and agent.brain(CHAT).spec.chat_shared is False


def test_profile_question_needs_a_quote_or_thumbs():
    from agentd.questions import Question, bind

    q = Question("q", s.QuestionKind.PROFILE, "?", time.time() + 60, prompt_ref="p")
    assert not bind(q, s.Reply(conv=CHAT, text="yes"), time.time()).bound
    assert not bind(q, s.Reply(conv=CHAT, text="Yes, go ahead", quoted_ref="p"), time.time()).bound
    assert bind(q, s.Reply(conv=CHAT, text="yes", quoted_ref="p"), time.time()).outcome is s.ReplyOutcome.APPROVED
    assert bind(q, s.Reply(conv=CHAT, reaction="\U0001F44D", reaction_on="p"), time.time()).bound

