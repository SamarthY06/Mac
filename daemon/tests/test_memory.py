from __future__ import annotations

from pathlib import Path

from macd.brain import memory
from macd.brain.memory import Memory, MemoryFile, grounded


def test_grounding_accepts_user_facts_and_refuses_page_facts() -> None:
    said = "Hey Mac, I prefer short answers.\nMy team's WhatsApp group is Timetable, by the way."
    assert grounded("Prefers short answers", said)
    assert grounded("Team's WhatsApp group is called Timetable", said)
    assert grounded("User's WhatsApp group: Timetable", said)
    # from a web page / tool result the user never mentioned
    assert not grounded("Lives at 42 Elm Street, Springfield; phone 555-0199", said)
    assert not grounded("Favourite framework is Django and deploys on Kubernetes", said)
    assert not grounded("", said)
    assert not grounded("the and of", said)  # no content words → refuse


def test_sections_routing_and_dedupe() -> None:
    mem = Memory()
    assert mem.remember("Prefers short answers", "preference") == "added"
    assert mem.remember("Works on the Mac voice agent", "project") == "added"
    assert mem.remember("Priya is a colleague", "person") == "added"
    assert mem.remember("Weekly timetable update on Mondays", "task") == "added"
    assert mem.remember("Name is Nikhil", "about", kept=True) == "added"
    assert mem.remember("Likes jazz", "bogus-kind") == "added"  # unknown kind → About you
    assert mem.sections["Preferences"] == ["Prefers short answers"]
    assert mem.sections["Projects & work"] == ["Works on the Mac voice agent"]
    assert mem.sections["People"] == ["Priya is a colleague"]
    assert mem.sections["Recurring tasks"] == ["Weekly timetable update on Mondays"]
    assert mem.sections["About you"] == ["Name is Nikhil [kept]", "Likes jazz"]
    # near-duplicate (≥ 0.85) replaces instead of appending, keeps the [kept] tag
    assert mem.remember("Prefers short answers.", "preference") == "updated"
    assert mem.remember("Name is Nikhil!", "about") == "updated"
    assert mem.sections["Preferences"] == ["Prefers short answers"]
    assert mem.sections["About you"][0] == "Name is Nikhil! [kept]"
    rendered = mem.render()
    assert rendered.startswith("# Memory\n\n## About you\n- Name is Nikhil! [kept]")
    assert Memory.parse(rendered).sections == mem.sections


def test_forget() -> None:
    mem = Memory()
    mem.remember("Prefers short answers", "preference")
    mem.remember("Team's WhatsApp group is Timetable", "preference")
    mem.remember("Priya is a colleague", "person")
    assert mem.forget("the timetable group") == 1
    assert mem.sections["Preferences"] == ["Prefers short answers"]
    assert mem.forget("nothing like this at all") == 0
    assert mem.forget("Priya") == 1 and mem.sections["People"] == []


def test_recent_context_limit_and_replace() -> None:
    mem = Memory()
    mem.set_recent_context(["x" * 600, "y" * 600, "z"])
    assert mem.sections["Recent context"] == ["x" * 600]  # 1 KB cap
    mem.set_recent_context(["fresh"])
    assert mem.sections["Recent context"] == ["fresh"]


def test_consolidation_preserves_kept_lines() -> None:
    old = Memory()
    old.remember("Name is Nikhil", "about", kept=True)
    old.remember("Always use Chrome for the web", "preference", kept=True)
    for i in range(40):
        old.remember(f"Stale detail number {i} about something", "project")
    rewritten = "# Memory\n## About you\n- Name is Nikhil [kept]\n## Projects & work\n- Works on Mac\n"
    new = old.consolidate(rewritten)
    assert new.sections["About you"] == ["Name is Nikhil [kept]"]
    assert new.sections["Preferences"] == ["Always use Chrome for the web [kept]"]  # re-added
    assert new.sections["Projects & work"] == ["Works on Mac"]
    assert len(new.render().encode()) < memory.CONSOLIDATE_TARGET


def test_memory_file_roundtrip_and_legacy_format(tmp_path: Path) -> None:
    f = MemoryFile(tmp_path / "memory.md")
    assert f.load().sections["About you"] == []
    (tmp_path / "memory.md").write_text("- 2026-11-07: likes tea\n")  # old flat format
    assert f.load().sections["About you"] == ["likes tea"]
    assert f.remember("Prefers short answers", "preference") == "added"
    f.add_recent_context("Was drafting the weekly update")
    assert "## Preferences\n- Prefers short answers" in f.text()
    assert "## Recent context\n- Was drafting the weekly update" in f.text()
    assert f.forget("tea") == 1
    assert "likes tea" not in f.text()
    assert not f.needs_consolidation()


# ----------------------------------------------------------------------------- F11 writers

def _distinct(seed: str) -> str:
    """A fact no other fact dedupes into (Memory.remember merges near-duplicates)."""
    import hashlib

    return hashlib.sha256(seed.encode()).hexdigest()[:40]


def test_f11_two_writers_never_lose_a_fact(tmp_path: Path) -> None:
    """Two MemoryFile instances (two brains) on one memory.md, writing at the same time."""
    import threading

    path = tmp_path / "memory.md"
    a, b = MemoryFile(path), MemoryFile(path)
    facts = [_distinct(f"t{i}") for i in range(60)]

    def write(f: MemoryFile, items: list[str]) -> None:
        for x in items:
            f.remember(x, "project")

    threads = [threading.Thread(target=write, args=(a, facts[0::2])),
               threading.Thread(target=write, args=(b, facts[1::2]))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    saved = MemoryFile(path).load().sections["Projects & work"]
    assert sorted(saved) == sorted(facts)
    assert (tmp_path / "memory.md.lock").exists()


def test_f11_single_shared_instance_without_flock(tmp_path: Path) -> None:
    import threading

    f = MemoryFile(tmp_path / "memory.md", flock=False)  # remote mode: one writer, no flock
    items = [_distinct(f"s{i}") for i in range(40)]
    threads = [threading.Thread(target=f.remember, args=(x, "task")) for x in items]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(f.load().sections["Recurring tasks"]) == sorted(items)
    assert not (tmp_path / "memory.md.lock").exists()


_WRITER = """
import hashlib, sys
from pathlib import Path
from macd.brain.memory import MemoryFile
f = MemoryFile(Path(sys.argv[1]))
for i in range(int(sys.argv[3])):
    f.remember(sys.argv[2] + " " + hashlib.sha256(f"{sys.argv[2]}{i}".encode()).hexdigest()[:40], "project")
"""


def test_f11_two_processes_never_lose_a_fact(tmp_path: Path) -> None:
    import subprocess
    import sys

    path = tmp_path / "memory.md"
    procs = [subprocess.Popen([sys.executable, "-c", _WRITER, str(path), who, "40"]) for who in ("voice", "chat")]
    for p in procs:
        assert p.wait(60) == 0
    saved = MemoryFile(path).load().sections["Projects & work"]
    assert len(saved) == 80
    assert {s.split()[0] for s in saved} == {"voice", "chat"}


def test_flock_held_elsewhere_times_out_instead_of_hanging(tmp_path: Path) -> None:
    import fcntl
    import os

    import pytest

    path = tmp_path / "memory.md"
    fd = os.open(tmp_path / "memory.md.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)  # another process (another open file description) holds it
    try:
        f = MemoryFile(path, flock_timeout_s=0.2)
        with pytest.raises(TimeoutError):
            f.remember("Something new", "about")
        assert f.text() == ""
    finally:
        os.close(fd)
    assert MemoryFile(path).remember("Something new", "about") == "added"
