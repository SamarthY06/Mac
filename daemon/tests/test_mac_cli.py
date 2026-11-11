"""`mac` (macsvc.cli) as a real process in a temp HOME and MAC_HOME: the fake bridge for linking,
a stubbed launchctl, and a test macsvc (agentd fake brain, chatd fake channel) for the chats ops.
Never the real ~/.mac-agent, LaunchAgent or WhatsApp."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

FAKE_BRIDGE = Path(__file__).resolve().parent / "chatd" / "fake_bridge.py"
FAMILY, WORK, TIMETABLE = "120363000000000001@g.us", "120363000000000002@g.us", "120363000000000003@g.us"
CHATS = [{"id": FAMILY, "subject": "Family", "members": 4, "owner_only": False, "is_self": False},
         {"id": TIMETABLE, "subject": " Timetable ", "members": 30, "owner_only": False, "is_self": False},
         {"id": WORK, "subject": "Work", "members": 6, "owner_only": False, "is_self": False}]
LAUNCHCTL = """#!/bin/bash
echo "$*" >>"$STUB_DIR/calls"
case "$1" in
  print) [[ -f "$STUB_DIR/loaded" ]] ;;
  bootstrap) touch "$STUB_DIR/loaded" ;;
  bootout) rm -f "$STUB_DIR/loaded" ;;
esac
"""


class World:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.home = root / "h"
        self.mac = self.home / ".mac-agent"
        self.chat = self.mac / "chat"
        self.fake = root / "fb"
        self.stub = root / "stub"
        for d in (self.chat, self.fake, self.stub):
            d.mkdir(parents=True)
        (self.stub / "launchctl").write_text(LAUNCHCTL)
        (self.stub / "launchctl").chmod(0o755)
        self.env = {**os.environ, "HOME": str(self.home), "MAC_HOME": str(self.mac),
                    "MAC_LOG_DIR": str(root / "logs"), "LAUNCHCTL": str(self.stub / "launchctl"),
                    "STUB_DIR": str(self.stub)}
        self.env.pop("PYTHONPATH", None)
        command = json.dumps([sys.executable, str(FAKE_BRIDGE), "--fake-dir", str(self.fake)])
        self.config(f"[channels.whatsapp]\ncommand = {command}\n")

    def config(self, text: str) -> None:
        (self.chat / "config.toml").write_text(text)

    def bridge(self, **cfg: Any) -> None:
        (self.fake / "fake.json").write_text(json.dumps({"chats": CHATS, **cfg}))

    async def cli(self, *args: str, stdin: bytes | None = None, timeout: float = 60) -> tuple[int, str, str]:
        p = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "macsvc.cli", *args, env=self.env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = await asyncio.wait_for(p.communicate(stdin or b""), timeout)
        assert p.returncode is not None
        return p.returncode, out.decode(), err.decode()

    def pairing(self) -> dict[str, Any]:
        return json.loads((self.chat / "pairing.json").read_text())


@pytest.fixture
def w() -> Iterator[World]:
    root = Path(tempfile.mkdtemp(prefix="mc", dir="/tmp"))  # short: unix socket paths cap at 104 bytes
    yield World(root)
    shutil.rmtree(root, ignore_errors=True)


async def scan(w: World) -> None:
    r, wr = await asyncio.open_unix_connection(str(w.fake / "c.sock"))
    wr.write(b'{"cmd": "scan"}\n')
    await wr.drain()
    assert json.loads(await r.readline())["ok"]
    wr.close()


async def test_link_shows_a_qr_writes_pairing_0600_and_never_offers_timetable(w: World) -> None:
    w.bridge(linked=False)
    p = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "macsvc.cli", "whatsapp", "link", env=w.env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.stdout is not None and p.stdin is not None
    seen = b""
    while b"\x1b[0m" not in seen:  # the QR's first line
        seen += await asyncio.wait_for(p.stdout.readline(), 20)
    assert b"Link a device" in seen and b"\xe2\x96\x80" in seen  # the half-block QR
    await scan(w)
    p.stdin.write(b"1\n")
    p.stdin.close()
    out = (seen + await asyncio.wait_for(p.stdout.read(), 30)).decode()
    assert await p.wait() == 0, (await p.stderr.read()).decode() if p.stderr else ""
    assert "Linked." in out and "1. Family (4 people)" in out and "2. Work" in out
    assert "Timetable" not in out
    assert 'mac chats add "Family"' in out  # the services aren't running: how to add it later

    path = w.chat / "pairing.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    pairing = w.pairing()
    assert pairing["Family"] == {"id": FAMILY, "members": 4, "owner_only": False}
    assert pairing["Timetable"]["id"] == TIMETABLE  # kept, so the never-list blocks it

    from agentd.profiles import load_never_set, read_pairing
    assert read_pairing(path) is not None
    assert TIMETABLE in load_never_set(w.chat / "config.toml", path).jids


@pytest.mark.parametrize("answer", ["timetable", "  TIME\u200btable "])
async def test_link_refuses_a_never_listed_pick(w: World, answer: str) -> None:
    w.bridge(linked=True)
    code, out, err = await w.cli("whatsapp", "link", stdin=f"{answer}\n".encode())
    assert code == 1 and "Mac never uses Timetable." in err
    assert "already linked" in out


async def test_link_refuses_while_the_store_is_locked(w: World) -> None:
    w.bridge(linked=True)
    store = w.chat / "whatsapp" / "device.db"
    store.parent.mkdir(parents=True)
    with open(f"{store}.lock", "w") as lock:  # noqa: ASYNC230 (held across the run)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # another bridge holds it
        code, _, err = await w.cli("whatsapp", "link")
    assert code == 1 and "in use by another Mac process" in err
    assert not (w.chat / "pairing.json").exists()


async def test_link_stops_the_running_service_and_starts_it_again(w: World) -> None:
    w.bridge(linked=True)
    (w.stub / "loaded").touch()
    code, out, _ = await w.cli("whatsapp", "link", stdin=b"\n")
    assert code == 0 and "Stopping Mac's services" in out and "Starting Mac's services again." in out
    calls = (w.stub / "calls").read_text().splitlines()
    assert [c.split()[0] for c in calls if c.split()[0] != "print"] == ["bootout", "bootstrap"]
    assert (w.stub / "loaded").exists()


async def wait_for(pred: Any, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "condition not met in time"
        await asyncio.sleep(0.1)


async def test_chats_ops_go_through_agentd_with_the_app_token(w: World) -> None:
    groups = "[" + ", ".join("{" + ", ".join(f"{k} = {json.dumps(v)}" for k, v in c.items()) + "}"
                             for c in CHATS) + "]"
    w.config(f"[channels.fake]\ngroups = {groups}\n")
    sup = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "macsvc", "--test", "--sock", str(w.mac / "run" / "agent.sock"),
        "--chatd-arg", "--channel fake", env=w.env, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    token = w.mac / "run" / "app-token"
    try:
        await wait_for(lambda: token.exists() and (w.chat / "chats.json").exists())
        assert stat.S_IMODE(token.stat().st_mode) == 0o600 and len(token.read_text().strip()) == 64

        code, out, _ = await w.cli("chats")
        assert code == 0 and "Mac answers in no groups yet" in out
        code, out, err = await w.cli("chats", "add", " family ")
        assert code == 0, err
        assert out.strip() == 'Added Family. Mac answers your messages that start with "Mac" there.'
        code, _, err = await w.cli("chats", "add", "Timetable")
        assert code == 1 and "Mac never uses Timetable." in err
        code, _, err = await w.cli("chats", "add", "Nope")
        assert code == 1 and 'no group called "Nope"' in err
        code, _, err = await w.cli("chats", "add", "Family")
        assert code == 1

        code, out, _ = await w.cli("chats", "deny", "Family", "chrome,", "the", "web")
        assert code == 0 and "Not allowed: Chrome, the web." in out
        code, out, _ = await w.cli("chats", "allow", "family", "Chrome")
        assert code == 0 and "Not allowed: the web." in out
        code, _, err = await w.cli("chats", "deny", "Family", "rockets")
        assert code == 1 and "is not something Mac can allow or deny" in err
        code, out, _ = await w.cli("chats", "off", "Family")
        assert code == 0 and out.startswith("Family: off")
        code, out, _ = await w.cli("chats", "list")
        assert code == 0 and out.startswith("Family: off,")
        code, out, _ = await w.cli("chats", "remove", "Family")
        assert code == 0 and "Removed Family." in out
        code, _, err = await w.cli("chats", "on", "Family")
        assert code == 1 and "no settings for a group" in err
    finally:
        sup.terminate()
        await asyncio.wait_for(sup.wait(), 40)
    assert not token.exists()  # removed at stop


async def test_chats_ops_without_the_services_say_how_to_start_them(w: World) -> None:
    code, _, err = await w.cli("chats")
    assert code == 1 and "scripts/services.sh install" in err
