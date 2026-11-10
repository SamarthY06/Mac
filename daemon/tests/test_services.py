"""macsvc and `python -m chatd` as real processes, in a temp HOME and MAC_HOME, with agentd's
fake brain and chatd's fake channel (or a stand-in bridge that exits at once). Never the real
~/.mac-agent or WhatsApp."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

TOKEN = "a" * 64


@pytest.fixture
def world() -> Iterator[dict[str, Any]]:
    root = Path(tempfile.mkdtemp(prefix="sv", dir="/tmp"))  # short: unix socket paths cap at 104 bytes
    home = root / "h"
    mac = home / ".mac-agent"
    (mac / "chat").mkdir(parents=True)
    env = {**os.environ, "HOME": str(home), "MAC_HOME": str(mac), "MAC_LOG_DIR": str(root / "logs")}
    yield {"root": root, "mac": mac, "env": env, "sock": root / "a.sock", "status": mac / "run" / "services.json"}
    shutil.rmtree(root, ignore_errors=True)


def config(w: dict[str, Any], text: str = "") -> None:
    (w["mac"] / "chat" / "config.toml").write_text(text)


async def until(pred: Callable[[], Any], timeout: float = 20.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with_value = pred()
        if with_value:
            return with_value
        await asyncio.sleep(0.05)
    raise AssertionError("condition not met in time")


def status(w: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return json.loads(w["status"].read_text())
    except (OSError, ValueError):
        return None


def child(w: dict[str, Any], name: str) -> dict[str, Any]:
    return (status(w) or {}).get("children", {}).get(name, {})


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def macsvc(w: dict[str, Any], *chatd_args: str) -> asyncio.subprocess.Process:
    extra = [a for x in chatd_args for a in ("--chatd-arg", x)]
    return await asyncio.create_subprocess_exec(
        sys.executable, "-m", "macsvc", "--test", "--sock", str(w["sock"]), *extra, env=w["env"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def stop(proc: asyncio.subprocess.Process) -> int:
    if proc.returncode is None:
        proc.send_signal(signal.SIGTERM)
    return await asyncio.wait_for(proc.wait(), 40)


async def test_supervisor_starts_both_writes_status_and_restarts_a_crashed_chatd(world: dict[str, Any]) -> None:
    config(world)
    sup = await macsvc(world, "--channel fake")
    try:
        await until(lambda: child(world, "agentd").get("state") == "running"
                    and child(world, "chatd").get("state") == "running")
        await until(lambda: ((status(world) or {}).get("whatsapp") or {}).get("state") == "connected")
        assert world["sock"].exists()
        assert stat.S_IMODE(world["status"].stat().st_mode) == 0o600
        first = child(world, "chatd")["pid"]
        agent_pid = child(world, "agentd")["pid"]

        os.kill(first, signal.SIGKILL)
        await until(lambda: child(world, "chatd").get("restarts") == 1
                    and child(world, "chatd").get("state") == "running")
        assert child(world, "chatd")["pid"] != first
        assert child(world, "agentd")["pid"] == agent_pid  # agentd untouched
        log = (world["root"] / "logs" / "services.log").read_text()
        assert "chatd exited with code -9" in log and "chatd restarts in" in log
    finally:
        assert await stop(sup) == 0
    final = status(world)
    assert final is not None and final["stopping"] is True
    assert {c["state"] for c in final["children"].values()} == {"stopped"}
    assert not alive(first) and not alive(agent_pid)


@pytest.mark.parametrize(("code", "words"), [(3, "budget is spent"), (75, "in use by another run")])
async def test_chatd_is_not_restarted_after_the_bridge_exits_3_or_75(world: dict[str, Any], code: int,
                                                                      words: str) -> None:
    bridge = json.dumps([sys.executable, "-c", f"import sys; sys.exit({code})"])
    config(world, f"budget = 5\n[channels.whatsapp]\ncommand = {bridge}\n")
    sup = await macsvc(world, "--channel whatsapp")
    try:
        await until(lambda: child(world, "chatd").get("state") == "stopped"
                    and child(world, "chatd").get("last_exit") == code)
        await asyncio.sleep(1.5)  # longer than the first backoff: no restart came
        c = child(world, "chatd")
        assert c["state"] == "stopped" and c["restarts"] == 0 and words in c["why"]
        assert child(world, "agentd")["state"] == "running"  # agentd stays up
        assert (status(world) or {})["whatsapp"] is None
    finally:
        assert await stop(sup) == 0


async def test_killing_the_supervisor_closes_the_lifelines_and_both_children_exit(world: dict[str, Any]) -> None:
    config(world)
    sup = await macsvc(world, "--channel fake")
    await until(lambda: child(world, "agentd").get("state") == "running"
                and child(world, "chatd").get("state") == "running")
    pids = [child(world, "agentd")["pid"], child(world, "chatd")["pid"]]
    sup.kill()  # SIGKILL: no clean-up code runs, only the pipes close
    await sup.wait()
    await until(lambda: not any(alive(p) for p in pids))


async def chatd(w: dict[str, Any], *args: str) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable, "-m", "chatd", "--test", "--sock", str(w["sock"]), *args, env=w["env"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def last_words(out: bytes) -> dict[str, Any]:
    return json.loads(out.decode().strip().splitlines()[-1])["stopping"]


async def test_chatd_stops_cleanly_on_lifeline_eof(world: dict[str, Any]) -> None:
    config(world)
    proc = await chatd(world, "--channel", "fake")
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps({"token": TOKEN}).encode() + b"\n")
    await proc.stdin.drain()
    line = await asyncio.wait_for(proc.stdout.readline(), 15)
    assert json.loads(line) == {"link": {"state": "connected", "detail": ""}}
    proc.stdin.close()
    out = await asyncio.wait_for(proc.stdout.read(), 15)
    assert await proc.wait() == 0
    assert last_words(out)["code"] == 0


@pytest.mark.parametrize("setup", ["nothing", "bad_pairing"])
async def test_chatd_refuses_to_start_when_the_never_set_is_unreadable(world: dict[str, Any], setup: str) -> None:
    if setup == "bad_pairing":  # no config and a pairing.json that is not JSON
        (world["mac"] / "chat" / "pairing.json").write_text("{not json")
    proc = await chatd(world, "--channel", "fake")
    out, _ = await asyncio.wait_for(proc.communicate(json.dumps({"token": TOKEN}).encode() + b"\n"), 15)
    assert proc.returncode == 2
    assert "never-list" in last_words(out)["why"]


async def test_chatd_refuses_a_missing_token(world: dict[str, Any]) -> None:
    config(world)
    proc = await chatd(world, "--channel", "fake")
    out, _ = await asyncio.wait_for(proc.communicate(b"short\n"), 15)
    assert proc.returncode == 2 and last_words(out)["code"] == 2
