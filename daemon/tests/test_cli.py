from __future__ import annotations

import asyncio
import io
import json
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from macd import cli
from macd.control import ControlServer


def test_parser() -> None:
    p = cli.build_parser()
    a = p.parse_args(["ask", "what's up", "--wait"])
    assert a.cmd == "ask" and a.text == "what's up" and a.wait
    assert p.parse_args(["mute", "on"]).value == "on"
    assert p.parse_args(["logs", "-n", "5", "-f"]).follow
    assert p.parse_args(["secret", "set", "LIVEKIT_URL"]).name == "LIVEKIT_URL"
    assert p.parse_args(["doctor", "--deep"]).deep and p.parse_args(["confirm", "no"]).answer == "no"
    assert p.parse_args(["voice", "set", "x.wav"]).path == "x.wav" and p.parse_args(["voice", "reset"]).voice_cmd == "reset"
    assert p.parse_args(["settings", "set", "a=1", "b=2"]).pairs == ["a=1", "b=2"] and p.parse_args(["voices"]).cmd == "voices"
    for cmd in ("status", "say x", "stop", "listen", "reset", "remote-link", "models", "audio-reinit", "off", "on", "remote-end"):
        p.parse_args(cmd.split())
    with pytest.raises(SystemExit):
        p.parse_args(["mute", "maybe"])


class FakeOrch:
    def __init__(self) -> None:
        self.server: ControlServer | None = None
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.muted = False

    async def handle(self, cmd: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((cmd, args))
        if cmd == "status":
            return {"state": "idle", "turn_id": None, "muted": self.muted, "uptime_s": 1.0, "voice_engine": "kokoro",
                    "metrics": {"latency_ms": {"turn_total": {"n": 1, "p50": 1200.0, "p95": 1300.0}},
                                "counters": {"turns": 1}}}
        if cmd == "ask":
            asyncio.get_running_loop().call_later(0.05, self._play_turn, 9)
            return {"turn_id": 9}
        if cmd == "mute":
            self.muted = bool(args["value"])
            return {"muted": self.muted}
        if cmd in ("off", "on"):
            return {"off": cmd == "off"}
        if cmd == "voice_clone":
            return {"engine": "clone", "clone_sample": args["path"]}
        if cmd == "voice_reset":
            return {"engine": "kokoro", "clone_sample": None}
        if cmd == "settings_get":
            return {"name": "Mac", "wake.phrases": ["hey mac"]}
        if cmd == "settings_set":
            return {"applied": {k: v for k, v in args["changes"].items() if k != "bad"},
                    "errors": {"bad": "setting not allowed: bad"} if "bad" in args["changes"] else {}}
        if cmd == "voices_list":
            return {"voices": [{"id": "af_heart", "name": "Heart", "lang": "en-us", "gender": "f"}]}
        if cmd == "voice_preview":
            return {"queued": True}
        if cmd == "connect_check":
            return {"claude": {"installed": True, "version": "2.1.319", "min_ok": True, "signed_in": "yes",
                               "auth_method": "claude_ai", "fix": ""},
                    "chrome": {"browser_running": False, "browsers": [], "extension_installed": True, "native_host": True,
                               "enabled_by_default": True, "connected": "unknown", "fix": "Chrome isn't running"},
                    "ready": False}
        if cmd == "remote_link":
            return {"url": "https://meet.livekit.io/custom?x=1#token=t", "expires_in_s": 900, "pin": "1234",
                    "configured": True}
        return {}

    def _play_turn(self, tid: int) -> None:
        assert self.server is not None
        pub = self.server.publish
        pub({"event": "state", "state": "thinking", "turn_id": tid})
        pub({"event": "transcript", "role": "assistant", "kind": "filler", "text": "On it.", "turn_id": tid})
        pub({"event": "activity", "kind": "tool_start", "tool": "Bash", "text": "Running a command", "turn_id": tid})
        pub({"event": "transcript", "role": "assistant", "kind": "answer", "text": "Two plus two is four.", "turn_id": tid})
        pub({"event": "transcript", "role": "assistant", "kind": "answer", "text": "Other turn.", "turn_id": 3})
        pub({"event": "activity", "kind": "turn_end", "turn_id": tid, "text": "Two plus two is four."})


@pytest.fixture
async def daemon() -> AsyncIterator[tuple[FakeOrch, Path]]:
    d = Path(tempfile.mkdtemp(prefix="mc", dir="/tmp"))
    sock = d / "control.sock"
    orch = FakeOrch()
    srv = ControlServer(orch.handle, sock)
    orch.server = srv
    await srv.start()
    try:
        yield orch, sock
    finally:
        await srv.close()
        shutil.rmtree(d, ignore_errors=True)


async def run_cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = await asyncio.to_thread(cli.main, argv)
    out, err = capsys.readouterr()
    return code, out, err


async def test_status_roundtrip(daemon: tuple[FakeOrch, Path], capsys: pytest.CaptureFixture[str]) -> None:
    _orch, sock = daemon
    code, out, _ = await run_cli(["--socket", str(sock), "status"], capsys)
    assert code == 0 and "state: idle" in out and "turn_total" in out and "remote: off" in out
    code, out, _ = await run_cli(["--socket", str(sock), "--json", "status"], capsys)
    assert json.loads(out)["state"] == "idle"


async def test_ask_wait_prints_answer(daemon: tuple[FakeOrch, Path], capsys: pytest.CaptureFixture[str]) -> None:
    orch, sock = daemon
    code, out, err = await run_cli(["--socket", str(sock), "ask", "what is two plus two", "--wait", "--timeout", "5"],
                                   capsys)
    assert code == 0
    assert out.strip() == "Two plus two is four."
    assert "Running a command" in err and "Other turn." not in out
    assert ("ask", {"text": "what is two plus two"}) in orch.calls


async def test_simple_commands(daemon: tuple[FakeOrch, Path], capsys: pytest.CaptureFixture[str]) -> None:
    orch, sock = daemon
    s = ["--socket", str(sock)]
    assert (await run_cli([*s, "mute", "on"], capsys))[1].strip() == "muted"
    assert orch.muted
    assert (await run_cli([*s, "say", "hello"], capsys))[0] == 0
    assert (await run_cli([*s, "confirm", "yes"], capsys))[1].strip() == "approved"
    assert (await run_cli([*s, "stop"], capsys))[0] == 0
    assert (await run_cli([*s, "listen"], capsys))[0] == 0
    assert (await run_cli([*s, "off"], capsys))[1].strip() == "listening off"
    assert (await run_cli([*s, "on"], capsys))[1].strip() == "listening on"
    code, out, _ = await run_cli([*s, "remote-link"], capsys)
    assert code == 0 and out.startswith("https://meet.livekit.io/")
    assert (await run_cli([*s, "remote-end"], capsys))[0] == 0
    sample = str(Path("/tmp/me.wav").resolve())  # macctl resolves the path (/private/tmp on macOS)
    code, out, _ = await run_cli([*s, "voice", "set", "/tmp/me.wav"], capsys)
    assert code == 0 and out.strip() == f"voice: clone ({sample})"
    assert (await run_cli([*s, "voice", "reset"], capsys))[1].strip() == "voice: kokoro"
    assert (await run_cli([*s, "voice", "status"], capsys))[1].strip() == "voice: kokoro"
    assert (await run_cli([*s, "voice", "preview", "af_heart"], capsys))[1].strip() == "playing a sample of af_heart"
    code, out, _ = await run_cli([*s, "voices"], capsys)
    assert code == 0 and out.strip() == "af_heart     Heart  en-us f"
    code, out, _ = await run_cli([*s, "settings", "get"], capsys)
    assert code == 0 and json.loads(out)["name"] == "Mac"
    code, out, err = await run_cli([*s, "settings", "set", "name=Jarvis", "tts.speed=1.2", "bad=1"], capsys)
    assert code == 1 and "name = 'Jarvis'" in out and "tts.speed = '1.2'" in out and "bad: setting not allowed" in err
    assert orch.calls[-1][1] == {"changes": {"name": "Jarvis", "tts.speed": "1.2", "bad": "1"}}
    assert (await run_cli([*s, "settings", "set", "novalue"], capsys))[0] == 1
    code, out, _ = await run_cli([*s, "connect", "--deep"], capsys)
    assert code == 1 and "Claude Code: 2.1.319" in out and "fix: Chrome isn't running" in out and "not ready" in out
    assert orch.calls[-1] == ("connect_check", {"deep": True})
    code, out, _ = await run_cli([*s, "--json", "connect"], capsys)
    assert code == 1 and json.loads(out)["ready"] is False
    cmds = [c for c, _ in orch.calls]
    assert cmds[:12] == ["mute", "say", "confirm", "stop", "listen", "off", "on", "remote_link", "remote_end", "voice_clone",
                         "voice_reset", "status"]
    assert cmds[12:] == ["voice_preview", "voices_list", "settings_get", "settings_set", "connect_check", "connect_check"]
    assert orch.calls[9][1] == {"path": sample} and orch.calls[2][1] == {"approved": True}


async def test_daemon_not_running(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = await run_cli(["--socket", "/tmp/definitely-not-a-mac-socket", "status"], capsys)
    assert code == 1 and "isn't running" in err


def test_logs_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text("\n".join(json.dumps({"ts": 1.0, "mono": 1.0, "type": "t", "i": i}) for i in range(100)) + "\n")
    monkeypatch.setattr(cli.config, "EVENT_LOG", log)
    assert cli.main(["logs", "-n", "3"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3 and "i=99" in lines[-1]


def test_secret_set_reads_stdin(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    stored: dict[str, str] = {}
    monkeypatch.setattr(cli.config, "set_secret", lambda n, v: stored.__setitem__(n, v))
    monkeypatch.setattr("sys.stdin", io.StringIO("sekrit\n"))
    assert cli.main(["secret", "set", "LIVEKIT_API_SECRET"]) == 0
    assert stored == {"LIVEKIT_API_SECRET": "sekrit"}
    assert "sekrit" not in capsys.readouterr().out


async def test_daemon_voice_hooks_with_fake_stack(tmp_path: Path) -> None:
    """__main__._voice_clone/_voice_reset: set_voice runs in a thread, confirmation is spoken."""
    import argparse

    from macd import config
    from macd.__main__ import Daemon

    class FakeStack:
        voice_engine = "kokoro"
        clone_sample: str | None = None
        voice_error: str | None = None

        def set_voice(self, path: str | None) -> str:
            if path is not None and path.endswith("bad.wav"):
                raise ValueError("not speech")
            self.voice_engine = "clone" if path else "kokoro"
            self.clone_sample = path
            return "Done. This is my new voice." if path else "Back to my default voice."

    class FakeOrch:
        said: ClassVar[list[str]] = []
        remote: ClassVar[dict[str, Any]] = {}

        def say(self, text: str) -> None:
            self.said.append(text)

    d = Daemon(config.Config(), argparse.Namespace(audio="file", input_file=None, fake_brain=True, log_level="INFO"))
    d.stack, d.orch = FakeStack(), FakeOrch()  # type: ignore[assignment]
    data = await d._voice_clone(str(tmp_path / "me.wav"))
    assert data["engine"] == "clone" and data["voice_engine"] == {"engine": "clone", "sample": "me.wav"}
    assert d.orch.said == ["Done. This is my new voice."] and d._status()["voice_engine"] == "clone"
    with pytest.raises(ValueError, match="could not use that voice sample"):
        await d._voice_clone(str(tmp_path / "bad.wav"))
    data = await d._voice_reset()
    assert data == {"engine": "kokoro", "clone_sample": None, "voice_engine": {"engine": "kokoro", "sample": None}}
    assert d.orch.said[-1] == "Back to my default voice."
    d.stack = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="unavailable"):
        await d._voice_reset()
