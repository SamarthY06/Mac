"""macctl: command-line client for the macd control socket."""

from __future__ import annotations

import argparse
import getpass
import itertools
import json
import os
import socket
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

from macd import config


class DaemonError(RuntimeError):
    pass


class ControlClient:
    """Tiny synchronous NDJSON client (one request at a time + optional event stream)."""

    def __init__(self, path: Path | str | None = None, timeout: float = 35.0) -> None:
        self.path = str(path or config.CONTROL_SOCK)
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._buf = b""
        self._ids = itertools.count(1)
        self._events: list[dict[str, Any]] = []

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def connect(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        try:
            s.connect(self.path)
        except (FileNotFoundError, ConnectionRefusedError) as e:
            s.close()
            raise DaemonError("Mac isn't running. Open Mac.app (or run `macd` for headless use).") from e
        self._sock = s

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _readline(self, timeout: float | None) -> dict[str, Any] | None:
        assert self._sock is not None
        deadline = None if timeout is None else time.monotonic() + timeout
        while b"\n" not in self._buf:
            if deadline is not None:
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._sock.settimeout(left)
            else:
                self._sock.settimeout(None)
            try:
                chunk = self._sock.recv(65536)
            except TimeoutError:
                return None
            if not chunk:
                raise DaemonError("daemon closed the connection")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return json.loads(line)

    def request(self, cmd: str, **args: Any) -> dict[str, Any]:
        assert self._sock is not None, "not connected"
        mid = next(self._ids)
        self._sock.sendall((json.dumps({"id": mid, "cmd": cmd, **args}) + "\n").encode())
        while True:
            msg = self._readline(self.timeout)
            if msg is None:
                raise DaemonError(f"timed out waiting for reply to {cmd}")
            if "event" in msg:
                self._events.append(msg)
                continue
            if msg.get("id") == mid:
                if not msg.get("ok"):
                    raise DaemonError(str(msg.get("error", "unknown error")))
                return msg.get("data") or {}

    def events(self, timeout: float | None = None) -> Iterator[dict[str, Any]]:
        """Yield events (buffered first). Stops when `timeout` elapses without events."""
        while True:
            while self._events:
                yield self._events.pop(0)
            msg = self._readline(timeout)
            if msg is None:
                return
            if "event" in msg:
                yield msg


# ----------------------------------------------------------------------------- commands


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _fmt_status(d: dict[str, Any]) -> str:
    head = f"state: {d.get('state')}  turn: {d.get('turn_id')}  muted: {d.get('muted')}  "
    head += f"off: {d.get('off')}  brain alive: {d.get('brain_alive')}  voice: {d.get('voice_engine') or '-'}  "
    if not d.get("voice", True):
        head += f"VOICE UNAVAILABLE ({d.get('voice_reason') or 'unknown'})  "
    remote = d.get("remote") or {}
    remote_s = "phone" if remote.get("present") else ("link ready" if remote.get("active") else "off")
    head += f"remote: {remote_s}  uptime: {d.get('uptime_s')}s"
    lines = [head]
    if d.get("turn"):
        t = d["turn"]
        lines.append(f"current: {t.get('text')!r} (tool: {t.get('tool')}, {t.get('age_s')}s)")
    if d.get("listeners"):
        lines.append("listeners: " + ", ".join(d["listeners"]))
    for k in ("brain", "audio", "remote"):
        if d.get(k):
            lines.append(f"{k}: {json.dumps(d[k], default=str)}")
    m = d.get("metrics") or {}
    lat = m.get("latency_ms") or {}
    if lat:
        lines.append("latency (ms)        n     p50     p95")
        for name, v in lat.items():
            p50 = "-" if v.get("p50") is None else f"{v['p50']:.0f}"
            p95 = "-" if v.get("p95") is None else f"{v['p95']:.0f}"
            lines.append(f"  {name:<26}{v.get('n', 0):>4} {p50:>7} {p95:>7}")
    if m.get("counters"):
        lines.append("counters: " + ", ".join(f"{k}={v}" for k, v in sorted(m["counters"].items())))
    return "\n".join(lines)


def cmd_ask(client: ControlClient, text: str, wait: bool, timeout: float) -> int:
    if not wait:
        data = client.request("ask", text=text)
        print(f"turn {data.get('turn_id')}")
        return 0
    client.request("subscribe")
    tid = client.request("ask", text=text)["turn_id"]
    answer: list[str] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for ev in client.events(timeout=min(10.0, max(0.1, deadline - time.monotonic()))):
            if ev.get("turn_id") != tid:
                continue
            kind = ev.get("event")
            if kind == "transcript" and ev.get("role") == "assistant":
                if ev.get("kind") != "filler":
                    answer.append(str(ev.get("text", "")))
                    print(f"  > {ev.get('text')}", file=sys.stderr)
            elif kind == "activity":
                k = ev.get("kind")
                if k == "turn_end":
                    print(" ".join(answer).strip() or str(ev.get("text", "")).strip())
                    return 1 if ev.get("errored") or ev.get("cancelled") else 0
                if k in ("tool_start", "narration", "steer", "confirm_result", "status"):
                    print(f"  . {k}: {ev.get('text') or ev.get('tool') or ''}", file=sys.stderr)
            elif kind == "error":
                print(f"  ! {ev.get('message')}", file=sys.stderr)
            elif kind == "state":
                print(f"  [{ev.get('state')}]", file=sys.stderr)
            if time.monotonic() > deadline:
                break
        # Quiet for a while (long tool call?): make sure the turn is still alive.
        st = client.request("status")
        if st.get("turn_id") != tid and not client._events:
            print(" ".join(answer).strip())
            return 0
    print("timed out waiting for the turn to finish", file=sys.stderr)
    return 2


def fmt_connect(d: dict[str, Any]) -> str:
    cl, ch = d.get("claude") or {}, d.get("chrome") or {}
    mark = lambda ok: "✓" if ok else "✗"
    lines = [
        f" {mark(cl.get('installed'))} Claude Code: {cl.get('version') or 'not installed'}"
        + ("" if cl.get("min_ok") or not cl.get("installed") else " (too old)"),
        f" {mark(cl.get('signed_in') == 'yes')} Signed in: {cl.get('signed_in')} ({cl.get('auth_method')})",
    ]
    if cl.get("fix"):
        lines.append(f"     fix: {cl['fix']}")
    lines += [
        f" {mark(ch.get('browser_running'))} Browser running: {', '.join(ch.get('browsers') or []) or 'no'}",
        f" {mark(ch.get('extension_installed'))} Claude in Chrome extension installed",
        f" {mark(ch.get('native_host'))} Native host registered",
        f" {mark(ch.get('enabled_by_default'))} Enabled by default (/chrome): {ch.get('enabled_by_default')}",
        f" {'✓' if ch.get('connected') == 'connected' else '-'} Connected: {ch.get('connected')}",
    ]
    if ch.get("fix"):
        lines.append(f"     fix: {ch['fix']}")
    lines.append(f" {'✓ READY' if d.get('ready') else '✗ not ready'}")
    return "\n".join(lines)


def cmd_logs(n: int, follow: bool, raw: bool) -> int:
    path = config.EVENT_LOG
    if not path.exists():
        print(f"no event log at {path}", file=sys.stderr)
        return 1

    def show(line: str) -> None:
        if raw:
            print(line.rstrip("\n"))
            return
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            print(line.rstrip("\n"))
            return
        ts = time.strftime("%H:%M:%S", time.localtime(rec.pop("ts", 0)))
        rec.pop("mono", None)
        typ = rec.pop("type", "?")
        tid = rec.pop("turn_id", None)
        extra = " ".join(f"{k}={json.dumps(v, default=str)}" for k, v in rec.items())
        print(f"{ts} {typ:<16} {'' if tid is None else f'#{tid} '}{extra}")

    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = _tail(fh, n)
        for line in lines:
            show(line)
        if not follow:
            return 0
        try:
            while True:
                line = fh.readline()
                if line:
                    show(line)
                    continue
                time.sleep(0.25)
                try:
                    if os.stat(path).st_ino != os.fstat(fh.fileno()).st_ino:  # rotated
                        fh.close()
                        fh = open(path, encoding="utf-8", errors="replace")  # noqa: SIM115
                except FileNotFoundError:
                    pass
        except KeyboardInterrupt:
            return 0


def _tail(fh: Any, n: int) -> list[str]:
    fh.seek(0, os.SEEK_END)
    end = fh.tell()
    block = 64 * 1024
    pos = max(0, end - block)
    while True:
        fh.seek(pos)
        data = fh.read()
        lines = data.splitlines(keepends=True)
        if len(lines) > n or pos == 0:
            fh.seek(0, os.SEEK_END)
            return lines[-n:] if n > 0 else []
        pos = max(0, pos - block)


def cmd_models() -> int:
    from macd.speech import models

    return models.main([])


def cmd_secret_set(name: str) -> int:
    if sys.stdin.isatty():
        value = getpass.getpass(f"{name}: ")
    else:
        value = sys.stdin.read().strip()
    if not value:
        print("empty value; nothing stored", file=sys.stderr)
        return 1
    config.set_secret(name, value)
    print(f"stored {name} in the Keychain (service {config.KEYCHAIN_SERVICE})")
    return 0


# ----------------------------------------------------------------------------- argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="macctl", description="Control the Mac voice agent daemon.")
    p.add_argument("--socket", default=None, help="control socket path (default: %(default)s)")
    p.add_argument("--json", action="store_true", help="print raw JSON replies")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="show state, latency metrics and counters")
    a = sub.add_parser("ask", help="ask Mac something (as if spoken)")
    a.add_argument("text")
    a.add_argument("--wait", action="store_true", help="stream progress and print the answer")
    a.add_argument("--timeout", type=float, default=900.0)
    cf = sub.add_parser("confirm", help="answer a pending 'Shall I go ahead?'")
    cf.add_argument("answer", choices=["yes", "no"])
    s = sub.add_parser("say", help="speak text")
    s.add_argument("text")
    sub.add_parser("stop", help="interrupt the current turn / stop speaking")
    m = sub.add_parser("mute", help="mute or unmute the microphone")
    m.add_argument("value", choices=["on", "off"])
    sub.add_parser("listen", help="push-to-talk: listen for one utterance without the wake word")
    sub.add_parser("off", help="switch listening off (no wake word, mic released) until `macctl on`")
    sub.add_parser("on", help="switch listening back on")
    sub.add_parser("reset", help="start a fresh conversation")
    sub.add_parser("remote-link", help="create a LiveKit link to talk to your Mac from your phone")
    sub.add_parser("remote-end", help="end the current phone session")
    v = sub.add_parser("voice", help="choose Mac's voice (default Kokoro, or a clone of a short recording)")
    vsub = v.add_subparsers(dest="voice_cmd", required=True)
    vs = vsub.add_parser("set", help="clone the voice in a wav/m4a recording (10 to 30 s of clean speech)")
    vs.add_argument("path")
    vsub.add_parser("reset", help="back to the default voice")
    vsub.add_parser("status", help="show the active voice engine")
    vp = vsub.add_parser("preview", help="play a sample of a Kokoro voice without switching to it")
    vp.add_argument("voice")
    vp.add_argument("--text", default=None)
    sub.add_parser("voices", help="list the available Kokoro voices")
    cc = sub.add_parser("connect", help="check that Claude Code and Claude in Chrome are set up")
    cc.add_argument("--deep", action="store_true", help="also start Claude and ask it whether Chrome is connected")
    st = sub.add_parser("settings", help="read or change settings (saved to config.toml, applied live)")
    stsub = st.add_subparsers(dest="settings_cmd", required=True)
    stsub.add_parser("get", help="print all settings")
    ss = stsub.add_parser("set", help="change settings: key=value [key=value …]  (lists: a,b)")
    ss.add_argument("pairs", nargs="+", metavar="key=value")
    sub.add_parser("audio-reinit", help="re-open the audio devices")
    lg = sub.add_parser("logs", help="tail the event log")
    lg.add_argument("-n", type=int, default=40)
    lg.add_argument("-f", "--follow", action="store_true")
    lg.add_argument("--raw", action="store_true")
    d = sub.add_parser("doctor", help="check permissions, models, Claude login, …")
    d.add_argument("--deep", action="store_true", help="also run a real Claude query (slow)")
    sub.add_parser("models", help="download missing speech models")
    sec = sub.add_parser("secret", help="manage secrets in the Keychain")
    secsub = sec.add_subparsers(dest="secret_cmd", required=True)
    ss = secsub.add_parser("set", help="store a secret (value read from stdin)")
    ss.add_argument("name")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except DaemonError as e:
        print(f"macctl: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def _run(args: argparse.Namespace) -> int:
    if args.cmd == "logs":
        return cmd_logs(args.n, args.follow, args.raw)
    if args.cmd == "doctor":
        from macd import doctor

        return doctor.main(["--deep"] if args.deep else [])
    if args.cmd == "models":
        return cmd_models()
    if args.cmd == "secret":
        return cmd_secret_set(args.name)

    timeout = 90.0 if args.cmd == "remote-link" else 35.0
    with ControlClient(args.socket, timeout=timeout) as c:
        if args.cmd == "status":
            data = c.request("status")
            print(json.dumps(data, indent=2, default=str) if args.json else _fmt_status(data))
            return 0
        if args.cmd == "ask":
            return cmd_ask(c, args.text, args.wait, args.timeout)
        if args.cmd == "say":
            c.request("say", text=args.text)
            return 0
        if args.cmd == "confirm":
            c.request("confirm", approved=args.answer == "yes")
            print("approved" if args.answer == "yes" else "denied")
            return 0
        if args.cmd == "stop":
            c.request("stop")
            return 0
        if args.cmd == "mute":
            data = c.request("mute", value=args.value == "on")
            print("muted" if data.get("muted") else "unmuted")
            return 0
        if args.cmd == "listen":
            c.request("listen")
            print("listening…")
            return 0
        if args.cmd in ("off", "on"):
            data = c.request(args.cmd)
            print("listening off" if data.get("off") else "listening on")
            return 0
        if args.cmd == "reset":
            c.request("reset")
            print("new conversation")
            return 0
        if args.cmd == "remote-link":
            data = c.request("remote_link")
            if args.json:
                _print_json(data)
            elif not data.get("configured", True) or data.get("enabled") is False:
                print(data.get("hint", "LiveKit isn't set up."), file=sys.stderr)
                return 1
            else:
                print(data.get("url", ""))
                print(f"PIN: {data.get('pin')}  (say it on the phone; link valid for "
                      f"{int(data.get('expires_in_s', 0)) // 60} min)", file=sys.stderr)
            return 0
        if args.cmd == "remote-end":
            c.request("remote_end")
            return 0
        if args.cmd == "voice":
            if args.voice_cmd == "set":
                data = c.request("voice_clone", path=str(Path(args.path).expanduser().resolve()))
                print(f"voice: {data.get('engine')} ({data.get('clone_sample')})")
            elif args.voice_cmd == "reset":
                data = c.request("voice_reset")
                print(f"voice: {data.get('engine')}")
            elif args.voice_cmd == "preview":
                c.request("voice_preview", voice=args.voice, text=args.text)
                print(f"playing a sample of {args.voice}")
            else:
                st = c.request("status")
                print(f"voice: {st.get('voice_engine')}" + (f" ({st['clone_sample']})" if st.get("clone_sample") else ""))
            return 0
        if args.cmd == "connect":
            data = c.request("connect_check", deep=args.deep)
            if args.json:
                _print_json(data)
            else:
                print(fmt_connect(data))
            return 1 if not data.get("ready") else 0
        if args.cmd == "voices":
            for v in c.request("voices_list").get("voices", []):
                print(f"{v.get('id', '?'):<12} {v.get('name', '')}  {v.get('lang', '')} {v.get('gender', '')}".rstrip())
            return 0
        if args.cmd == "settings":
            if args.settings_cmd == "get":
                _print_json(c.request("settings_get"))
                return 0
            changes: dict[str, Any] = {}
            for pair in args.pairs:
                key, sep, value = pair.partition("=")
                if not sep or not key.strip():
                    raise DaemonError(f"expected key=value, got {pair!r}")
                changes[key.strip()] = value
            res = c.request("settings_set", changes=changes)
            for k, v in (res.get("applied") or {}).items():
                print(f"{k} = {v!r}")
            for k, msg in (res.get("errors") or {}).items():
                print(f"{k}: {msg}", file=sys.stderr)
            return 1 if res.get("errors") else 0
        if args.cmd == "audio-reinit":
            c.request("audio_reinit")
            return 0
    raise DaemonError(f"unknown command {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
