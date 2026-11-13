"""`python -m macsvc`: keeps agentd and chatd running (the launchd LaunchAgent runs this).

Mac.app becomes the supervisor later (W8, builder G); until then this does its job:

- **Tokens.** One chat, one app and one voice token per boot of macsvc, never in env. agentd gets
  `{"tokens": {<chat>: "chat", <app>: "app", <voice>: "voice"}}` and chatd `{"token": <chat>}` as
  the first line of stdin, their lifeline. Both pipes stay open; restarts reuse the same tokens.
  The app token is also written to ~/.mac-agent/run/app-token (0600, in the 0700 run dir the gate
  denies) for the `mac chats` CLI, and the voice token to run/voice-token for macd's local-mode
  task and stop_other links (tasks DESIGN D1); both are removed at stop. The chat token is never
  on disk.
- **Restarts.** A child that exits restarts with backoff 1, 2, 4 ... 60 s; 5 exits within 10 minutes
  stop it for good (the bridge breaker's numbers). chatd exiting 2 (config), 3 (budget spent) or 75
  (WhatsApp store in use) is not restarted, nor agentd exiting 2 (bad tokens); the other child
  keeps running.
- **Stop.** SIGTERM or SIGINT closes chatd's lifeline, then agentd's, and waits; a child that has
  not exited after the grace period is killed with its process group.
- **Logs** go to ~/Library/Logs/Mac/services.log (rotating): starts, exit codes and restarts only,
  plus the last stderr lines of a child that crashed. No message text passes through here.
- **Status** goes to ~/.mac-agent/run/services.json (0600) after every change: each child's state
  (running, restarting, stopped) with the reason in plain words, and chatd's WhatsApp link state.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import fcntl
import json
import logging
import logging.handlers
import os
import secrets
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("macsvc")

BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0)
BREAKER = (5, 600.0)  # exits within seconds
GRACE_S = 15.0
SOCKET_WAIT_S = 15.0
STDERR_TAIL = 10

WHY = {  # exit code -> plain words, per child
    "chatd": {2: "Stopped: the chat settings or the WhatsApp helper need fixing.",
              3: "Stopped: the WhatsApp message budget is spent.",
              75: "Stopped: WhatsApp is in use by another run."},
    "agentd": {2: "Stopped: agentd refused its tokens."},
}


@dataclass
class Child:
    name: str
    argv: list[str]
    first_line: bytes  # written to the lifeline (stdin) at every start
    no_restart: frozenset[int] = frozenset()
    state: str = "stopped"  # running | restarting | stopped
    why: str = "Not started yet."
    pid: int | None = None
    restarts: int = 0
    last_exit: int | None = None
    since: float = field(default_factory=time.time)
    proc: asyncio.subprocess.Process | None = None
    link: dict[str, str] | None = None  # chatd's WhatsApp link: {state, detail}
    stopping_why: str = ""  # chatd's own last words ({"stopping": ...} on stdout)
    tail: collections.deque[str] = field(default_factory=lambda: collections.deque(maxlen=STDERR_TAIL))


class Supervisor:
    def __init__(self, children: list[Child], status_path: Path, *, sock: Path | None = None,
                 backoff: tuple[float, ...] = BACKOFF_S, breaker: tuple[int, float] = BREAKER,
                 grace_s: float = GRACE_S) -> None:
        self.children = children
        self.status_path = status_path
        self.sock = sock
        self.backoff = backoff
        self.breaker = breaker
        self.grace_s = grace_s
        self.stopping = asyncio.Event()
        self.started = time.time()

    # -- status --------------------------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        chatd = next((c for c in self.children if c.name == "chatd"), None)
        return {
            "pid": os.getpid(), "started_at": self.started, "updated_at": time.time(),
            "stopping": self.stopping.is_set(),
            "children": {c.name: {"state": c.state, "why": c.why, "pid": c.pid, "restarts": c.restarts,
                                  "last_exit": c.last_exit, "since": c.since} for c in self.children},
            "whatsapp": chatd.link if chatd is not None and chatd.state == "running" else None,
        }

    def write_status(self) -> None:
        from agentd.stores import atomic_write_json

        try:
            atomic_write_json(self.status_path, self.snapshot())  # small; fine on the loop
        except OSError as e:
            log.warning("could not write %s (%s)", self.status_path.name, type(e).__name__)

    def _set(self, c: Child, state: str, why: str) -> None:
        c.state, c.why, c.since = state, why, time.time()
        self.write_status()

    # -- running -------------------------------------------------------------------------------------
    async def run(self) -> None:
        self.write_status()
        tasks = []
        for i, c in enumerate(self.children):
            tasks.append(asyncio.create_task(self._keep(c), name=f"macsvc:{c.name}"))
            if i == 0 and self.sock is not None:
                await self._wait_socket()
        await self.stopping.wait()
        log.info("stopping")
        for c, t in reversed(list(zip(self.children, tasks, strict=True))):
            await self._stop_child(c, t)
        self.write_status()
        log.info("stopped")

    async def _wait_socket(self) -> None:
        deadline = time.monotonic() + SOCKET_WAIT_S
        while not self.stopping.is_set() and time.monotonic() < deadline:
            if self.sock is not None and self.sock.exists():
                return
            await asyncio.sleep(0.1)

    async def _keep(self, c: Child) -> None:
        loop = asyncio.get_running_loop()
        exits: collections.deque[float] = collections.deque()
        limit, window = self.breaker
        while not self.stopping.is_set():
            try:
                code = await self._run_once(c)
            except OSError as e:
                log.error("%s could not start (%s)", c.name, type(e).__name__)
                self._set(c, "stopped", f"Stopped: {c.name} could not start.")
                return
            c.last_exit, c.pid, c.link = code, None, None
            if self.stopping.is_set():
                self._set(c, "stopped", "Stopped by request.")
                return
            log.warning("%s exited with code %s", c.name, code)
            for line in c.tail:
                log.warning("%s stderr: %s", c.name, line)
            if code in c.no_restart:
                why = WHY.get(c.name, {}).get(code, f"Stopped: {c.name} exited with code {code}.")
                if c.stopping_why:
                    why = f"{why} {c.stopping_why}"
                log.error("%s will not be restarted (exit %s)", c.name, code)
                self._set(c, "stopped", why)
                return
            now = loop.time()
            exits.append(now)
            while exits and exits[0] < now - window:
                exits.popleft()
            if len(exits) >= limit:
                log.error("%s keeps crashing (%d exits in %.0f s); not restarting", c.name, len(exits), window)
                self._set(c, "stopped", f"Stopped: {c.name} keeps crashing. Restart the services.")
                return
            delay = self.backoff[min(len(exits), len(self.backoff)) - 1]
            log.info("%s restarts in %.0f s", c.name, delay)
            self._set(c, "restarting", f"{c.name} stopped unexpectedly. Restarting it.")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.stopping.wait(), delay)
            c.restarts += 1
        self._set(c, "stopped", "Stopped by request.")

    async def _run_once(self, c: Child) -> int:
        c.tail.clear()
        c.stopping_why = ""
        proc = await asyncio.create_subprocess_exec(
            *c.argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True)
        c.proc, c.pid = proc, proc.pid
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        with contextlib.suppress(ConnectionError):
            proc.stdin.write(c.first_line)
            await proc.stdin.drain()
        log.info("%s started (pid %d)", c.name, proc.pid)
        self._set(c, "running", "Running.")
        readers = [asyncio.create_task(self._read_stdout(c, proc.stdout)),
                   asyncio.create_task(self._read_stderr(c, proc.stderr))]
        try:
            code = await proc.wait()
        finally:
            for r in readers:
                with contextlib.suppress(TimeoutError, Exception):
                    await asyncio.wait_for(r, 2.0)
                r.cancel()
            c.proc = None
        return code

    async def _read_stdout(self, c: Child, stream: asyncio.StreamReader) -> None:
        """chatd's status lines: `{"link": {state, detail}}` and `{"stopping": {code, why}}`."""
        async for raw in stream:
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            link = obj.get("link")
            if isinstance(link, dict):
                c.link = {"state": str(link.get("state", "")), "detail": str(link.get("detail", ""))}
                log.info("%s: WhatsApp is %s", c.name, c.link["state"])
                self.write_status()
            stop = obj.get("stopping")
            if isinstance(stop, dict) and stop.get("code"):
                c.stopping_why = str(stop.get("why", ""))[:200]

    async def _read_stderr(self, c: Child, stream: asyncio.StreamReader) -> None:
        async for raw in stream:
            c.tail.append(raw.decode("utf-8", "replace").rstrip()[:500])

    async def _stop_child(self, c: Child, task: asyncio.Task[None]) -> None:
        proc = c.proc
        if proc is not None and proc.stdin is not None:
            with contextlib.suppress(Exception):
                proc.stdin.close()  # the lifeline
        try:
            await asyncio.wait_for(asyncio.shield(task), self.grace_s)
        except TimeoutError:
            log.warning("%s did not stop in %.0f s; killing it", c.name, self.grace_s)
            if c.proc is not None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(c.proc.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                await task


# -- wiring ----------------------------------------------------------------------------------------------
def build_children(args: argparse.Namespace, tok: str, app_tok: str, voice_tok: str) -> list[Child]:
    py = sys.executable
    agentd = [py, "-m", "agentd"] + (["--fake-brain"] if args.test else []) + \
        (["--socket", args.sock] if args.sock else [])
    chatd = [py, "-m", "chatd"] + (["--test"] if args.test else []) + (["--sock", args.sock] if args.sock else []) + \
        [a for extra in args.chatd_arg or [] for a in extra.split()]
    tokens = {tok: "chat", app_tok: "app", voice_tok: "voice"}
    return [
        Child("agentd", agentd, (json.dumps({"tokens": tokens}) + "\n").encode(), frozenset({2})),
        Child("chatd", chatd, (json.dumps({"token": tok}) + "\n").encode(), frozenset({2, 3, 75})),
    ]


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(log_dir / "services.log", maxBytes=1_000_000, backupCount=3)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [fh]
    root.setLevel(logging.INFO)


def take_lock(path: Path) -> int | None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def write_token(path: Path, token: str) -> None:
    """Atomically, 0600 from the start, in a 0700 dir."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    os.replace(tmp, path)


async def amain(args: argparse.Namespace, mac_home: Path) -> int:
    sock = Path(args.sock) if args.sock else mac_home / "run" / "agent.sock"
    tok, app_tok, voice_tok = secrets.token_hex(32), secrets.token_hex(32), secrets.token_hex(32)
    app_path = mac_home / "run" / "app-token"
    voice_path = mac_home / "run" / "voice-token"
    write_token(app_path, app_tok)
    write_token(voice_path, voice_tok)
    sup = Supervisor(build_children(args, tok, app_tok, voice_tok), mac_home / "run" / "services.json", sock=sock)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, sup.stopping.set)
    try:
        await sup.run()
    finally:
        for path in (app_path, voice_path):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
    return 0


def main(argv: list[str] | None = None) -> int:
    from macd import config

    ap = argparse.ArgumentParser(prog="macsvc", description="Keeps agentd and chatd running.")
    ap.add_argument("--sock", help="agentd's socket (default ~/.mac-agent/run/agent.sock)")
    ap.add_argument("--test", action="store_true", help="agentd --fake-brain and chatd --test (tests only)")
    ap.add_argument("--chatd-arg", action="append", help="extra chatd arguments (tests)")
    args = ap.parse_args(argv)

    mac_home = Path(config.HOME)
    setup_logging(Path(config.LOG_DIR))
    lock = take_lock(mac_home / "run" / "services.lock")
    if lock is None:
        print("macsvc is already running.", file=sys.stderr)
        log.error("another macsvc is already running; exiting")
        return 1
    log.info("starting (pid %d)", os.getpid())
    try:
        return asyncio.run(amain(args, mac_home))
    finally:
        os.close(lock)


if __name__ == "__main__":
    sys.exit(main())
