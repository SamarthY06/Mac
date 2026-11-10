"""`python -m agentd`: the backend service (W0, W1, api §3, §8).

Mac.app writes `{"tokens": {<token>: <kind>}}` as the first line of agentd's stdin (the lifeline)
and keeps the pipe open; EOF means an orderly shutdown (V4). `--dev` instead makes tokens, writes
them to ~/.mac-agent/run/dev-tokens.json (0600) and warns at every start.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import logging
import logging.handlers
import os
import secrets
import signal
import sys
import threading
import tomllib
from pathlib import Path
from typing import Any

from agentd import schema as s
from agentd.service import AgentPaths, Service

log = logging.getLogger("agentd")

TOKEN_READ_TIMEOUT_S = 10.0
_KINDS = {s.ClientKind.VOICE.value, s.ClientKind.CHAT.value, s.ClientKind.APP.value}


def parse_tokens(line: str) -> dict[str, s.ClientKind]:
    """The lifeline's first line -> {token: kind}. Only voice, chat and app are accepted."""
    obj = json.loads(line)
    raw = obj.get("tokens") if isinstance(obj, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise ValueError("expected {\"tokens\": {<token>: <kind>}}")
    out = {}
    for tok, kind in raw.items():
        if not isinstance(tok, str) or len(tok) < 32 or kind not in _KINDS:
            raise ValueError("a token is too short or has an unknown kind")
        out[tok] = s.ClientKind(kind)
    return out


def dev_tokens(run_dir: Path) -> dict[str, s.ClientKind]:
    by_kind = {k: secrets.token_hex(32) for k in sorted(_KINDS)}
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = run_dir / "dev-tokens.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(by_kind, f)
    os.chmod(path, 0o600)
    log.warning("DEV MODE: client tokens written to %s. Never use --dev for normal running.", path)
    return {tok: s.ClientKind(k) for k, tok in by_kind.items()}


def take_pid_lock(path: Path) -> int | None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def _setup_logging(log_dir: Path, verbose: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.handlers.RotatingFileHandler(log_dir / "agentd.log", maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers[:] = [fh, sh]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


def _chat_brain_cfg(path: Path, base: Any) -> Any:
    """[chat_brain] of ~/.mac-agent/agent/config.toml over macd's BrainConfig defaults."""
    import dataclasses

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f).get("chat_brain") or {}
    except FileNotFoundError:
        data = {}
    known = {f.name for f in dataclasses.fields(base)}
    return dataclasses.replace(base, **{k: v for k, v in data.items() if k in known})


async def serve(args: argparse.Namespace, tokens: dict[str, s.ClientKind], stop: asyncio.Event) -> None:
    from agentd.brains import claude_factory, fake_factory
    from agentd.router import WarmHaiku
    from agentd.server import Server
    from macd import config
    from macd.brain.memory import MemoryFile

    paths = AgentPaths.default()
    if args.fake_brain:
        factory, model = fake_factory(), None
    else:
        from macd.brain.claude_brain import default_cli_path

        cfg = config.load(write_default=False)
        chat_cfg = _chat_brain_cfg(paths.mac_home / "agent" / "config.toml", config.BrainConfig())
        factory = claude_factory(cfg, chat_cfg)
        model = WarmHaiku(default_cli_path(cfg))
    svc = Service(paths, factory, router_model=model, memory_file=MemoryFile(paths.memory))
    await svc.start()
    server = Server(svc, tokens, Path(args.socket) if args.socket else paths.sock)
    await server.start()
    try:
        await stop.wait()
    finally:
        log.info("shutting down")
        await server.close()
        await svc.close()
        if model is not None:
            await model.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentd", description="Mac's backend service.")
    ap.add_argument("--dev", action="store_true", help="make tokens and write dev-tokens.json (development only)")
    ap.add_argument("--socket", help="socket path (default ~/.mac-agent/run/agent.sock)")
    ap.add_argument("--fake-brain", action="store_true", help="scripted brains, no claude (development only)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    paths = AgentPaths.default()
    _setup_logging(paths.log_dir, args.verbose)
    lock_fd = take_pid_lock(paths.mac_home / "run" / "agent.pid.lock")
    if lock_fd is None:
        print("agentd is already running.", file=sys.stderr)
        return 1

    if args.dev:
        tokens = dev_tokens(paths.mac_home / "run")
    else:
        line = _read_first_line(TOKEN_READ_TIMEOUT_S)
        try:
            tokens = parse_tokens(line or "")
        except ValueError as e:
            log.error("no valid tokens on the lifeline (%s); exiting", e)
            return 2

    loop = asyncio.new_event_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    if not args.dev:
        threading.Thread(target=_watch_lifeline, args=(loop, stop), daemon=True, name="lifeline").start()
    try:
        loop.run_until_complete(serve(args, tokens, stop))
    finally:
        loop.close()
        os.close(lock_fd)
    return 0


def _read_first_line(timeout: float) -> str | None:
    box: list[str] = []
    t = threading.Thread(target=lambda: box.append(sys.stdin.buffer.readline().decode("utf-8", "replace")), daemon=True)
    t.start()
    t.join(timeout)
    return box[0] if box else None


def _watch_lifeline(loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> None:
    """Block on stdin until EOF (Mac.app is gone), then stop in order (V4)."""
    with contextlib.suppress(Exception):
        while sys.stdin.buffer.read1(65536):
            pass
    log.info("lifeline closed")
    with contextlib.suppress(RuntimeError):
        loop.call_soon_threadsafe(stop.set)


if __name__ == "__main__":
    sys.exit(main())
