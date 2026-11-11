"""`python -m chatd`: the chat service (W8 phase 3, api §1).

The token comes on the lifeline: `{"token": "<hex>"}` as the first line of stdin (Mac.app's and
macsvc's convention), or the first line of `--token-fd N`, which is then the lifeline instead.
EOF on the lifeline, or SIGTERM, is an orderly stop.

chatd reads `~/.mac-agent/chat/config.toml` (process keys only; a missing file means defaults),
builds the channel with an empty allowlist (agentd's profiles fill it) and the never-set from
`agentd.profiles.load_never_set`, failing closed when that cannot be read, and runs ChatRunner.

While WhatsApp is connected, chatd keeps `~/.mac-agent/chat/chats.json` (0600): the bridge's
`list_chats` (id, subject, members, owner_only, is_self), refreshed on every connect, every 5
minutes and on SIGUSR1. `mac chats add` resolves a group name to its id from it.

stdout carries one JSON line per change for a supervisor, never message text:
`{"link": {"state", "detail"}}` and, last, `{"stopping": {"code", "why"}}`.

Exit codes: 0 asked to stop; 1 crash or the bridge keeps crashing (restart with backoff);
2 bad config, unreadable never-set, missing or mismatched bridge, or agentd refused us; 3 the
bridge's budget is spent; 75 the bridge's store is in use. 2, 3 and 75 must not be restarted.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import logging.handlers
import os
import signal
import sys
import threading
import tomllib
from pathlib import Path
from typing import IO, Any

log = logging.getLogger("chatd")

EXIT_OK, EXIT_CRASH, EXIT_CONFIG, EXIT_BUDGET, EXIT_LOCKED = 0, 1, 2, 3, 75
TOKEN_READ_TIMEOUT_S = 10.0
POLL_S = 0.5
CHATS_EVERY_S = 300.0
CHAT_KEYS = ("id", "subject", "members", "owner_only", "is_self")
REPO = Path(__file__).resolve().parents[3]


class Stop(Exception):
    def __init__(self, code: int, why: str) -> None:
        super().__init__(why)
        self.code, self.why = code, why


def parse_token(line: str) -> str:
    """`{"token": "<hex>"}` (or the bare token) -> the token."""
    line = line.strip()
    tok: Any = line
    if line.startswith("{"):
        obj = json.loads(line)
        tok = obj.get("token") if isinstance(obj, dict) else None
    if not isinstance(tok, str) or len(tok) < 32:
        raise ValueError("expected {\"token\": <token>} on the first line")
    return tok


def load_config(path: Path) -> dict[str, Any]:
    from chatd.channels.registry import process_config

    try:
        with open(path, "rb") as f:
            return process_config(tomllib.load(f))
    except FileNotFoundError:
        return {}


def emit(out: IO[str], obj: dict[str, Any]) -> None:
    with contextlib.suppress(Exception):
        out.write(json.dumps(obj) + "\n")
        out.flush()


def default_bridge(mac_home: Path) -> Path:
    installed = mac_home / "bin" / "wa-bridge"
    return installed if installed.exists() else REPO / "bridges" / "whatsapp" / "bin" / "wa-bridge"


def channel_cfg(cfg: dict[str, Any], name: str, mac_home: Path) -> dict[str, Any]:
    channels = cfg.get("channels")
    sec = dict(channels.get(name) or {}) if isinstance(channels, dict) else {}
    if "budget" in cfg and "budget" not in sec:
        sec["budget"] = cfg["budget"]
    if name == "whatsapp":  # MAC_HOME-relative defaults (the adapter's own are under ~/.mac-agent)
        sec.setdefault("bridge", str(default_bridge(mac_home)))
        sec.setdefault("store", str(mac_home / "chat" / "whatsapp" / "device.db"))
        sec.setdefault("media_dir", str(mac_home / "chat" / "whatsapp" / "media"))
    return sec


def bridge_stopped(adapter: Any) -> Stop | None:
    """The WhatsApp bridge stopped for good (its supervisor gave up): the matching exit."""
    from chatd.channels import whatsapp as wa

    if not isinstance(adapter, wa.WhatsAppChannel):
        return None
    code, detail = adapter.exit_code, adapter.status().detail
    if code == wa.EXIT_BAD_ARGS:
        return Stop(EXIT_CONFIG, "The WhatsApp helper refused its settings or doesn't match this version.")
    if code == wa.EXIT_BUDGET:
        return Stop(EXIT_BUDGET, "The WhatsApp message budget is spent.")
    if code == wa.EXIT_LOCKED:
        return Stop(EXIT_LOCKED, "WhatsApp is in use by another run.")
    if detail == wa.NOT_INSTALLED:
        return Stop(EXIT_CONFIG, "The WhatsApp helper isn't installed.")
    if detail == wa.CRASHING:
        return Stop(EXIT_CRASH, "The WhatsApp helper keeps stopping.")
    return None


async def write_chats(adapter: Any, path: Path) -> None:
    """The bridge's chat list to `chats.json` for `mac chats add` (subjects never logged)."""
    from agentd.stores import atomic_write_json

    try:
        chats = await adapter.list_chats()
    except Exception as e:  # noqa: BLE001 - offline, timeout: the next refresh tries again
        log.info("chatd: chat list not refreshed (%s)", type(e).__name__)
        return
    rows = [{k: c[k] for k in CHAT_KEYS if k in c} for c in chats if isinstance(c.get("id"), str)]
    await asyncio.to_thread(atomic_write_json, path, rows)
    log.info("chatd: chat list refreshed (%d chats)", len(rows))


async def serve(args: argparse.Namespace, token: str, stop: asyncio.Event, out: IO[str],
                refresh: asyncio.Event | None = None) -> int:
    from agentd import profiles as pf
    from chatd.channels.registry import ChannelContext, SeenIds, create
    from chatd.governor import Budget, Governor
    from chatd.runner import ChatRunner
    from macd import config

    mac_home = Path(config.HOME)
    chat_dir = mac_home / "chat"
    try:
        cfg = load_config(chat_dir / "config.toml")
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise Stop(EXIT_CONFIG, f"The chat config can't be read ({type(e).__name__}).") from e
    never = await asyncio.to_thread(pf.load_never_set, chat_dir / "config.toml", chat_dir / "pairing.json")
    if not never.ok:
        raise Stop(EXIT_CONFIG, "The never-list can't be read, so chat stays off.")
    log.info("chatd: never-set has %d chat(s)", len(never.jids))

    name = args.channel or str(cfg.get("channel", "whatsapp"))
    budget = cfg.get("budget")
    if budget is not None and (not isinstance(budget, int) or budget < 1):
        raise Stop(EXIT_CONFIG, "The chat config's budget must be a positive whole number.")
    chat_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    gov = await Governor(None if args.test else chat_dir / "governor.json", Budget(total=budget)).load()
    seen = await SeenIds(chat_dir / "seen.json").load()
    ctx = ChannelContext(governor=gov, chats=frozenset(), never_chats=never.jids, seen=seen,
                         dry_run=bool(cfg.get("dry_run", False)), test=args.test)
    try:
        adapter = create(name, channel_cfg(cfg, name, mac_home), ctx)
    except (ValueError, TypeError) as e:
        raise Stop(EXIT_CONFIG, f"The chat channel can't start ({e}).") from e

    sock = Path(args.sock) if args.sock else mac_home / "run" / "agent.sock"
    runner = ChatRunner(adapter, sock=sock, token=token, state_dir=chat_dir)
    await runner.start()
    log.info("chatd: running channel %s", name)
    shown: tuple[str, str] | None = None
    refresh = refresh or asyncio.Event()
    listing: asyncio.Task[None] | None = None
    listed_at = 0.0
    loop = asyncio.get_running_loop()
    try:
        while not stop.is_set():
            link = runner.link or adapter.status()
            if (link.state.value, link.detail) != shown:
                if link.state.value == "connected":
                    refresh.set()  # every (re)connect
                shown = (link.state.value, link.detail)
                emit(out, {"link": {"state": shown[0], "detail": shown[1]}})
            due = refresh.is_set() or loop.time() - listed_at >= CHATS_EVERY_S
            if shown[0] == "connected" and due and (listing is None or listing.done()) \
                    and hasattr(adapter, "list_chats"):
                refresh.clear()
                listed_at = loop.time()
                listing = asyncio.create_task(write_chats(adapter, chat_dir / "chats.json"))
            if runner.fatal is not None:
                raise Stop(EXIT_CONFIG, "agentd refused chat's token or version.")
            if (gone := bridge_stopped(adapter)) is not None:
                raise gone
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), POLL_S)
        return EXIT_OK
    finally:
        log.info("chatd: stopping")
        if listing is not None:
            listing.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await listing
        await runner.close()


def _setup_logging(log_dir: Path, verbose: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.handlers.RotatingFileHandler(log_dir / "chatd.log", maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers[:] = [fh, sh]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


def _read_first_line(fd: int, timeout: float) -> str | None:
    """The lifeline's first line. Raw os.read, not a buffered stream: a reader thread still blocked
    in a buffered read aborts the interpreter at exit."""
    box: list[bytes] = []

    def read() -> None:
        buf = b""
        while not buf.endswith(b"\n") and len(buf) < 4096:
            chunk = os.read(fd, 1)
            if not chunk:
                break
            buf += chunk
        box.append(buf)

    t = threading.Thread(target=read, daemon=True)
    t.start()
    t.join(timeout)
    return box[0].decode("utf-8", "replace") if box else None


def _watch_lifeline(fd: int, loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> None:
    """Block on the lifeline until EOF (the supervisor is gone), then stop in order."""
    with contextlib.suppress(OSError):
        while os.read(fd, 65536):
            pass
    log.info("chatd: lifeline closed")
    with contextlib.suppress(RuntimeError):
        loop.call_soon_threadsafe(stop.set)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="chatd", description="Mac's chat service (WhatsApp).")
    ap.add_argument("--token-fd", type=int, help="read the token from this fd, which is then the lifeline")
    ap.add_argument("--sock", help="agentd's socket (default ~/.mac-agent/run/agent.sock)")
    ap.add_argument("--channel", help="channel to run (default: config `channel`, else whatsapp)")
    ap.add_argument("--test", action="store_true", help="test mode: test adapters, governor in memory")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    from macd import config

    _setup_logging(Path(config.LOG_DIR) / "chat", args.verbose)
    out = sys.stdout
    lifeline = args.token_fd if args.token_fd is not None else sys.stdin.fileno()
    try:
        token = parse_token(_read_first_line(lifeline, TOKEN_READ_TIMEOUT_S) or "")
    except ValueError as e:
        log.error("chatd: no valid token on the lifeline (%s); exiting", e)
        emit(out, {"stopping": {"code": EXIT_CONFIG, "why": "No valid token on the lifeline."}})
        return EXIT_CONFIG

    loop = asyncio.new_event_loop()
    stop = asyncio.Event()
    refresh = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    loop.add_signal_handler(signal.SIGUSR1, refresh.set)  # `mac chats add`: refresh chats.json now
    threading.Thread(target=_watch_lifeline, args=(lifeline, loop, stop), daemon=True, name="lifeline").start()
    try:
        code, why = loop.run_until_complete(serve(args, token, stop, out, refresh)), "Stopped."
    except Stop as e:
        code, why = e.code, e.why
        log.error("chatd: %s (exit %d)", why, code)
    finally:
        loop.close()
    emit(out, {"stopping": {"code": code, "why": why}})
    return code


if __name__ == "__main__":
    sys.exit(main())
