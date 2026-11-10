"""FakeBridge: a stand-in for `wa-bridge` that speaks the stdio protocol (docs/whatsapp/bridge-protocol.md)
as a real child process, with no WhatsApp behind it. The adapter under test spawns it exactly as it
spawns the real binary:

    python fake_bridge.py --fake-dir DIR --store ... --media-dir ... --allow ... --deny ... [flags]

It keeps the bridge's rules: flag validation (exit 2), the store flock (exit 75), `hello` first,
the outbound checks in order (denied, not_allowed, rate, budget with exit 3, link state), the
inbound filter (§5), no download for voice/audio/video/stickers, `--dry-run`, `--test`, and the
stdin lifeline (EOF -> exit 0). Stderr carries ids, kinds and lengths only.

Tests drive it through `DIR/c.sock` (one JSON object per line, one reply per line):

    {"cmd": "inbound", "chat", "msg", "text", "from_me", "sender_device", "ts", "now", "replayed",
                       "media": {"kind", "name", "mime"}, "quoted": {...}, "reaction": {...}}
                                             ("now": the test clock, for the 7-day age rule)
    {"cmd": "emit", "event": {...}}          raw event, unfiltered (tests the adapter's own re-checks)
    {"cmd": "chat_changed", "chat", "subject", "members", "owner_only"}
    {"cmd": "offline"} | {"cmd": "reconnect"} | {"cmd": "scan"} | {"cmd": "exit", "code": n}
    {"cmd": "stats"}  -> {"writes": [...], "dropped": {...}, "allow", "deny", "argv", "downloads"}

and `DIR/fake.json`, read at every start: {"linked": true, "exit_codes": [1, 1], "protocol": "1",
"delay": {"send_text": 0.3}, "hang": ["status"], "chats": [...]}. Each start appends a line to
`DIR/starts`; start number i exits at once with `exit_codes[i]` when there is one.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

PROTOCOL = "1"
OWN_JID, OWN_LID, OWN_DEVICE = "447700900000@s.whatsapp.net", "100000000000001@lid", 7
EDIT_WINDOW_S = 15 * 60
MAX_AGE_S = 7 * 86400
UNSUPPORTED = {"voice", "audio", "video", "sticker"}
PREFIX = "Mac · "


def jids(s: str) -> list[str]:
    out = [p.strip() for p in s.split(",") if p.strip()]
    for j in out:
        if j.count("@") != 1 or j.startswith("@") or j.endswith("@") or any(c in j for c in " :"):
            raise ValueError("not a chat JID")
    return out


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="wa-bridge", exit_on_error=False, add_help=False)
    p.add_argument("--fake-dir", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--media-dir", required=True)
    p.add_argument("--allow", default="")
    p.add_argument("--deny", default="")
    p.add_argument("--trusted-devices", default="0")
    p.add_argument("--max-per-min", type=int, default=20)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--test", action="store_true")
    a = p.parse_args(argv)
    a.allow, a.deny = jids(a.allow), jids(a.deny)
    a.trusted = [int(x) for x in a.trusted_devices.split(",") if x.strip()]
    if set(a.allow) & set(a.deny) or not a.trusted or a.max_per_min < 1:
        raise ValueError("bad flags")
    if a.test and a.budget is None:
        raise ValueError("--test requires --budget")
    return a


def log(msg: str) -> None:
    sys.stderr.write(f"fake-bridge {msg}\n")
    sys.stderr.flush()


class Fake:
    def __init__(self, a: argparse.Namespace, cfg: dict[str, Any], start_no: int) -> None:
        self.a, self.cfg, self.start_no = a, cfg, start_no
        self.allow, self.deny = set(a.allow), set(a.deny)
        self.linked = bool(cfg.get("linked", True))
        self.connected = False
        self.pairing = False
        self.hello = False
        self.used = 0
        self.n = 0
        self.sent: dict[str, float] = {}  # our message id -> sent time
        self.sends: list[float] = []  # new-message times (rate)
        self.writes: list[dict[str, Any]] = []  # network writes (never under --dry-run); no text
        self.dropped: dict[str, int] = {}
        self.downloads = 0
        self.ops: list[str] = []  # every request op, in arrival order
        self.media = Path(a.media_dir)

    # -- output ------------------------------------------------------------------------------------
    def out(self, obj: dict[str, Any]) -> None:
        sys.stdout.buffer.write(json.dumps(obj, ensure_ascii=False).encode() + b"\n")
        sys.stdout.buffer.flush()

    def event(self, name: str, **fields: Any) -> None:
        self.out({"ev": name, **fields})

    def ok(self, rid: Any, data: dict[str, Any] | None = None) -> None:
        self.out({"id": rid, "ok": True, "data": data or {}})

    def fail(self, rid: Any, code: str, error: str) -> None:
        self.out({"id": rid, "ok": False, "code": code, "error": error})

    def exit(self, code: int) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)

    # -- commands ------------------------------------------------------------------------------------
    async def handle(self, line: bytes) -> None:
        try:
            r = json.loads(line)
        except ValueError:
            r = None
        if not isinstance(r, dict) or not isinstance(r.get("id"), int):
            self.fail(None, "bad_frame", "invalid JSON object")
            return
        rid, op = r["id"], r.get("op", "")
        self.ops.append(op)
        log(f"req id={rid} op={op}")
        if op == "hello":
            await self.op_hello(rid, r)
            return
        if not self.hello:
            self.fail(rid, "bad_args", "hello must be the first command")
            return
        if op in self.cfg.get("hang", []):
            return  # never answers (timeout tests)
        if (d := self.cfg.get("delay", {}).get(op)) is not None:
            await asyncio.sleep(float(d))  # answered late, so out of order
        h = getattr(self, f"op_{op}", None)
        if h is None:
            self.fail(rid, "unknown_op", "unknown op")
            return
        await h(rid, r)

    async def op_hello(self, rid: Any, r: dict[str, Any]) -> None:
        if r.get("protocol") != self.cfg.get("protocol", PROTOCOL):
            self.fail(rid, "version_mismatch", "this bridge speaks protocol " + self.cfg.get("protocol", PROTOCOL))
            self.exit(2)
        self.hello = True
        d: dict[str, Any] = {"protocol": PROTOCOL, "bridge_version": "fake", "linked": self.linked,
                             "allow": sorted(self.allow), "deny": list(self.a.deny), "dry_run": self.a.dry_run,
                             "test": self.a.test}
        if self.linked:
            d.update(jid=OWN_JID, lid=OWN_LID, device=OWN_DEVICE)
        if self.a.budget is not None:
            d["budget_left"] = self.a.budget - self.used
        self.ok(rid, d)
        if self.linked:
            self.connect()

    def connect(self) -> None:
        self.connected, self.pairing = True, False
        self.event("connected", jid=OWN_JID, lid=OWN_LID, device=OWN_DEVICE)

    def check_out(self, rid: Any, chat: str, new: bool, counted: bool = True) -> bool:
        """The outbound rules (§3), in order. False: a failure was sent."""
        now = time.time()
        if chat in self.deny:
            self.fail(rid, "denied", "the chat is on the never-list")
        elif chat not in self.allow:
            self.fail(rid, "not_allowed", "the chat is not allowlisted")
        elif new and len([t for t in self.sends if now - t < 60]) >= self.a.max_per_min:
            self.fail(rid, "rate", "the per-minute cap was reached")
        elif counted and self.a.budget is not None and self.used >= self.a.budget:
            self.fail(rid, "budget", "the budget is spent")
            self.event("budget_exhausted", used=self.used)
            self.exit(3)
        elif not self.linked:
            self.fail(rid, "not_linked", "no device is linked")
        elif not self.connected:
            self.fail(rid, "offline", "WhatsApp is offline")
        else:
            if counted:
                self.used += 1
            if new:
                self.sends.append(now)
            return True
        return False

    def new_id(self) -> str:
        self.n += 1
        mid = f"dry-{self.n}" if self.a.dry_run else f"FAKE{self.n:04d}"
        self.sent[mid] = time.time()
        return mid

    def write(self, **w: Any) -> None:
        if not self.a.dry_run:
            self.writes.append(w)

    async def op_send_text(self, rid: Any, r: dict[str, Any]) -> None:
        chat, text = r.get("chat", ""), r.get("text")
        if not chat or not isinstance(text, str) or not text:
            self.fail(rid, "bad_args", "chat and text are required")
            return
        if self.check_out(rid, chat, new=True):
            mid = self.new_id()
            self.write(op="send_text", chat=chat, msg=mid, len=len(text), reply_to=r.get("reply_to"))
            self.ok(rid, {"msg": mid})

    async def op_edit(self, rid: Any, r: dict[str, Any]) -> None:
        chat, msg, text = r.get("chat", ""), r.get("msg", ""), r.get("text")
        if not chat or not msg or not isinstance(text, str):
            self.fail(rid, "bad_args", "chat, msg and text are required")
            return
        if msg not in self.sent:
            self.fail(rid, "not_found", "unknown message")
        elif time.time() - self.sent[msg] > EDIT_WINDOW_S:
            self.fail(rid, "edit_window", "too old to edit")
        elif self.check_out(rid, chat, new=False):
            self.write(op="edit", chat=chat, msg=msg, len=len(text))
            self.ok(rid)

    async def op_react(self, rid: Any, r: dict[str, Any]) -> None:
        chat, msg, emoji = r.get("chat", ""), r.get("msg", ""), r.get("emoji")
        if not chat or not msg or not isinstance(emoji, str) or not isinstance(r.get("from_me"), bool):
            self.fail(rid, "bad_args", "chat, msg, from_me and emoji are required")
            return
        if self.check_out(rid, chat, new=False):
            self.write(op="react", chat=chat, msg=msg, from_me=r["from_me"], emoji=emoji)
            self.ok(rid)

    async def op_send_media(self, rid: Any, r: dict[str, Any]) -> None:
        chat, path, kind = r.get("chat", ""), Path(r.get("path", "")), r.get("kind")
        if not chat or kind not in ("document", "image"):
            self.fail(rid, "bad_args", "chat, path and kind (document|image) are required")
            return
        if not path.is_file():
            self.fail(rid, "bad_args", "the file could not be read")
            return
        if path.stat().st_size > 100 * 1024 * 1024:
            self.fail(rid, "too_large", "over 100 MB")
            return
        if self.check_out(rid, chat, new=True):
            mid = self.new_id()
            self.write(op="send_media", chat=chat, msg=mid, kind=kind, path=str(path), file_name=r.get("file_name"),
                       mime=r.get("mime"), caption_len=len(r.get("caption") or ""))
            self.ok(rid, {"msg": mid})

    async def op_presence(self, rid: Any, r: dict[str, Any]) -> None:
        if r.get("state") not in ("available", "unavailable"):
            self.fail(rid, "bad_args", "state must be available or unavailable")
            return
        self.write(op="presence", state=r["state"])
        self.ok(rid)

    async def op_chat_presence(self, rid: Any, r: dict[str, Any]) -> None:
        chat, state = r.get("chat", ""), r.get("state")
        if not chat or state not in ("composing", "paused"):
            self.fail(rid, "bad_args", "chat and state (composing|paused) are required")
            return
        if self.check_out(rid, chat, new=False, counted=False):
            self.write(op="chat_presence", chat=chat, state=state)
            self.ok(rid)

    async def op_set_allow(self, rid: Any, r: dict[str, Any]) -> None:
        if self.a.test:
            self.fail(rid, "test_mode", "set_allow is disabled under --test")
            return
        allow = r.get("allow")
        if not isinstance(allow, list):
            self.fail(rid, "bad_args", "allow is required")
            return
        if set(allow) & self.deny:
            self.fail(rid, "denied", "a JID is on the never-list")
            return
        self.allow = set(allow)
        self.ok(rid, {"allow": sorted(self.allow)})

    async def op_status(self, rid: Any, r: dict[str, Any]) -> None:
        d: dict[str, Any] = {"linked": self.linked, "connected": self.connected,
                             "sent_last_min": len([t for t in self.sends if time.time() - t < 60]),
                             "total_outbound": self.used, "dropped": dict(self.dropped)}
        if self.a.budget is not None:
            d["budget_left"] = self.a.budget - self.used
        self.ok(rid, d)

    async def op_list_chats(self, rid: Any, r: dict[str, Any]) -> None:
        self.ok(rid, {"chats": self.cfg.get("chats", [])})

    async def op_pair(self, rid: Any, r: dict[str, Any]) -> None:
        if self.a.test:
            self.fail(rid, "test_mode", "pair is disabled under --test")
        elif self.linked:
            self.fail(rid, "already_linked", "a device is already linked")
        else:
            self.pairing = True
            self.ok(rid)
            self.event("qr", code="fake-qr-1", expires_s=20)

    async def op_logout(self, rid: Any, r: dict[str, Any]) -> None:
        if self.a.test:
            self.fail(rid, "test_mode", "logout is disabled under --test")
            return
        if not self.linked:
            self.fail(rid, "not_linked", "no device is linked")
            return
        self.linked = self.connected = False
        self.ok(rid)
        self.event("logged_out", reason="logout")

    # -- control (tests) ---------------------------------------------------------------------------------
    def inbound(self, c: dict[str, Any]) -> dict[str, Any]:
        chat, msg, text = c.get("chat", ""), c.get("msg", ""), c.get("text", "")
        device = int(c.get("sender_device", 0))
        now = float(c.get("now") or time.time())  # the test's clock, never the wall clock
        ts = float(c.get("ts") or now)
        why = ""
        if chat in self.deny:
            why = "denied"
        elif chat not in self.allow:
            why = "not_allowed"
        elif not c.get("from_me", True):
            why = "not_owner"
        elif device not in self.a.trusted:
            why = "untrusted_device"
        elif device == OWN_DEVICE or msg in self.sent or text.startswith(PREFIX):
            why = "echo"
        elif c.get("broadcast") or now - ts > MAX_AGE_S:
            why = "old_or_broadcast"
        if why:
            self.dropped[why] = self.dropped.get(why, 0) + 1
            log(f"dropped{{{why}}}" + ("" if why in ("denied", "not_allowed") else f" msg={msg}"))
            return {"ok": True, "emitted": False, "reason": why}
        ev: dict[str, Any] = {"ev": "message", "chat": chat, "msg": msg, "sender_device": device, "from_me": True, "text": text,
                              "sent_at": ts, "replayed": bool(c.get("replayed")), "attachments": [],
                              "unsupported": ""}
        if c.get("quoted"):
            ev["quoted"] = c["quoted"]
        if c.get("reaction"):
            ev["reaction"], ev["text"] = c["reaction"], ""
        media = c.get("media") or {}
        if media.get("kind") in UNSUPPORTED:
            ev["unsupported"] = media["kind"]  # never downloaded
        elif media.get("kind") in ("image", "document"):
            self.media.mkdir(parents=True, exist_ok=True, mode=0o700)
            name = media.get("name", "file")
            path = self.media / f"{msg}-{name}"
            path.write_bytes(b"fake media")
            self.downloads += 1
            ev["attachments"] = [{"kind": media["kind"], "path": str(path), "mime": media.get("mime", ""),
                                  "name": name, "size": path.stat().st_size}]
        self.out(ev)
        log(f"message msg={msg} device={device} len={len(text)} media={media.get('kind', '')}")
        return {"ok": True, "emitted": True}

    def control(self, c: dict[str, Any]) -> dict[str, Any]:
        cmd = c.get("cmd")
        if cmd == "inbound":
            return self.inbound(c)
        if cmd == "emit":
            self.out(c["event"])
        elif cmd == "chat_changed":
            if c["chat"] in self.allow and c["chat"] not in self.deny:
                self.event("chat_changed", chat=c["chat"], subject=c.get("subject", ""),
                           members=c.get("members", 1), owner_only=c.get("owner_only", True))
        elif cmd == "offline":
            self.connected = False
            self.event("offline", reason="network")
        elif cmd == "reconnect":
            self.connect()
        elif cmd == "scan":
            if not self.pairing:
                return {"ok": False}
            self.linked = True
            self.connect()
        elif cmd == "exit":
            self.exit(int(c.get("code", 0)))
        elif cmd == "stats":
            return {"ok": True, "writes": self.writes, "dropped": self.dropped, "allow": sorted(self.allow),
                    "deny": sorted(self.deny), "argv": sys.argv[1:], "downloads": self.downloads, "ops": self.ops,
                    "used": self.used}
        return {"ok": True}


async def main(argv: list[str]) -> int:
    try:
        a = parse(argv)
    except (ValueError, argparse.ArgumentError, SystemExit):
        log("bad flags")
        return 2
    fake_dir = Path(a.fake_dir)
    fake_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = fake_dir / "fake.json"
    cfg: dict[str, Any] = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    starts = fake_dir / "starts"
    start_no = len(starts.read_text().splitlines()) if starts.exists() else 0
    with starts.open("a") as f:
        f.write(f"{os.getpid()}\n")
    codes = cfg.get("exit_codes", [])
    if start_no < len(codes) and codes[start_no] is not None:
        log(f"scripted exit {codes[start_no]}")
        return int(codes[start_no])

    store = Path(a.store)
    store.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = open(f"{store}.lock", "w")  # noqa: ASYNC230, SIM115 (held for the process lifetime)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("store locked")
        return 75

    fake = Fake(a, cfg, start_no)
    loop = asyncio.get_running_loop()
    sock = fake_dir / "c.sock"
    with contextlib.suppress(FileNotFoundError):
        sock.unlink()

    async def serve(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        while line := await r.readline():
            w.write(json.dumps(fake.control(json.loads(line))).encode() + b"\n")
            await w.drain()
        w.close()

    server = await asyncio.start_unix_server(serve, path=str(sock))
    fake.event("ready", protocol=PROTOCOL, bridge_version="fake", linked=fake.linked)
    log(f"ready linked={fake.linked} allow={len(fake.allow)} deny={len(fake.deny)}")

    eof = asyncio.Event()
    tasks: set[asyncio.Task[None]] = set()

    def feed(line: bytes) -> None:
        t = loop.create_task(fake.handle(line))
        tasks.add(t)
        t.add_done_callback(tasks.discard)

    def reader() -> None:
        for line in sys.stdin.buffer:
            loop.call_soon_threadsafe(feed, line)
        loop.call_soon_threadsafe(eof.set)

    threading.Thread(target=reader, daemon=True).start()
    await eof.wait()  # the lifeline: chatd is gone
    log("stdin closed; exiting")
    server.close()
    return 0


if __name__ == "__main__":
    code = asyncio.run(main(sys.argv[1:]))
    sys.stdout.flush()
    os._exit(code)
