"""`mac` (scripts/mac): link your WhatsApp and choose the groups Mac answers in.

    mac whatsapp link                      show a QR code, link, pick groups
    mac chats [list]                       the groups Mac has settings for
    mac chats add "<group>"                answer in a group (by its WhatsApp name)
    mac chats on|off "<group>"             turn a group on or off
    mac chats allow|deny "<group>" <what>  e.g. deny "Family" chrome, the web
    mac chats remove "<group>"             forget a group's settings
    mac status                             the services' status (scripts/services.sh status)

`link` runs the WhatsApp bridge itself, so it stops the background services while it runs (the
bridge's data allows one user at a time) and starts them again afterwards. It writes
~/.mac-agent/chat/pairing.json (0600): every chat the bridge lists, `{name: {id, members,
owner_only}}`, the shape agentd reads for the never-list and its one-time seed. Never-listed
groups ("Timetable") are kept in that file, so they stay blocked, but they are never offered.

The `chats` commands go through agentd's app-kind profile ops with the per-boot app token that
macsvc writes to ~/.mac-agent/run/app-token. `add` finds the group's id in the chat list chatd
keeps (chat/chats.json), asking chatd to refresh it when the name is not there yet.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

LABEL = "com.mac-agent.services"
CHANNEL = "whatsapp"
PROTOCOL = "2"
WAIT_AGENT_S = 30.0
PAIR_WAIT_S = 300.0
LIST_RETRIES = 10


class CliError(Exception):
    """A refusal or failure, in plain words."""


# -- paths and names ---------------------------------------------------------------------------------------
def mac_home() -> Path:
    from macd import config

    return Path(config.HOME)


def norm(name: str) -> str:
    from agentd.profiles import normalize_subject

    return normalize_subject(name)


def never(name: str) -> bool:
    from agentd.profiles import is_never_subject

    return is_never_subject(name)


def capability_names() -> dict[str, str]:
    from agentd import schema as s

    return {**{k: k for k in s.CAPABILITIES}, **{k.replace("_", " "): k for k in s.CAPABILITIES},
            **{w.casefold(): k for k, w in s.CAPABILITY_WORDS.items()}}


def parse_capabilities(words: list[str]) -> list[str]:
    """`chrome, the web` -> ["chrome", "web"]. Several are separated by commas."""
    from agentd import schema as s

    names, out = capability_names(), []
    for part in " ".join(words).split(","):
        key = names.get(" ".join(part.split()).casefold())
        if key is None:
            options = ", ".join(s.CAPABILITY_WORDS.values())
            raise CliError(f'"{part.strip()}" is not something Mac can allow or deny. Use one of: {options}.')
        out.append(key)
    if not out:
        raise CliError("Say what to allow or deny, for example: the web")
    return out


# -- the LaunchAgent ---------------------------------------------------------------------------------------
class Service:
    def __init__(self) -> None:
        self.launchctl = os.environ.get("LAUNCHCTL", "launchctl")
        self.domain = f"gui/{os.getuid()}"
        self.plist = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"

    def _run(self, *args: str) -> int:
        return subprocess.run([self.launchctl, *args], capture_output=True, check=False).returncode

    def loaded(self) -> bool:
        return self._run("print", f"{self.domain}/{LABEL}") == 0

    def stop(self) -> None:
        self._run("bootout", f"{self.domain}/{LABEL}")
        deadline = time.monotonic() + 60
        while self.loaded():
            if time.monotonic() > deadline:
                raise CliError("Mac's services did not stop. Try again in a minute.")
            time.sleep(0.5)

    def start(self) -> bool:
        return self._run("bootstrap", self.domain, str(self.plist)) == 0


# -- the bridge ---------------------------------------------------------------------------------------------
class BridgeFailed(CliError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class Bridge:
    """The WhatsApp bridge as a child process, speaking its stdio protocol."""

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        self.proc: asyncio.subprocess.Process | None = None
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.replies: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self.next_id = 0
        self.reader: asyncio.Task[None] | None = None

    async def start(self) -> dict[str, Any]:
        """Start it and say hello. Raises CliError when the store is in use (exit 75)."""
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *self.argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, limit=2 * 1024 * 1024)
        except OSError as e:
            raise CliError("The WhatsApp helper isn't installed. Run install.sh, or scripts/services.sh install.") from e
        self.reader = asyncio.create_task(self._read())
        ready = await self.event("ready", timeout=30)
        if ready is None:
            code = await self.proc.wait()
            if code == 75:
                raise CliError("WhatsApp's data is in use by another Mac process (the background services or a "
                               "test run). Stop it with scripts/services.sh uninstall, then try again.")
            raise CliError(f"The WhatsApp helper stopped (code {code}).")
        return await self.call("hello", protocol=PROTOCOL)

    async def _read(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        async for raw in self.proc.stdout:
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            if "ev" in obj:
                self.events.put_nowait(obj)
            elif (fut := self.replies.pop(obj.get("id"), None)) is not None and not fut.done():
                fut.set_result(obj)
        self.events.put_nowait({"ev": "_exit"})
        for fut in self.replies.values():
            if not fut.done():
                fut.set_result({"ok": False, "code": "offline", "error": "the helper stopped"})

    async def call(self, op: str, timeout: float = 30.0, **args: Any) -> dict[str, Any]:
        assert self.proc is not None and self.proc.stdin is not None
        self.next_id += 1
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self.replies[self.next_id] = fut
        self.proc.stdin.write((json.dumps({"id": self.next_id, "op": op, **args}) + "\n").encode())
        await self.proc.stdin.drain()
        try:
            r = await asyncio.wait_for(fut, timeout)
        except TimeoutError as e:
            raise CliError("The WhatsApp helper did not answer.") from e
        if not r.get("ok"):
            raise BridgeFailed(str(r.get("code", "")), str(r.get("error", "")))
        data = r.get("data")
        return data if isinstance(data, dict) else {}

    async def event(self, *names: str, timeout: float) -> dict[str, Any] | None:
        """The next event among `names` (others are skipped); None if the helper stopped or time ran out."""
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            left = deadline - asyncio.get_running_loop().time()
            if left <= 0:
                return None
            try:
                ev = await asyncio.wait_for(self.events.get(), left)
            except TimeoutError:
                return None
            if ev["ev"] == "_exit":
                return None
            if ev["ev"] in names:
                return ev

    async def close(self) -> None:
        if self.proc is None:
            return
        if self.proc.stdin is not None:
            with contextlib.suppress(Exception):
                self.proc.stdin.close()  # the lifeline: the bridge disconnects and exits 0
        try:
            await asyncio.wait_for(self.proc.wait(), 10)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                self.proc.kill()
            await self.proc.wait()
        if self.reader is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.reader, 2)


def bridge_argv(home: Path) -> tuple[list[str], frozenset[str]]:
    """The same bridge, store and never-list chatd would use (chat/config.toml)."""
    from agentd.profiles import load_never_set
    from chatd.__main__ import channel_cfg, load_config

    chat = home / "chat"
    cfg = load_config(chat / "config.toml")
    sec = channel_cfg(cfg, CHANNEL, home)
    ns = load_never_set(chat / "config.toml", chat / "pairing.json")
    deny = ns.jids if ns.ok else frozenset()
    command = [str(x) for x in sec.get("command") or [Path(sec["bridge"]).expanduser()]]
    return [*command, "--store", str(Path(sec["store"]).expanduser()),
            "--media-dir", str(Path(sec["media_dir"]).expanduser()),
            "--allow=", f"--deny={','.join(sorted(deny))}"], deny


# -- QR in the terminal -----------------------------------------------------------------------------------
def qr_text(code: str) -> str:
    """The code as a QR, two rows per line, black on white whatever the terminal's colours."""
    try:
        import Quartz
        from Foundation import NSData
    except ImportError:
        return f"(Can't draw the QR code here. Code: {code})"
    f = Quartz.CIFilter.filterWithName_("CIQRCodeGenerator")
    raw = code.encode()
    f.setValue_forKey_(NSData.dataWithBytes_length_(raw, len(raw)), "inputMessage")
    f.setValue_forKey_("L", "inputCorrectionLevel")
    img = f.outputImage()
    ext = img.extent()
    cg = Quartz.CIContext.contextWithOptions_(None).createCGImage_fromRect_(img, ext)
    w, h = int(ext.size.width), int(ext.size.height)
    bpr, bpp = Quartz.CGImageGetBytesPerRow(cg), Quartz.CGImageGetBitsPerPixel(cg) // 8
    px = bytes(Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(cg)))
    pad = 2
    dark = [[False] * (w + 2 * pad) for _ in range(pad)]
    dark += [[False] * pad + [px[y * bpr + x * bpp] < 128 for x in range(w)] + [False] * pad for y in range(h)]
    dark += [[False] * (w + 2 * pad) for _ in range(pad + 1)]
    lines = []
    for y in range(0, len(dark) - 1, 2):
        cells = []
        for top, bottom in zip(dark[y], dark[y + 1], strict=True):
            fg = 30 if top else 97  # the upper half block in the top colour, on the bottom colour
            bg = 40 if bottom else 107
            cells.append(f"\x1b[{fg};{bg}m▀")
        lines.append("".join(cells) + "\x1b[0m")
    return "\n".join(lines)


# -- agentd as the app ------------------------------------------------------------------------------------
class Agent:
    def __init__(self, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        self.r, self.w, self.next_id = r, w, 0

    @classmethod
    async def connect(cls, home: Path, wait_s: float = 0.0) -> Agent:
        token_path, sock = home / "run" / "app-token", home / "run" / "agent.sock"
        deadline = time.monotonic() + wait_s
        while True:
            try:
                token = token_path.read_text().strip()
                r, w = await asyncio.open_unix_connection(str(sock), limit=4 * 1024 * 1024)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise CliError("Mac's services aren't running. Start them with: scripts/services.sh install") from None
                await asyncio.sleep(0.5)
        from agentd import schema as s

        a = cls(r, w)
        await a.call("hello", api=s.API_VERSION, token=token, client="mac-cli/1", client_epoch="cli",
                     conversations=[])
        return a

    async def call(self, op: str, **args: Any) -> dict[str, Any]:
        self.next_id += 1
        self.w.write((json.dumps({"id": self.next_id, "op": op, **args}) + "\n").encode())
        await self.w.drain()
        while True:
            line = await asyncio.wait_for(self.r.readline(), 30)
            if not line:
                raise CliError("Mac's agent closed the connection.")
            obj = json.loads(line)
            if obj.get("id") != self.next_id:
                continue  # events and notices
            if not obj.get("ok"):
                raise CliError(str(obj.get("error") or obj.get("code") or "Mac's agent refused."))
            data = obj.get("data")
            return data if isinstance(data, dict) else {}

    async def close(self) -> None:
        self.w.close()
        with contextlib.suppress(Exception):
            await self.w.wait_closed()

    async def profiles(self) -> list[dict[str, Any]]:
        rows = (await self.call("profile_list")).get("profiles") or []
        return [p for p in rows if p.get("channel") == CHANNEL]

    async def find(self, name: str) -> dict[str, Any]:
        hits = [p for p in await self.profiles() if norm(p.get("display_name", "")) == norm(name)]
        if not hits:
            raise CliError(f'Mac has no settings for a group called "{name.strip()}". See them with: mac chats')
        if len(hits) > 1:
            raise CliError(f'More than one group is called "{name.strip()}". Rename one in WhatsApp first.')
        return hits[0]

    async def add(self, chat: dict[str, Any]) -> str:
        name = str(chat.get("subject") or "").strip()
        try:
            await self.call("profile_create", channel=CHANNEL, chat_id=chat["id"], display_name=name[:80] or chat["id"],
                            owner_only=bool(chat.get("owner_only", False)))
        except CliError as e:
            return f"{name}: {e}"
        how = "answers your own messages" if chat.get("owner_only") else 'answers your messages that start with "Mac"'
        return f"Added {name}. Mac {how} there."


# -- the chat list ---------------------------------------------------------------------------------------
def write_pairing(home: Path, chats: list[dict[str, Any]]) -> None:
    """Every listed chat as {name: {id, members, owner_only}}. Two chats with one name get
    invisible word joiners appended (a format character, dropped by the never-list's match), so
    two groups both named Timetable both stay blocked."""
    from agentd.stores import atomic_write_json

    out: dict[str, dict[str, Any]] = {}
    for c in chats:
        name = str(c.get("subject") or "").strip() or str(c["id"])
        while name in out:
            name += "⁠"
        out[name] = {"id": c["id"], "members": int(c.get("members") or 0), "owner_only": bool(c.get("owner_only"))}
    (home / "chat").mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write_json(home / "chat" / "pairing.json", out)


def known_chats(home: Path) -> list[dict[str, Any]]:
    """chatd's fresh list first, then the list saved when linking."""
    out: list[dict[str, Any]] = []
    with contextlib.suppress(OSError, ValueError):
        rows = json.loads((home / "chat" / "chats.json").read_text())
        out += [r for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)]
    with contextlib.suppress(OSError, ValueError):
        for name, e in json.loads((home / "chat" / "pairing.json").read_text()).items():
            if isinstance(e, dict) and isinstance(e.get("id"), str):
                out.append({**e, "subject": name})
    return out


def match(chats: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    seen, out = set(), []
    for c in chats:
        if norm(str(c.get("subject") or "")) == norm(name) and c["id"] not in seen:
            seen.add(c["id"])
            out.append(c)
    return out


def ask_chatd_to_refresh(home: Path) -> bool:
    """SIGUSR1 to the running chatd (pid from services.json); True if one was asked."""
    try:
        st = json.loads((home / "run" / "services.json").read_text())
        pid = int(st["children"]["chatd"]["pid"])
        if st["children"]["chatd"]["state"] != "running":
            return False
        os.kill(pid, signal.SIGUSR1)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


async def resolve(home: Path, name: str) -> dict[str, Any]:
    if never(name):
        raise CliError("Mac never uses Timetable.")
    hits = match(known_chats(home), name)
    if not hits and ask_chatd_to_refresh(home):
        path = home / "chat" / "chats.json"
        before = path.stat().st_mtime if path.exists() else 0.0
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not hits:
            await asyncio.sleep(0.3)
            if path.exists() and path.stat().st_mtime != before:
                hits = match(known_chats(home), name)
                before = path.stat().st_mtime
    if not hits:
        raise CliError(f'WhatsApp has no group called "{name.strip()}" that Mac can see. Check the name, '
                       "or run: mac whatsapp link")
    if len(hits) > 1:
        raise CliError(f'More than one group is called "{name.strip()}". Rename one in WhatsApp first.')
    return hits[0]


# -- commands ---------------------------------------------------------------------------------------------
def pick(chats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offered = [c for c in chats if not never(str(c.get("subject") or ""))]
    if not offered:
        print("WhatsApp lists no groups yet. Add one later with: mac chats add \"<group>\"")
        return []
    print("\nYour WhatsApp chats:")
    for i, c in enumerate(offered, 1):
        kind = "just you" if c.get("owner_only") else f"{c.get('members', '?')} people"
        print(f"  {i:>2}. {str(c.get('subject') or '').strip() or c['id']} ({kind})")
    print("\nWhich should Mac answer in? Type numbers (for example: 1 3) or a name. Press Return for none.")
    try:
        line = input("> ").strip()
    except EOFError:
        return []
    if not line:
        return []
    tokens = line.replace(",", " ").split()
    if all(t.isdigit() for t in tokens):
        out = []
        for t in tokens:
            if not 1 <= int(t) <= len(offered):
                raise CliError(f"There is no number {t} in the list.")
            out.append(offered[int(t) - 1])
        return out
    if never(line):
        raise CliError("Mac never uses Timetable.")
    hits = match(offered, line)
    if len(hits) != 1:
        raise CliError(f'No single chat is called "{line}". Run mac whatsapp link again.')
    return hits


async def link(home: Path) -> int:
    svc = Service()
    was_running = svc.loaded()
    if was_running:
        print("Stopping Mac's services while linking (WhatsApp's data allows one user at a time).")
        svc.stop()
    picks: list[dict[str, Any]] = []
    bridge: Bridge | None = None
    try:
        argv, _ = bridge_argv(home)
        bridge = Bridge(argv)
        hello = await bridge.start()
        if not hello.get("linked"):
            await bridge.call("pair")
            print("On your phone, open WhatsApp > Settings > Linked devices > Link a device, then scan this:\n")
            deadline = time.monotonic() + PAIR_WAIT_S
            while True:
                ev = await bridge.event("qr", "connected", "error", timeout=max(1.0, deadline - time.monotonic()))
                if ev is None:
                    raise CliError("Linking stopped before your phone scanned the code. Run mac whatsapp link again.")
                if ev["ev"] == "qr":
                    print(qr_text(str(ev.get("code", ""))) + "\n", flush=True)
                elif ev["ev"] == "connected":
                    break
                else:
                    raise CliError("The QR code expired. Run mac whatsapp link again.")
            print("Linked.")
        else:
            print("This Mac is already linked to your WhatsApp.")
        chats: list[dict[str, Any]] = []
        for _ in range(LIST_RETRIES):  # right after linking, WhatsApp may still be syncing groups
            with contextlib.suppress(BridgeFailed):
                data = await bridge.call("list_chats")
                chats = [c for c in data.get("chats") or [] if isinstance(c, dict) and isinstance(c.get("id"), str)]
            if chats:
                break
            await asyncio.sleep(1)
        write_pairing(home, chats)
        picks = pick(chats)
    finally:
        if bridge is not None:
            await bridge.close()
        if was_running:
            print("Starting Mac's services again." if svc.start() else
                  "Mac's services did not start again. Run: scripts/services.sh restart")
    if not picks:
        return 0
    if not svc.loaded():
        print("\nMac's services aren't running, so the groups aren't added yet. Run:")
        print("  scripts/services.sh install")
        for c in picks:
            print(f'  mac chats add "{str(c.get("subject") or "").strip()}"')
        return 0
    agent = await Agent.connect(home, wait_s=WAIT_AGENT_S)
    try:
        for c in picks:
            print(await agent.add(c))
    finally:
        await agent.close()
    return 0


def describe(p: dict[str, Any]) -> str:
    from agentd import schema as s

    caps = p.get("capabilities") or {}
    off = [s.CAPABILITY_WORDS[k] for k in s.CAPABILITIES if not caps.get(k)]
    state = "on" if p.get("enabled") else "off"
    trigger = 'messages starting with "Mac"' if p.get("trigger") == "prefix" else "all your messages"
    tail = f"Not allowed: {', '.join(off)}." if off else "Everything allowed."
    return f"{p.get('display_name')}: {state}, answers {trigger}. {tail}"


async def chats_cmd(home: Path, args: list[str]) -> int:
    sub = args[0] if args else "list"
    if sub not in ("list", "add", "on", "off", "allow", "deny", "remove"):
        raise CliError(f"Unknown command: mac chats {sub}. See: mac help")
    if sub != "list" and len(args) < 2:
        raise CliError(f'Name the group, for example: mac chats {sub} "Family"')
    if sub in ("allow", "deny") and len(args) < 3:
        raise CliError(f'Say what to {sub}, for example: mac chats {sub} "Family" the web')
    agent = await Agent.connect(home)
    try:
        if sub == "list":
            rows = await agent.profiles()
            if not rows:
                print('Mac answers in no groups yet. Add one with: mac chats add "<group>"')
            for p in sorted(rows, key=lambda p: norm(p.get("display_name", ""))):
                print(describe(p))
            return 0
        name = args[1]
        if sub == "add":
            msg = await agent.add(await resolve(home, name))
            print(msg)
            return 0 if msg.startswith("Added") else 1
        p = await agent.find(name)
        if sub == "remove":
            await agent.call("profile_delete", channel=CHANNEL, chat_id=p["chat_id"])
            print(f"Removed {p['display_name']}. Mac no longer answers there.")
            return 0
        if sub in ("on", "off"):
            patch: dict[str, Any] = {"enabled": sub == "on"}
        else:
            patch = {"capabilities": {k: sub == "allow" for k in parse_capabilities(args[2:])}}
        new = await agent.call("profile_set", channel=CHANNEL, chat_id=p["chat_id"], rev=p["rev"], patch=patch)
        print(describe(new.get("profile") or new))
        return 0
    finally:
        await agent.close()


HELP = __doc__.split("\n\n")[1] if __doc__ else ""


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    home = mac_home()
    try:
        if args[:2] == ["whatsapp", "link"]:
            return asyncio.run(link(home))
        if args[:1] == ["chats"]:
            return asyncio.run(chats_cmd(home, args[1:]))
        print("Usage:\n" + HELP)
        return 0 if args[:1] in ([], ["help"], ["-h"], ["--help"]) else 2
    except CliError as e:
        print(str(e), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
