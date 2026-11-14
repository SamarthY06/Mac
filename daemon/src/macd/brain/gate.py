"""Safety gate (ARCHITECTURE §6): classify tool calls and hold confirmation tokens.

`classify()` is a pure function (no command is ever executed) that maps a Claude tool call
to a `GateVerdict`:

- ``catastrophic``  - always denied (rm of roots/home, disk erase, sudo, fork bombs, ...)
- ``sensitive_read`` - always denied (keys, keychains, cloud creds, browser profiles, ...)
- ``outward``        - allowed only with a confirmation token minted from a human "yes"
- ``allow``

`analyze()` returns the richer `GateDecision` (verdict + internal reason code + reason).
Writes to the workspace CLAUDE.md / memory.md are reported with code ``memory_write`` and
verdict ``sensitive_read`` (a hard deny), GateVerdict itself is unchanged.

Desktop and browser rules are heuristics (ARCHITECTURE §6): the frontmost app is queried at
decision time; coordinate clicks, Return and line breaks in a messaging app are outward; Chrome
clicks/keys are outward when the element label matches an outward verb OR the tab's host is a
mail/chat/social/payment site (hosts tracked from navigate/tabs_context). Scripts passed as
files to osascript/shells/interpreters are read and classified. Remaining blind spots (prompt +
event log only): values computed at run time (`$(base64 -d …)`), clicks on unknown hosts whose
labels avoid the verb list, and read-only exfiltration through GET URLs.

Chat profile (WhatsApp ARCHITECTURE W3.3), switched on through `GateContext`:
- ``whatsapp_ui_block``: every use of WhatsApp's own screens is catastrophic with code
  ``whatsapp_ui``, reads included (M1): Claude in Chrome on a web.whatsapp.com tab or URL, the
  mac ui/screenshot/open_app tools with WhatsApp targeted or frontmost, and osascript, ``open``
  or ``whatsapp://`` URLs aimed at WhatsApp. The voice profile keeps today's behaviour (V6,
  Option A: WhatsApp sends behind the spoken confirmation).
- ``never_chat_names``: an outward action or a confirmation summary naming a never-listed chat
  (for example "Timetable") is catastrophic with code ``never_chat``.
- ``.env`` files are readable only inside ``env_exempt_dir`` (default: the workspace), and only
  when that folder is one of Mac's own workspaces under ``~/.mac-agent`` (F10).
- ``capabilities`` (v3.4, W9.5): the chat's granted tool classes (a set, or a callable read at
  every decision; None = voice, no check). Every tool has one row in ``TOOL_MAP``; a call needs
  every class `tool_classes()` gives it; an unmapped tool (user MCP servers, Skill, ...) needs none.
  Order: whatsapp_ui and the never-list, then a missing capability, then the rules above.

`ConfirmationGate` is owned by the orchestrator and shared with the brain: the model asks
via ``mcp__mac__request_confirmation``; the orchestrator speaks the summary, listens for a
human yes/no and calls ``resolve()``. Approval mints a token (single turn, TTL, max uses)
that the PreToolUse hook consumes when it lets an outward action through.
"""

from __future__ import annotations

import asyncio
import fnmatch
import functools
import hashlib
import logging
import os
import posixpath
import re
import sys
import time
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentd.schema import (  # one table for what the user sees (UX5)
    CAPABILITIES,
    CAPABILITY_WORDS,
)
from macd import config
from macd.brain import shell
from macd.brain.prompts import check_persona
from macd.contracts import GateVerdict

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- results

_RANK: dict[str, int] = {"allow": 0, "outward": 1, "sensitive_read": 2, "catastrophic": 3}

MEMORY_WRITE = "memory_write"  # internal reason code (verdict: sensitive_read → deny)


@dataclass(frozen=True)
class GateDecision:
    verdict: GateVerdict
    code: str = "ok"
    reason: str = ""
    target: str = ""  # normalised recipient/host/label; part of the confirmation fingerprint
    detail: str = ""  # shown with the confirmation, not part of the fingerprint (set_persona's text)

    @property
    def always_denied(self) -> bool:
        return self.verdict in ("catastrophic", "sensitive_read")


ALLOW = GateDecision("allow")


def _worst(decisions: Iterable[GateDecision]) -> GateDecision:
    best = ALLOW
    for d in decisions:
        if _RANK[d.verdict] > _RANK[best.verdict]:
            best = d
    return best


# ----------------------------------------------------------------------------- tables
# Paths use "~" for the user's home; matching is case-insensitive (APFS default).

SENSITIVE_ROOTS: tuple[str, ...] = (
    "~/.ssh",
    "~/Library/Keychains",
    "/Library/Keychains",
    "~/.aws",
    "~/.config/gh",
    "~/.config/gcloud",
    "~/.azure",
    "~/.kube/config",
    "~/.docker/config.json",
    "~/.netrc",
    "~/.gnupg",
    "~/.git-credentials",
    "~/.npmrc",
    "~/.pypirc",
    "~/.password-store",
    "~/.claude/.credentials.json",
    "~/Library/Cookies",
    "~/Library/Safari",
    "~/Library/Containers/com.apple.Safari",
    "~/Library/Application Support/Google/Chrome",
    "~/Library/Application Support/Chromium",
    "~/Library/Application Support/BraveSoftware",
    "~/Library/Application Support/Arc",
    "~/Library/Application Support/Microsoft Edge",
    "~/Library/Application Support/Vivaldi",
    "~/Library/Application Support/Firefox",
    "~/Library/Application Support/1Password",
)

# I1: Mac's own policy, tokens, binaries and WhatsApp keys, under ~/.mac-agent (and MAC_HOME).
# Never read or written by any tool: the profile store, pairing.json, chat config, the bridge binary,
# dev tokens, device.db. The workspace and agent/chats/* stay usable.
MAC_PRIVATE: tuple[str, ...] = ("agent/state", "run", "bin", "chat", "config.toml")
MAC_READABLE: tuple[str, ...] = ("chat/whatsapp/media",)  # inbound attachments: read, never written
# H1 (security audit): nothing under ~/.mac-agent is written by a tool except these two (release
# snapshots, models, voice, state.json, logs included), and nothing that starts Mac or a shell.
MAC_WRITABLE: tuple[str, ...] = ("workspace", "agent/chats", "agent/chat-workspace")
WRITE_DENIED: tuple[str, ...] = ("~/Library/LaunchAgents", "~/.local/bin")
# The owner's own shell and Claude Code setup: never from chat; by voice behind a spoken yes.
OWNER_CONFIG: tuple[str, ...] = (
    "~/.claude/settings*.json", "~/.claude.json", "~/.claude/CLAUDE.md", "~/.claude/hooks",
    "~/.zshrc", "~/.zprofile", "~/.bashrc", "~/.bash_profile", "~/.profile",
)
PROTECTED_WRITE = "protected_write"
OWNER_CONFIG_WRITE = "owner_config"


@functools.cache
def _own_code_roots() -> tuple[str, ...]:
    """Mac's own code: the daemon's src and venvs, the running interpreter's venv, and the repo."""
    here = Path(__file__).resolve()
    daemon = here.parents[3]  # .../daemon/src/macd/brain/gate.py
    roots = {str(daemon / "src"), str(daemon / ".venv"), str(Path(sys.prefix).resolve())}
    venv = Path(sys.prefix).resolve()
    if venv.name == ".venv" and venv.parent.name == "daemon":  # a snapshot run uses the repo's venv
        roots |= {str(venv.parent / "src")}
    return tuple(sorted(roots))


@functools.cache
def _repo_roots() -> tuple[str, ...]:
    """The repo (and its daemon dir) for the catastrophic rm rule."""
    out = {str(Path(__file__).resolve().parents[4]), str(Path(__file__).resolve().parents[3])}
    venv = Path(sys.prefix).resolve()
    if venv.name == ".venv" and venv.parent.name == "daemon":
        out |= {str(venv.parent), str(venv.parent.parent)}
    return tuple(sorted(out))

SENSITIVE_BASENAMES: tuple[str, ...] = (
    "*.pem", "*.key", "*.p12", "*.pfx", "*.keychain", "*.keychain-db",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*", ".netrc", ".pgpass", ".env*",
)
# Representative names used to decide whether a *glob* could select a sensitive file.
_SENSITIVE_SAMPLES: tuple[str, ...] = (
    "id_rsa", "id_ed25519", "server.pem", "server.key", "cert.p12", "login.keychain-db",
    ".env", ".env.local",
)
ENV_TEMPLATE_NAMES = {".env.example", ".env.sample", ".env.template", ".env.dist"}

# Deleting / recursively chmod-ing these is catastrophic. "*" matches one path segment.
PROTECTED_ROOTS: tuple[str, ...] = (
    "/", "/System", "/Library", "/Applications", "/Users", "/Users/*", "/usr", "/bin",
    "/sbin", "/etc", "/var", "/private", "/private/etc", "/private/var", "/opt", "/Volumes",
    "/Volumes/*", "/dev", "/cores",
    "~", "~/Documents", "~/Desktop", "~/Library", "~/Pictures", "~/Movies", "~/Music",
    "~/Downloads", "~/.mac-agent",
)

OUTWARD_VERBS = re.compile(
    r"\b(send(s|ing)?|post(s|ed|ing)?|publish(es|ed|ing)?|submit(s|ted|ting)?|buy(s|ing)?|"
    r"purchas(e|es|ed|ing)|pay(s|ing|ment)?|check ?out|order(s|ed|ing)?|place order|"
    r"delet(e|es|ed|ing)|remove account|transfer(s|red|ring)?|confirm(s|ed|ing)?|"
    r"tweet(s|ed|ing)?|repl(y|ies|ied|ying)|share now)\b",
    re.IGNORECASE,
)
MESSAGING_APPS = re.compile(
    r"whatsapp|slack|telegram|discord|messages|signal|teams|messenger|wechat|mail|outlook|"
    r"spark|airmail|linkedin|twitter|\bx\b",
    re.IGNORECASE,
)

# Chrome tabs on these hosts: any click / Enter is outward (labels come from the page itself).
OUTWARD_HOSTS = re.compile(
    r"(^|\.)(mail\.google\.com|outlook\.(live|office)\.com|mail\.yahoo\.com|mail\.proton\.me|"
    r"web\.whatsapp\.com|slack\.com|discord\.com|web\.telegram\.org|teams\.microsoft\.com|"
    r"messenger\.com|linkedin\.com|twitter\.com|x\.com|facebook\.com|instagram\.com|"
    r"threads\.net|reddit\.com|paypal\.com|stripe\.com|amazon\.[a-z.]+|ebay\.[a-z.]+|venmo\.com|"
    r"coinbase\.com|binance\.com|robinhood\.com|[a-z0-9-]*bank[a-z0-9-]*\.[a-z.]+)$",
    re.IGNORECASE,
)
# Unknown user MCP tools (setting_sources=["user"]) whose name suggests a write/send.
UNKNOWN_MCP_VERBS = re.compile(
    r"send|post|create|delete|update|write|publish|reply|pay|transfer|upload|remove", re.IGNORECASE)

SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "fish", "tcsh", "csh"}
INTERPRETERS = {
    "python", "python3", "python2", "pypy", "pypy3", "perl", "ruby", "node", "deno", "bun",
    "php", "lua", "tclsh", "swift",
}
SUDO_LIKE = {"sudo", "su", "doas", "pkexec", "run0"}
# Commands that only inspect metadata (names, sizes, types), never file contents.
METADATA_ONLY = {
    "ls", "stat", "test", "[", "[[", "file", "du", "echo", "printf", "cd", "pushd", "popd",
    "mkdir", "which", "type", "true", "false", "eza", "exa", "tree", "basename", "dirname",
    "realpath", "readlink", "wc",
}
READ_ONLY_TEXT = METADATA_ONLY | {"cat", "head", "tail", "less", "more", "grep", "rg", "bat", "diff"}
BULK_COPY = {"tar", "gtar", "bsdtar", "zip", "7z", "7za", "ditto", "cp", "rsync", "scp", "rclone",
             "pax", "cpio", "grep", "egrep", "fgrep"}
REMOTE_SHELLS = {"ssh", "sftp", "mosh", "ftp", "lftp", "telnet", "socat"}
NETCATS = {"nc", "ncat", "netcat"}
MAILERS = {"mail", "mailx", "sendmail", "mutt", "msmtp", "neomutt"}

# (command, subcommand words) prefixes that publish/share something.
OUTWARD_SUBCOMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("git", ("push",)), ("git", ("send-email",)),
    ("npm", ("publish",)), ("npm", ("unpublish",)), ("pnpm", ("publish",)),
    ("yarn", ("publish",)), ("yarn", ("npm", "publish")), ("bun", ("publish",)),
    ("cargo", ("publish",)), ("gem", ("push",)), ("twine", ("upload",)),
    ("poetry", ("publish",)), ("uv", ("publish",)), ("docker", ("push",)),
    ("podman", ("push",)), ("hatch", ("publish",)), ("flit", ("publish",)),
    ("gh", ("pr", "create")), ("gh", ("pr", "merge")), ("gh", ("pr", "comment")),
    ("gh", ("pr", "review")), ("gh", ("pr", "close")), ("gh", ("issue", "create")),
    ("gh", ("issue", "comment")), ("gh", ("issue", "close")), ("gh", ("release", "create")),
    ("gh", ("repo", "create")), ("gh", ("repo", "delete")), ("gh", ("gist", "create")),
    ("gh", ("secret", "set")), ("shortcuts", ("run",)),
    ("aws", ("s3", "cp")), ("aws", ("s3", "mv")), ("aws", ("s3", "sync")),
    ("gsutil", ("cp",)), ("gcloud", ("storage", "cp")),
)

# AppleScript / JXA patterns (applied to the script text, lowercased).
_AS_ADMIN = re.compile(r"with\s+administrator\s+privileges")
_AS_SEND = re.compile(r"(?<![\w.])send\b|\.send\s*\(")
_AS_SEND_APPS = re.compile(r"\b(mail|messages|outlook|spark|airmail|whatsapp|telegram|slack)\b")
_AS_RETURN = re.compile(
    r"keystroke\s+return|key\s+code\s+(36|76)\b|keystroke\s+\"\\r\"|ascii\s+character\s+13|"
    r"keystroke\s+linefeed"
)
_AS_CLICK_VERB = re.compile(r"click\b[^\n]*\"([^\"]*)\"")
_AS_DELETE = re.compile(r"\bdelete\b[^\n]*\b(message|messages|note|notes|event|events|reminder|"
                        r"reminders|mailbox|account|contact|file|folder|item|items)\b|empty\s+trash")
_AS_DO_SHELL = re.compile(r"do\s+shell\s+script\s+\"((?:[^\"\\]|\\.)*)\"")

_FORK_BOMB = re.compile(r"(\S+)\s*\(\s*\)\s*\{[^}]*\1\s*\|\s*\1\s*&[^}]*\}")
_FORK_BOMB_LITERAL = re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}")

# A string literal inside a script is re-read as a shell command line only when it starts like
# one (a command name or a path, possibly after a substitution): `cat ~/.ssh/id_rsa`, `/bin/sh -c`.
# Regexes (`(?is)<(script|style)[^>]*>.*?</\\1>`, `[\\s\\S]*?`) and globs are not command lines;
# parsed as one, `<` became an input redirect from the "file" `.*?`, which as a glob matches
# `.env` (live 2026-11-11: job-hunt HTML stripping denied as a credential read).
_COMMAND_LIKE = re.compile(r"^\s*(\$\(|`|[\w./~-]+(\s|$))")
_INTERP_DELETE = re.compile(r"rmtree|os\.remove|os\.unlink|unlink(sync)?\s*\(|rmdir|rm_rf|rimraf|"
                            r"fs\.rm|shutil\.move|file\.delete|remove_dir_all")
_INTERP_UPLOAD = re.compile(
    r"requests\.(post|put|patch|delete)|httpx\.(post|put|patch|delete)|smtplib|"
    r"urlopen\([^)]*data\s*=|method\s*[:=]\s*['\"](post|put|patch|delete)|sendmail|"
    r"net::http::post|\.send_message\(|socket\.(create_connection|socket)",
    re.IGNORECASE,
)
_JS_OUTWARD = re.compile(
    r"\.submit\s*\(|requestsubmit\s*\(|sendbeacon|method\s*:\s*['\"](post|put|patch|delete)|"
    r"xmlhttprequest",
    re.IGNORECASE,
)
_JS_CLICK = re.compile(r"\.click\s*\(")
_STRING_LITERAL = re.compile(r"""(['"])((?:\\.|(?!\1).)*)\1""", re.DOTALL)

# WhatsApp's own screens (chat profile hard block, M1).
WHATSAPP_HOSTS = re.compile(r"(^|\.)(whatsapp\.com|whatsapp\.net|wa\.me)$", re.IGNORECASE)
_WA_URL = re.compile(r"whatsapp://|(?<![a-z0-9.-])(?:https?://)?(?:[a-z0-9-]+\.)*(?:whatsapp\.com|wa\.me)(?:[/:?#]|$)",
                     re.IGNORECASE)
_WA_SCRIPT = re.compile(r"whatsapp://|tell\s+application\s+\\?[\"']?whatsapp|application\s+id\s+\\?[\"']?"
                        r"net\.whatsapp|net\.whatsapp\.whatsapp|application\s*\(\s*[\"']whatsapp",
                        re.IGNORECASE)
WHATSAPP_UI = "whatsapp_ui"
NEVER_CHAT = "never_chat"
REASON_WHATSAPP_UI = ("WhatsApp's own screens are off limits on this channel, reading included; "
                      "your replies reach the user through Mac's own link")
MAC_SEND_FILE = "mcp__mac__send_file"
MAC_TEST_OUTWARD = "mcp__mac__test_outward"
MAC_CONFIRM = "mcp__mac__request_confirmation"
MAC_SET_PERSONA = "mcp__mac__set_persona"
CAPABILITY = "capability"  # reason code prefix: capability:<name> (W9.5)
# A scheduled run (tasks DESIGN 9.3): nobody is there to say yes, so an outward step is refused at
# once and the model lists it in its report instead.
UNATTENDED_REASON = ("a scheduled task can't send, post, pay, delete or change things outside the Mac "
                     "on its own. Skip this step and list it in your report under 'Waiting for your OK:'")

# Mapped tools whose free-text input still gets the X1 never-list and secret-path scan (a skill's
# arguments reach other tools; tasks D8 keeps Skill in the "" row, not the unclassified rule).
_UNPARSED_TOO = frozenset({"Skill"})

UI_RETURN_COMBOS = {"return", "enter", "cmd+return", "cmd+enter", "ctrl+return", "ctrl+enter",
                    "shift+return", "shift+enter", "cmd+shift+d", "kp_enter"}


# ----------------------------------------------------------------------------- rules/context


@dataclass
class GateContext:
    """Environment for classification (all optional; defaults are the real machine)."""

    home: str = field(default_factory=lambda: str(Path.home()))
    workspace: str = field(default_factory=lambda: str(config.WORKSPACE_DIR))
    user: str = field(default_factory=lambda: os.environ.get("USER") or Path.home().name)
    element_label: Callable[[str], str | None] | None = None  # mac ui_tree registry
    chrome_ref_label: Callable[[str], str | None] | None = None  # Chrome find/read_page refs
    chrome_tab_host: Callable[[Any], str | None] | None = None  # tabId -> host (navigate/tabs)
    current_app: Callable[[], str] | None = None  # frontmost app, queried at decision time
    read_file: Callable[[str], str | None] = lambda p: _read_small_file(p)  # script files
    # Profile switches (WhatsApp ARCHITECTURE W3.1/W3.3); the defaults are the voice profile.
    whatsapp_ui_block: bool = False  # chat: WhatsApp's own screens are catastrophic (M1)
    never_chat_names: tuple[str, ...] = ()  # chat: outward actions naming these are catastrophic
    env_exempt_dir: str | None = None  # .env readable here (None: the workspace); F10 limits it
    memory_path: str | None = None  # an injected shared memory.md is a memory file too
    # mac.send_file is outward (a bound yes): profile.confirm_files (P6); a callable is read per call
    confirm_send_file: bool | Callable[[], bool] = False
    # Chat profile capabilities (W9.5): granted classes, or a callable read at each decision.
    # None = voice: no capability check at all.
    capabilities: frozenset[str] | Callable[[], frozenset[str]] | None = None
    display_name: str = ""  # the chat's name, for capability denials and the set_persona target
    # Scheduled runs (tasks 9.3): True while nobody can answer a confirmation; read per decision.
    unattended: Callable[[], bool] | None = None


def _read_small_file(path: str, limit: int = 256 * 1024) -> str | None:
    try:
        with open(path, "rb") as f:
            return f.read(limit).decode("utf-8", "replace")
    except OSError:
        return None


# ----------------------------------------------------------------------------- path helpers


def _segments(path: str) -> list[str]:
    return [s for s in path.lower().split("/") if s]


def _seg_match(target: list[str], pattern: list[str], target_is_glob: bool) -> bool:
    if len(target) != len(pattern):
        return False
    for t, p in zip(target, pattern, strict=True):
        if p == "*" or t == p:
            continue
        if target_is_glob and fnmatch.fnmatchcase(p, t):
            continue
        return False
    return True


class _Paths:
    """Normalisation and table lookups bound to one GateContext."""

    def __init__(self, ctx: GateContext) -> None:
        self.ctx = ctx
        self.home = posixpath.normpath(ctx.home)
        self.workspace = posixpath.normpath(ctx.workspace).lower()
        self.sensitive = [self._abs(self.expand_tilde(p), "/").lower() for p in SENSITIVE_ROOTS]
        mac_homes = {posixpath.join(self.home, ".mac-agent").lower(), posixpath.normpath(str(config.HOME)).lower()}
        self.mac_private = [posixpath.join(h, sub) for h in sorted(mac_homes) for sub in MAC_PRIVATE]
        self.mac_readable = [posixpath.join(h, sub) for h in sorted(mac_homes) for sub in MAC_READABLE]
        self.sensitive_names = tuple(n.lower() for n in SENSITIVE_BASENAMES)
        self.protected = [_segments(self.expand_tilde(p)) for p in [*PROTECTED_ROOTS, *_repo_roots()]]
        self.mac_homes = sorted(mac_homes)
        self.mac_writable = [posixpath.join(h, sub) for h in self.mac_homes for sub in MAC_WRITABLE]
        self.write_denied = [*self.mac_homes, *(r.lower() for r in _own_code_roots()),
                             *(self._abs(self.expand_tilde(p), "/").lower() for p in WRITE_DENIED)]
        self.owner_config = [self._abs(self.expand_tilde(p), "/").lower() for p in OWNER_CONFIG]
        self.memory_files = {
            posixpath.join(self.workspace, "claude.md"),
            posixpath.join(self.workspace, "memory.md"),
            str(config.MEMORY_PATH).lower(),
            str(config.WORKSPACE_DIR / "CLAUDE.md").lower(),
        }
        if ctx.memory_path:
            self.memory_files.add(posixpath.normpath(self.expand_tilde(ctx.memory_path)).lower())
        self.env_exempt = self._env_exempt_dir(ctx)

    def _env_exempt_dir(self, ctx: GateContext) -> str | None:
        """F10: the .env exemption holds only inside one of Mac's own workspaces (under
        ~/.mac-agent), never in a real project folder used as the working directory."""
        raw = ctx.env_exempt_dir if ctx.env_exempt_dir is not None else ctx.workspace
        if not raw:
            return None
        d = posixpath.normpath(self.expand_tilde(raw)).lower()
        own = [posixpath.join(self.home.lower(), ".mac-agent"),
               posixpath.normpath(str(config.HOME)).lower()]
        return d if any(self._under(d, root) for root in own) else None

    def expand_tilde(self, p: str) -> str:
        if p == "~" or p.startswith("~/"):
            return self.home + p[1:]
        m = re.match(r"~([A-Za-z0-9_.-]+)(/.*)?$", p)
        if m:
            user = m.group(1)
            base = "/var/root" if user == "root" else f"/Users/{user}"
            return base + (m.group(2) or "")
        return p

    @staticmethod
    def _abs(p: str, cwd: str) -> str:
        if not p.startswith("/"):
            p = posixpath.join(cwd, p)
        p = posixpath.normpath(p)
        return "/" + p.lstrip("/")

    def norm(self, p: str, cwd: str) -> str:
        return self._abs(self.expand_tilde(p), cwd)

    # -- tables ------------------------------------------------------------------------
    def is_protected(self, path: str, glob: bool) -> bool:
        segs = _segments(path)
        return any(_seg_match(segs, pat, glob) for pat in self.protected)

    def _under(self, p: str, root: str) -> bool:
        return p == root or p.startswith(root.rstrip("/") + "/")

    def sensitive_reason(self, raw: str, cwd: str, glob: bool = False, write: bool = False) -> str | None:
        """Return a reason string if `raw` (a path-ish word) names sensitive data. `write`: the
        path is written (Mac's inbound media folder is readable, never writable)."""
        if not raw or raw.startswith("-") and "=" not in raw:
            return None
        path = self.norm(raw, cwd)
        lowered = path.lower()
        candidates = {lowered}
        if not glob:
            try:
                candidates.add(os.path.realpath(path).lower())
            except (OSError, ValueError):
                pass
        for cand in candidates:
            for root in self.sensitive:
                if self._under(cand, root):
                    return f"{raw} is inside a protected secrets location"
                if glob and _seg_match(_segments(cand)[: len(_segments(root))], _segments(root), True):
                    return f"{raw} may match a protected secrets location"
            readable = not write and not glob and any(self._under(cand, r) for r in self.mac_readable)
            for root in () if readable else self.mac_private:
                if self._under(cand, root) or (
                        glob and _seg_match(_segments(cand)[: len(_segments(root))], _segments(root), True)):
                    return f"{raw} is Mac's own settings, tokens or keys"
        base = posixpath.basename(lowered)
        if base in ENV_TEMPLATE_NAMES:
            return None
        for pat in self.sensitive_names:
            hit = (fnmatch.fnmatchcase(base, pat) or
                   (glob and any(fnmatch.fnmatchcase(s, base) and fnmatch.fnmatchcase(s, pat)
                                 for s in _SENSITIVE_SAMPLES) and base != "*"))
            if not hit:
                continue
            if pat == ".env*" and self.env_exempt and self._under(lowered, self.env_exempt):
                continue
            return f"{raw} looks like a credential or key file"
        return None

    def is_ancestor_of_sensitive(self, raw: str, cwd: str) -> bool:
        p = self.norm(raw, cwd).lower().rstrip("/") + "/"
        return any(root.startswith(p) for root in [*self.sensitive, *self.mac_private])

    def write_decision(self, raw: str, cwd: str, glob: bool = False) -> GateDecision | None:
        """H1: a write to Mac's own code, state or launch config, or to the owner's shell and
        Claude Code setup. Mac's: denied everywhere. The owner's: denied in chat, a spoken yes by
        voice (it is the owner speaking)."""
        if not raw or raw.startswith("-") and "=" not in raw:
            return None
        path = self.norm(raw, cwd).lower()
        cands = {path}
        if not glob:
            try:
                cands.add(os.path.realpath(self.norm(raw, cwd)).lower())
            except (OSError, ValueError):
                pass
        for cand in cands:
            if any(fnmatch.fnmatchcase(cand, pat) or self._under(cand, pat) for pat in self.owner_config):
                if self.ctx.capabilities is not None:
                    return GateDecision("sensitive_read", OWNER_CONFIG_WRITE,
                                        f"{raw} is your shell or Claude Code setup; Mac never changes it from a chat")
                return GateDecision("outward", OWNER_CONFIG_WRITE, f"changes your shell or Claude Code setup ({raw})",
                                    target=posixpath.basename(cand))
            if any(self._under(cand, r) for r in self.write_denied) and not any(
                    self._under(cand, w) for w in self.mac_writable):
                return GateDecision("sensitive_read", PROTECTED_WRITE,
                                    f"{raw} is Mac's own code, settings or startup files; no tool may change it")
        return None

    def is_memory_file(self, raw: str, cwd: str) -> bool:
        return self.norm(raw, cwd).lower() in self.memory_files


# ----------------------------------------------------------------------------- bash analysis


@dataclass
class _Cmd:
    name: str
    args: list[shell.Word]
    cmd: shell.SimpleCommand
    via_xargs: bool = False


class _BashAnalyzer:
    MAX_DEPTH = 6

    def __init__(self, paths: _Paths) -> None:
        self.p = paths
        self.env: dict[str, list[str]] = {
            "HOME": [paths.home], "USER": [paths.ctx.user], "LOGNAME": [paths.ctx.user],
            "TMPDIR": ["/tmp"],
        }
        self.cwd = paths.ctx.workspace
        self.commands: set[str] = set()  # every command name seen, nested ones included (tool_classes)

    # -- entry -------------------------------------------------------------------------
    def analyze(self, src: str, depth: int = 0) -> GateDecision:
        if depth > self.MAX_DEPTH:
            return GateDecision("catastrophic", "too_nested", "command nesting is too deep to verify")
        found: list[GateDecision] = []
        if _FORK_BOMB_LITERAL.search(src) or _FORK_BOMB.search(src):
            found.append(GateDecision("catastrophic", "fork_bomb", "fork bomb"))
        script = shell.parse(src)
        for sub in script.substitutions:
            found.append(self.analyze(sub, depth + 1))
        for pipeline in script.pipelines:
            found.append(self._pipeline(pipeline, depth))
        return _worst(found)

    # -- expansion ---------------------------------------------------------------------
    _VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[^}]*\}|\$([A-Za-z_][A-Za-z0-9_]*|[0-9@*#?$!-])")

    def expand(self, word: shell.Word) -> list[str]:
        text = word.text
        values: list[str] = [""]
        pos = 0
        for m in self._VAR.finditer(text):
            name = m.group(1) or m.group(2)
            alts = self.env.get(name, [""])
            if name == "PWD":
                alts = [self.cwd]
            lit = text[pos : m.start()]
            values = [v + lit + a for v in values for a in alts][:64]
            pos = m.end()
        values = [v + text[pos:] for v in values]
        out: list[str] = []
        for v in values:
            for e in shell.brace_expand(v):
                out.append(self.p.expand_tilde(e))
        return out

    def words(self, ws: Iterable[shell.Word]) -> list[tuple[str, bool]]:
        out: list[tuple[str, bool]] = []
        for w in ws:
            out.extend((e, w.glob) for e in self.expand(w))
        return out

    # -- wrappers ----------------------------------------------------------------------
    def _unwrap(self, cmd: shell.SimpleCommand) -> _Cmd | None:
        ws = list(cmd.words)
        if len(ws) >= 3 and ws[0].text in ("for", "select") and ws[2].text == "in":
            self.env[ws[1].text] = [e for e, _ in self.words(ws[3:])][:64] or [""]
            return None
        via_xargs = False
        assignments: list[tuple[str, list[str]]] = []
        while ws:
            head = ws[0].text
            m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", head, re.DOTALL)
            if m:
                assignments.append((m.group(1), self.expand(shell.Word(m.group(2), ws[0].glob))))
                ws.pop(0)
                continue
            name = posixpath.basename(head.lstrip("\\")) if head else head
            if name in ("{", "}", "!", "if", "then", "else", "elif", "do", "while", "until",
                        "fi", "done", "esac", "time", "nohup", "builtin", "noglob", "-",
                        "unbuffer", "stdbuf", "caffeinate", "chronic"):
                ws.pop(0)
                while name in ("time", "stdbuf", "caffeinate") and ws and ws[0].text.startswith("-"):
                    ws.pop(0)
                continue
            if name == "command":
                ws.pop(0)
                if ws and ws[0].text in ("-v", "-V"):
                    return None
                while ws and ws[0].text.startswith("-"):
                    ws.pop(0)
                continue
            if name == "exec":
                ws.pop(0)
                while ws and ws[0].text.startswith("-"):
                    flag = ws.pop(0).text
                    if flag == "-a" and ws:
                        ws.pop(0)
                continue
            if name == "env":
                ws.pop(0)
                while ws and (ws[0].text.startswith("-") or "=" in ws[0].text):
                    flag = ws.pop(0).text
                    if flag in ("-u", "-P", "-C") and ws:
                        ws.pop(0)
                    elif flag in ("-S", "--split-string") and ws:
                        split = shell.parse(ws.pop(0).text)
                        extra = split.pipelines[0][0].words if split.pipelines else []
                        ws = extra + ws
                    elif "=" in flag and not flag.startswith("-"):
                        k, _, v = flag.partition("=")
                        assignments.append((k, [v]))
                continue
            if name in ("nice", "renice", "ionice", "timeout", "gtimeout", "arch", "watch"):
                ws.pop(0)
                while ws and (ws[0].text.startswith("-") or re.fullmatch(r"[\d.]+[smhd]?", ws[0].text)):
                    flag = ws.pop(0).text
                    if flag in ("-n", "-s", "-k", "--signal", "--kill-after", "-c") and ws:
                        ws.pop(0)
                continue
            if name == "xargs":
                ws.pop(0)
                via_xargs = True
                while ws and ws[0].text.startswith("-"):
                    flag = ws.pop(0).text
                    if flag in ("-I", "-J", "-n", "-P", "-L", "-s", "-E", "-d", "-R", "-S") and ws:
                        ws.pop(0)
                continue
            break
        if not ws:
            for k, v in assignments:  # bare assignment: persists for later commands
                self.env[k] = v
            return None
        name = posixpath.basename(ws[0].text.lstrip("\\")).lower()
        return _Cmd(name=name, args=ws[1:], cmd=cmd, via_xargs=via_xargs)

    # -- pipelines ---------------------------------------------------------------------
    def _pipeline(self, pipeline: list[shell.SimpleCommand], depth: int) -> GateDecision:
        found: list[GateDecision] = []
        all_words = [w for c in pipeline for w in c.words]
        prev_text: str | None = None
        prev_name = ""
        for idx, sc in enumerate(pipeline):
            c = self._unwrap(sc)
            if c is None and (sc.redirects_in or sc.redirects_out):
                c = _Cmd(name=":", args=[], cmd=sc)  # `done < file`, `} < file`, `exec 3<file`
            if c is None:
                prev_text, prev_name = None, ""
                continue
            stdin = sc.stdin_text if sc.stdin_text is not None else prev_text
            feeds = depth > 0 or idx < len(pipeline) - 1  # output becomes someone else's input
            found.append(self._command(c, depth, stdin, all_words, feeds))
            if prev_name in ("curl", "wget") and (c.name in SHELLS or c.name in INTERPRETERS):
                found.append(GateDecision("outward", "remote_exec",
                                          "runs code downloaded from the internet"))
            prev_text = None
            if c.name in ("echo", "printf"):
                prev_text = " ".join(e for e, _ in self.words(c.args))
            elif c.name == "cat" and not [w for w in c.args if not w.text.startswith("-")]:
                prev_text = stdin
            prev_name = c.name
        return _worst(found)

    def _command(self, c: _Cmd, depth: int, stdin: str | None,
                 pipeline_words: list[shell.Word], feeds: bool = False) -> GateDecision:
        found: list[GateDecision] = []
        self.commands.add(c.name)
        if c.name in SUDO_LIKE:
            return GateDecision("catastrophic", "sudo", f"{c.name} (privilege escalation) is never allowed")
        found.append(self._catastrophic(c, pipeline_words))
        found.append(self._sensitive(c, pipeline_words, feeds))
        found.append(self._memory(c))
        found.append(self._protected_write(c))
        found.append(self._outward(c))
        found.append(self._nested(c, depth, stdin))
        self._track_cwd(c)
        return _worst(found)

    # -- nested interpreters -----------------------------------------------------------
    def _nested(self, c: _Cmd, depth: int, stdin: str | None) -> GateDecision:
        args = [w.text for w in c.args]
        positional = [a for a in args if not a.startswith("-")]
        if c.name in SHELLS:
            payload = _flag_payload(args, ("c",))
            if payload is None:
                payload = self._script_file(positional[0]) if positional else stdin
            return self.analyze(payload, depth + 1) if payload else ALLOW
        if c.name == "eval":
            return self.analyze(" ".join(args), depth + 1)
        if c.name in ("source", "."):
            payload = self._script_file(positional[0]) if positional else None
            return self.analyze(payload, depth + 1) if payload else ALLOW
        if c.name == "osascript":
            return self._osascript(args, stdin, depth)
        if c.name in INTERPRETERS or re.fullmatch(r"python3(\.\d+)?", c.name):
            payload = _flag_payload(args, ("c", "e", "E", "r"))
            if payload is None:
                payload = self._script_file(positional[0]) if positional else stdin
            return self._interpreter(payload, depth) if payload else ALLOW
        if c.name == "find":
            return self._find_exec(c, depth)
        return ALLOW

    def _script_file(self, path: str) -> str | None:
        """Contents of a script passed by path (an injected model can write-then-run)."""
        for cand in self.expand(shell.Word(path)):
            text = self.p.ctx.read_file(self.p.norm(cand, self.cwd))
            if text is not None:
                return text
        return None

    def _interpreter(self, payload: str, depth: int) -> GateDecision:
        found: list[GateDecision] = []
        literals = [m.group(2) for m in _STRING_LITERAL.finditer(payload)]
        for lit in literals:
            if _COMMAND_LIKE.match(lit):
                found.append(self.analyze(lit, depth + 1))
            reason = self.p.sensitive_reason(lit, self.cwd)
            if reason and ("/" in lit or lit.startswith("~")):
                found.append(GateDecision("sensitive_read", "sensitive_path", reason))
            if "/" in lit or lit.startswith("~"):  # a script naming a protected path may write it
                found.append(self.p.write_decision(lit, self.cwd) or ALLOW)
            if _INTERP_DELETE.search(payload) and self.p.is_protected(self.p.norm(lit, self.cwd), False) \
                    and ("/" in lit or lit.startswith("~")):
                found.append(GateDecision("catastrophic", "rm_root", f"deletes {lit}"))
        if _INTERP_UPLOAD.search(payload):
            found.append(GateDecision("outward", "network_upload", "script sends data over the network"))
        if "with administrator privileges" in payload.lower():
            found.append(GateDecision("catastrophic", "admin", "requests administrator privileges"))
        return _worst(found)

    def _osascript(self, args: list[str], stdin: str | None, depth: int) -> GateDecision:
        parts = [args[i + 1] for i in range(len(args) - 1) if args[i] == "-e"]
        if stdin:
            parts.append(stdin)
        if not parts:  # `osascript file`: read it (a compiled .scpt is opaque → outward)
            files = [a for i, a in enumerate(args) if not a.startswith("-") and (i == 0 or args[i - 1] != "-l")]
            if files:
                text = self._script_file(files[0])
                if text is None:
                    return ALLOW  # not there yet; the run will fail
                if "\x00" in text:
                    return GateDecision("outward", "osascript_binary", "runs a compiled AppleScript",
                                        target=posixpath.basename(files[0]))
                parts.append(text)
        script = "\n".join(parts)
        if not script.strip():
            return ALLOW
        return _classify_applescript(script, lambda s: self.analyze(s, depth + 1))

    def _find_exec(self, c: _Cmd, depth: int) -> GateDecision:
        args = [w.text for w in c.args]
        if "-exec" not in args and "-execdir" not in args and "-ok" not in args:
            return ALLOW
        roots = [a for a in _find_roots(args)] or ["."]
        found: list[GateDecision] = []
        for i, a in enumerate(args):
            if a in ("-exec", "-execdir", "-ok", "-okdir"):
                end = next((j for j in range(i + 1, len(args)) if args[j] in (";", "+")), len(args))
                inner = args[i + 1 : end]
                for root in roots:
                    cmdline = " ".join(_quote(root if x == "{}" else x) for x in inner)
                    found.append(self.analyze(cmdline, depth + 1))
        return _worst(found)

    # -- catastrophic ------------------------------------------------------------------
    def _catastrophic(self, c: _Cmd, pipeline_words: list[shell.Word]) -> GateDecision:
        name = c.name
        raw = [w.text for w in c.args]
        if name in ("rm", "srm", "shred", "trash", "gio"):
            return self._rm(c, pipeline_words)
        if name in ("mkfs", "newfs", "fdisk", "gpt") or name.startswith(("mkfs.", "newfs_")):
            return GateDecision("catastrophic", "disk_format", f"{name} formats disks")
        if name == "diskutil" and raw:
            sub = raw[0].lower()
            if sub.startswith(("erase", "zero", "secureerase", "partition", "reformat")) or (
                sub in ("apfs", "cs", "appleraid", "ar") and any(
                    a.lower().startswith(("delete", "erase")) for a in raw[1:])):
                return GateDecision("catastrophic", "disk_erase", f"diskutil {sub} erases disks")
        if name == "dd":
            for a in raw:
                if a.startswith("of=") and a[3:].startswith("/dev/"):
                    return GateDecision("catastrophic", "dd_device", "dd writes to a raw device")
        if name in ("chmod", "chown", "chgrp", "chflags", "xattr") and any(
                _has_short(a, "R") or a == "--recursive" for a in raw if a.startswith("-")):
            for t, g in self.words(a for a in c.args if not a.text.startswith("-")):
                if self.p.is_protected(self.p.norm(t, self.cwd), g):
                    return GateDecision("catastrophic", "chmod_root", f"recursive {name} on {t}")
        if name == "mv":
            targets = [(t, g) for t, g in self.words(c.args) if not t.startswith("-")]
            for t, g in targets[:-1]:
                if self.p.is_protected(self.p.norm(t, self.cwd), g):
                    return GateDecision("catastrophic", "mv_root", f"moves {t}")
        if name == "rsync" and any(a.startswith("--delete") for a in raw):
            targets = [t for t, _ in self.words(c.args) if not t.startswith("-")]
            if targets and self.p.is_protected(self.p.norm(targets[-1], self.cwd), False):
                return GateDecision("catastrophic", "rsync_delete", f"rsync --delete into {targets[-1]}")
        if name == "find" and ("-delete" in raw or _find_exec_rm(raw)):
            for root in _find_roots(raw):
                for t in self.expand(shell.Word(root)):
                    if self.p.is_protected(self.p.norm(t, self.cwd), False):
                        return GateDecision("catastrophic", "find_delete", f"find {t} deletes files")
        if name == "launchctl" and raw and raw[0].lower() in ("bootout", "unload", "remove", "disable", "kill") \
                and any("com.mac-agent" in a.lower() for a in raw[1:]):
            return GateDecision("catastrophic", "mac_service", f"launchctl {raw[0]} stops Mac's own services")
        if name in ("csrutil", "spctl") and any(a in ("disable", "--master-disable") for a in raw):
            return GateDecision("catastrophic", "security_off", f"{name} disables system security")
        for w in c.cmd.redirects_out:
            for t in self.expand(w):
                if re.match(r"/dev/r?disk\d", t):
                    return GateDecision("catastrophic", "dd_device", "writes to a raw disk device")
        return ALLOW

    def _rm(self, c: _Cmd, pipeline_words: list[shell.Word]) -> GateDecision:
        recursive = c.name in ("srm", "trash", "gio")
        targets: list[tuple[str, bool]] = []
        end_opts = False
        for w in c.args:
            t = w.text
            if not end_opts and t == "--":
                end_opts = True
                continue
            if not end_opts and t.startswith("--"):
                if t == "--no-preserve-root":
                    return GateDecision("catastrophic", "rm_root", "rm --no-preserve-root")
                if t in ("--recursive", "--dir"):
                    recursive = True
                continue
            if not end_opts and t.startswith("-") and len(t) > 1:
                recursive = recursive or _has_short(t, "r") or _has_short(t, "R")
                continue
            targets.extend((e, w.glob) for e in self.expand(w))
        if c.via_xargs and recursive:
            for w in pipeline_words:
                for e in self.expand(w):
                    if e.startswith(("/", "~")) and self.p.is_protected(self.p.norm(e, self.cwd), w.glob):
                        return GateDecision("catastrophic", "rm_root", f"xargs rm on {e}")
        for t, glob in targets:
            if not t:
                continue
            path = self.p.norm(t, self.cwd)
            if (recursive or glob) and self.p.is_protected(path, glob):
                return GateDecision("catastrophic", "rm_root",
                                    f"recursive delete of {t} ({path}) would destroy the system or home")
        return ALLOW

    # -- sensitive ---------------------------------------------------------------------
    def _sensitive(self, c: _Cmd, pipeline_words: list[shell.Word] | None = None,
                   feeds: bool = False) -> GateDecision:
        raw = [w.text for w in c.args]
        if c.name == "security" and raw:
            sub = next((a for a in raw if not a.startswith("-")), "")
            if sub in ("find-generic-password", "find-internet-password", "dump-keychain",
                       "export", "find-key", "export-identity", "dump-trust-settings") or (
                    sub == "find-certificate" and "-p" in raw):
                return GateDecision("sensitive_read", "keychain_cli", f"security {sub} reads the keychain")
        # Redirects first: `echo key >> ~/.ssh/authorized_keys` is a write by a "harmless" command.
        for t, glob in self.words(c.cmd.redirects_out):
            reason = self.p.sensitive_reason(t, self.cwd, glob, write=True)
            if reason:
                return GateDecision("sensitive_read", "sensitive_redirect", reason)
        for t, glob in self.words(c.cmd.redirects_in):
            reason = self.p.sensitive_reason(t, self.cwd, glob)
            if reason:
                return GateDecision("sensitive_read", "sensitive_redirect", reason)
        # `echo ~/.ssh/id_rsa | cat` / `cat $(echo ~/.ssh/id_rsa | cat)`: the echoed path is
        # what the next command reads, so scan it; a plain `echo` is just text.
        if c.name in METADATA_ONLY and not (c.name in ("echo", "printf") and feeds):
            return ALLOW
        words = list(c.args)
        if c.via_xargs and pipeline_words:  # `ls ~/.ssh | xargs cat`: inputs come from the pipe
            words += pipeline_words
        for t, glob in self.words(words):
            for cand in _path_candidates(t):
                reason = self.p.sensitive_reason(cand, self.cwd, glob)
                if reason:
                    return GateDecision("sensitive_read", "sensitive_path", reason)
        if c.name in BULK_COPY and _is_recursive_copy(c.name, raw):
            for t, _ in self.words(c.args):
                if not t.startswith("-") and (t.startswith(("/", "~", ".")) or "/" in t) \
                        and self.p.is_ancestor_of_sensitive(t, self.cwd):
                    return GateDecision("sensitive_read", "bulk_copy_sensitive",
                                        f"{c.name} of {t} would include protected secrets")
        return ALLOW

    def _protected_write(self, c: _Cmd) -> GateDecision:
        """H1: redirects into, and any non-read-only command naming, a write-protected path."""
        found: list[GateDecision] = []
        for t, glob in self.words(c.cmd.redirects_out):
            found.append(self.p.write_decision(t, self.cwd, glob) or ALLOW)
        if c.name not in READ_ONLY_TEXT and c.name not in ("source", "."):  # sourcing reads
            for t, glob in self.words(c.args):
                for cand in _path_candidates(t):
                    found.append(self.p.write_decision(cand, self.cwd, glob) or ALLOW)
        return _worst(found)

    def _memory(self, c: _Cmd) -> GateDecision:
        for t, _ in self.words(c.cmd.redirects_out):
            if self.p.is_memory_file(t, self.cwd):
                return GateDecision("sensitive_read", MEMORY_WRITE, "memory files are append-only via remember")
        if c.name in READ_ONLY_TEXT:
            return ALLOW
        for t, _ in self.words(c.args):
            for cand in _path_candidates(t):
                if self.p.is_memory_file(cand, self.cwd):
                    return GateDecision("sensitive_read", MEMORY_WRITE, "memory files are append-only via remember")
        return ALLOW

    # -- outward -----------------------------------------------------------------------
    def _outward(self, c: _Cmd) -> GateDecision:
        name = c.name
        raw = [w.text for w in c.args]
        words = [r for r in raw if not r.startswith("-")]
        host = _url_host(words)
        for cmd, sub in OUTWARD_SUBCOMMANDS:
            if name == cmd and _has_subcommand(raw, sub):
                return GateDecision("outward", f"{cmd}_{'_'.join(sub)}", f"{cmd} {' '.join(sub)}",
                                    target=" ".join([name, *words[: len(sub) + 1]]))
        if name == "curl" and _curl_uploads(raw):
            return GateDecision("outward", "http_upload", f"curl sends data to {host or 'a server'}", target=host)
        if name == "wget" and any(a.startswith(("--post-data", "--post-file", "--body-data", "--body-file"))
                                  or re.match(r"--method=?(post|put|patch|delete)", a, re.IGNORECASE) for a in raw):
            return GateDecision("outward", "http_upload", f"wget sends data to {host or 'a server'}", target=host)
        if name in ("http", "https", "xh", "xhs", "httpie") and _httpie_uploads(raw):
            return GateDecision("outward", "http_upload", f"httpie sends data to {host or 'a server'}",
                                target=host)
        if name == "gh" and raw[:1] == ["api"] and (
                any(a in ("-f", "-F", "--field", "--raw-field", "--input") for a in raw) or
                any(re.fullmatch(r"(-X|--method)=?(post|put|patch|delete)?", a, re.IGNORECASE) and
                    (a.upper().endswith(("POST", "PUT", "PATCH", "DELETE")) or
                     (i + 1 < len(raw) and raw[i + 1].upper() in ("POST", "PUT", "PATCH", "DELETE")))
                    for i, a in enumerate(raw))):
            return GateDecision("outward", "gh_api_write", "gh api write")
        if name in NETCATS and not any(_has_short(a, "l") or _has_short(a, "z") for a in raw if a.startswith("-")):
            return GateDecision("outward", "netcat", f"{name} opens a raw network connection",
                                target=words[0] if words else "")
        if name in REMOTE_SHELLS:
            return GateDecision("outward", "remote_shell", f"{name} connects to a remote host",
                                target=words[0] if words else "")
        if name in ("scp", "rsync") and any(_is_remote_spec(a) for a in words):
            remote = next(a for a in words if _is_remote_spec(a))
            return GateDecision("outward", "remote_copy", f"{name} copies to {remote}", target=remote)
        if name in MAILERS:
            return GateDecision("outward", "mail", f"{name} sends email", target=words[-1] if words else "")
        return ALLOW

    def _track_cwd(self, c: _Cmd) -> None:
        if c.name not in ("cd", "pushd"):
            return
        args = [t for t, _ in self.words(c.args) if not t.startswith("-")]
        target = args[0] if args else self.p.home
        if target != "-":
            self.cwd = self.p.norm(target, self.cwd)


# ----------------------------------------------------------------------------- bash helpers


def _quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _has_short(flag: str, letter: str) -> bool:
    return flag.startswith("-") and not flag.startswith("--") and letter in flag[1:]


def _flag_payload(args: list[str], letters: tuple[str, ...]) -> str | None:
    """`bash -lc 'cmd'` / `python3 -c 'code'` / `perl -e'code'` → the payload, else None."""
    for i, a in enumerate(args):
        if not a.startswith("-"):
            return None  # first positional is a script file: not inspectable
        if a.startswith("--") or not any(ch in a[1:] for ch in letters):
            continue
        if a[-1] in letters:
            return args[i + 1] if i + 1 < len(args) else None
        if a[1] in letters:
            return a[2:]  # payload attached: -e'code'
    return None


def _path_candidates(t: str) -> list[str]:
    """A word plus path-ish pieces inside it (`--key=~/x`, `@file`, `if=...`, `host:path`)."""
    out = [t]
    if "=" in t:
        out.append(t.split("=", 1)[1])
    if t.startswith("@"):
        out.append(t[1:])
    if ":" in t and not re.match(r"^[a-z][a-z0-9+.-]*://", t, re.IGNORECASE):
        out.append(t.split(":", 1)[1])
    return [o for o in out if o]


def _find_roots(args: list[str]) -> list[str]:
    roots = []
    for a in args:
        if a.startswith("-") or a in ("(", "!", ")"):
            break
        roots.append(a)
    return roots


def _find_exec_rm(args: list[str]) -> bool:
    for i, a in enumerate(args):
        if a in ("-exec", "-execdir", "-ok", "-okdir"):
            seg = " ".join(args[i + 1 : i + 8])
            if re.search(r"(^|[/\s'\"])(rm|srm|shred|unlink)\b", seg):
                return True
    return False


def _is_recursive_copy(name: str, raw: list[str]) -> bool:
    if name in ("cp", "scp"):
        return any(_has_short(a, "r") or _has_short(a, "R") or _has_short(a, "a") or a == "--recursive"
                   for a in raw)
    if name in ("zip",):
        return any(_has_short(a, "r") for a in raw)
    if name in ("grep", "egrep", "fgrep"):
        return any(_has_short(a, "r") or _has_short(a, "R") or a.startswith("--recursive") for a in raw)
    return True


def _has_subcommand(raw: list[str], sub: tuple[str, ...]) -> bool:
    positional: list[str] = []
    skip = False
    for a in raw:
        if skip:
            skip = False
            continue
        if a in ("-C", "-c", "--git-dir", "--work-tree", "-R", "--repo", "--prefix", "--cwd",
                 "--workspace", "--profile", "--region"):
            skip = True
            continue
        if a.startswith("-"):
            continue
        positional.append(a)
        if len(positional) >= len(sub):
            break
    return tuple(p.lower() for p in positional[: len(sub)]) == sub


_CURL_DATA_LONG = ("--data", "--data-binary", "--data-raw", "--data-urlencode", "--data-ascii",
                   "--json", "--form", "--form-string", "--upload-file", "-d", "-F", "-T")


def _curl_uploads(raw: list[str]) -> bool:
    for i, a in enumerate(raw):
        if a.startswith(_CURL_DATA_LONG):
            return True
        if a.startswith(("--request", "-X")):
            method = a.split("=", 1)[1] if "=" in a else (a[2:] if a.startswith("-X") and len(a) > 2
                                                          else (raw[i + 1] if i + 1 < len(raw) else ""))
            if method.upper() in ("POST", "PUT", "PATCH", "DELETE"):
                return True
        if a.startswith("-") and not a.startswith("--") and len(a) > 2:
            cluster = a[1:]
            for j, ch in enumerate(cluster):
                if ch in "dFT":
                    return True
                if ch == "X":
                    method = cluster[j + 1 :] or (raw[i + 1] if i + 1 < len(raw) else "")
                    if method.upper() in ("POST", "PUT", "PATCH", "DELETE"):
                        return True
                    break
                if ch in "oHAeuUwxbcEK":  # options that take an argument: rest is the value
                    break
    return False


def _httpie_uploads(raw: list[str]) -> bool:
    if any(a in ("-f", "--form", "--multipart", "--raw") for a in raw):
        return True
    positional = [a for a in raw if not a.startswith("-")]
    if not positional:
        return False
    method = positional[0].upper()
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        return True
    items = positional[2:] if method in ("GET", "HEAD", "OPTIONS") else positional[1:]
    return any(re.match(r"^[^=:@/]+(:=@?|=@?|@)", it) and not re.match(r"^[^=:@/]+==", it)
               for it in items)


def _url_host(words: list[str]) -> str:
    for w in words:
        m = re.match(r"^(?:[a-z][a-z0-9+.-]*://)?([A-Za-z0-9.-]+(?::\d+)?)(?:/|$)", w)
        if m and ("." in m.group(1) or "localhost" in m.group(1) or ":" in m.group(1)) and "=" not in w:
            return m.group(1).lower()
    return ""


def _is_remote_spec(a: str) -> bool:
    if re.match(r"^rsync://", a):
        return True
    m = re.match(r"^([^/:@\s]+@)?([A-Za-z0-9._-]+):", a)
    return bool(m) and not a.startswith("/")


# ----------------------------------------------------------------------------- AppleScript


def _classify_applescript(script: str, analyze_shell: Callable[[str], GateDecision]) -> GateDecision:
    low = script.lower()
    found: list[GateDecision] = []
    if _AS_ADMIN.search(low):
        return GateDecision("catastrophic", "admin", "AppleScript asks for administrator privileges")
    for m in _AS_DO_SHELL.finditer(script):
        found.append(analyze_shell(m.group(1).replace('\\"', '"').replace("\\\\", "\\")))
    app = _AS_SEND_APPS.search(low) or MESSAGING_APPS.search(low)
    app_name = app.group(0) if app else ""
    if _AS_SEND.search(low) and _AS_SEND_APPS.search(low):
        found.append(GateDecision("outward", "osascript_send", f"AppleScript sends via {app_name}",
                                  target=app_name))
    if _AS_RETURN.search(low) and MESSAGING_APPS.search(low):
        found.append(GateDecision("outward", "osascript_keystroke_send",
                                  f"AppleScript presses Return in {app_name}", target=app_name))
    for m in _AS_CLICK_VERB.finditer(script):
        if OUTWARD_VERBS.search(m.group(1)):
            found.append(GateDecision("outward", "osascript_click", f"AppleScript clicks '{m.group(1)}'",
                                      target=m.group(1)))
    if _AS_DELETE.search(low):
        found.append(GateDecision("outward", "osascript_delete", "AppleScript deletes items", target=app_name))
    return _worst(found)


# ----------------------------------------------------------------------------- per-tool


_SHELL_TOOLS = {"Bash", "Monitor"}  # Monitor's `command` is a shell command, classified the same way
_FILE_WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
_FILE_READ_TOOLS = {"Read", "NotebookRead"}
_CHROME_CLICKS = {"left_click", "right_click", "double_click", "triple_click", "left_click_drag"}


def _file_tool(name: str, inp: dict[str, Any], paths: _Paths) -> GateDecision:
    cwd = paths.ctx.workspace
    if name in _FILE_WRITE_TOOLS:
        target = str(inp.get("file_path") or inp.get("notebook_path") or "")
        if target and paths.is_memory_file(target, cwd):
            return GateDecision("sensitive_read", MEMORY_WRITE, "memory files are append-only via remember")
        reason = paths.sensitive_reason(target, cwd, write=True) if target else None
        if reason:
            return GateDecision("sensitive_read", "sensitive_write", reason)
        return (paths.write_decision(target, cwd) if target else None) or ALLOW
    if name in _FILE_READ_TOOLS:
        target = str(inp.get("file_path") or inp.get("notebook_path") or "")
        reason = paths.sensitive_reason(target, cwd) if target else None
        return GateDecision("sensitive_read", "sensitive_path", reason) if reason else ALLOW
    if name == "Grep":
        base = str(inp.get("path") or cwd)
        checks = [(base, False)]
        if inp.get("glob"):
            checks.append((posixpath.join(base, str(inp["glob"])), True))
        for p, g in checks:
            reason = paths.sensitive_reason(p, cwd, g)
            if reason:
                return GateDecision("sensitive_read", "sensitive_path", reason)
        return ALLOW
    if name == "Glob":
        base = str(inp.get("path") or cwd)
        pattern = str(inp.get("pattern") or "")
        for p, g in ((base, False), (posixpath.join(base, pattern) if pattern else base, True)):
            reason = paths.sensitive_reason(p, cwd, g)
            if reason and not (g and posixpath.basename(pattern) in ("*", "**")
                               and not paths.sensitive_reason(posixpath.dirname(p), cwd, True)):
                return GateDecision("sensitive_read", "sensitive_path", reason)
        return ALLOW
    return ALLOW


def _mac_ui(name: str, inp: dict[str, Any], ctx: GateContext) -> GateDecision:
    app = ""
    if ctx.current_app:
        try:
            app = ctx.current_app() or ""
        except Exception:  # noqa: BLE001
            app = ""
    messaging = bool(MESSAGING_APPS.search(app))
    if name == "mcp__mac__ui_click":
        eid = inp.get("element_id")
        label = ""
        if eid not in (None, "") and ctx.element_label:
            label = ctx.element_label(str(eid)) or ""
        if label and OUTWARD_VERBS.search(label):
            return GateDecision("outward", "ui_click_verb", f"clicks '{label}' in {app or 'an app'}",
                                target=f"{app}:{label}")
        if not label and messaging:  # coordinate click in a messaging app: could be Send
            return GateDecision("outward", "ui_click_coord", f"clicks somewhere in {app}", target=app)
        return ALLOW
    if name == "mcp__mac__ui_key":
        combo = re.sub(r"\s+", "", str(inp.get("combo", "")).lower())
        if combo in UI_RETURN_COMBOS and messaging:
            return GateDecision("outward", "ui_key_send", f"presses {combo} in {app}", target=app)
        return ALLOW
    if name == "mcp__mac__ui_type":
        text = str(inp.get("text", ""))
        if ("\n" in text or "\r" in text) and messaging:
            return GateDecision("outward", "ui_type_send", f"types a line break in {app}", target=app)
    return ALLOW


def _chrome_action(action: dict[str, Any], ctx: GateContext) -> GateDecision:
    kind = str(action.get("action", ""))
    labels = [str(action.get("action_summary") or "")]
    ref = action.get("ref")
    if ref and ctx.chrome_ref_label:
        labels.append(ctx.chrome_ref_label(str(ref)) or "")
    label = " ".join(x for x in labels if x)
    host = ""
    if ctx.chrome_tab_host and action.get("tabId") is not None:
        host = ctx.chrome_tab_host(action["tabId"]) or ""
    enter = kind == "key" and bool(re.search(r"\b(enter|return)\b", str(action.get("text", "")), re.IGNORECASE))
    newline = kind == "type" and bool(re.search(r"[\r\n]$", str(action.get("text", ""))))
    acts = kind in _CHROME_CLICKS or enter or newline
    if acts and OUTWARD_VERBS.search(label):
        return GateDecision("outward", f"chrome_{kind}_verb", f"browser {kind}: {label}",
                            target=f"{host}:{label}")
    if acts and host and OUTWARD_HOSTS.search(host):
        return GateDecision("outward", f"chrome_{kind}_host", f"browser {kind} on {host}", target=host)
    return ALLOW


def _iter_dicts(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_dicts(v)


def _chrome(name: str, inp: dict[str, Any], ctx: GateContext) -> GateDecision:
    found = [_chrome_action(d, ctx) for d in _iter_dicts(inp) if "action" in d]
    if name.endswith("javascript_tool"):
        code = " ".join(str(v) for d in _iter_dicts(inp) for v in d.values() if isinstance(v, str))
        if _JS_OUTWARD.search(code) or (_JS_CLICK.search(code) and OUTWARD_VERBS.search(code)):
            found.append(GateDecision("outward", "chrome_js", "page script submits or sends data",
                                      target="javascript"))
    return _worst(found)


# ----------------------------------------------------------------------------- public API


def _app_name(ctx: GateContext) -> str:
    if not ctx.current_app:
        return ""
    try:
        return ctx.current_app() or ""
    except Exception:  # noqa: BLE001
        return ""


def _strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def _whatsapp_ui(tool_name: str, inp: dict[str, Any], ctx: GateContext) -> GateDecision | None:
    """Chat profile (M1): any use of WhatsApp's own screens, reads included. None: not WhatsApp."""
    hit = GateDecision("catastrophic", WHATSAPP_UI, REASON_WHATSAPP_UI, target="whatsapp")
    texts = list(_strings(inp))
    if tool_name.startswith("mcp__claude-in-chrome__"):
        if any(_WA_URL.search(t) for t in texts):
            return hit
        if ctx.chrome_tab_host:
            for d in _iter_dicts(inp):
                for key in ("tabId", "tab_id"):
                    if d.get(key) is not None and WHATSAPP_HOSTS.search(ctx.chrome_tab_host(d[key]) or ""):
                        return hit
        return None
    if tool_name.startswith("mcp__mac__ui_") or tool_name in ("mcp__mac__screenshot", "mcp__mac__open_app"):
        named = " ".join(str(inp.get(k, "")) for k in ("name", "app"))
        if "whatsapp" in named.lower() or "whatsapp" in _app_name(ctx).lower():
            return hit
        if any(_WA_URL.search(t) for t in texts):
            return hit
        return None
    if tool_name in _SHELL_TOOLS:
        cmd = str(inp.get("command", ""))
        low = cmd.lower()
        scripted = "whatsapp" in low and re.search(r"\b(osascript|jxa|system events)\b", low)
        if _WA_SCRIPT.search(cmd) or _WA_URL.search(cmd) or scripted or _open_targets_whatsapp(cmd):
            return hit
        return None
    if any(_WA_URL.search(t) for t in texts) and tool_name.startswith("mcp__mac__"):
        return hit
    return None


def _open_targets_whatsapp(cmd: str) -> bool:
    """`open -a WhatsApp`, `open -b net.whatsapp.WhatsApp`, `open /Applications/WhatsApp.app`."""
    for pipeline in shell.parse(cmd).pipelines:
        for sc in pipeline:
            words = [w.text for w in sc.words]
            while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
                words.pop(0)
            if words and posixpath.basename(words[0]) == "open" and any(
                    "whatsapp" in w.lower() for w in words[1:]):
                return True
    return False


def _never_chat(tool_name: str, inp: dict[str, Any], d: GateDecision, ctx: GateContext) -> GateDecision | None:
    """Chat profile: an outward action whose target names a never-listed chat, or a confirmation
    summary that does (the summary must name the destination, so every send passes here)."""
    names = [n.strip().lower() for n in ctx.never_chat_names if n.strip()]
    if not names:
        return None
    if tool_name == MAC_CONFIRM:
        texts = [str(inp.get("summary", "")), str(inp.get("question", ""))]
    elif d.verdict == "outward":
        texts = [d.target, d.reason]
    else:
        return None
    return _never_named(" ".join(texts), ctx)


def _plain(text: str) -> str:
    """L1: NFKC, format characters (zero-width and the like) dropped, case-folded."""
    t = unicodedata.normalize("NFKC", text)
    return "".join(c for c in t if unicodedata.category(c) != "Cf").casefold()


def _never_named(text: str, ctx: GateContext) -> GateDecision | None:
    """`text` names a never-listed chat (NFKC, case-folded, at word boundaries)."""
    blob = _plain(text)
    for raw in ctx.never_chat_names:
        n = _plain(raw).strip()
        if n and re.search(rf"(?<![^\W_]){re.escape(n)}(?![^\W_])", blob):
            return GateDecision("catastrophic", NEVER_CHAT, f"'{n}' is on the never-list; Mac never reads "
                                f"from or sends to it", target=n)
    return None


_LEAF_DEPTH, _LEAF_MAX = 4, 200


def _leaves(obj: Any, depth: int = 0) -> Iterable[str]:
    if isinstance(obj, str):
        yield unicodedata.normalize("NFKC", obj)
    elif depth < _LEAF_DEPTH and isinstance(obj, dict | list):
        for v in obj.values() if isinstance(obj, dict) else obj:
            yield from _leaves(v, depth + 1)


def _unparsed(tool_name: str, inp: dict[str, Any], d: GateDecision, ctx: GateContext) -> GateDecision | None:
    """X1: a tool the gate has no parser for (user MCP servers, Skill, ...): every string in its
    input (depth 4, 200 leaves) is checked against the never-list, reads included, and every
    path-like one against the secrets and ~/.mac-agent rules."""
    paths = _Paths(ctx)
    write = d.verdict == "outward" or bool(UNKNOWN_MCP_VERBS.search(tool_name.rsplit("__", 1)[-1]))
    for i, leaf in enumerate(_leaves(inp)):
        if i >= _LEAF_MAX:
            break
        if ctx.never_chat_names:
            never = _never_named(leaf, ctx)
            if never is not None:
                return never
        raw = leaf.strip()
        if raw.lower().startswith("file://"):
            raw = raw[7:]
        if raw.startswith(("/", "~")):
            reason = paths.sensitive_reason(raw, ctx.workspace, write=write)
            if reason:
                return GateDecision("sensitive_read", "sensitive_write" if write else "sensitive_path", reason)
            guarded = paths.write_decision(raw, ctx.workspace) if write else None
            if guarded is not None:
                return guarded
    return None


def _confirm_files(ctx: GateContext) -> bool:
    flag = ctx.confirm_send_file
    if not callable(flag):
        return bool(flag)
    try:
        return bool(flag())
    except Exception:  # the live profile could not be read: ask
        log.exception("gate: confirm_send_file unavailable")
        return True


def _mac_send_file(inp: dict[str, Any], paths: _Paths, ctx: GateContext) -> GateDecision:
    target = str(inp.get("path", ""))
    reason = paths.sensitive_reason(target, ctx.workspace) if target else None
    if reason:
        return GateDecision("sensitive_read", "sensitive_path", reason)
    if target and paths.is_memory_file(target, ctx.workspace):
        return GateDecision("sensitive_read", "sensitive_path", f"{target} is Mac's private memory")
    if _confirm_files(ctx):
        return GateDecision("outward", "send_file", "sends a file into this chat",
                            target=posixpath.basename(target))
    return ALLOW


def analyze(tool_name: str, tool_input: dict[str, Any] | None,
            context: GateContext | None = None) -> GateDecision:
    """Classify one tool call. Never runs anything (reads script files passed by path)."""
    inp = tool_input or {}
    ctx = context or GateContext()
    try:
        if ctx.whatsapp_ui_block:
            wa = _whatsapp_ui(tool_name, inp, ctx)
            if wa is not None:
                return wa
        d, commands = _analyze(tool_name, inp, ctx)
        if ctx.never_chat_names:
            never = _never_chat(tool_name, inp, d, ctx)
            if never is not None:
                return never
        if not map_rows(tool_name) or tool_name in _UNPARSED_TOO:
            unparsed = _unparsed(tool_name, inp, d, ctx)
            if unparsed is not None:
                return unparsed
    except Exception:
        log.exception("gate: classifier error for %s", tool_name)
        if ctx.capabilities is not None:  # N2: a chat never gets a yes for a call it could not check
            return GateDecision("sensitive_read", "classifier_error", "Mac could not check this action")
        return GateDecision("outward", "classifier_error", "could not verify this action")
    if ctx.capabilities is None or d.verdict == "catastrophic":  # keep the stronger denial
        return d
    return _capability(tool_name, d, commands, ctx) or d


def _analyze(tool_name: str, inp: dict[str, Any], ctx: GateContext) -> tuple[GateDecision, frozenset[str]]:
    """The decision, plus the command names a Bash call runs (empty for other tools)."""
    paths = _Paths(ctx)
    if tool_name in _SHELL_TOOLS:
        bash = _BashAnalyzer(paths)
        return bash.analyze(str(inp.get("command", ""))), frozenset(bash.commands)
    return _analyze_tool(tool_name, inp, ctx, paths), frozenset()


def _analyze_tool(tool_name: str, inp: dict[str, Any], ctx: GateContext, paths: _Paths) -> GateDecision:
    if tool_name == MAC_SEND_FILE:
        return _mac_send_file(inp, paths, ctx)
    if tool_name == MAC_SET_PERSONA:
        return _mac_set_persona(inp, ctx)
    if tool_name == MAC_TEST_OUTWARD:  # harmless tool for the live test: always needs a yes
        return GateDecision("outward", "test_outward", "test action that needs confirmation",
                            target=str(inp.get("target", "test")))
    if tool_name in _FILE_WRITE_TOOLS | _FILE_READ_TOOLS | {"Grep", "Glob"}:
        return _file_tool(tool_name, inp, paths)
    if tool_name.startswith("mcp__mac__ui_"):
        return _mac_ui(tool_name, inp, ctx)
    if tool_name.startswith("mcp__claude-in-chrome__"):
        return _chrome(tool_name, inp, ctx)
    if tool_name.startswith("mcp__") and not tool_name.startswith("mcp__mac__"):
        short = tool_name.rsplit("__", 1)[-1]
        if UNKNOWN_MCP_VERBS.search(short):  # user's own MCP servers: name-based, conservative
            return GateDecision("outward", "mcp_write", f"{short} looks like it sends or changes something",
                                target=tool_name)
    return ALLOW


# ----------------------------------------------------------------------------- capabilities (W9)

# W9.1 (P5): every tool a chat brain can call has exactly ONE row ("" = no capability needed).
# Patterns are fnmatch. A tool that matches no row (user MCP servers, Skill, harness tools) is
# allowed in chat and logged by name (user decision, v3.5); the gate's other rules still apply.
TOOL_MAP: dict[str, tuple[str, ...]] = {
    "": ("mcp__mac__say", "mcp__mac__request_confirmation", "mcp__mac__notify",
         "mcp__mac__switch_workspace", "mcp__mac__set_persona", "TodoWrite", "Task", "Agent",
         "ToolSearch",  # ToolSearch only loads deferred tools; each one is gated when called
         "Skill",  # a skill only reaches other tools, each gated by its own row (tasks D8)
         "SubagentHandback",  # a sub-agent handing its result to its parent: nothing leaves the session
         # task tools talk to agentd's task table; grounding in the user's words is their rule (9.2)
         "mcp__mac__task_create", "mcp__mac__task_list", "mcp__mac__task_update", "mcp__mac__task_delete",
         "mcp__mac__task_run_now", "mcp__mac__task_continue", "mcp__mac__task_stop", "mcp__mac__task_history"),
    "files_read": ("Read", "NotebookRead", "Glob", "Grep", "LS", "mcp__mac__remember", "mcp__mac__forget",
                   "mcp__mac__rewrite_memory"),
    "files_write": ("Write", "Edit", "MultiEdit", "NotebookEdit", "Bash", "BashOutput", "KillShell",
                    "Monitor", "TaskStop"),  # Monitor runs a shell command; TaskStop stops one
    "chrome": ("mcp__claude-in-chrome__*",),
    "web": ("WebSearch", "WebFetch"),
    "desktop": ("mcp__mac__ui_*", "mcp__mac__screenshot", "mcp__mac__open_app", "mcp__mac__frontmost"),
    "send_file": ("mcp__mac__send_file",),
    "outward_sends": ("mcp__mac__test_outward",),
    "read_chat": (),  # no tool: the group's messages and members arrive as context (W10.6)
}
# X2: an unclassified tool could read, write, fetch or send, so it needs all four levers on.
UNCLASSIFIED_NEEDS: tuple[str, ...] = ("files_read", "files_write", "web", "outward_sends")
# Bash commands that drive the screen or other apps (osascript UI scripting, `open`): + desktop.
_DESKTOP_COMMANDS = frozenset({"osascript", "open", "screencapture", "cliclick", "shortcuts"})
# Outward tools whose yes is their own rule, not an outward send (send_file: confirm_files).
_OWN_CONFIRM = frozenset({MAC_SEND_FILE, MAC_SET_PERSONA})


def map_rows(tool_name: str) -> list[str]:
    """The TOOL_MAP rows (capability names, "" = none needed) that list `tool_name`."""
    return [cap for cap, pats in TOOL_MAP.items() if any(fnmatch.fnmatchcase(tool_name, p) for p in pats)]


def _classes(tool_name: str, d: GateDecision, commands: frozenset[str]) -> set[str]:
    rows = map_rows(tool_name)
    classes = set(rows) - {""}  # an unmapped tool needs no capability (user decision, v3.5)
    if commands & _DESKTOP_COMMANDS:
        classes.add("desktop")
    if d.verdict == "outward" and tool_name not in _OWN_CONFIRM:
        classes.add("outward_sends")  # a webmail send: chrome + outward_sends
    return classes


def tool_classes(tool_name: str, tool_input: dict[str, Any] | None,
                 context: GateContext | None = None) -> set[str]:
    """The capabilities a call needs (W9.1): its TOOL_MAP row (none for an unmapped tool), +
    desktop for a Bash call that runs osascript or `open`, + outward_sends for any outward
    verdict. Uses the same parsing as `analyze`."""
    ctx = context or GateContext()
    d, commands = _analyze(tool_name, tool_input or {}, ctx)
    return _classes(tool_name, d, commands)


def _granted(ctx: GateContext) -> frozenset[str]:
    caps = ctx.capabilities
    try:
        got = caps() if callable(caps) else caps
        return frozenset(got or ())
    except Exception:  # the live profile could not be read: fail closed
        log.exception("gate: capabilities unavailable")
        return frozenset()


def is_unattended(ctx: GateContext | None) -> bool:
    """True while a scheduled run owns this brain (nobody can answer a confirmation). An
    unreadable flag counts as unattended (fail closed)."""
    flag = ctx.unattended if ctx is not None else None
    if flag is None:
        return False
    try:
        return bool(flag())
    except Exception:
        log.exception("gate: unattended flag unavailable")
        return True


def _capability(tool_name: str, d: GateDecision, commands: frozenset[str],
                ctx: GateContext) -> GateDecision | None:
    granted = _granted(ctx)  # one read of the live profile per decision
    where = f"Mac \u25b8 WhatsApp \u25b8 {ctx.display_name}" if ctx.display_name else "Mac \u25b8 WhatsApp"
    if not map_rows(tool_name):
        # X2: a tool Mac can't classify runs only while every lever it could bypass is on (the
        # default, so the allow-all decision holds); otherwise turning one off would mean nothing.
        off = [c for c in UNCLASSIFIED_NEEDS if c not in granted]
        if not off:
            log.info("gate: unmapped tool %s in a chat: allowed (all levers on)", tool_name)
            return None
        if off == ["outward_sends"] and is_unattended(ctx):  # a run: only sending is off, say why (9.3)
            return GateDecision("sensitive_read", f"{CAPABILITY}:outward_sends", UNATTENDED_REASON, target="outward_sends")
        w = CAPABILITY_WORDS[off[0]]
        return GateDecision("sensitive_read", f"{CAPABILITY}:{off[0]}",
                            f"this chat has {w} off, so tools Mac can't classify are off too. You can turn "
                            f'it on in {where}, or say "Mac, allow {w}"', target=off[0])
    classes = _classes(tool_name, d, commands)
    missing = [c for c in CAPABILITIES if c in classes and c not in granted]
    if not missing:
        return None
    cap = missing[0]  # `code` stays internal; the reason is plain words the model relays (UX5)
    if cap == "outward_sends" and is_unattended(ctx):  # a run: nobody to ask, say so plainly (9.3)
        return GateDecision("sensitive_read", f"{CAPABILITY}:{cap}", UNATTENDED_REASON, target=cap)
    return GateDecision("sensitive_read", f"{CAPABILITY}:{cap}",
                        f"this chat can't use {CAPABILITY_WORDS[cap]}. You can turn it on in {where}, "
                        f'or say "Mac, allow {CAPABILITY_WORDS[cap]}"', target=cap)


def _mac_set_persona(inp: dict[str, Any], ctx: GateContext) -> GateDecision:
    """A bound yes for every valid new persona; the yes is tied to this exact text. An invalid
    text is allowed through so the tool itself returns the error, without asking anyone."""
    text, problem = check_persona(inp.get("text"))
    if problem:
        return ALLOW
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    # UX4: the target is the chat's name (display only; the digest in the code binds the yes)
    return GateDecision("outward", f"set_persona:{digest}", "changes the instructions for this chat",
                        target=ctx.display_name or "this chat", detail=text)


def classify(tool_name: str, tool_input: dict[str, Any] | None,
             context: GateContext | None = None) -> GateVerdict:
    return analyze(tool_name, tool_input, context).verdict


# ----------------------------------------------------------------------------- tokens


@dataclass
class _Token:
    turn_id: int
    expires_at: float
    action: str  # fingerprint of the confirmed action


@dataclass
class _Pending:
    turn_id: int
    summary: str
    future: asyncio.Future[bool]


class ConfirmationGate:
    """Human-in-the-loop confirmation tokens (ARCHITECTURE §6.2).

    The model asks through ``mcp__mac__request_confirmation``; the CONFIRM BrainEvent tells
    the orchestrator to speak the summary; a human yes/no arrives via ``resolve()``. Approval
    mints ONE token for ONE action: the fingerprint of the outward action the hook just denied.
    Only that action may use it, as often as needed (idempotent retry), until the TTL expires,
    the turn changes or ``revoke_all()``. A token is never minted without an action to bind to.
    """

    def __init__(self, *, ttl_s: float = 120.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl_s = ttl_s
        self._clock = clock
        self._token: _Token | None = None
        self._pending: _Pending | None = None

    @property
    def pending(self) -> tuple[int, str] | None:
        p = self._pending
        return (p.turn_id, p.summary) if p and not p.future.done() else None

    async def request(self, turn_id: int, summary: str, timeout: float, action: str) -> bool:
        """Wait for resolve(); False on timeout/revoke, if another request is pending, or if
        there is no `action` (fingerprint of a denied outward call) to bind the token to."""
        if not action:
            log.warning("gate: refusing to mint an unbound token (no denied action)")
            return False
        if self.pending is not None:
            log.warning("gate: confirmation already pending; rejecting second request")
            return False
        fut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending = _Pending(turn_id, summary, fut)
        tool, code = (action.split("|") + ["", ""])[:2]  # M1: never message text; a hash to correlate
        log.info("gate: confirmation requested turn=%d tool=%s code=%s ref=%s", turn_id, tool, code,
                 hashlib.sha256(f"{action}\n{summary}".encode()).hexdigest()[:12])
        try:
            approved = await asyncio.wait_for(asyncio.shield(fut), timeout)
        except TimeoutError:
            log.info("gate: confirmation timed out turn=%d", turn_id)
            approved = False
        finally:
            if self._pending is not None and self._pending.future is fut:
                self._pending = None
            if not fut.done():
                fut.set_result(False)
        if approved:
            self._token = _Token(turn_id, self._clock() + self.ttl_s, action)
            log.info("gate: token minted turn=%d action=%s", turn_id, action)
        return approved

    def resolve(self, approved: bool) -> None:
        p = self._pending
        if p is None or p.future.done():
            log.debug("gate: resolve(%s) with nothing pending", approved)
            return
        p.future.set_result(bool(approved))

    def has_token(self, turn_id: int, action: str | None = None) -> bool:
        t = self._token
        if not t or t.turn_id != turn_id or self._clock() >= t.expires_at:
            return False
        return action is None or t.action == action

    def consume(self, turn_id: int, action: str) -> bool:
        """True if the token covers exactly `action` (retries of it are fine)."""
        if not self.has_token(turn_id, action):
            return False
        log.info("gate: token used turn=%d action=%s", turn_id, action)
        return True

    def revoke_all(self) -> None:
        self._token = None
        self.resolve(False)
