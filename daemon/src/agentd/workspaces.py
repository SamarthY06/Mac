"""Folders, approved workspaces and the settings hash (W3.5, F2, M2).

Claude may work anywhere in home except the skip list. A folder's own project settings load only
for a workspace you approved, and the approval is pinned to a hash of everything those settings
can run. The hash is re-checked at every turn start; on a mismatch the folder drops to user
settings until it is approved again.

Everything here that touches the disk is blocking and is called through `asyncio.to_thread` (V7).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentd import offload
from agentd import schema as s
from agentd.stores import JsonStore

log = logging.getLogger(__name__)

SKIP_UNDER_HOME = ("Library", ".Trash", ".cache", ".npm", ".cargo", ".rustup", ".local/share",
                   ".mac-agent", ".claude")
SKIP_NAMES = frozenset({"node_modules", ".git", ".venv"})
SKIP_SUFFIXES = (".photoslibrary",)
CHAT_APPROVAL_MAX_BYTES = 2 * 1024 * 1024  # a bigger .claude/ tree is approved in the app only
MAX_WALK_DIRS = 20_000  # bound on the nested-CLAUDE.md walk
MAX_IMPORT_DEPTH = 5
SPOTLIGHT_MAX_NAME = 80
SPOTLIGHT_TIMEOUT_S = 5.0
SEEDS = ("~/Desktop/Personal/Resume",)
_PROJECT_FILES = (".mcp.json", "CLAUDE.md", "CLAUDE.local.md")
_IMPORT_RE = re.compile(r"(?:^|\s)@((?:~/|\.{1,2}/|/)?[\w.\-/]+)")


# ----------------------------------------------------------------------------- paths


def under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def is_skipped(path: Path, home: Path) -> bool:
    """True for anything outside home or on the skip list (never routed to or searched)."""
    path = Path(os.path.abspath(path))
    if not under(path, home):
        return True
    rel = path.relative_to(home)
    if any(under(path, home / d) for d in SKIP_UNDER_HOME):
        return True
    return any(p in SKIP_NAMES or p.endswith(SKIP_SUFFIXES) for p in rel.parts)


def display(path: str | Path | None, home: Path) -> str | None:
    """`~/...` form for status and lists."""
    if path is None:
        return None
    p = Path(path)
    return "~/" + p.relative_to(home).as_posix() if under(p, home) and p != home else str(p)


def resolve_user_path(raw: str, home: Path) -> Path | None:
    """A path from a request (`~/x`, `/Users/.../x`) as an absolute path under home, or None."""
    if not raw or "\x00" in raw:
        return None
    p = Path(raw)
    if raw.startswith("~"):
        p = home / raw[1:].lstrip("/")
    if not p.is_absolute():
        return None
    p = Path(os.path.normpath(p))
    return p if under(p, home) and p != home else None


# ----------------------------------------------------------------------------- settings hash (M2)


@dataclass
class SettingsScan:
    digest: str
    claude_bytes: int = 0  # size of the .claude/ tree
    has_project_settings: bool = False  # the folder itself has .claude/, .mcp.json or CLAUDE.md
    hooks: list[str] = field(default_factory=list)  # command strings
    skills: list[str] = field(default_factory=list)
    servers: list[str] = field(default_factory=list)  # "name: command"
    files: list[str] = field(default_factory=list)  # hashed keys (tests, diagnostics)


def _walk_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for f in sorted(filenames):
            yield Path(dirpath) / f


def _commands(obj: Any) -> Iterator[str]:
    """Every `command` string in a settings object (hooks, statusLine, ...)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "command" and isinstance(v, str):
                yield v
            else:
                yield from _commands(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _commands(v)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _command_files(command: str, folder: Path, home: Path) -> list[Path]:
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    out = []
    for w in words:
        w = w.replace("${CLAUDE_PROJECT_DIR}", str(folder)).replace("$CLAUDE_PROJECT_DIR", str(folder))
        if w.startswith("~"):
            w = str(home / w[1:].lstrip("/"))
        if "/" not in w and "." not in w:
            continue
        p = Path(w) if os.path.isabs(w) else folder / w
        p = Path(os.path.normpath(p))
        if under(p, folder) and p.is_file():
            out.append(p)
    return out


def _imports(md: Path, home: Path, depth: int, seen: set[Path]) -> Iterator[Path]:
    if depth > MAX_IMPORT_DEPTH:
        return
    try:
        text = md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for m in _IMPORT_RE.finditer(text):
        raw = m.group(1)
        p = home / raw[2:] if raw.startswith("~/") else (Path(raw) if raw.startswith("/") else md.parent / raw)
        p = Path(os.path.normpath(p))
        if p in seen or not under(p, home) or not p.is_file():
            continue
        seen.add(p)
        yield p
        yield from _imports(p, home, depth + 1, seen)


def settings_hash(folder: Path, home: Path) -> SettingsScan:
    """Blocking. SHA-256 over a path-sorted list of (path, content) covering the M2 scope:
    the whole .claude/ tree, .mcp.json, CLAUDE.md, CLAUDE.local.md, every nested CLAUDE.md (skip
    list applied), @import targets inside home, ancestor CLAUDE.md and CLAUDE.local.md up to home,
    and every file a settings `command` names inside the folder."""
    folder = Path(os.path.abspath(folder))
    files: set[Path] = set()
    scan = SettingsScan(digest="")
    claude = folder / ".claude"
    if claude.is_dir():
        scan.has_project_settings = True
        for p in _walk_files(claude):
            if p.is_file():
                files.add(p)
                try:
                    scan.claude_bytes += p.stat().st_size
                except OSError:
                    pass
        for name in ("settings.json", "settings.local.json"):
            for cmd in _commands(_load_json(claude / name)):
                scan.hooks.append(cmd)
                files.update(_command_files(cmd, folder, home))
        skills = claude / "skills"
        if skills.is_dir():
            scan.skills = sorted(p.name for p in skills.iterdir() if p.is_dir())
    for name in _PROJECT_FILES:
        if (folder / name).is_file():
            scan.has_project_settings = True
            files.add(folder / name)
    mcp = _load_json(folder / ".mcp.json")
    if isinstance(mcp, dict) and isinstance(mcp.get("mcpServers"), dict):
        for sname, spec in sorted(mcp["mcpServers"].items()):
            cmd = spec.get("command", spec.get("url", "")) if isinstance(spec, dict) else ""
            args = " ".join(spec.get("args", [])) if isinstance(spec, dict) and isinstance(spec.get("args"), list) else ""
            scan.servers.append(f"{sname}: {cmd} {args}".strip())
    # nested CLAUDE.md below the folder (skip list applied, bounded walk)
    for visited, (dirpath, dirnames, filenames) in enumerate(os.walk(folder), 1):
        if visited > MAX_WALK_DIRS:
            files.add(Path("/walk-truncated"))
            break
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_NAMES and not d.endswith(SKIP_SUFFIXES) and d != ".claude"
                             and not is_skipped(Path(dirpath) / d, home))
        if "CLAUDE.md" in filenames:
            files.add(Path(dirpath) / "CLAUDE.md")
    # ancestors up to and including home
    p = folder.parent
    while under(p, home):
        for name in ("CLAUDE.md", "CLAUDE.local.md"):
            if (p / name).is_file():
                files.add(p / name)
        if p == home:
            break
        p = p.parent
    seen: set[Path] = set()
    for md in [f for f in files if f.name in ("CLAUDE.md", "CLAUDE.local.md")]:
        files.update(_imports(md, home, 1, seen))

    h = hashlib.sha256()
    for f in sorted(files, key=lambda x: _key(x, folder, home)):
        key = _key(f, folder, home)
        scan.files.append(key)
        try:
            content = hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else "missing"
        except OSError:
            content = "unreadable"
        h.update(f"{key}\0{content}\n".encode())
    scan.digest = h.hexdigest()
    return scan


def _key(f: Path, folder: Path, home: Path) -> str:
    if under(f, folder):
        return "folder:" + f.relative_to(folder).as_posix()
    if under(f, home):
        return "home:" + f.relative_to(home).as_posix()
    return "abs:" + f.as_posix()


def approval_text(name: str, scan: SettingsScan) -> str:
    """The folder_approval prompt (W3.5). Hooks by command string (80 chars), servers by command."""
    parts = [f"skills: {', '.join(scan.skills)}" if scan.skills else "no skills",
             f"hooks: {'; '.join(c[:80] for c in scan.hooks)}" if scan.hooks else "no hooks",
             f"servers: {'; '.join(x[:80] for x in scan.servers)}" if scan.servers else "no extra servers"]
    return (f"{name} has its own Claude setup ({'; '.join(parts)}). Use {name} as a workspace? "
            "Reply to this message with yes.")


# ----------------------------------------------------------------------------- Spotlight (N-a)


def spotlight_query(name: str) -> str | None:
    """The mdfind query for a folder named `name`, or None for a name we refuse (N-a)."""
    if not name or len(name) > SPOTLIGHT_MAX_NAME or "*" in name or "?" in name:
        return None
    if any(ord(c) < 32 for c in name):
        return None
    esc = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'kMDItemContentType == "public.folder" && kMDItemFSName == "{esc}"c'


def _run_mdfind(args: list[str]) -> str:
    r = subprocess.run(args, capture_output=True, text=True, timeout=SPOTLIGHT_TIMEOUT_S, check=False)
    return r.stdout


def spotlight_find(name: str, home: Path, run: Callable[[list[str]], str] = _run_mdfind) -> list[Path]:
    """Blocking. Folders in home named `name` (case-insensitive), skip-list hits dropped."""
    q = spotlight_query(name)
    if q is None:
        return []
    try:
        out = run(["mdfind", "-onlyin", str(home), q])
    except (OSError, subprocess.SubprocessError):
        log.warning("router: mdfind failed")
        return []
    found = []
    for line in out.splitlines():
        p = Path(line.strip())
        if line.strip() and p.is_absolute() and not is_skipped(p, home) and p.is_dir():
            found.append(p)
    return sorted(set(found))


# ----------------------------------------------------------------------------- store


@dataclass
class Trust:
    approved: bool  # an approval exists
    matches: bool  # and the current hash equals the approved one
    scan: SettingsScan

    @property
    def setting_sources(self) -> list[str]:
        return ["user", "project", "local"] if self.approved and self.matches else ["user"]

    @property
    def needs_approval(self) -> bool:
        return self.scan.has_project_settings and not (self.approved and self.matches)


class WorkspaceStore:
    """`~/.mac-agent/agent/state/workspaces.json`: {"approved": {path: {hash, at, aliases}},
    "declined": {path: hash}}. Paths are absolute."""

    def __init__(self, path: Path, home: Path, *, seeds: tuple[str, ...] = SEEDS,
                 hasher: Callable[[Path, Path], SettingsScan] = settings_hash) -> None:
        self.home = Path(home)
        self._store = JsonStore(path)
        self._seeds = seeds
        self._hasher = hasher
        self.approved: dict[str, dict[str, Any]] = {}
        self.declined: dict[str, str] = {}

    async def load(self) -> None:
        d = await self._store.load({})
        d = d if isinstance(d, dict) else {}
        self.approved = {k: v for k, v in (d.get("approved") or {}).items() if isinstance(v, dict)}
        self.declined = {k: v for k, v in (d.get("declined") or {}).items() if isinstance(v, str)}

    async def _save(self) -> None:
        await self._store.save(lambda: {"approved": self.approved, "declined": self.declined})

    async def scan(self, folder: Path) -> SettingsScan:
        return await offload.run(self._hasher, folder, self.home)  # a process, not a thread (V7)

    async def trust(self, folder: Path) -> Trust:
        scan = await self.scan(folder)
        entry = self.approved.get(str(folder))
        return Trust(entry is not None, entry is not None and entry.get("hash") == scan.digest, scan)

    async def approve(self, folder: Path, digest: str | None = None) -> str:
        """Approve `folder` pinned to `digest` (the hash shown in the prompt), or the current one."""
        if digest is None:
            digest = (await self.scan(folder)).digest
        entry = self.approved.get(str(folder), {})
        self.approved[str(folder)] = {"hash": digest, "at": time.time(), "aliases": entry.get("aliases", [])}
        self.declined.pop(str(folder), None)
        await self._save()
        log.info("workspaces: approved %s", folder.name)
        return digest

    async def decline(self, folder: Path, digest: str) -> None:
        self.declined[str(folder)] = digest
        await self._save()

    def was_declined(self, folder: Path, digest: str) -> bool:
        return self.declined.get(str(folder)) == digest

    async def remove(self, folder: Path) -> None:
        self.approved.pop(str(folder), None)
        await self._save()

    def entries(self) -> list[tuple[Path, list[str]]]:
        """Approved folders that still exist, with their aliases."""
        return [(Path(p), list(e.get("aliases") or [])) for p, e in sorted(self.approved.items())]

    async def listing(self) -> list[s.Workspace]:
        out = []
        paths = [Path(p) for p in sorted(self.approved)]
        for seed in self._seeds:
            p = resolve_user_path(seed, self.home)
            if p is not None and p not in paths and await asyncio.to_thread(p.is_dir):
                paths.append(p)
        for p in paths:
            entry = self.approved.get(str(p))
            changed = False
            if entry is not None:
                changed = (await self.scan(p)).digest != entry.get("hash")
            out.append(s.Workspace(path=display(p, self.home) or str(p), name=p.name,
                                   approved=entry is not None, settings_changed=changed))
        return out
