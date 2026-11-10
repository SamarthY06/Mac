"""Atomic JSON stores (W0): one writer per file, mkstemp in the same dir, fsync, os.replace.

Writes run in a thread (V7) under an asyncio.Lock, so the event loop never waits on fsync and two
writes of one store never interleave. A corrupt file is moved aside and the store starts empty.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def atomic_write_json(path: Path, obj: Any) -> None:
    """Blocking: write `obj` as JSON to `path` atomically (0600). Call it from a thread."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


def read_json(path: Path, default: Any) -> Any:
    """Blocking: the JSON in `path`, or `default` if missing. A corrupt file is kept aside."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        aside = path.with_name(path.name + ".corrupt")
        log.error("store: %s is not valid JSON; moved to %s and starting empty", path, aside.name)
        with contextlib.suppress(OSError):
            os.replace(path, aside)
        return default


class JsonStore:
    """One JSON file with a single async writer. `save(snapshot)` takes the snapshot inside the
    lock, so the last write always carries the latest state."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()

    async def load(self, default: Any) -> Any:
        return await asyncio.to_thread(read_json, self.path, default)

    async def save(self, snapshot: Callable[[], Any]) -> None:
        async with self._lock:
            obj = snapshot()
            await asyncio.to_thread(atomic_write_json, self.path, obj)
