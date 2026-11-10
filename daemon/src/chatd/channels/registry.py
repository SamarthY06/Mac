"""Channel registry (W2.5): a channel is one module under chatd/channels/ with
`create(cfg, ctx) -> ChannelAdapter`, plus one line in ADAPTERS.

`ChannelContext` is what chatd hands every adapter: the governor (R1), the allowlist and the
never-list (R2), the handled-ids store (R4) and the dry-run flag. `SeenIds` is the persisted
`seen.json` (F15): the last 2,000 handled message ids, marked only after agentd accepted.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentd.stores import atomic_write_json, read_json
from chatd.channels.base import ChannelAdapter, DestinationRefused
from chatd.governor import Governor

log = logging.getLogger(__name__)

ADAPTERS: dict[str, str] = {
    "cli": "chatd.channels.cli",
    "whatsapp": "chatd.channels.whatsapp",
}
TEST_ADAPTERS: dict[str, str] = {"fake": "chatd.channels.fake"}  # only with ctx.test (AG25)

SEEN_MAX = 2_000
MAX_AGE_S = 7 * 86400.0  # older messages are never acted on (W5.2 check 5)


class SeenIds:
    """Handled message ids, oldest first, persisted atomically (None path: memory only)."""

    def __init__(self, path: Path | None, maxlen: int = SEEN_MAX) -> None:
        self.path = Path(path) if path else None
        self.maxlen = maxlen
        self._ids: OrderedDict[str, None] = OrderedDict()
        self._lock = asyncio.Lock()

    async def load(self) -> SeenIds:
        if self.path is not None:
            raw = await asyncio.to_thread(read_json, self.path, [])
            for i in raw if isinstance(raw, list) else []:
                self._ids[str(i)] = None
        return self

    def __contains__(self, key: str) -> bool:
        return key in self._ids

    async def add(self, key: str) -> None:
        async with self._lock:
            self._ids[key] = None
            self._ids.move_to_end(key)
            while len(self._ids) > self.maxlen:
                self._ids.popitem(last=False)
            if self.path is not None:
                await asyncio.to_thread(atomic_write_json, self.path, list(self._ids))


@dataclass
class ChannelContext:
    governor: Governor
    chats: frozenset[str] = frozenset()  # allowlisted chat ids: empty until agentd's profiles arrive (W9.8)
    never_chats: frozenset[str] = frozenset()  # wins over `chats`
    seen: SeenIds = field(default_factory=lambda: SeenIds(None))
    dry_run: bool = False
    test: bool = False
    clock: Callable[[], float] = time.time

    def allowed(self, chat_id: str) -> bool:
        return chat_id in self.chats and chat_id not in self.never_chats

    def check_destination(self, chat_id: str) -> None:
        """R2: raise before any I/O for a never-listed or non-allowlisted chat."""
        if not self.allowed(chat_id):
            raise DestinationRefused("That chat is not one Mac may write to.")

    def too_old(self, sent_at: float) -> bool:
        return bool(sent_at) and self.clock() - sent_at > MAX_AGE_S


# W9.9: chatd config keeps process keys only. The per-chat keys moved to agentd's profiles (its
# one-time seed read them); chatd ignores them and drops them the next time it writes the config.
PER_CHAT_KEYS = ("chats", "allow_shared_chat", "trigger")


def process_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """A copy of chatd's config without the per-chat keys of any `[channels.<name>]` section, with
    one log line naming what was dropped (never the values)."""
    out = dict(cfg)
    channels = cfg.get("channels")
    if not isinstance(channels, dict):
        return out
    out["channels"], dropped = {}, []
    for name, sec in channels.items():
        if isinstance(sec, dict):
            dropped += [f"{name}.{k}" for k in PER_CHAT_KEYS if k in sec]
            sec = {k: v for k, v in sec.items() if k not in PER_CHAT_KEYS}
        out["channels"][name] = sec
    if dropped:
        log.warning("chatd: config keys %s are now per-chat profiles in the Mac app; ignored "
                    "(they are removed the next time the config is written)", ", ".join(dropped))
    return out


def create(name: str, cfg: dict[str, Any], ctx: ChannelContext) -> ChannelAdapter:
    module = ADAPTERS.get(name) or (TEST_ADAPTERS.get(name) if ctx.test else None)
    if module is None:
        raise ValueError(f"unknown channel {name!r}")
    adapter: ChannelAdapter = importlib.import_module(module).create(cfg, ctx)
    return adapter
