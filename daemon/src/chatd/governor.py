"""The outbound governor (W5.4, F5): every send, edit, reaction and typing update of every adapter
is charged here first (R1), and the counters are persisted, so a crash loop cannot reset them.

Budgets: 20 sends and 30 edits per rolling minute, 60 new messages per hour and 300 per day.
Reactions and typing count toward the per-minute sends (typing as 0.1). An optional lifetime
`total` (tests and the live take) caps every outbound action but typing; once it is spent the
governor stays disabled.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agentd.stores import atomic_write_json, read_json

log = logging.getLogger(__name__)

Kind = Literal["send", "edit", "react", "typing"]
_DAY, _HOUR, _MIN = 86400.0, 3600.0, 60.0


class BudgetExceeded(Exception):
    """An outbound action over budget. Adapters raise it before any I/O."""


@dataclass(frozen=True)
class Budget:
    sends_per_min: float = 20
    edits_per_min: int = 30
    new_per_hour: int = 60
    new_per_day: int = 300
    total: int | None = None  # lifetime cap on send/edit/react (None = none)


class Governor:
    def __init__(self, path: Path | None = None, budget: Budget | None = None, *,
                 clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path) if path else None  # None: counters in memory only (test mode)
        self.budget = budget or Budget()
        self.clock = clock
        self.log: list[tuple[float, str]] = []  # (time, kind) within the last day
        self.used = 0  # actions counted toward `total`
        self.counts: dict[str, int] = {"send": 0, "edit": 0, "react": 0, "typing": 0}
        self._lock = asyncio.Lock()

    async def load(self) -> Governor:
        if self.path is None:
            return self
        raw = await asyncio.to_thread(read_json, self.path, {})
        if isinstance(raw, dict):
            now = self.clock()
            self.log = [(float(t), str(k)) for t, k in raw.get("log", [])
                        if isinstance(t, int | float) and now - t < _DAY]
            self.used = int(raw.get("used", 0) or 0)
        return self

    @property
    def disabled(self) -> bool:
        return self.budget.total is not None and self.used >= self.budget.total

    def _since(self, seconds: float, kinds: tuple[str, ...]) -> float:
        cut = self.clock() - seconds
        return sum(0.1 if k == "typing" else 1.0 for t, k in self.log if t > cut and k in kinds)

    def allowed(self, kind: Kind) -> bool:
        b = self.budget
        if kind != "typing" and self.disabled:
            return False
        if kind == "edit":
            return self._since(_MIN, ("edit",)) + 1 <= b.edits_per_min
        weight = 0.1 if kind == "typing" else 1.0
        if self._since(_MIN, ("send", "react", "typing")) + weight > b.sends_per_min + 1e-9:
            return False
        if kind == "send":
            return (self._since(_HOUR, ("send",)) + 1 <= b.new_per_hour
                    and self._since(_DAY, ("send",)) + 1 <= b.new_per_day)
        return True

    async def charge(self, kind: Kind, chat_id: str = "") -> None:
        """Count one action, or raise BudgetExceeded (nothing counted)."""
        async with self._lock:
            if not self.allowed(kind):
                log.warning("governor: %s over budget; dropped", kind)
                raise BudgetExceeded(f"{kind} over budget")
            now = self.clock()
            self.log = [(t, k) for t, k in self.log if now - t < _DAY]
            self.log.append((now, kind))
            self.counts[kind] = self.counts.get(kind, 0) + 1
            if kind != "typing":
                self.used += 1
            if self.path is not None:
                snap = {"log": [[t, k] for t, k in self.log], "used": self.used}
                await asyncio.to_thread(atomic_write_json, self.path, snap)
