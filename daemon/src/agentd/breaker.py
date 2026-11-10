"""One rate-limit breaker per agentd (W3.6, F20): one Claude plan means one breaker.

Any conversation's `error{kind: rate_limit}` trips it. While it is open, every new turn in every
conversation ends at once with `error{kind: rate_limit}` and the reset time, without starting
`claude`. It closes by itself at the reset time.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

log = logging.getLogger(__name__)

DEFAULT_COOLOFF_S = 300.0  # when Claude gives no reset time
MAX_COOLOFF_S = 6 * 3600.0  # never trust a reset time further out than this


class PlanBreaker:
    def __init__(self, *, clock: Callable[[], float] = time.time,
                 default_cooloff_s: float = DEFAULT_COOLOFF_S) -> None:
        self._clock = clock
        self.default_cooloff_s = default_cooloff_s
        self.reset_at = 0.0  # wall time; 0 = closed

    def trip(self, reset_at: float | None = None) -> None:
        now = self._clock()
        until = reset_at if reset_at and reset_at > now else now + self.default_cooloff_s
        until = min(until, now + MAX_COOLOFF_S)
        if until > self.reset_at:
            self.reset_at = until
            log.warning("breaker: open until %s", time.strftime("%H:%M:%S", time.localtime(until)))

    def is_open(self) -> bool:
        return self._clock() < self.reset_at

    def close(self) -> None:
        self.reset_at = 0.0

    def message(self) -> str:
        """Plain words with the reset time, for `error{kind: rate_limit}`."""
        at = time.strftime("%I:%M %p", time.localtime(self.reset_at)).lstrip("0")
        return f"I've hit the Claude usage limit. It resets at {at}."
