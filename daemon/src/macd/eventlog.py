"""Append-only JSONL event log + in-process latency metrics.

EventLog
- `emit()` never blocks the asyncio loop: records go into a bounded `queue.Queue` that a
  single daemon writer thread drains to disk. When the queue is full the record is dropped
  and `dropped` is incremented (the drop count is written into the next record that fits).
- Size-based rotation: events.jsonl -> events.jsonl.1 ... .N (default 10 MB x 5).
- Every record has `ts` (unix epoch, float), `mono` (time.monotonic) and `type`;
  `turn_id` is included when given.

Metrics
- counters (`incr`) and latency histograms (`observe`, milliseconds) over a bounded
  window of recent samples; `snapshot()` -> {"counters": {...}, "latency_ms": {name:
  {"n", "p50", "p95", "last"}}}.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_BACKUPS = 5

_STOP = object()


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return f"<ndarray shape={o.shape}>"
    except ImportError:  # pragma: no cover - numpy always present in practice
        pass
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    if hasattr(o, "value") and not callable(o.value):  # enums
        return o.value
    return repr(o)


class EventLog:
    def __init__(
        self,
        path: Path | str,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backups: int = DEFAULT_BACKUPS,
        queue_size: int = 10_000,
    ) -> None:
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.backups = backups
        self._q: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.dropped = 0
        self.written = 0
        self.write_errors = 0
        self._closed = False

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._writer, name="eventlog-writer", daemon=True)
        self._thread.start()

    def close(self, timeout: float = 2.0) -> None:
        """Flush and stop the writer thread (blocking; call via asyncio.to_thread on the loop)."""
        if self._closed:
            return
        self._closed = True
        if self._thread is None:
            return
        try:
            self._q.put(_STOP, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout)
        self._thread = None

    # ------------------------------------------------------------------ producer side

    def emit(self, type: str, turn_id: int | None = None, **fields: Any) -> bool:
        """Queue one record. Never blocks. Returns False if dropped."""
        if self._closed:
            return False
        rec: dict[str, Any] = {"ts": round(time.time(), 6), "mono": round(time.monotonic(), 6), "type": type}
        if turn_id is not None:
            rec["turn_id"] = turn_id
        rec.update(fields)
        try:
            self._q.put_nowait(rec)
            return True
        except queue.Full:
            with self._lock:
                self.dropped += 1
            return False

    # ------------------------------------------------------------------ writer thread

    def _writer(self) -> None:
        fh = None
        size = 0
        reported_drops = 0
        try:
            while True:
                item = self._q.get()
                batch = [item]
                # Drain whatever else is ready so we write in batches.
                while len(batch) < 512:
                    try:
                        batch.append(self._q.get_nowait())
                    except queue.Empty:
                        break
                stop = False
                lines: list[str] = []
                for rec in batch:
                    if rec is _STOP:
                        stop = True
                        continue
                    with self._lock:
                        drops = self.dropped
                    if drops != reported_drops:
                        rec = dict(rec, eventlog_dropped=drops)
                        reported_drops = drops
                    try:
                        lines.append(json.dumps(rec, default=_json_default, ensure_ascii=False) + "\n")
                    except (TypeError, ValueError):
                        self.write_errors += 1
                if lines:
                    try:
                        if fh is None:
                            fh, size = self._open()
                        pending: list[bytes] = []
                        pending_n = 0
                        for line in lines:
                            data = line.encode("utf-8")
                            if size + pending_n + len(data) > self.max_bytes and size + pending_n > 0:
                                if pending:
                                    fh.write(b"".join(pending))
                                fh.close()
                                fh = None
                                self._rotate()
                                fh, size = self._open()
                                pending, pending_n = [], 0
                            pending.append(data)
                            pending_n += len(data)
                        if pending:
                            fh.write(b"".join(pending))
                        size += pending_n
                        self.written += len(lines)
                    except OSError:
                        self.write_errors += 1
                        if fh is not None:
                            try:
                                fh.close()
                            except OSError:
                                pass
                        fh = None
                if stop:
                    return
        finally:
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass

    def _open(self) -> tuple[Any, int]:
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        fh = os.fdopen(fd, "ab", buffering=0)
        return fh, os.fstat(fd).st_size

    def _rotate(self) -> None:
        if self.backups <= 0:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            return
        for i in range(self.backups - 1, 0, -1):
            src = self.path.with_name(f"{self.path.name}.{i}")
            if src.exists():
                os.replace(src, self.path.with_name(f"{self.path.name}.{i + 1}"))
        if self.path.exists():
            os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))


class NullEventLog:
    """Drop-in that records nothing (tests)."""

    dropped = 0

    def start(self) -> None:
        return

    def close(self, timeout: float = 0.0) -> None:
        return

    def emit(self, type: str, turn_id: int | None = None, **fields: Any) -> bool:
        return True


# ----------------------------------------------------------------------------- metrics

def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return math.nan
    k = (len(sorted_vals) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


class Metrics:
    """Counters + latency histograms. Loop-thread only (no locking needed)."""

    def __init__(self, window: int = 500) -> None:
        self.counters: collections.Counter[str] = collections.Counter()
        self._window = window
        self._hist: dict[str, collections.deque[float]] = {}
        self.started = time.monotonic()

    def incr(self, name: str, n: int = 1) -> None:
        self.counters[name] += n

    def observe(self, name: str, ms: float) -> None:
        if ms < 0 or math.isnan(ms):
            return
        h = self._hist.get(name)
        if h is None:
            h = self._hist[name] = collections.deque(maxlen=self._window)
        h.append(float(ms))

    def observe_since(self, name: str, t0_mono: float) -> float:
        ms = (time.monotonic() - t0_mono) * 1000.0
        self.observe(name, ms)
        return ms

    def latency(self, name: str) -> dict[str, float | int]:
        vals = sorted(self._hist.get(name, ()))
        last = self._hist[name][-1] if self._hist.get(name) else math.nan
        return {
            "n": len(vals),
            "p50": round(_percentile(vals, 0.5), 1) if vals else None,  # type: ignore[dict-item]
            "p95": round(_percentile(vals, 0.95), 1) if vals else None,  # type: ignore[dict-item]
            "last": round(last, 1) if vals else None,  # type: ignore[dict-item]
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "uptime_s": round(time.monotonic() - self.started, 1),
            "counters": dict(self.counters),
            "latency_ms": {name: self.latency(name) for name in self._hist},
        }
