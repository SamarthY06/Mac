from __future__ import annotations

import json
import time
from pathlib import Path

from macd.eventlog import EventLog, Metrics


def read_all(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_records_have_required_fields(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    log.start()
    log.emit("state", turn_id=3, state="thinking")
    log.emit("daemon_start")
    log.close()
    recs = read_all(tmp_path / "events.jsonl")
    assert [r["type"] for r in recs] == ["state", "daemon_start"]
    assert recs[0]["turn_id"] == 3 and recs[0]["state"] == "thinking"
    assert "turn_id" not in recs[1]
    for r in recs:
        assert isinstance(r["ts"], float) and isinstance(r["mono"], float)


def test_rotation(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog(path, max_bytes=4_000, backups=2)
    log.start()
    for i in range(400):
        log.emit("tick", i=i, pad="x" * 50)
    log.close()
    assert path.exists()
    assert (tmp_path / "events.jsonl.1").exists()
    assert (tmp_path / "events.jsonl.2").exists()
    assert not (tmp_path / "events.jsonl.3").exists()
    for p in (path, tmp_path / "events.jsonl.1", tmp_path / "events.jsonl.2"):
        assert p.stat().st_size <= 4_000 + 200
        read_all(p)  # every file is valid JSONL
    # newest records are in the live file
    assert read_all(path)[-1]["i"] == 399


def test_emit_never_blocks_and_counts_drops(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl", queue_size=50)
    # writer not started: the queue fills and further emits are dropped immediately
    t0 = time.monotonic()
    results = [log.emit("spam", i=i) for i in range(10_000)]
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5
    assert results.count(True) == 50
    assert log.dropped == 10_000 - 50
    log.start()
    deadline = time.monotonic() + 2
    while not log.emit("after") and time.monotonic() < deadline:
        time.sleep(0.01)
    log.close()
    recs = read_all(tmp_path / "events.jsonl")
    assert recs[0]["eventlog_dropped"] >= 10_000 - 50
    assert recs[-1]["type"] == "after"


def test_unserializable_values_are_stringified(tmp_path: Path) -> None:
    import numpy as np

    log = EventLog(tmp_path / "e.jsonl")
    log.start()
    log.emit("x", obj=object(), n=np.float32(1.5), s={1, 2})
    log.close()
    rec = read_all(tmp_path / "e.jsonl")[0]
    assert rec["n"] == 1.5 and rec["s"] == [1, 2] and isinstance(rec["obj"], str)


def test_metrics_percentiles() -> None:
    m = Metrics(window=100)
    for v in range(1, 101):
        m.observe("turn_total", float(v))
    m.incr("turns")
    m.incr("turns", 2)
    snap = m.snapshot()
    lat = snap["latency_ms"]["turn_total"]
    assert lat["n"] == 100
    assert abs(lat["p50"] - 50.5) < 0.6
    assert abs(lat["p95"] - 95.05) < 0.6
    assert snap["counters"]["turns"] == 3
    assert m.latency("wake_to_earcon")["p50"] is None and "wake_to_earcon" not in snap["latency_ms"]
    # bounded window
    for _ in range(500):
        m.observe("turn_total", 1.0)
    assert m.snapshot()["latency_ms"]["turn_total"]["n"] == 100
