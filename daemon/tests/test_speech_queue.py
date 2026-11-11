from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import numpy as np
import pytest
from orch_fakes import FakeOutput, FakeTTS, wait_until

from macd.speech_queue import SYSTEM_TURN, SpeechQueue, resume_offset


@pytest.fixture
async def running() -> AsyncIterator[tuple[SpeechQueue, FakeTTS, FakeOutput]]:
    tts = FakeTTS(delay=0.0)
    out = FakeOutput(tts, play_s=0.01)
    q = SpeechQueue(tts, {"local": out})
    task = asyncio.create_task(q.run())
    try:
        yield q, tts, out
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_ordering_and_callbacks() -> None:
    tts = FakeTTS(delay=0.002)
    out = FakeOutput(tts, play_s=0.005)
    starts: list[str] = []
    idles: list[int] = []
    q = SpeechQueue(tts, {"local": out}, on_start=lambda text, kind: starts.append(text), on_idle=lambda: idles.append(1))
    task = asyncio.create_task(q.run())
    try:
        q.set_current_turn(1)
        for t in ("One.", "Two.", "Three."):
            q.say(1, t, "answer")
        await asyncio.wait_for(q.drain(1), 2)
        assert out.spoken == ["One.", "Two.", "Three."]
        assert starts == ["One.", "Two.", "Three."]
        await wait_until(lambda: idles == [1])
        assert not q.speaking
    finally:
        task.cancel()


async def test_prefetch_overlaps_synthesis_with_playback() -> None:
    tts = FakeTTS(delay=0.05)
    out = FakeOutput(tts, play_s=0.05)
    q = SpeechQueue(tts, {"local": out})
    task = asyncio.create_task(q.run())
    try:
        q.set_current_turn(1)
        for i in range(4):
            q.say(1, f"Sentence {i}.", "answer")
        t0 = time.monotonic()
        await asyncio.wait_for(q.drain(1), 3)
        elapsed = time.monotonic() - t0
        # serial would be 4 * (0.05 + 0.05) = 0.4 s; pipelined ≈ 0.05 + 4 * 0.05 = 0.25 s
        assert elapsed < 0.36, elapsed
        # synthesis of item 2 started before item 1 finished playing
        play1_start = out.play_started[0][1]
        synth2_start = dict(tts.started)["Sentence 1."]
        assert synth2_start < play1_start + 0.05
        assert out.spoken == [f"Sentence {i}." for i in range(4)]
    finally:
        task.cancel()


async def test_stale_items_dropped_including_inflight_synthesis() -> None:
    tts = FakeTTS(delay=0.1)
    out = FakeOutput(tts, play_s=0.01)
    q = SpeechQueue(tts, {"local": out})
    task = asyncio.create_task(q.run())
    try:
        q.set_current_turn(1)
        q.say(1, "Old turn sentence.", "answer")
        q.say(1, "Another old one.", "answer")
        await wait_until(lambda: tts.started, msg="synthesis started")
        q.set_current_turn(2)  # synthesis for turn 1 is in flight
        q.say(2, "New turn.", "answer")
        await asyncio.wait_for(q.drain(2), 2)
        await asyncio.sleep(0.15)
        assert out.spoken == ["New turn."]
        assert q.dropped >= 2
        assert not q.say(1, "late", "answer")  # enqueue for stale turn refused
    finally:
        task.cancel()


async def test_cancel_turn_latency_under_50ms() -> None:
    tts = FakeTTS()
    out = FakeOutput(tts, play_s=5.0)
    q = SpeechQueue(tts, {"local": out})
    task = asyncio.create_task(q.run())
    try:
        q.set_current_turn(3)
        q.say(3, "A very long answer.", "answer")
        q.say(3, "Queued follow-up.", "answer")
        await wait_until(lambda: out.play_started)
        t0 = time.monotonic()
        q.cancel_turn(3)
        assert out.flushes >= 1  # synchronous flush
        await wait_until(lambda: out.cancelled_at, timeout=1)
        assert out.cancelled_at[0] - t0 < 0.05
        await asyncio.wait_for(q.drain(3), 0.5)
        assert out.spoken == []
        assert q.queued == 0
    finally:
        task.cancel()


async def test_drain_semantics(running: tuple[SpeechQueue, FakeTTS, FakeOutput]) -> None:
    q, _tts, out = running
    await asyncio.wait_for(q.drain(42), 0.1)  # nothing queued → immediate
    q.set_current_turn(5)
    q.say(5, "Hello.", "answer")
    q.say(5, "World.", "answer")
    assert q.pending(5) == 2
    await asyncio.wait_for(q.drain(5), 1)
    assert out.spoken == ["Hello.", "World."]
    assert q.pending(5) == 0


async def test_system_turn_survives_generation_change(running: tuple[SpeechQueue, FakeTTS, FakeOutput]) -> None:
    q, _tts, out = running
    q.set_current_turn(7)
    q.say(SYSTEM_TURN, "Okay, stopped.", "filler")
    q.set_current_turn(8)
    await asyncio.wait_for(q.drain(SYSTEM_TURN), 1)
    assert out.spoken == ["Okay, stopped."]


async def test_routing_by_output_and_pcm_items() -> None:
    tts = FakeTTS()
    local, remote = FakeOutput(tts, "local"), FakeOutput(tts, "remote")
    q = SpeechQueue(tts, {"local": local, "remote": remote})
    task = asyncio.create_task(q.run())
    try:
        q.set_current_turn(1)
        q.say(1, "For the phone.", "answer", output="remote")
        q.say(1, "For the room.", "answer")
        q.play_pcm(1, np.zeros(100, dtype=np.float32), 24_000, "earcon", output="local")
        q.say(1, "Unknown output falls back.", "answer", output="nope")
        await asyncio.wait_for(q.drain(1), 1)
        assert remote.spoken == ["For the phone."]
        assert local.spoken == ["For the room.", "<earcon@24000>", "Unknown output falls back."]
    finally:
        task.cancel()


async def test_tts_failure_skips_item(running: tuple[SpeechQueue, FakeTTS, FakeOutput]) -> None:
    q, tts, out = running

    async def boom(text: str) -> np.ndarray:
        if "bad" in text:
            raise RuntimeError("kokoro crashed")
        return tts.encode(text)

    tts.synthesize = boom  # type: ignore[method-assign]
    q.set_current_turn(1)
    q.say(1, "bad sentence", "answer")
    q.say(1, "Good sentence.", "answer")
    await asyncio.wait_for(q.drain(1), 1)
    assert out.spoken == ["Good sentence."]


async def test_bounded_queue_sheds_low_priority() -> None:
    tts = FakeTTS()
    q = SpeechQueue(tts, {"local": FakeOutput(tts)}, maxsize=3)
    q.set_current_turn(1)
    q.say(1, "Narrating.", "narration")
    q.say(1, "A.", "answer")
    q.say(1, "B.", "answer")
    q.say(1, "C.", "answer")
    assert q.queued == 3
    assert [it.text for it in q._q] == ["A.", "B.", "C."]


async def test_priority_item_jumps_the_queue() -> None:
    tts = FakeTTS()
    out = FakeOutput(tts, play_s=0.02)
    q = SpeechQueue(tts, {"local": out})
    task = asyncio.create_task(q.run())
    try:
        q.set_current_turn(1)
        q.say(1, "First sentence.", "answer")
        q.say(1, "Second sentence.", "answer")
        q.say(1, "Third sentence.", "answer")
        await wait_until(lambda: out.play_started)
        q.say(1, "Got it.", "filler", priority=True)
        await asyncio.wait_for(q.drain(1), 2)
        assert out.spoken == ["First sentence.", "Got it.", "Second sentence.", "Third sentence."]
    finally:
        task.cancel()


# ----------------------------------------------------------------------------- pause / resume (barge-in)


@pytest.fixture
async def slow() -> AsyncIterator[tuple[SpeechQueue, FakeTTS, FakeOutput]]:
    """1 s of PCM per text, played in 1 s: playback position tracks wall time."""
    tts = FakeTTS(samples=16_000)
    out = FakeOutput(tts, play_s=1.0)
    q = SpeechQueue(tts, {"local": out})
    task = asyncio.create_task(q.run())
    try:
        yield q, tts, out
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_resume_offset_snaps_back_to_a_word_gap() -> None:
    sr = 16_000
    pcm = np.full(sr, 0.5, dtype=np.float32)
    pcm[int(0.50 * sr):int(0.56 * sr)] = 0.0  # a gap between two words
    at = resume_offset(pcm, sr, int(0.8 * sr))
    assert int(0.50 * sr) <= at < int(0.56 * sr)  # replays the cut word from its start
    assert resume_offset(pcm, sr, 100) == 0
    assert resume_offset(pcm, sr, 10 * sr) <= sr  # clamped to the item


async def test_pause_holds_the_remainder_and_resume_plays_it_first(slow) -> None:
    q, _, out = slow
    q.set_current_turn(1)
    q.say(1, "first sentence here", "answer")
    q.say(1, "second sentence", "answer")
    await wait_until(lambda: out.play_started)
    await asyncio.sleep(0.2)
    assert q.pause() and q.paused and q.speaking and q.playing_turn == 1
    await asyncio.sleep(0.1)
    assert len(out.play_started) == 1 and out.cancelled_at  # cut, and nothing else started
    assert q.pending(1) == 2  # remainder + the queued sentence: drain() still waits
    assert q.resume() and not q.paused
    await asyncio.wait_for(q.drain(1), 5)
    assert out.spoken == ["first sentence here", "second sentence"]
    assert q.words_heard(1) == 5


async def test_pause_ignores_system_speech_and_silence(slow) -> None:
    q, _, out = slow
    assert not q.pause()  # nothing playing
    q.say(SYSTEM_TURN, "Okay, stopped.")
    await wait_until(lambda: out.play_started)
    assert not q.pause() and not q.paused


async def test_cancel_while_paused_counts_what_was_heard_and_unpauses(slow) -> None:
    q, _, out = slow
    q.set_current_turn(1)
    q.say(1, "one two three four five six seven eight nine ten", "answer")
    await wait_until(lambda: out.play_started)
    await asyncio.sleep(0.5)
    q.pause()
    q.cancel_turn(1)
    await asyncio.sleep(0.02)  # the consumer releases the cut item
    assert not q.paused and q.pending(1) == 0
    assert 2 <= q.words_heard(1) <= 6  # about half of it, from the resume point
    q.say(SYSTEM_TURN, "Okay, stopped.")
    await wait_until(lambda: "Okay, stopped." in out.spoken, timeout=3)


async def test_flush_turn_keeps_the_turn_live(slow) -> None:
    q, _, out = slow
    q.set_current_turn(1)
    q.say(1, "a b c d", "answer")
    q.say(1, "never heard", "answer")
    await wait_until(lambda: out.play_started)
    await asyncio.sleep(0.25)
    q.flush_turn(1)
    await asyncio.sleep(0.02)
    assert q.pending(1) == 0 and q.words_heard(1) == 1  # a quarter of four words
    assert q.say(1, "the reply", "answer")  # unlike cancel_turn, later speech still plays
    await asyncio.wait_for(q.drain(1), 5)
    assert out.spoken == ["the reply"]


async def test_pause_during_synthesis_holds_the_whole_item() -> None:
    tts = FakeTTS(delay=0.1, samples=1600)
    out = FakeOutput(tts, play_s=0.1)
    q = SpeechQueue(tts, {"local": out})
    task = asyncio.create_task(q.run())
    try:
        q.set_current_turn(1)
        q.say(1, "on it", "filler")
        await wait_until(lambda: tts.started)
        assert q.pause()
        await asyncio.sleep(0.2)
        assert out.play_started == [] and q.pending(1) == 1
        q.resume()
        await asyncio.wait_for(q.drain(1), 2)
        assert out.spoken == ["on it"]
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
