from __future__ import annotations

import asyncio

import pytest

from macd.brain import fake_brain as fb
from macd.contracts import BrainErrorKind, BrainEvent
from macd.contracts import BrainEventKind as K


async def collect(brain: fb.FakeBrain, until_done: int = 1, timeout: float = 3) -> list[BrainEvent]:
    out: list[BrainEvent] = []
    done = 0

    async def run() -> None:
        nonlocal done
        async for ev in brain.events():
            out.append(ev)
            if ev.kind in (K.DONE, K.ERROR):
                done += 1
                if done >= until_done:
                    return

    await asyncio.wait_for(run(), timeout)
    return out


async def test_scripted_turn() -> None:
    brain = fb.FakeBrain({"hi": [fb.say("One sec."), *fb.tool("Bash", {"command": "ls"}), fb.text("Hello!")]}.get)
    await brain.start()
    await brain.ask(1, "hi")
    evs = await collect(brain)
    assert [e.kind for e in evs] == [K.SAY, K.TOOL_START, K.TOOL_END, K.TEXT, K.DONE]
    assert all(e.turn_id == 1 for e in evs)
    assert evs[-1].data == {"interrupted": False}
    await brain.close()


async def test_default_reply_and_callable_script() -> None:
    brain = fb.FakeBrain(lambda t: [fb.text(t.upper())])
    await brain.start()
    await brain.ask(1, "abc")
    evs = await collect(brain)
    assert evs[0].text == "ABC"
    await brain.close()


async def test_interrupt_drains_and_next_ask_waits() -> None:
    long = [fb.text(f"{i} ", delay=0.05) for i in range(50)]
    brain = fb.FakeBrain({"long": long, "short": [fb.text("51")]}.get, drain_s=0.1)
    await brain.start()
    await brain.ask(1, "long")
    await asyncio.sleep(0.12)
    await brain.interrupt(1)
    await brain.ask(2, "short")
    evs = await collect(brain, until_done=2)
    t1 = [e for e in evs if e.turn_id == 1]
    assert t1[-1].kind == K.DONE and t1[-1].data["interrupted"] is True
    assert len([e for e in t1 if e.kind == K.TEXT]) < 10
    # no turn-1 event after turn 2 began
    first2 = next(i for i, e in enumerate(evs) if e.turn_id == 2)
    assert all(e.turn_id == 2 for e in evs[first2:])
    assert [e.text for e in evs if e.turn_id == 2 and e.kind == K.TEXT] == ["51"]
    await brain.close()


async def test_interrupt_finished_turn_is_noop() -> None:
    brain = fb.FakeBrain()
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    await brain.interrupt(1)
    await brain.ask(2, "y")
    evs = await collect(brain)
    assert [e.kind for e in evs] == [K.TEXT, K.DONE] and evs[0].turn_id == 2
    await brain.close()


async def test_interrupt_before_start() -> None:
    brain = fb.FakeBrain({"a": [fb.text("a", delay=0.2)]}.get)
    await brain.start()
    await brain.ask(1, "a")
    await brain.ask(2, "a")
    await brain.interrupt(2)
    evs = await collect(brain, until_done=2)
    t2 = [e for e in evs if e.turn_id == 2]
    assert [e.kind for e in t2] == [K.DONE] and t2[0].data["interrupted"]
    await brain.close()


async def test_error_is_terminal() -> None:
    brain = fb.FakeBrain({"x": [fb.error(BrainErrorKind.AUTH, "login"), fb.text("never")]}.get)
    await brain.start()
    await brain.ask(1, "x")
    evs = await collect(brain)
    assert [e.kind for e in evs] == [K.ERROR]
    assert evs[0].error_kind == BrainErrorKind.AUTH
    await brain.close()


async def test_steer_appends_to_running_turn() -> None:
    brain = fb.FakeBrain({"go": [fb.text("a", delay=0.05), fb.text("b", delay=0.05)]}.get,
                         steer_script=lambda t: [fb.text(f"steer:{t}")])
    await brain.start()
    await brain.ask(1, "go")
    await asyncio.sleep(0.02)
    await brain.steer(1, "also")
    evs = await collect(brain)
    texts = [e.text for e in evs if e.kind == K.TEXT]
    assert texts[-1] == "steer:also" and "a" in texts
    assert brain.steers == [(1, "also")]
    await brain.close()


async def test_reset_and_records() -> None:
    brain = fb.FakeBrain()
    await brain.start()
    await brain.reset()
    await brain.ask(3, "q")
    await collect(brain)
    assert brain.resets == 1 and brain.asks == [(3, "q")]
    await brain.warm()
    assert brain.alive and brain.warms == 1
    await brain.sleep()
    assert not brain.alive and brain.sleeps == 1
    assert await brain.handoff() is True and brain.handoffs == 1
    assert (await brain.connect_check())["auth"] == "ok" and brain.connect_checks == 1
    await brain.set_context(locked=True)
    assert brain.locked
    await brain.set_name("Jarvis")
    await brain.set_language("hi")
    assert brain.name == "Jarvis" and brain.lang == "hi"
    await brain.close()
    assert brain.closed


async def test_summarize_is_scripted_and_none_while_busy() -> None:
    brain = fb.FakeBrain()
    await brain.start()
    assert await brain.summarize("notes please", 120) == "Notes from the fake brain."
    brain.summary_reply = None
    assert await brain.summarize("again", 5) is None
    brain.summary_reply = RuntimeError("limit")
    with pytest.raises(RuntimeError):
        await brain.summarize("x", 1)
    brain.summary_reply = "late notes"
    brain._script = lambda t: [fb.text("slow reply", delay=0.3)]
    await brain.ask(1, "slow")
    await asyncio.sleep(0.05)  # a turn in flight: the notes query is refused, nothing is interrupted
    assert await brain.summarize("busy", 1) is None
    await collect(brain)
    await brain.close()
    assert [p for p, _ in brain.summaries] == ["notes please", "again", "x", "busy"]
