"""Desktop lock (W3.4, F3): bounded waits, turn token, idle release, release while confirming,
teardown, holder visibility, and the polled handle the brain's DesktopGuard uses."""

from __future__ import annotations

import asyncio

from agentd_kit import CHAT, CHAT_TOKEN, confirm_step, slow

from agentd.desktop import DesktopLock, TurnToken
from macd.brain import fake_brain as fb
from macd.brain.hooks import DesktopGuard

V = "voice:local"


def tok(conv: str, n: int = 1, what: str = "", label: str = "") -> TurnToken:
    stop = {"v": False}
    t = TurnToken(conv, n, label or conv.split(":")[0], what, lambda: stop["v"])
    t.stop = stop  # type: ignore[attr-defined]
    return t


async def test_bounded_wait_times_out():
    lock = DesktopLock()
    a, b = tok(CHAT, what="find jobs"), tok(V, what="open my calendar")
    assert await lock.acquire(a, 1)
    t0 = asyncio.get_running_loop().time()
    assert not await lock.acquire(b, 0.05)
    assert 0.04 < asyncio.get_running_loop().time() - t0 < 0.5
    assert lock.status().holder == "whatsapp" and lock.status().what == "find jobs"


async def test_waiter_gets_lock_on_release_in_order():
    lock = DesktopLock()
    a, b, c = tok(CHAT), tok(V), tok("cli:1")
    await lock.acquire(a, 1)
    fb_ = asyncio.create_task(lock.acquire(b, 1))
    fc = asyncio.create_task(lock.acquire(c, 1))
    await asyncio.sleep(0.01)
    lock.release(a)
    assert await fb_ and lock.holder is b
    lock.release(b)
    assert await fc


async def test_stopped_turns_waiter_never_acquires():
    lock = DesktopLock()
    a, b = tok(CHAT), tok(V)
    await lock.acquire(a, 1)
    wait = asyncio.create_task(lock.acquire(b, 1))
    await asyncio.sleep(0.01)
    b.stop["v"] = True
    lock.release(a)
    assert await wait is False and lock.holder is None


async def test_granted_then_cancelled_is_released():
    lock = DesktopLock()
    a, b = tok(CHAT), tok(V)
    await lock.acquire(a, 1)
    wait = asyncio.create_task(lock.acquire(b, 1))
    await asyncio.sleep(0.01)
    lock._free()  # grant to b ...
    b.stop["v"] = True  # ... whose turn stopped before it resumed
    assert await wait is False and lock.holder is None


async def test_cancelled_waiting_task_leaks_nothing():
    lock = DesktopLock()
    a, b = tok(CHAT), tok(V)
    await lock.acquire(a, 1)
    wait = asyncio.create_task(lock.acquire(b, 5))
    await asyncio.sleep(0.01)
    wait.cancel()
    await asyncio.gather(wait, return_exceptions=True)
    lock.release(a)
    assert lock.holder is None and not lock._waiters


async def test_idle_release_and_touch():
    lock = DesktopLock(idle_release_s=0.1)
    a = tok(CHAT)
    await lock.acquire(a, 1)
    await asyncio.sleep(0.07)
    lock.touch(a)
    await asyncio.sleep(0.07)
    assert lock.holder is a
    await asyncio.sleep(0.1)
    assert lock.holder is None


async def test_release_conv_and_cancel_turn():
    lock = DesktopLock()
    a, b = tok(CHAT), tok(V)
    await lock.acquire(a, 1)
    wait = asyncio.create_task(lock.acquire(b, 5))
    await asyncio.sleep(0.01)
    lock.cancel_turn(V)  # teardown of the waiter's conversation
    assert await wait is False
    lock.release_conv(CHAT)  # confirmation opened in chat
    assert lock.holder is None


async def test_dead_holder_is_dropped():
    lock = DesktopLock()
    a, b = tok(CHAT), tok(V)
    await lock.acquire(a, 1)
    a.stop["v"] = True
    assert await lock.acquire(b, 0.01)


async def test_handles_with_the_brains_desktop_guard():
    lock = DesktopLock()
    chat, voice = lock.handle(CHAT, "chat", lambda: "find jobs"), lock.handle(V, "voice", lambda: "calendar")
    g_chat = DesktopGuard([chat], wait_s=1, poll_s=0.01)
    g_voice = DesktopGuard([voice], wait_s=0.1, poll_s=0.01)
    assert await g_chat.acquire(1, "t1", lambda: False) == "ok"
    assert lock.status().holder == "chat" and lock.status().what == "find jobs"
    assert await g_voice.acquire(1, "t2", lambda: False) == "busy"
    lock.release_conv(CHAT)  # agentd frees it while the chat confirmation is open
    assert await g_voice.acquire(1, "t2", lambda: False) == "ok"
    g_voice.release()
    assert lock.holder is None


async def test_service_releases_on_confirm_and_on_turn_end(agent):
    step = confirm_step()
    step.delay = 0.2
    agent.scripts[CHAT] = lambda t: [fb.say("browsing"), step, *slow(n=3, delay=0.01)]
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "email"})
    await c.event("say")
    conv = agent.svc.convs[CHAT]
    assert conv.desktop.try_acquire()
    st = (await c.call({"op": "status", "conv": CHAT}))["data"]
    assert st["desktop"] == {"holder": "chat", "what": "email"}
    ev = await c.event("confirm")
    assert agent.svc.desktop.holder is None  # released while confirming
    assert conv.desktop.try_acquire()  # the next desktop tool re-acquires
    await c.call({"op": "reply", "conv": CHAT, "text": "yes"})
    assert ev["question_id"]
    await c.event("done")
    assert agent.svc.desktop.holder is None  # teardown


async def test_voice_answers_while_chat_turn_runs(agent):
    agent.scripts[CHAT] = lambda t: slow(n=200, delay=0.01)
    chat = await agent.client(CHAT_TOKEN, [CHAT])
    await chat.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "long"})
    await chat.event("say")
    voice = await agent.client()
    await voice.call({"op": "ask", "conv": V, "turn": 1, "text": "what time is it"})
    assert (await voice.event("text", timeout=1))["text"] == "Okay."
