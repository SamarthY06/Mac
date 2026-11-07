from __future__ import annotations

import asyncio

from macd.brain.gate import ConfirmationGate


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


async def approve(gate: ConfirmationGate, turn: int, action: str = "Bash|git_push|git push") -> None:
    task = asyncio.create_task(gate.request(turn, "x", timeout=5, action=action))
    await asyncio.sleep(0)
    gate.resolve(True)
    assert await task is True


async def test_approve_mints_bound_token() -> None:
    gate = ConfirmationGate(clock=Clock())
    task = asyncio.create_task(gate.request(7, "Send Bob hi", timeout=5, action="Bash|osascript_send|messages"))
    await asyncio.sleep(0)
    assert gate.pending == (7, "Send Bob hi")
    gate.resolve(True)
    assert await task is True
    assert gate.pending is None
    assert gate.has_token(7)
    assert gate.consume(7, "Bash|osascript_send|messages")
    assert gate.consume(7, "Bash|osascript_send|messages")  # idempotent retry of the same action
    assert not gate.consume(7, "Bash|http_upload|attacker.example")  # a different action: never
    assert not gate.has_token(8)


async def test_unbound_mint_is_refused() -> None:
    gate = ConfirmationGate()
    assert await gate.request(1, "x", timeout=1, action="") is False  # no denied action → no token
    assert gate.pending is None
    assert not gate.has_token(1)
    assert not gate.consume(1, "Bash|git_push|git push origin")


async def test_decline_no_token() -> None:
    gate = ConfirmationGate()
    task = asyncio.create_task(gate.request(1, "x", timeout=5, action="a"))
    await asyncio.sleep(0)
    gate.resolve(False)
    assert await task is False
    assert not gate.has_token(1)


async def test_timeout_returns_false() -> None:
    gate = ConfirmationGate()
    assert await gate.request(1, "x", timeout=0.05, action="a") is False
    assert gate.pending is None
    assert not gate.has_token(1)


async def test_token_ttl() -> None:
    clock = Clock()
    gate = ConfirmationGate(clock=clock)
    await approve(gate, 3, "a")
    clock.t += 119
    assert gate.has_token(3)
    clock.t += 2
    assert not gate.has_token(3)
    assert not gate.consume(3, "a")


async def test_token_single_turn() -> None:
    gate = ConfirmationGate()
    await approve(gate, 3, "a")
    assert not gate.consume(4, "a")
    assert gate.consume(3, "a")


async def test_revoke_all_cancels_pending_and_token() -> None:
    gate = ConfirmationGate()
    await approve(gate, 1, "a")
    task = asyncio.create_task(gate.request(1, "b", timeout=5, action="b"))
    await asyncio.sleep(0)
    gate.revoke_all()
    assert await task is False
    assert not gate.has_token(1)


async def test_only_one_pending() -> None:
    gate = ConfirmationGate()
    first = asyncio.create_task(gate.request(1, "a", timeout=5, action="a"))
    await asyncio.sleep(0)
    assert await gate.request(1, "b", timeout=5, action="b") is False
    assert gate.pending == (1, "a")
    gate.resolve(True)
    assert await first is True


async def test_resolve_without_pending_is_noop() -> None:
    gate = ConfirmationGate()
    gate.resolve(True)
    assert not gate.has_token(1)


async def test_new_approval_replaces_token() -> None:
    gate = ConfirmationGate()
    await approve(gate, 1, "a")
    await approve(gate, 2, "b")
    assert not gate.has_token(1)
    assert gate.has_token(2, "b") and not gate.has_token(2, "a")
