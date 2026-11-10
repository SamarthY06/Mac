"""The adapter contract suite (W7.1, cases C1-C9), shared by every chat channel.

A channel's test module subclasses `AdapterContract` and provides a `harness` fixture returning a
`Harness` for its adapter (the fake channel and the cli channel here; WhatsApp over FakeBridge in
test_whatsapp.py). The harness knows how to make the adapter, how to put an inbound message "on
the wire" and how many writes reached the wire; the contract checks the rules R1-R5.

    class TestMyChannel(AdapterContract):
        @pytest.fixture
        def harness(self, tmp_path): return MyHarness(tmp_path)

Cases a channel cannot have (a terminal has no other members) are listed in `Harness.lacks`
and skipped with that reason.
"""

from __future__ import annotations

import abc
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pytest

from chatd.channels.base import (
    Cap,
    ChannelAdapter,
    ChannelStatus,
    ChatChanged,
    ContextMessage,
    DestinationRefused,
    InboundMessage,
    LinkState,
    MessageRef,
)
from chatd.channels.registry import ChannelContext, SeenIds
from chatd.governor import Budget, BudgetExceeded, Governor

SECRET = "the secret plan nobody else may read"


class Clock:
    def __init__(self, t: float = 1_800_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


@dataclass
class Rig:
    """One adapter under test with everything around it."""

    adapter: ChannelAdapter
    ctx: ChannelContext
    clock: Clock
    delivered: list[InboundMessage] = field(default_factory=list)
    statuses: list[ChannelStatus] = field(default_factory=list)
    changes: list[ChatChanged] = field(default_factory=list)
    contexts: list[ContextMessage] = field(default_factory=list)

    async def start(self) -> Rig:
        async def sink(m: InboundMessage) -> None:
            self.delivered.append(m)

        async def changed(c: ChatChanged) -> None:
            self.changes.append(c)

        async def context(c: ContextMessage) -> None:
            self.contexts.append(c)

        if "on_context" in inspect.signature(self.adapter.start).parameters:
            await self.adapter.start(sink, self.statuses.append, changed, on_context=context)
        else:
            await self.adapter.start(sink, self.statuses.append, changed)
        return self


class Harness(abc.ABC):
    chat: str = "chat-a"  # allowlisted
    other: str = "chat-b"  # not allowlisted
    never: str = "chat-never"  # allowlisted AND never-listed: never wins
    lacks: ClassVar[dict[str, str]] = {}  # case -> why the channel cannot have it

    def __init__(self, root: Path) -> None:
        self.root = root

    def context(self, clock: Clock, *, total: int | None = None, dry_run: bool = False,
                seen: Path | None = None) -> ChannelContext:
        gov = Governor(None, Budget(total=total), clock=clock)
        return ChannelContext(governor=gov, chats=frozenset({self.chat, self.never}),
                              never_chats=frozenset({self.never}),
                              seen=SeenIds(seen or self.root / "seen.json"), dry_run=dry_run, clock=clock)

    @abc.abstractmethod
    async def make(self, ctx: ChannelContext) -> ChannelAdapter: ...

    @abc.abstractmethod
    async def inject(self, rig: Rig, *, chat_id: str, text: str, msg_id: str, owner: bool = True,
                     replayed: bool = False, echo_of: str = "", unsupported: str = "") -> None:
        """Put one inbound message on the wire. `echo_of`: the id of a message Mac sent (an echo)."""

    @abc.abstractmethod
    def writes(self, rig: Rig) -> int:
        """How many outbound writes reached the wire."""

    async def inject_context(self, rig: Rig, *, chat_id: str, text: str, msg_id: str, quote_of: str = "") -> None:
        """Put another member's message in a context chat on the wire (`quote_of`: it replies to
        that message). Channels without other members list "context" in `lacks`."""
        raise NotImplementedError

    async def settle(self, rig: Rig) -> None:
        """Wait until injected messages were processed (default: nothing to wait for)."""

    @abc.abstractmethod
    async def lifecycle(self, rig: Rig) -> list[LinkState]:
        """Drive the channel through its link states; return the states on_status must have seen
        (in order, starting after `start`)."""

    async def rig(self, *, total: int | None = None, dry_run: bool = False, seen: Path | None = None) -> Rig:
        clock = Clock()
        await (ctx := self.context(clock, total=total, dry_run=dry_run, seen=seen)).seen.load()
        return await Rig(await self.make(ctx), ctx, clock).start()


class AdapterContract:
    """Subclass and provide a `harness` fixture."""

    def need(self, harness: Harness, case: str) -> None:
        if case in harness.lacks:
            pytest.skip(harness.lacks[case])

    async def deliver(self, harness: Harness, rig: Rig, **kw: Any) -> list[InboundMessage]:
        before = len(rig.delivered)
        await harness.inject(rig, **kw)
        await harness.settle(rig)
        return rig.delivered[before:]

    # C1
    async def test_c1_owner_message_delivered_once(self, harness: Harness) -> None:
        rig = await harness.rig()
        got = await self.deliver(harness, rig, chat_id=harness.chat, text="find me 20 jobs", msg_id="m1")
        assert len(got) == 1
        m = got[0]
        assert (m.channel, m.chat_id, m.text) == (rig.adapter.name, harness.chat, "find me 20 jobs")
        assert m.ref.chat_id == harness.chat and m.ref.message_id
        assert len(rig.delivered) == 1
        await rig.adapter.close()

    # C2
    async def test_c2_replay_once_and_redelivered_after_a_crash_before_mark_handled(self, harness: Harness) -> None:
        self.need(harness, "replay")
        seen = harness.root / "seen-c2.json"
        rig = await harness.rig(seen=seen)
        got = await self.deliver(harness, rig, chat_id=harness.chat, text="one", msg_id="r1", replayed=True)
        assert len(got) == 1 and got[0].replayed
        await rig.adapter.mark_handled(got[0].ref)
        assert await self.deliver(harness, rig, chat_id=harness.chat, text="one", msg_id="r1", replayed=True) == []
        got2 = await self.deliver(harness, rig, chat_id=harness.chat, text="two", msg_id="r2", replayed=True)
        assert len(got2) == 1  # not marked: agentd never accepted it
        await rig.adapter.close()
        rig2 = await harness.rig(seen=seen)  # "crash" and restart with the same handled-ids store
        assert await self.deliver(harness, rig2, chat_id=harness.chat, text="one", msg_id="r1", replayed=True) == []
        again = await self.deliver(harness, rig2, chat_id=harness.chat, text="two", msg_id="r2", replayed=True)
        assert [m.ref.message_id for m in again] == [got2[0].ref.message_id]
        await rig2.adapter.close()

    # C3
    async def test_c3_non_owner_never_delivered_or_logged(self, harness: Harness,
                                                         caplog: pytest.LogCaptureFixture) -> None:
        self.need(harness, "non_owner")
        caplog.set_level(logging.DEBUG)
        rig = await harness.rig()
        assert await self.deliver(harness, rig, chat_id=harness.chat, text=SECRET, msg_id="n1", owner=False) == []
        assert SECRET not in caplog.text
        await rig.adapter.close()

    # C4
    async def test_c4_echoes_dropped(self, harness: Harness) -> None:
        rig = await harness.rig()
        ref = await rig.adapter.send_text(harness.chat, "Mac · On it")
        assert await self.deliver(harness, rig, chat_id=harness.chat, text="Mac · On it", msg_id=ref.message_id,
                                  echo_of=ref.message_id) == []
        assert await self.deliver(harness, rig, chat_id=harness.chat, text="Mac · Found 20 roles", msg_id="e2") == []
        assert len(await self.deliver(harness, rig, chat_id=harness.chat, text="thanks", msg_id="e3")) == 1
        await rig.adapter.close()

    # C4b: the shared-chat trigger lives in the runner, so a "Mac, ..." command must reach it intact
    async def test_c4b_mac_prefixed_owner_text_is_not_an_echo(self, harness: Harness) -> None:
        rig = await harness.rig()
        for i, text in enumerate(("Mac, find me 20 jobs", "mac stop", "MAC: status")):
            got = await self.deliver(harness, rig, chat_id=harness.chat, text=text, msg_id=f"p{i}")
            assert [m.text for m in got] == [text]
        await rig.adapter.close()

    # C5
    async def test_c5_other_and_never_listed_chats(self, harness: Harness) -> None:
        rig = await harness.rig()
        for chat in (harness.other, harness.never):
            assert await self.deliver(harness, rig, chat_id=chat, text="hi", msg_id=f"x-{chat}") == []
        writes, used = harness.writes(rig), dict(rig.ctx.governor.counts)
        for chat in (harness.other, harness.never):
            ref = MessageRef(chat, "m")
            for call in (rig.adapter.send_text(chat, "Mac · hi"), rig.adapter.edit_text(ref, "Mac · x"),
                         rig.adapter.react(ref, "👀"), rig.adapter.set_typing(chat, True),
                         rig.adapter.send_file(chat, harness.root / "f.txt")):
                with pytest.raises(DestinationRefused):
                    await call
        assert harness.writes(rig) == writes and rig.ctx.governor.counts == used  # refused before any I/O
        await rig.adapter.close()

    # C6
    async def test_c6_governor_on_every_op_and_exhaustion(self, harness: Harness, tmp_path: Path) -> None:
        rig = await harness.rig(total=4)
        gov, a = rig.ctx.governor, rig.adapter
        f = tmp_path / "a.txt"
        f.write_text("x")
        ref = await a.send_text(harness.chat, "Mac · one")
        assert gov.counts["send"] == 1
        if Cap.EDIT in a.caps:
            assert await a.edit_text(ref, "Mac · two")
            assert gov.counts["edit"] == 1
        if Cap.REACT in a.caps:
            await a.react(ref, "👀")
            assert gov.counts["react"] == 1
        if Cap.TYPING in a.caps:
            await a.set_typing(harness.chat, True)
            assert gov.counts["typing"] == 1
        await a.send_file(harness.chat, f)
        while not gov.disabled:
            await a.send_text(harness.chat, "Mac · filler")
        writes = harness.writes(rig)
        with pytest.raises(BudgetExceeded):
            await a.send_text(harness.chat, "Mac · one too many")
        with pytest.raises(BudgetExceeded):
            await a.send_file(harness.chat, f)
        assert harness.writes(rig) == writes
        await a.close()

    # C7
    async def test_c7_edit_past_the_window_returns_false(self, harness: Harness) -> None:
        self.need(harness, "edit_window")
        rig = await harness.rig()
        a = rig.adapter
        ref = await a.send_text(harness.chat, "Mac · On it")
        assert Cap.EDIT in a.caps and a.limits.edit_window_s > 0
        rig.clock.advance(a.limits.edit_window_s - 1)
        assert await a.edit_text(ref, "Mac · still")
        rig.clock.advance(2)
        writes = harness.writes(rig)
        assert await a.edit_text(ref, "Mac · too late") is False  # the runner sends a new message
        assert harness.writes(rig) == writes
        await a.close()

    # C8
    async def test_c8_status_transitions(self, harness: Harness) -> None:
        rig = await harness.rig()
        assert rig.statuses and rig.adapter.status() == rig.statuses[-1]
        assert rig.statuses[-1].state in (LinkState.CONNECTED, LinkState.UNLINKED)
        start = len(rig.statuses)
        expected = await harness.lifecycle(rig)
        assert [x.state for x in rig.statuses[start:]] == expected
        assert rig.adapter.status() == rig.statuses[-1]
        assert all(isinstance(x, ChannelStatus) for x in rig.statuses)
        await rig.adapter.close()

    # C9
    async def test_c9_dry_run_writes_nothing(self, harness: Harness, tmp_path: Path) -> None:
        rig = await harness.rig(dry_run=True)
        a = rig.adapter
        f = tmp_path / "a.txt"
        f.write_text("x")
        ref = await a.send_text(harness.chat, "Mac · hi")
        assert ref.message_id
        await a.edit_text(ref, "Mac · edited")
        await a.react(ref, "👀")
        await a.set_typing(harness.chat, True)
        await a.send_file(harness.chat, f)
        assert harness.writes(rig) == 0
        await a.close()

    # unsupported media (W5.5 voice note): flagged, never downloaded
    async def test_unsupported_media_flagged_not_downloaded(self, harness: Harness) -> None:
        self.need(harness, "unsupported")
        rig = await harness.rig()
        got = await self.deliver(harness, rig, chat_id=harness.chat, text="", msg_id="v1", unsupported="voice")
        assert len(got) == 1 and got[0].unsupported == "voice" and got[0].attachments == ()
        await rig.adapter.close()

    # C10 (v3.5, W10.1): context is data. A member's "Mac, stop", or a "yes" quoting Mac's prompt,
    # reaches only on_context, never the sink that leads to commands, replies, steers and asks.
    async def test_c10_context_never_reaches_the_trigger_sink(self, harness: Harness) -> None:
        self.need(harness, "context")
        rig = await harness.rig()
        prompt = await rig.adapter.send_text(harness.chat, "Mac · Send it? Reply yes to this message.")
        try:
            await harness.inject_context(rig, chat_id=harness.chat, text="Mac, stop", msg_id="c1")
        except NotImplementedError:
            await rig.adapter.close()
            pytest.skip("this harness cannot put context on the wire yet")
        await harness.inject_context(rig, chat_id=harness.chat, text="yes", msg_id="c2", quote_of=prompt.message_id)
        await harness.settle(rig)
        assert rig.delivered == []
        assert [c.text for c in rig.contexts] == ["Mac, stop", "yes"]
        assert all(not c.from_owner and c.chat_id == harness.chat for c in rig.contexts)
        await rig.adapter.close()
