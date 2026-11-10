"""The adapter contract (C1-C9) for the fake and cli channels, plus the registry, the handled-ids
store and the persisted governor (F5)."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import ClassVar

import pytest
from contract import AdapterContract, Clock, Harness, Rig

from chatd.channels import registry
from chatd.channels.base import ChannelAdapter, LinkState
from chatd.channels.cli import CliChannel
from chatd.channels.fake import FakeChannel
from chatd.channels.registry import ChannelContext, SeenIds
from chatd.governor import Budget, BudgetExceeded, Governor


class FakeHarness(Harness):
    async def make(self, ctx: ChannelContext) -> ChannelAdapter:
        return FakeChannel({}, ctx)

    async def inject(self, rig: Rig, *, chat_id: str, text: str, msg_id: str, owner: bool = True,
                     replayed: bool = False, echo_of: str = "", unsupported: str = "") -> None:
        a = rig.adapter
        assert isinstance(a, FakeChannel)
        await a.inject(text, chat_id=chat_id, msg_id=echo_of or msg_id, from_owner=owner, replayed=replayed,
                       unsupported=unsupported)  # type: ignore[arg-type]

    def writes(self, rig: Rig) -> int:
        assert isinstance(rig.adapter, FakeChannel)
        return len(rig.adapter.wire)

    async def lifecycle(self, rig: Rig) -> list[LinkState]:
        a = rig.adapter
        assert isinstance(a, FakeChannel)
        await a.unlink()
        await a.link()
        assert a.status().qr
        await a.complete_pairing()
        await a.go_offline()
        return [LinkState.UNLINKED, LinkState.PAIRING, LinkState.CONNECTED, LinkState.OFFLINE]


class CliHarness(Harness):
    lacks: ClassVar[dict[str, str]] = {"non_owner": "a terminal has no other members",
             "replay": "a terminal has no offline history",
             "unsupported": "a terminal only carries text"}

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.io: dict[int, tuple[asyncio.StreamReader, io.StringIO]] = {}

    async def make(self, ctx: ChannelContext) -> ChannelAdapter:
        reader, out = asyncio.StreamReader(), io.StringIO()
        a = CliChannel({"chat_id": self.chat, "input": reader, "output": out}, ctx)
        self.io[id(a)] = (reader, out)
        return a

    async def inject(self, rig: Rig, *, chat_id: str, text: str, msg_id: str, owner: bool = True,
                     replayed: bool = False, echo_of: str = "", unsupported: str = "") -> None:
        if chat_id != self.chat:
            return  # a terminal only ever types into its own chat
        self.io[id(rig.adapter)][0].feed_data((text + "\n").encode())

    async def settle(self, rig: Rig) -> None:
        for _ in range(10):
            await asyncio.sleep(0)

    def writes(self, rig: Rig) -> int:
        return len(self.io[id(rig.adapter)][1].getvalue().splitlines())

    async def lifecycle(self, rig: Rig) -> list[LinkState]:
        await rig.adapter.unlink()
        await rig.adapter.link()
        self.io[id(rig.adapter)][0].feed_eof()
        await self.settle(rig)
        return [LinkState.UNLINKED, LinkState.CONNECTED, LinkState.OFFLINE]


class TestFakeChannelContract(AdapterContract):
    @pytest.fixture
    def harness(self, tmp_path: Path) -> Harness:
        return FakeHarness(tmp_path)


class TestCliChannelContract(AdapterContract):
    @pytest.fixture
    def harness(self, tmp_path: Path) -> Harness:
        return CliHarness(tmp_path)


async def test_never_list_wins_over_the_allowlist_on_the_cli(tmp_path: Path) -> None:
    clock = Clock()
    ctx = ChannelContext(governor=Governor(clock=clock), chats=frozenset({"local"}), never_chats=frozenset({"local"}),
                         clock=clock)
    reader, out = asyncio.StreamReader(), io.StringIO()
    got: list[object] = []

    async def sink(m: object) -> None:
        got.append(m)

    async def changed(c: object) -> None: ...

    a = CliChannel({"input": reader, "output": out}, ctx)
    await a.start(sink, lambda s: None, changed)
    reader.feed_data(b"hello\n")
    for _ in range(10):
        await asyncio.sleep(0)
    assert got == [] and out.getvalue() == ""
    await a.close()


async def test_registry_creates_known_channels(tmp_path: Path) -> None:
    ctx = ChannelContext(governor=Governor(), chats=frozenset({"x"}))
    with pytest.raises(ValueError):
        registry.create("fake", {}, ctx)  # test adapters are not registered in production (AG25)
    test_ctx = ChannelContext(governor=Governor(), chats=frozenset({"x"}), test=True)
    assert isinstance(registry.create("fake", {}, test_ctx), FakeChannel)
    assert isinstance(registry.create("cli", {"input": asyncio.StreamReader()}, ctx), CliChannel)
    with pytest.raises(ValueError):
        registry.create("telegram", {}, ctx)
    assert set(registry.ADAPTERS) == {"cli", "whatsapp"}


async def test_seen_ids_persist_and_cap(tmp_path: Path) -> None:
    p = tmp_path / "seen.json"
    seen = await SeenIds(p, maxlen=3).load()
    for i in range(5):
        await seen.add(f"c:{i}")
    again = await SeenIds(p, maxlen=3).load()
    assert "c:4" in again and "c:2" in again and "c:1" not in again


async def test_governor_limits_and_persistence(tmp_path: Path) -> None:
    clock = Clock()
    p = tmp_path / "governor.json"
    gov = await Governor(p, Budget(sends_per_min=3, edits_per_min=2, new_per_hour=4), clock=clock).load()
    for _ in range(3):
        await gov.charge("send")
    with pytest.raises(BudgetExceeded):
        await gov.charge("send")  # 20/min style cap
    await gov.charge("edit")
    await gov.charge("edit")
    with pytest.raises(BudgetExceeded):
        await gov.charge("edit")
    clock.advance(61)
    await gov.charge("send")
    with pytest.raises(BudgetExceeded):
        await gov.charge("send")  # 4 per hour
    # F5: a restart (or a crash loop) does not reset the counters
    gov2 = await Governor(p, Budget(sends_per_min=3, edits_per_min=2, new_per_hour=4), clock=clock).load()
    assert not gov2.allowed("send")
    clock.advance(3601)
    assert gov2.allowed("send")


async def test_governor_typing_counts_a_tenth() -> None:
    gov = Governor(None, Budget(sends_per_min=1))
    for _ in range(9):
        await gov.charge("typing")
    with pytest.raises(BudgetExceeded):
        await gov.charge("react")  # 0.9 + 1 > 1
    await gov.charge("typing")  # 1.0


async def test_governor_total_disables_everything_but_typing() -> None:
    gov = Governor(None, Budget(total=2))
    await gov.charge("send")
    await gov.charge("react")
    assert gov.disabled
    for kind in ("send", "edit", "react"):
        with pytest.raises(BudgetExceeded):
            await gov.charge(kind)  # type: ignore[arg-type]
    await gov.charge("typing")


def test_config_keeps_process_keys_only(caplog: pytest.LogCaptureFixture) -> None:
    """W9.9: the per-chat keys moved to agentd's profiles; chatd drops them, naming the keys only."""
    process_config = registry.process_config
    cfg = {"channels": {"whatsapp": {"enabled": True, "never_chats": ["777@g.us"], "chats": ["123@g.us"],
                                     "allow_shared_chat": True, "trigger": {"123@g.us": "prefix"},
                                     "dry_run": False}}}
    with caplog.at_level("WARNING"):
        out = process_config(cfg)
    assert out["channels"]["whatsapp"] == {"enabled": True, "never_chats": ["777@g.us"], "dry_run": False}
    assert "whatsapp.chats" in caplog.text and "123@g.us" not in caplog.text
    assert cfg["channels"]["whatsapp"]["chats"] == ["123@g.us"]  # the input is not changed
    caplog.clear()
    assert process_config({"channels": {"cli": {"enabled": True}}}) == {"channels": {"cli": {"enabled": True}}}
    assert caplog.text == ""


async def test_set_allowed_is_recorded_not_applied_by_test_adapters(tmp_path: Path) -> None:
    """The fake and the terminal keep their fixed chats; production starts empty (W9.8 step 6)."""
    assert ChannelContext(governor=Governor(None)).chats == frozenset()
    ctx = ChannelContext(governor=Governor(None), chats=frozenset({"a"}))
    fake = FakeChannel({}, ctx)
    await fake.set_allowed(frozenset({"b"}))
    assert fake.allowed_sets == [{"b"}] and ctx.chats == {"a"}
    cli = CliChannel({}, ChannelContext(governor=Governor(None), chats=frozenset({"local"})))
    await cli.set_allowed(frozenset())
    assert cli.ctx.chats == {"local"}
