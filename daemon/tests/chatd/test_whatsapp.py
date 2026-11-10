"""The WhatsApp channel over FakeBridge: the adapter contract (C1-C9) plus the WhatsApp cases (W7.1):
rename, collision, Timetable, set_allow, untrusted device, chat_changed pause signal, voice note,
the bridge's exit codes, timeouts, out-of-order answers and the outbound arguments.

FakeBridge (fake_bridge.py) is a real child process speaking the bridge protocol; nothing here
touches WhatsApp or the real store under ~/.mac-agent (every path is a temp dir).
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import sys
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from contract import SECRET, AdapterContract, Clock, Harness, Rig

from chatd.channels import registry
from chatd.channels import whatsapp as wa
from chatd.channels.base import (
    ChannelAdapter,
    ChatChanged,
    DestinationRefused,
    LinkState,
    MessageRef,
)
from chatd.channels.registry import ChannelContext
from chatd.channels.whatsapp import BridgeError, WhatsAppChannel
from chatd.governor import BudgetExceeded

FAKE = Path(__file__).with_name("fake_bridge.py")
PREP = "120363000000000001@g.us"  # "Interview Prep"
OTHER = "120363000000000002@g.us"  # another group (also called "Interview Prep" in the collision case)
TIMETABLE = "120363000000000009@g.us"  # never-listed


def control(fake_dir: Path, **cmd: Any) -> dict[str, Any]:
    """One command to a FakeBridge's control socket. Blocking: async tests use `acontrol`.
    A cold bridge may not have opened its socket yet, so connecting is retried for up to 20 s."""
    deadline = time.monotonic() + 20
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(10)
        while True:
            try:
                s.connect(str(fake_dir / "c.sock"))
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.02)
        s.sendall(json.dumps(cmd).encode() + b"\n")
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf) if buf else {}  # "exit" answers by exiting


async def acontrol(fake_dir: Path, **cmd: Any) -> dict[str, Any]:
    """`control` off the event loop, so the adapter keeps reading the bridge while we wait."""
    return await asyncio.to_thread(control, fake_dir, **cmd)


async def until(pred: Callable[[], bool], timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not pred():
        if loop.time() > end:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


def starts(fake_dir: Path) -> int:
    p = fake_dir / "starts"
    return len(p.read_text().splitlines()) if p.exists() else 0


class WaHarness(Harness):
    chat, other, never = PREP, OTHER, TIMETABLE

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.n = 0
        self.dirs: dict[int, Path] = {}
        self.made: list[WhatsAppChannel] = []
        self.cfg: dict[str, Any] = {}  # extra adapter config for the next make()
        self.fake: dict[str, Any] = {}  # fake.json for the next make()

    def config(self, fake_dir: Path) -> dict[str, Any]:
        return {"command": [sys.executable, str(FAKE), "--fake-dir", str(fake_dir)],
                "store": str(self.root / "wa" / "device.db"), "media_dir": str(self.root / "wa" / "media"),
                "log_path": str(self.root / "wa-bridge.log"), "request_timeout_s": 15.0, "connect_wait_s": 10.0,
                "backoff": [0.01], "typing": True, **self.cfg}

    async def make(self, ctx: ChannelContext) -> ChannelAdapter:
        self.n += 1
        d = self.root / f"f{self.n}"
        d.mkdir()
        if self.fake:
            (d / "fake.json").write_text(json.dumps(self.fake))
        a = WhatsAppChannel(self.config(d), ctx)
        self.dirs[id(a)] = d
        self.made.append(a)
        return a

    def dir(self, rig: Rig) -> Path:
        return self.dirs[id(rig.adapter)]

    async def inject(self, rig: Rig, *, chat_id: str, text: str, msg_id: str, owner: bool = True,
                     replayed: bool = False, echo_of: str = "", unsupported: str = "", **extra: Any) -> None:
        cmd: dict[str, Any] = {"chat": chat_id, "msg": echo_of or msg_id, "text": text, "from_me": owner,
                               "ts": rig.clock(), "now": rig.clock(), "replayed": replayed, **extra}
        if unsupported:
            cmd["media"] = {"kind": unsupported}
        await acontrol(self.dir(rig), cmd="inbound", **cmd)

    async def inject_context(self, rig: Rig, *, chat_id: str, text: str, msg_id: str, quote_of: str = "") -> None:
        a = adapter(rig)
        if chat_id not in a.context_ids:
            await a.set_allowed(rig.ctx.chats, a.context_ids | {chat_id})
        extra: dict[str, Any] = {"quote_of": quote_of} if quote_of else {}
        await self.inject(rig, chat_id=chat_id, text=text, msg_id=msg_id, owner=False,
                          sender="222@s.whatsapp.net", sender_name="Arjun", **extra)

    def stats(self, rig: Rig) -> dict[str, Any]:
        return control(self.dir(rig), cmd="stats")

    def writes(self, rig: Rig) -> int:
        return len(self.stats(rig)["writes"])

    async def settle(self, rig: Rig) -> None:
        a = rig.adapter
        assert isinstance(a, WhatsAppChannel)
        await a.bridge_status()  # every line the bridge wrote before its answer has been read
        await a._events.join()

    async def lifecycle(self, rig: Rig) -> list[LinkState]:
        a, d = rig.adapter, self.dir(rig)
        await acontrol(d, cmd="offline")
        await until(lambda: a.status().state is LinkState.OFFLINE)
        await acontrol(d, cmd="reconnect")
        await until(lambda: a.status().state is LinkState.CONNECTED)
        await a.unlink()
        await until(lambda: a.status().state is LinkState.UNLINKED)
        await a.link()
        await until(lambda: a.status().state is LinkState.PAIRING)
        assert a.status().qr == "fake-qr-1"
        await acontrol(d, cmd="scan")
        await until(lambda: a.status().state is LinkState.CONNECTED)
        return [LinkState.OFFLINE, LinkState.CONNECTED, LinkState.UNLINKED, LinkState.PAIRING, LinkState.CONNECTED]

    async def rig(self, *, total: int | None = None, dry_run: bool = False, seen: Path | None = None) -> Rig:
        rig = await super().rig(total=total, dry_run=dry_run, seen=seen)
        await connected(rig)
        return rig

    async def rig_with(self, *, test: bool = False, total: int | None = None, dry_run: bool = False,
                       cfg: dict[str, Any] | None = None, fake: dict[str, Any] | None = None,
                       chats: frozenset[str] | None = None, clock_t: float | None = None,
                       wait: bool = True) -> Rig:
        """`wait`: return only once the bridge is connected (a cold bridge under load is slow)."""
        self.cfg, self.fake = cfg or {}, fake or {}
        clock = Clock() if clock_t is None else Clock(clock_t)
        ctx = self.context(clock, total=total, dry_run=dry_run)
        ctx.test = test
        if chats is not None:
            ctx.chats = chats
        await ctx.seen.load()
        rig = await Rig(await self.make(ctx), ctx, clock).start()
        self.cfg, self.fake = {}, {}
        if wait:
            await connected(rig)
        return rig


async def connected(rig: Rig) -> None:
    await until(lambda: rig.adapter.status().state is LinkState.CONNECTED, timeout=30)


@pytest.fixture
async def harness(short_root: Path) -> AsyncIterator[WaHarness]:
    h = WaHarness(short_root)
    yield h
    for a in h.made:  # a failed test must not leave a bridge running
        await a.close()


class TestWhatsAppContract(AdapterContract):
    pass


def adapter(rig: Rig) -> WhatsAppChannel:
    assert isinstance(rig.adapter, WhatsAppChannel)
    return rig.adapter


async def deliver(h: WaHarness, rig: Rig, **kw: Any) -> list[Any]:
    before = len(rig.delivered)
    kw.setdefault("text", "")
    await h.inject(rig, **kw)
    await h.settle(rig)
    return rig.delivered[before:]


async def emit(h: WaHarness, rig: Rig, event: dict[str, Any]) -> None:
    """A raw event, past the bridge's own filter (tests the adapter's re-checks)."""
    await acontrol(h.dir(rig), cmd="emit", event=event)
    await h.settle(rig)


def raw_message(rig: Rig, **kw: Any) -> dict[str, Any]:
    return {"ev": "message", "chat": PREP, "msg": "raw-1", "sender_device": 0, "from_me": True, "text": "hi",
            "sent_at": rig.clock(), "replayed": False, "attachments": [], "unsupported": "", **kw}


# -- registry and process flags ---------------------------------------------------------------------------
async def test_registered_and_flags_from_config(harness: WaHarness) -> None:
    assert registry.ADAPTERS["whatsapp"] == "chatd.channels.whatsapp"
    rig = await harness.rig_with(test=True, dry_run=True, cfg={"budget": 9, "max_per_min": 5,
                                                               "trusted_devices": [0, 3]})
    argv = harness.stats(rig)["argv"]
    assert f"--allow={PREP}" in argv  # the never-listed chat is never on --allow
    assert f"--deny={TIMETABLE}" in argv
    assert {"--trusted-devices=0,3", "--max-per-min=5", "--budget=9", "--dry-run", "--test"} <= set(argv)
    assert harness.stats(rig)["allow"] == [PREP] and harness.stats(rig)["deny"] == [TIMETABLE]
    await rig.adapter.close()


def test_test_mode_needs_a_budget(harness: WaHarness) -> None:
    ctx = harness.context(Clock())
    ctx.test = True
    with pytest.raises(ValueError):
        WhatsAppChannel({}, ctx)


# -- Timetable, rename, collision, set_allow ------------------------------------------------------------------
async def test_timetable_never_read_or_written(harness: WaHarness) -> None:
    rig = await harness.rig()
    a = adapter(rig)
    assert await deliver(harness, rig, chat_id=TIMETABLE, text="timetable", msg_id="t1") == []
    assert harness.stats(rig)["dropped"] == {"denied": 1}  # the bridge dropped it
    await emit(harness, rig, raw_message(rig, chat=TIMETABLE, msg="t2"))  # and the adapter re-checks
    await emit(harness, rig, {"ev": "chat_changed", "chat": TIMETABLE, "subject": "Timetable", "members": 3,
                              "owner_only": False})
    assert rig.delivered == [] and rig.changes == []
    with pytest.raises(DestinationRefused):
        await a.send_text(TIMETABLE, "Mac · hi")
    with pytest.raises(DestinationRefused):
        await a.set_allow([PREP, TIMETABLE])
    with pytest.raises(BridgeError) as e:  # the bridge refuses it too, and nothing changes
        await a.request("set_allow", allow=[PREP, TIMETABLE])
    assert e.value.code == "denied"
    with pytest.raises(BridgeError) as e:
        await a.request("send_text", chat=TIMETABLE, text="Mac · hi")
    assert e.value.code == "denied"
    assert harness.stats(rig)["allow"] == [PREP] and harness.writes(rig) == 0
    assert TIMETABLE in rig.ctx.never_chats
    await a.close()


async def test_rename_changes_nothing(harness: WaHarness) -> None:
    rig = await harness.rig()
    await acontrol(harness.dir(rig), cmd="chat_changed", chat=PREP, subject="Job hunt", members=1, owner_only=True)
    await harness.settle(rig)
    assert rig.changes == [ChatChanged(PREP, "Job hunt", owner_only=True, extra={"members": 1})]
    got = await deliver(harness, rig, chat_id=PREP, text="find me 20 jobs", msg_id="m1")
    assert [m.text for m in got] == ["find me 20 jobs"]  # JIDs are compared, never subjects
    await adapter(rig).send_text(PREP, "Mac · On it")
    await rig.adapter.close()


async def test_subject_collision_only_the_allowlisted_jid(harness: WaHarness) -> None:
    chats = [{"id": PREP, "subject": "Interview Prep", "members": 1, "owner_only": True, "is_self": False},
             {"id": OTHER, "subject": "Interview Prep", "members": 4, "owner_only": False, "is_self": False}]
    rig = await harness.rig_with(fake={"chats": chats})
    a = adapter(rig)
    assert [c["id"] for c in await a.list_chats()] == [PREP, OTHER]
    assert await deliver(harness, rig, chat_id=OTHER, text="hi", msg_id="o1") == []
    await emit(harness, rig, raw_message(rig, chat=OTHER, msg="o2"))
    assert rig.delivered == []
    with pytest.raises(DestinationRefused):
        await a.send_text(OTHER, "Mac · hi")
    assert len(await deliver(harness, rig, chat_id=PREP, text="hi", msg_id="p1")) == 1
    await a.close()


async def test_set_allow_refused_under_test(harness: WaHarness) -> None:
    rig = await harness.rig_with(test=True, cfg={"budget": 5})
    a = adapter(rig)
    with pytest.raises(BridgeError) as e:
        await a.set_allow([PREP, OTHER])
    assert e.value.code == "test_mode"
    with pytest.raises(BridgeError) as e:  # and the bridge refuses it on its own
        await a.request("set_allow", allow=[PREP, OTHER])
    assert e.value.code == "test_mode"
    for call in (a.link, a.unlink):
        with pytest.raises(BridgeError) as e:
            await call()
        assert e.value.code == "test_mode"
    assert harness.stats(rig)["allow"] == [PREP] and rig.ctx.chats == {PREP, TIMETABLE}
    await a.close()


async def test_set_allow_outside_test(harness: WaHarness) -> None:
    rig = await harness.rig()
    a = adapter(rig)
    await a.set_allow([PREP, OTHER])
    assert harness.stats(rig)["allow"] == [PREP, OTHER] and rig.ctx.allowed(OTHER)
    await a.send_text(OTHER, "Mac · hi")
    assert len(await deliver(harness, rig, chat_id=OTHER, text="hi", msg_id="o1")) == 1
    await a.close()


# -- who can drive the Mac -----------------------------------------------------------------------------------
async def test_untrusted_and_own_devices_dropped(harness: WaHarness) -> None:
    rig = await harness.rig_with(cfg={"trusted_devices": [0]})
    assert await deliver(harness, rig, chat_id=PREP, text="hi", msg_id="d1", sender_device=3) == []
    assert harness.stats(rig)["dropped"] == {"untrusted_device": 1}
    await emit(harness, rig, raw_message(rig, msg="d2", sender_device=3))  # past the bridge: re-checked
    await emit(harness, rig, raw_message(rig, msg="d3", sender_device=7))  # the bridge's own device (echo)
    await emit(harness, rig, raw_message(rig, msg="d4", sender_device=None))
    await emit(harness, rig, raw_message(rig, msg="d5", sent_at=rig.clock() - 8 * 86400))  # too old
    await emit(harness, rig, raw_message(rig, msg="d6", text="Mac · Found 20 roles"))  # our prefix
    assert rig.delivered == []
    await emit(harness, rig, raw_message(rig, msg="d7"))
    assert [m.ref.message_id for m in rig.delivered] == ["d7"]
    await rig.adapter.close()


async def test_trusted_devices_from_config(harness: WaHarness) -> None:
    rig = await harness.rig_with(cfg={"trusted_devices": [0, 3]})
    got = await deliver(harness, rig, chat_id=PREP, text="from my laptop", msg_id="d1", sender_device=3)
    assert len(got) == 1
    await rig.adapter.close()


async def test_chat_changed_to_shared_is_the_pause_signal(harness: WaHarness) -> None:
    rig = await harness.rig()
    await acontrol(harness.dir(rig), cmd="chat_changed", chat=PREP, subject="Interview Prep", members=2, owner_only=False)
    await harness.settle(rig)
    assert rig.changes == [ChatChanged(PREP, "Interview Prep", owner_only=False, extra={"members": 2})]
    await acontrol(harness.dir(rig), cmd="chat_changed", chat=OTHER, subject="x", members=2, owner_only=False)
    await harness.settle(rig)
    assert len(rig.changes) == 1  # other chats never reach chatd
    await rig.adapter.close()


# -- inbound mapping -----------------------------------------------------------------------------------------
async def test_voice_note_unsupported_and_never_downloaded(harness: WaHarness) -> None:
    rig = await harness.rig()
    media = harness.root / "wa" / "media"
    for kind in ("voice", "audio", "video", "sticker"):
        got = await deliver(harness, rig, chat_id=PREP, text="", msg_id=f"v-{kind}", unsupported=kind)
        assert [(m.unsupported, m.attachments) for m in got] == [(kind, ())]
    assert harness.stats(rig)["downloads"] == 0 and not (media.exists() and any(media.iterdir()))
    await emit(harness, rig, raw_message(rig, msg="v2", unsupported="voice",
                                         attachments=[{"kind": "document", "path": str(media / "x"), "mime": ""}]))
    assert rig.delivered[-1].attachments == ()  # a flagged message never carries files
    await rig.adapter.close()


async def test_documents_and_images_mapped(harness: WaHarness, tmp_path: Path) -> None:
    rig = await harness.rig()
    got = await deliver(harness, rig, chat_id=PREP, text="here is my cv", msg_id="f1",
                        media={"kind": "document", "name": "cv.pdf", "mime": "application/pdf"})
    (att,) = got[0].attachments
    assert (att.kind, att.mime, att.name) == ("document", "application/pdf", "cv.pdf")
    assert att.path.is_file() and att.path.is_relative_to((harness.root / "wa" / "media").resolve())
    outside = tmp_path / "evil.pdf"
    outside.write_text("x")
    await emit(harness, rig, raw_message(rig, msg="f2", attachments=[
        {"kind": "document", "path": str(outside), "mime": "application/pdf", "name": "evil.pdf", "size": 1}]))
    assert rig.delivered[-1].attachments == ()  # only files inside the media folder
    await rig.adapter.close()


async def test_quotes_reactions_and_replayed(harness: WaHarness) -> None:
    rig = await harness.rig()
    a = adapter(rig)
    ref = await a.send_text(PREP, "Mac · Send it?")
    got = await deliver(harness, rig, chat_id=PREP, text="yes", msg_id="q1", replayed=True,
                        quoted={"msg": ref.message_id, "text": "Mac · Send it?", "from_owner": True})
    m = got[0]
    assert m.replayed and m.reply_to is not None
    assert (m.reply_to.ref, m.reply_to.from_owner) == (ref, True)
    got = await deliver(harness, rig, chat_id=PREP, text="", msg_id="q2",
                        reaction={"emoji": "👍", "target_msg": ref.message_id})
    assert (got[0].reaction, got[0].text, got[0].reply_to.ref) == ("👍", "", ref)
    got = await deliver(harness, rig, chat_id=PREP, text="what about this", msg_id="q3",
                        quoted={"msg": "THEIRS1", "text": "a message from the other member", "from_owner": False})
    assert got[0].reply_to.from_owner is False and got[0].reply_to.text == "a message from the other member"
    await a.close()


# -- outbound -----------------------------------------------------------------------------------------------------
async def test_outbound_arguments(harness: WaHarness, tmp_path: Path) -> None:
    rig = await harness.rig()
    a = adapter(rig)
    (req,) = await deliver(harness, rig, chat_id=PREP, text="find me 20 jobs", msg_id="REQ1",
                           quoted={"msg": "THEIRS1", "text": "x", "from_owner": False})
    await a.react(req.ref, "👀")
    status = await a.send_text(PREP, "Mac · On it", reply_to=req.ref)
    await a.send_text(PREP, "Mac · About that", reply_to=MessageRef(PREP, "THEIRS1"))
    assert await a.edit_text(status, "Mac · Found 20 roles")
    assert await a.edit_text(MessageRef(PREP, "UNKNOWN"), "Mac · x") is False  # the bridge: not_found
    png, pdf = tmp_path / "chart.png", tmp_path / "jobs.pdf"
    png.write_bytes(b"\x89PNG")
    pdf.write_bytes(b"%PDF")
    await a.send_file(PREP, png)
    await a.send_file(PREP, pdf, caption="the list")
    await a.set_typing(PREP, True)
    w = harness.stats(rig)["writes"]
    assert [x["op"] for x in w] == ["react", "send_text", "send_text", "edit", "send_media", "send_media",
                                    "chat_presence"]
    assert (w[0]["msg"], w[0]["from_me"], w[0]["emoji"]) == ("REQ1", True, "👀")
    assert w[1]["reply_to"] == {"msg": "REQ1", "from_me": True}
    assert w[2]["reply_to"] == {"msg": "THEIRS1", "from_me": False}
    assert (w[3]["msg"], w[3]["op"]) == (status.message_id, "edit")
    assert (w[4]["kind"], w[4]["mime"], w[4]["file_name"]) == ("image", "image/png", "chart.png")
    assert (w[5]["kind"], w[5]["mime"], w[5]["caption_len"]) == ("document", "application/pdf", 8)
    assert w[6]["state"] == "composing"
    with pytest.raises(BridgeError):
        await a.send_file(PREP, tmp_path / "missing.pdf")
    await a.close()


async def test_typing_and_presence_gated_by_config(harness: WaHarness) -> None:
    rig = await harness.rig_with(cfg={"typing": False})
    a = adapter(rig)
    await a.set_typing(PREP, True)
    assert harness.writes(rig) == 0 and rig.ctx.governor.counts["typing"] == 0
    await a.close()
    rig = await harness.rig_with(cfg={"presence": "available"})
    await harness.settle(rig)
    assert [w for w in harness.stats(rig)["writes"] if w["op"] == "presence"] == [
        {"op": "presence", "state": "available"}]
    await rig.adapter.close()


async def test_answers_out_of_order(harness: WaHarness) -> None:
    rig = await harness.rig_with(fake={"delay": {"send_text": 0.5}})
    a = adapter(rig)
    (req,) = await deliver(harness, rig, chat_id=PREP, text="go", msg_id="REQ1")
    done: list[str] = []

    async def track(name: str, coro: Any) -> Any:
        r = await coro
        done.append(name)
        return r

    ref, _ = await asyncio.gather(track("send", a.send_text(PREP, "Mac · On it")), track("react", a.react(req.ref, "👀")))
    assert done == ["react", "send"] and ref.message_id.startswith("FAKE")
    await a.close()


# -- timeouts and the supervisor --------------------------------------------------------------------------
async def test_timeout_degrades_then_recovers_and_restarts(harness: WaHarness) -> None:
    rig = await harness.rig_with(cfg={"request_timeout_s": 0.3}, fake={"hang": ["status"]})
    a, d = adapter(rig), harness.dir(rig)
    with pytest.raises(BridgeError) as e:
        await a.bridge_status()
    assert e.value.code == "timeout" and a.degraded
    assert a.status().state is LinkState.ERROR and a.status().detail == wa.NOT_ANSWERING
    await a.send_text(PREP, "Mac · hi")  # any answer clears it
    assert not a.degraded and a.status().state is LinkState.CONNECTED
    for _ in range(2):  # two in a row: the bridge is restarted
        with pytest.raises(BridgeError):
            await a.bridge_status()
    await until(lambda: starts(d) == 2 and a.status().state is LinkState.CONNECTED)
    await a.close()


async def test_exit_1_restarts_with_backoff(harness: WaHarness) -> None:
    rig = await harness.rig_with(fake={"exit_codes": [1, 1]})
    a, d = adapter(rig), harness.dir(rig)
    await until(lambda: a.status().state is LinkState.CONNECTED)
    assert starts(d) == 3 and a.exit_code == 1
    assert wa.RESTARTING in [s.detail for s in rig.statuses]
    await a.close()


async def test_exit_1_breaker_after_five_crashes(harness: WaHarness) -> None:
    rig = await harness.rig_with(fake={"exit_codes": [1] * 10}, wait=False)
    a, d = adapter(rig), harness.dir(rig)
    await until(lambda: a.status().detail == wa.CRASHING)
    await asyncio.sleep(0.1)
    assert starts(d) == 5 and a.status().state is LinkState.ERROR
    await a.close()


@pytest.mark.parametrize("fake", [{"exit_codes": [2]}, {"protocol": "1"}])
async def test_exit_2_never_restarts(harness: WaHarness, fake: dict[str, Any]) -> None:
    rig = await harness.rig_with(fake=fake, wait=False)
    a, d = adapter(rig), harness.dir(rig)
    await until(lambda: a.status().detail == wa.BAD_BRIDGE)
    await asyncio.sleep(0.2)
    assert starts(d) == 1 and a.exit_code == 2 and a.status().state is LinkState.ERROR
    with pytest.raises(BridgeError):
        await a.send_text(PREP, "Mac · hi")
    await a.close()


async def test_exit_3_budget_spent_never_restarts(harness: WaHarness) -> None:
    rig = await harness.rig_with(test=True, cfg={"budget": 2})
    a, d = adapter(rig), harness.dir(rig)
    await a.send_text(PREP, "Mac · one")
    await a.send_text(PREP, "Mac · two")
    with pytest.raises(BudgetExceeded):
        await a.send_text(PREP, "Mac · three")  # the bridge's own budget: fatal
    await until(lambda: a.exit_code == 3)
    await asyncio.sleep(0.2)
    assert starts(d) == 1 and a.status().detail == wa.BUDGET_SPENT
    used = dict(rig.ctx.governor.counts)
    with pytest.raises(BudgetExceeded):
        await a.send_text(PREP, "Mac · four")  # refused before the governor and any I/O
    assert rig.ctx.governor.counts == used
    await a.close()


async def test_exit_75_in_use_by_a_test_run_then_resume(harness: WaHarness) -> None:
    first = await harness.rig()
    second = await harness.rig_with(wait=False)  # same store: the lock is held
    b, d2 = adapter(second), harness.dir(second)
    assert b.status() == second.statuses[-1]
    assert (b.status().state, b.status().detail, b.exit_code) == (LinkState.ERROR, wa.IN_USE, 75)
    await asyncio.sleep(0.2)
    assert starts(d2) == 1  # never retried
    await first.adapter.close()
    await b.resume()  # `chatd resume`
    assert b.status().state is LinkState.CONNECTED and starts(d2) == 2
    await b.close()


async def test_exit_0_and_close_lifeline(harness: WaHarness) -> None:
    rig = await harness.rig()
    a, d = adapter(rig), harness.dir(rig)
    await acontrol(d, cmd="exit", code=0)
    await until(lambda: a.status().detail == wa.STOPPED)
    assert starts(d) == 1
    await a.close()
    rig = await harness.rig()
    a = adapter(rig)
    await a.close()  # stdin EOF: the bridge exits 0 by itself
    assert a.exit_code == 0


async def test_missing_bridge_binary(harness: WaHarness) -> None:
    rig = await harness.rig_with(cfg={"command": [str(harness.root / "no-such-bridge")]}, wait=False)
    assert (rig.adapter.status().state, rig.adapter.status().detail) == (LinkState.ERROR, wa.NOT_INSTALLED)
    await rig.adapter.close()


async def test_logged_out_reasons(harness: WaHarness) -> None:
    rig = await harness.rig()
    await emit(harness, rig, {"ev": "logged_out", "reason": "banned"})
    assert (rig.adapter.status().state, rig.adapter.status().detail) == (LinkState.UNLINKED, wa.BANNED)
    await emit(harness, rig, {"ev": "logged_out", "reason": "phone"})
    assert rig.adapter.status().detail == wa.UNLINKED_PHONE
    await rig.adapter.close()


async def test_unlinked_start(harness: WaHarness) -> None:
    rig = await harness.rig_with(fake={"linked": False}, wait=False)
    assert rig.adapter.status().state is LinkState.UNLINKED
    with pytest.raises(BridgeError) as e:
        await adapter(rig).send_text(PREP, "Mac · hi")
    assert e.value.code == "not_linked"
    await rig.adapter.close()


# -- privacy --------------------------------------------------------------------------------------------------
async def test_no_message_text_in_any_log(harness: WaHarness, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    subject = "Secret Subject 42"
    rig = await harness.rig_with(fake={"chats": [{"id": PREP, "subject": subject}]})
    a = adapter(rig)
    await deliver(harness, rig, chat_id=PREP, text=SECRET, msg_id="s1")
    await deliver(harness, rig, chat_id=PREP, text=SECRET, msg_id="s2", owner=False)
    await deliver(harness, rig, chat_id=PREP, text="x", msg_id="s3",
                  quoted={"msg": "T", "text": SECRET, "from_owner": False})
    await deliver(harness, rig, chat_id=OTHER, text=SECRET, msg_id="s4")
    ref = await a.send_text(PREP, f"Mac · {SECRET}")
    await a.edit_text(ref, f"Mac · {SECRET} edited")
    await a.list_chats()
    await acontrol(harness.dir(rig), cmd="chat_changed", chat=PREP, subject=subject, members=1, owner_only=True)
    await harness.settle(rig)
    with pytest.raises(BridgeError):
        await a.request("send_text", chat=PREP, text="")  # an error answer
    await a.close()
    bridge_log = (harness.root / "wa-bridge.log").read_text()
    assert "msg=s2" in bridge_log or "dropped" in bridge_log  # the bridge log is written
    for text in (caplog.text, bridge_log):
        assert SECRET not in text and subject not in text


# -- v3.4 allowlist from agentd's profiles (W9.8) ----------------------------------------------------------
async def test_set_allowed_filters_the_never_list(harness: WaHarness) -> None:
    rig = await harness.rig_with(chats=frozenset())
    a = adapter(rig)
    assert "--allow=" in harness.stats(rig)["argv"]  # fail closed: empty until agentd's data arrives
    assert await deliver(harness, rig, chat_id=PREP, text="hi", msg_id="m0") == []
    await a.set_allowed([PREP, OTHER, TIMETABLE])
    assert harness.stats(rig)["allow"] == [PREP, OTHER] and rig.ctx.chats == {PREP, OTHER}
    assert len(await deliver(harness, rig, chat_id=PREP, text="hi", msg_id="m1")) == 1
    await a.set_allowed([OTHER])
    assert harness.stats(rig)["allow"] == [OTHER] and not rig.ctx.allowed(PREP)
    with pytest.raises(DestinationRefused):
        await a.send_text(PREP, "Mac · hi")
    await a.close()


async def test_set_allowed_never_sent_under_test(harness: WaHarness) -> None:
    rig = await harness.rig_with(test=True, cfg={"budget": 5})
    a = adapter(rig)
    await a.set_allowed([PREP, OTHER])
    st = harness.stats(rig)
    assert "set_allow" not in st["ops"] and st["allow"] == [PREP] and not rig.ctx.allowed(OTHER)
    await acontrol(harness.dir(rig), cmd="offline")
    await acontrol(harness.dir(rig), cmd="reconnect")
    await harness.settle(rig)
    await until(lambda: a._resync is not None and a._resync.done())
    assert "set_allow" not in harness.stats(rig)["ops"]
    await a.close()


async def test_allowlist_resent_after_restart_and_connected(harness: WaHarness) -> None:
    rig = await harness.rig_with(chats=frozenset())
    a, d = adapter(rig), harness.dir(rig)
    calls: list[str] = []

    async def member_check() -> None:
        calls.append(a.status().state.value)

    a.on_connected = member_check
    await until(lambda: a._resync is not None and a._resync.done())  # the start's own resync
    calls.clear()
    await a.set_allowed([PREP])
    await acontrol(d, cmd="offline")
    await acontrol(d, cmd="reconnect")
    await until(lambda: calls == ["connected"])
    assert harness.stats(rig)["ops"].count("set_allow") == 2  # set_allowed, then again on connected
    await acontrol(d, cmd="exit", code=1)  # a crash: the new bridge must not be deaf
    await until(lambda: starts(d) == 2 and len(calls) == 2)
    st = harness.stats(rig)
    assert f"--allow={PREP}" in st["argv"] and "set_allow" in st["ops"] and st["allow"] == [PREP]
    assert len(await deliver(harness, rig, chat_id=PREP, text="hi", msg_id="m1")) == 1
    await a.close()


async def test_set_allowed_while_the_bridge_is_down(harness: WaHarness) -> None:
    rig = await harness.rig_with(chats=frozenset())
    a, d = adapter(rig), harness.dir(rig)
    await acontrol(d, cmd="exit", code=0)
    await until(lambda: a.status().detail == wa.STOPPED)
    await a.set_allowed([PREP, TIMETABLE])  # no error; applied at the next start
    assert rig.ctx.chats == {PREP}
    await a.resume()
    assert harness.stats(rig)["allow"] == [PREP]
    await a.close()


# -- review fixes -------------------------------------------------------------------------------------------
async def test_far_future_clock(harness: WaHarness) -> None:
    """The age rule uses the test clock in the bridge too, never the wall clock."""
    rig = await harness.rig_with(clock_t=4_000_000_000.0)  # year 2096
    assert len(await deliver(harness, rig, chat_id=PREP, text="hi", msg_id="m1")) == 1
    old = rig.clock() - 8 * 86400
    assert await deliver(harness, rig, chat_id=PREP, text="hi", msg_id="m2", ts=old) == []
    assert harness.stats(rig)["dropped"] == {"old_or_broadcast": 1}
    await rig.adapter.close()


async def test_owner_rechecked_in_the_adapter(harness: WaHarness) -> None:
    rig = await harness.rig()
    await emit(harness, rig, raw_message(rig, msg="n1", from_me=False))
    await emit(harness, rig, raw_message(rig, msg="n2", from_me="yes"))
    missing = raw_message(rig, msg="n4")
    del missing["from_me"]
    await emit(harness, rig, missing)  # an event without the field is dropped
    assert rig.delivered == []
    await emit(harness, rig, raw_message(rig, msg="n3", from_me=True))
    assert [m.ref.message_id for m in rig.delivered] == ["n3"]
    await rig.adapter.close()


async def test_set_allow_narrows_before_the_bridge_answers(harness: WaHarness) -> None:
    rig = await harness.rig_with(cfg={"request_timeout_s": 0.3}, fake={"hang": ["set_allow"]})
    a = adapter(rig)
    with pytest.raises(BridgeError):
        await a.set_allow([OTHER])
    assert not rig.ctx.allowed(PREP) and not rig.ctx.allowed(OTHER)  # never the old, wider set
    with pytest.raises(DestinationRefused):
        await a.send_text(PREP, "Mac · hi")
    await a.close()


# -- protocol 2 (v3.5, W10): context, members, mentions, history, quotes -----------------------------------
ARJUN = {"jid": "222@s.whatsapp.net", "number": "222", "name": "Arjun"}
PRIYA = {"jid": "333@s.whatsapp.net", "number": "333", "name": "Priya"}


async def test_context_reaches_only_on_context(harness: WaHarness) -> None:
    rig = await harness.rig()
    a = adapter(rig)
    await a.set_allowed(rig.ctx.chats, {PREP, TIMETABLE})  # a never-listed chat never becomes context
    assert a.context_ids == {PREP} and harness.stats(rig)["context"] == [PREP]
    await harness.inject(rig, chat_id=PREP, text="x" * 5000, msg_id="c1", owner=False, sender_name="Arjun",
                         media={"kind": "image", "name": "cv.png"})
    await harness.inject(rig, chat_id=PREP, text="", msg_id="c2", owner=False, reaction="👍")  # never context
    await harness.settle(rig)
    assert rig.delivered == [] and len(rig.contexts) == 1
    c = rig.contexts[0]
    assert (c.sender_name, len(c.text), c.from_owner, c.history) == ("Arjun", 4000, False, False)
    assert c.media == {"kind": "image", "name": "cv.png"}
    assert harness.stats(rig)["downloads"] == 0  # context media is never downloaded
    await a.set_allowed(rig.ctx.chats, ())  # read_chat off: members' messages are dropped again
    await harness.inject(rig, chat_id=PREP, text="hi", msg_id="c3", owner=False)
    await harness.settle(rig)
    assert len(rig.contexts) == 1 and harness.stats(rig)["dropped"].get("not_owner") == 2
    await a.close()


async def test_context_flags_and_trusted_default(harness: WaHarness) -> None:
    rig = await harness.rig()
    argv = harness.stats(rig)["argv"]
    assert "--trusted-devices=all" in argv and "--context=" in argv
    await adapter(rig).close()


async def test_context_event_rechecked_in_the_adapter(harness: WaHarness) -> None:
    rig = await harness.rig()
    a = adapter(rig)
    sent = await a.send_text(PREP, "Mac · hi")
    ev = {"ev": "context", "chat": PREP, "msg": "c1", "sender": ARJUN["jid"], "from_owner": False, "at": 1.0,
          "text": "hi", "media": {"kind": "image", "path": "/etc/passwd"}}
    await emit(harness, rig, ev)  # not a context chat
    await a.set_allowed(rig.ctx.chats, {PREP})
    await emit(harness, rig, {**ev, "msg": sent.message_id})  # our own message
    await emit(harness, rig, {**ev, "msg": "c2", "text": "Mac · echo"})
    await emit(harness, rig, ev)
    assert [c.ref.message_id for c in rig.contexts] == ["c1"] and rig.contexts[0].media == {"kind": "image"}
    await a.close()


async def test_revoked_reaches_on_context_as_deleted(harness: WaHarness) -> None:
    rig = await harness.rig()
    a = adapter(rig)
    await a.set_allowed(rig.ctx.chats, {PREP})
    await emit(harness, rig, {"ev": "revoked", "chat": "120363000000000077@g.us", "msg": "x1"})  # not allowed
    await emit(harness, rig, {"ev": "revoked", "chat": PREP, "msg": "c9"})
    assert [(c.chat_id, c.ref.message_id, c.deleted, c.text) for c in rig.contexts] == [(PREP, "c9", True, "")]
    await a.close()


async def test_set_allow_context_must_be_inside_allow(harness: WaHarness) -> None:
    rig = await harness.rig()
    a = adapter(rig)
    await a.set_allow([PREP, OTHER], [OTHER, "999@g.us"])  # the adapter keeps only the allowed part
    assert a.context_ids == {OTHER} and harness.stats(rig)["context"] == [OTHER]
    await a.set_allow([PREP])  # context omitted: narrowed with the allowlist
    assert a.context_ids == frozenset() and harness.stats(rig)["context"] == []
    with pytest.raises(BridgeError) as e:  # the bridge refuses a context outside allow on its own
        await a._call("set_allow", allow=[PREP], context=[OTHER])
    assert e.value.code == "bad_args"
    await a.close()


async def test_members_and_mentions(harness: WaHarness) -> None:
    rig = await harness.rig_with(fake={"members": {PREP: [ARJUN, PRIYA]}})
    a = adapter(rig)
    assert await a.members(PREP) == [ARJUN, PRIYA]
    with pytest.raises(DestinationRefused):
        await a.members(TIMETABLE)
    await a.send_text(PREP, "Mac · @222 and @999 look", mentions=[ARJUN["jid"], PRIYA["jid"], "999@s.whatsapp.net"])
    ref = await a.send_text(PREP, "Mac · plain")
    assert await a.edit_text(ref, "Mac · now @333", mentions=[PRIYA["jid"]])
    writes = harness.stats(rig)["writes"]
    assert [w.get("mentions") for w in writes if w["op"] in ("send_text", "edit")] == [[ARJUN["jid"]], [],
                                                                                     [PRIYA["jid"]]]
    await a.close()


async def test_request_history(harness: WaHarness) -> None:
    rig = await harness.rig()
    a = adapter(rig)
    await harness.inject(rig, chat_id=PREP, text="first", msg_id="m1")
    await harness.settle(rig)
    before = MessageRef(PREP, "m1")
    assert not await a.request_history(PREP, before)  # not a context chat
    await a.set_allowed(rig.ctx.chats, {PREP})
    assert await a.request_history(PREP, before, count=500)
    assert next(w for w in harness.stats(rig)["writes"] if w["op"] == "history")["count"] == 50
    assert not await a.request_history(PREP, before)  # once per chat per hour
    owner = {"chat": PREP, "msg": "h1", "text": "old owner text", "from_me": True}
    await acontrol(harness.dir(rig), cmd="history_blob",
                   messages=[owner, {"chat": PREP, "msg": "h2", "text": "old member text", "sender_name": "Arjun"}])
    await harness.settle(rig)
    assert [m.ref.message_id for m in rig.delivered] == ["m1"]  # history is never a trigger, even the owner's
    assert [(c.ref.message_id, c.history, c.from_owner) for c in rig.contexts] == [("h1", True, True),
                                                                                  ("h2", True, False)]
    await acontrol(harness.dir(rig), cmd="history_blob", messages=[{**owner, "msg": "h3"}])  # nothing open
    await harness.settle(rig)
    assert len(rig.contexts) == 2
    await a.close()


async def test_request_history_off_under_dry_run(harness: WaHarness) -> None:
    rig = await harness.rig(dry_run=True)
    a = adapter(rig)
    await a.set_allowed(rig.ctx.chats, {PREP})
    assert not await a.request_history(PREP, MessageRef(PREP, "m1"))
    assert harness.stats(rig)["writes"] == []
    await a.close()


async def test_extended_quote(harness: WaHarness, short_root: Path) -> None:
    rig = await harness.rig()
    a = adapter(rig)
    media = short_root / "wa" / "media"
    media.mkdir(parents=True, exist_ok=True)
    (media / "q1-cv.pdf").write_bytes(b"%PDF")
    outside = short_root / "secret.pdf"
    outside.write_bytes(b"x")
    links = ["https://example.com/job", "javascript:alert(1)", *[f"http://x/{i}" for i in range(12)]]
    quoted = {"msg": "q1", "text": "the role", "from_owner": False, "sender": ARJUN["jid"], "sender_name": "Arjun",
              "at": 5.0, "links": links,
              "media": {"kind": "document", "name": "cv.pdf", "mime": "application/pdf", "size": 4,
                        "path": str(media / "q1-cv.pdf")}}
    await emit(harness, rig, raw_message(rig, msg="m1", text="look", quoted=quoted))
    await emit(harness, rig, raw_message(rig, msg="m2", text="look",
                                         quoted={**quoted, "media": {**quoted["media"], "path": str(outside)}}))
    await emit(harness, rig, raw_message(rig, msg="m3", text="look", quoted={"msg": "zz", "kind": "other_chat",
                                                                            "text": "leak"}))
    q1, q2, q3 = (m.reply_to for m in rig.delivered)
    assert q1 is not None and q2 is not None and q3 is not None
    assert (q1.sender_name, q1.at, q1.links[0], len(q1.links)) == ("Arjun", 5.0, "https://example.com/job", 10)
    assert q1.media is not None and q1.media.path == (media / "q1-cv.pdf").resolve()
    assert q2.media is not None and q2.media.path is None  # a file outside the media folder is ignored
    assert (q3.kind, q3.text) == ("other_chat", "")
    await a.close()
