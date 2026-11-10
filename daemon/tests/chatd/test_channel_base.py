"""chatd.channels.base: the adapter contract types exist with the W5.1 defaults, and a minimal
adapter satisfies the Protocol."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from chatd.channels import base as b
from macd.contracts import BrainEventKind


def test_defaults_match_design():
    lim = b.Limits()
    assert (lim.max_text, lim.edit_window_s, lim.min_edit_interval_s, lim.max_file_mb, lim.text_format) == \
        (4000, 0.0, 4.0, 100, "plain")
    assert [s.value for s in b.LinkState] == ["unlinked", "pairing", "connected", "offline", "error"]
    assert b.Cap.NONE.value == 0 and (b.Cap.EDIT | b.Cap.FILES) & b.Cap.FILES


def test_inbound_message_is_frozen_and_owner_shaped():
    ref = b.MessageRef("1203@g.us", "3EB0")
    m = b.InboundMessage(channel="whatsapp", chat_id="1203@g.us", ref=ref, text="hi",
                         attachments=(b.Attachment("document", Path("/tmp/a.pdf"), "application/pdf"),),
                         reply_to=b.Quoted(ref, "earlier", True))
    assert m.unsupported == "" and not m.replayed
    with pytest.raises(FrozenInstanceError):
        m.text = "x"  # type: ignore[misc]


class _Mini:
    name = "fake"
    caps = b.Cap.EDIT
    limits = b.Limits()

    async def start(self, sink, on_status, on_chat_changed): ...
    async def close(self): ...
    def status(self): return b.ChannelStatus(b.LinkState.UNLINKED)
    async def mark_handled(self, ref): ...
    async def send_text(self, chat_id, text, reply_to=None): return b.MessageRef(chat_id, "1")
    async def edit_text(self, ref, text): return True
    async def react(self, ref, emoji): ...
    async def set_typing(self, chat_id, on): ...
    async def send_file(self, chat_id, path, caption=""): return b.MessageRef(chat_id, "2")
    async def link(self): ...
    async def unlink(self): ...
    async def list_chats(self): return []


def test_protocol_is_satisfiable():
    assert isinstance(_Mini(), b.ChannelAdapter)


def test_brain_event_kind_attach_exists():
    assert BrainEventKind.ATTACH.value == "attach"
