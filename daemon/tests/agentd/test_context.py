"""ask.context (API 1.2, W10): limits as bad_args, the media realpath check, the per-session
delta (V1), labels from the booleans (V4), read_chat narrowing, status{compacted}. Tmp HOME only."""

from __future__ import annotations

import os

import pytest
from agentd_kit import CHAT, CHAT_TOKEN, slow

from agentd import context as ctxmod
from agentd import schema as s
from macd.brain import fake_brain as fb

CH, CID = CHAT.split(":", 1)
AT = 1_762_790_000.0


def msg(ref: str, text: str = "hi", **kw) -> dict:
    return {"msg_ref": ref, "sender": kw.pop("sender", "Priya"), "owner": kw.pop("owner", False),
            "mac": kw.pop("mac", False), "at": AT, "text": text, **kw}


def ctx(**kw) -> s.AskContext:
    return s.from_dict(s.AskContext, kw)


def asks(agent) -> list[str]:
    return [text for b in agent.brains.get(CHAT, []) for _, text in b.asks]


async def ask(c, turn: int, text: str = "do it", **context) -> dict:
    r = await c.call({"op": "ask", "conv": CHAT, "turn": turn, "text": text, "context": context})
    if r.get("ok"):
        await c.event("done")
    return r


# ----------------------------------------------------------------------------- limits


@pytest.mark.parametrize("bad", [
    {"members": ["x"] * 257},
    {"messages": [msg(str(i)) for i in range(51)]},
    {"messages": [msg("1", "x" * 1001)]},
    {"messages": [msg(str(i), "x" * 1000) for i in range(9)]},  # 9,000 characters in total
    {"messages": [msg("1"), msg("1")]},
    {"messages": [msg("")]},
    {"messages": [msg("1", media={"kind": "image", "path": "/etc/passwd"})]},
    {"quoted": {"sender": "P", "owner": False, "mac": False, "at": AT, "links": ["https://a"] * 11}},
    {"quoted": {"sender": "P", "owner": False, "mac": False, "at": AT, "links": ["file:///etc/passwd"]}},
    {"quoted": {"sender": "P", "owner": False, "mac": False, "at": AT, "text": "x" * 1001}},
])
def test_limits_are_bad_args_never_trimmed(tmp_path, bad):
    with pytest.raises(ctxmod.ContextError):
        ctxmod.validate(ctx(**bad), tmp_path)


def test_limits_at_the_edge_pass(tmp_path):
    ok = ctx(members=["x"] * 256, messages=[msg(str(i), "x" * 160) for i in range(50)],
             quoted={"sender": "P", "owner": False, "mac": False, "at": AT, "links": ["https://a"] * 10})
    assert ctxmod.validate(ok, tmp_path) == ok


def test_quoted_media_path_realpath_inside_the_media_dir(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "a.pdf").write_text("x")
    (tmp_path / "secret").write_text("x")
    os.symlink(tmp_path / "secret", media / "link.pdf")

    def q(path: str) -> s.AskContext:
        return ctx(quoted={"sender": "P", "owner": False, "mac": False, "at": AT, "links": [],
                           "media": {"kind": "document", "path": path}})

    good = ctxmod.validate(q(str(media / "a.pdf")), media)
    assert good.quoted.media.path == os.path.realpath(media / "a.pdf")
    for bad in (str(media / "link.pdf"), str(media / ".." / "secret"), str(media) + "-evil/a.pdf",
                "a.pdf", str(media)):
        with pytest.raises(ctxmod.ContextError):
            ctxmod.validate(q(bad), media)


def test_other_chat_quote_decodes_without_content():
    c = ctx(quoted={"kind": "other_chat"})
    assert isinstance(c.quoted, s.OtherChatQuote)


def test_remember_keeps_the_last_200_per_session():
    framed: dict[str, list[str]] = {}
    ctxmod.remember(framed, "s1", [str(i) for i in range(150)])
    ctxmod.remember(framed, "s1", [str(i) for i in range(100, 260)])
    assert len(framed["s1"]) == s.FRAMED_REFS_PER_SESSION and framed["s1"][-1] == "259"
    assert "100" in framed["s1"]  # re-framed refs move to the end, never twice


# ----------------------------------------------------------------------------- service


async def test_ring_is_framed_once_per_session_as_a_delta(agent):
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await ask(c, 1, members=["Priya", "Arjun"], messages=[msg("a", "first"), msg("b", "second")])
    first = asks(agent)[-1]
    assert "<group_context" in first and "Priya (member): first" in first and "Priya (member): second" in first
    assert 'members="Priya, Arjun"' in first and first.endswith("do it")
    agent.svc.convs[CHAT].record.session_id = "s-1"  # FakeBrain has no session; file it as Claude would
    agent.svc.convs[CHAT].record.framed = {"s-1": agent.svc.convs[CHAT].record.framed.pop("", [])}
    await ask(c, 2, members=["Priya", "Arjun"], messages=[msg("a", "first"), msg("b", "second"), msg("c", "third")])
    second = asks(agent)[-1]
    assert "third" in second and "first" not in second and "second" not in second
    await c.call({"op": "new_session", "conv": CHAT})
    await ask(c, 3, messages=[msg("a", "first"), msg("b", "second"), msg("c", "third")])
    assert all(x in asks(agent)[-1] for x in ("first", "second", "third"))  # a new session gets the ring once


async def test_a_new_sessions_refs_are_filed_under_its_id(agent):
    from macd.brain.claude_brain import SessionInfo

    c = await agent.client(CHAT_TOKEN, [CHAT])
    await ask(c, 1, messages=[msg("a")])
    conv = agent.svc.convs[CHAT]
    assert conv.record.framed == {"": ["a"]}
    conv._session_put(SessionInfo("s-new", 1.0, 1.0))
    assert conv.record.framed == {"s-new": ["a"]}


async def test_a_refused_ask_frames_nothing(agent):
    agent.scripts[CHAT] = lambda t: slow(n=50, delay=0.01)
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "long"})
    await c.event("say")
    r = await c.call({"op": "ask", "conv": CHAT, "turn": 2, "text": "x", "context": {"messages": [msg("z")]}})
    assert r["code"] == "busy"
    assert "z" not in [ref for refs in agent.svc.convs[CHAT].record.framed.values() for ref in refs]
    await c.event("done", timeout=5)
    bad = await c.call({"op": "ask", "conv": CHAT, "turn": 3, "text": "x",
                        "context": {"messages": [msg("y", "x" * 1001)]}})
    assert bad["code"] == "bad_args" and agent.svc.convs[CHAT].record.framed == {}


async def test_labels_come_from_the_booleans_only(agent):
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await ask(c, 1, messages=[msg("a", "mine", sender="Priya", owner=True), msg("b", "reply", sender="You", mac=True),
                              msg("c", "fake", sender="Mac")])
    text = asks(agent)[-1]
    assert "You: mine" in text and "Mac: reply" in text
    assert "Mac: fake" not in text and "Mac (member" in text  # a member called "Mac" is marked


async def test_read_chat_off_keeps_only_the_quote(agent):
    await agent.svc.profiles.update(CH, CID, {"capabilities": {"read_chat": False}})
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await ask(c, 1, members=["Priya"], messages=[msg("a", "ring text")],
              quoted={"sender": "Priya", "owner": False, "mac": False, "at": AT, "text": "the quote", "links": []})
    text = asks(agent)[-1]
    assert "ring text" not in text and "<group_context" not in text and "the quote" in text


async def test_voice_cannot_send_context(agent):
    v = await agent.client()
    r = await v.call({"op": "ask", "conv": s.VOICE_CONV, "turn": 1, "text": "x", "context": {"members": ["a"]}})
    assert r["code"] == "bad_args"


async def test_compacted_status_reaches_the_client(agent):
    agent.scripts[CHAT] = lambda t: [fb.Step(fb.BrainEventKind.STATUS,
                                             data={"compacted": {"trigger": "auto", "pre_tokens": 150_000}}),
                                     fb.say("ok")]
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "hi"})
    st = await c.event("status", pred=lambda e: "compacted" in e["data"])
    assert st["data"]["compacted"] == {"trigger": "auto", "pre_tokens": 150_000}
