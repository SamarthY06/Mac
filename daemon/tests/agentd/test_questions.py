"""Questions (W3.2, M5, F14): binding rules, one open question, Skipped on replace, change-it,
catch-up and route answers through the service."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from agentd_kit import CHAT, CHAT_TOKEN, confirm_step, slow

from agentd import schema as s
from agentd.questions import Question, bind, yes_no
from macd.brain import fake_brain as fb

K = s.QuestionKind
NOW = 1000.0


def q(kind: K, ref: str = "p1", options: int = 2) -> Question:
    opts = [{"label": "Resume", "keys": ["resume", "yes"]}, {"label": "generally", "keys": ["general", "no"]}]
    return Question("q1", kind, "?", NOW + 60, prompt_ref=ref, payload={"options": opts[:options]})


def reply(**kw) -> s.Reply:
    return s.Reply(conv=CHAT, **kw)


@pytest.mark.parametrize("text,want", [("yes", "yes"), ("Yes!", "yes"), ("send it", "yes"), ("go ahead.", "yes"),
                                       ("\U0001F44D", "yes"), ("\U0001F44D\U0001F3FD", "yes"), ("no", "no"),
                                       ("Cancel", "no"), ("don’t", "no"), ("yes, but to Ann", None),
                                       ("maybe", None)])
def test_yes_no(text, want):
    assert yes_no(text) == want


def test_confirm_outcomes():
    assert bind(q(K.CONFIRM), reply(text="yes"), NOW).outcome is s.ReplyOutcome.APPROVED
    assert bind(q(K.CONFIRM), reply(text="no", quoted_ref="p1"), NOW).outcome is s.ReplyOutcome.DENIED
    b = bind(q(K.CONFIRM), reply(text="yes, but subject Hi"), NOW)
    assert b.outcome is s.ReplyOutcome.CHANGED and b.steer_text == "yes, but subject Hi"
    assert bind(q(K.CONFIRM), reply(reaction="\U0001F44D", reaction_on="p1"), NOW).outcome is s.ReplyOutcome.APPROVED


def test_f14_old_prompts_never_bind():
    assert not bind(q(K.CONFIRM), reply(reaction="\U0001F44D", reaction_on="old"), NOW).bound
    assert not bind(q(K.CONFIRM), reply(text="yes", quoted_ref="old"), NOW).bound
    assert not bind(q(K.CONFIRM), reply(reaction="❤️", reaction_on="p1"), NOW).bound
    assert not bind(q(K.CONFIRM, ref=""), reply(reaction="\U0001F44D", reaction_on=""), NOW).bound
    assert not bind(q(K.CONFIRM), reply(text="yes"), NOW + 61).bound  # expired
    assert not bind(None, reply(text="yes"), NOW).bound


def test_folder_approval_needs_quote_or_reaction():
    assert not bind(q(K.FOLDER_APPROVAL), reply(text="yes"), NOW).bound
    assert bind(q(K.FOLDER_APPROVAL), reply(text="yes", quoted_ref="p1"), NOW).outcome is s.ReplyOutcome.APPROVED
    assert bind(q(K.FOLDER_APPROVAL), reply(reaction="\U0001F44D", reaction_on="p1"), NOW).bound
    assert not bind(q(K.FOLDER_APPROVAL), reply(text="sure thing buddy", quoted_ref="p1"), NOW).bound


def test_route_catch_up_restart_outcomes():
    b = bind(q(K.ROUTE), reply(text="do it in resume"), NOW)
    assert b.outcome is s.ReplyOutcome.ANSWERED and b.choice == 0
    assert bind(q(K.ROUTE), reply(text="no"), NOW).choice == 1
    assert not bind(q(K.ROUTE), reply(text="what's the weather"), NOW).bound
    assert bind(q(K.CATCH_UP), reply(text="yes"), NOW).choice == 0
    assert bind(q(K.RESTART), reply(text="no"), NOW).outcome is s.ReplyOutcome.DENIED
    assert not bind(q(K.RESTART), reply(text="find more jobs"), NOW).bound


# ----------------------------------------------------------------------------- service


async def test_chat_confirm_by_quoted_yes_and_skipped_replace(agent):
    agent.scripts[CHAT] = lambda t: [confirm_step("first"), confirm_step("second")]
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "email"})
    first = await c.event("confirm")
    await c.call({"op": "question_posted", "conv": CHAT, "question_id": first["question_id"], "msg_ref": "p1"})
    r = await c.call({"op": "reply", "conv": CHAT, "text": "no", "quoted_ref": "p1"})
    assert r["data"] == {"bound": True, "outcome": "denied"}
    second = await c.event("confirm")
    await c.call({"op": "question_posted", "conv": CHAT, "question_id": second["question_id"], "msg_ref": "p2"})
    # a thumbs up on the OLD prompt never approves the new one (F14)
    r = await c.call({"op": "reply", "conv": CHAT, "reaction": "\U0001F44D", "reaction_on": "p1"})
    assert r["data"] == {"bound": False}
    r = await c.call({"op": "reply", "conv": CHAT, "reaction": "\U0001F44D", "reaction_on": "p2"})
    assert r["data"] == {"bound": True, "outcome": "approved"}
    await c.event("done")
    assert agent.brain(CHAT).confirm_results == [False, True]


async def test_change_it_denies_then_steers(agent):
    agent.scripts[CHAT] = lambda t: [confirm_step(), *slow(n=5, delay=0.02)]
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "email"})
    await c.event("confirm")
    r = await c.call({"op": "reply", "conv": CHAT, "text": "make the subject shorter"})
    assert r["data"] == {"bound": True, "outcome": "changed"}
    assert (await c.event("question_closed"))["outcome"] == "changed"
    assert (await c.event("text"))["text"] == "not sent"
    b = agent.brain(CHAT)
    assert b.steers and b.steers[-1][1] == "make the subject shorter"


async def test_new_question_closes_old_with_skipped(agent):
    conv = agent.svc.conversation(CHAT)
    c = await agent.client(CHAT_TOKEN, [CHAT])
    q1 = conv.open_question(K.CATCH_UP, "one?")
    q2 = conv.open_question(K.ROUTE, "two?", {"options": []})
    closed = await c.event("question_closed")
    assert closed == {**closed, "question_id": q1.question_id, "outcome": "skipped"}
    assert conv.question is q2 and conv.record.question.question_id == q2.question_id


async def test_catch_up_question_for_old_replayed_message(agent):
    c = await agent.client(CHAT_TOKEN, [CHAT])
    meta = {"replayed": True, "sent_at": time.time() - 7200, "msg_ref": "m1"}
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "find me 20 jobs", "meta": meta})
    qe = await c.event("question")
    assert qe["kind"] == "catch_up" and qe["text"].endswith('while I was asleep: "find me 20 jobs". Still want it?')
    r = await c.call({"op": "reply", "conv": CHAT, "text": "yes"})
    assert r["data"] == {"bound": True, "outcome": "answered"}
    await c.event("done")
    assert agent.brain(CHAT).asks[0][1] == "find me 20 jobs"


async def test_catch_up_no_ends_turn_without_brain(agent):
    c = await agent.client(CHAT_TOKEN, [CHAT])
    meta = {"replayed": True, "sent_at": time.time() - 7200}
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "x", "meta": meta})
    await c.event("question")
    await c.call({"op": "reply", "conv": CHAT, "text": "no"})
    assert (await c.event("done"))["data"]["interrupted"] is True
    assert CHAT not in agent.brains


async def test_route_question_answer_picks_folder(agent):
    proj = agent.paths.home / "Blog"
    proj.mkdir()
    agent.sessions = [SimpleNamespace(cwd=str(proj), last_modified=time.time() * 1000, session_id="b1",
                                      custom_title="Post", summary="")]
    agent.model.reply = '{"folder": "Blog", "session": "new", "confidence": 0.3}'
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "write something"})
    qe = await c.event("question")
    assert qe["kind"] == "route" and qe["text"] == "Should I do this in Blog, or just generally?"
    # a steer while asking is kept for the brain; an unrelated reply does not bind
    assert (await c.call({"op": "steer", "conv": CHAT, "text": "about agents"}))["data"] == {"accepted": True}
    assert (await c.call({"op": "reply", "conv": CHAT, "text": "hmm"}))["data"] == {"bound": False}
    r = await c.call({"op": "reply", "conv": CHAT, "text": "in blog"})
    assert r["data"] == {"bound": True, "outcome": "answered"}
    await c.event("done")
    b = agent.brain(CHAT)
    assert b.spec.cwd == proj and b.asks[0][1] == "write something" and b.steers[0][1] == "about agents"


async def test_interrupt_while_asking_closes_question(agent):
    agent.model.reply = '{"folder": "Blog", "session": "new", "confidence": 0.3}'
    (agent.paths.home / "Blog").mkdir()
    agent.sessions = [SimpleNamespace(cwd=str(agent.paths.home / "Blog"), last_modified=time.time() * 1000,
                                      session_id="b1", custom_title="", summary="")]
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "x"})
    await c.event("question")
    assert (await c.call({"op": "interrupt", "conv": CHAT}))["data"] == {"stopped": True}
    assert (await c.event("question_closed"))["outcome"] == "skipped"
    assert (await c.event("done"))["data"]["interrupted"] is True
    assert (await c.call({"op": "status", "conv": CHAT}))["data"].get("question") is None


async def test_confirm_expires_and_follow_up_text_is_ordinary(agent):
    agent.scripts[CHAT] = lambda t: [confirm_step()]
    conv = agent.svc.conversation(CHAT)
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "email"})
    await c.event("confirm")
    conv._expire(conv.question.question_id)
    assert (await c.event("question_closed"))["outcome"] == "expired"
    assert (await c.call({"op": "reply", "conv": CHAT, "text": "yes"}))["data"] == {"bound": False}
    agent.brain(CHAT).gate.resolve(False)
    await c.event("done")
    assert agent.brain(CHAT).confirm_results == [False]


def test_steps_have_no_em_dashes():
    import agentd.router as r
    import agentd.service as sv
    import agentd.workspaces as w

    for mod in (r, sv, w):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "—" not in src and "–" not in src, mod.__name__
    assert fb  # FakeBrain is the brain under test
