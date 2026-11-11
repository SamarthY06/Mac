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


def q(kind: K, ref: str = "p1", options: int = 2, yes: int | None = 0) -> Question:
    opts = [{"label": "Resume", "keys": ["resume", "1"]}, {"label": "generally", "keys": ["general", "2"]}]
    return Question("q1", kind, "?", NOW + 60, prompt_ref=ref, payload={"options": opts[:options], "yes": yes})


def reply(**kw) -> s.Reply:
    return s.Reply(conv=CHAT, **kw)


@pytest.mark.parametrize("text,want", [("yes", "yes"), ("Yes!", "yes"), ("send it", "yes"), ("go ahead.", "yes"),
                                       ("\U0001F44D", "yes"), ("\U0001F44D\U0001F3FD", "yes"), ("no", "no"),
                                       ("Cancel", "no"), ("don’t", "no"), ("yes, but to Ann", None),
                                       ("maybe", None)])
def test_yes_no(text, want):
    assert yes_no(text) == want


@pytest.mark.parametrize("text,want", [("Yes, go ahead", "yes"), ("yes send it", "yes"), ("sure, thanks", "yes"),
                                       ("ok do it", "yes"), ("absolutely", "yes"), ("no not now", "no"),
                                       ("not yet", "no"), ("yes, but to Ann", None), ("hmm", None),
                                       ("no, send it to Bob", None)])
def test_yes_no_widened_reuses_the_voice_rule(text, want):
    """AG6: confirm, catch-up and restart take natural yeses (deterministic, macd.phrases)."""
    assert yes_no(text, strict=False) == want


def test_widened_yes_binds_confirm_but_never_folder_approval():
    assert bind(q(K.CONFIRM), reply(text="Yes, go ahead"), NOW).outcome is s.ReplyOutcome.APPROVED
    assert bind(q(K.CONFIRM), reply(text="sure, thanks"), NOW).outcome is s.ReplyOutcome.APPROVED
    assert bind(q(K.RESTART), reply(text="ok do it"), NOW).outcome is s.ReplyOutcome.APPROVED
    assert not bind(q(K.FOLDER_APPROVAL), reply(text="Yes, go ahead", quoted_ref="p1"), NOW).bound


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


def test_folder_approval_takes_an_exact_yes_or_no():
    assert bind(q(K.FOLDER_APPROVAL), reply(text="yes"), NOW).outcome is s.ReplyOutcome.APPROVED
    assert bind(q(K.FOLDER_APPROVAL), reply(text="no"), NOW).outcome is s.ReplyOutcome.DENIED
    assert not bind(q(K.FOLDER_APPROVAL), reply(text="sure thing buddy"), NOW).bound
    assert not bind(q(K.PROFILE), reply(text="yes"), NOW).bound  # profile changes stay quote-only
    assert bind(q(K.FOLDER_APPROVAL), reply(text="yes", quoted_ref="p1"), NOW).outcome is s.ReplyOutcome.APPROVED
    assert bind(q(K.FOLDER_APPROVAL), reply(reaction="\U0001F44D", reaction_on="p1"), NOW).bound
    assert not bind(q(K.FOLDER_APPROVAL), reply(text="sure thing buddy", quoted_ref="p1"), NOW).bound


def test_route_catch_up_restart_outcomes():
    b = bind(q(K.ROUTE), reply(text="Resume."), NOW)
    assert b.outcome is s.ReplyOutcome.ANSWERED and b.choice == 0
    b = bind(q(K.ROUTE), reply(text="do it in resume"), NOW)  # P2-1: not the whole reply, so the router reads it
    assert b.outcome is s.ReplyOutcome.ANSWERED and b.choice is None and b.steer_text == "do it in resume"
    for text in ("yes, but in Notes", "no, use the general one", "1 and also 2"):
        assert bind(q(K.ROUTE), reply(text=text), NOW).choice is None
    assert bind(q(K.ROUTE), reply(text="no"), NOW).choice == 1
    assert bind(q(K.CATCH_UP), reply(text="yes"), NOW).choice == 0
    assert bind(q(K.RESTART), reply(text="no"), NOW).outcome is s.ReplyOutcome.DENIED
    # AG4: free text closes route, catch-up and restart questions as an answer for the model
    for kind in (K.ROUTE, K.CATCH_UP, K.RESTART):
        b = bind(q(kind), reply(text="the job one"), NOW)
        assert b.bound and b.outcome is s.ReplyOutcome.ANSWERED and b.choice is None
        assert b.steer_text == "the job one"
    assert not bind(q(K.CONFIRM, options=0), reply(reaction="\U0001F44D", reaction_on="x"), NOW).bound


def test_route_yes_binds_to_what_the_question_asked():
    """AG3: "Start fresh, or pick up the job search?" with yes meaning fresh (option 1)."""
    assert bind(q(K.ROUTE, yes=1), reply(text="yes"), NOW).choice == 1
    assert bind(q(K.ROUTE, yes=1), reply(text="no"), NOW).choice == 0
    no_yes = q(K.ROUTE, yes=None)  # the model did not say: only a number or label binds
    b = bind(no_yes, reply(text="yes"), NOW)
    assert b.choice is None and b.steer_text == "yes"
    assert bind(no_yes, reply(text="2"), NOW).choice == 1
    assert bind(no_yes, reply(text=" General! "), NOW).choice == 1
    assert bind(no_yes, reply(text="general please"), NOW).choice is None  # P2-1: free text for the router


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


async def test_catch_up_is_the_models_call_with_the_time_it_was_sent(agent):
    """AG5: no fixed 60-minute question; the model is told when the message was sent and decides."""
    c = await agent.client(CHAT_TOKEN, [CHAT])
    meta = {"replayed": True, "sent_at": time.time() - 7200, "msg_ref": "m1"}
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "find me 20 jobs", "meta": meta})
    await c.event("done")
    assert [e for e in c.drain() if e.get("ev") == "question"] == []
    text = agent.brain(CHAT).asks[0][1]
    assert text.startswith("[Note from Mac: the user sent this at ") and "2 h ago, while Mac was asleep" in text
    assert text.endswith("]\n\nfind me 20 jobs")
    await c.call({"op": "ask", "conv": CHAT, "turn": 2, "text": "and 5 more",
                  "meta": {"replayed": True, "sent_at": time.time() - 30}})
    await c.event("done")
    assert agent.brain(CHAT).asks[-1][1] == "and 5 more"  # a fresh replay needs no note


async def test_route_question_answer_picks_folder(agent):
    proj = agent.paths.home / "Blog"
    proj.mkdir()
    agent.sessions = [SimpleNamespace(cwd=str(proj), last_modified=time.time() * 1000, session_id="b1",
                                      custom_title="Post", summary="")]
    agent.model.reply = '{"folder": "Blog", "session": "new", "confidence": 0.3}'
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "write something"})
    qe = await c.event("question")
    assert qe["kind"] == "route" and qe["text"] == "Should I do this in Blog? Say no to do it generally."
    # a steer while asking is kept for the brain; a reaction on something else does not bind
    assert (await c.call({"op": "steer", "conv": CHAT, "text": "about agents"}))["data"] == {"accepted": True}
    assert (await c.call({"op": "reply", "conv": CHAT, "reaction": "\U0001F44D", "reaction_on": "x"})
            )["data"] == {"bound": False}
    r = await c.call({"op": "reply", "conv": CHAT, "text": "in blog"})
    assert r["data"] == {"bound": True, "outcome": "answered"}
    await c.event("done")
    b = agent.brain(CHAT)
    assert b.spec.cwd == proj and b.asks[0][1] == "write something" and b.steers[0][1] == "about agents"


async def test_free_text_route_answer_goes_back_to_the_model(agent):
    """AG4: "the blog one" closes the route question at once; the model decides with it."""
    proj = agent.paths.home / "Blog"
    proj.mkdir()
    agent.sessions = [SimpleNamespace(cwd=str(proj), last_modified=time.time() * 1000, session_id="b1",
                                      custom_title="Post", summary="")]
    agent.model.reply = '{"folder": "Blog", "session": "new", "confidence": 0.3, "ask": "Blog or not?"}'
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "write something"})
    qe = await c.event("question")
    assert qe["text"] == "Blog or not? (1 Blog, 2 generally)"
    r = await c.call({"op": "reply", "conv": CHAT, "text": "the one about my posts"})
    assert r["data"] == {"bound": True, "outcome": "answered"}
    assert (await c.event("question_closed"))["outcome"] == "answered"
    await c.event("done")
    assert agent.brain(CHAT).spec.cwd == proj
    assert "They answered: 'the one about my posts'" in agent.model.calls[-1]


async def test_free_text_restart_answer_is_a_new_ask_with_context(short_root):
    """AG4: a restart question answered in words runs those words, with the old request as context."""
    from agentd_kit import make_agent, stop_agent

    a = await make_agent(short_root, scripts={CHAT: lambda t: slow(n=20, delay=0.05)})
    c = await a.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "find me 20 jobs"})
    await c.event("say")
    await stop_agent(a)
    a = await make_agent(short_root, boot="b2", scripts={CHAT: lambda t: [fb.text("Found.")]})
    try:
        c = await a.client(CHAT_TOKEN, [CHAT])
        st = (await c.call({"op": "status", "conv": CHAT}))["data"]
        assert st["question"]["kind"] == "restart"
        r = await c.call({"op": "reply", "conv": CHAT, "text": "just the remote ones please"})
        assert r["data"] == {"bound": True, "outcome": "answered"}
        await c.event("done")
        text = a.brain(CHAT).asks[0][1]
        assert 'working on "find me 20 jobs"' in text and text.endswith("\n\njust the remote ones please")
    finally:
        await stop_agent(a)


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
