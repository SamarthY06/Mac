"""Agentic-manager findings for agentd (docs/reviews/agentic-manager.md, section A) and the brain
policy agentd hands each profile: switch_workspace (AG1), never-list (AG7), shared chats, the
model's confirm question, and the voice rename / reset_voice tools."""

from __future__ import annotations

import asyncio

import pytest
from agentd_kit import (
    CHAT,
    CHAT_TOKEN,
    NEVER_CHAT,
    OTHER_CHAT,
    confirm_step,
    slow,
    write_chat_config,
)

from agentd import brains
from agentd import schema as s
from macd.brain import fake_brain as fb
from macd.contracts import BrainEventKind

V = s.VOICE_CONV


def switch(folder: str, mode: str = "new") -> fb.Step:
    return fb.Step(BrainEventKind.STATUS, data={"switch_workspace": {"folder": folder, "mode": mode,
                                                                     "tools_run": 0}})


def first_switches(folder: str, mode: str = "new"):
    n = [0]

    def script(text: str) -> list[fb.Step]:
        n[0] += 1
        if n[0] == 1:
            return [fb.text("Looking."), switch(folder, mode), *slow("old", n=50, delay=0.02)]
        return [fb.text(f"Done in run {n[0]}.")]

    return script


async def collect(c, until: str = "done") -> list[dict]:
    out = []
    while True:
        e = await c.event(timeout=5)
        out.append(e)
        if e["ev"] == until:
            return out


# ----------------------------------------------------------------------------- AG1


async def test_switch_workspace_moves_and_reruns_the_request(agent):
    proj = agent.paths.home / "Resume"
    proj.mkdir()
    agent.mdfind_out = f"{proj}\n"
    agent.scripts[CHAT] = first_switches("Resume")
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "update my CV"})
    evs = await collect(c)
    texts = [e["text"] for e in evs if e["ev"] == "text"]
    assert texts == ["Looking.", "Done in run 2."]  # nothing of the moving run after the switch
    assert not any(e["ev"] == "say" for e in evs)
    folders = [e["data"]["folder"] for e in evs if e["ev"] == "status" and "folder" in e.get("data", {})]
    assert folders[-1] == "~/Resume"
    assert [e for e in evs if e["ev"] == "done"] == [evs[-1]] and not evs[-1]["data"].get("interrupted")
    assert len({e["turn_uid"] for e in evs if "turn_uid" in e}) == 1  # one turn for the client
    first, second = agent.brains[CHAT]
    assert first.interrupts and second.spec.cwd == proj and second.asks[0][1] == "update my CV"
    assert agent.svc.convs[CHAT].record.folder == str(proj)


def switch_after_work(folder: str):
    n = [0]

    def script(text: str) -> list[fb.Step]:
        n[0] += 1
        if n[0] == 1:
            return [*fb.tool("Read"), fb.Step(BrainEventKind.STATUS, data={"switch_workspace": {
                "folder": folder, "mode": "new", "tools_run": 1}}), fb.text("Stopping here.", delay=0.05)]
        return [fb.text(f"Done in run {n[0]}.")]

    return script


async def test_switch_after_work_ran_asks_first_and_yes_moves(agent):
    proj = agent.paths.home / "Resume"
    proj.mkdir()
    agent.mdfind_out = f"{proj}\n"
    agent.scripts[CHAT] = switch_after_work("Resume")
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "update my CV"})
    q = await c.event("question")
    assert q["kind"] == "route" and q["text"] == "I already started on this here. Move it to ~/Resume and start over there?"
    assert (await c.call({"op": "reply", "conv": CHAT, "text": "yes"}))["data"]["bound"] is True
    evs = await collect(c)
    assert [e["text"] for e in evs if e["ev"] == "text"] == ["Done in run 2."]
    assert agent.brain(CHAT).spec.cwd == proj


async def test_switch_after_work_ran_and_the_user_stays(agent):
    proj = agent.paths.home / "Resume"
    proj.mkdir()
    agent.mdfind_out = f"{proj}\n"
    agent.scripts[CHAT] = switch_after_work("Resume")
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "update my CV"})
    await c.event("question")
    await c.call({"op": "reply", "conv": CHAT, "text": "no"})
    done = await c.event("done")
    assert not done["data"].get("interrupted")  # the run ended on its own; nothing re-ran
    assert not agent.brain(CHAT).interrupts
    assert len(agent.brains[CHAT]) == 1 and len(agent.brain(CHAT).asks) == 1
    assert agent.svc.convs[CHAT].record.folder != str(proj)


async def test_switch_question_free_text_stays_and_starts_as_the_next_request(agent):
    """P2-1: "yes, but in Notes" is not a yes; it stays, then runs as a new ask with context."""
    proj = agent.paths.home / "Resume"
    proj.mkdir()
    agent.mdfind_out = f"{proj}\n"
    agent.scripts[CHAT] = switch_after_work("Resume")
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "update my CV"})
    await c.event("question")
    await c.call({"op": "reply", "conv": CHAT, "text": "yes, but put it in Notes"})
    await c.event("done")
    await c.event("done", timeout=5)  # the follow-up request ran too
    asks = [text for b in agent.brains[CHAT] for _, text in b.asks]
    assert asks[-1].endswith("yes, but put it in Notes") and '"update my CV"' in asks[-1]
    assert agent.svc.convs[CHAT].record.folder != str(proj)


async def test_switch_workspace_to_an_unknown_folder_is_refused_and_the_turn_carries_on(agent):
    agent.mdfind_out = ""
    agent.scripts[CHAT] = lambda t: [switch("Nowhere"), fb.text("Here then.", delay=0.1)]
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "x"})
    evs = await collect(c)
    assert [e["text"] for e in evs if e["ev"] == "text"] == ["Here then."]
    b = agent.brain(CHAT)
    assert len(agent.brains[CHAT]) == 1 and not b.interrupts
    assert b.steers and b.steers[0][1].startswith("Mac did not move this request: there is no folder called Nowhere")


async def test_switch_after_the_brain_already_ended_still_moves(agent):
    """The DONE can beat the folder check: it is held, then the request moves."""
    proj = agent.paths.home / "Blog"
    proj.mkdir()
    agent.mdfind_out = f"{proj}\n"
    n = [0]

    def script(text: str) -> list[fb.Step]:
        n[0] += 1
        return [switch("Blog")] if n[0] == 1 else [fb.text("Blogged.")]

    agent.scripts[CHAT] = script
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "post it"})
    evs = await collect(c)
    assert [e["text"] for e in evs if e["ev"] == "text"] == ["Blogged."]
    assert agent.brain(CHAT).spec.cwd == proj


async def test_a_request_moves_once_and_stop_still_works_while_moving(agent):
    proj = agent.paths.home / "Resume"
    proj.mkdir()
    agent.mdfind_out = f"{proj}\n"
    agent.scripts[CHAT] = lambda t: [switch("Resume"), *slow("w", n=100, delay=0.02)]
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "x"})
    await c.event("status", pred=lambda e: e.get("data", {}).get("folder") == "~/Resume")
    await asyncio.sleep(0.1)
    second = agent.brain(CHAT)
    assert any("already moved once" in txt for _, txt in second.steers)
    await c.call({"op": "interrupt", "conv": CHAT})
    assert (await c.event("done"))["data"]["interrupted"] is True


async def test_voice_never_switches(agent):
    agent.scripts[V] = lambda t: [switch("Resume"), fb.text("ok", delay=0.05)]
    c = await agent.client()
    await c.call({"op": "ask", "conv": V, "turn": 1, "text": "x"})
    await c.event("done")
    assert len(agent.brains[V]) == 1 and not agent.brain(V).interrupts


# ----------------------------------------------------------------------------- AG7, shared chats


async def test_never_list_names_fixed_plus_config_and_only_for_chat_brains(agent):
    write_chat_config(agent.paths.mac_home, [CHAT, OTHER_CHAT, NEVER_CHAT], [NEVER_CHAT],
                      'never_chat_names = ["Family"]\n')
    c = await agent.client(CHAT_TOKEN, [CHAT])  # the hello re-reads the never-set
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "hi"})
    await c.event("done")
    assert agent.brain(CHAT).spec.never_chat_names == ("Timetable", "Family")
    v = await agent.client()
    await v.call({"op": "ask", "conv": V, "turn": 1, "text": "hi"})
    await v.event("done")
    assert agent.brain(V).spec.never_chat_names == ()  # V6: by voice any chat, behind a spoken confirm


async def test_shared_chat_and_confirm_files_come_from_the_profile(agent):
    channel, _, cid = CHAT.partition(":")
    await agent.svc.profiles.update(channel, cid, {"shared": True, "confirm_files": False})
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "hi"})
    await c.event("done")
    spec = agent.brain(CHAT).spec
    assert spec.chat_shared is True and spec.confirm_send_file() is False  # P6: confirm_files, never shared
    await agent.svc.profiles.update(channel, cid, {"confirm_files": True})
    assert spec.confirm_send_file() is True  # live: read per send, no new brain
    assert spec.capabilities() == frozenset(s.CAPABILITIES)


def _spy_factory(monkeypatch, names):
    import inspect

    from macd import config
    from macd.brain import claude_brain as cb

    seen: list[dict] = []

    class Spy:
        def __init__(self, cfg, gate, **kw):
            seen.append({**kw, "_chrome": cfg.brain.chrome})

    params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
              inspect.Parameter("cfg", inspect.Parameter.POSITIONAL_OR_KEYWORD),
              inspect.Parameter("gate", inspect.Parameter.POSITIONAL_OR_KEYWORD),
              *(inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=None) for n in names)]
    Spy.__init__.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    monkeypatch.setattr(cb, "ClaudeBrain", Spy)
    return brains.claude_factory(config.Config()), seen


def _spec(profile, **kw):
    from pathlib import Path

    return brains.BrainSpec(conv="c", profile=profile, cwd=Path("/tmp"), setting_sources=["user"], gate=None,
                            session_store=None, memory_file=None, desktop=None, recent_user_text=list, **kw)


ALL_ARGS = [*brains.CLAUDE_ARGS, *brains.CHAT_SAFETY_ARGS, "on_alive", "on_rename", "on_reset_voice",
            "on_go_to_sleep", "chat_persona", "display_name", "chat_shared", "on_set_persona", "channel",
            "switch_workspace", "send_file", "whatsapp_ui_block", "snapshot"]


def test_factory_passes_profile_policy_to_claude_brain(monkeypatch):
    """Chat brains get the live capabilities, confirm_send_file, the never-list and the persona;
    voice brains get none of them, only the voice callbacks."""
    from macd.brain import claude_brain as cb

    make, seen = _spy_factory(monkeypatch, sorted({*ALL_ARGS, *cb.CHAT_PROFILE}))

    def caps():
        return frozenset({"web"})

    def alive(_a):
        return None

    def alive_true():
        return True

    def rename(_n):
        return None

    make(_spec(brains.CHAT, never_chat_names=("Timetable",), capabilities=caps, confirm_send_file=alive_true,
               chat_persona="Reply in Hindi", display_name="Book club", chat_shared=True, chrome=False,
               on_alive=alive, on_rename=rename))
    make(_spec(brains.VOICE, never_chat_names=("Timetable",), on_alive=alive, on_rename=rename,
               on_go_to_sleep=alive))
    chat, voice = seen
    assert chat["capabilities"] is caps and chat["confirm_send_file"] is alive_true
    assert chat["never_chat_names"] == ["Timetable"] and chat["chat_persona"] == "Reply in Hindi"
    assert chat["display_name"] == "Book club" and chat["chat_shared"] is True and chat["_chrome"] is False
    assert chat["on_alive"] is alive and "on_rename" not in chat
    assert chat["channel"] == "chat" and chat["switch_workspace"] is True
    assert chat["snapshot"] is False and "snapshot" not in voice  # W10.7: persona edits reach a resumed session
    for k in ("never_chat_names", "capabilities", "confirm_send_file", "channel", "chat_persona"):
        assert k not in voice
    assert voice["on_rename"] is rename and voice["on_alive"] is alive and voice["on_go_to_sleep"] is alive
    assert "on_reset_voice" not in voice  # not given: not passed


@pytest.mark.parametrize("missing", brains.CHAT_SAFETY_ARGS)
def test_chat_brain_refused_when_claude_brain_lacks_a_safety_argument(monkeypatch, missing):
    """Fail closed: a dropped `capabilities` would mean everything is allowed."""
    from macd.brain import claude_brain as cb

    with pytest.raises(RuntimeError, match=missing):  # at factory start or at the chat brain build
        make, _ = _spy_factory(monkeypatch, sorted({*ALL_ARGS, *cb.CHAT_PROFILE} - {missing}))
        make(_spec(brains.CHAT, never_chat_names=("Timetable",), capabilities=lambda: frozenset(),
                   confirm_send_file=lambda: True))


def test_extra_cannot_replace_a_safety_argument(monkeypatch):
    """N5: `spec.extra` goes in first, so the policy below always wins."""
    from macd.brain import claude_brain as cb

    make, seen = _spy_factory(monkeypatch, sorted({*ALL_ARGS, *cb.CHAT_PROFILE}))

    def caps():
        return frozenset()

    make(_spec(brains.CHAT, never_chat_names=("Timetable",), capabilities=caps, confirm_send_file=lambda: True,
               extra={"capabilities": None, "never_chat_names": [], "confirm_send_file": None}))
    assert seen[0]["capabilities"] is caps and seen[0]["never_chat_names"] == ["Timetable"]
    assert seen[0]["confirm_send_file"]() is True


def test_chat_brain_refused_without_its_policy(monkeypatch):
    from macd.brain import claude_brain as cb

    make, _ = _spy_factory(monkeypatch, sorted({*ALL_ARGS, *cb.CHAT_PROFILE}))
    with pytest.raises(RuntimeError):
        make(_spec(brains.CHAT, never_chat_names=("Timetable",)))  # no capabilities callable
    with pytest.raises(RuntimeError):
        make(_spec(brains.CHAT, capabilities=lambda: frozenset(), confirm_send_file=lambda: True))  # no never-list
    with pytest.raises(RuntimeError):  # no live confirm_send_file
        make(_spec(brains.CHAT, never_chat_names=("Timetable",), capabilities=lambda: frozenset()))


# ----------------------------------------------------------------------------- confirm, voice tools


async def test_confirm_carries_the_models_question(agent):
    step = confirm_step()
    step.data["question"] = "Send the draft to Bob?"
    agent.scripts[V] = lambda t: [step] if t == "send it" else [confirm_step()]
    c = await agent.client()
    await c.call({"op": "ask", "conv": V, "turn": 1, "text": "send it"})
    ev = await c.event("confirm")
    assert ev["question"] == "Send the draft to Bob?" and ev["summary"] == "Send the email?"
    st = (await c.call({"op": "status", "conv": V}))["data"]["question"]
    assert st["kind"] == "confirm" and st["question"] == "Send the draft to Bob?"
    assert st["target"] == "bob@example.com"
    await c.call({"op": "confirm", "conv": V, "question_id": ev["question_id"], "approved": False})
    await c.event("done")
    await c.call({"op": "ask", "conv": V, "turn": 2, "text": "again"})
    ev = await c.event("confirm")
    assert ev.get("question") is None  # absent: the client falls back to a neutral question
    st = (await c.call({"op": "status", "conv": V}))["data"]["question"]
    assert st.get("question") is None and st["target"] == "bob@example.com"


async def test_voice_rename_and_reset_voice_tools_push_status(agent):
    c = await agent.client()
    await c.call({"op": "warm", "conv": V})
    b = agent.brain(V)
    assert agent.svc.convs[V].brain is b
    assert b.spec.on_rename("Jarvis") is None
    assert (await c.event("status", pred=lambda e: "rename" in e.get("data", {})))["data"] == {
        "rename": {"name": "Jarvis"}}
    assert b.spec.on_reset_voice() is None
    assert (await c.event("status", pred=lambda e: "reset_voice" in e.get("data", {})))["data"] == {
        "reset_voice": {}}
    assert b.spec.on_go_to_sleep() is None
    assert (await c.event("status", pred=lambda e: "go_to_sleep" in e.get("data", {})))["data"] == {
        "go_to_sleep": {}}
    chat = await agent.client(CHAT_TOKEN, [CHAT])
    await chat.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "hi"})
    await chat.event("done")
    cs = agent.brain(CHAT).spec
    assert cs.on_rename is None and cs.on_reset_voice is None and cs.on_go_to_sleep is None
