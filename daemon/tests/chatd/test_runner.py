"""ChatRunner against a real agentd (Service + Server from the agentd test kit, FakeBrain scripts)
and the fake channel: the W5.5 timeline, budgets, commands, questions, catch-up, the loop breaker,
restart notices and the stream.json resume paths (V3, B1)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from agentd_kit import (
    APP_TOKEN,
    CHAT,
    CHAT_TOKEN,
    OTHER_CHAT,
    Agent,
    confirm_step,
    make_agent,
    stop_agent,
)

from agentd import schema as s
from agentd.service import RESTART_TEXT
from chatd import format as fmt
from chatd.agent_client import AgentClient, AgentFatal, dev_token, read_lifeline_token
from chatd.channels.base import Cap, ChatChanged, ContextMessage, MessageRef
from chatd.channels.fake import FakeChannel
from chatd.channels.registry import ChannelContext, SeenIds
from chatd.context import Ring, resolve_mentions
from chatd.governor import Governor
from chatd.runner import ChatRunner, parse_command, strip_trigger
from macd.brain import fake_brain as fb
from macd.contracts import BrainEventKind

CHAT_ID = CHAT.split(":", 1)[1]
OTHER_ID = OTHER_CHAT.split(":", 1)[1]
DASHES = re.compile("[‒–—―]")
_CHANNELS: list[FakeChannel] = []


@pytest.fixture(autouse=True)
def _every_outbound_text_is_clean() -> Any:
    """Across every runner test: each text Mac sent or edited starts with "Mac · " and has no dash."""
    _CHANNELS.clear()
    yield
    for ch in _CHANNELS:
        for text in ch.texts():
            assert text.startswith(fmt.PREFIX), text
            assert not DASHES.search(text), text


def channel(root: Path, **cfg: Any) -> FakeChannel:
    ctx = ChannelContext(governor=Governor(root / "governor.json"), chats=frozenset({CHAT_ID}),
                         seen=SeenIds(root / "seen.json"))
    ch = FakeChannel({"name": "whatsapp", "min_edit_interval_s": 0.05, **cfg}, ctx)
    _CHANNELS.append(ch)
    return ch


async def agentd(root: Path, *, shared: bool = False, **kw: Any) -> Agent:
    """A real agentd; its seed gives CHAT and OTHER_CHAT enabled profiles (W9.10)."""
    agent = await make_agent(root, **kw)
    if shared:
        await agent.svc.profiles.update("whatsapp", CHAT_ID, {"shared": True})
    return agent


async def runner(agent: Agent, ch: FakeChannel, root: Path, **kw: Any) -> ChatRunner:
    await ch.ctx.seen.load()
    r = ChatRunner(ch, sock=agent.sock, token=CHAT_TOKEN, state_dir=root / "chat",
                   catch_up_window_s=kw.pop("catch_up_window_s", 0.2), status_after_s=kw.pop("status_after_s", 0.3),
                   backoff=(0.05,), **kw)
    await r.start()
    await until(r.client.connected.is_set)
    await r.settle()
    return r


async def until(pred: Callable[[], Any], timeout: float = 4.0) -> Any:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met in time")


def closed(prompt: Any, outcome: str) -> bool:
    """A closed question keeps its text, with the outcome below it."""
    return prompt.text == f"{prompt.history[0]}\n\n{fmt.strip_prefix(outcome)}"


def stream(root: Path) -> dict[str, Any]:
    return json.loads((root / "chat" / "stream.json").read_text())["convs"][CHAT]


@dataclass
class World:
    root: Path
    agent: Agent
    ch: FakeChannel
    r: ChatRunner

    def conv(self) -> Any:
        return self.agent.svc.convs[CHAT]

    def brain(self) -> Any:
        return self.agent.brain(CHAT)

    async def say(self, text: str = "", **kw: Any) -> str:
        mid = kw.pop("msg_id", None) or f"u{time.monotonic_ns()}"
        assert await self.ch.inject(text, chat_id=CHAT_ID, msg_id=mid, **kw)
        return mid

    async def done(self, mid: str, emoji: str = "✅") -> None:
        await until(lambda: self.ch.reactions.get(mid, [""])[-1] == emoji)
        await self.r.settle()

    async def restart_runner(self, **kw: Any) -> None:
        await self.r.close()
        self.r = await runner(self.agent, self.ch, self.root, **kw)


@pytest.fixture
async def world(short_root: Path) -> AsyncIterator[World]:
    agent = await agentd(short_root)
    ch = channel(short_root)
    w = World(short_root, agent, ch, await runner(agent, ch, short_root))
    yield w
    await w.r.close()
    await stop_agent(w.agent)


_FRAMED = re.compile(r"<(group_context|quoted_message)\b[^>]*>.*?</\1>\n?", re.DOTALL)


def request(text: str) -> str:
    """The owner's request inside an ask agentd framed: without the group context blocks (W10.9)."""
    return _FRAMED.sub("", text).strip()


def script(d: dict[str, list[fb.Step]]) -> Callable[[str], list[fb.Step] | None]:
    return lambda text: d.get(request(text))


def attach(path: Path, attach_id: str = "a1", delay: float = 0.0) -> fb.Step:
    return fb.Step(BrainEventKind.ATTACH, data={"attach_id": attach_id, "path": str(path), "caption": "jobs"},
                   delay=delay)


def tool(name: str, narration: str) -> list[fb.Step]:
    return [fb.Step(BrainEventKind.TOOL_START, tool_name=name, data={"narration": narration}),
            fb.Step(BrainEventKind.TOOL_END, tool_name=name, delay=0.02)]


# -- the timeline and the budget (W5.5, U27) --------------------------------------------------------------


async def test_timeline_status_edited_into_the_answer_then_file_then_check(world: World) -> None:
    f = world.root / "jobs.csv"
    f.write_text("company,role\n")
    world.agent.scripts[CHAT] = script({"find me 20 jobs": [
        fb.say("Searching LinkedIn and Indeed", delay=0.15),
        *tool("mcp__claude-in-chrome__navigate", "Opening LinkedIn"),
        fb.Step(BrainEventKind.TEXT, text="Let me read the log.", data={"tool_follows": True}),
        *tool("Read", "Reading applications_log.csv"),
        fb.say("12 so far", delay=0.15),
        attach(f),
        fb.text("Found 20 new AI engineer roles — none already in your log.", delay=0.1),
    ]})
    t0 = time.monotonic()
    req = await world.say("find me 20 jobs")
    await until(lambda: world.ch.reactions.get(req))
    assert time.monotonic() - t0 < 1.0 and world.ch.reactions[req] == ["👀"]  # 👀 under 1 s
    await world.done(req)

    sent = world.ch.sent()
    assert len(sent) == 2  # one status message, one file
    status, file = sent
    assert status.reply_to == req
    assert status.history[0] == "Mac · On it\nSearching LinkedIn and Indeed"  # no bare "On it" (UX1)
    assert any("Searching LinkedIn and Indeed" in h for h in status.history)
    assert any("12 so far" in h for h in status.history)
    assert status.text == "Mac · Found 20 new AI engineer roles, none already in your log."
    assert file.path == str(f) and file.caption == ""  # no caption
    ops = [(w.op, w.message_id) for w in world.ch.wire]
    last_edit = max(i for i, (op, m) in enumerate(ops) if op == "edit" and m == status.message_id)
    file_at = next(i for i, (op, _) in enumerate(ops) if op == "send_file")
    check_at = next(i for i, w in enumerate(world.ch.wire) if w.op == "react" and w.text == "✅")
    assert last_edit < file_at < check_at
    assert world.ch.reactions[req] == ["👀", "✅"]  # one reaction slot
    for text in world.ch.texts():  # where and what, never how
        for leak in ("Opening LinkedIn", "Reading applications_log", "navigate", "Read", "Let me read the log"):
            assert leak not in text
    await until(lambda: not world.conv().record.unacked)  # acked once applied


async def test_task_budget_with_a_confirmation(world: World) -> None:
    f = world.root / "resume.pdf"
    f.write_text("pdf")
    world.agent.scripts[CHAT] = script({"email the recruiter": [
        fb.say("Writing the email", delay=0.1), attach(f, "r1"),
        confirm_step("Ready to email Bob, resume attached. Send the email?", "bob@example.com")]})
    req = await world.say("email the recruiter")
    prompt = await until(lambda: next((m for m in world.ch.sent() if "Send the email?" in m.text), None))
    # the model's words plus the hook-derived target; a summary that asks already gets no fallback (UX4)
    assert prompt.text == "Mac · Ready to email Bob, resume attached. Send the email?\nTo: bob@example.com"
    await until(lambda: world.conv().question is not None and world.conv().question.prompt_ref == prompt.message_id)
    await world.say("yes", reply_to=prompt.message_id)
    await world.done(req)
    assert closed(prompt, fmt.APPROVED)  # never claims it was sent; the answer says what happened
    assert world.brain().confirm_results == [True]
    sent = world.ch.sent()
    statuses = [m for m in sent if m.reply_to == req]
    files = [m for m in sent if m.path]
    questions = [m for m in sent if m is prompt]
    # UX3: the status froze when the question went out; the answer lands below the question
    assert len(statuses) == 2 and len(files) == 1 and len(questions) == 1 and len(sent) == 4
    assert "Writing the email" in statuses[0].text and statuses[1].text == "Mac · sent"
    assert sent.index(prompt) < sent.index(statuses[1])
    assert world.ch.reactions[req] == ["👀", "✅"]


async def test_thumbs_up_on_the_prompt_approves_and_other_text_is_a_change(world: World) -> None:
    world.agent.scripts[CHAT] = script({"post it": [confirm_step("Post the update?", "linkedin.com")],
                                        "send it again": [confirm_step("Send it?", "x@y.z")]})
    req = await world.say("post it")
    prompt = await until(lambda: next((m for m in world.ch.sent() if "Post the update?" in m.text), None))
    await until(lambda: world.conv().question is not None and world.conv().question.prompt_ref)
    await world.say(reaction="👍", reply_to=prompt.message_id)
    await world.done(req)
    assert world.brain().confirm_results == [True]
    req2 = await world.say("send it again")
    p2 = await until(lambda: next((m for m in world.ch.sent() if "Send it?" in m.text), None))
    await until(lambda: world.conv().question is not None and world.conv().question.prompt_ref)
    await world.say("also attach the cover letter")  # neither yes nor no: change it (U11)
    await until(lambda: closed(p2, fmt.CHANGING))
    await world.done(req2)
    assert world.brain().confirm_results == [True, False]


async def test_quick_answer_without_progress_is_one_reply(world: World) -> None:
    req = await world.say("hi")  # default FakeBrain reply "Okay."
    await world.done(req)
    sent = world.ch.sent()
    assert [m.text for m in sent] == ["Mac · Okay."] and sent[0].reply_to == req


async def test_long_answer_becomes_a_summary_and_a_file(world: World) -> None:
    long = "\n".join(f"Line {i}: " + "x" * 80 for i in range(200))
    world.agent.scripts[CHAT] = script({"report": [fb.text(long)]})
    req = await world.say("report")
    await world.done(req)
    sent = world.ch.sent()
    assert len(sent) == 2 and sent[1].path.endswith(".md") and sent[1].caption == ""
    assert sent[0].text == fmt.prefixed(fmt.FULL_ANSWER)  # no invented summary (AG21)
    assert Path(sent[1].path).read_text() == long


async def test_error_is_one_line_and_cross(world: World) -> None:
    world.agent.scripts[CHAT] = script({"go": [fb.say("Working", delay=0.05),
                                               fb.error(fb.BrainErrorKind.AUTH, "not logged in", delay=0.1)]})
    req = await world.say("go")
    await world.done(req, "❌")
    assert world.ch.sent()[0].text == fmt.ERRORS["auth"]
    assert len(world.ch.sent()) == 1


# -- commands (W5.3) -------------------------------------------------------------------------------------------


def test_parse_commands_with_or_without_slash() -> None:
    assert parse_command("stop") == ("stop", "") == parse_command("/Stop.")
    assert parse_command("new chat") == ("new", "") == parse_command("start fresh")
    assert parse_command("/new chat in Resume") == ("new", "Resume")
    assert parse_command("What are you doing?") == ("status", "") == parse_command("/status")
    assert parse_command("help") == ("help", "") == parse_command("/help")
    assert parse_command("stop the music in the kitchen") is None
    for free_text in ("stop it", "start over", "how's it going?", "what's happening", "new"):
        assert parse_command(free_text) is None  # only the exact commands; the rest is the model's


async def test_help_status_new_and_stop_when_idle(world: World) -> None:
    for text in ("help", "/help"):
        mid = await world.say(text)
        await until(lambda m=mid: any(x.reply_to == m for x in world.ch.sent()))
    helps = [m for m in world.ch.sent() if m.text == fmt.HELP.format(folder=fmt.HELP_FOLDER)]
    assert len(helps) == 2 and len(fmt.HELP.splitlines()) == 5 and "new chat in Documents" in helps[0].text
    await world.say("status")
    await until(lambda: world.ch.sent()[-1].text == "Mac · Nothing running.")
    await world.say("stop")
    await until(lambda: world.ch.sent()[-1].text == fmt.NOTHING_TO_STOP)
    await world.say("new chat")
    await until(lambda: world.ch.sent()[-1].text == fmt.FRESH)
    assert world.conv().record.fresh_next
    await world.say("/new chat in Resume")
    await until(lambda: world.ch.sent()[-1].text == "Mac · Fresh start in Resume.")
    assert world.conv().record.fresh_folder == "Resume"
    assert not world.agent.brains.get(CHAT)  # no command reached a brain


async def test_mid_turn_steer_progress_status_new_and_stop(world: World) -> None:
    world.agent.scripts[CHAT] = script({"long task": [fb.say("Reading the job boards", delay=0.05),
                                                      *fb_slow(200)]})
    req = await world.say("long task")
    await until(lambda: any("Reading the job boards" in m.text for m in world.ch.sent()))
    steer = await world.say("only remote roles")
    await until(lambda: world.ch.reactions.get(steer) == ["➕"])
    assert world.brain().steers and world.brain().steers[-1][1] == "only remote roles"
    progress = await world.say("how's it going?")  # free text: a steer, answered by the model
    await until(lambda: world.ch.reactions.get(progress) == ["➕"])
    assert world.brain().steers[-1][1] == "how's it going?"
    await world.say("status")
    await until(lambda: world.ch.sent()[-1].text.startswith("Mac · Reading the job boards, "))  # AG24
    await world.say("new chat")
    await until(lambda: world.ch.sent()[-1].text == fmt.STOP_FIRST)
    await world.say("/stop")
    await world.done(req, "✋")
    status = next(m for m in world.ch.sent() if m.reply_to == req)
    assert status.text == fmt.STOPPED
    await until(lambda: world.conv().running() is None)
    await world.r.settle()
    assert status.text == fmt.STOPPED and world.ch.reactions[req] == ["👀", "✋"]


def fb_slow(n: int, delay: float = 0.02) -> list[fb.Step]:
    return [fb.Step(BrainEventKind.TOOL_START, tool_name="Bash", delay=delay) for _ in range(n)]


async def test_voice_note_gets_one_line_and_no_download(world: World) -> None:
    mid = await world.say(msg_id="v1", unsupported="voice")
    await until(lambda: world.ch.sent())
    await world.r.settle()
    assert [(m.text, m.reply_to) for m in world.ch.sent()] == [(fmt.VOICE_NOTE, mid)]
    assert world.ch.downloads == 0 and not world.agent.brains.get(CHAT)
    assert not await world.ch.inject(chat_id=CHAT_ID, msg_id="v1", unsupported="voice", replayed=True)  # handled


# -- catch-up and the loop breaker (W5.4) ------------------------------------------------------------------


async def test_catch_up_is_coalesced_into_one_ask(world: World) -> None:
    now = time.time()
    ids = [await world.say(t, replayed=True, sent_at=now - 600 + i) for i, t in
           enumerate(["find jobs", "in London", "remote is fine too"])]
    await world.done(ids[-1])
    assert len(world.brain().asks) == 1
    text = world.brain().asks[0][1]
    assert "remote is fine too" in text and '"find jobs"' in text and '"in London"' in text
    assert world.ch.reactions == {ids[-1]: ["👀", "✅"]}
    for mid in ids:  # every one was marked handled, so none comes back
        assert not await world.ch.inject("x", chat_id=CHAT_ID, msg_id=mid, replayed=True)


async def test_catch_up_older_than_an_hour_is_one_ask_the_model_decides(world: World) -> None:
    old = time.time() - 7200
    ids = [await world.say(t, replayed=True, sent_at=old + i) for i, t in enumerate(["a", "b", "find me 20 jobs"])]
    await world.done(ids[-1])
    assert len(world.brain().asks) == 1 and "find me 20 jobs" in world.brain().asks[0][1]
    assert [m.text for m in world.ch.sent()] == ["Mac · Okay."]  # no canned catch-up wording from chatd


async def test_loop_breaker_pauses_once(world: World) -> None:
    req = await world.say("hi")
    await world.done(req)
    for i in range(6):
        await world.say("Okay.", msg_id=f"loop{i}")
        await world.r.settle()
        await until(lambda: world.conv().running() is None)
    await until(lambda: world.conv().record.paused == "loop")
    assert [m.text for m in world.ch.sent()].count(fmt.LOOP_PAUSED) == 1
    asked = len(world.brain().asks) + len(world.brain().steers)
    await world.say("Okay.")
    await world.r.settle()
    assert len(world.brain().asks) + len(world.brain().steers) == asked
    assert [m.text for m in world.ch.sent()].count(fmt.LOOP_PAUSED) == 1


async def test_members_changed_pauses_the_chat(world: World) -> None:
    await world.ch.change_chat(ChatChanged(CHAT_ID, "Interview Prep", owner_only=False))
    await until(lambda: world.conv().record.paused == "members_changed")


async def test_hello_and_goodbye_once_per_marker(world: World) -> None:
    assert await world.r.say_once(CHAT_ID, f"hello:{CHAT_ID}:0", fmt.HELLO)
    assert not await world.r.say_once(CHAT_ID, f"hello:{CHAT_ID}:0", fmt.HELLO)
    await world.restart_runner()
    assert not await world.r.say_once(CHAT_ID, f"hello:{CHAT_ID}:0", fmt.HELLO)
    assert await world.r.say_once(CHAT_ID, f"goodbye:{CHAT_ID}:0", fmt.GOODBYE)
    assert not await world.r.say_once(CHAT_ID, f"goodbye:{CHAT_ID}:0", fmt.GOODBYE)
    assert [m.text for m in world.ch.sent()] == [fmt.HELLO, fmt.GOODBYE]


# -- restarts: stream.json, B1, V3 ----------------------------------------------------------------------------


async def test_resume_keeps_editing_the_same_status_message(world: World) -> None:
    steps = [fb.say(f"Step {i}", delay=0.1) for i in range(12)] + [fb.text("All done.")]
    world.agent.scripts[CHAT] = script({"task": steps})
    req = await world.say("task")
    await until(lambda: any("Step 1" in m.text for m in world.ch.sent()))
    st = stream(world.root)
    uid = world.conv().record.running.uid
    status = world.ch.sent()[0]
    assert st["running_turn_uid"] == uid and st["status_msg_ref"] == status.message_id
    assert st["boot"] == world.agent.svc.boot and st["last_applied_seq"] > 0
    await world.r.close()  # chatd goes away mid-turn; the turn keeps running (B1)
    await asyncio.sleep(0.4)
    edits_before = len(status.history)
    await world.restart_runner()
    await world.done(req)
    assert len(status.history) > edits_before  # progress edits resumed on the same message
    assert status.text == "Mac · All done." and len(world.ch.sent()) == 1
    assert sum(1 for h in status.history if h == "Mac · All done.") == 1  # the final events applied once
    assert stream(world.root)["running_turn_uid"] == ""


async def test_stop_after_a_chatd_restart_interrupts_the_running_turn(world: World) -> None:
    world.agent.scripts[CHAT] = script({"long": [fb.say("Working", delay=0.05), *fb_slow(300)]})
    req = await world.say("long")
    await until(lambda: any("Working" in m.text for m in world.ch.sent()))
    await world.restart_runner()
    await world.say("stop")
    await world.done(req, "✋")
    await until(lambda: world.conv().running() is None)
    assert world.brain().interrupts
    assert next(m for m in world.ch.sent() if m.reply_to == req).text == fmt.STOPPED


async def test_question_opened_while_chatd_was_down_is_posted_and_answerable(world: World) -> None:
    world.agent.scripts[CHAT] = script({"email": [fb.say("Drafting", delay=0.05),
                                                  fb.Step(BrainEventKind.SAY, text="Almost", delay=0.4),
                                                  confirm_step("Send it to Bob?", "bob@example.com")]})
    req = await world.say("email")
    await until(lambda: any("Drafting" in m.text for m in world.ch.sent()))
    await world.r.close()
    await until(lambda: world.conv().question is not None)
    await world.restart_runner()
    prompt = await until(lambda: next((m for m in world.ch.sent() if "Send it to Bob?" in m.text), None))
    await until(lambda: world.conv().question is not None and world.conv().question.prompt_ref == prompt.message_id)
    await world.say("yes")
    await world.done(req)
    assert world.brain().confirm_results == [True]
    assert closed(prompt, fmt.APPROVED)  # never claims it was sent; the answer says what happened


async def test_question_from_status_after_a_gap_is_posted(world: World) -> None:
    world.agent.scripts[CHAT] = script({"email": [fb.say("Drafting", delay=0.05),
                                                  confirm_step("Send it to Bob?", "bob@example.com")]})
    await world.say("email")
    await until(lambda: any("Send it to Bob?" in m.text for m in world.ch.sent()))
    await world.r.close()
    _tamper(world.root, boot="0000000000000000", questions={})  # unknown boot: gap, prompt record lost
    await world.restart_runner()
    prompts = await until(lambda: [m for m in world.ch.sent() if "Send it to Bob?" in m.text][1:] or None)
    await until(lambda: world.conv().question.prompt_ref == prompts[-1].message_id)
    await world.say("no", reply_to=prompts[-1].message_id)
    await until(lambda: closed(prompts[-1], fmt.DECLINED))


def _tamper(root: Path, **fields: Any) -> None:
    p = root / "chat" / "stream.json"
    raw = json.loads(p.read_text())
    raw["convs"][CHAT].update(fields)
    p.write_text(json.dumps(raw))


async def test_gap_rebuilds_from_unacked_and_acks(world: World) -> None:
    f = world.root / "list.csv"
    f.write_text("x")
    world.agent.scripts[CHAT] = script({"send file": [fb.say("Collecting", delay=0.05), attach(f, "a1", delay=0.3),
                                                      fb.text("Here it is.")]})
    req = await world.say("send file")
    await until(lambda: any("Collecting" in m.text for m in world.ch.sent()))
    await world.r.close()
    await until(lambda: world.conv().record.unacked)
    _tamper(world.root, boot="0000000000000000")
    await world.restart_runner()
    await world.done(req)
    sent = world.ch.sent()
    assert sent[0].text == "Mac · Here it is." and len([m for m in sent if m.path == str(f)]) == 1
    await until(lambda: not world.conv().record.unacked)
    assert stream(world.root)["sent_attach_ids"] == ["a1"]


async def test_attach_ids_are_never_sent_twice(world: World) -> None:
    f = world.root / "list.csv"
    f.write_text("x")
    world.agent.scripts[CHAT] = script({"send file": [fb.say("Collecting", delay=0.05), attach(f, "a1", delay=0.3),
                                                      fb.text("Here it is.")]})
    req = await world.say("send file")
    await until(lambda: any("Collecting" in m.text for m in world.ch.sent()))
    await world.r.close()
    await until(lambda: world.conv().record.unacked)
    _tamper(world.root, boot="0000000000000000", sent_attach_ids=["a1"])  # it went out before a crash
    await world.restart_runner()
    await world.done(req)
    assert not [m for m in world.ch.sent() if m.path]


async def test_restart_notice_once_and_the_answer_binds_to_it(short_root: Path) -> None:
    calls = {"n": 0}

    def scr(text: str) -> list[fb.Step]:
        calls["n"] += 1
        if calls["n"] == 1:
            return [fb.say("Working", delay=0.05), *fb_slow(60)]
        return [fb.text("Picked it up and finished.")]

    agent = await agentd(short_root, scripts={CHAT: scr})
    ch = channel(short_root)
    r = await runner(agent, ch, short_root)
    w = World(short_root, agent, ch, r)
    try:
        req = await w.say("long task")
        status = await until(lambda: next((m for m in ch.sent() if "Working" in m.text), None))
        await stop_agent(agent)  # agentd dies mid-turn; the record keeps `running`
        w.agent = agent = await agentd(short_root, scripts={CHAT: scr})
        restart = fmt.prefixed(RESTART_TEXT)  # agentd's words only; chatd has no copy
        await until(lambda: status.text == restart)
        await w.r.settle()
        assert ch.reactions[req] == ["👀", "❌"]
        await until(lambda: w.conv().question is not None and w.conv().question.prompt_ref == status.message_id)
        n_sent = len(ch.sent())
        await w.restart_runner()  # and chatd restarts too: no second notice, no second prompt
        await w.r.settle()
        assert len(ch.sent()) == n_sent and status.text == restart
        await w.say("yes")
        await until(lambda: any(m.text == "Mac · Picked it up and finished." for m in ch.sent()))
        await until(lambda: not w.conv().record.unacked)
        assert [m.text for m in ch.sent()].count(restart) == 1
    finally:
        await w.r.close()
        await stop_agent(w.agent)


async def test_messages_while_agentd_is_down_are_handled_after_it_returns(short_root: Path) -> None:
    agent = await agentd(short_root)
    ch = channel(short_root)
    r = await runner(agent, ch, short_root)
    try:
        await stop_agent(agent)
        await until(lambda: not r.client.connected.is_set())
        mid = f"late{time.monotonic_ns()}"
        assert await ch.inject("are you there", chat_id=CHAT_ID, msg_id=mid)
        await r.settle()
        assert not ch.reactions and f"{CHAT_ID}:{mid}" not in ch.ctx.seen  # R4: not marked yet
        agent = await agentd(short_root)
        await until(lambda: ch.reactions.get(mid, [""])[-1] == "✅")
        assert f"{CHAT_ID}:{mid}" in ch.ctx.seen
    finally:
        await r.close()
        await stop_agent(agent)


# -- smaller pieces --------------------------------------------------------------------------------------------


async def test_rollover_and_throttle_of_status_edits(world: World) -> None:
    clock = {"t": 1000.0}
    r = world.r
    r.clock = lambda: clock["t"]
    st = r.state[CHAT]
    from chatd.runner import Turn

    ref = await world.ch.send_text(CHAT_ID, "Mac · On it")
    t = Turn(uid="u", status=ref.message_id, status_sent_at=1000.0, shown="Mac · On it", last_edit=1000.0,
             started_at=1000.0, folder="Resume", say="Searching")
    st.turn = t
    assert not await r._edit_status(st, t)  # inside min_edit_interval: deferred
    clock["t"] += 1
    assert await r._edit_status(st, t)
    assert world.ch.messages[ref.message_id].text == "Mac · On it, in Resume\nSearching"
    clock["t"] += 240
    t.say = "12 so far"
    assert await r._edit_status(st, t)
    assert world.ch.messages[ref.message_id].text == "Mac · On it, in Resume\n12 so far · 4 min"
    clock["t"] += 900 - 120  # near the 15 min edit window: a fresh status message
    t.say = "18 so far"
    assert await r._edit_status(st, t)
    assert t.status != ref.message_id and world.ch.messages[t.status].text.endswith("18 so far · 17 min")
    st.turn = None


async def test_no_edit_channel_drops_progress_but_shows_the_answer(short_root: Path) -> None:
    agent = await agentd(short_root)
    ch = channel(short_root, caps=Cap.REACT | Cap.FILES, edit_window_s=0.0)
    agent.scripts[CHAT] = script({"go": [fb.say(f"step {i}", delay=0.05) for i in range(5)] + [fb.text("Done it.")]})
    r = await runner(agent, ch, short_root)
    try:
        req = f"r{time.monotonic_ns()}"
        await ch.inject("go", chat_id=CHAT_ID, msg_id=req)
        await until(lambda: ch.reactions.get(req, [""])[-1] == "✅")
        texts = [m.text for m in ch.sent()]
        assert texts[-1] == "Mac · Done it." and len(texts) <= 2
    finally:
        await r.close()
        await stop_agent(agent)


def test_format_strings_have_no_dashes_and_the_prefix() -> None:
    for name in dir(fmt):
        v = getattr(fmt, name)
        if isinstance(v, str) and name.isupper() and not name.startswith("_"):
            assert not DASHES.search(v), name
    src = Path(fmt.__file__).parent
    for f in [src / "format.py", src / "runner.py", *(src / "channels").glob("*.py")]:
        assert not DASHES.search(f.read_text()), f"{f.name} contains a literal em or en dash"
    assert fmt.render("Results — 3–5 roles") == "Results, 3-5 roles"
    assert fmt.outbound("hi") == "Mac · hi" and fmt.outbound("Mac · hi") == "Mac · hi"


def test_render_markdown_for_whatsapp() -> None:
    md = "## Top roles\n**Anthropic**, *London*\n\n| Company | City |\n|---|---|\n| Wayve | remote |\n\n---\n* one"
    assert fmt.render(md) == "*Top roles*\n*Anthropic*, _London_\n\nWayve: remote\n\n- one"
    assert fmt.render(md, "plain") == "Top roles\nAnthropic, London\n\nWayve: remote\n\n- one"
    assert fmt.render("| a | b | c |\n|--|--|--|\n| x | 1 | 2 |") == "x: b 1, c 2"
    assert fmt.rate_limit_line("I've hit the Claude usage limit. It resets at 3:40 PM.") == \
        "Mac · I've hit the Claude usage limit. It resets at 3:40 PM. Say \"go\" after that and I'll carry on."
    assert fmt.ago(7300) == "2 hours ago" and fmt.folder_name("~/Desktop/Personal/Resume") == "Resume"


# -- the agent client ------------------------------------------------------------------------------------------


def test_lifeline_token() -> None:
    assert read_lifeline_token('{"token": "' + "c" * 64 + '"}\n') == "c" * 64
    with pytest.raises(ValueError):
        read_lifeline_token('{"token": "short"}')


async def test_client_bad_token_is_fatal_and_reconnects_with_the_same_epoch(short_root: Path) -> None:
    agent = await agentd(short_root)
    try:
        bad = AgentClient(agent.sock, "x" * 64, [CHAT], on_event=lambda e: None,
                          on_connect=_noop, backoff=(0.01,))
        with pytest.raises(AgentFatal):
            await asyncio.wait_for(bad.run(), 3)
        events: list[s.Event] = []
        good = AgentClient(agent.sock, CHAT_TOKEN, [CHAT], on_event=events.append, on_connect=_noop,
                           backoff=(0.05,))
        task = asyncio.create_task(good.run())
        await until(good.connected.is_set)
        epoch = good.epoch
        await stop_agent(agent)
        await until(lambda: not good.connected.is_set())
        agent = await agentd(short_root)
        await until(lambda: good.connected.is_set() and good.connects == 2)
        assert good.epoch == epoch
        data = await good.call("status", conv=CHAT)
        assert isinstance(data, s.ConversationStatus) and data.conv == CHAT
        await good.close()
        await asyncio.wait_for(task, 3)
    finally:
        await stop_agent(agent)


async def _noop(hello: s.HelloData) -> None:
    return None


async def test_e2e_with_a_real_agentd_process(short_root: Path, tmp_path: Path) -> None:
    """`python -m agentd --fake-brain --dev` in a temp MAC_HOME, the dev token, one turn."""
    home = tmp_path / "mh"
    (home / "chat").mkdir(parents=True)
    (home / "chat" / "config.toml").write_text(f'[channels.whatsapp]\nchats = ["{CHAT_ID}"]\n')  # the seed
    sock = short_root / "a.sock"
    env = {**os.environ, "MAC_HOME": str(home), "MAC_LOG_DIR": str(tmp_path / "logs")}
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "agentd", "--fake-brain", "--dev", "--socket", str(sock), env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    r = None
    try:
        await until(lambda: sock.exists() and (home / "run" / "dev-tokens.json").exists(), timeout=15)
        ch = channel(short_root)
        await ch.ctx.seen.load()
        r = ChatRunner(ch, sock=sock, token=dev_token(home / "run"), state_dir=short_root / "c",
                       backoff=(0.05,))
        await r.start()
        await until(r.client.connected.is_set, timeout=10)
        await ch.inject("hello", chat_id=CHAT_ID, msg_id="e2e1")
        await until(lambda: ch.reactions.get("e2e1", [""])[-1] == "✅", timeout=10)
        assert [m.text for m in ch.sent()] == ["Mac · Okay."]
    finally:
        if r is not None:
            await r.close()
        proc.terminate()
        await asyncio.wait_for(proc.wait(), 10)



# -- shared chats: the "Mac" prefix trigger ---------------------------------------------------------------


@pytest.fixture
async def shared(short_root: Path) -> AsyncIterator[World]:
    agent = await agentd(short_root, shared=True)
    ch = channel(short_root)
    w = World(short_root, agent, ch, await runner(agent, ch, short_root))
    yield w
    await w.r.close()
    await stop_agent(w.agent)


def test_prefix_trigger_w9_6() -> None:
    assert strip_trigger("Mac, find me 20 jobs") == "find me 20 jobs"
    assert strip_trigger("mac, find jobs") == "find jobs" and strip_trigger("MAC: status") == "status"
    assert strip_trigger("Mac find jobs") == "find jobs" and strip_trigger("  Mac. stop") == "stop"
    assert strip_trigger("Mac!") == "" and strip_trigger("Mac") == ""
    for text in ("Macintosh is slow", "Mac's battery", "Macbook", "find me jobs, Mac", "Mac · Okay.",
                 "macros", "Mac; stop", "Mac-stop"):
        assert strip_trigger(text) is None, text
    assert strip_trigger("Jarvis, stop", "Jarvis") == "stop" and strip_trigger("Mac, stop", "Jarvis") is None


async def test_shared_chat_ignores_unprefixed_owner_text_silently(shared: World) -> None:
    for i, kw in enumerate(({"text": "find me 20 jobs"}, {"text": "yes"}, {"unsupported": "voice"},
                            {"text": "old one", "replayed": True, "sent_at": time.time() - 60},
                            {"text": "Okay."})):
        mid = f"ign{i}"
        assert await shared.ch.inject(chat_id=CHAT_ID, msg_id=mid, **kw)
    await asyncio.sleep(0.4)  # past the catch-up window
    await shared.r.settle()
    assert shared.ch.wire == [] and not shared.agent.brains.get(CHAT)
    assert not any(f"{CHAT_ID}:ign{i}" in shared.ch.ctx.seen for i in range(5))  # not marked


async def test_shared_chat_runs_prefixed_text_and_commands(shared: World) -> None:
    req = await shared.say("Mac, find me 20 jobs")
    await shared.done(req)
    assert request(shared.brain().asks[0][1]) == "find me 20 jobs"
    await shared.say("mac help")
    await until(lambda: shared.ch.sent()[-1].text == fmt.HELP_PREFIX.format(w="Mac", folder=fmt.HELP_FOLDER))
    assert '"Mac stop" stops' in shared.ch.sent()[-1].text
    await shared.say("Mac: status")
    await until(lambda: shared.ch.sent()[-1].text.startswith("Mac · Nothing running."))


async def test_shared_chat_binds_unprefixed_replies_on_prompts_only(shared: World) -> None:
    shared.agent.scripts[CHAT] = script({"email bob": [confirm_step("Send the email?", "bob@example.com")],
                                         "post it": [confirm_step("Post it?", "linkedin.com")]})
    req = await shared.say("Mac email bob")
    prompt = await until(lambda: next((m for m in shared.ch.sent() if "Send the email?" in m.text), None))
    await until(lambda: shared.conv().question is not None and shared.conv().question.prompt_ref)
    await shared.say("yes")  # unquoted and unprefixed: ignored in a shared chat
    await shared.say(reaction="👍", reply_to=req)  # a 👍 on something that is not a prompt: ignored
    await shared.r.settle()
    assert shared.conv().question is not None
    await shared.say("yes", reply_to=prompt.message_id)  # quotes Mac's prompt: binds
    await shared.done(req)
    assert closed(prompt, fmt.APPROVED)  # never claims it was sent; the answer says what happened
    req2 = await shared.say("mac, post it")
    p2 = await until(lambda: next((m for m in shared.ch.sent() if "Post it?" in m.text), None))
    await until(lambda: shared.conv().question is not None and shared.conv().question.prompt_ref)
    await shared.say(reaction="👍", reply_to=p2.message_id)
    await shared.done(req2)
    assert shared.brain().confirm_results == [True, True]


async def test_shared_chat_catch_up_only_counts_prefixed_messages(shared: World) -> None:
    now = time.time()
    await shared.say("chatting with my friend", replayed=True, sent_at=now - 300)
    a = await shared.say("Mac find jobs", replayed=True, sent_at=now - 200)
    b = await shared.say("Mac, in London", replayed=True, sent_at=now - 100)
    await shared.done(b)
    text = shared.brain().asks[0][1]
    assert len(shared.brain().asks) == 1 and "in London" in text and '"find jobs"' in text
    # an unprefixed message is never a request; it is background context (W10.2)
    assert "friend" not in request(text) and "You: chatting with my friend" in text
    assert f"{CHAT_ID}:{a}" in shared.ch.ctx.seen


async def test_confirm_prompt_uses_the_models_question(world: World) -> None:
    """AG23: when the CONFIRM carries the model's question (AG10), it is shown instead of the fallback."""
    ev = s.ConfirmEvent(conv=CHAT, boot=world.agent.svc.boot, seq=10**6, turn=1, turn_uid="x:u1",
                        question_id="q1", summary="Ready to delete old_resume.pdf.", target="~/old_resume.pdf")
    ev.question = "Delete it?"  # type: ignore[attr-defined]
    await world.r._apply(ev)
    assert world.ch.sent()[-1].text == "Mac · Ready to delete old_resume.pdf. Delete it?\nTo: ~/old_resume.pdf"


def test_echo_prefix_is_one_constant_shared_with_the_bridge() -> None:
    filter_go = Path(__file__).resolve().parents[3] / "bridges/whatsapp/internal/bridge/filter.go"
    if not filter_go.exists():
        pytest.skip("wa-bridge sources not present")
    m = re.search(r'const EchoPrefix = "([^"]*)"', filter_go.read_text())
    assert m and m.group(1) == fmt.PREFIX


# -- per-chat profiles (v3.4: W9.3, W9.6-W9.8) -------------------------------------------------------------


async def app_set(agent: Agent, chat_id: str, patch: dict[str, Any]) -> None:
    """The Mac app's `profile_set` on a real agentd; the chat client gets `profile_changed`."""
    c = await agent.client(APP_TOKEN, subscribe=False)
    try:
        rows = (await c.call({"op": "profile_list"}))["data"]["profiles"]
        rev = next(p["rev"] for p in rows if p["chat_id"] == chat_id)
        r = await c.call({"op": "profile_set", "channel": "whatsapp", "chat_id": chat_id, "rev": rev, "patch": patch})
        assert r["ok"], r
    finally:
        await c.close()


def both_chats(root: Path, **cfg: Any) -> FakeChannel:
    ch = channel(root, **cfg)
    ch.ctx.chats = frozenset({CHAT_ID, OTHER_ID})
    return ch


async def test_chats_and_allowlist_come_from_profiles(short_root: Path) -> None:
    agent = await agentd(short_root)
    ch = both_chats(short_root)
    r = await runner(agent, ch, short_root)
    try:
        assert r.client.conversations == []  # hello sends none; agentd grants the live set
        assert set(r.chats) == {CHAT, OTHER_CHAT} and ch.allowed_sets[-1] == {CHAT_ID, OTHER_ID}
        rev = r.profiles_rev
        await app_set(agent, OTHER_ID, {"enabled": False})
        await until(lambda: OTHER_CHAT not in r.chats)
        assert ch.allowed_sets[-1] == {CHAT_ID} and r.profiles_rev == rev + 1
        assert not await ch.inject("hi", chat_id=OTHER_ID, msg_id="o1") or not ch.reactions.get("o1")
        await app_set(agent, OTHER_ID, {"enabled": True})
        await until(lambda: OTHER_CHAT in r.chats and ch.allowed_sets[-1] == {CHAT_ID, OTHER_ID})
        await r.settle()
        agent.scripts[OTHER_CHAT] = script({"hello": [fb.text("Hi.")]})
        assert await ch.inject("hello", chat_id=OTHER_ID, msg_id="o2")
        await until(lambda: ch.reactions.get("o2", [""])[-1] == "✅")  # subscribed live, no reconnect
    finally:
        await r.close()
        await stop_agent(agent)


async def test_profile_changed_rev_rules(world: World) -> None:
    r, rev = world.r, world.r.profiles_rev
    p = world.agent.svc.profiles.get("whatsapp", OTHER_ID)
    assert p is not None
    off = s.ProfileChanged(rev=rev, channel="whatsapp", chat_id=OTHER_ID,
                           profile=s.Profile(**{**p.__dict__, "enabled": False}))
    await r._profile_changed(off)  # rev <= ours: ignored
    assert OTHER_CHAT in r.chats
    await r._profile_changed(s.ProfileChanged(rev=rev + 1, channel="telegram", chat_id="x", deleted=True))
    assert r.profiles_rev == rev + 1 and OTHER_CHAT in r.chats  # another channel: counted, nothing else
    lists = len(world.ch.allowed_sets)
    await r._profile_changed(s.ProfileChanged(rev=rev + 5, channel="whatsapp", chat_id=OTHER_ID, deleted=True))
    assert r.profiles_rev == world.agent.svc.profiles.rev  # a jump re-lists: agentd's truth wins
    assert OTHER_CHAT in r.chats and len(world.ch.allowed_sets) == lists + 1


async def test_prefix_word_and_shared_trigger_from_the_profile(world: World) -> None:
    await app_set(world.agent, CHAT_ID, {"trigger": "prefix", "prefix": "Jarvis"})
    await until(lambda: world.r.profiles[CHAT_ID].prefix == "Jarvis")
    await world.say("find me jobs")  # no prefix: ignored silently
    await world.say("Mac, find me jobs")  # not this chat's word
    req = await world.say("jarvis: find me jobs")
    await world.done(req)
    assert [request(a[1]) for a in world.brain().asks] == ["find me jobs"]
    await app_set(world.agent, CHAT_ID, {"trigger": "all", "prefix": "Mac", "shared": True})
    await until(lambda: world.r.profiles[CHAT_ID].shared)
    await world.say("status")  # shared: the prefix is needed whatever `trigger` says
    await world.r.settle()
    assert not any(m.text.startswith("Mac · Nothing running") for m in world.ch.sent())
    await world.say("Mac, status")
    await until(lambda: world.ch.sent()[-1].text.startswith("Mac · Nothing running."))


async def test_quoted_reply_in_a_prefix_chat_is_never_a_command(shared: World) -> None:
    shared.agent.scripts[CHAT] = script({"email bob": [confirm_step("Send the email?", "bob@example.com")]})
    req = await shared.say("Mac email bob")
    prompt = await until(lambda: next((m for m in shared.ch.sent() if "Send the email?" in m.text), None))
    await until(lambda: shared.conv().question is not None and shared.conv().question.prompt_ref)
    await shared.say("stop", reply_to=prompt.message_id)  # an answer to the question, not the command
    await shared.done(req)
    assert closed(prompt, fmt.DECLINED) and shared.brain().confirm_results == [False]


async def test_add_group_resolves_the_subject_and_binds_a_quoted_yes(short_root: Path) -> None:
    groups = [{"id": "555@g.us", "subject": "Book club", "members": 4, "owner_only": False},
              {"id": "556@g.us", "subject": "Twins", "members": 2, "owner_only": False},
              {"id": "557@g.us", "subject": " twins ", "members": 2, "owner_only": False},
              {"id": "777@g.us", "subject": "Timetable", "members": 9, "owner_only": False}]
    agent = await agentd(short_root)
    ch = channel(short_root, groups=groups)
    w = World(short_root, agent, ch, await runner(agent, ch, short_root))
    try:
        chess = await w.say("add group Chess")  # no such group: an ordinary request (P2-2)
        await w.done(chess)
        assert request(w.brain().asks[0][1]) == "add group Chess"
        await w.say("add group twins")
        await until(lambda: ch.sent()[-1].text == fmt.MANY_GROUPS.format(subject="twins"))
        await w.say("add group timetable")
        await until(lambda: ch.sent()[-1].text == fmt.NEVER_LISTED)
        await w.say("/add group  BOOK CLUB ")
        await until(lambda: w.conv().question is not None and w.conv().question.prompt_ref)
        await w.say("yes")  # unquoted: never approves a profile question (it is just a message)
        await until(lambda: w.conv().question is None)
        await w.r.settle()
        assert agent.svc.profiles.get("whatsapp", "555@g.us") is None
        await w.say("add group book club")
        await until(lambda: w.conv().question is not None and w.conv().question.prompt_ref)
        prompt = ch.messages[w.conv().question.prompt_ref]
        assert "\u201cBook club\u201d" in prompt.text
        await w.say("yes", reply_to=prompt.message_id)
        await until(lambda: "555@g.us" in w.r.profiles and "555@g.us" in ch.allowed_sets[-1])
        p = w.r.profiles["555@g.us"]
        assert p.shared  # from the member list; capabilities are agentd's default
        await until(lambda: "Book club" in ch.sent()[-1].text and "now" in ch.sent()[-1].text)  # agentd's result
        assert agent.svc.profiles.get("whatsapp", "777@g.us") is None
    finally:
        await w.r.close()
        await stop_agent(agent)


async def test_allow_needs_a_thumbs_up_and_disallow_is_at_once(world: World) -> None:
    await world.say("disallow chrome")
    await until(lambda: world.ch.sent() and world.ch.sent()[-1].text == "Mac · Chrome is off in this chat.")
    assert not world.agent.svc.profiles.get("whatsapp", CHAT_ID).capabilities.chrome
    await until(lambda: not world.r.profiles[CHAT_ID].capabilities.chrome)  # profile_changed applied
    for free in ("allow everything", "Allow it.", "disallow cookies"):  # not commands: the model's (P2-2)
        await world.done(await world.say(free))
    await world.say("Mac, allow Chrome")  # the trigger word works in every chat (P2-3)
    await until(lambda: world.conv().question is not None and world.conv().question.prompt_ref)
    prompt = world.ch.messages[world.conv().question.prompt_ref]
    await world.say(reaction="👍", reply_to=prompt.message_id)
    await until(lambda: world.r.profiles[CHAT_ID].capabilities.chrome)
    assert [request(a[1]) for a in world.brain().asks] == ["allow everything", "Allow it.", "disallow cookies"]
    await world.say("disallow sending messages for you")  # the plain words name it too
    await until(lambda: world.ch.sent()[-1].text == "Mac · Sending messages for you is off in this chat.")


async def test_members_are_compared_on_every_bridge_connect(short_root: Path) -> None:
    agent = await agentd(short_root, shared=True)
    ch = channel(short_root, groups=[{"id": CHAT_ID, "subject": "Interview prep", "members": 3, "owner_only": False}])
    w = World(short_root, agent, ch, await runner(agent, ch, short_root))
    try:
        await ch.reconnect()
        await w.r.settle()
        assert not w.conv().record.paused and w.r.state[CHAT].members == 3
        ch.chats[0]["members"] = 2  # someone left: nothing changes
        await ch.reconnect()
        await w.r.settle()
        assert not w.conv().record.paused
        ch.chats[0]["members"] = 4  # someone joined while the bridge was down
        await ch.reconnect()
        await until(lambda: w.conv().record.paused == "members_changed")
    finally:
        await w.r.close()
        await stop_agent(agent)


async def test_an_owner_only_chat_that_gains_members_pauses(world: World) -> None:
    await world.ch.change_chat(ChatChanged(CHAT_ID, "Interview prep", owner_only=False, extra={"members": 2}))
    await until(lambda: world.conv().record.paused == "members_changed")


# -- phase 2 review fixes ---------------------------------------------------------------------------------


def test_owner_commands_are_exact_capability_names_only() -> None:
    """P2-2: allow/disallow need a real capability (key, key with spaces, or its plain words)."""
    assert parse_command("allow chrome") == ("allow", "chrome") == parse_command("Allow Chrome.")
    assert parse_command("allow files read") == ("allow", "files_read") == parse_command("/allow files_read")
    assert parse_command("disallow the web") == ("disallow", "web")
    assert parse_command("disallow sending messages for you") == ("disallow", "outward_sends")
    for free in ("allow it", "Allow it.", "disallow cookies", "allow me to explain", "allow"):
        assert parse_command(free) is None
    assert parse_command("add group of five people") == ("add", "of five people")  # a candidate only


def _ev(world: World, cls: type, seq: int, **kw: Any) -> Any:
    return cls(conv=CHAT, boot=world.r.state[CHAT].boot or world.agent.svc.boot, seq=seq, **kw)


async def test_desktop_wait_line_clears_when_the_tool_ends(world: World) -> None:
    """UX2: the wait note goes away as soon as the tool that waited ends."""
    base = 10**6
    await world.r._apply(_ev(world, s.SayEvent, base, turn=1, turn_uid="x:u1", text="Opening the site"))
    await world.r._apply(_ev(world, s.StatusEvent, base + 1, turn=1, turn_uid="x:u1",
                             data={"waiting_for": "desktop"}))
    st = world.r.state[CHAT]
    assert st.turn is not None and st.turn.note == fmt.DESKTOP_WAIT
    world.r.clock = lambda: time.time() + 10  # past the edit throttle
    await world.r._apply(_ev(world, s.ToolEndEvent, base + 2, turn=1, turn_uid="x:u1", tool="navigate",
                             tool_use_id="t1"))
    assert st.turn.note == "" and fmt.DESKTOP_WAIT not in world.ch.sent()[-1].text


async def test_persona_confirm_shows_question_target_then_the_text(world: World) -> None:
    """UX4: summary and question, then To:, then the detail; no fallback after a summary that asks."""
    ev = _ev(world, s.ConfirmEvent, 10**6, turn=1, turn_uid="x:u1", question_id="q1",
             summary="Change how I write in this chat.", detail="Be brief. Use British spelling.",
             target="Interview prep")
    await world.r._apply(ev)
    assert world.ch.sent()[-1].text == ("Mac · Change how I write in this chat. Shall I go ahead?"
                                        "\nTo: Interview prep\n\nBe brief. Use British spelling.")
    ev2 = _ev(world, s.ConfirmEvent, 10**6 + 1, turn=1, turn_uid="x:u1", question_id="q2",
              summary="Send the email to bob@example.com?", target="bob@example.com")
    await world.r._apply(ev2)
    assert world.ch.sent()[-1].text == "Mac · Send the email to bob@example.com?"


async def test_non_confirm_questions_stop_asking_when_closed(world: World) -> None:
    """UX8: an expired or declined profile/folder/route prompt is edited so it no longer asks."""
    for i, (kind, outcome, want) in enumerate(((s.QuestionKind.PROFILE, s.CloseOutcome.EXPIRED, fmt.NO_ANSWER_LEFT),
                                               (s.QuestionKind.FOLDER_APPROVAL, s.CloseOutcome.DENIED, fmt.DECLINED),
                                               (s.QuestionKind.ROUTE, s.CloseOutcome.EXPIRED, fmt.NO_ANSWER_LEFT))):
        qid, seq = f"q{i}", 10**6 + 2 * i
        await world.r._apply(_ev(world, s.QuestionEvent, seq, question_id=qid, kind=kind,
                                 text="Reply yes to this message.", expires_at=time.time() + 600))
        prompt = world.ch.sent()[-1]
        await world.r._apply(_ev(world, s.QuestionClosedEvent, seq + 1, question_id=qid, outcome=outcome))
        assert closed(prompt, want)


async def test_forbidden_conversation_re_lists_and_says_nothing(world: World) -> None:
    """I3: agentd refuses the chat (never-listed with no notice seen): chatd re-lists, drops it,
    narrows the allowlist and says nothing there."""
    await world.agent.svc.profiles.update("whatsapp", CHAT_ID, {"enabled": False})  # no notice sent
    n = len(world.ch.wire)
    await world.say("find me jobs")
    await until(lambda: CHAT not in world.r.chats)
    await world.r.settle()
    assert len(world.ch.wire) == n and CHAT_ID not in world.ch.allowed_sets[-1]


async def test_a_group_renamed_to_timetable_is_never_listed(world: World) -> None:
    """N6: a profiled chat renamed to a never subject (format characters and case ignored)."""
    sent: list[tuple[str, dict[str, Any]]] = []
    real = world.r.client.call

    async def spy(op: str, **args: Any) -> Any:
        sent.append((op, args))
        return await real(op, **args)

    world.r.client.call = spy  # type: ignore[method-assign]
    await world.ch.change_chat(ChatChanged(CHAT_ID, " Time\u200btable ", owner_only=True))
    await until(lambda: any(op == "profile_list" for op, _ in sent))
    assert ("pause", {"conv": CHAT, "value": True, "reason": "never_listed"}) in sent
    await world.ch.change_chat(ChatChanged(CHAT_ID, "Timetables", owner_only=True))
    await world.r.settle()
    assert sum(op == "pause" for op, _ in sent) == 1


async def test_catch_up_passes_earlier_messages_whole(world: World) -> None:
    now = time.time()
    long = "x" * 600
    await world.say(long, replayed=True, sent_at=now - 200)
    b = await world.say("and in London", replayed=True, sent_at=now - 100)
    await world.done(b)
    assert long in world.brain().asks[0][1]


async def test_client_fails_waiting_calls_at_once_when_agentd_drops(short_root: Path) -> None:
    """code-quality 10: a socket that drops during on_connect fails the next calls at once."""
    agent = await agentd(short_root)
    results: list[float] = []

    async def on_connect(hello: s.HelloData) -> None:
        await stop_agent(agent)
        t0 = time.monotonic()
        try:
            await client.call("profile_list")
        except Exception:  # noqa: BLE001
            results.append(time.monotonic() - t0)
        await client.close()

    client = AgentClient(agent.sock, CHAT_TOKEN, [], on_event=lambda e: None, on_connect=on_connect,
                         backoff=(0.05,))
    await asyncio.wait_for(client.run(), 5)
    assert results and results[0] < 2.0


# -- v3.5: group context, mentions, quoted (W10, W7.7 runner) ----------------------------------------------

PRIYA = {"jid": "111@s.whatsapp.net", "number": "447700900111", "name": "Priya"}
ARJUN = {"jid": "222@s.whatsapp.net", "number": "447700900222", "name": "Arjun"}


def test_ring_limits_order_and_dedupe() -> None:
    r = Ring()
    for i in range(60):
        r.add(f"m{i}", sender="Priya", at=float(i), text="x")
    assert len(r) == 50 and r.messages()[0].msg_ref == "m10"  # the newest 50
    r = Ring()
    r.add("long", at=1.0, text="y" * 5000)
    assert r.messages()[0].text == "y" * 999 + "…"  # each text clipped to 1,000, marked as partial
    for i in range(9):
        r.add(f"k{i}", at=2.0 + i, text="z" * 1000)
    assert r.chars() <= 8000 and r.messages()[0].msg_ref == "k1"  # 8,000 in all: oldest go first
    r.add("k8", at=99.0, text="dup")
    assert [m.msg_ref for m in r.messages()].count("k8") == 1
    r = Ring()
    r.add("live", at=10.0, text="now")
    r.add("early", at=5.0, text="before", earlier=True)  # history goes in time order
    assert [m.msg_ref for m in r.messages()] == ["early", "live"] and r.messages()[0].earlier
    assert [m.msg_ref for m in r.messages(exclude="live")] == ["early"]


def test_mentions_are_resolved_deterministically() -> None:
    members = [PRIYA, ARJUN, {"jid": "333@s", "number": "4477333", "name": "Sam"},
               {"jid": "334@s", "number": "4477334", "name": "Sam"}, {"jid": "335@s", "number": "4477335", "name": "Mac"}]
    text, jids = resolve_mentions("@Priya and @arjun, ask @Sam or @Mac or @Zoe. email@Priya.com", members)
    assert text == "@447700900111 and @447700900222, ask @Sam or @Mac or @Zoe. email@Priya.com"
    assert jids == (PRIYA["jid"], ARJUN["jid"])
    long = [{"jid": "p@s", "number": "1", "name": "Priya"}, {"jid": "pk@s", "number": "2", "name": "Priya K"}]
    assert resolve_mentions("hi @Priya K!", long) == ("hi @2!", ("pk@s",))  # the longest name wins
    many = [{"jid": f"{i}@s", "number": str(i), "name": f"M{i}"} for i in range(7)]
    text, jids = resolve_mentions(" ".join(f"@M{i}" for i in range(7)), many)
    assert len(jids) == 5 and text.endswith("@M5 @M6")  # at most 5; the rest stay plain


@pytest.fixture
async def group(short_root: Path) -> AsyncIterator[World]:
    agent = await agentd(short_root, shared=True)
    ch = channel(short_root, members={CHAT_ID: [PRIYA, ARJUN]})
    w = World(short_root, agent, ch, await runner(agent, ch, short_root))
    yield w
    await w.r.close()
    await stop_agent(w.agent)


def spy(r: ChatRunner) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    real = r.client.call

    async def call(op: str, **args: Any) -> Any:
        calls.append((op, args))
        return await real(op, **args)

    r.client.call = call  # type: ignore[method-assign]
    return calls


async def test_context_is_data_and_never_triggers(group: World) -> None:
    group.agent.scripts[CHAT] = script({"email bob": [confirm_step("Send the email?", "bob@example.com")]})
    calls = spy(group.r)
    req = await group.say("Mac email bob")
    prompt = await until(lambda: next((m for m in group.ch.sent() if "Send the email?" in m.text), None))
    await until(lambda: group.conv().question is not None and group.conv().question.prompt_ref)
    n_ops, n_wire = len(calls), len(group.ch.wire)
    assert await group.ch.inject_context("Mac, stop", chat_id=CHAT_ID, msg_id="c1")
    assert await group.ch.inject_context("yes", chat_id=CHAT_ID, msg_id="c2")  # a member's yes binds nothing
    await group.r.settle()
    assert calls[n_ops:] == [] and len(group.ch.wire) == n_wire and group.conv().question is not None
    assert [m.msg_ref for m in group.r.rings[CHAT_ID].messages()][-2:] == ["c1", "c2"]
    await group.say("no", reply_to=prompt.message_id)
    await group.done(req)


async def test_every_ask_carries_the_whole_ring_and_steers_none(group: World) -> None:
    group.agent.scripts[CHAT] = script({"go": [fb.say("Working", delay=0.05), *fb_slow(40)],
                                        "again": [fb.text("Again.")]})
    calls = spy(group.r)
    await group.ch.inject_context("can you look at this? https://example.com/job", chat_id=CHAT_ID, msg_id="c1")
    await group.ch.inject_context("looks senior", chat_id=CHAT_ID, msg_id="c2", sender_name="Arjun")
    await group.say("just chatting")  # your own non-trigger message: context too
    req = await group.say("Mac, go")
    await until(lambda: any("Working" in m.text for m in group.ch.sent()))
    await group.say("Mac, also check London")  # mid-turn: a steer, no context
    await until(lambda: group.brain().steers)
    await group.say("Mac stop")
    await group.done(req, "✋")
    req2 = await group.say("Mac, again")
    await group.done(req2)
    asks = [a for op, a in calls if op == "ask"]
    steers = [a for op, a in calls if op == "steer"]
    assert len(asks) == 2 and steers and all("context" not in a for a in steers)
    ctx = asks[0]["context"]
    assert ctx.members == ["Priya", "Arjun"]
    assert [(m.msg_ref, m.sender, m.owner, m.mac) for m in ctx.messages][:2] == \
        [("c1", "Priya", False, False), ("c2", "Arjun", False, False)]
    assert ctx.messages[2].owner and ctx.messages[2].text == "just chatting"
    assert all(m.msg_ref != req for m in ctx.messages)  # the trigger itself is left out
    refs2 = [m.msg_ref for m in asks[1]["context"].messages]
    assert refs2[:3] == [m.msg_ref for m in ctx.messages]  # the whole ring again: agentd frames the delta
    text = group.brain().asks[0][1]
    assert "<group_context" in text and "can you look at this?" in text and request(text) == "go"


async def test_an_ask_with_an_empty_ring_requests_history(group: World) -> None:
    """After the ask is accepted; the hourly bound is the bridge's (W10.3)."""
    req = await group.say("Mac, hi")
    await group.done(req)
    assert group.ch.history_requests == [(CHAT_ID, req, 30)]
    await group.done(await group.say("Mac, hi again"))  # the ring holds Mac's reply now: no request
    assert len(group.ch.history_requests) == 1
    await group.ch.inject_context("from before", chat_id=CHAT_ID, msg_id="h1", at=time.time() - 9000, history=True)
    await group.r.settle()
    first = group.r.rings[CHAT_ID].messages()[0]
    assert (first.msg_ref, first.earlier) == ("h1", True)


async def test_a_deleted_message_leaves_the_ring_for_good(group: World) -> None:
    """Deleted on WhatsApp: never in a later ask's context, and a late history page can't bring it back."""
    calls = spy(group.r)
    await group.ch.inject_context("keep this", chat_id=CHAT_ID, msg_id="c1")
    await group.ch.inject_context("delete me", chat_id=CHAT_ID, msg_id="c2")
    await group.r.settle()
    await group.r._context_sink(ContextMessage(channel="fake", chat_id=CHAT_ID, ref=MessageRef(CHAT_ID, "c2"),
                                               deleted=True))
    await group.ch.inject_context("delete me", chat_id=CHAT_ID, msg_id="c2", history=True)
    await group.done(await group.say("Mac, hi"))
    ask = next(a for op, a in calls if op == "ask")
    assert [m.msg_ref for m in ask["context"].messages] == ["c1"]


async def test_new_chat_starts_with_no_earlier_messages(group: World) -> None:
    """"new chat" drops the ring (Mac's old replies included) and fetches no history to refill it."""
    calls = spy(group.r)
    await group.ch.inject_context("old talk", chat_id=CHAT_ID, msg_id="c1")
    await group.done(await group.say("Mac, hi"))
    await group.say("Mac new chat")
    await until(lambda: any(op == "new_session" for op, _ in calls))
    await group.r.settle()
    n_hist = len(group.ch.history_requests)
    await group.done(await group.say("Mac, hello"))
    ask = [a for op, a in calls if op == "ask"][-1]
    assert ask.get("context") is None or not ask["context"].messages
    assert len(group.ch.history_requests) == n_hist


async def test_an_answer_that_tags_is_a_new_message_and_joins_the_ring(group: World) -> None:
    """Edits ping nobody: with tags, the answer is a new message and the status is left as it was."""
    group.agent.scripts[CHAT] = script({"tag them": [fb.say("Looking", delay=0.05),
                                                     fb.text("@Priya and @Arjun, the job is senior. @Zoe too.",
                                                             delay=0.2)]})
    req = await group.say("Mac, tag them")
    await group.done(req)
    status, answer = [m for m in group.ch.sent() if m.reply_to == req]
    assert status.text == "Mac · On it\nLooking"  # not edited into the answer: shown once
    assert answer.text == "Mac · @447700900111 and @447700900222, the job is senior. @Zoe too."
    tagged = [w for w in group.ch.wire if w.mentions]
    assert len(tagged) == 1 and tagged[0].op == "send_text" and tagged[0].mentions == (PRIYA["jid"], ARJUN["jid"])
    assert all(not w.mentions for w in group.ch.wire if w.op == "edit")
    assert len(group.ch.wire) == 4  # 👀, status, answer, ✅
    mine = group.r.rings[CHAT_ID].messages()[-1]
    assert mine.mac and mine.msg_ref == answer.message_id and "@Priya" in (mine.text or "")


async def test_the_assistant_name_is_never_taggable(group: World) -> None:
    group.r.assistant_name = "Jarvis"
    group.r.members[CHAT_ID] = [PRIYA, {"jid": "9@s", "number": "4479", "name": "jarvis"}]
    assert group.r._mentions(group.r.state[CHAT], "@Jarvis and @Priya") == \
        ("@Jarvis and @447700900111", (PRIYA["jid"],))


async def test_the_assistant_name_comes_from_the_voice_config(short_root: Path) -> None:
    from macd import config as mcfg

    mcfg.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    mcfg.CONFIG_PATH.write_text('name = "Friday"\n')
    agent = await agentd(short_root)
    ch = channel(short_root)
    r = await runner(agent, ch, short_root)
    try:
        assert r.assistant_name == "Friday"
    finally:
        await r.close()
        await stop_agent(agent)


async def test_read_chat_off_drops_the_ring_and_keeps_quoted(group: World) -> None:
    calls = spy(group.r)
    await group.ch.inject_context("look https://example.com/a", chat_id=CHAT_ID, msg_id="c1")
    await group.r.settle()
    assert group.r.rings[CHAT_ID].messages()
    await group.say("Mac, disallow read_chat")
    await until(lambda: CHAT_ID not in group.ch.context_sets[-1])
    assert CHAT_ID not in group.r.rings and CHAT_ID not in group.r.members
    assert not await group.ch.inject_context("more", chat_id=CHAT_ID, msg_id="c2") or CHAT_ID not in group.r.rings
    req = await group.say("Mac, open that link", reply_to="c1")
    await group.done(req)
    ctx = next(a for op, a in calls if op == "ask")["context"]
    assert ctx.messages is None and ctx.members is None
    q = ctx.quoted
    assert (q.text, q.sender, q.owner, q.mac, q.links) == ("look https://example.com/a", "Priya", False, False,
                                                           ["https://example.com/a"])
    # a mention without read_chat stays plain text: no members known
    assert group.r._mentions(group.r.state[CHAT], "@Priya") == ("@Priya", ())


async def test_compaction_is_logged_never_posted(world: World, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    n = len(world.ch.wire)
    await world.r._apply(_ev(world, s.StatusEvent, 10**6, data={"compacted": {"trigger": "auto", "pre_tokens": 150000,
                                                                             "post_tokens": 20000}}))
    assert len(world.ch.wire) == n and "compacted (auto, 150000 -> 20000 tokens)" in caplog.text
