"""API contract (docs/agentd-api.md, W7.1): the real agentd server with FakeBrain conversations."""

from __future__ import annotations

import asyncio
import os
import stat
import time
from types import SimpleNamespace

import pytest
from agentd_kit import (
    APP_TOKEN,
    CHAT,
    CHAT_TOKEN,
    NEVER_CHAT,
    TEST_TOKEN,
    VOICE_TOKEN,
    Client,
    confirm_step,
    make_agent,
    slow,
    stop_agent,
)

from agentd import schema as s
from macd.brain import fake_brain as fb

V = s.VOICE_CONV


# ----------------------------------------------------------------------------- transport


async def test_socket_is_private(agent):
    assert stat.S_IMODE(os.stat(agent.sock).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(agent.sock.parent).st_mode) == 0o700


async def test_peer_with_other_uid_is_closed_without_reply(short_root):
    a = await make_agent(short_root, uid=os.getuid() + 1)
    try:
        c = await Client.open(a.sock)
        c.send({"op": "hello"})
        await asyncio.wait_for(c.closed.wait(), 2)
        assert c.frames == []
    finally:
        await stop_agent(a)


async def test_bad_json_is_bad_frame_with_null_id_and_closes(agent):
    c = await Client.open(agent.sock)
    c.w.write(b"{not json\n")
    await asyncio.wait_for(c.closed.wait(), 2)
    assert c.errors == [{"id": None, "ok": False, "code": "bad_frame", "error": c.errors[0]["error"]}]


async def test_line_over_one_mib_is_bad_frame(agent):
    c = await Client.open(agent.sock)
    c.w.write(b'{"id": 1, "op": "hello", "x": "' + b"a" * (s.MAX_LINE_BYTES + 10) + b'"}\n')
    await asyncio.wait_for(c.closed.wait(), 3)
    assert c.errors and c.errors[0]["code"] == "bad_frame" and c.errors[0]["id"] is None


# ----------------------------------------------------------------------------- hello and identity


async def test_first_message_must_be_hello(agent):
    c = await Client.open(agent.sock)
    r = await c.call({"op": "status"})
    assert r["code"] == "hello_required"
    await asyncio.wait_for(c.closed.wait(), 2)


async def test_hello_reply_and_kind_from_token_not_payload(agent):
    c = await Client.open(agent.sock)
    r = await c.hello(VOICE_TOKEN, [V], kind="app", client_kind="app")
    assert r["ok"]
    d = r["data"]
    assert d["api"] == "1.2" and d["kind"] == "voice" and d["boot"] == agent.svc.boot
    assert d["conversations"] == [V] and d["interrupted"] == []
    s.decode_response(r, "hello")
    # the voice token still may not use an app op
    r = await c.call({"op": "ws_approve", "path": "~/x"})
    assert r["code"] == "forbidden"


async def test_unknown_token_and_test_kind_are_unauthorized(agent):
    for tok in ("x" * 64, TEST_TOKEN):
        c = await Client.open(agent.sock)
        r = await c.hello(tok, [])
        assert r["code"] == "unauthorized"
        await asyncio.wait_for(c.closed.wait(), 2)


async def test_version_mismatch_plain_message_and_close(agent):
    c = await Client.open(agent.sock)
    r = await c.hello(VOICE_TOKEN, [V], api="1.1")
    assert r["code"] == "version_mismatch"
    assert r["error"] == ("agentd speaks API 1.2 but this client speaks 1.1. "
                          "Restart Mac to update all parts together.")
    await asyncio.wait_for(c.closed.wait(), 2)


@pytest.mark.parametrize("token,convs", [
    (CHAT_TOKEN, [V]), (VOICE_TOKEN, [CHAT]), (CHAT_TOKEN, ["whatsapp:notlisted@g.us"]),
    (CHAT_TOKEN, [NEVER_CHAT]), (CHAT_TOKEN, ["telegram:123@g.us"]), (APP_TOKEN, [V]),
    (CHAT_TOKEN, [CHAT, "whatsapp:nope"]),
])
async def test_forbidden_conversations(agent, token, convs):
    c = await Client.open(agent.sock)
    r = await c.hello(token, convs)
    assert r["code"] == "forbidden_conversation"


async def test_chat_config_path_cannot_come_from_the_request(agent, short_root):
    other = short_root / "evil.toml"
    other.write_text('[channels.whatsapp]\nchats = ["evil@g.us"]\n')
    c = await Client.open(agent.sock)
    r = await c.hello(CHAT_TOKEN, ["whatsapp:evil@g.us"], config=str(other), config_path=str(other),
                      chat_config=str(other))
    assert r["code"] == "forbidden_conversation"
    assert agent.paths.chat_config == agent.paths.mac_home / "chat" / "config.toml"


async def test_chat_ops_on_an_ungranted_conversation(agent):
    c = await agent.client(CHAT_TOKEN, [CHAT])
    r = await c.call({"op": "ask", "conv": "whatsapp:555@g.us", "turn": 1, "text": "hi"})
    assert r["code"] == "forbidden_conversation"


async def test_ops_outside_kind_forbidden_unknown_op_bad_args_unknown_fields(agent):
    c = await agent.client(CHAT_TOKEN, [CHAT])
    assert (await c.call({"op": "confirm", "conv": CHAT, "question_id": "x", "approved": True}))["code"] == "forbidden"
    assert (await c.call({"op": "teleport"}))["code"] == "unknown_op"
    assert (await c.call({"op": "ask", "conv": CHAT, "turn": True, "text": "x"}))["code"] == "bad_args"
    assert (await c.call({"op": "ask", "conv": CHAT, "text": "x"}))["code"] == "bad_args"
    r = await c.call({"op": "status", "conv": CHAT, "whatever": 1})
    assert r["ok"]
    app = await agent.client(APP_TOKEN, [])
    assert (await app.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "x"}))["code"] == "forbidden"


async def test_voice_steer_and_interrupt_need_turn(agent):
    c = await agent.client()
    assert (await c.call({"op": "steer", "conv": V, "text": "x"}))["code"] == "bad_args"


# ----------------------------------------------------------------------------- turns


async def test_voice_ask_event_stream_and_turn_uid(agent):
    agent.scripts[V] = lambda t: [fb.text("Sure."), *fb.tool("mcp__mac__calendar"), fb.say("Looking"),
                                  fb.text("Two meetings.")]
    c = await agent.client()
    r = await c.call({"op": "ask", "conv": V, "turn": 1, "text": "calendar"})
    uid = r["data"]["turn_uid"]
    assert uid == f"{agent.svc.boot}:{V}:e1:1"
    evs = []
    while not evs or evs[-1]["ev"] != "done":
        evs.append(await c.event())
    assert [e["ev"] for e in evs] == ["text", "tool_start", "tool_end", "say", "text", "done"]
    assert all(e["turn"] == 1 and e["turn_uid"] == uid and e["boot"] == agent.svc.boot for e in evs)
    assert [e["seq"] for e in evs] == sorted(e["seq"] for e in evs)
    assert evs[1]["tool_use_id"] == evs[2]["tool_use_id"] and evs[1]["tool_use_id"]
    assert "interrupted" not in evs[-1]["data"] and evs[-1]["data"]["duration_s"] >= 0
    # the brain saw an internal turn id, not the client's number
    assert agent.brain(V).asks[0][1] == "calendar"


async def test_busy_and_voice_steer_after_done_continues_the_turn(agent):
    agent.scripts[V] = lambda t: slow(n=10, delay=0.02) if t == "long" else [fb.text(f"re: {t}")]
    c = await agent.client()
    uid = (await c.call({"op": "ask", "conv": V, "turn": 1, "text": "long"}))["data"]["turn_uid"]
    assert (await c.call({"op": "ask", "conv": V, "turn": 2, "text": "again"}))["code"] == "busy"
    assert (await c.call({"op": "steer", "conv": V, "turn": 1, "text": "more"}))["data"] == {"accepted": True}
    await c.event("done")
    # local ClaudeBrain semantics: a steer for the just-finished turn is a new query on that turn
    assert (await c.call({"op": "steer", "conv": V, "turn": 1, "text": "late"}))["data"] == {"accepted": True}
    ev = await c.event("text", pred=lambda e: e["text"] == "re: late")
    assert ev["turn"] == 1 and ev["turn_uid"] == uid
    assert (await c.event("done"))["turn_uid"] == uid
    # never for an older turn, another epoch, or a stopped turn
    assert (await c.call({"op": "steer", "conv": V, "turn": 7, "text": "x"}))["data"] == {"accepted": False}
    await c.call({"op": "ask", "conv": V, "turn": 2, "text": "long"})
    await c.call({"op": "interrupt", "conv": V, "turn": 2})
    await c.event("done", pred=lambda e: e["turn"] == 2)
    assert (await c.call({"op": "steer", "conv": V, "turn": 2, "text": "x"}))["data"] == {"accepted": False}
    assert (await c.call({"op": "interrupt", "conv": V, "turn": 2}))["data"] == {"stopped": False}


async def test_voice_steer_crossing_done_keeps_its_events(agent):
    agent.scripts[V] = lambda t: [fb.text(f"re: {t}")]
    c = await agent.client()
    uid = (await c.call({"op": "ask", "conv": V, "turn": 1, "text": "hi"}))["data"]["turn_uid"]
    await c.event("done")
    conv = agent.svc.convs[V]
    internal = conv.last_finished.internal
    await conv.brain.steer(internal, "crossed")  # the steer reached the brain just after its DONE
    ev = await c.event("text", pred=lambda e: e["text"] == "re: crossed")
    assert ev["turn_uid"] == uid and (await c.event("done"))["turn_uid"] == uid
    assert conv.running() is None and agent.svc.convs[V].record.running is None


async def test_brain_error_text_passes_through_empty(agent):
    from macd.contracts import BrainErrorKind

    agent.scripts[V] = lambda t: [fb.error(BrainErrorKind.NETWORK, "")]
    c = await agent.client()
    await c.call({"op": "ask", "conv": V, "turn": 1, "text": "hi"})
    ev = await c.event("error")
    assert ev["kind"] == "network" and ev["message"] == ""


async def test_voice_side_connection_without_voice_conv_is_neutral(agent):
    from macd import config

    config.update_state(brain={"session_id": "voice-s1", "updated_at": 1.0})
    agent.scripts[V] = lambda t: slow(n=100, delay=0.01)
    main = await agent.client(epoch="e1")
    await main.call({"op": "ask", "conv": V, "turn": 1, "text": "long"})
    await main.event("say")
    side = await agent.client(VOICE_TOKEN, [], epoch="other")  # local-mode stop_other connection
    assert (await side.call({"op": "stop_other"}))["data"] == {"stopped": []}
    await side.close()
    await asyncio.sleep(0.4)  # beyond the 0.2 s grace
    assert agent.svc.convs[V].running() is not None  # neither epoch rule touched the voice turn
    assert agent.brain(V).interrupts == []
    side2 = await Client.open(agent.sock)
    await side2.hello(VOICE_TOKEN, [], epoch="x")
    assert agent.svc.convs[V].record.voice_migrated is True  # migrated by the real hello only


async def test_voice_side_connection_does_not_migrate(agent):
    side = await Client.open(agent.sock)
    assert (await side.hello(VOICE_TOKEN, [], epoch="x"))["ok"]
    assert s.VOICE_CONV not in agent.svc.convs


async def test_voice_ask_does_not_wait_for_the_record_fsync(agent, monkeypatch):
    from agentd import stores

    real = stores.atomic_write_json
    gate = __import__("threading").Event()

    def slow_write(path, obj):
        gate.wait(2)
        real(path, obj)

    c = await agent.client()
    monkeypatch.setattr(stores, "atomic_write_json", slow_write)
    t0 = time.monotonic()
    await c.call({"op": "ask", "conv": V, "turn": 1, "text": "hi"})
    await c.event("text")
    assert time.monotonic() - t0 < 0.5  # the fsync is still blocked, the voice turn is not
    gate.set()
    await c.event("done")
    for _ in range(50):
        await asyncio.sleep(0.02)
        if not agent.svc._save_pending:
            break
    await agent.svc.persist()
    rec = __import__("json").loads((agent.paths.state_dir / "conversations.json").read_text())
    assert rec["conversations"][V]["running"] is None


async def test_chat_steer_false_after_done_and_unacked_until_ack(agent):
    out = agent.root / "jobs.csv"
    out.write_text("a,b\n")
    agent.scripts[CHAT] = lambda t: [fb.say("Finding jobs"),
                                     fb.Step(fb.BrainEventKind.ATTACH,
                                             data={"attach_id": "att1", "path": str(out), "caption": ""}),
                                     fb.text("Found 20.")]
    c = await agent.client(CHAT_TOKEN, [CHAT])
    uid = (await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "find jobs"}))["data"]["turn_uid"]
    att = await c.event("attach")
    assert att["attach_id"] == "att1" and att["path"] == str(out)
    await c.event("done")
    assert (await c.call({"op": "steer", "conv": CHAT, "text": "late"}))["data"] == {"accepted": False}
    st = (await c.call({"op": "status", "conv": CHAT}))["data"]
    s.decode_response({"id": 1, "ok": True, "data": st}, "status")
    assert st["unacked"] == [{"turn_uid": uid, "answer": "Found 20.",
                              "attaches": [{"attach_id": "att1", "path": str(out), "caption": ""}]}]
    assert (await c.call({"op": "ack", "conv": CHAT, "turn_uid": uid}))["ok"]
    assert (await c.call({"op": "status", "conv": CHAT}))["data"]["unacked"] == []


async def test_voice_turn_never_in_unacked(agent):
    c = await agent.client()
    await c.call({"op": "ask", "conv": V, "turn": 1, "text": "hi"})
    await c.event("done")
    assert (await c.call({"op": "status", "conv": V}))["data"]["unacked"] == []


async def test_responses_come_back_in_request_order(agent):
    c = await agent.client(CHAT_TOKEN, [CHAT])
    futs = [c.send({"op": "status", "conv": CHAT}) for _ in range(30)]
    futs.insert(10, c.send({"op": "teleport"}))
    await asyncio.gather(*futs)
    ids = [f["id"] for f in c.frames if "ev" not in f]
    assert ids == sorted(ids)


# ----------------------------------------------------------------------------- V1: confirmations


async def test_voice_confirm_with_question_id_approves(agent):
    agent.scripts[V] = lambda t: [confirm_step()]
    c = await agent.client()
    await c.call({"op": "ask", "conv": V, "turn": 1, "text": "email bob"})
    ev = await c.event("confirm")
    assert ev["target"] == "bob@example.com" and ev["summary"] == "Send the email?"
    st = (await c.call({"op": "status", "conv": V}))["data"]
    assert st["question"]["kind"] == "confirm" and st["question"]["question_id"] == ev["question_id"]
    r = await c.call({"op": "confirm", "conv": V, "question_id": ev["question_id"], "approved": True})
    assert r["data"] == {"applied": True}
    closed = await c.event("question_closed")
    assert closed["outcome"] == "approved"
    assert (await c.event("text"))["text"] == "sent"
    assert agent.brain(V).confirm_results == [True]


async def test_stale_question_never_approves(agent):
    agent.scripts[V] = lambda t: [confirm_step()]
    c = await agent.client()
    await c.call({"op": "ask", "conv": V, "turn": 1, "text": "email bob"})
    ev = await c.event("confirm")
    r = await c.call({"op": "confirm", "conv": V, "question_id": "nope", "approved": True})
    assert r["code"] == "stale_question"
    assert agent.brain(V).gate.pending is not None  # still waiting, not approved
    await c.call({"op": "confirm", "conv": V, "question_id": ev["question_id"], "approved": False})
    assert (await c.event("text"))["text"] == "not sent"
    r = await c.call({"op": "confirm", "conv": V, "question_id": ev["question_id"], "approved": True})
    assert r["code"] == "stale_question"
    assert agent.brain(V).confirm_results == [False]


async def test_confirm_racing_a_new_confirm_never_approves_the_new_one(agent):
    agent.scripts[V] = lambda t: [confirm_step("first?"), confirm_step("second?")]
    c = await agent.client()
    await c.call({"op": "ask", "conv": V, "turn": 1, "text": "two"})
    first = await c.event("confirm")
    agent.brain(V).gate.resolve(False)  # the first one times out inside the brain
    second = await c.event("confirm", pred=lambda e: e["summary"] == "second?")
    r = await c.call({"op": "confirm", "conv": V, "question_id": first["question_id"], "approved": True})
    assert r["code"] == "stale_question"
    await c.call({"op": "confirm", "conv": V, "question_id": second["question_id"], "approved": False})
    await c.event("done")
    assert agent.brain(V).confirm_results == [False, False]


async def test_cancel_sequence_in_order_leaves_no_token_window(agent):
    agent.scripts[V] = lambda t: [confirm_step(), *slow(n=50, delay=0.01)]
    c = await agent.client()
    await c.call({"op": "ask", "conv": V, "turn": 1, "text": "email"})
    ev = await c.event("confirm")
    f1 = c.send({"op": "confirm", "conv": V, "question_id": ev["question_id"], "approved": False})
    f2 = c.send({"op": "revoke_all", "conv": V})
    f3 = c.send({"op": "interrupt", "conv": V, "turn": 1})
    r1, r2, r3 = await asyncio.gather(f1, f2, f3)
    assert r1["data"] == {"applied": True} and r2["ok"] and r3["data"] == {"stopped": True}
    done = await c.event("done")
    assert done["data"]["interrupted"] is True
    gate = agent.brain(V).gate
    assert agent.brain(V).confirm_results == [False] and not gate.has_token(1)


async def test_revoke_all_denies_open_confirm(agent):
    agent.scripts[V] = lambda t: [confirm_step()]
    c = await agent.client()
    await c.call({"op": "ask", "conv": V, "turn": 1, "text": "email"})
    await c.event("confirm")
    await c.call({"op": "revoke_all", "conv": V})
    assert (await c.event("question_closed"))["outcome"] == "denied"
    assert (await c.event("text"))["text"] == "not sent"


# ----------------------------------------------------------------------------- B2: long ops


@pytest.mark.parametrize("op", ["connect_check", "handoff"])
async def test_long_ops_accept_at_once_and_interrupt_within_100ms(agent, op):
    agent.scripts[V] = lambda t: slow(n=500, delay=0.01)
    c = await agent.client()
    await c.call({"op": "warm", "conv": V})
    b = agent.brain(V)
    b.connect_delay = b.handoff_delay = 0.4
    await c.call({"op": "ask", "conv": V, "turn": 1, "text": "long"})
    await c.event("say")
    t0 = time.monotonic()
    f_long = c.send({"op": op, "conv": V})
    f_int = c.send({"op": "interrupt", "conv": V, "turn": 1})
    r_long, r_int = await asyncio.gather(f_long, f_int)
    done = await c.event("done", timeout=1)
    assert time.monotonic() - t0 < 0.1
    assert r_int["data"] == {"stopped": True} and done["data"]["interrupted"] is True
    assert r_long["data"]["accepted"] is True
    key = "connect_check" if op == "connect_check" else "handoff_done"
    st = await c.event("status", timeout=4, pred=lambda e: key in e["data"])
    if op == "connect_check":
        assert st["data"]["connect_check"]["check_id"] == r_long["data"]["check_id"]
        assert st["data"]["connect_check"]["auth"] == "ok"
    else:
        assert st["data"]["handoff_done"] == {"ok": True}


# ----------------------------------------------------------------------------- B1 / V2: epochs


async def test_chat_new_epoch_sees_and_stops_the_running_turn(agent):
    agent.scripts[CHAT] = lambda t: slow(n=300, delay=0.01)
    c1 = await agent.client(CHAT_TOKEN, [CHAT], epoch="old")
    uid = (await c1.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "long"}))["data"]["turn_uid"]
    last = await c1.event("say")
    await c1.close()
    c2 = await Client.open(agent.sock)
    await c2.hello(CHAT_TOKEN, [CHAT], epoch="new")
    r = await c2.call({"op": "subscribe", "conv": CHAT, "after": {"boot": last["boot"], "seq": last["seq"]}})
    assert r["ok"]
    ev = await c2.event("say")
    assert ev["turn_uid"] == uid and ev["seq"] == last["seq"] + 1  # replay, no gap, same turn
    assert (await c2.call({"op": "status", "conv": CHAT}))["data"]["running"]["uid"] == uid
    assert (await c2.call({"op": "steer", "conv": CHAT, "text": "also x"}))["data"] == {"accepted": True}
    assert (await c2.call({"op": "steer", "conv": CHAT, "turn_uid": "zz", "text": "x"}))["data"] == {"accepted": False}
    assert (await c2.call({"op": "interrupt", "conv": CHAT}))["data"] == {"stopped": True}
    done = await c2.event("done")
    assert done["turn_uid"] == uid and done["data"]["interrupted"] is True


async def test_voice_new_epoch_interrupts_old_turn_and_filters_its_events(agent):
    agent.scripts[V] = lambda t: slow(n=300, delay=0.01) if t == "long" else None
    c1 = await agent.client(epoch="old")
    await c1.call({"op": "ask", "conv": V, "turn": 1, "text": "long"})
    await c1.event("say")
    c2 = await agent.client(epoch="new")
    # the old turn is interrupted; the new client never sees its events
    await c1.event("done", timeout=2)
    assert (await c2.call({"op": "interrupt", "conv": V, "turn": 1}))["data"] == {"stopped": False}
    r = await c2.call({"op": "ask", "conv": V, "turn": 1, "text": "short"})
    assert r["ok"]
    done = await c2.event("done")
    assert done["turn_uid"] == r["data"]["turn_uid"]
    assert all(f.get("turn_uid", r["data"]["turn_uid"]).split(":")[-2] == "new"
               for f in c2.frames if "ev" in f)


async def test_voice_gone_over_grace_interrupts_turn(agent):
    agent.scripts[V] = lambda t: slow(n=300, delay=0.01)
    c = await agent.client(epoch="e1")
    await c.call({"op": "ask", "conv": V, "turn": 1, "text": "long"})
    await c.event("say")
    await c.close()
    await asyncio.sleep(0.6)  # voice_grace_s is 0.2 in tests
    assert agent.brain(V).interrupts
    assert agent.svc.convs[V].running() is None


async def test_chat_disconnect_keeps_turn_running(agent):
    agent.scripts[CHAT] = lambda t: slow(n=60, delay=0.01)
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "long"})
    await c.event("say")
    await c.close()
    await asyncio.sleep(0.4)
    assert agent.brain(CHAT).interrupts == []
    c2 = await agent.client(CHAT_TOKEN, [CHAT], epoch="e2")
    for _ in range(50):
        if not (await c2.call({"op": "status", "conv": CHAT}))["data"].get("running"):
            break
        await asyncio.sleep(0.05)
    st = (await c2.call({"op": "status", "conv": CHAT}))["data"]
    assert len(st["unacked"]) == 1 and st["unacked"][0]["answer"] == ""


# ----------------------------------------------------------------------------- V3: replay and gaps


async def test_subscribe_after_replays_exactly_the_missed_events(agent):
    c = await agent.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "hi"})
    evs = [await c.event()]
    while evs[-1]["ev"] != "done":
        evs.append(await c.event())
    c2 = await agent.client(CHAT_TOKEN, [CHAT], subscribe=False)
    r = await c2.call({"op": "subscribe", "conv": CHAT, "after": {"boot": agent.svc.boot, "seq": evs[1]["seq"]}})
    assert r["data"] == {"boot": agent.svc.boot, "seq": evs[-1]["seq"]}
    await asyncio.sleep(0.05)
    assert [e["seq"] for e in c2.drain()] == [e["seq"] for e in evs[2:]]


async def test_gap_for_unknown_boot_and_too_old_seq(agent):
    c = await agent.client(CHAT_TOKEN, [CHAT], subscribe=False)
    r = await c.call({"op": "subscribe", "conv": CHAT, "after": {"boot": "deadbeef", "seq": 1}})
    assert r["code"] == "gap" and r["data"] == {"boot": agent.svc.boot, "oldest_seq": 1}
    stream = agent.svc.conversation(CHAT).stream
    for _ in range(s.REPLAY_BUFFER_EVENTS + 5):
        stream.emit(s.StatusEvent, data={})
    r = await c.call({"op": "subscribe", "conv": CHAT, "after": {"boot": agent.svc.boot, "seq": 3}})
    assert r["code"] == "gap" and r["data"]["oldest_seq"] == 6
    r = await c.call({"op": "subscribe", "conv": CHAT, "after": {"boot": agent.svc.boot, "seq": 5}})
    assert r["ok"]


async def test_slow_client_is_closed_and_can_resume(short_root):
    a = await make_agent(short_root, queue_events=50)
    try:
        c = await a.client(CHAT_TOKEN, [CHAT])
        stream = a.svc.conversation(CHAT).stream
        for _ in range(60):  # synchronous: the writer cannot run in between
            stream.emit(s.StatusEvent, data={})
        await asyncio.wait_for(c.closed.wait(), 2)
        assert c.errors[-1]["code"] == "slow_client" and c.errors[-1]["id"] is None
        c2 = await a.client(CHAT_TOKEN, [CHAT], subscribe=False)
        r = await c2.call({"op": "subscribe", "conv": CHAT, "after": {"boot": a.svc.boot, "seq": 10}})
        assert r["ok"]
        await asyncio.sleep(0.05)
        assert len(c2.drain()) == 50
    finally:
        await stop_agent(a)


LATER = lambda: time.time() + 3 * 3600  # a reboot hours later: the cut turn is stale


async def test_interrupted_notice_once_across_three_restarts(short_root):
    a = await make_agent(short_root, scripts={CHAT: lambda t: slow(n=20, delay=0.05)})
    c = await a.client(CHAT_TOKEN, [CHAT])
    uid = (await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "find jobs"}))["data"]["turn_uid"]
    await c.event("say")
    await stop_agent(a)  # running stays in the record, like a crash
    seen = []
    for i in range(3):
        a = await make_agent(short_root, boot=f"b{i}", clock=LATER)
        c = await Client.open(a.sock)
        r = await c.hello(CHAT_TOKEN, [CHAT])
        seen += r["data"]["interrupted"]
        if i == 0:
            st = (await c.call({"op": "status", "conv": CHAT}))["data"]
            assert st["question"]["kind"] == "restart" and st.get("running") is None
        await stop_agent(a)
    assert seen == [{"conv": CHAT, "uid": uid, "text": "find jobs"}]


async def test_a_recent_restart_resumes_the_request_by_itself(short_root):
    """Owner, 2026-11-12: a restart must not lose the request. Same session, a note, no question."""
    a = await make_agent(short_root, scripts={CHAT: lambda t: slow(n=20, delay=0.05)})
    c = await a.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "find jobs"})
    await c.event("say")
    await stop_agent(a)
    a = await make_agent(short_root, boot="b2", scripts={CHAT: lambda t: [fb.text("Found.")]})
    try:
        c = await a.client(CHAT_TOKEN, [CHAT])
        assert (await c.event("text"))["text"] == "Found."
        ask = a.brain(CHAT).asks[-1][1]
        assert ask.endswith("find jobs") and "restarted" in ask and "do not start over" in ask
        st = (await c.call({"op": "status", "conv": CHAT}))["data"]
        assert st.get("question") is None
    finally:
        await stop_agent(a)


async def test_restart_question_yes_reruns_the_request(short_root):
    a = await make_agent(short_root, scripts={CHAT: lambda t: slow(n=20, delay=0.05)})
    c = await a.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "find jobs"})
    await c.event("say")
    await stop_agent(a)
    a = await make_agent(short_root, boot="b2", scripts={CHAT: lambda t: [fb.text("Found.")]}, clock=LATER)
    try:
        c = await a.client(CHAT_TOKEN, [CHAT])
        q = (await c.call({"op": "status", "conv": CHAT}))["data"]["question"]
        await c.call({"op": "question_posted", "conv": CHAT, "question_id": q["question_id"], "msg_ref": "m9"})
        r = await c.call({"op": "reply", "conv": CHAT, "text": "yes", "quoted_ref": "m9"})
        assert r["data"] == {"bound": True, "outcome": "approved"}
        assert (await c.event("text"))["text"] == "Found."
        assert a.brain(CHAT).asks[-1][1] == "find jobs"
    finally:
        await stop_agent(a)


async def test_restart_never_approves_an_old_question(short_root):
    a = await make_agent(short_root)
    a.model.reply = '{"folder": "Resume", "session": "new", "confidence": 0.1}'
    proj = a.paths.home / "Resume"
    proj.mkdir()
    a.svc.conversation(CHAT).record.folder = str(proj)
    a.sessions = [SimpleNamespace(cwd=str(proj), last_modified=time.time() * 1000, session_id="s1",
                                  custom_title="Jobs", summary="")]
    c = await a.client(CHAT_TOKEN, [CHAT])
    await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "do the thing"})
    q = await c.event("question")
    await c.call({"op": "question_posted", "conv": CHAT, "question_id": q["question_id"], "msg_ref": "p1"})
    await stop_agent(a)
    a = await make_agent(short_root, boot="b2", clock=LATER)
    try:
        c = await a.client(CHAT_TOKEN, [CHAT])
        st = (await c.call({"op": "status", "conv": CHAT}))["data"]
        assert st["question"]["kind"] == "restart"
        r = await c.call({"op": "reply", "conv": CHAT, "text": "yes", "quoted_ref": "p1"})
        assert r["data"] == {"bound": False}
    finally:
        await stop_agent(a)


# ----------------------------------------------------------------------------- app, pause, misc


async def test_app_status_all_ws_list_and_cancel(agent):
    agent.scripts[CHAT] = lambda t: slow(n=300, delay=0.01)
    chat = await agent.client(CHAT_TOKEN, [CHAT])
    await chat.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "long"})
    await chat.event("say")
    app = await agent.client(APP_TOKEN, [])
    st = (await app.call({"op": "status"}))["data"]
    s.decode_response({"id": 1, "ok": True, "data": st}, "status")
    assert [x["conv"] for x in st["conversations"]] == [CHAT] and st["conversations"][0]["running"]["text"] == "long"
    ws = (await app.call({"op": "ws_list"}))["data"]
    assert ws == {"workspaces": []}
    assert (await app.call({"op": "cancel", "conv": CHAT}))["data"] == {"stopped": True}
    assert (await chat.event("done"))["data"]["interrupted"] is True
    assert (await app.call({"op": "cancel", "conv": "whatsapp:nope"}))["code"] == "bad_args"


async def test_pause_blocks_ask_and_only_app_unpauses(agent):
    c = await agent.client(CHAT_TOKEN, [CHAT])
    assert (await c.call({"op": "pause", "conv": CHAT, "value": True, "reason": "loop"}))["data"] == {"paused": "loop"}
    assert (await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "x"}))["code"] == "paused"
    assert (await c.call({"op": "pause", "conv": CHAT, "value": False}))["code"] == "forbidden"
    app = await agent.client(APP_TOKEN, [])
    assert (await app.call({"op": "pause", "conv": CHAT, "value": False}))["data"] == {"paused": ""}
    assert (await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "x"}))["ok"]


async def test_stop_other_cancels_chat_turns_only(agent):
    agent.scripts[CHAT] = lambda t: slow(n=300, delay=0.01)
    chat = await agent.client(CHAT_TOKEN, [CHAT])
    await chat.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "long"})
    await chat.event("say")
    voice = await agent.client()
    assert (await voice.call({"op": "stop_other"}))["data"] == {"stopped": [CHAT]}
    assert (await chat.event("done"))["data"]["interrupted"] is True


async def test_voice_settings_reach_the_brain(agent):
    c = await agent.client()
    await c.call({"op": "voice_settings", "model": "opus", "chrome": False, "setting_sources": ["user"],
                  "name": "Jarvis", "language": "fr-fr"})
    await c.call({"op": "ask", "conv": V, "turn": 1, "text": "hi"})
    await c.event("done")
    b = agent.brain(V)
    assert b.name == "Jarvis" and b.lang == "fr-fr" and b.spec.model == "opus" and b.spec.chrome is False
    assert b.spec.cwd == agent.paths.voice_cwd


async def test_question_kind_single_enum():
    assert [k.value for k in s.QuestionKind] == ["confirm", "folder_approval", "route", "catch_up", "restart",
                                                 "profile"]
    with pytest.raises(s.SchemaError):
        s.QuestionEvent(conv=CHAT, boot="b", seq=1, question_id="q", kind=s.QuestionKind.CONFIRM, text="", expires_at=0)
    s.QuestionStatus(question_id="q", kind=s.QuestionKind.CONFIRM, text="")


async def test_brain_liveness_is_pushed_on_every_change(agent):
    c = await agent.client()
    alive = lambda ev: ev.get("data", {}).get("alive") is not None
    await c.call({"op": "warm", "conv": V})
    assert (await c.event("status", pred=alive))["data"] == {"alive": True}
    await c.call({"op": "sleep", "conv": V})  # idle shutdown looks the same: the process went away
    assert (await c.event("status", pred=alive))["data"] == {"alive": False}
    await c.call({"op": "warm", "conv": V})
    assert (await c.event("status", pred=alive))["data"] == {"alive": True}
    conv = agent.svc.convs[V]
    conv._on_alive(True)  # a repeated callback with no change pushes nothing
    first = agent.brain(V)
    await first._out.put(None)  # the brain's event stream ends: it crashed
    assert (await c.event("status", pred=alive))["data"] == {"alive": False}
    assert conv.brain is None
    assert (await c.call({"op": "status", "conv": V}))["data"]["alive"] is False
    await c.call({"op": "warm", "conv": V})  # a new brain, not the dead one
    assert (await c.event("status", pred=alive))["data"] == {"alive": True}
    assert agent.brain(V) is not first


async def test_brain_on_alive_callback_pushes_status(agent):
    c = await agent.client()
    await c.call({"op": "warm", "conv": V})
    alive = lambda ev: ev.get("data", {}).get("alive") is not None
    await c.event("status", pred=alive)
    b = agent.brain(V)
    b.alive = False  # ClaudeBrain's idle shutdown, reported through on_alive
    b.spec.on_alive(False)
    assert (await c.event("status", pred=alive))["data"] == {"alive": False}


async def test_chat_brain_reporting_back_after_a_background_task_is_a_new_turn(agent):
    """Claude answers a finished background task on its own (WAKE_TURN_BASE ids): a new turn with
    its own uid, so chatd adopts it and posts it as a new message (live 2026-11-11)."""
    from macd.brain.claude_brain import WAKE_TURN_BASE
    from macd.contracts import BrainEvent, BrainEventKind

    c = await agent.client(CHAT_TOKEN, [CHAT])
    uid = (await c.call({"op": "ask", "conv": CHAT, "turn": 1, "text": "scan in the background"}))["data"]["turn_uid"]
    await c.event("done")
    conv = agent.svc.convs[CHAT]
    wake = WAKE_TURN_BASE + 1
    await conv.brain._out.put(BrainEvent(BrainEventKind.TEXT, wake, text="Found 3 roles."))
    await conv.brain._out.put(BrainEvent(BrainEventKind.DONE, wake))
    ev = await c.event("text", pred=lambda e: e["text"] == "Found 3 roles.")
    assert ev["turn_uid"] != uid
    assert (await c.event("done"))["turn_uid"] == ev["turn_uid"]
    assert conv.running() is None and conv.record.running is None
