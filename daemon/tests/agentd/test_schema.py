"""agentd.schema matches docs/agentd-api.md: every op, response, event, error code and the
kind policy, plus a round trip of every example in the doc."""

from __future__ import annotations

import json
import re
from dataclasses import fields
from pathlib import Path
from typing import get_args

import pytest

from agentd import schema as s

DOC = (Path(__file__).resolve().parents[3] / "docs" / "agentd-api.md").read_text()


def _section(title_prefix: str) -> str:
    parts = re.split(r"^## ", DOC, flags=re.MULTILINE)
    return next(p for p in parts if p.startswith(title_prefix))


def _table_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if line.startswith("|") and not re.match(r"^\|[-| ]+\|$", line):
            cells, cur, code = [], "", False  # a | inside `code` is not a cell border
            for ch in line.strip()[1:-1]:
                if ch == "`":
                    code = not code
                if ch == "|" and not code:
                    cells.append(cur.strip())
                    cur = ""
                else:
                    cur += ch
            rows.append([*cells, cur.strip()])
    return rows


def _first_col_names(text: str) -> list[tuple[str, list[str]]]:
    out = []
    for cells in _table_rows(text):
        m = re.fullmatch(r"`(\w+)`", cells[0])
        if m:
            out.append((m.group(1), cells))
    return out


def _strip_nested(t: str) -> str:
    prev = None
    while prev != t:
        prev = t
        t = re.sub(r"\[[^\[\]]*\]|\{[^{}]*\}", "", t)
    return t


def _names(arg_text: str) -> set[str]:
    """`conv, turn:int, attachments?:[...]` -> {conv, turn, attachments}. Per-kind labels
    ("voice:", "chat:") and prose after a period are dropped."""
    t = re.sub(r"\b(voice|chat):", ",", arg_text)
    t = _strip_nested(t.replace("`", ""))
    names = set()
    for piece in re.split(r"[,.]", t):
        m = re.match(r"\s*(\w+)\??\s*(:|$)", piece)
        if m and m.group(1) not in ("Voice", "Chat"):
            names.add(m.group(1))
    return names


def _field_names(cls) -> set[str]:
    return {f.name for f in fields(cls)}


# ----------------------------------------------------------------------------- doc vs schema

OP_ROWS = _first_col_names(_section("4. Ops"))
EV_ROWS = _first_col_names(_section("6. Events"))
CODE_ROWS = _first_col_names(_section("7. Error codes"))


def test_every_doc_op_is_in_the_schema_and_back():
    doc_ops = {name for name, _ in OP_ROWS} | {"hello"}
    assert doc_ops == set(s.OPS)


@pytest.mark.parametrize("name,cells", OP_ROWS, ids=[n for n, _ in OP_ROWS])
def test_op_args_match_doc(name, cells):
    doc_args = _names(cells[1]) if len(cells) > 1 else set()
    assert doc_args <= _field_names(s.OPS[name]), f"{name}: doc args {doc_args}"


def _data_keys(col: str) -> tuple[bool, set[str]]:
    col = col.replace("`", "")
    m = re.search(r"\[\{([^}]*)\}\]", col)
    if m:
        return True, _names(m.group(1))
    m = re.search(r"\{([^{}]*)\}", col)
    return False, (_names(m.group(1)) if m else set())


@pytest.mark.parametrize("name,cells", [r for r in OP_ROWS if r[0] != "status"],
                         ids=[n for n, _ in OP_ROWS if n != "status"])
def test_response_data_matches_doc(name, cells):
    is_list, keys = _data_keys(cells[2])
    tp = s.RESPONSE_DATA[name]
    if is_list:
        (tp,) = get_args(tp)
    assert keys <= _field_names(tp), f"{name}: doc data keys {keys}"


def test_every_doc_event_is_in_the_schema_and_back():
    assert {name for name, _ in EV_ROWS} == set(s.EVENTS)


@pytest.mark.parametrize("name,cells", EV_ROWS, ids=[n for n, _ in EV_ROWS])
def test_event_fields_match_doc(name, cells):
    assert _names(cells[1]) <= _field_names(s.EVENTS[name])


def test_error_codes_match_doc():
    assert {name for name, _ in CODE_ROWS} == {c.value for c in s.ErrorCode}


def test_event_error_kinds_match_doc():
    row = dict(EV_ROWS)["error"]
    kinds = set(re.findall(r"\b([a-z_]+)\b", row[2].split("∈")[1].split(".")[0]))
    assert kinds == {k.value for k in s.EventErrorKind}


def test_single_question_kind_enum_matches_doc():
    assert "One enum in `agentd/schema.py`: `confirm|folder_approval|route|catch_up|restart`" in DOC
    assert [k.value for k in s.QuestionKind] == ["confirm", "folder_approval", "route", "catch_up", "restart"]


def test_close_and_reply_outcomes_match_doc():
    closed = dict(EV_ROWS)["question_closed"][1]
    assert set(closed.split("outcome:")[1].strip(" `").split("|")) == {o.value for o in s.CloseOutcome}
    reply = dict(OP_ROWS)["reply"][2]
    assert set(re.search(r"outcome\?: ([a-z|]+)", reply).group(1).split("|")) == {o.value for o in s.ReplyOutcome}


def test_kind_policy_matches_doc():
    rows = {c[0].strip("`"): c for c in _table_rows(_section("3. Identity")) if c[0].startswith("`")}
    for kind in (s.ClientKind.VOICE, s.ClientKind.CHAT, s.ClientKind.APP):
        doc = {x.strip() for x in rows[kind.value][2].replace("(all conversations)", "").split(",")}
        assert doc | {"hello"} == s.KIND_OPS[kind], kind


def test_api_version_and_limits_match_doc():
    assert DOC.startswith(f"# agentd API, version {s.API_VERSION}")
    assert "at most 1 MiB per line" in DOC and s.MAX_LINE_BYTES == 1 << 20
    assert "last 2,000 events" in DOC and s.REPLAY_BUFFER_EVENTS == 2000
    assert "queue of 10,000 events" in DOC and s.CLIENT_QUEUE_EVENTS == 10_000


# ----------------------------------------------------------------------------- doc examples

def _json_blocks() -> list[dict]:
    return [json.loads(b) for b in re.findall(r"```json\n(.*?)```", DOC, flags=re.DOTALL)]


def test_doc_has_the_three_json_examples():
    assert len(_json_blocks()) == 3


def test_hello_request_example_round_trips():
    ex = _json_blocks()[0]
    req_id, op = s.decode_request(ex)
    assert isinstance(op, s.Hello) and op.conversations == ["voice:local"]
    assert s.encode_request(req_id, op) == ex


def test_hello_reply_example_round_trips():
    ex = _json_blocks()[1]
    r = s.decode_response(ex, "hello")
    assert isinstance(r.data, s.HelloData) and r.data.kind is s.ClientKind.VOICE
    assert r.data.interrupted[0].conv == "voice:local"
    assert s.encode_ok(r.id, r.data) == ex


def test_status_reply_example_round_trips():
    ex = _json_blocks()[2]
    data = s._coerce(s.RESPONSE_DATA["status"], ex, "data")
    assert isinstance(data, s.ConversationStatus)
    assert data.question.kind is s.QuestionKind.CONFIRM and data.unacked[0].attaches[0].caption == ""
    assert s.to_json(data) == ex


def test_version_mismatch_text_is_the_doc_text():
    m = re.search(r'`error: "(agentd speaks API 1\.0 [^"]+)"`', DOC)
    assert m and "Restart Mac to update all parts together." in m.group(1)


# ----------------------------------------------------------------------------- every op, response and event

REQUESTS = {
    "hello": {"api": "1.0", "token": "ab" * 32, "client": "chatd/0.9.0", "client_epoch": "cd" * 16,
              "conversations": ["whatsapp:1203@g.us"]},
    "ask": {"conv": "whatsapp:1203@g.us", "turn": 3, "text": "find me 20 jobs",
            "attachments": [{"path": "/tmp/a.pdf", "kind": "document", "name": "a.pdf"}],
            "meta": {"replayed": False, "sent_at": 1762790000.5, "msg_ref": "3EB0"}},
    "steer": {"conv": "whatsapp:1203@g.us", "turn_uid": "b:c:e:3", "text": "only London"},
    "interrupt": {"conv": "voice:local", "turn": 2},
    "new_session": {"conv": "whatsapp:1203@g.us", "folder": "~/Desktop/Personal/Mac"},
    "reset": {"conv": "voice:local"}, "warm": {"conv": "voice:local"}, "sleep": {"conv": "voice:local"},
    "handoff": {"conv": "voice:local"}, "cancel": {"conv": "whatsapp:1203@g.us"}, "stop_other": {},
    "confirm": {"conv": "voice:local", "question_id": "q1", "approved": True},
    "revoke_all": {"conv": "voice:local"},
    "question_posted": {"conv": "whatsapp:1203@g.us", "question_id": "q2", "msg_ref": "3EB1"},
    "reply": {"conv": "whatsapp:1203@g.us", "text": "yes", "quoted_ref": "3EB1", "reaction": "👍", "reaction_on": "3EB1"},
    "set_context": {"conv": "voice:local", "locked": True},
    "set_language": {"conv": "voice:local", "lang": "fr-fr"},
    "set_name": {"conv": "voice:local", "name": "Jarvis"},
    "voice_settings": {"model": "", "chrome": True, "setting_sources": ["user"], "name": "Mac", "language": "en-us"},
    "connect_check": {"conv": "voice:local", "deep": True},
    "subscribe": {"conv": "whatsapp:1203@g.us", "after": {"boot": "00ff", "seq": 41}},
    "ack": {"conv": "whatsapp:1203@g.us", "turn_uid": "b:c:e:3"},
    "status": {"conv": "voice:local"},
    "pause": {"conv": "whatsapp:1203@g.us", "value": True, "reason": "loop"},
    "ws_list": {}, "ws_approve": {"path": "~/Desktop/Personal/Resume"}, "ws_remove": {"path": "~/x"},
}

RESPONSES = {
    "hello": {"api": "1.0", "server": "agentd/0.9.0", "kind": "chat", "boot": "00ff",
              "conversations": ["whatsapp:1203@g.us"], "interrupted": []},
    "ask": {"turn_uid": "00ff:whatsapp:1203@g.us:cdcd:3"}, "steer": {"accepted": False},
    "interrupt": {"stopped": True}, "new_session": {}, "reset": {}, "warm": {}, "sleep": {},
    "handoff": {"accepted": True}, "cancel": {"stopped": False}, "stop_other": {"stopped": ["whatsapp:1203@g.us"]},
    "confirm": {"applied": True}, "revoke_all": {}, "question_posted": {},
    "reply": {"bound": True, "outcome": "approved"}, "set_context": {}, "set_language": {}, "set_name": {},
    "voice_settings": {}, "connect_check": {"accepted": True, "check_id": "c1"},
    "subscribe": {"boot": "00ff", "seq": 41}, "ack": {},
    "status": {"conversations": [{"conv": "voice:local", "alive": False, "session_title": "", "paused": "",
                                  "unacked": [], "loop_lag_ms_p99": 0.4}]},
    "pause": {"paused": True}, "ws_list": [{"path": "~/R", "name": "R", "approved": True, "settings_changed": False}],
    "ws_approve": {"settings_hash": "ab" * 32}, "ws_remove": {},
}

T = {"conv": "voice:local", "boot": "00ff", "seq": 7, "turn": 2, "turn_uid": "00ff:voice:local:cdcd:2"}
NT = {"conv": "whatsapp:1203@g.us", "boot": "00ff", "seq": 8}
EVENTS = [
    {"ev": "text", **T, "text": "Looking now.", "tool_follows": True},
    {"ev": "say", **T, "text": "Checking LinkedIn"},
    {"ev": "tool_start", **T, "tool": "mcp__claude-in-chrome__navigate", "tool_use_id": "tu1", "narration": "Opening Chrome."},
    {"ev": "tool_end", **T, "tool": "mcp__claude-in-chrome__navigate", "tool_use_id": "tu1", "is_error": False},
    {"ev": "confirm", **T, "question_id": "q1", "summary": "Email recruiter. Shall I go ahead?", "target": "recruiter@acme.com"},
    {"ev": "question", **NT, "question_id": "q2", "kind": "route", "text": "Mac · Resume, or generally?", "expires_at": 1762790000.0},
    {"ev": "question_closed", **NT, "question_id": "q2", "outcome": "skipped"},
    {"ev": "attach", **T, "attach_id": "a1", "path": "/Users/x/Documents/Mac/jobs.csv", "caption": ""},
    {"ev": "status", **NT, "data": {"handoff_done": {"ok": True}}},
    {"ev": "status", **T, "data": {"connect_check": {"check_id": "c1", "auth": "ok"}}},
    {"ev": "done", **T, "data": {"interrupted": False, "duration_s": 12.5, "cost": 0.03}},
    {"ev": "error", **T, "kind": "rate_limit", "message": "I've hit the Claude usage limit."},
    {"ev": "error", **NT, "kind": "paused", "message": "Paused."},
    {"ev": "interrupted", **NT, "uid": "00aa:whatsapp:1203@g.us:cdcd:1", "text": "find me 20 jobs"},
]


def test_examples_cover_every_op_and_event():
    assert set(REQUESTS) == set(s.OPS) == set(RESPONSES)
    assert {e["ev"] for e in EVENTS} == set(s.EVENTS)


@pytest.mark.parametrize("op", sorted(REQUESTS))
def test_request_round_trip(op):
    wire = {"id": 9, "op": op, **REQUESTS[op]}
    req_id, parsed = s.decode_request(s.parse_line(s.encode_line(wire)))
    assert req_id == 9 and parsed.OP == op
    assert s.encode_request(req_id, parsed) == wire


@pytest.mark.parametrize("op", sorted(RESPONSES))
def test_response_round_trip(op):
    wire = {"id": 9, "ok": True, "data": RESPONSES[op]}
    r = s.decode_response(s.parse_line(s.encode_line(wire)), op)
    assert s.encode_ok(r.id, r.data) == wire


@pytest.mark.parametrize("wire", EVENTS, ids=[f"{e['ev']}{i}" for i, e in enumerate(EVENTS)])
def test_event_round_trip(wire):
    ev = s.decode_event(s.parse_line(s.encode_line(wire)))
    assert s.encode_event(ev) == wire


@pytest.mark.parametrize("code", list(s.ErrorCode))
def test_error_response_round_trip(code):
    wire = s.encode_error(4, code, "plain words")
    r = s.decode_response(wire)
    assert isinstance(r, s.ErrorResponse) and r.code is code and s.encode_error(r.id, r.code, r.error) == wire


def test_gap_error_carries_position():
    wire = s.encode_error(5, s.ErrorCode.GAP, "too old", s.GapData(boot="00ff", oldest_seq=120))
    assert wire["data"] == {"boot": "00ff", "oldest_seq": 120}
    r = s.decode_response(wire)
    assert s.from_dict(s.GapData, r.data).oldest_seq == 120


def test_frame_level_error_has_null_id():
    r = s.decode_response(s.encode_error(None, s.ErrorCode.BAD_FRAME, "invalid JSON"))
    assert r.id is None and r.code is s.ErrorCode.BAD_FRAME


# ----------------------------------------------------------------------------- validation

def test_unknown_fields_are_ignored_everywhere():
    _, op = s.decode_request({"id": 1, "op": "ack", "conv": "c", "turn_uid": "u", "future": 1})
    assert s.encode_request(1, op) == {"id": 1, "op": "ack", "conv": "c", "turn_uid": "u"}
    ev = s.decode_event({"ev": "say", **T, "text": "x", "new_field": [1]})
    assert not hasattr(ev, "new_field")
    r = s.decode_response({"id": 1, "ok": True, "data": {"accepted": True, "extra": 2}}, "steer")
    assert r.data == s.AcceptedData(accepted=True)


def test_unknown_op_and_event():
    with pytest.raises(s.SchemaError) as e:
        s.decode_request({"id": 1, "op": "fly"})
    assert e.value.code is s.ErrorCode.UNKNOWN_OP
    with pytest.raises(s.SchemaError):
        s.decode_event({"ev": "nope", **NT})


@pytest.mark.parametrize("bad", [
    {"id": 1, "op": "ask", "conv": "c", "text": "hi"},                    # missing turn
    {"id": 1, "op": "ask", "conv": "c", "turn": True, "text": "hi"},      # bool is not an int
    {"id": 1, "op": "ask", "conv": "c", "turn": "3", "text": "hi"},
    {"id": 1, "op": "confirm", "conv": "c", "question_id": "q", "approved": 1},
    {"id": 1, "op": "pause", "conv": "c", "value": True, "reason": "bored"},
    {"id": 1, "op": "ask", "conv": "c", "turn": 1, "text": "x", "attachments": [{"path": "p", "kind": "audio", "name": "n"}]},
    {"id": True, "op": "status"},
    {"id": 1, "op": "subscribe", "conv": "c", "after": {"boot": "b"}},
])
def test_bad_args(bad):
    with pytest.raises(s.SchemaError) as e:
        s.decode_request(bad)
    assert e.value.code is s.ErrorCode.BAD_ARGS


def test_bad_frames():
    for line in (b"not json", b"[1,2]", b"\xff\xfe", b'{"id": 1' + b" " * (1 << 20) + b"}"):
        with pytest.raises(s.SchemaError) as e:
            s.parse_line(line)
        assert e.value.code is s.ErrorCode.BAD_FRAME


def test_question_event_rejects_confirm_kind_but_status_allows_it():
    with pytest.raises(s.SchemaError):
        s.decode_event({"ev": "question", **NT, "question_id": "q", "kind": "confirm", "text": "t", "expires_at": 1.0})
    with pytest.raises(s.SchemaError):
        s.decode_event({"ev": "question", **NT, "question_id": "q", "kind": "chat", "text": "t", "expires_at": 1.0})
    st = s.from_dict(s.QuestionStatus, {"question_id": "q", "kind": "confirm", "text": "t"})
    assert st.kind is s.QuestionKind.CONFIRM


def test_turn_events_need_turn_and_uid():
    with pytest.raises(s.SchemaError):
        s.decode_event({"ev": "text", **NT, "text": "x"})
    assert s.decode_event({"ev": "status", **NT, "data": {}}).turn is None


def test_kind_policy_checks():
    s.check_op_for_kind(s.ClientKind.VOICE, s.Steer(conv="voice:local", turn=1, text="x"))
    s.check_op_for_kind(s.ClientKind.CHAT, s.Steer(conv="whatsapp:x", text="x"))
    with pytest.raises(s.SchemaError) as e:
        s.check_op_for_kind(s.ClientKind.VOICE, s.Interrupt(conv="voice:local"))
    assert e.value.code is s.ErrorCode.BAD_ARGS
    for kind, op in ((s.ClientKind.CHAT, s.Confirm(conv="c", question_id="q", approved=True)),
                     (s.ClientKind.VOICE, s.Reply(conv="voice:local", text="yes")),
                     (s.ClientKind.APP, s.Ask(conv="c", turn=1, text="x")),
                     (s.ClientKind.CHAT, s.WsApprove(path="~"))):
        with pytest.raises(s.SchemaError) as e:
            s.check_op_for_kind(kind, op)
        assert e.value.code is s.ErrorCode.FORBIDDEN


def test_turn_uid_format():
    assert s.turn_uid("00ff", "voice:local", "cdcd", 2) == "00ff:voice:local:cdcd:2"


def test_encode_line_is_one_compact_line():
    raw = s.encode_line(s.encode_event(s.decode_event(EVENTS[0])))
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1 and b'", "' not in raw
