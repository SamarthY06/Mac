"""Security audit 2026-11-10 (docs/reviews/security-audit-2026-11-10.md), builder B: H1 write
protection of Mac's own code, state and startup files and the owner's shell/Claude setup, the
launchctl and rm rules, M1 (no message text in the gate log) and L1 (zero-width never-list)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from agentd.schema import CAPABILITIES
from macd.brain.gate import ConfirmationGate, GateContext, analyze

HOME = "/Users/alice"
MAC = f"{HOME}/.mac-agent"
WS = f"{MAC}/workspace"
REPO = str(Path(__file__).resolve().parents[2])  # .../Mac
ALL = frozenset(CAPABILITIES)


def chat(**kw: Any) -> GateContext:
    return GateContext(home=HOME, workspace=f"{MAC}/agent/chats/abc", user="alice", whatsapp_ui_block=True,
                       never_chat_names=("Timetable",), capabilities=ALL, **kw)


def voice(**kw: Any) -> GateContext:
    return GateContext(home=HOME, workspace=WS, user="alice", **kw)


def writes(path: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        ("Write", {"file_path": path, "content": "x"}),
        ("Edit", {"file_path": path, "old_string": "a", "new_string": "b"}),
        ("Bash", {"command": f"echo x >> {path}"}),
        ("Bash", {"command": f"cp /tmp/evil {path}"}),
        ("Bash", {"command": f"sed -i '' s/a/b/ {path}"}),
        ("Bash", {"command": f"python3 -c 'open(\"{path}\", \"w\").write(\"x\")'"}),
        ("mcp__filesystem__write_file", {"path": path, "content": "x"}),
    ]


MAC_OWN = [
    f"{MAC}/release/current/daemon/src/macd/brain/gate.py",
    f"{MAC}/release/.complete",
    f"{MAC}/state.json",
    f"{MAC}/models/x.onnx",
    f"{MAC}/voice/sample.wav",
    f"{MAC}/chat/whatsapp/media/in/a.jpg",
    f"{REPO}/daemon/src/macd/brain/hooks.py",
    f"{REPO}/daemon/.venv/lib/python3.12/site-packages/evil.pth",
    f"{HOME}/Library/LaunchAgents/evil.plist",
    f"{HOME}/.local/bin/claude",
]
OWNER = [f"{HOME}/.claude/settings.json", f"{HOME}/.claude/settings.local.json", f"{HOME}/.claude.json",
         f"{HOME}/.claude/CLAUDE.md", f"{HOME}/.claude/hooks/pre.sh", f"{HOME}/.zshrc", f"{HOME}/.zprofile",
         f"{HOME}/.bashrc", f"{HOME}/.bash_profile", f"{HOME}/.profile"]


@pytest.mark.parametrize("path", MAC_OWN)
@pytest.mark.parametrize("ctx", [chat, voice])
def test_mac_own_code_state_and_startup_files_are_never_written(path: str, ctx: Any) -> None:
    for tool, inp in writes(path):
        d = analyze(tool, inp, ctx())
        assert d.always_denied, (tool, inp, d)


@pytest.mark.parametrize("path", OWNER)
def test_owner_setup_never_from_chat_and_behind_a_yes_by_voice(path: str) -> None:
    for tool, inp in writes(path):
        assert analyze(tool, inp, chat()).always_denied, (tool, inp)
        if not tool.startswith("mcp__"):  # an unparsed MCP write is outward anyway
            d = analyze(tool, inp, voice())
            assert (d.verdict, d.code) == ("outward", "owner_config"), (tool, inp, d)
    assert analyze("Read", {"file_path": path}, chat()).verdict == "allow"  # reading is fine
    assert analyze("Bash", {"command": f"cat {path}"}, voice()).verdict == "allow"


def test_usable_folders_stay_writable() -> None:
    for p in (f"{WS}/notes.md", f"{MAC}/agent/chats/abc/out.csv", f"{MAC}/agent/chat-workspace/a.md",
              f"{HOME}/Documents/Mac/list.csv", f"{HOME}/Desktop/Personal/other/app.py"):
        assert analyze("Write", {"file_path": p, "content": "x"}, chat()).verdict == "allow", p
        assert analyze("Bash", {"command": f"echo x > {p}"}, voice()).verdict == "allow", p
    assert analyze("Read", {"file_path": f"{MAC}/chat/whatsapp/media/in/a.jpg"}, chat()).verdict == "allow"
    assert analyze("Bash", {"command": "source ~/.zshrc && ls"}, voice()).verdict == "allow"
    assert analyze("Bash", {"command": f"git -C {REPO} status"}, voice()).verdict == "allow"


@pytest.mark.parametrize("cmd", [
    "launchctl bootout gui/501/com.mac-agent.services",
    "launchctl unload ~/Library/LaunchAgents/com.mac-agent.services.plist",
    "launchctl remove com.mac-agent.agentd",
    f"rm -rf {REPO}",
    f"rm -rf {REPO}/daemon",
    "rm -rf ~/.mac-agent",
])
@pytest.mark.parametrize("ctx", [chat, voice])
def test_stopping_or_deleting_mac_is_catastrophic(cmd: str, ctx: Any) -> None:
    assert analyze("Bash", {"command": cmd}, ctx()).verdict == "catastrophic", cmd


def test_launchctl_of_other_services_is_unchanged() -> None:
    assert analyze("Bash", {"command": "launchctl list"}, voice()).verdict == "allow"
    assert not analyze("Bash", {"command": "launchctl bootout gui/501/com.example.app"}, voice()).always_denied


async def test_confirmation_log_holds_no_message_text(caplog) -> None:
    import asyncio

    gate = ConfirmationGate()
    secret = "send 'I love you Priya, meet at 9' to Priya"
    with caplog.at_level(logging.DEBUG, logger="macd.brain.gate"):
        asyncio.get_running_loop().call_later(0.01, gate.resolve, True)
        assert await gate.request(1, secret, 2, "mcp__x__send|mcp_write|priya")
    text = caplog.text
    assert "love" not in text and "meet at 9" not in text and secret not in text
    assert "tool=mcp__x__send" in text and "code=mcp_write" in text and "ref=" in text


@pytest.mark.parametrize("name", ["Time\u200btable", "Time‌table", "﻿Timetable", "Time­table"])
def test_never_list_ignores_zero_width_and_format_characters(name: str) -> None:
    c = chat()
    assert analyze("mcp__mac__request_confirmation", {"summary": f"send hi to {name}"}, c).code == "never_chat"
    assert analyze("mcp__whatsapp__list_messages", {"chat": name}, c).code == "never_chat"
