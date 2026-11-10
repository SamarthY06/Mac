"""Gate profile rules (WhatsApp ARCHITECTURE W3.3): M1 whatsapp_ui for the chat profile (reads
included), the never-list for chat, voice Option A (V6), F10 .env exemption, mac.send_file."""

from __future__ import annotations

from typing import Any

import pytest

from macd.brain.gate import GateContext, analyze

HOME = "/Users/alice"
WS = f"{HOME}/.mac-agent/workspace"
CHAT_WS = f"{HOME}/.mac-agent/agent/chat-workspace"
PROJECT = f"{HOME}/Desktop/Personal/Mac"
TABS = {"7": "web.whatsapp.com", "8": "example.com", "9": "mail.google.com"}
CHROME = "mcp__claude-in-chrome__"


def chat(app: str = "", **kw: Any) -> GateContext:
    return GateContext(home=HOME, workspace=CHAT_WS, user="alice", whatsapp_ui_block=True,
                       never_chat_names=("Timetable",), chrome_tab_host=lambda t: TABS.get(str(t)),
                       current_app=lambda: app, **kw)


def voice(app: str = "") -> GateContext:
    return GateContext(home=HOME, workspace=WS, user="alice", chrome_tab_host=lambda t: TABS.get(str(t)),
                       current_app=lambda: app)


# ----------------------------------------------------------------------------- M1 (chat)

WHATSAPP_UI_CALLS: list[tuple[str, dict[str, Any], str]] = [
    (f"{CHROME}navigate", {"url": "https://web.whatsapp.com/", "tabId": 8}, ""),
    (f"{CHROME}navigate", {"url": "web.whatsapp.com", "tabId": 8}, ""),
    (f"{CHROME}navigate", {"url": "https://wa.me/15551234567", "tabId": 8}, ""),
    (f"{CHROME}read_page", {"tabId": 7}, ""),
    (f"{CHROME}get_page_text", {"tabId": 7}, ""),
    (f"{CHROME}find", {"tabId": 7, "query": "chat list"}, ""),
    (f"{CHROME}computer", {"action": "screenshot", "tabId": 7}, ""),
    (f"{CHROME}computer", {"action": "left_click", "tabId": 7, "coordinate": [10, 10]}, ""),
    (f"{CHROME}javascript_tool", {"tabId": 7, "text": "document.title"}, ""),
    (f"{CHROME}tabs_create_mcp", {"url": "https://web.whatsapp.com"}, ""),
    ("mcp__mac__screenshot", {}, "WhatsApp"),
    ("mcp__mac__ui_tree", {"app": "WhatsApp"}, ""),
    ("mcp__mac__ui_tree", {}, "WhatsApp"),
    ("mcp__mac__ui_click", {"x": 1, "y": 2}, "WhatsApp"),
    ("mcp__mac__ui_type", {"text": "hello"}, "WhatsApp"),
    ("mcp__mac__ui_key", {"combo": "cmd+f"}, "WhatsApp"),
    ("mcp__mac__open_app", {"name": "WhatsApp"}, ""),
    ("Bash", {"command": "osascript -e 'tell application \"WhatsApp\" to activate'"}, ""),
    ("Bash", {"command": "osascript -e 'tell application \"System Events\" to tell process \"WhatsApp\" "
                         "to get name of every window'"}, ""),
    ("Bash", {"command": "osascript -l JavaScript -e 'Application(\"WhatsApp\").activate()'"}, ""),
    ("Bash", {"command": "open -a WhatsApp"}, ""),
    ("Bash", {"command": "open -b net.whatsapp.WhatsApp"}, ""),
    ("Bash", {"command": "open 'whatsapp://send?phone=1555&text=hi'"}, ""),
    ("Bash", {"command": "cd /tmp && open /Applications/WhatsApp.app"}, ""),
]


@pytest.mark.parametrize("tool,inp,app", WHATSAPP_UI_CALLS)
def test_chat_profile_whatsapp_ui_is_catastrophic_reads_included(tool: str, inp: dict[str, Any], app: str) -> None:
    d = analyze(tool, inp, chat(app))
    assert (d.verdict, d.code) == ("catastrophic", "whatsapp_ui"), (tool, inp, d)
    assert "—" not in d.reason


@pytest.mark.parametrize("tool,inp,app", [
    (f"{CHROME}get_page_text", {"tabId": 8}, ""),
    (f"{CHROME}navigate", {"url": "https://example.com/whatsapp-business-guide", "tabId": 8}, ""),
    (f"{CHROME}navigate", {"url": "https://java.me", "tabId": 8}, ""),
    ("mcp__mac__screenshot", {}, "Finder"),
    ("mcp__mac__open_app", {"name": "Notes"}, ""),
    ("Bash", {"command": "cat whatsapp_notes.md"}, ""),
    ("Bash", {"command": "ls ~/Downloads"}, ""),
    ("Read", {"file_path": f"{CHAT_WS}/whatsapp.txt"}, ""),
])
def test_chat_profile_other_screens_still_work(tool: str, inp: dict[str, Any], app: str) -> None:
    assert analyze(tool, inp, chat(app)).verdict == "allow"


def test_voice_profile_keeps_whatsapp_behind_confirmation_option_a() -> None:
    # V6, Option A: by voice Mac may read and send in WhatsApp; every send needs the spoken yes.
    read = analyze(f"{CHROME}get_page_text", {"tabId": 7}, voice())
    assert read.verdict == "allow"
    click = analyze(f"{CHROME}computer", {"action": "left_click", "tabId": 7, "coordinate": [1, 2]}, voice())
    assert click.verdict == "outward" and click.target == "web.whatsapp.com"
    assert analyze("mcp__mac__ui_key", {"combo": "return"}, voice("WhatsApp")).verdict == "outward"
    assert analyze("mcp__mac__screenshot", {}, voice("WhatsApp")).verdict == "allow"
    assert analyze("mcp__mac__open_app", {"name": "WhatsApp"}, voice()).verdict == "allow"
    # Timetable by voice is allowed behind the confirmation that names it (user decision V6)
    assert analyze("mcp__mac__request_confirmation",
                   {"summary": "send 'hi' to Timetable on WhatsApp"}, voice()).verdict == "allow"


# ----------------------------------------------------------------------------- never-list (chat)

def test_chat_never_list_confirmation_summary_is_catastrophic() -> None:
    d = analyze("mcp__mac__request_confirmation", {"summary": "send the notes to the Timetable group"}, chat())
    assert (d.verdict, d.code) == ("catastrophic", "never_chat")
    ok = analyze("mcp__mac__request_confirmation", {"summary": "email resume.pdf to recruiter@acme.com"}, chat())
    assert ok.verdict == "allow"


def test_chat_never_list_outward_target_is_catastrophic() -> None:
    d = analyze("Bash", {"command": "osascript -e 'tell application \"Messages\" to send \"hi\" to "
                                    "chat \"Timetable\"'"}, chat())
    assert d.verdict == "outward"  # the target is only "messages": the confirmation summary that
    # must name the chat is where the never-list catches it (previous test)
    ui = analyze("mcp__mac__ui_click", {"element_id": "4"}, GateContext(
        home=HOME, workspace=CHAT_WS, user="alice", never_chat_names=("Timetable",),
        element_label=lambda e: "AXButton Send to Timetable", current_app=lambda: "Slack"))
    assert (ui.verdict, ui.code) == ("catastrophic", "never_chat")
    # a word that merely contains the name is not a hit, and reads are untouched
    assert analyze("Read", {"file_path": f"{CHAT_WS}/timetable.md"}, chat()).verdict == "allow"


# ----------------------------------------------------------------------------- F10 .env

def test_env_files_readable_only_in_macs_own_workspace() -> None:
    home_ctx = GateContext(home=HOME, workspace=WS, user="alice")
    assert analyze("Read", {"file_path": f"{WS}/.env"}, home_ctx).verdict == "allow"
    assert analyze("Bash", {"command": "cat .env"}, home_ctx).verdict == "allow"
    # cwd = a real project: its .env holds real keys (F10)
    proj = GateContext(home=HOME, workspace=PROJECT, user="alice")
    for tool, inp in (("Read", {"file_path": f"{PROJECT}/.env"}), ("Bash", {"command": "cat .env"}),
                      ("Bash", {"command": f"cat {PROJECT}/daemon/.env.local"}),
                      ("mcp__mac__send_file", {"path": f"{PROJECT}/.env"})):
        assert analyze(tool, inp, proj).verdict == "sensitive_read", (tool, inp)
    # an explicit exemption pointing at a project is ignored; the chat workspace is honoured
    sneaky = GateContext(home=HOME, workspace=PROJECT, user="alice", env_exempt_dir=PROJECT)
    assert analyze("Read", {"file_path": f"{PROJECT}/.env"}, sneaky).verdict == "sensitive_read"
    routed = GateContext(home=HOME, workspace=PROJECT, user="alice", env_exempt_dir=CHAT_WS)
    assert analyze("Read", {"file_path": f"{CHAT_WS}/.env"}, routed).verdict == "allow"
    assert analyze("Read", {"file_path": f"{PROJECT}/.env"}, routed).verdict == "sensitive_read"
    assert analyze("Read", {"file_path": f"{PROJECT}/.env.example"}, proj).verdict == "allow"


# ----------------------------------------------------------------------------- send_file / test tool

def test_send_file_rules() -> None:
    ctx = chat()
    assert analyze("mcp__mac__send_file", {"path": f"{CHAT_WS}/report.pdf"}, ctx).verdict == "allow"
    assert analyze("mcp__mac__send_file", {"path": "~/.ssh/id_ed25519"}, ctx).verdict == "sensitive_read"
    assert analyze("mcp__mac__send_file", {"path": f"{HOME}/keys/server.pem"}, ctx).verdict == "sensitive_read"
    mem = GateContext(home=HOME, workspace=CHAT_WS, user="alice", memory_path=f"{WS}/memory.md")
    assert analyze("mcp__mac__send_file", {"path": f"{WS}/memory.md"}, mem).verdict == "sensitive_read"
    shared = chat(confirm_send_file=True)
    d = analyze("mcp__mac__send_file", {"path": f"{CHAT_WS}/report.pdf"}, shared)
    assert d.verdict == "outward" and d.target == "report.pdf"


def test_injected_memory_path_is_a_memory_file() -> None:
    ctx = GateContext(home=HOME, workspace=PROJECT, user="alice", memory_path=f"{WS}/memory.md")
    d = analyze("Write", {"file_path": f"{WS}/memory.md", "content": "x"}, ctx)
    assert d.code == "memory_write"
    assert analyze("Edit", {"file_path": f"{PROJECT}/CLAUDE.md"}, ctx).code == "memory_write"


def test_test_outward_tool_always_needs_confirmation() -> None:
    d = analyze("mcp__mac__test_outward", {"target": "Interview Prep"}, chat())
    assert (d.verdict, d.code, d.target) == ("outward", "test_outward", "Interview Prep")
