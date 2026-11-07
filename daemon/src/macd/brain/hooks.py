"""PreToolUse hook: the capability gate the model cannot bypass (ARCHITECTURE §6).

An outward action is allowed only with a confirmation token bound to that action's
fingerprint (tool, gate code, normalised target). When the hook denies an outward call it
records the fingerprint as the *pending action*; ``mcp__mac__request_confirmation`` binds the
token to it, so a "yes" for "send Priya a WhatsApp" cannot be spent on `curl -F` or `git push`.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from macd.brain.gate import (
    MEMORY_WRITE,
    ConfirmationGate,
    GateContext,
    GateDecision,
    analyze,
)

log = logging.getLogger(__name__)

CONFIRM_TOOL = "mcp__mac__request_confirmation"
MEMORY_TOOLS = {"mcp__mac__remember", "mcp__mac__forget", "mcp__mac__rewrite_memory"}
MemoryCheck = Callable[[str, dict[str, Any]], tuple[bool, str]]  # (tool, args) -> (allowed, how|reason)

REASON_CANCELLED = "The user cancelled this request. Stop now and do not call any more tools."
REASON_MEMORY_WRITE = ("Direct edits to CLAUDE.md or memory.md are not allowed. Use mcp__mac__remember "
                       "with a one-line fact, and only when the user asked you to remember something.")
REASON_OUTWARD = ("This action needs the user's spoken confirmation. First call "
                  f"{CONFIRM_TOOL} with a one-sentence summary (who or what it goes to, and the gist). "
                  "If it says the user approved, retry exactly the same action. If declined, don't.")


def _deny(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


@dataclass(frozen=True)
class ActionFingerprint:
    """What a confirmation token is bound to."""

    turn_id: int
    tool: str
    code: str
    target: str

    @property
    def key(self) -> str:
        return f"{self.tool}|{self.code}|{self.target.strip().lower()}"


@dataclass(frozen=True)
class HookDecision:
    t: float
    turn_id: int
    tool_name: str
    tool_use_id: str
    allowed: bool
    verdict: str
    code: str
    reason: str


class PreToolHook:
    """Callable SDK hook (`HookCallback`). `decisions` keeps recent outcomes for tests/diagnostics."""

    def __init__(self, gate: ConfirmationGate, current_turn: Callable[[], int],
                 is_cancelled: Callable[[int], bool], memory_check: MemoryCheck, *,
                 context: GateContext | None = None,
                 restrict: Callable[[str], str | None] | None = None,
                 on_decision: Callable[[HookDecision], None] | None = None) -> None:
        self.gate = gate
        self.current_turn = current_turn
        self.is_cancelled = is_cancelled
        self.memory_check = memory_check
        self.restrict = restrict  # e.g. "memory tools only" during a session handoff
        self.context = context
        self.on_decision = on_decision
        self.decisions: deque[HookDecision] = deque(maxlen=256)
        self.pending_action: ActionFingerprint | None = None  # last denied outward action

    def pending_for(self, turn_id: int) -> ActionFingerprint | None:
        p = self.pending_action
        return p if p is not None and p.turn_id == turn_id else None

    def _record(self, turn: int, name: str, tool_use_id: str, allowed: bool,
                decision: GateDecision, reason: str) -> None:
        rec = HookDecision(time.time(), turn, name, tool_use_id, allowed, decision.verdict,
                           decision.code, reason)
        self.decisions.append(rec)
        level = logging.INFO if allowed and decision.verdict == "allow" else logging.WARNING
        log.log(level, "gate %s turn=%d tool=%s verdict=%s code=%s target=%r %s",
                "ALLOW" if allowed else "DENY", turn, name, decision.verdict, decision.code,
                decision.target, reason)
        if self.on_decision:
            try:
                self.on_decision(rec)
            except Exception:
                log.exception("hook on_decision callback failed")

    def decide(self, name: str, tool_input: dict[str, Any], tool_use_id: str = "") -> tuple[bool, str]:
        turn = self.current_turn()
        if self.is_cancelled(turn):
            d = GateDecision("allow", "cancelled", REASON_CANCELLED)
            self._record(turn, name, tool_use_id, False, d, REASON_CANCELLED)
            return False, REASON_CANCELLED
        restricted = self.restrict(name) if self.restrict else None
        if restricted:
            d = GateDecision("allow", "restricted", restricted)
            self._record(turn, name, tool_use_id, False, d, restricted)
            return False, restricted
        if name in MEMORY_TOOLS:
            allowed, how = self.memory_check(name.removeprefix("mcp__mac__"), tool_input)
            if not allowed:
                reason = f"Memory not changed: {how}. Do not retry."
                d = GateDecision("sensitive_read", "memory_not_grounded", reason)
                self._record(turn, name, tool_use_id, False, d, reason)
                return False, reason
        d = analyze(name, tool_input, self.context)
        if d.verdict == "catastrophic":
            reason = (f"Blocked: {d.reason}. This is never allowed. Don't try another way; "
                      "tell the user you can't do that.")
        elif d.verdict == "sensitive_read" and d.code == MEMORY_WRITE:
            reason = REASON_MEMORY_WRITE
        elif d.verdict == "sensitive_read":
            reason = (f"Blocked: {d.reason}. Reading or copying secrets, keys, keychains or browser "
                      "profiles is never allowed. Don't try another way; tell the user.")
        elif d.verdict == "outward":
            fp = ActionFingerprint(turn, name, d.code, d.target)
            if self.gate.consume(turn, fp.key):
                self._record(turn, name, tool_use_id, True, d, "confirmed by the user")
                return True, ""
            self.pending_action = fp
            reason = f"{REASON_OUTWARD} (Reason: {d.reason}.)"
        else:
            self._record(turn, name, tool_use_id, True, d, "")
            return True, ""
        self._record(turn, name, tool_use_id, False, d, reason)
        return False, reason

    async def __call__(self, input_data: dict[str, Any], tool_use_id: str | None,
                       context: Any = None) -> dict[str, Any]:
        if input_data.get("hook_event_name", "PreToolUse") != "PreToolUse":
            return {}
        name = str(input_data.get("tool_name", ""))
        tool_input = input_data.get("tool_input") or {}
        tuid = tool_use_id or str(input_data.get("tool_use_id", ""))
        try:
            allowed, reason = self.decide(name, tool_input, tuid)
        except Exception:  # fail closed
            log.exception("pre-tool hook failed for %s", name)
            return _deny("Safety check failed for this action; it was blocked.")
        return {} if allowed else _deny(reason)
