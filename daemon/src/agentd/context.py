"""`ask.context` (API 1.2, W10): validation, the per-session delta (V1) and the hand-off to
prompts.frame_context, which alone renders and escapes (V4, W10.9).

Everything here is untrusted chat content. agentd never trims it: anything over a limit is
`bad_args`. The labels "You" and "Mac" come from the booleans only, never from `sender`.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agentd import schema as s


class ContextError(ValueError):
    """A limit or the media path check failed (bad_args)."""


def validate(ctx: s.AskContext, media_dir: Path) -> s.AskContext:
    members = ctx.members or []
    if len(members) > s.CONTEXT_MAX_MEMBERS:
        raise ContextError(f"context.members: at most {s.CONTEXT_MAX_MEMBERS}")
    msgs = ctx.messages or []
    if len(msgs) > s.CONTEXT_MAX_MESSAGES:
        raise ContextError(f"context.messages: at most {s.CONTEXT_MAX_MESSAGES}")
    total, refs = 0, set()
    for i, m in enumerate(msgs):
        if not m.msg_ref.strip():
            raise ContextError(f"context.messages[{i}].msg_ref: must not be empty")
        if m.msg_ref in refs:
            raise ContextError(f"context.messages[{i}].msg_ref: repeated")
        refs.add(m.msg_ref)
        n = len(m.text or "")
        if n > s.CONTEXT_MAX_TEXT:
            raise ContextError(f"context.messages[{i}].text: at most {s.CONTEXT_MAX_TEXT} characters")
        total += n
        if m.media is not None and m.media.path is not None:
            raise ContextError(f"context.messages[{i}].media.path: only a quoted message carries a file")
    if total > s.CONTEXT_MAX_CHARS:
        raise ContextError(f"context.messages: at most {s.CONTEXT_MAX_CHARS} characters in total")
    q = ctx.quoted
    if isinstance(q, s.QuotedMessage):
        if len(q.text or "") > s.CONTEXT_MAX_TEXT:
            raise ContextError(f"context.quoted.text: at most {s.CONTEXT_MAX_TEXT} characters")
        if len(q.links) > s.CONTEXT_MAX_LINKS:
            raise ContextError(f"context.quoted.links: at most {s.CONTEXT_MAX_LINKS}")
        for link in q.links:
            if not link.lower().startswith(("http://", "https://")) or any(c.isspace() for c in link):
                raise ContextError("context.quoted.links: http(s) URLs only")
        if q.media is not None and q.media.path is not None:
            q = dataclasses.replace(q, media=dataclasses.replace(q.media, path=_media_path(q.media.path,
                                                                                         media_dir)))
    return dataclasses.replace(ctx, quoted=q)


def _media_path(raw: str, media_dir: Path) -> str:
    """`realpath` inside the media dir, never a string prefix (W10.5): symlinks and `..` resolved."""
    if not raw or "\x00" in raw or not os.path.isabs(raw):
        raise ContextError("context.quoted.media.path: must be an absolute path")
    real, root = os.path.realpath(raw), os.path.realpath(media_dir)
    if os.path.commonpath([real, root]) != root or real == root:
        raise ContextError("context.quoted.media.path: must be inside the chat media folder")
    return real


def narrow(ctx: s.AskContext, read_chat: bool) -> s.AskContext:
    """`read_chat` off: only the quote (the owner chose it); no members, no messages (W10.6)."""
    return ctx if read_chat else s.AskContext(quoted=ctx.quoted)


def delta(ctx: s.AskContext | None, framed: Iterable[str]) -> list[s.ContextMessage]:
    """V1: the ring entries not yet framed into this session, in the client's (time) order."""
    if ctx is None or not ctx.messages:
        return []
    seen = set(framed)
    return [m for m in ctx.messages if m.msg_ref not in seen]


def remember(framed: dict[str, list[str]], session: str, refs: Iterable[str], *, max_sessions: int = 20) -> None:
    """Add `refs` to the session's framed set, the last FRAMED_REFS_PER_SESSION kept."""
    new = list(refs)
    fresh = set(new)
    cur = [r for r in framed.pop(session, []) if r not in fresh]
    cur.extend(new)
    framed[session] = cur[-s.FRAMED_REFS_PER_SESSION:]
    while len(framed) > max_sessions:
        framed.pop(next(iter(framed)))


def render(text: str, ctx: s.AskContext | None, msgs: list[s.ContextMessage], *, display_name: str,
           owner_name: str, assistant_name: str) -> str:
    """The user turn: the framed blocks (prompts.frame_context renders and escapes), then `text`."""
    if ctx is None or (not msgs and not ctx.members and ctx.quoted is None):
        return text
    from macd.brain import prompts

    quoted: dict[str, Any] | None = None
    if ctx.quoted is not None:
        quoted = dataclasses.asdict(ctx.quoted)
    return prompts.frame_context(
        text, display_name=display_name, members=tuple(ctx.members or ()),
        messages=tuple(dataclasses.asdict(m) for m in msgs), quoted=quoted,
        owner_name=owner_name, assistant_name=assistant_name)
