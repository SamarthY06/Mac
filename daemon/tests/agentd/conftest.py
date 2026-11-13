"""agentd test fixtures (helpers live in agentd_kit.py)."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from agentd_kit import Agent, make_agent, stop_agent


@pytest.fixture
def short_root() -> AsyncIterator[Path]:
    d = Path(tempfile.mkdtemp(prefix="ad", dir="/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
async def agent(short_root: Path) -> AsyncIterator[Agent]:
    a = await make_agent(short_root)
    yield a
    await stop_agent(a)


@pytest.fixture
async def env(short_root: Path) -> AsyncIterator[tuple[Agent, Any]]:
    """The scheduled-task scenario agent: a settable clock, a default report script, no osascript,
    the invariants checked and I9 (no orphan tasks) asserted at teardown."""
    from tasks_kit import Clock, report, stop
    from tasks_kit import agent as task_agent

    clock = Clock()
    a = await task_agent(short_root, clock)
    a.scripts["*"] = lambda text: report()
    yield a, clock
    await stop(a)


