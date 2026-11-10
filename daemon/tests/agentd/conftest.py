"""agentd test fixtures (helpers live in agentd_kit.py)."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

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


