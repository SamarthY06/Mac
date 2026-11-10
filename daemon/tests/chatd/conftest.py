"""chatd test fixtures. The runner tests reuse the agentd test kit (a real agentd Service and
Server with scripted FakeBrains), so tests/agentd is put on the import path."""

from __future__ import annotations

import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agentd"))


@pytest.fixture
def short_root() -> Iterator[Path]:
    """A short directory for unix sockets (macOS caps socket paths at 104 bytes)."""
    d = Path(tempfile.mkdtemp(prefix="cd", dir="/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)
