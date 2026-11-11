"""scripts/services.sh against a throwaway git repo, in a temp HOME, with launchctl stubbed.
Never the real LaunchAgent, ~/.mac-agent or the live service."""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "services.sh"
LAUNCHCTL = """#!/bin/bash
echo "$*" >>"$STUB_DIR/calls"
case "$1" in
  print) [[ -f "$STUB_DIR/loaded" ]] ;;
  bootstrap)  # fails for a plist naming the sha in $STUB_DIR/fail
    if [[ -s "$STUB_DIR/fail" ]] && grep -q "$(cat "$STUB_DIR/fail")" "$3"; then exit 5; fi
    touch "$STUB_DIR/loaded" ;;
  bootout) rm -f "$STUB_DIR/loaded" ;;
  *) true ;;
esac
"""
MAKEFILE = "build:\n\tmkdir -p bin && cp version.txt bin/wa-bridge && chmod +x bin/wa-bridge\n"


class World:
    def __init__(self, root: Path) -> None:
        self.root, self.repo, self.home = root, root / "repo", root / "home"
        self.stub = root / "stub"
        self.mac = self.home / ".mac-agent"
        self.releases = self.mac / "release"
        self.plist_path = self.home / "Library" / "LaunchAgents" / "com.mac-agent.services.plist"
        for d in (self.repo / "scripts", self.repo / "daemon" / ".venv" / "bin",
                  self.repo / "daemon" / "src" / "chatd", self.repo / "bridges" / "whatsapp", self.stub, self.home):
            d.mkdir(parents=True)
        shutil.copy(SCRIPT, self.repo / "scripts" / "services.sh")
        (self.repo / "daemon" / ".venv" / "bin" / "python").symlink_to(sys.executable)
        (self.repo / ".gitignore").write_text(".venv/\nbin/\n")
        (self.repo / "bridges" / "whatsapp" / "Makefile").write_text(MAKEFILE)
        (self.stub / "launchctl").write_text(LAUNCHCTL)
        (self.stub / "launchctl").chmod(0o755)
        env = {k: v for k, v in os.environ.items() if k not in ("MAC_HOME", "PYTHONPATH")}
        self.env = {**env, "HOME": str(self.home), "LAUNCHCTL": str(self.stub / "launchctl"), "MAC_BRIDGE": "build",
                    "STUB_DIR": str(self.stub), "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
                    # isolated from the user's git config (signing, hooks, templates)
                    "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_COUNT": "2", "GIT_CONFIG_KEY_0": "commit.gpgsign", "GIT_CONFIG_VALUE_0": "false",
                    "GIT_CONFIG_KEY_1": "tag.gpgsign", "GIT_CONFIG_VALUE_1": "false"}
        self.git("init", "-q")

    def git(self, *args: str) -> str:
        return subprocess.run(["git", "-C", str(self.repo), *args], env=self.env, check=True,
                              capture_output=True, text=True).stdout.strip()

    def commit(self, version: str, release: str = "0.1.0") -> str:
        (self.repo / "VERSION").write_text(f"{release}\n")
        (self.repo / "daemon" / "src" / "chatd" / "__init__.py").write_text(f"VERSION = {version!r}\n")
        (self.repo / "bridges" / "whatsapp" / "version.txt").write_text(f"bridge {version}\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", f"version {version}")
        return self.git("rev-parse", "HEAD")

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([str(self.repo / "scripts" / "services.sh"), *args], env=self.env,
                              capture_output=True, text=True, timeout=60, check=False)

    def plist(self) -> dict[str, Any]:
        return plistlib.loads(self.plist_path.read_bytes())

    def current(self) -> str:
        return os.readlink(self.releases / "current")

    def bridge(self) -> str:
        return (self.mac / "bin" / "wa-bridge").read_text()

    def calls(self) -> list[str]:
        p = self.stub / "calls"
        return p.read_text().splitlines() if p.exists() else []

    def snapshots(self) -> set[str]:
        return {p.name for p in self.releases.iterdir() if p.is_dir() and not p.is_symlink()}


@pytest.fixture
def w(tmp_path: Path) -> World:
    return World(tmp_path)


def test_install_runs_a_frozen_snapshot_of_head_not_the_working_tree(w: World) -> None:
    sha = w.commit("v1")
    r = w.run("install")
    assert r.returncode == 0, r.stderr
    snap = w.releases / sha
    assert w.current() == sha and (snap / ".complete").exists()
    p = w.plist()
    assert p["ProgramArguments"] == [str(w.repo / "daemon" / ".venv" / "bin" / "python"), "-m", "macsvc"]
    assert p["WorkingDirectory"] == p["EnvironmentVariables"]["PYTHONPATH"] == str(snap / "daemon" / "src")
    assert p["RunAtLoad"] is True and p["KeepAlive"] == {"SuccessfulExit": False}
    assert "MAC_HOME" not in p["EnvironmentVariables"]
    assert w.bridge() == "bridge v1\n" and os.access(w.mac / "bin" / "wa-bridge", os.X_OK)
    assert any(c.startswith("bootstrap gui/") for c in w.calls())

    # The snapshot's code wins over the venv's editable install of the repo (this test's venv).
    out = subprocess.run([sys.executable, "-c", "import chatd; print(chatd.__file__, chatd.VERSION)"],
                         env={**w.env, "PYTHONPATH": p["EnvironmentVariables"]["PYTHONPATH"]},
                         cwd=p["WorkingDirectory"], capture_output=True, text=True, check=True).stdout
    assert out.split() == [str(snap / "daemon" / "src" / "chatd" / "__init__.py"), "v1"]

    # Editing the working tree changes nothing the service runs.
    (w.repo / "daemon" / "src" / "chatd" / "__init__.py").write_text("raise SystemExit('half-done edit')\n")
    assert (snap / "daemon" / "src" / "chatd" / "__init__.py").read_text() == "VERSION = 'v1'\n"


def test_install_and_update_refuse_a_dirty_tree_unless_allowed(w: World) -> None:
    sha = w.commit("v1")
    (w.repo / "daemon" / "src" / "chatd" / "__init__.py").write_text("VERSION = 'dirty'\n")
    r = w.run("install")
    assert r.returncode == 1 and "uncommitted changes" in r.stderr and "--allow-dirty" in r.stderr
    assert not w.plist_path.exists() and w.calls() == []
    r = w.run("install", "--allow-dirty")
    assert r.returncode == 0, r.stderr
    assert w.current() == sha  # HEAD, never the dirty edit
    assert (w.releases / sha / "daemon" / "src" / "chatd" / "__init__.py").read_text() == "VERSION = 'v1'\n"
    assert w.run("update").returncode == 1
    assert w.run("update", "--bogus").returncode == 1


def test_update_switches_restarts_keeps_three_and_rollback_goes_back(w: World) -> None:
    shas = [w.commit("v1")]
    assert w.run("install").returncode == 0
    for v in ("v2", "v3", "v4"):
        shas.append(w.commit(v))
        r = w.run("update")
        assert r.returncode == 0, r.stderr
        assert w.current() == shas[-1] and w.bridge() == f"bridge {v}\n"
        assert w.plist()["EnvironmentVariables"]["PYTHONPATH"] == str(w.releases / shas[-1] / "daemon" / "src")
    calls = w.calls()
    assert sum(c.startswith("bootout") for c in calls) == 3 and sum(c.startswith("bootstrap") for c in calls) == 4
    assert w.snapshots() == set(shas[1:])  # the oldest was pruned

    r = w.run("rollback")
    assert r.returncode == 0, r.stderr
    assert w.current() == shas[2] and w.bridge() == "bridge v3\n"
    assert w.plist()["WorkingDirectory"] == str(w.releases / shas[2] / "daemon" / "src")
    r = w.run("rollback")  # a second rollback goes forward again
    assert r.returncode == 0 and w.current() == shas[3]

    r = w.run("status")
    assert r.returncode == 0 and f"Running commit: {shas[3][:12]} (version v4)." in r.stdout


def test_a_failed_bridge_build_switches_nothing(w: World) -> None:
    sha = w.commit("v1")
    assert w.run("install").returncode == 0
    (w.repo / "bridges" / "whatsapp" / "Makefile").write_text("build:\n\tfalse\n")
    w.git("commit", "-qam", "broken bridge")
    r = w.run("update")
    assert r.returncode == 1 and "did not build" in r.stderr
    assert w.current() == sha and w.snapshots() == {sha}
    assert not any(p.name.startswith(".new.") for p in w.releases.iterdir())


def _fake_release(w: World, release: str, body: str, *, corrupt: bool = False) -> str:
    """A file:// stand-in for github.com/<owner>/OpenMac/releases/download/v<release>/."""
    d = w.root / "rel" / f"v{release}"
    d.mkdir(parents=True)
    asset = d / "wa-bridge-darwin-arm64"
    asset.write_text(body)
    asset.chmod(0o755)
    digest = subprocess.run(["shasum", "-a", "256", str(asset)], capture_output=True, text=True, check=True).stdout.split()[0]
    if corrupt:
        digest = "0" * 64
    (d / "SHA256SUMS").write_text(f"{digest}  wa-bridge-darwin-arm64\n")
    return "file://" + str(w.root / "rel")


def test_without_go_the_bridge_is_downloaded_for_the_snapshots_version(w: World) -> None:
    sha = w.commit("v1", release="0.1.0")
    w.env["MAC_BRIDGE"] = "download"
    w.env["OPENMAC_RELEASES_URL"] = _fake_release(w, "0.1.0", "#!/bin/sh\necho prebuilt 0.1.0\n")
    r = w.run("install")
    assert r.returncode == 0, r.stderr
    assert w.current() == sha and w.bridge() == "#!/bin/sh\necho prebuilt 0.1.0\n"
    assert (w.releases / sha / "bridges" / "whatsapp" / "bin" / ".source").read_text() == "v0.1.0\n"
    assert "Downloading the WhatsApp bridge v0.1.0" in r.stderr


def test_a_bad_checksum_or_missing_release_switches_nothing(w: World) -> None:
    sha = w.commit("v1")
    assert w.run("install").returncode == 0  # built (MAC_BRIDGE=build)
    w.env["MAC_BRIDGE"] = "download"
    w.env["OPENMAC_RELEASES_URL"] = _fake_release(w, "0.2.0", "tampered\n", corrupt=True)
    w.commit("v2", release="0.2.0")
    r = w.run("update")
    assert r.returncode == 1 and "checksum mismatch" in r.stderr and "Install Go" in r.stderr
    assert w.current() == sha and w.snapshots() == {sha} and w.bridge() == "bridge v1\n"
    assert not any(p.name.startswith(".new.") for p in w.releases.iterdir())
    w.env["OPENMAC_RELEASES_URL"] = "file://" + str(w.root / "nowhere")
    r = w.run("update")
    assert r.returncode == 1 and w.current() == sha and w.snapshots() == {sha}


def test_rollback_without_an_earlier_snapshot_and_uninstall(w: World) -> None:
    w.commit("v1")
    assert w.run("rollback").returncode == 1  # not installed
    assert w.run("install").returncode == 0
    r = w.run("rollback")
    assert r.returncode == 1 and "no earlier snapshot" in r.stderr
    r = w.run("uninstall")
    assert r.returncode == 0 and not w.plist_path.exists() and not (w.stub / "loaded").exists()
    assert w.snapshots()  # kept for later


def test_update_messages_and_rollback_rejects_allow_dirty(w: World) -> None:
    sha1 = w.commit("v1")
    assert w.run("install").returncode == 0
    (w.releases / "current").unlink()  # an install from before snapshots: no current
    r = w.run("update")
    assert r.returncode == 0 and r.stdout.strip() == f"Now on commit {sha1[:12]}. Restarted."
    r = w.run("update")
    assert r.stdout.strip() == f"Already on commit {sha1[:12]}. Restarted."
    sha2 = w.commit("v2")
    r = w.run("update")
    assert r.stdout.strip() == (f"Updated from {sha1[:12]} to {sha2[:12]}. Restarted. "
                                "To go back: scripts/services.sh rollback")
    r = w.run("rollback", "--allow-dirty")
    assert r.returncode == 1 and "Unknown option" in r.stderr and w.current() == sha2


def test_prune_deletes_only_real_snapshots_inside_release(w: World) -> None:
    outside = w.root / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("x")
    fake_sha = "f" * 40
    shas = [w.commit("v1")]
    assert w.run("install").returncode == 0
    (w.releases / fake_sha).symlink_to(outside)  # a sha-named symlink pointing out
    (w.releases / "notes").mkdir()  # not a snapshot name
    (w.releases / ".new.stale1").mkdir()  # left by a crashed run
    with (w.releases / "history").open("a") as f:
        f.write(f"../../precious\ncurrent\n.lock\n{fake_sha}\n")
    for v in ("v2", "v3", "v4", "v5"):
        shas.append(w.commit(v))
        assert w.run("update").returncode == 0
    assert (outside / "keep.txt").exists() and (w.releases / fake_sha).is_symlink()
    assert (w.releases / "notes").is_dir() and not (w.releases / ".new.stale1").exists()
    assert w.current() == shas[-1] and w.snapshots() == {*shas[-3:], "notes"}
    assert (w.releases / "history").read_text().split() == shas[-3:]


def test_a_second_run_waits_for_the_lock(w: World) -> None:
    sha = w.commit("v1")
    assert w.run("install").returncode == 0
    lock = w.releases / ".lock"
    lock.mkdir()
    (lock / "pid").write_text(f"{os.getpid()}\n")  # a live owner
    (w.releases / ".new.building").mkdir()  # its staging dir
    w.commit("v2")
    for args in (("update",), ("rollback",), ("uninstall",)):
        r = w.run(*args)
        assert r.returncode == 1 and "in progress" in r.stderr
    assert (w.releases / ".new.building").is_dir() and w.current() == sha and w.plist_path.exists()

    dead = subprocess.Popen(["true"])
    dead.wait()
    (lock / "pid").write_text(f"{dead.pid}\n")  # owner gone: the lock is taken over
    r = w.run("update")
    assert r.returncode == 0, r.stderr
    assert not lock.exists() and not (w.releases / ".new.building").exists()


def test_a_snapshot_that_will_not_start_goes_back_to_the_previous_one(w: World) -> None:
    sha1 = w.commit("v1")
    assert w.run("install").returncode == 0
    sha2 = w.commit("v2")
    (w.stub / "fail").write_text(sha2)
    r = w.run("update")
    assert r.returncode == 1
    assert f"Commit {sha2[:12]} did not start, so Mac went back to commit {sha1[:12]}." in r.stderr
    assert f"running commit {sha1[:12]} again" in r.stderr
    assert w.current() == sha1 and w.bridge() == "bridge v1\n" and (w.stub / "loaded").exists()
    assert w.plist()["WorkingDirectory"] == str(w.releases / sha1 / "daemon" / "src")
    assert (w.releases / "history").read_text().split()[-1] == sha1

    (w.stub / "fail").write_text("")
    assert w.run("update").returncode == 0 and w.current() == sha2
    (w.stub / "fail").write_text(sha1)  # a rollback that won't start returns to where it was
    r = w.run("rollback")
    assert r.returncode == 1 and f"Commit {sha1[:12]} did not start, so Mac went back to commit {sha2[:12]}." in r.stderr
    assert w.current() == sha2 and w.bridge() == "bridge v2\n" and (w.stub / "loaded").exists()


def test_a_first_install_that_will_not_start_says_so(w: World) -> None:
    sha = w.commit("v1")
    (w.stub / "fail").write_text(sha)
    r = w.run("install")
    assert r.returncode == 1
    assert f"Commit {sha[:12]} did not start. Mac's services are stopped." in r.stderr
    assert not (w.stub / "loaded").exists() and not (w.releases / ".lock").exists()
