"""`macctl doctor`: environment checks with fix hints.

Each check returns a Check(status, name, detail, fix). Status: "ok" | "warn" | "fail" | "skip".
Checks are synchronous (this runs in the CLI process, not the daemon loop). Probes that
depend on TCC (mic, Accessibility, Automation) are attributed to the process running doctor —
your terminal — and are labelled "(this terminal)"; rows labelled "(daemon)" come from macd's
own status (voice, Claude sign-in) and reflect what Mac.app can actually do.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from macd import config

MARK = {"ok": "✓", "warn": "!", "fail": "✗", "skip": "-"}


@dataclass
class Check:
    status: str
    name: str
    detail: str = ""
    fix: str = ""


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


# ----------------------------------------------------------------------------- checks


def check_arch() -> Check:
    m = platform.machine()
    if m == "arm64":
        return Check("ok", "Apple Silicon", m)
    return Check("warn", "Apple Silicon", f"machine is {m}", "Mac is tuned for M-series chips; expect higher latency.")


def check_macos() -> Check:
    ver = platform.mac_ver()[0]
    if not ver:
        return Check("fail", "macOS version", "not macOS", "Mac only runs on macOS 14+.")
    major = int(ver.split(".")[0])
    if major >= 14:
        return Check("ok", "macOS version", ver)
    return Check("warn", "macOS version", ver, "macOS 14 (Sonoma) or newer is recommended.")


def find_claude(cfg: config.Config) -> str | None:
    from macd.brain.claude_brain import default_cli_path

    path = default_cli_path(cfg)
    return path if path and Path(path).exists() else None


def check_claude(cfg: config.Config, deep: bool) -> list[Check]:
    path = find_claude(cfg)
    if path is None:
        return [Check("fail", "claude CLI", "not found", "Install Claude Code: https://claude.com/claude-code")]
    out = _run([path, "--version"], 15)
    ver = out.stdout.strip() if out and out.returncode == 0 else "unknown version"
    checks = [Check("ok", "claude CLI", f"{path} ({ver})")]
    if cfg.brain.auth == "api_key":
        has = config.get_secret("ANTHROPIC_API_KEY") is not None
        checks.append(Check("ok" if has else "fail", "Anthropic API key", "present" if has else "missing",
                            "" if has else "macctl secret set ANTHROPIC_API_KEY"))
    if deep:
        res = _run([path, "-p", "--output-format", "json", "--max-turns", "1", "Reply OK"], 60)
        if res is None:
            checks.append(Check("fail", "claude login", "query timed out (60 s)", "Run `claude` in a terminal and sign in."))
        elif res.returncode != 0:
            err = (res.stderr or res.stdout).strip().splitlines()[-1:] or ["error"]
            checks.append(Check("fail", "claude login", err[0][:160], "Run `claude` in a terminal and sign in."))
        else:
            try:
                data = json.loads(res.stdout)
                ok = not data.get("is_error")
                detail = str(data.get("result", ""))[:60]
            except json.JSONDecodeError:
                ok, detail = True, res.stdout.strip()[:60]
            checks.append(Check("ok" if ok else "fail", "claude login", f"replied: {detail}",
                                "" if ok else "Run `claude` in a terminal and sign in."))
    else:
        res = _run([path, "auth", "status"], 15)
        if res is None or "unknown" in (res.stderr + res.stdout).lower() and res.returncode != 0:
            checks.append(Check("skip", "claude login", "not checked (use --deep)"))
        elif res.returncode == 0:
            checks.append(Check("ok", "claude login", _auth_detail(res.stdout)))
        else:
            checks.append(Check("fail", "claude login", (res.stderr or res.stdout).strip()[:120], "Run `claude` in a terminal and sign in."))
    return checks


def _auth_detail(stdout: str) -> str:
    """`claude auth status` prints JSON; show the useful fields on one line."""
    text = stdout.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text.splitlines()[0][:80] if text else "logged in"
    if not isinstance(data, dict):
        return "logged in"
    parts = [f"{k}={data[k]}" for k in ("loggedIn", "authMethod", "email", "subscriptionType") if k in data]
    return ", ".join(parts)[:80] or "logged in"


def check_models() -> Check:
    try:
        from macd.speech.models import missing_models
    except ImportError as e:
        return Check("skip", "speech models", f"model registry unavailable ({e})")
    try:
        missing = missing_models()
    except Exception as e:  # noqa: BLE001
        return Check("fail", "speech models", repr(e), "macctl models")
    if not missing:
        return Check("ok", "speech models", str(config.MODELS_DIR))
    return Check("fail", "speech models", "missing: " + ", ".join(map(str, missing)), "macctl models")


def connect_checks(cfg: config.Config) -> list[Check]:
    """Same data as `macctl connect` (static part; no Claude process is started)."""
    from macd import connect

    d = connect.check_all(cfg)
    cl, ch = d["claude"], d["chrome"]
    out = [
        Check("ok" if cl["installed"] and cl["min_ok"] else "fail", "Claude Code", cl.get("version") or "not installed",
              cl["fix"] if not (cl["installed"] and cl["min_ok"]) else ""),
        Check("ok" if cl["signed_in"] == "yes" and cl["auth_method"] != "api_key" else "fail", "Claude sign-in",
              f"{cl['signed_in']} ({cl['auth_method']})", cl["fix"] if cl["signed_in"] != "yes" or cl["auth_method"] == "api_key" else ""),
        Check("ok" if ch["browser_running"] else "warn", "browser running", ", ".join(ch["browsers"]) or "no",
              connect.FIX_BROWSER if not ch["browser_running"] else ""),
        Check("ok" if ch["extension_installed"] else "fail", "Claude in Chrome extension",
              "installed" if ch["extension_installed"] else "not found", "" if ch["extension_installed"] else connect.FIX_EXTENSION),
        Check("ok" if ch["native_host"] else "fail", "Claude in Chrome connected",
              f"native host {'ok' if ch['native_host'] else 'missing'}, enabled by default {ch['enabled_by_default']}",
              "" if ch["native_host"] else connect.FIX_CONNECT),
    ]
    return out


def check_pocket_model() -> Check:
    """Optional: the Pocket TTS model used for voice cloning."""
    from macd.speech.models import missing_models

    try:
        missing = missing_models(["pocket"])
    except (KeyError, ValueError):
        return Check("skip", "voice clone model", "not supported by this build")
    if missing:
        return Check("warn", "voice clone model", "Pocket TTS not downloaded (optional)", "macctl models")
    return Check("ok", "voice clone model", "Pocket TTS present")


def check_mic() -> Check:
    try:
        import numpy as np
        import sounddevice as sd
    except Exception as e:  # noqa: BLE001
        return Check("fail", "microphone", f"sounddevice unavailable: {e}")
    frames: list[np.ndarray] = []

    def cb(indata, n, t, status):  # type: ignore[no-untyped-def]
        frames.append(indata.copy())

    try:
        with sd.InputStream(samplerate=16000, channels=1, dtype="float32", callback=cb):
            sd.sleep(250)
    except Exception as e:  # noqa: BLE001
        return Check("fail", "microphone", f"cannot open input: {e}",
                     "System Settings > Privacy & Security > Microphone: allow Mac (or your terminal).")
    if not frames:
        return Check("fail", "microphone", "no frames delivered", "Check the input device.")
    pcm = np.concatenate(frames)
    if float(np.max(np.abs(pcm))) == 0.0:
        return Check("fail", "microphone", "all-zero input (permission denied?)",
                     "Grant Microphone access to OpenValet.app (TCC attributes this probe to the calling process).")
    return Check("ok", "microphone (this terminal)", f"{len(pcm)} samples, peak {float(np.max(np.abs(pcm))):.3f}")


def check_accessibility() -> Check:
    try:
        from ApplicationServices import AXIsProcessTrusted
    except Exception as e:  # noqa: BLE001
        return Check("skip", "Accessibility", f"pyobjc unavailable: {e}")
    if AXIsProcessTrusted():
        return Check("ok", "Accessibility (this terminal)", "trusted")
    return Check("fail", "Accessibility (this terminal)", "not trusted",
                 "System Settings > Privacy & Security > Accessibility: enable Mac (and your terminal for dev).")


def check_automation() -> Check:
    res = _run(["osascript", "-e", 'tell application "System Events" to name of first process'], 15)
    if res is not None and res.returncode == 0:
        return Check("ok", "Automation (this terminal)", res.stdout.strip()[:40])
    detail = (res.stderr.strip()[:120] if res else "timed out") or "denied"
    return Check("fail", "Automation (this terminal)", detail,
                 "System Settings > Privacy & Security > Automation: allow System Events.")


def daemon_status(timeout: float = 3.0) -> dict | None:  # type: ignore[type-arg]
    from macd.cli import ControlClient, DaemonError

    try:
        with ControlClient(timeout=timeout) as c:
            return c.request("status")
    except (DaemonError, OSError, ValueError):
        return None


def check_daemon() -> tuple[Check, dict | None]:  # type: ignore[type-arg]
    st = daemon_status()
    if st is None:
        return Check("warn", "daemon", f"not reachable at {config.CONTROL_SOCK}", "Start OpenValet.app or run `macd`."), None
    return Check("ok", "daemon", f"state {st.get('state')}, up {st.get('uptime_s')}s"), st


def check_chrome(st: dict | None) -> Check:  # type: ignore[type-arg]
    if st is None:
        return Check("skip", "Claude in Chrome", "daemon not running")
    brain = st.get("brain") or {}
    chrome = brain.get("chrome") if isinstance(brain, dict) else None
    if chrome in (True, "connected"):
        return Check("ok", "Claude in Chrome", "connected")
    if chrome is None:
        return Check("skip", "Claude in Chrome", "status not reported yet (ask Mac something first)")
    return Check("fail", "Claude in Chrome", str(chrome),
                 "Install the Claude in Chrome extension, sign in, and keep Chrome running.")


def check_livekit() -> Check:
    names = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    missing = [n for n in names if not config.get_secret(n)]
    if not missing:
        return Check("ok", "LiveKit credentials", "present (values not shown)")
    return Check("warn", "LiveKit credentials", "missing: " + ", ".join(missing),
                 "Optional (remote link). macctl secret set LIVEKIT_URL  (etc.)")


def check_sleep() -> Check:
    res = _run(["pmset", "-g"], 10)
    if res is None or res.returncode != 0:
        return Check("skip", "sleep settings", "pmset unavailable")
    sleep_val = None
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "sleep":
            sleep_val = parts[1]
    if sleep_val is None:
        return Check("skip", "sleep settings", "unknown")
    if sleep_val == "0":
        return Check("ok", "sleep settings", "system sleep disabled")
    return Check("warn", "sleep settings", f"system sleeps after {sleep_val} min",
                 'System Settings > Battery/Energy > Options: enable "Prevent automatic sleeping '
                 'when the display is off" (on power adapter) so Mac stays listening.')


# ----------------------------------------------------------------------------- runner


def run_checks(deep: bool = False, cfg: config.Config | None = None) -> list[Check]:
    cfg = cfg or config.load(write_default=False)
    checks: list[Check] = [check_arch(), check_macos()]
    checks += check_claude(cfg, deep)
    checks += connect_checks(cfg)
    daemon, st = check_daemon()
    if st is not None:
        voice_ok = st.get("voice", True)
        checks.append(Check("ok" if voice_ok else "fail", "voice (daemon)",
                            "speech models loaded" if voice_ok else str(st.get("voice_reason")),
                            "" if voice_ok else "Re-run install.sh (it downloads the models)."))
        engine = st.get("voice_engine")
        if engine:
            sample = st.get("clone_sample")
            checks.append(Check("ok", "voice engine (daemon)", engine + (f" ({sample})" if sample else "")))
        auth = (st.get("brain") or {}).get("auth")
        if auth in ("ok", "signed_out", "missing"):
            checks.append(Check("ok" if auth == "ok" else "fail", "Claude sign-in (daemon)", auth,
                                "" if auth == "ok" else "Run `claude` in a terminal and sign in."))
    steps: list[Callable[[], Check]] = [
        check_models, check_pocket_model, check_mic, check_accessibility, check_automation, lambda: daemon,
        check_livekit, check_sleep,
    ]
    if deep:
        steps.insert(5, lambda: check_chrome(st))
    for step in steps:
        try:
            checks.append(step())
        except Exception as e:  # noqa: BLE001 - a broken probe must not break doctor
            checks.append(Check("fail", getattr(step, "__name__", "check"), repr(e)))
    return checks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="macctl doctor")
    p.add_argument("--deep", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    checks = run_checks(deep=args.deep)
    if args.json:
        print(json.dumps([c.__dict__ for c in checks], indent=2))
    else:
        for c in checks:
            line = f" {MARK.get(c.status, '?')} {c.name}"
            if c.detail:
                line += f": {c.detail}"
            print(line)
            if c.fix and c.status in ("fail", "warn"):
                print(f"     fix: {c.fix}")
    return 1 if any(c.status == "fail" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
