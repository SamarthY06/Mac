"""macd: the Mac voice-agent daemon.

Wiring (one asyncio loop):
  AudioInputs → Listeners → Orchestrator → Brain
                              │
                              ▼
                         SpeechQueue → TTS → AudioOutput(s)
  ControlServer (control.sock) ⇄ control.handle_command / Orchestrator.publish
  EventLog + Metrics, LiveKitBridge (on demand)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import logging
import logging.handlers
import os
import signal
import sys
from typing import IO, Any

from macd import config
from macd.audio.earcons import earcon
from macd.brain.claude_brain import ClaudeBrain, default_cli_path
from macd.brain.fake_brain import FakeBrain
from macd.brain.gate import ConfirmationGate
from macd.brain.mac_tools import notify_argv
from macd.connect import ConnectChecker
from macd.contracts import AudioInput, AudioOutput, Brain
from macd.control import ControlServer, Hooks, handle_command
from macd.eventlog import EventLog, Metrics
from macd.orchestrator import Orchestrator
from macd.phrases import CLONE_LOAD_FAILED, PREWARM_PHRASES
from macd.remote.livekit_bridge import (
    SETUP_HINT,
    LiveKitBridge,
    RemoteUnavailable,
    configured,
)
from macd.settings import Settings, wake_phrases_for
from macd.speech.models import missing_models
from macd.speech_queue import SpeechQueue

log = logging.getLogger("macd")

SHUTDOWN_TIMEOUT_S = 5.0  # Mac.app's supervisor SIGKILLs after 10 s
SWIFT_CONNECT_WAIT_S = 3.0
AUTH_PROBE_TIMEOUT_S = 20.0
AUTH_REPROBE_S = 300.0
EXIT_ALREADY_RUNNING = 3  # Mac.app treats this as "an external macd is running"


# ----------------------------------------------------------------------------- helpers


def setup_logging(level: str) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    try:
        fh = logging.handlers.RotatingFileHandler(config.LOG_DIR / "macd.log", maxBytes=5_000_000, backupCount=3)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError as e:
        print(f"macd: cannot open log file: {e}", file=sys.stderr)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    for noisy in ("livekit", "httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def acquire_lock() -> IO[str] | None:
    fh = open(config.RUN_DIR / "macd.lock", "a+")  # noqa: SIM115 - held for the process lifetime
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh


# ----------------------------------------------------------------------------- daemon


class Daemon:
    def __init__(self, cfg: config.Config, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.stop = asyncio.Event()
        self.eventlog = EventLog(config.EVENT_LOG)
        self.metrics = Metrics()
        for name in ("kws_hits", "wakes", "false_wakes", "wakes_suppressed", "stop_words", "turns"):
            self.metrics.counters[name] = 0  # visible in `macctl status` before the first event
        self.stack: Any = None
        self.voice_reason = ""
        self.inputs: list[AudioInput] = []
        self.output: AudioOutput | None = None
        self.link: Any = None
        self.backend = "none"
        self.brain_status: dict[str, Any] = {"chrome": "unknown", "auth": "unknown"}
        self.bridge: LiveKitBridge | None = None
        self.orch: Orchestrator | None = None
        self.speech: SpeechQueue | None = None
        self.brain: Brain | None = None
        self.control: ControlServer | None = None
        self.settings: Settings | None = None
        self.connect: ConnectChecker | None = None
        self._bg: set[asyncio.Task[Any]] = set()

    # ------------------------------------------------------------------ setup

    async def _load_speech(self) -> None:
        missing = missing_models()
        if missing:
            self.voice_reason = "speech models missing"
            log.error("speech models missing: %s. Re-run install.sh (it downloads them).", ", ".join(missing))
            self.eventlog.emit("models_missing", missing=list(missing))
            await self._notify("Mac", "Speech models are missing. Re-run install.sh.", None)
            return
        from macd.speech.factory import build_speech

        try:
            self.stack = await asyncio.to_thread(build_speech, self.cfg, metrics=self.metrics)
        except Exception:
            self.voice_reason = "speech models failed to load"
            log.exception("failed to load the speech stack; running without voice")
            await self._notify("Mac", "Speech models failed to load. See macd.log, or re-run install.sh.", None)

    async def _setup_audio(self) -> None:
        backend = self.args.audio or self.cfg.audio.backend
        if backend == "file":
            from macd.audio.file_backend import CaptureOutput, FileInput

            if not self.args.input_file:
                raise SystemExit("--audio file needs --input-file PATH")
            self.inputs = [FileInput(self.args.input_file, realtime=True, name="local")]
            self.output = CaptureOutput(realtime=True, name="capture")
            self.backend = "file"
            return
        if backend in ("auto", "swift"):
            from macd.audio.swift_link import SwiftAudioLink

            link = SwiftAudioLink(config.AUDIO_SOCK)
            await link.start()
            connected = backend == "swift"
            if not connected:
                try:
                    await asyncio.wait_for(link.connected.wait(), SWIFT_CONNECT_WAIT_S)
                    connected = True
                except TimeoutError:
                    log.info("audio: Mac.app not connected after %.0fs; using sounddevice", SWIFT_CONNECT_WAIT_S)
            if connected:
                self.link, self.inputs, self.output, self.backend = link, [link.input], link.output, "swift"
                return
            await link.close()
        from macd.audio.sd_backend import SoundDeviceInput, SoundDeviceOutput

        self.inputs = [SoundDeviceInput(self.cfg, name="local")]
        self.output = SoundDeviceOutput(self.cfg)
        self.backend = "sounddevice"

    def _make_brain(self, gate: ConfirmationGate) -> Brain:
        if self.args.fake_brain:
            return FakeBrain()

        def chrome_status(connected: bool) -> None:
            self.brain_status["chrome"] = "connected" if connected else "disconnected"

        return ClaudeBrain(self.cfg, gate, chrome_status_cb=chrome_status, on_notify=self._notify_app,
                           recent_user_text=lambda n=10: self.orch.recent_user_text(n) if self.orch else [])

    def _make_listener(self, inp: AudioInput) -> Any:
        from macd.speech.factory import make_listener

        output = self.output
        return make_listener(inp, self.stack, self.cfg, aec=lambda: bool(output is not None and output.aec),
                             metrics=self.metrics, eventlog=self.eventlog)

    def _spawn(self, coro: Any, name: str) -> None:
        t = asyncio.create_task(coro, name=name)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    # ------------------------------------------------------------------ hooks

    def _notify_app(self, title: str, body: str, path: str | None) -> bool:
        """Route a notification to Mac.app when it is subscribed (it owns the notification
        grant and reveals `path` on click). False → the caller falls back to osascript."""
        if self.control is None or self.control.subscriber_count == 0:
            return False
        self.control.publish({"event": "notify", "title": title, "body": body, "path": path})
        self.eventlog.emit("notify", title=title, body=body, path=path)
        return True

    async def _notify(self, title: str, body: str, path: str | None) -> None:
        """Daemon-originated notification (e.g. models missing): app if subscribed, else osascript."""
        if self._notify_app(title, body, path):
            return
        with contextlib.suppress(Exception):
            proc = await asyncio.create_subprocess_exec(*notify_argv(title, body), stdout=asyncio.subprocess.DEVNULL,
                                                        stderr=asyncio.subprocess.DEVNULL)
            await asyncio.wait_for(proc.wait(), 5)

    async def _remote_link(self) -> dict[str, Any]:
        if not configured():
            return {"configured": False, "hint": SETUP_HINT}
        if not self.cfg.remote.enabled:
            return {"configured": True, "enabled": False,
                    "hint": "Remote access is off. Enable it from the phone panel (remote_enable) or set "
                            "[remote] enabled = true in ~/.mac-agent/config.toml."}
        if self.bridge is None:
            assert self.orch is not None and self.speech is not None
            self.bridge = LiveKitBridge(self.cfg, self.orch, self.speech, self._make_listener)
        try:
            return await self.bridge.create_link()
        except RemoteUnavailable as e:
            raise ValueError(str(e)) from None

    async def _remote_enable(self) -> None:
        """Runtime enable from the app's phone panel; remembered in state.json (honoured at boot)."""
        self.cfg.remote.enabled = True
        await asyncio.to_thread(config.update_state, remote_enabled=True)

    async def _voice_clone(self, path: str) -> dict[str, Any]:
        """Switch TTS to a cloned voice (Pocket TTS); the confirmation is spoken by the new voice."""
        if self.stack is None:
            raise ValueError("voice is unavailable (speech models not loaded)")
        try:
            confirmation = await asyncio.to_thread(self.stack.set_voice, path)
        except (FileNotFoundError, ValueError, OSError) as e:
            raise ValueError(f"could not use that voice sample: {e}") from None
        self.eventlog.emit("voice_clone", path=path, engine=self.stack.voice_engine)
        if self.orch is not None and confirmation:
            self.orch.say(confirmation)
        return self._voice_reply()

    def _voice_reply(self) -> dict[str, Any]:
        engine, sample = self.stack.voice_engine, self.stack.clone_sample
        return {"engine": engine, "clone_sample": sample,
                "voice_engine": {"engine": engine, "sample": os.path.basename(sample) if sample else None}}

    async def _voice_reset(self) -> dict[str, Any]:
        if self.stack is None:
            raise ValueError("voice is unavailable (speech models not loaded)")
        confirmation = await asyncio.to_thread(self.stack.set_voice, None)
        self.eventlog.emit("voice_reset", engine=self.stack.voice_engine)
        if self.orch is not None and confirmation:
            self.orch.say(confirmation)
        return self._voice_reply()

    async def _remote_end(self) -> None:
        if self.bridge is not None:
            await self.bridge.close()

    async def _audio_reinit(self) -> None:
        if self.backend == "sounddevice" and self.output is not None:
            with contextlib.suppress(Exception):
                await self.output.close()
            await self.output.start()
        self.eventlog.emit("audio_reinit", backend=self.backend)

    async def _power(self, event: str) -> None:
        if event == "did_wake" and self.backend == "sounddevice":
            await self._audio_reinit()

    def _status(self) -> dict[str, Any]:
        st: dict[str, Any] = {
            "audio": {
                "backend": self.backend,
                "aec": bool(self.output.aec) if self.output is not None else False,
                "connected": bool(self.link.connected.is_set()) if self.link is not None else None,
            },
            "brain": dict(self.brain_status, kind=type(self.brain).__name__ if self.brain else None),
            "connect": self.connect.snapshot() if self.connect is not None else None,
            "voice_engine": self.stack.voice_engine if self.stack is not None else None,
            "clone_sample": self.stack.clone_sample if self.stack is not None else None,
        }
        if self.bridge is not None:
            st["remote"] = {**(self.orch.remote if self.orch else {}), **self.bridge.status()}
        else:
            st["remote"] = {**(self.orch.remote if self.orch else {}), "configured": configured()}
        if self.control is not None:
            st["control_clients"] = self.control.client_count
        return st

    async def _auth_probe_loop(self) -> None:
        """status.brain.auth: ok | signed_out | missing (UX#8); re-probed every 5 min until ok."""
        if self.args.fake_brain:
            self.brain_status["auth"] = "ok"
            return
        while True:
            cli = default_cli_path(self.cfg)
            if not cli:
                self.brain_status["auth"] = "missing"
            else:
                try:
                    proc = await asyncio.create_subprocess_exec(cli, "auth", "status", stdout=asyncio.subprocess.DEVNULL,
                                                                stderr=asyncio.subprocess.DEVNULL)
                    rc = await asyncio.wait_for(proc.wait(), AUTH_PROBE_TIMEOUT_S)
                    self.brain_status["auth"] = "ok" if rc == 0 else "signed_out"
                except (TimeoutError, OSError):
                    self.brain_status["auth"] = "unknown"
            if self.brain_status["auth"] == "ok":
                return
            await asyncio.sleep(AUTH_REPROBE_S)

    # ------------------------------------------------------------------ run

    async def run(self) -> int:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.stop.set)
        if os.environ.get("MAC_SUPERVISED") == "1":
            self._watch_stdin(loop)  # Mac.app holds our stdin open: EOF means the app died
        self.eventlog.start()
        self.eventlog.emit("daemon_start", pid=os.getpid(), audio=self.args.audio, version=_version())
        try:
            await self._load_speech()
            await self._setup_audio()
            assert self.output is not None
            tts = self.stack.tts if self.stack is not None else None
            self.speech = SpeechQueue(tts, {"local": self.output}, default_output="local",
                                      on_drop=lambda item, why: self.metrics.incr(f"speech_dropped_{why}"))
            gate = ConfirmationGate()
            self.brain = self._make_brain(gate)
            self.orch = Orchestrator(self.cfg, brain=self.brain, gate=gate, speech=self.speech,
                                     eventlog=self.eventlog, metrics=self.metrics, earcon=earcon,
                                     voice_reset=self._voice_reset, rename=self._rename)
            self.orch.set_voice(self.stack is not None, self.voice_reason)
            if self.stack is not None:
                for i, inp in enumerate(self.inputs):
                    self.orch.add_listener(self._make_listener(inp), requires_wake=inp.requires_wake_word,
                                           primary=i == 0)
            state = config.read_state()
            if state.get("remote_enabled"):
                self.cfg.remote.enabled = True
            if state.get("listening_off"):
                log.info("listening was switched off before the last restart; staying off")
                self.orch.set_off(True, persist=False)
            speech = self.speech
            self.settings = Settings(self.cfg, stack=self.stack, brain=self.brain, speak=self.orch.say,
                                     play=lambda pcm, sr: speech.play_pcm(0, pcm, sr, "say"),
                                     prewarm=self._prewarm_fillers)
            self.connect = ConnectChecker(self.cfg, brain=self.brain,
                                          brain_chrome=lambda: str(self.brain_status.get("chrome", "unknown")))
            hooks = Hooks(remote_link=self._remote_link, remote_end=self._remote_end, remote_enable=self._remote_enable,
                          settings=self.settings, connect_check=self.connect.command,
                          audio_reinit=self._audio_reinit, voice_clone=self._voice_clone,
                          voice_reset=self._voice_reset,
                          power=self._power, status=self._status)
            orch = self.orch
            self.control = ControlServer(lambda cmd, args: handle_command(orch, hooks, cmd, args), config.CONTROL_SOCK)
            self.orch.subscribe(self.control.publish)
            await self.control.start()
            for inp in self.inputs:
                try:
                    await inp.start()
                except Exception:
                    log.exception("audio input %s failed to start", inp.name)
            try:
                await self.output.start()
            except Exception:
                log.exception("audio output failed to start")
            await self.brain.start()  # creates the brain's owner task; it supervises itself
            self._spawn(self._auth_probe_loop(), "auth-probe")
            if self.stack is not None:
                self._spawn(self._prewarm(), "prewarm")
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.orch.run(), name="orchestrator")
                log.info("macd ready (audio=%s, voice=%s)", self.backend, self.stack is not None)
                self.eventlog.emit("daemon_ready", audio=self.backend, voice=self.stack is not None)
                await self.stop.wait()
                log.info("shutting down")
                self.orch.close()
        finally:
            await self._shutdown()
        return 0

    async def _prewarm_fillers(self) -> None:
        if self.stack is not None:
            await self.stack.tts.prewarm(list(PREWARM_PHRASES))

    async def _rename(self, name: str) -> None:
        """Spoken "call yourself Jarvis": name + wake phrases through the same settings path."""
        if self.settings is not None:
            res = await self.settings.set({"name": name, "wake.phrases": wake_phrases_for(name)})
            if res["errors"]:
                log.warning("rename to %r failed: %s", name, res["errors"])

    async def _prewarm(self) -> None:
        await self.stack.warmup()  # one STT + TTS inference
        await self.stack.tts.prewarm(list(PREWARM_PHRASES))  # filler cache
        log.info("speech warm; %d phrases cached (engine %s)", len(PREWARM_PHRASES), self.stack.voice_engine)
        if self.stack.voice_error:  # persisted clone could not be loaded → Kokoro fallback, say so once
            log.error("cloned voice unavailable: %s", self.stack.voice_error)
            self.eventlog.emit("voice_clone_failed", error=self.stack.voice_error)
            if self.orch is not None:
                self.orch.say(CLONE_LOAD_FAILED)

    def _watch_stdin(self, loop: asyncio.AbstractEventLoop) -> None:
        fd = sys.stdin.fileno()

        def on_readable() -> None:
            try:
                data = os.read(fd, 4096)
            except OSError:
                data = b""
            if not data:
                loop.remove_reader(fd)
                if not self.stop.is_set():
                    log.info("stdin closed (supervisor gone); shutting down")
                    self.stop.set()

        try:
            loop.add_reader(fd, on_readable)
        except (OSError, ValueError) as e:
            log.warning("cannot watch stdin: %s", e)

    async def _shutdown(self) -> None:
        async def steps() -> None:
            for t in list(self._bg):
                t.cancel()
            if self.brain is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self.brain.close(), 3.0)
            if self.bridge is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self.bridge.close(), 2.0)
            for inp in self.inputs:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(inp.close(), 1.0)
            if self.output is not None:
                with contextlib.suppress(Exception):
                    self.output.flush()
                    await asyncio.wait_for(self.output.close(), 1.0)
            if self.link is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self.link.close(), 1.0)
            if self.control is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self.control.close(), 1.0)

        try:
            await asyncio.wait_for(steps(), SHUTDOWN_TIMEOUT_S)
        except TimeoutError:
            log.warning("shutdown timed out after %.0fs", SHUTDOWN_TIMEOUT_S)
        if self.stack is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self.stack.close)
        self.eventlog.emit("daemon_stop")
        await asyncio.to_thread(self.eventlog.close)


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("macd")
    except Exception:  # noqa: BLE001
        return "dev"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="macd", description="Mac voice agent daemon")
    p.add_argument("--audio", choices=["auto", "swift", "sounddevice", "file"], default=None,
                   help="audio backend (default: config audio.backend)")
    p.add_argument("--input-file", default=None, help="WAV file for --audio file (testing)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fake-brain", action="store_true", help="use a scripted brain instead of Claude (testing)")
    g.add_argument("--no-brain", dest="fake_brain", action="store_true", help="alias for --fake-brain")
    p.add_argument("--log-level", default=os.environ.get("MAC_LOG_LEVEL", "INFO"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config.ensure_dirs()
    setup_logging(args.log_level)
    # GIL hand-off latency: onnxruntime calls on the tts_pool thread hold the GIL in bursts; a
    # 1 ms switch interval keeps the event loop's per-frame time ~1 ms while TTS runs (process-wide).
    sys.setswitchinterval(0.001)
    lock = acquire_lock()
    if lock is None:
        print("macd is already running", file=sys.stderr)
        return EXIT_ALREADY_RUNNING
    try:
        cfg = config.load()
    except Exception as e:  # noqa: BLE001
        log.error("invalid config %s: %s", config.CONFIG_PATH, e)
        return 2
    try:
        return asyncio.run(Daemon(cfg, args).run())
    except KeyboardInterrupt:
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
