#!/usr/bin/env python3
"""Fake macd for end-to-end testing of Mac.app's AudioLink + ControlClient (stdlib only).

Serves $MAC_HOME/run/audio.sock and $MAC_HOME/run/control.sock, waits for the app to
connect, then runs checks:

  1. HELLO received (JSON with version/input_device/aec)
  2. MIC frames: rate ≈ 50/s, every payload 640 bytes
  3. PLAY 0.5 s quiet tone (24 kHz, amplitude 0.05) in 20 ms chunks + END → DRAINED
     (and DRAINED arrives no earlier than ~the audio duration)
  4. END with no PLAY → immediate DRAINED
  5. FLUSH(id) mid-stream → DRAINED(id) quickly
  6. FLUSH(0) mid-stream → DRAINED for the open stream
  7. off switch: state=off → MIC frames stop + HELLO(input "none"); state=idle →
     fresh HELLO with a real input device and MIC frames resume at ~50/s
  8. control: app sends {"cmd":"subscribe"}; we push state/transcript events and log
     any commands the app sends (replying ok; remote_link gets a fake url)

Usage:
  MAC_HOME=$(mktemp -d) python3 scripts/fake_daemon.py [--timeout 60] [--hold]
  then launch:  open --env MAC_HOME=$MAC_HOME --env MAC_SUPERVISE=0 build/Mac.app

Exit code 0 if all checks pass. --hold keeps serving (for manual UI poking) after checks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

MIC, PLAY, FLUSH, DRAINED, END, HELLO = 1, 2, 3, 4, 5, 6
NAMES = {1: "MIC", 2: "PLAY", 3: "FLUSH", 4: "DRAINED", 5: "END", 6: "HELLO"}


def frame(t: int, payload: bytes) -> bytes:
    return struct.pack("<BI", t, len(payload)) + payload


def u32(v: int) -> bytes:
    return struct.pack("<I", v)


def tone(seconds: float, rate: int = 24000, freq: float = 440.0, amp: float = 0.05) -> bytes:
    n = int(seconds * rate)
    return b"".join(
        struct.pack("<h", int(amp * 32767 * math.sin(2 * math.pi * freq * i / rate))) for i in range(n)
    )


def log(*a: object) -> None:
    print(f"[fake {time.strftime('%H:%M:%S')}]", *a, flush=True)


class AudioSide:
    def __init__(self) -> None:
        self.conn: socket.socket | None = None
        self.hello: dict | None = None
        self.hellos: list[dict] = []
        self.mic_times: list[float] = []
        self.mic_sizes: set[int] = set()
        self.mic_last: bytes = b""
        self.drained: dict[int, float] = {}
        self.cv = threading.Condition()
        self.wlock = threading.Lock()

    def send(self, t: int, payload: bytes) -> None:
        assert self.conn
        with self.wlock:
            self.conn.sendall(frame(t, payload))

    def reader(self, conn: socket.socket) -> None:
        buf = b""
        while True:
            try:
                chunk = conn.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while len(buf) >= 5:
                t, n = struct.unpack_from("<BI", buf)
                if n > 4 * 1024 * 1024:
                    log("FRAME TOO LARGE", n)
                    return
                if len(buf) < 5 + n:
                    break
                p, buf = buf[5 : 5 + n], buf[5 + n :]
                with self.cv:
                    if t == MIC:
                        self.mic_times.append(time.monotonic())
                        self.mic_sizes.add(len(p))
                        self.mic_last = p
                    elif t == HELLO:
                        self.hello = json.loads(p)
                        self.hellos.append(self.hello)
                        log("HELLO", self.hello)
                    elif t == DRAINED:
                        sid = struct.unpack("<I", p)[0]
                        self.drained[sid] = time.monotonic()
                        log("DRAINED", sid)
                    else:
                        log("unexpected frame", NAMES.get(t, t))
                    self.cv.notify_all()
        log("audio connection closed")

    def wait(self, pred, timeout: float) -> bool:
        with self.cv:
            return self.cv.wait_for(pred, timeout)


class ControlSide:
    def __init__(self) -> None:
        self.conn: socket.socket | None = None
        self.cmds: list[dict] = []
        self.status_polls = 0
        self.voice_engine = "kokoro"      # B2: status.voice_engine is "kokoro" | "clone"
        self.settings: dict = {           # settings_get / settings_set (B2's flat dotted keys)
            "name": "Mac", "user_name": "", "results_dir": "~/Documents/Mac",
            "wake.phrases": ["hey mac", "ok mac"], "listen.followup_s": 6.0, "tts.voice": "af_heart",
            "tts.speed": 1.0, "tts.lang": "en-us", "tts.engine": "kokoro", "brain.idle_shutdown_min": 10,
            "brain.chrome": True, "brain.setting_sources": ["user"], "remote.enabled": False,
            "clone_sample": None, "stt_languages": ["en"],
        }
        self.voices = [
            {"id": "af_heart", "name": "Heart", "lang": "en-us", "gender": "f"},
            {"id": "af_bella", "name": "Bella", "lang": "en-us", "gender": "f"},
            {"id": "am_adam", "name": "Adam", "lang": "en-us", "gender": "m"},
            {"id": "bf_emma", "name": "Emma", "lang": "en-gb", "gender": "f"},
            {"id": "bm_george", "name": "George", "lang": "en-gb", "gender": "m"},
            {"id": "ff_siwis", "name": "Siwis", "lang": "fr", "gender": "f"},
        ]
        self.clone_sample: str | None = None
        self.cv = threading.Condition()
        self.wlock = threading.Lock()

    def send(self, obj: dict) -> None:
        if not self.conn:
            return
        with self.wlock:
            self.conn.sendall((json.dumps(obj) + "\n").encode())

    def reader(self, conn: socket.socket) -> None:
        f = conn.makefile("rb")
        for line in f:
            try:
                msg = json.loads(line)
            except ValueError:
                log("bad control line", line[:100])
                continue
            log("control cmd", msg)
            with self.cv:
                self.cmds.append(msg)
                self.cv.notify_all()
            reply: dict = {"id": msg.get("id"), "ok": True, "data": {}}
            if msg.get("cmd") == "remote_link":
                reply["data"] = {"url": "https://meet.livekit.io/custom?liveKitUrl=wss://fake&token=TEST"}
            elif msg.get("cmd") == "status":
                # 1st poll: brain not probed yet (no block → UI must show "unknown", never ✓);
                # later polls: the realistic block (B2's shape).
                self.status_polls += 1
                reply["data"] = {"state": "idle", "voice": True, "voice_reason": "",
                                 "voice_engine": self.voice_engine, "clone_sample": self.clone_sample,
                                 "remote": {"active": False, "present": False, "verified": False, "pin": None}}
                if self.status_polls > 1:
                    reply["data"]["brain"] = {"alive": True, "auth": "ok", "chrome": "connected", "kind": "FakeBrain"}
            elif msg.get("cmd") == "remote_end":
                reply["data"] = {"ended": True}
            elif msg.get("cmd") == "settings_get":
                reply["data"] = dict(self.settings)
            elif msg.get("cmd") == "settings_set":
                applied, errors = {}, {}
                for k, v in (msg.get("changes") or {}).items():
                    if k == "name":
                        v = str(v).strip()
                        if not (1 <= len(v) <= 32) or not all(c.isalnum() or c in " -" for c in v):
                            errors[k] = "name must be 1–32 letters, digits, spaces or dashes"
                            continue
                        applied["wake.phrases"] = [f"hey {v.lower()}", f"ok {v.lower()}"]
                    elif k == "tts.speed":
                        if not (0.5 <= float(v) <= 2.0):
                            errors[k] = "tts.speed must be between 0.5 and 2"
                            continue
                        v = round(float(v), 2)
                    elif k == "listen.followup_s" and not (0 <= float(v) <= 60):
                        errors[k] = "followup_s must be between 0 and 60"
                        continue
                    elif k == "brain.idle_shutdown_min" and not (1 <= int(v) <= 720):
                        errors[k] = "idle_shutdown_min must be between 1 and 720"
                        continue
                    applied[k] = v
                self.settings.update(applied)
                reply["data"] = {"applied": applied, "errors": errors}
            elif msg.get("cmd") == "voices_list":
                reply["data"] = {"voices": list(self.voices)}
            elif msg.get("cmd") == "voice_preview":
                vid = msg.get("voice")
                if not any(v["id"] == vid for v in self.voices):
                    reply = {"id": msg.get("id"), "ok": False, "error": f"cannot preview voice '{vid}': unknown voice"}
                else:
                    reply["data"] = {"queued": True, "voice": vid, "text": "Hi, I'm Mac. This is how I sound."}
            elif msg.get("cmd") in ("voice_clone", "voice_reset"):
                # Voice cloning takes a moment: reply after 1 s from a side thread.
                path = msg.get("path")
                if msg.get("cmd") == "voice_clone" and (not isinstance(path, str) or not os.path.exists(path)):
                    reply = {"id": msg.get("id"), "ok": False, "error": "Sample file not found."}
                else:
                    if msg.get("cmd") == "voice_clone":
                        self.voice_engine, self.clone_sample = "clone", path
                        self.settings.update({"tts.engine": "clone", "clone_sample": path})
                    else:
                        self.voice_engine, self.clone_sample = "kokoro", None
                        self.settings.update({"tts.engine": "kokoro", "clone_sample": None})
                    sample = os.path.basename(self.clone_sample) if self.clone_sample else None
                    reply["data"] = {"engine": self.voice_engine, "clone_sample": self.clone_sample,
                                     "voice_engine": {"engine": self.voice_engine, "sample": sample}}
                threading.Timer(1.0, self.send, args=(reply,)).start()
                continue
            self.send(reply)
        log("control connection closed")


def serve(path: Path, on_conn) -> socket.socket:
    if path.exists():
        path.unlink()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(path))
    os.chmod(path, 0o600)
    s.listen(4)

    def loop() -> None:
        while True:
            try:
                c, _ = s.accept()
            except OSError:
                return
            on_conn(c)

    threading.Thread(target=loop, daemon=True).start()
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--hold", action="store_true")
    ap.add_argument("--mic-seconds", type=float, default=5.0)
    args = ap.parse_args()

    home = Path(os.environ.get("MAC_HOME", "")).expanduser()
    if not str(home) or str(home) == ".":
        print("set MAC_HOME", file=sys.stderr)
        return 2
    run = home / "run"
    run.mkdir(parents=True, exist_ok=True)
    os.chmod(run, 0o700)

    audio, control = AudioSide(), ControlSide()

    def on_audio(c: socket.socket) -> None:
        log("audio client connected")
        audio.conn = c
        audio.send(HELLO, json.dumps({"version": 1, "role": "macd", "sample_rate": 16000}).encode())
        threading.Thread(target=audio.reader, args=(c,), daemon=True).start()

    def on_control(c: socket.socket) -> None:
        log("control client connected")
        control.conn = c
        threading.Thread(target=control.reader, args=(c,), daemon=True).start()

    servers = [serve(run / "audio.sock", on_audio), serve(run / "control.sock", on_control)]
    log(f"listening in {run}")
    results: dict[str, object] = {}
    ok = True

    def check(name: str, cond: bool, detail: object) -> None:
        nonlocal ok
        ok &= bool(cond)
        results[name] = {"pass": bool(cond), "detail": detail}
        log(("PASS " if cond else "FAIL ") + name, detail)

    # --- connect + HELLO
    got = audio.wait(lambda: audio.hello is not None, args.timeout)
    check("hello", got and {"version", "input_device", "aec"} <= set(audio.hello or {}), audio.hello)
    if not got:
        print(json.dumps(results, indent=2))
        return 1

    # --- control subscribe + events
    with control.cv:
        control.cv.wait_for(lambda: any(c.get("cmd") == "subscribe" for c in control.cmds), 10)
    check("control_subscribe", any(c.get("cmd") == "subscribe" for c in control.cmds),
          [c.get("cmd") for c in control.cmds])
    with control.cv:
        control.cv.wait_for(lambda: any(c.get("cmd") == "status" for c in control.cmds), 10)
    check("control_status_polled", any(c.get("cmd") == "status" for c in control.cmds),
          [c.get("cmd") for c in control.cmds])
    # (a second poll — with the brain block — is checked at the end of the run)
    for ev in (
        {"event": "state", "state": "listening"},
        {"event": "transcript", "role": "user", "text": "what's the weather like", "final": True},
        {"event": "state", "state": "thinking"},
        {"event": "activity", "text": "Opening weather.com"},
        {"event": "state", "state": "speaking"},
        {"event": "transcript", "role": "assistant", "text": "It's sunny and 21 degrees."},
        {"event": "remote", "active": True, "pin": "1234", "identity": "phone"},
        {"event": "remote", "active": False, "pin": ""},
        {"event": "notify", "title": "fake_daemon test", "body": "Click to reveal the Mac folder", "path": str(home)},
    ):
        control.send(ev)
        time.sleep(0.15)

    # --- AEC policy: the state events above went idle→listening→thinking→speaking, so the
    # engine must now run WITH voice processing (HELLO aec=true, real input device).
    got = audio.wait(lambda: any(h.get("input_device") not in (None, "none") and h.get("aec") for h in audio.hellos), 10)
    check("aec_on_during_turn", got, audio.hellos[-1] if audio.hellos else None)
    # --- MIC rate (measure only once the engine is up: HELLO with a real input device)
    audio.wait(lambda: any(h.get("input_device") not in (None, "none") for h in audio.hellos), 10)
    time.sleep(1.0)  # let the engine settle
    with audio.cv:
        start_n = len(audio.mic_times)
    t0 = time.monotonic()
    time.sleep(args.mic_seconds)
    with audio.cv:
        n = len(audio.mic_times) - start_n
        sizes = sorted(audio.mic_sizes)
    rate = n / (time.monotonic() - t0)
    check("mic_rate", 45 <= rate <= 55, f"{n} frames in {args.mic_seconds:.1f}s = {rate:.1f}/s")
    check("mic_payload_640", sizes == [640], sizes)
    with audio.cv:
        last = audio.mic_last
    samples = struct.unpack(f"<{len(last)//2}h", last) if last else ()
    rms = math.sqrt(sum(x * x for x in samples) / max(1, len(samples))) / 32768
    log(f"info: last mic frame RMS = {rms:.5f} (0.0 exactly = silence: mic permission not granted or muted)")

    # --- PLAY + END → DRAINED
    pcm = tone(0.5)
    chunk = 24000 * 2 // 50  # 20 ms
    t_play = time.monotonic()
    for i in range(0, len(pcm), chunk):
        audio.send(PLAY, u32(1) + u32(24000) + pcm[i : i + chunk])
    audio.send(END, u32(1))
    got = audio.wait(lambda: 1 in audio.drained, 10)
    dt = audio.drained.get(1, time.monotonic()) - t_play
    check("play_end_drained", got and dt >= 0.45, f"DRAINED(1) after {dt:.3f}s (audio 0.500s)")

    # --- END without PLAY
    t = time.monotonic()
    audio.send(END, u32(7))
    got = audio.wait(lambda: 7 in audio.drained, 3)
    check("end_without_play", got, f"{(audio.drained.get(7, t) - t)*1000:.0f} ms")

    # --- FLUSH(id)
    pcm3 = tone(3.0, freq=330)
    for i in range(0, len(pcm3), chunk):
        audio.send(PLAY, u32(2) + u32(24000) + pcm3[i : i + chunk])
    time.sleep(0.3)
    t = time.monotonic()
    audio.send(FLUSH, u32(2))
    got = audio.wait(lambda: 2 in audio.drained, 3)
    lat = audio.drained.get(2, t) - t
    check("flush_id_drained", got and lat < 0.5, f"DRAINED(2) {lat*1000:.0f} ms after FLUSH(2)")

    # --- FLUSH(0)
    for i in range(0, len(pcm3), chunk):
        audio.send(PLAY, u32(3) + u32(16000) + pcm3[i : i + chunk])
    time.sleep(0.3)
    t = time.monotonic()
    audio.send(FLUSH, u32(0))
    got = audio.wait(lambda: 3 in audio.drained, 3)
    lat = audio.drained.get(3, t) - t
    check("flush_all_drained", got and lat < 0.5, f"DRAINED(3) {lat*1000:.0f} ms after FLUSH(0)")

    # PLAY for a flushed stream must be ignored (no second DRAINED, no crash)
    audio.send(PLAY, u32(3) + u32(24000) + pcm[:chunk])
    with audio.cv:
        n_hello = len(audio.hellos)
    control.send({"event": "state", "state": "idle"})
    # idle → voice processing off: a fresh HELLO with aec=false and a real device
    got = audio.wait(lambda: len(audio.hellos) > n_hello and audio.hellos[-1].get("aec") is False
                     and audio.hellos[-1].get("input_device") not in (None, "none"), 6)
    check("aec_off_when_idle", got, audio.hellos[-1] if audio.hellos else None)
    time.sleep(1.0)
    with audio.cv:
        before = len(audio.mic_times)
    time.sleep(1.0)
    with audio.cv:
        alive = len(audio.mic_times) > before + 30
    check("still_streaming_after_tests", alive, f"{len(audio.mic_times) - before} mic frames in 1 s after AEC-off rebuild")

    # --- off switch: state=off must release the engine (no MIC frames), state=idle resumes
    with audio.cv:
        n_hello = len(audio.hellos)
    control.send({"event": "state", "state": "off"})
    got = audio.wait(lambda: len(audio.hellos) > n_hello and audio.hellos[-1].get("input_device") == "none", 5)
    check("off_hello_none", got, audio.hellos[-1] if audio.hellos else None)
    time.sleep(1.0)  # in-flight frames settle
    with audio.cv:
        n_off = len(audio.mic_times)
    time.sleep(1.5)
    with audio.cv:
        n_off2 = len(audio.mic_times)
        n_hello = len(audio.hellos)
    check("off_mic_stopped", n_off2 == n_off, f"{n_off2 - n_off} MIC frames in 1.5 s while off")
    control.send({"event": "state", "state": "idle"})
    got = audio.wait(lambda: len(audio.hellos) > n_hello and audio.hellos[-1].get("input_device") not in (None, "none"), 5)
    check("on_fresh_hello", got, audio.hellos[-1] if audio.hellos else None)
    time.sleep(1.0)
    with audio.cv:
        n_on = len(audio.mic_times)
    t0 = time.monotonic()
    time.sleep(2.0)
    with audio.cv:
        n_on2 = len(audio.mic_times)
    rate = (n_on2 - n_on) / (time.monotonic() - t0)
    check("on_mic_resumed", 45 <= rate <= 55, f"{rate:.1f} frames/s after turning on")

    with control.cv:
        control.cv.wait_for(lambda: control.status_polls >= 2, 12)
    check("status_polled_both_shapes", control.status_polls >= 2,
          f"{control.status_polls} status polls (1st without brain block, then with auth=ok/chrome=connected)")

    print(json.dumps({"ok": ok, "results": results, "commands": control.cmds}, indent=2), flush=True)
    if args.hold:
        log("holding (Ctrl-C to exit)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    for s in servers:
        s.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
