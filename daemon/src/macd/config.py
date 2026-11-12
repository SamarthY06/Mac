"""Configuration: ~/.mac-agent/config.toml (+ MAC_* env overrides) and secrets.

Paths are centralised here so every subsystem agrees on them.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HOME = Path(os.environ.get("MAC_HOME", Path.home() / ".mac-agent")).expanduser()
RUN_DIR = HOME / "run"
MODELS_DIR = Path(os.environ.get("MAC_MODELS_DIR", HOME / "models")).expanduser()
WORKSPACE_DIR = HOME / "workspace"
LOG_DIR = Path(os.environ.get("MAC_LOG_DIR", Path.home() / "Library/Logs/Mac")).expanduser()
CONFIG_PATH = HOME / "config.toml"
STATE_PATH = HOME / "state.json"
VOICE_DIR = HOME / "voice"  # cloned-voice reference sample
MEMORY_PATH = WORKSPACE_DIR / "memory.md"
CONTROL_SOCK = RUN_DIR / "control.sock"
AUDIO_SOCK = RUN_DIR / "audio.sock"
EVENT_LOG = LOG_DIR / "events.jsonl"
KEYCHAIN_SERVICE = "mac-agent"


@dataclass
class WakeConfig:
    phrases: list[str] = field(default_factory=lambda: ["hey mac", "ok mac"])
    kws_threshold: float = 0.1  # sherpa-onnx keywords_threshold (lower = more sensitive; 0.25 missed accented speech)
    kws_score: float = 1.5  # sherpa-onnx keywords_score (boost; higher values LOWER recall on this model)
    verify: bool = True  # stage-2 STT verification of wake hits
    accept_similar_names: bool = True  # stage 2 also accepts "hey mark/make/mic..." (what STT hears for accented "mac")
    # Stage 2 hearing no words at all is no evidence against the hit: Parakeet often returns ""
    # for a lone short "Hey Mac" (measured: 6 of 50 clear wakes). Words that are not the
    # phrase still reject. Cost: a rare noise hit makes Mac listen for a few seconds.
    accept_unheard: bool = True


@dataclass
class ListenConfig:
    silence_ms: int = 700  # end-of-utterance silence (without the Smart Turn model)
    # Smart Turn (semantic end of turn): after turn_check_ms of silence the model decides from the
    # audio whether the sentence is finished; if not ("tell me about..."), wait for more speech up to
    # turn_max_silence_ms. Pipecat's pattern (their stop_secs is 3 s).
    smart_turn: bool = True
    turn_check_ms: int = 300
    turn_max_silence_ms: int = 2000
    turn_threshold: float = 0.5
    max_utterance_s: float = 60.0
    no_speech_timeout_s: float = 5.0  # after wake, give up if nothing is said
    followup_s: float = 6.0  # after speaking, listen without wake word
    followup_after_question_s: float = 15.0  # longer window when Mac just asked a question
    confirm_window_s: float = 10.0
    vad_threshold: float = 0.5
    tail_guard_ms: int = 300  # ignore input right after playback ends (no-AEC only)
    # Talking over Mac (AEC only): VAD onset this long pauses speech; it stops for real only on
    # this many words that are not a backchannel/echo, otherwise it resumes (LiveKit/Vapi style).
    barge_in: bool = True
    barge_in_onset_ms: int = 300
    barge_in_min_words: int = 3  # Pipecat MinWords default; live test: a misheard "mm-hmm" became "An engine."


@dataclass
class TTSConfig:
    voice: str = "af_heart"
    speed: float = 1.05
    lang: str = "en-us"
    engine: str = "kokoro"  # "kokoro" | "clone" (Kyutai Pocket TTS with a reference sample)
    clone_sample: str = ""  # path of the reference wav (copied under ~/.mac-agent/voice/)
    clone_threads: int = 2


@dataclass
class STTConfig:
    model: str = "parakeet-tdt-0.6b-v2-int8"
    threads: int = 4
    lazy: bool = False


@dataclass
class BrainConfig:
    auth: str = "claude_cli"  # "claude_cli" (your Claude Code login) | "api_key"
    cli_path: str = ""  # empty = autodetect `claude` on PATH / ~/.local/bin
    model: str = ""  # empty = Claude Code default
    chrome: bool = True
    # True: the hook refuses every outward action (send, post, pay, delete, upload) until the user
    # says yes to that exact action. False: the model decides when to ask; the hook only blocks
    # dangerous actions (destructive commands, secrets, Mac's own files, the never-list).
    confirm_outward: bool = True
    # "auto": Claude Code's own permission classifier runs on every tool call; what it escalates is
    # put to the user as a confirmation (spoken, or a question in the chat). "bypassPermissions":
    # no classifier; only this project's hook decides.
    permission_mode: str = "auto"
    setting_sources: list[str] = field(default_factory=lambda: ["user"])  # load ~/.claude MCP servers/skills/plugins; [] to isolate
    session_idle_rotate_min: int = 180  # after this long idle, handoff() writes memory and the next turn starts a fresh session
    idle_shutdown_min: int = 10  # kill the claude process after this long with no work
    warm_on_wake: bool = True  # start the claude process as soon as the wake word is heard
    max_turn_s: int = 900
    narration_silence_s: float = 6.0
    mode: str = "local"  # "local" (brain inside macd) | "remote" (agentd's voice:local over agent.sock, W6)


@dataclass
class AudioConfig:
    backend: str = "auto"  # "auto" (swift if app link present, else sounddevice) | "swift" | "sounddevice"
    input_device: str = ""  # sounddevice only; "" = built-in mic preferred
    prefer_builtin_mic: bool = True


@dataclass
class RemoteConfig:
    enabled: bool = False
    token_ttl_s: int = 900
    empty_room_leave_s: int = 60


@dataclass
class Config:
    name: str = "Mac"
    user_name: str = ""
    results_dir: str = "~/Documents/Mac"  # where Mac saves lists, drafts and files it produces
    wake: WakeConfig = field(default_factory=WakeConfig)
    listen: ListenConfig = field(default_factory=ListenConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)


def _merge(obj: Any, data: dict[str, Any], prefix: str = "") -> None:
    for f in dataclasses.fields(obj):
        cur = getattr(obj, f.name)
        if dataclasses.is_dataclass(cur):
            _merge(cur, data.get(f.name, {}) or {}, f"{prefix}{f.name}_")
            continue
        if f.name in data:
            setattr(obj, f.name, type(cur)(data[f.name]) if not isinstance(cur, list) else list(data[f.name]))
        env = os.environ.get(f"MAC_{prefix}{f.name}".upper())
        if env is not None:
            if isinstance(cur, bool):
                setattr(obj, f.name, env.lower() in ("1", "true", "yes", "on"))
            elif isinstance(cur, list):
                setattr(obj, f.name, [s.strip() for s in env.split(",") if s.strip()])
            else:
                setattr(obj, f.name, type(cur)(env))


def _to_toml(obj: Any, section: str = "") -> str:
    lines: list[str] = []
    subs: list[str] = []
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if dataclasses.is_dataclass(v):
            subs.append(f"\n[{f.name}]\n" + _to_toml(v, f.name))
        elif isinstance(v, bool):
            lines.append(f"{f.name} = {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{f.name} = {v}")
        elif isinstance(v, list):
            lines.append(f"{f.name} = [" + ", ".join(f'"{x}"' for x in v) + "]")
        else:
            lines.append(f'{f.name} = "{v}"')
    return "\n".join(lines) + "".join(subs) + "\n"


def to_dict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def save(cfg: Config, path: Path | None = None) -> None:
    """Write the whole config to config.toml atomically (tomli_w). Env overrides are not saved."""
    import os as _os

    import tomli_w

    path = path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text("# Mac agent configuration. Edit here or in the app's Settings window.\n" + tomli_w.dumps(to_dict(cfg)))
    _os.replace(tmp, path)


SETTABLE_KEYS: dict[str, type] = {
    # dotted key -> type; the subset the Settings window / `settings_set` may change at runtime
    "name": str, "user_name": str, "results_dir": str,
    "wake.phrases": list, "wake.kws_threshold": float,
    "listen.silence_ms": int, "listen.smart_turn": bool, "listen.turn_max_silence_ms": int, "listen.followup_s": float, "listen.max_utterance_s": float,
    "tts.voice": str, "tts.speed": float, "tts.lang": str, "tts.engine": str,
    "brain.idle_shutdown_min": int, "brain.model": str, "brain.chrome": bool, "brain.confirm_outward": bool,
    "brain.setting_sources": list, "remote.enabled": bool,
}


def set_key(cfg: Config, dotted: str, value: Any) -> Any:
    """Set one SETTABLE key on cfg with type coercion; returns the coerced value. ValueError if unknown."""
    if dotted not in SETTABLE_KEYS:
        raise ValueError(f"setting not allowed: {dotted}")
    typ = SETTABLE_KEYS[dotted]
    if typ is bool and isinstance(value, str):
        value = value.lower() in ("1", "true", "yes", "on")
    elif typ is list:
        value = [str(v).strip() for v in (value.split(",") if isinstance(value, str) else value) if str(v).strip()]
    else:
        value = typ(value)
    obj: Any = cfg
    *parents, leaf = dotted.split(".")
    for part in parents:
        obj = getattr(obj, part)
    setattr(obj, leaf, value)
    return value


def get_key(cfg: Config, dotted: str) -> Any:
    obj: Any = cfg
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def ensure_dirs() -> None:
    old = os.umask(0o077)
    try:
        for d in (HOME, RUN_DIR, MODELS_DIR, WORKSPACE_DIR, LOG_DIR):
            d.mkdir(parents=True, exist_ok=True)
        os.chmod(RUN_DIR, 0o700)
    finally:
        os.umask(old)


def load(path: Path | None = None, write_default: bool = True) -> Config:
    path = path or CONFIG_PATH
    cfg = Config()
    data: dict[str, Any] = {}
    if path.exists():
        data = tomllib.loads(path.read_text())
    elif write_default:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Mac agent configuration. Delete this file to reset to defaults.\n" + _to_toml(cfg))
    _merge(cfg, data)
    return cfg


def read_state() -> dict[str, Any]:
    """Daemon-wide persisted state (session id, listening_off, ...). Missing/corrupt -> {}."""
    import json

    try:
        data = json.loads(STATE_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def update_state(**fields: Any) -> dict[str, Any]:
    """Merge `fields` into state.json atomically (tmp file + rename). Returns the new state."""
    import json
    import os as _os

    state = read_state()
    state.update(fields)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=0, sort_keys=True))
    _os.replace(tmp, STATE_PATH)
    return state


def get_secret(name: str) -> str | None:
    """Secret lookup order: env var -> macOS Keychain (service mac-agent) -> .env files."""
    if os.environ.get(name):
        return os.environ[name]
    import logging
    import subprocess

    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", name, "-w"],
            check=False, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:  # keychain unavailable: fall through to .env
        logging.getLogger(__name__).debug("keychain lookup for %s failed: %s", name, e)
    for env_file in (HOME / ".env", Path(__file__).resolve().parents[3] / ".env"):
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                k, _, v = line.partition("=")
                if k.strip() == name and v.strip():
                    return v.strip().strip('"').strip("'")
    return None


def set_secret(name: str, value: str) -> None:
    import subprocess

    subprocess.run(
        ["security", "add-generic-password", "-U", "-s", KEYCHAIN_SERVICE, "-a", name, "-w", value],
        check=True, capture_output=True, timeout=10,
    )
