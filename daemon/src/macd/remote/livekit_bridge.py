"""LiveKit bridge: talk to your Mac from anywhere (ARCHITECTURE §10, N7).

`create_link()` makes a random room `mac-<128-bit hex>`, joins it as `mac-agent`, mints a
single-identity (`phone`), room-scoped, mic-only, short-TTL token and returns a LiveKit Meet
URL. A 4-digit PIN is shown by the
app; the orchestrator refuses remote commands until the phone speaks it. The phone's mic is a
`RemoteInput` (no wake word, 16 kHz frames) and speech for remote turns goes to `RemoteOutput`
(the `mac-voice` track). The session ends on `remote_end`, when the room has been empty for
`cfg.remote.empty_room_leave_s`, or when nobody joined before the token expired. One session at
a time. Secrets come from config.get_secret and are never logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import logging
import secrets as _secrets
import time
import urllib.parse
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from macd import config
from macd.audio.resample import float_to_int16, int16_to_float, resample
from macd.contracts import FRAME_SAMPLES, SAMPLE_RATE, AudioInput, Listener

if TYPE_CHECKING:  # pragma: no cover
    from livekit import rtc

    from macd.orchestrator import Orchestrator
    from macd.speech_queue import SpeechQueue

log = logging.getLogger(__name__)

AGENT_IDENTITY = "mac-agent"
PHONE_IDENTITY = "phone"
VOICE_TRACK = "mac-voice"
OUT_SAMPLE_RATE = 48_000
OUT_CHUNK = OUT_SAMPLE_RATE // 50  # 20 ms
MEET_BASE = "https://meet.livekit.io/custom"
AGENT_TOKEN_TTL_S = 12 * 3600
SECRET_NAMES = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
SETUP_HINT = "LiveKit isn't set up: run install.sh and choose 'Set up LiveKit', or `macctl secret set LIVEKIT_URL` (etc.)."


class RemoteUnavailable(RuntimeError):
    pass


# ----------------------------------------------------------------------------- tokens


@dataclass(frozen=True)
class Credentials:
    url: str
    api_key: str
    api_secret: str

    def __repr__(self) -> str:  # never leak secrets via repr/logging
        return f"Credentials(url={self.url!r}, api_key=<redacted>, api_secret=<redacted>)"


def configured(get_secret: Callable[[str], str | None] = config.get_secret) -> bool:
    return all(get_secret(n) for n in SECRET_NAMES)


def load_credentials(get_secret: Callable[[str], str | None] = config.get_secret) -> Credentials:
    values = {n: get_secret(n) for n in SECRET_NAMES}
    missing = [n for n, v in values.items() if not v]
    if missing:
        raise RemoteUnavailable(f"{SETUP_HINT} Missing: {', '.join(missing)}.")
    return Credentials(url=values["LIVEKIT_URL"] or "", api_key=values["LIVEKIT_API_KEY"] or "",
                       api_secret=values["LIVEKIT_API_SECRET"] or "")


def new_room_name() -> str:
    return f"mac-{_secrets.token_hex(16)}"


def new_pin() -> str:
    return f"{_secrets.randbelow(10_000):04d}"


def mint_token(creds: Credentials, room: str, identity: str, ttl_s: int, *, agent: bool = False) -> str:
    from livekit import api

    grants = api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True, can_publish_data=agent,
                             can_publish_sources=None if agent else ["microphone"])
    tok = (api.AccessToken(creds.api_key, creds.api_secret).with_identity(identity)
           .with_name("Mac" if agent else "Phone").with_grants(grants)
           .with_ttl(datetime.timedelta(seconds=ttl_s)))
    return tok.to_jwt()


def meet_url(livekit_url: str, token: str) -> str:
    # Query string (not fragment) because the stock Meet page reads it there; the protection is
    # the 15-min single-identity mic-only token plus the spoken PIN gate, not URL secrecy.
    q = urllib.parse.urlencode({"liveKitUrl": livekit_url, "token": token}, quote_via=urllib.parse.quote)
    return f"{MEET_BASE}?{q}"


# ----------------------------------------------------------------------------- audio I/O


class RemoteInput:
    """AudioInput fed by a LiveKit AudioStream (16 kHz mono float32, FRAME_SAMPLES frames)."""

    requires_wake_word = False

    def __init__(self, identity: str, max_frames: int = 250) -> None:
        self.name = f"remote:{identity}"
        self.identity = identity
        self._q: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=max_frames)
        self._carry = np.zeros(0, dtype=np.float32)
        self._closed = False
        self.overruns = 0

    async def start(self) -> None:
        return

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._put(None)

    def _put(self, item: np.ndarray | None) -> None:
        while True:
            try:
                self._q.put_nowait(item)
                return
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    self._q.get_nowait()  # drop oldest
                    self.overruns += 1

    def feed(self, pcm: np.ndarray) -> None:
        """Accept any-length 16 kHz float32 audio; re-chunk into exact frames."""
        if self._closed:
            return
        buf = np.concatenate([self._carry, pcm.astype(np.float32, copy=False)]) if self._carry.size else pcm
        n = (len(buf) // FRAME_SAMPLES) * FRAME_SAMPLES
        for i in range(0, n, FRAME_SAMPLES):
            self._put(np.array(buf[i : i + FRAME_SAMPLES], dtype=np.float32))
        self._carry = np.array(buf[n:], dtype=np.float32)

    async def frames(self) -> AsyncIterator[np.ndarray]:
        while True:
            item = await self._q.get()
            if item is None:
                return
            yield item


class RemoteOutput:
    """AudioOutput that publishes speech on the `mac-voice` track."""

    name = "remote"

    def __init__(self, source: rtc.AudioSource | None = None) -> None:
        self._source = source

    async def start(self) -> None:
        return

    async def close(self) -> None:
        self.flush()
        self._source = None

    @property
    def aec(self) -> bool:
        return True  # phones / browsers run their own echo cancellation

    def flush(self) -> None:
        if self._source is not None:
            with contextlib.suppress(Exception):
                self._source.clear_queue()

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        from livekit import rtc

        src = self._source
        if src is None:
            return
        data = float_to_int16(resample(np.asarray(pcm, dtype=np.float32).reshape(-1), sample_rate, OUT_SAMPLE_RATE))
        try:
            for i in range(0, len(data), OUT_CHUNK):
                chunk = data[i : i + OUT_CHUNK]
                if len(chunk) < OUT_CHUNK:
                    chunk = np.concatenate([chunk, np.zeros(OUT_CHUNK - len(chunk), dtype=np.int16)])
                await src.capture_frame(rtc.AudioFrame(chunk.tobytes(), OUT_SAMPLE_RATE, 1, OUT_CHUNK))
            await src.wait_for_playout()
        except asyncio.CancelledError:
            self.flush()
            raise


# ----------------------------------------------------------------------------- bridge


@dataclass(eq=False)
class _Session:
    room_name: str
    pin: str
    room: Any
    source: Any
    output: RemoteOutput
    expires_at: float
    created: float = field(default_factory=time.monotonic)
    inputs: dict[str, RemoteInput] = field(default_factory=dict)
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    ever_joined: bool = False
    empty_since: float | None = None
    closed: bool = False


class LiveKitBridge:
    def __init__(
        self,
        cfg: config.Config,
        orch: Orchestrator,
        speech: SpeechQueue,
        make_listener: Callable[[AudioInput], Listener],
        *,
        get_secret: Callable[[str], str | None] = config.get_secret,
    ) -> None:
        self.cfg = cfg
        self.orch = orch
        self.speech = speech
        self.make_listener = make_listener
        self._get_secret = get_secret
        self._session: _Session | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ public

    @property
    def active(self) -> bool:
        return self._session is not None and not self._session.closed

    def status(self) -> dict[str, Any]:
        s = self._session
        if s is None or s.closed:
            return {"active": False, "configured": configured(self._get_secret)}
        return {"active": True, "configured": True, "room": s.room_name, "pin": s.pin,
                "participants": sorted(s.room.remote_participants.keys()),
                "age_s": round(time.monotonic() - s.created, 1)}

    async def create_link(self) -> dict[str, Any]:
        """Start (or restart) the remote session and return the phone link + PIN."""
        async with self._lock:
            creds = load_credentials(self._get_secret)
            if self._session is not None:
                await self._teardown(self._session, "replaced by a new link")
            from livekit import rtc

            room_name = new_room_name()
            ttl = int(self.cfg.remote.token_ttl_s)
            agent_token = mint_token(creds, room_name, AGENT_IDENTITY, AGENT_TOKEN_TTL_S, agent=True)
            phone_token = mint_token(creds, room_name, PHONE_IDENTITY, ttl)
            room = rtc.Room()
            source = rtc.AudioSource(OUT_SAMPLE_RATE, 1)
            sess = _Session(room_name=room_name, pin=new_pin(), room=room, source=source,
                            output=RemoteOutput(source), expires_at=time.monotonic() + ttl)
            self._register(sess)
            try:
                await asyncio.wait_for(room.connect(creds.url, agent_token), timeout=20)
                track = rtc.LocalAudioTrack.create_audio_track(VOICE_TRACK, source)
                await room.local_participant.publish_track(
                    track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
            except Exception as e:  # noqa: BLE001 - any join failure is reported to the user
                sess.closed = True
                with contextlib.suppress(Exception):
                    await room.disconnect()
                with contextlib.suppress(Exception):
                    await source.aclose()
                raise RemoteUnavailable(f"could not join the LiveKit room: {type(e).__name__}: {e}") from None
            self._session = sess
            self.speech.add_output("remote", sess.output)
            self.orch.on_remote_session(sess.pin)
            for p in list(room.remote_participants.values()):
                self._on_participant_connected(sess, p)
            self._spawn(sess, self._watch(sess), "watchdog")
            log.info("remote session started in room %s", room_name)
            return {"url": meet_url(creds.url, phone_token), "room": room_name, "pin": sess.pin,
                    "expires_in_s": ttl, "configured": True}

    async def close(self) -> None:
        async with self._lock:
            if self._session is not None:
                await self._teardown(self._session, "ended")

    # ------------------------------------------------------------------ room events

    def _register(self, sess: _Session) -> None:
        room = sess.room

        @room.on("participant_connected")
        def _pc(p: Any) -> None:
            self._on_participant_connected(sess, p)

        @room.on("participant_disconnected")
        def _pd(p: Any) -> None:
            self._on_participant_disconnected(sess, p)

        @room.on("track_subscribed")
        def _ts(track: Any, _publication: Any, participant: Any) -> None:
            self._on_track(sess, track, participant)

        @room.on("disconnected")
        def _dc(*_args: Any) -> None:
            if not sess.closed:
                self._spawn(sess, self._teardown_locked(sess, "room disconnected"), "teardown")

    def _spawn(self, sess: _Session, coro: Any, name: str) -> None:
        t = asyncio.get_running_loop().create_task(coro, name=f"livekit:{name}")
        sess.tasks.add(t)
        t.add_done_callback(sess.tasks.discard)

    def _on_participant_connected(self, sess: _Session, p: Any) -> None:
        if sess.closed:
            return
        sess.ever_joined = True
        sess.empty_since = None
        log.info("remote participant joined: %s", p.identity)
        self.orch.set_remote_present(True, p.identity)

    def _on_participant_disconnected(self, sess: _Session, p: Any) -> None:
        if sess.closed:
            return
        inp = sess.inputs.pop(p.identity, None)
        if inp is not None:
            self._spawn(sess, inp.close(), "input-close")
            self.orch.remove_listener(inp.name)
        if not sess.room.remote_participants:
            sess.empty_since = time.monotonic()
            self.orch.set_remote_present(False)

    def _on_track(self, sess: _Session, track: Any, participant: Any) -> None:
        from livekit import rtc

        if sess.closed or track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        ident = participant.identity
        old = sess.inputs.pop(ident, None)
        if old is not None:
            self._spawn(sess, old.close(), "input-close")
            self.orch.remove_listener(old.name)
        inp = RemoteInput(ident)
        sess.inputs[ident] = inp
        self._spawn(sess, self._pump(inp, track), f"pump:{ident}")
        self.orch.add_listener(self.make_listener(inp), requires_wake=False)

    async def _pump(self, inp: RemoteInput, track: Any) -> None:
        from livekit import rtc

        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1, frame_size_ms=20, capacity=100)
        try:
            async for ev in stream:
                inp.feed(int16_to_float(np.frombuffer(ev.frame.data, dtype=np.int16)))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("remote audio stream failed")
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()

    async def _watch(self, sess: _Session) -> None:
        """Auto-leave: room empty for empty_room_leave_s, or nobody joined before expiry."""
        leave_after = float(self.cfg.remote.empty_room_leave_s)
        while not sess.closed:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            if not sess.ever_joined and now > sess.expires_at:
                await self._teardown_locked(sess, "nobody joined before the link expired")
                return
            if sess.empty_since is not None and now - sess.empty_since >= leave_after:
                await self._teardown_locked(sess, "room empty")
                return

    async def _teardown_locked(self, sess: _Session, why: str) -> None:
        async with self._lock:
            await self._teardown(sess, why)

    async def _teardown(self, sess: _Session, why: str) -> None:
        if sess.closed:
            return
        sess.closed = True
        if self._session is sess:
            self._session = None
        log.info("remote session %s ending: %s", sess.room_name, why)
        me = asyncio.current_task()
        for t in list(sess.tasks):
            if t is not me:
                t.cancel()
        for inp in list(sess.inputs.values()):
            await inp.close()
            self.orch.remove_listener(inp.name)
        sess.inputs.clear()
        self.speech.remove_output("remote")
        await sess.output.close()
        self.orch.on_remote_session(None)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(sess.room.disconnect(), timeout=5)
        with contextlib.suppress(Exception):
            await sess.source.aclose()
