from __future__ import annotations

import asyncio
import re
import time
import urllib.parse
from typing import ClassVar

import numpy as np
import pytest

from macd import config
from macd.contracts import FRAME_SAMPLES
from macd.remote import livekit_bridge as lb

FAKE = lb.Credentials(url="wss://example.livekit.cloud", api_key="APIfake123", api_secret="s" * 48)


def test_room_names_are_random_128_bit() -> None:
    names = {lb.new_room_name() for _ in range(50)}
    assert len(names) == 50
    for n in names:
        assert re.fullmatch(r"mac-[0-9a-f]{32}", n)


def test_phone_token_grants_and_ttl() -> None:
    from livekit import api

    room = lb.new_room_name()
    tok = lb.mint_token(FAKE, room, lb.PHONE_IDENTITY, 900)
    claims = api.TokenVerifier(FAKE.api_key, FAKE.api_secret).verify(tok)
    assert claims.identity == "phone"
    assert claims.video is not None
    assert claims.video.room == room and claims.video.room_join is True
    assert not claims.video.room_admin and not claims.video.room_create and not claims.video.room_list
    assert claims.video.can_publish_sources == ["microphone"]
    import jwt

    raw = jwt.decode(tok, options={"verify_signature": False})
    assert 890 <= raw["exp"] - int(time.time()) <= 905
    with pytest.raises(jwt.InvalidTokenError):
        api.TokenVerifier(FAKE.api_key, "wrong-secret" * 4).verify(tok)


def test_agent_token() -> None:
    from livekit import api

    tok = lb.mint_token(FAKE, "mac-x", lb.AGENT_IDENTITY, 3600, agent=True)
    claims = api.TokenVerifier(FAKE.api_key, FAKE.api_secret).verify(tok)
    assert claims.identity == "mac-agent" and claims.video.room == "mac-x"
    assert claims.video.can_publish and claims.video.can_subscribe


def test_meet_url_is_encoded() -> None:
    url = lb.meet_url("wss://x.livekit.cloud", "a.b+c/d=e")
    parsed = urllib.parse.urlparse(url)
    assert url.startswith("https://meet.livekit.io/custom?") and not parsed.fragment
    assert urllib.parse.parse_qs(parsed.query) == {"liveKitUrl": ["wss://x.livekit.cloud"], "token": ["a.b+c/d=e"]}
    assert "+" not in parsed.query.split("token=")[1] and "/" not in parsed.query.split("token=")[1]


def test_pin_and_configured() -> None:
    pins = {lb.new_pin() for _ in range(50)}
    assert all(len(p) == 4 and p.isdigit() for p in pins) and len(pins) > 10
    assert lb.configured(lambda n: "x") and not lb.configured(lambda n: None if n == "LIVEKIT_URL" else "x")


def test_credentials_never_in_repr() -> None:
    assert FAKE.api_secret not in repr(FAKE) and FAKE.api_key not in repr(FAKE)


def test_missing_credentials_message() -> None:
    with pytest.raises(lb.RemoteUnavailable) as ei:
        lb.load_credentials(lambda name: None)
    assert "LIVEKIT_API_SECRET" in str(ei.value)


async def test_remote_input_rechunks_and_bounds() -> None:
    inp = lb.RemoteInput("phone", max_frames=4)
    assert inp.name == "remote:phone" and inp.requires_wake_word is False
    inp.feed(np.ones(FRAME_SAMPLES + 100, dtype=np.float32))
    inp.feed(np.ones(FRAME_SAMPLES - 100, dtype=np.float32))
    for _ in range(10):
        inp.feed(np.zeros(FRAME_SAMPLES, dtype=np.float32))
    assert inp.overruns == 8  # 12 frames into a 4-frame queue: oldest dropped
    await inp.close()
    frames = [f async for f in inp.frames()]
    assert all(len(f) == FRAME_SAMPLES for f in frames)


async def test_bridge_requires_credentials() -> None:
    bridge = lb.LiveKitBridge(config.Config(), None, None, lambda inp: None, get_secret=lambda n: None)  # type: ignore[arg-type]
    with pytest.raises(lb.RemoteUnavailable):
        await bridge.create_link()
    assert not bridge.active and bridge.status() == {"active": False, "configured": False}


@pytest.mark.live
async def test_live_join_publish_leave() -> None:
    """Real LiveKit: join a random room, publish a tone on mac-voice, leave cleanly."""
    if not lb.configured():
        pytest.skip("LiveKit credentials not configured")

    class Orch:
        sessions: ClassVar[list[str | None]] = []
        presence: ClassVar[list[bool]] = []

        def on_remote_session(self, pin: str | None) -> None:
            self.sessions.append(pin)

        def set_remote_present(self, present: bool, identity: str | None = None) -> None:
            self.presence.append(present)

        def add_listener(self, *a: object, **k: object) -> None:
            return

        def remove_listener(self, name: str) -> None:
            return

    class Speech:
        outputs: ClassVar[dict[str, object]] = {}

        def add_output(self, name: str, out: object) -> None:
            self.outputs[name] = out

        def remove_output(self, name: str) -> None:
            self.outputs.pop(name, None)

    orch, speech = Orch(), Speech()
    bridge = lb.LiveKitBridge(config.Config(), orch, speech, lambda inp: None)  # type: ignore[arg-type]
    link = await bridge.create_link()
    try:
        assert link["url"].startswith("https://meet.livekit.io/custom?liveKitUrl=") and "&token=" in link["url"]
        assert re.fullmatch(r"mac-[0-9a-f]{32}", link["room"]) and link["pin"].isdigit()
        assert bridge.active and "remote" in speech.outputs and orch.sessions == [link["pin"]]
        sess = bridge._session
        assert sess is not None and sess.room.isconnected()
        assert any(p.name == lb.VOICE_TRACK for p in sess.room.local_participant.track_publications.values())
        t = np.arange(int(0.3 * 24_000)) / 24_000
        tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        t0 = time.monotonic()
        await asyncio.wait_for(speech.outputs["remote"].play(tone, 24_000), 5)  # type: ignore[attr-defined]
        assert time.monotonic() - t0 >= 0.2  # actually paced in real time
    finally:
        await bridge.close()
    assert not bridge.active and "remote" not in speech.outputs and orch.sessions[-1] is None
