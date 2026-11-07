import json

import numpy as np
import pytest

from macd.audio import protocol as P


def test_header_layout():
    b = P.encode_frame(P.FrameType.MIC, b"\x01\x02")
    assert b == b"\x01\x02\x00\x00\x00\x01\x02"


def test_play_roundtrip():
    pcm = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    b = P.encode_play(7, 24000, pcm)
    assert b[0] == 0x02
    (fr,) = P.FrameDecoder().feed(b)
    sid, sr, out = P.decode_play(fr.payload)
    assert (sid, sr) == (7, 24000)
    assert np.allclose(out, pcm, atol=1e-4)
    assert fr.payload[:8] == (7).to_bytes(4, "little") + (24000).to_bytes(4, "little")


def test_stream_id_frames():
    for enc, t in ((P.encode_flush, 3), (P.encode_drained, 4), (P.encode_end, 5)):
        b = enc(0xDEADBEEF)
        assert b[0] == t and b[1:5] == (4).to_bytes(4, "little")
        (fr,) = P.FrameDecoder().feed(b)
        assert P.decode_stream_id(fr.payload) == 0xDEADBEEF
    assert P.decode_stream_id(P.FrameDecoder().feed(P.encode_flush())[0].payload) == 0


def test_hello():
    b = P.encode_hello({"aec": True, "input_device": "MacBook Pro Microphone"})
    (fr,) = P.FrameDecoder().feed(b)
    h = P.decode_hello(fr.payload)
    assert h["aec"] is True and h["version"] == P.PROTOCOL_VERSION
    with pytest.raises(P.ProtocolError):
        P.decode_hello(b"[1]")
    with pytest.raises(P.ProtocolError):
        P.decode_hello(b"{nope")


def test_decoder_byte_by_byte_and_batched():
    pcm = np.linspace(-1, 1, 320, dtype=np.float32)
    stream = P.encode_mic(pcm) + P.encode_end(3) + P.encode_hello({"aec": False})
    dec = P.FrameDecoder()
    frames = []
    for i in range(len(stream)):
        frames += dec.feed(stream[i : i + 1])
    assert [f.type for f in frames] == [P.FrameType.MIC, P.FrameType.END, P.FrameType.HELLO]
    assert dec.pending == 0
    assert len(P.FrameDecoder().feed(stream)) == 3
    assert json.loads(frames[2].payload)["aec"] is False


def test_decoder_errors():
    with pytest.raises(P.ProtocolError):
        P.FrameDecoder().feed(b"\x09\x00\x00\x00\x00")
    with pytest.raises(P.ProtocolError):
        P.FrameDecoder(max_payload=10).feed(b"\x01\xff\x00\x00\x00")
    with pytest.raises(P.ProtocolError):
        P.bytes_to_pcm(b"\x00")
    with pytest.raises(P.ProtocolError):
        P.decode_stream_id(b"\x00\x00")


def test_pcm_clipping():
    out = P.bytes_to_pcm(P.pcm_to_bytes(np.array([2.0, -2.0], dtype=np.float32)))
    assert out[0] == pytest.approx(32767 / 32768) and out[1] == pytest.approx(-32767 / 32768)
