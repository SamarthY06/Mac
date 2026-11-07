"""Binary framing for audio.sock (Swift app <-> macd). See ARCHITECTURE.md §2.

Frame layout: ``type:u8 | len:u32le | payload`` (len = payload length in bytes).

| type | dir  | payload                                                    |
|------|------|------------------------------------------------------------|
| 0x01 | S->P | MIC: int16le PCM, 16 kHz mono                              |
| 0x02 | P->S | PLAY: stream_id:u32le, sample_rate:u32le, int16le PCM mono |
| 0x03 | P->S | FLUSH: stream_id:u32le (0 = everything)                    |
| 0x04 | S->P | DRAINED: stream_id:u32le                                   |
| 0x05 | P->S | END: stream_id:u32le                                       |
| 0x06 | both | HELLO: UTF-8 JSON {version, input_device, aec}             |

Pure module (no I/O) so it can be unit-tested and mirrored in Swift.
"""

from __future__ import annotations

import enum
import json
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from macd.audio.resample import float_to_int16, int16_to_float

PROTOCOL_VERSION = 1
HEADER = struct.Struct("<BI")
HEADER_SIZE = HEADER.size  # 5
U32 = struct.Struct("<I")
PLAY_HEADER = struct.Struct("<II")
MAX_PAYLOAD = 4 * 1024 * 1024  # sanity bound; a frame larger than this is a protocol error


class FrameType(enum.IntEnum):
    MIC = 0x01
    PLAY = 0x02
    FLUSH = 0x03
    DRAINED = 0x04
    END = 0x05
    HELLO = 0x06


class ProtocolError(ValueError):
    """Malformed frame on audio.sock."""


@dataclass(frozen=True)
class Frame:
    type: FrameType
    payload: bytes


def encode_frame(ftype: FrameType | int, payload: bytes = b"") -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(f"payload too large: {len(payload)}")
    return HEADER.pack(int(ftype), len(payload)) + payload


def pcm_to_bytes(pcm: np.ndarray) -> bytes:
    """float32 [-1,1] (or int16) -> int16le bytes."""
    x = pcm if pcm.dtype == np.int16 else float_to_int16(pcm)
    return x.astype("<i2", copy=False).tobytes()


def bytes_to_pcm(data: bytes) -> np.ndarray:
    """int16le bytes -> float32 [-1,1]. A trailing odd byte is a protocol error."""
    if len(data) % 2:
        raise ProtocolError("odd-length int16 payload")
    return int16_to_float(np.frombuffer(data, dtype="<i2"))


def encode_mic(pcm: np.ndarray) -> bytes:
    return encode_frame(FrameType.MIC, pcm_to_bytes(pcm))


def encode_play(stream_id: int, sample_rate: int, pcm: np.ndarray) -> bytes:
    return encode_frame(FrameType.PLAY, PLAY_HEADER.pack(stream_id, sample_rate) + pcm_to_bytes(pcm))


def encode_stream_id(ftype: FrameType, stream_id: int) -> bytes:
    if ftype not in (FrameType.FLUSH, FrameType.DRAINED, FrameType.END):
        raise ProtocolError(f"{ftype!r} does not carry a bare stream_id")
    return encode_frame(ftype, U32.pack(stream_id))


def encode_flush(stream_id: int = 0) -> bytes:
    return encode_stream_id(FrameType.FLUSH, stream_id)


def encode_drained(stream_id: int) -> bytes:
    return encode_stream_id(FrameType.DRAINED, stream_id)


def encode_end(stream_id: int) -> bytes:
    return encode_stream_id(FrameType.END, stream_id)


def encode_hello(info: dict[str, Any]) -> bytes:
    body = {"version": PROTOCOL_VERSION, **info}
    return encode_frame(FrameType.HELLO, json.dumps(body, separators=(",", ":")).encode())


def decode_stream_id(payload: bytes) -> int:
    if len(payload) != 4:
        raise ProtocolError(f"stream_id payload must be 4 bytes, got {len(payload)}")
    return U32.unpack(payload)[0]


def decode_play(payload: bytes) -> tuple[int, int, np.ndarray]:
    """-> (stream_id, sample_rate, float32 pcm)."""
    if len(payload) < PLAY_HEADER.size:
        raise ProtocolError("PLAY payload too short")
    sid, sr = PLAY_HEADER.unpack_from(payload)
    return sid, sr, bytes_to_pcm(payload[PLAY_HEADER.size :])


def decode_hello(payload: bytes) -> dict[str, Any]:
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"bad HELLO json: {e}") from e
    if not isinstance(obj, dict):
        raise ProtocolError("HELLO must be a JSON object")
    return obj


class FrameDecoder:
    """Incremental decoder: ``feed(bytes) -> list[Frame]``. Handles arbitrary splits."""

    def __init__(self, max_payload: int = MAX_PAYLOAD) -> None:
        self._buf = bytearray()
        self._max = max_payload

    def feed(self, data: bytes) -> list[Frame]:
        self._buf += data
        out: list[Frame] = []
        while len(self._buf) >= HEADER_SIZE:
            ftype, length = HEADER.unpack_from(self._buf)
            if length > self._max:
                raise ProtocolError(f"frame length {length} exceeds limit")
            try:
                ft = FrameType(ftype)
            except ValueError as e:
                raise ProtocolError(f"unknown frame type 0x{ftype:02x}") from e
            end = HEADER_SIZE + length
            if len(self._buf) < end:
                break
            out.append(Frame(ft, bytes(self._buf[HEADER_SIZE:end])))
            del self._buf[:end]
        return out

    @property
    def pending(self) -> int:
        return len(self._buf)
