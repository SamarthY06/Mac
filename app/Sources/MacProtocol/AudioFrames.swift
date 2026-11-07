// audio.sock wire format (ARCHITECTURE.md §2):  type:u8 | len:u32le | payload
import Foundation

public enum FrameType: UInt8, Sendable {
    case mic = 0x01      // S→P  int16le PCM, 16 kHz mono
    case play = 0x02     // P→S  stream_id:u32le, sample_rate:u32le, int16le PCM mono
    case flush = 0x03    // P→S  stream_id:u32le (0 = everything)
    case drained = 0x04  // S→P  stream_id:u32le
    case end = 0x05      // P→S  stream_id:u32le
    case hello = 0x06    // both JSON {version, input_device, aec}
}

public struct Hello: Codable, Equatable, Sendable {
    public static let protocolVersion = 1
    public var version: Int
    public var input_device: String
    public var aec: Bool
    public init(version: Int = Hello.protocolVersion, input_device: String, aec: Bool) {
        self.version = version
        self.input_device = input_device
        self.aec = aec
    }
}

/// A decoded audio-link message.
public enum AudioMessage: Equatable, Sendable {
    case mic(Data)
    case play(streamID: UInt32, sampleRate: UInt32, pcm: Data)
    case flush(streamID: UInt32)
    case drained(streamID: UInt32)
    case end(streamID: UInt32)
    case hello(Data)                      // raw JSON
    case unknown(type: UInt8, payload: Data)

    public var frameType: UInt8 {
        switch self {
        case .mic: return FrameType.mic.rawValue
        case .play: return FrameType.play.rawValue
        case .flush: return FrameType.flush.rawValue
        case .drained: return FrameType.drained.rawValue
        case .end: return FrameType.end.rawValue
        case .hello: return FrameType.hello.rawValue
        case .unknown(let t, _): return t
        }
    }

    public var payload: Data {
        switch self {
        case .mic(let d), .hello(let d): return d
        case .play(let sid, let sr, let pcm):
            var d = Data(capacity: 8 + pcm.count)
            d.appendLE(sid); d.appendLE(sr); d.append(pcm)
            return d
        case .flush(let s), .drained(let s), .end(let s):
            var d = Data(capacity: 4); d.appendLE(s); return d
        case .unknown(_, let p): return p
        }
    }

    /// Full wire encoding (header + payload).
    public func encoded() -> Data {
        let p = payload
        var d = Data(capacity: 5 + p.count)
        d.append(frameType)
        d.appendLE(UInt32(p.count))
        d.append(p)
        return d
    }

    public static func hello(_ h: Hello) -> AudioMessage {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        return .hello((try? enc.encode(h)) ?? Data("{}".utf8))
    }

    /// Parse one frame's payload. Returns nil if the payload is malformed for its type.
    public static func parse(type: UInt8, payload p: Data) -> AudioMessage? {
        guard let t = FrameType(rawValue: type) else { return .unknown(type: type, payload: p) }
        switch t {
        case .mic: return .mic(p)
        case .hello: return .hello(p)
        case .play:
            guard p.count >= 8 else { return nil }
            let pcm = p.subdata(in: (p.startIndex + 8)..<p.endIndex)
            return .play(streamID: p.u32le(at: 0), sampleRate: p.u32le(at: 4), pcm: pcm)
        case .flush, .drained, .end:
            guard p.count >= 4 else { return nil }
            let s = p.u32le(at: 0)
            switch t {
            case .flush: return .flush(streamID: s)
            case .drained: return .drained(streamID: s)
            default: return .end(streamID: s)
            }
        }
    }
}

public enum FrameDecoderError: Error, Equatable {
    case frameTooLarge(Int)
    case malformed(type: UInt8)
}

/// Incremental decoder: feed arbitrary byte chunks, get whole messages out.
public struct FrameDecoder: Sendable {
    public static let defaultMaxFrame = 4 * 1024 * 1024
    public let maxFrame: Int
    private var buf = Data()

    public init(maxFrame: Int = FrameDecoder.defaultMaxFrame) { self.maxFrame = maxFrame }

    public var bufferedBytes: Int { buf.count }

    /// Appends bytes and returns all complete messages. Throws on a protocol violation
    /// (the caller should drop the connection).
    public mutating func feed(_ chunk: Data) throws -> [AudioMessage] {
        buf.append(chunk)
        var out: [AudioMessage] = []
        var off = 0
        let n = buf.count
        while n - off >= 5 {
            let type = buf[buf.startIndex + off]
            let len = Int(buf.u32le(at: off + 1))
            if len > maxFrame { throw FrameDecoderError.frameTooLarge(len) }
            guard n - off - 5 >= len else { break }
            let s = buf.startIndex + off + 5
            let payload = buf.subdata(in: s..<(s + len))
            guard let msg = AudioMessage.parse(type: type, payload: payload) else {
                throw FrameDecoderError.malformed(type: type)
            }
            out.append(msg)
            off += 5 + len
        }
        if off > 0 { buf = buf.subdata(in: (buf.startIndex + off)..<buf.endIndex) }
        return out
    }

    public mutating func reset() { buf = Data() }
}

// MARK: - little-endian helpers

extension Data {
    mutating func appendLE(_ v: UInt32) {
        var le = v.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }

    func u32le(at offset: Int) -> UInt32 {
        let i = startIndex + offset
        return UInt32(self[i]) | UInt32(self[i + 1]) << 8 | UInt32(self[i + 2]) << 16 | UInt32(self[i + 3]) << 24
    }
}
