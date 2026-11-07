import XCTest
@testable import MacProtocol

final class ProtocolTests: XCTestCase {
    func testRoundTripAllTypes() throws {
        let msgs: [AudioMessage] = [
            .mic(Data(repeating: 7, count: 640)),
            .play(streamID: 42, sampleRate: 24000, pcm: Data([1, 0, 2, 0])),
            .flush(streamID: 0),
            .drained(streamID: 9),
            .end(streamID: 0xDEADBEEF),
            .hello(Hello(input_device: "MacBook Pro Microphone", aec: true)),
        ]
        var wire = Data()
        for m in msgs { wire.append(m.encoded()) }
        var dec = FrameDecoder()
        // Feed one byte at a time to exercise partial frames.
        var out: [AudioMessage] = []
        for b in wire { out += try dec.feed(Data([b])) }
        XCTAssertEqual(out, msgs)
        XCTAssertEqual(dec.bufferedBytes, 0)
    }

    func testExactHeaderLayout() {
        let d = AudioMessage.play(streamID: 1, sampleRate: 16000, pcm: Data([0xAA, 0xBB])).encoded()
        XCTAssertEqual([UInt8](d), [0x02, 10, 0, 0, 0, 1, 0, 0, 0, 0x80, 0x3E, 0, 0, 0xAA, 0xBB])
        XCTAssertEqual([UInt8](AudioMessage.drained(streamID: 5).encoded()), [0x04, 4, 0, 0, 0, 5, 0, 0, 0])
    }

    func testHelloJSON() throws {
        let m = AudioMessage.hello(Hello(input_device: "X", aec: false))
        guard case .hello(let json) = m else { return XCTFail() }
        let obj = try JSONSerialization.jsonObject(with: json) as! [String: Any]
        XCTAssertEqual(obj["version"] as? Int, 1)
        XCTAssertEqual(obj["input_device"] as? String, "X")
        XCTAssertEqual(obj["aec"] as? Bool, false)
    }

    func testOversizeFrameRejected() {
        var dec = FrameDecoder()
        var hdr = Data([0x02]); hdr.appendLE(UInt32(5 * 1024 * 1024))
        XCTAssertThrowsError(try dec.feed(hdr))
    }

    func testMalformedPlay() {
        var dec = FrameDecoder()
        var d = Data([0x02]); d.appendLE(3); d.append(contentsOf: [1, 2, 3])
        XCTAssertThrowsError(try dec.feed(d))
    }

    func testUnknownTypeTolerated() throws {
        var dec = FrameDecoder()
        var d = Data([0x7F]); d.appendLE(1); d.append(9)
        XCTAssertEqual(try dec.feed(d), [.unknown(type: 0x7F, payload: Data([9]))])
    }

    func testLineDecoder() {
        var ld = LineDecoder()
        XCTAssertEqual(ld.feed(Data("{\"a\":1}\n{\"b\"".utf8)), [Data("{\"a\":1}".utf8)])
        XCTAssertEqual(ld.feed(Data(":2}\r\n\n".utf8)), [Data("{\"b\":2}".utf8)])
    }
}
