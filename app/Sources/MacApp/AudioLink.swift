import Foundation
import MacProtocol

/// Receives messages from the daemon (PLAY/FLUSH/END/HELLO). Called on the audio queue.
protocol AudioLinkDelegate: AnyObject {
    func audioLinkDidConnect()
    func audioLinkDidDisconnect()
    func audioLink(received message: AudioMessage)
    /// Current HELLO to announce (input device / aec). Called on the reader thread.
    func audioLinkHello() -> Hello
}

/// Client of ~/.mac-agent/run/audio.sock (binary framing, ARCHITECTURE.md §2).
final class AudioLink: @unchecked Sendable {
    private let sock: UnixSocketClient
    private var decoder = FrameDecoder()      // reader thread only
    private let deliverQueue: DispatchQueue
    weak var delegate: AudioLinkDelegate?

    // Stats (guarded by lock)
    private let lock = NSLock()
    private var micFramesSent = 0

    init(path: String = Paths.audioSock.path, deliverQueue: DispatchQueue) {
        self.sock = UnixSocketClient(path: path, name: "audio", maxPendingDroppable: 50)
        self.deliverQueue = deliverQueue
        sock.onConnect = { [weak self] in self?.connected() }
        sock.onData = { [weak self] d in self?.received(d) }
        sock.onDisconnect = { [weak self] in
            guard let self else { return }
            self.decoder.reset()
            self.deliverQueue.async { self.delegate?.audioLinkDidDisconnect() }
        }
    }

    var isConnected: Bool { sock.isConnected }

    func start() { sock.start() }
    func stop() { sock.stop() }

    private func connected() {
        decoder.reset()
        if let hello = delegate?.audioLinkHello() {
            sock.send(AudioMessage.hello(hello).encoded())
            Log.info("audio: HELLO sent \(hello)", "link")
        }
        deliverQueue.async { [weak self] in self?.delegate?.audioLinkDidConnect() }
    }

    private func received(_ chunk: Data) {
        do {
            let msgs = try decoder.feed(chunk)
            guard !msgs.isEmpty else { return }
            deliverQueue.async { [weak self] in
                guard let d = self?.delegate else { return }
                for m in msgs { d.audioLink(received: m) }
            }
        } catch {
            Log.error("audio: protocol error \(error); dropping connection", "link")
            decoder.reset()
            sock.reconnect()
        }
    }

    // MARK: send

    func sendMic(_ pcm: Data) {
        guard sock.isConnected else { return }
        sock.send(AudioMessage.mic(pcm).encoded(), droppable: true)
        lock.withLock { micFramesSent += 1 }
    }

    func sendDrained(_ streamID: UInt32) {
        sock.send(AudioMessage.drained(streamID: streamID).encoded())
    }

    func sendHello(_ h: Hello) {
        guard sock.isConnected else { return }
        sock.send(AudioMessage.hello(h).encoded())
        Log.info("audio: HELLO sent \(h)", "link")
    }

    /// Returns and resets the number of MIC frames sent since the last call.
    func takeMicCount() -> (sent: Int, dropped: Int) {
        lock.withLock {
            defer { micFramesSent = 0 }
            return (micFramesSent, sock.droppedWrites)
        }
    }
}
