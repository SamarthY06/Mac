import Darwin
import Foundation

/// A reconnecting client for a unix-domain stream socket where the daemon is the server.
///
/// Threading: one dedicated reader thread (blocking connect/read loop) and one serial
/// write queue. The fd is only closed by the reader thread, after the write queue has
/// been drained, so a writer can never write to a recycled fd.
final class UnixSocketClient: @unchecked Sendable {
    let path: String
    let name: String
    var onConnect: (() -> Void)?
    var onData: ((Data) -> Void)?
    var onDisconnect: (() -> Void)?

    private let lock = NSLock()
    private var fd: Int32 = -1
    private var running = false
    private var thread: Thread?
    private let writeQueue: DispatchQueue
    private var pendingDroppable = 0
    private let maxPendingDroppable: Int
    private(set) var droppedWrites = 0

    init(path: String, name: String, maxPendingDroppable: Int = 100) {
        self.path = path
        self.name = name
        self.maxPendingDroppable = maxPendingDroppable
        self.writeQueue = DispatchQueue(label: "mac.sock.\(name).write", qos: .userInteractive)
    }

    var isConnected: Bool { lock.withLock { fd >= 0 } }

    func start() {
        lock.withLock {
            guard !running else { return }
            running = true
            let t = Thread { [weak self] in self?.runLoop() }
            t.name = "mac.sock.\(name)"
            t.qualityOfService = .userInteractive
            thread = t
            t.start()
        }
    }

    func stop() {
        lock.withLock {
            running = false
            if fd >= 0 { Darwin.shutdown(fd, SHUT_RDWR) }
        }
    }

    /// Force the current connection to drop (it will reconnect).
    func reconnect() {
        lock.withLock { if fd >= 0 { Darwin.shutdown(fd, SHUT_RDWR) } }
    }

    /// Queue bytes for sending. `droppable` writes (mic audio) are dropped instead of
    /// queued when the peer isn't keeping up or we're disconnected.
    func send(_ data: Data, droppable: Bool = false) {
        if droppable {
            let ok: Bool = lock.withLock {
                guard fd >= 0, pendingDroppable < maxPendingDroppable else { droppedWrites += 1; return false }
                pendingDroppable += 1
                return true
            }
            guard ok else { return }
        }
        writeQueue.async { [self] in
            if droppable { lock.withLock { pendingDroppable -= 1 } }
            let s = lock.withLock { fd }
            guard s >= 0 else { return }
            if !writeAll(s, data) {
                Log.warn("\(name): write failed (\(String(cString: strerror(errno)))); dropping connection", "sock")
                Darwin.shutdown(s, SHUT_RDWR)
            }
        }
    }

    private func writeAll(_ s: Int32, _ data: Data) -> Bool {
        data.withUnsafeBytes { raw -> Bool in
            guard var p = raw.baseAddress else { return true }
            var left = raw.count
            while left > 0 {
                let n = Darwin.write(s, p, left)
                if n < 0 {
                    if errno == EINTR { continue }
                    return false
                }
                left -= n
                p = p.advanced(by: n)
            }
            return true
        }
    }

    private func connectOnce() -> Int32 {
        let s = socket(AF_UNIX, SOCK_STREAM, 0)
        guard s >= 0 else { return -1 }
        var on: Int32 = 1
        setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &on, socklen_t(MemoryLayout<Int32>.size))
        // A stuck peer must not block the write queue forever.
        var tv = timeval(tv_sec: 2, tv_usec: 0)
        setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8)
        let cap = MemoryLayout.size(ofValue: addr.sun_path)
        guard bytes.count < cap else {
            Log.error("\(name): socket path too long (\(bytes.count) ≥ \(cap)): \(path)", "sock")
            close(s); return -1
        }
        withUnsafeMutableBytes(of: &addr.sun_path) { buf in
            buf.copyBytes(from: bytes)
            buf[bytes.count] = 0
        }
        addr.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
        let rc = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(s, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        if rc != 0 { close(s); return -1 }
        return s
    }

    private func runLoop() {
        var backoff = 0.2
        var loggedWaiting = false
        var buf = [UInt8](repeating: 0, count: 64 * 1024)
        while lock.withLock({ running }) {
            let s = connectOnce()
            if s < 0 {
                if !loggedWaiting { Log.info("\(name): waiting for \(path)", "sock"); loggedWaiting = true }
                Thread.sleep(forTimeInterval: backoff)
                backoff = min(backoff * 2, 3.0)
                continue
            }
            let keep = lock.withLock { () -> Bool in
                guard running else { return false }
                fd = s
                return true
            }
            if !keep { close(s); break }
            backoff = 0.2
            loggedWaiting = false
            Log.info("\(name): connected", "sock")
            onConnect?()
            while true {
                let n = buf.withUnsafeMutableBytes { read(s, $0.baseAddress, $0.count) }
                if n > 0 {
                    onData?(Data(buf[0..<n]))
                } else if n < 0 && errno == EINTR {
                    continue
                } else {
                    break
                }
            }
            lock.withLock { fd = -1 }
            writeQueue.sync {}  // make sure no writer still uses `s`
            close(s)
            Log.info("\(name): disconnected", "sock")
            onDisconnect?()
            Thread.sleep(forTimeInterval: 0.2)
        }
    }
}
