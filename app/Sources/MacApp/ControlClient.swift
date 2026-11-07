import Foundation
import MacProtocol

/// Client of ~/.mac-agent/run/control.sock (NDJSON, ARCHITECTURE.md §11).
///
/// Events and replies are delivered on the main queue.
final class ControlClient: @unchecked Sendable {
    typealias Reply = @MainActor @Sendable (_ ok: Bool, _ data: [String: Any], _ error: String?) -> Void

    private let sock: UnixSocketClient
    private var lines = LineDecoder()          // reader thread only
    private let lock = NSLock()
    private var replies: [String: Reply] = [:] // guarded by lock
    private var seq = 0                         // guarded by lock

    var onEvent: (@MainActor ([String: Any]) -> Void)?
    var onConnection: (@MainActor (Bool) -> Void)?

    init(path: String = Paths.controlSock.path) {
        sock = UnixSocketClient(path: path, name: "control")
        sock.onConnect = { [weak self] in
            guard let self else { return }
            self.lines = LineDecoder()
            self.send("subscribe")
            DispatchQueue.main.async { MainActor.assumeIsolated { self.onConnection?(true) } }
        }
        sock.onDisconnect = { [weak self] in
            guard let self else { return }
            let pending = self.lock.withLock { () -> [Reply] in
                defer { self.replies.removeAll() }
                return Array(self.replies.values)
            }
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    for r in pending { r(false, [:], "disconnected") }
                    self.onConnection?(false)
                }
            }
        }
        sock.onData = { [weak self] d in self?.received(d) }
    }

    var isConnected: Bool { sock.isConnected }
    func start() { sock.start() }
    func stop() { sock.stop() }

    /// Send a command. `args` are merged into the top-level object
    /// (e.g. `send("mute", ["value": true])` → `{"id":"a3","cmd":"mute","value":true}`).
    @discardableResult
    func send(_ cmd: String, _ args: [String: Any] = [:], reply: Reply? = nil) -> Bool {
        let id: String = lock.withLock {
            seq += 1
            let id = "app-\(seq)"
            if let reply { replies[id] = reply }
            return id
        }
        var obj = args
        obj["id"] = id
        obj["cmd"] = cmd
        guard sock.isConnected,
              var data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]) else {
            let r = lock.withLock { replies.removeValue(forKey: id) }
            if let r { DispatchQueue.main.async { MainActor.assumeIsolated { r(false, [:], "Mac daemon is not running") } } }
            return false
        }
        data.append(0x0A)
        sock.send(data)
        return true
    }

    private func received(_ chunk: Data) {
        for line in lines.feed(chunk) {
            guard let obj = (try? JSONSerialization.jsonObject(with: line)) as? [String: Any] else {
                Log.warn("control: bad JSON line: \(String(decoding: line.prefix(200), as: UTF8.self))", "control")
                continue
            }
            if obj["event"] != nil {
                let box = UncheckedBox(obj)
                DispatchQueue.main.async { MainActor.assumeIsolated { self.onEvent?(box.value) } }
            } else if let id = obj["id"] as? String {
                let r = lock.withLock { replies.removeValue(forKey: id) }
                guard let r else { continue }
                let ok = (obj["ok"] as? Bool) ?? (obj["error"] == nil)
                let box = UncheckedBox((obj["data"] as? [String: Any]) ?? [:])
                let err = (obj["error"] as? String) ?? ((obj["error"] as? [String: Any])?["message"] as? String)
                DispatchQueue.main.async { MainActor.assumeIsolated { r(ok, box.value, err) } }
            }
        }
    }
}

/// Moves a non-Sendable JSON value across a queue hop (it's never mutated afterwards).
struct UncheckedBox<T>: @unchecked Sendable { let value: T; init(_ v: T) { value = v } }
