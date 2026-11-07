import Foundation

/// Spawns and supervises the Python daemon `macd` as a CHILD process, so macOS TCC treats
/// Mac.app as the responsible process for its mic / automation / accessibility use.
///
/// - Command: `<daemonDir>/.venv/bin/python -m macd --audio swift`, where `daemonDir`
///   comes from `defaults read io.github.macagent.Mac daemonDir` (written by install.sh)
///   or env `MAC_DAEMON_DIR`.
/// - Restarts with exponential backoff 1 s → 30 s; backoff resets after 60 s of uptime.
/// - stdin is a pipe that is never written: if Mac.app dies (even SIGKILL), the daemon
///   sees EOF on stdin and can exit (env `MAC_PARENT_PIPE=stdin`).
/// - Exit code 3 = "macd is already running" (a dev `uv run macd` holds the lock): treated
///   as "external daemon present": 60 s backoff, no crash counting (N10).
/// - Quit: SIGTERM, then SIGKILL after 10 s (macd's own shutdown can take ~5 s) (N15).
/// - Disabled with env `MAC_SUPERVISE=0` (tests, or when running macd by hand).
@MainActor
final class Supervisor {
    enum Status: Equatable { case disabled, notConfigured(String), starting, running(pid: Int32), backoff(Double), external }

    /// macd's exit code when another instance holds ~/.mac-agent/run/macd.lock.
    static let alreadyRunningExit: Int32 = 3

    private(set) var status: Status = .starting { didSet { onStatus?(status) } }
    var onStatus: ((Status) -> Void)?

    private var process: Process?
    private var stdinPipe: Pipe?
    private var backoff: Double = 1
    private var startedAt = Date()
    private var restartWork: DispatchWorkItem?
    private var stopping = false
    private(set) var restarts = 0

    static let defaultsKey = "daemonDir"

    var enabled: Bool { ProcessInfo.processInfo.environment["MAC_SUPERVISE"] != "0" }

    var daemonDir: String? {
        if let e = ProcessInfo.processInfo.environment["MAC_DAEMON_DIR"], !e.isEmpty { return e }
        return UserDefaults.standard.string(forKey: Self.defaultsKey)
    }

    func start() {
        guard enabled else {
            status = .disabled
            Log.info("supervisor disabled (MAC_SUPERVISE=0)", "supervisor")
            return
        }
        stopping = false
        spawn()
    }

    private func spawn() {
        restartWork = nil
        guard !stopping else { return }
        guard let dir = daemonDir else {
            status = .notConfigured("daemonDir not set: run install.sh")
            Log.error("daemonDir not configured (defaults write io.github.macagent.Mac daemonDir <path>)", "supervisor")
            scheduleRestart()
            return
        }
        let python = URL(fileURLWithPath: dir).appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            status = .notConfigured("\(python.path) missing: run install.sh")
            Log.error("python not found at \(python.path)", "supervisor")
            scheduleRestart()
            return
        }

        let p = Process()
        p.executableURL = python
        p.arguments = ["-m", "macd", "--audio", "swift"]
        p.currentDirectoryURL = URL(fileURLWithPath: dir)
        var env = ProcessInfo.processInfo.environment
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let extra = ["\(home)/.local/bin", "/opt/homebrew/bin", "/usr/local/bin"]
        env["PATH"] = (extra + [env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"]).joined(separator: ":")
        env["PYTHONUNBUFFERED"] = "1"
        env["MAC_SUPERVISED"] = "1"
        env["MAC_PARENT_PIPE"] = "stdin"
        env["MAC_APP_PID"] = String(ProcessInfo.processInfo.processIdentifier)
        p.environment = env

        do {
            try FileManager.default.createDirectory(at: Paths.logDir, withIntermediateDirectories: true)
            if !FileManager.default.fileExists(atPath: Paths.daemonLog.path) {
                FileManager.default.createFile(atPath: Paths.daemonLog.path, contents: nil)
            }
            let out = try FileHandle(forWritingTo: Paths.daemonLog)
            try out.seekToEnd()
            let banner = "\n===== \(ISO8601DateFormatter().string(from: Date())) Mac.app starting macd (\(python.path) -m macd --audio swift) =====\n"
            try out.write(contentsOf: Data(banner.utf8))
            p.standardOutput = out
            p.standardError = out
            let inPipe = Pipe()
            p.standardInput = inPipe
            stdinPipe = inPipe
            p.terminationHandler = { [weak self] proc in
                let code = proc.terminationStatus
                let signalled = proc.terminationReason == .uncaughtSignal
                let reason = signalled ? "signal \(code)" : "exit \(code)"
                try? out.close()
                DispatchQueue.main.async {
                    MainActor.assumeIsolated { self?.exited(proc, reason: reason, external: !signalled && code == Self.alreadyRunningExit) }
                }
            }
            try p.run()
            process = p
            startedAt = Date()
            status = .running(pid: p.processIdentifier)
            Log.info("macd started pid=\(p.processIdentifier)", "supervisor")
        } catch {
            Log.error("failed to start macd: \(error)", "supervisor")
            scheduleRestart()
        }
    }

    private func exited(_ proc: Process, reason: String, external: Bool) {
        guard proc === process else { return }
        process = nil
        stdinPipe = nil
        let up = Date().timeIntervalSince(startedAt)
        Log.warn(String(format: "macd exited (%@) after %.1f s", reason, up), "supervisor")
        guard !stopping else { return }
        if external {
            // Someone else runs macd (e.g. `uv run macd` in a terminal). Not a crash: check
            // again in a minute, and keep the backoff untouched for real crashes.
            status = .external
            Log.info("another macd holds the lock; checking again in 60 s", "supervisor")
            scheduleRestart(after: 60, countAsCrash: false)
            return
        }
        if up >= 60 { backoff = 1 }
        scheduleRestart()
    }

    private func scheduleRestart(after fixed: Double? = nil, countAsCrash: Bool = true) {
        guard !stopping, restartWork == nil else { return }
        let delay = fixed ?? backoff
        if countAsCrash {
            backoff = min(backoff * 2, 30)
            restarts += 1
            status = status.isNotConfigured ? status : .backoff(delay)
        }
        Log.info("restarting macd in \(delay) s", "supervisor")
        let w = DispatchWorkItem { [weak self] in MainActor.assumeIsolated { self?.spawn() } }
        restartWork = w
        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: w)
    }

    /// Manual restart from the menu.
    func restartNow() {
        guard enabled else { return }
        backoff = 1
        if let p = process, p.isRunning {
            kill(p.processIdentifier, SIGTERM)  // terminationHandler schedules the restart
        } else {
            restartWork?.cancel(); restartWork = nil
            spawn()
        }
    }

    /// All live descendants of `pid` (children, grandchildren …), e.g. the `claude` CLI.
    nonisolated static func descendants(of pid: Int32) -> [Int32] {
        let ps = Process()
        ps.executableURL = URL(fileURLWithPath: "/bin/ps")
        ps.arguments = ["-axo", "pid=,ppid="]
        let out = Pipe(); ps.standardOutput = out
        guard (try? ps.run()) != nil else { return [] }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        ps.waitUntilExit()
        var children: [Int32: [Int32]] = [:]
        for line in String(decoding: data, as: UTF8.self).split(separator: "\n") {
            let f = line.split(separator: " ", omittingEmptySubsequences: true)
            if f.count == 2, let c = Int32(f[0]), let pp = Int32(f[1]) { children[pp, default: []].append(c) }
        }
        var out2: [Int32] = [], queue = [pid]
        while let p = queue.popLast() { for c in children[p] ?? [] { out2.append(c); queue.append(c) } }
        return out2
    }

    /// Blocking: used from applicationWillTerminate. SIGTERM, then SIGKILL after `timeout`;
    /// any descendants still alive afterwards (orphaned `claude` CLI, shells) are killed too.
    func stopSync(timeout: TimeInterval = 10) {
        stopping = true
        restartWork?.cancel(); restartWork = nil
        guard let p = process, p.isRunning else { return }
        let pid = p.processIdentifier
        let kids = Self.descendants(of: pid)
        Log.info("stopping macd pid=\(pid) (SIGTERM; \(kids.count) descendants)", "supervisor")
        kill(pid, SIGTERM)
        let deadline = Date().addingTimeInterval(timeout)
        while p.isRunning && Date() < deadline { usleep(50_000) }
        if p.isRunning {
            Log.warn("macd did not exit in \(timeout) s; SIGKILL", "supervisor")
            kill(pid, SIGKILL)
            p.waitUntilExit()
        }
        usleep(200_000)
        let orphans = kids.filter { kill($0, 0) == 0 }
        for o in orphans { kill(o, SIGKILL) }
        if !orphans.isEmpty { Log.warn("killed \(orphans.count) orphaned daemon descendants: \(orphans)", "supervisor") }
        Log.info("macd stopped", "supervisor")
        process = nil
    }
}

extension Supervisor.Status {
    var isNotConfigured: Bool { if case .notConfigured = self { return true }; return false }
    var label: String {
        switch self {
        case .disabled: return "Daemon supervision off"
        case .notConfigured(let why): return "Daemon not installed: \(why)"
        case .starting: return "Starting daemon…"
        case .running: return "Daemon running"
        case .backoff(let s): return String(format: "Daemon crashed: restarting in %.0f s", s)
        case .external: return "Using a macd started outside the app (e.g. uv run macd)"
        }
    }
}
