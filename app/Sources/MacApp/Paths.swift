import Foundation
import os

/// Filesystem layout shared with the daemon (daemon/src/macd/config.py).
/// `MAC_HOME` / `MAC_LOG_DIR` env vars override the defaults (used by tests).
enum Paths {
    static let env = ProcessInfo.processInfo.environment

    static let home: URL = {
        if let h = env["MAC_HOME"], !h.isEmpty { return URL(fileURLWithPath: (h as NSString).expandingTildeInPath) }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".mac-agent")
    }()
    static let runDir = home.appendingPathComponent("run")
    static let audioSock = runDir.appendingPathComponent("audio.sock")
    static let controlSock = runDir.appendingPathComponent("control.sock")
    static let configFile = home.appendingPathComponent("config.toml")
    static let logDir: URL = {
        if let d = env["MAC_LOG_DIR"], !d.isEmpty { return URL(fileURLWithPath: (d as NSString).expandingTildeInPath) }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/Mac")
    }()
    static let appLog = logDir.appendingPathComponent("app.log")
    static let daemonLog = logDir.appendingPathComponent("macd.stdout.log")

    static func flag(_ name: String) -> Bool {
        guard let v = env[name]?.lowercased() else { return false }
        return ["1", "true", "yes", "on"].contains(v)
    }
}

/// Tiny thread-safe file + unified logger (~/Library/Logs/Mac/app.log).
enum Log {
    private static let osLog = Logger(subsystem: "io.github.macagent.Mac", category: "app")
    private static let queue = DispatchQueue(label: "mac.log")
    nonisolated(unsafe) private static var handle: FileHandle?
    private static let fmt: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    static func info(_ msg: String, _ cat: String = "app") { write("INFO", cat, msg) }
    static func warn(_ msg: String, _ cat: String = "app") { write("WARN", cat, msg) }
    static func error(_ msg: String, _ cat: String = "app") { write("ERROR", cat, msg) }

    private static func write(_ level: String, _ cat: String, _ msg: String) {
        osLog.log("[\(cat, privacy: .public)] \(msg, privacy: .public)")
        let now = Date()
        queue.async {
            if handle == nil {
                try? FileManager.default.createDirectory(at: Paths.logDir, withIntermediateDirectories: true)
                if !FileManager.default.fileExists(atPath: Paths.appLog.path) {
                    FileManager.default.createFile(atPath: Paths.appLog.path, contents: nil)
                }
                handle = try? FileHandle(forWritingTo: Paths.appLog)
                _ = try? handle?.seekToEnd()
            }
            let line = "\(fmt.string(from: now)) \(level) [\(cat)] \(msg)\n"
            try? handle?.write(contentsOf: Data(line.utf8))
        }
    }
}
