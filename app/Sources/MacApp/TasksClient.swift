import Combine
import Foundation
import MacProtocol

// MARK: - Paths (agentd side)

extension Paths {
    /// agentd's API socket and the app token macsvc writes next to it (Mac-private/docs/agentd-api.md §1, §3).
    static let agentSock = runDir.appendingPathComponent("agent.sock")
    static let appToken = runDir.appendingPathComponent("app-token")
}

// MARK: - Wire models (DESIGN.md section 3; TaskView = TaskRecord minus anchor/limit_conv/current.report
// plus next_run, schedule_text, where)

struct TaskRun: Identifiable, Equatable {
    var runId: String
    var n: Int
    var trigger: String
    var status: String       // queued | running | ok | error | stopped | interrupted | limit | skipped | missed
    var queuedAt: Double?
    var startedAt: Double?
    var endedAt: Double?
    var errorKind: String
    var report: String
    var note: String
    var resetAt: Double?
    var delivery: String

    var id: String { runId }

    init?(_ d: [String: Any]) {
        guard let runId = d["run_id"] as? String else { return nil }
        self.runId = runId
        n = JSONish.int(d["n"]) ?? 0
        trigger = d["trigger"] as? String ?? ""
        status = d["status"] as? String ?? ""
        queuedAt = JSONish.double(d["queued_at"])
        startedAt = JSONish.double(d["started_at"])
        endedAt = JSONish.double(d["ended_at"])
        errorKind = d["error_kind"] as? String ?? ""
        report = d["report"] as? String ?? ""
        note = d["note"] as? String ?? ""
        resetAt = JSONish.double(d["reset_at"])
        delivery = d["delivery"] as? String ?? "none"
    }

    /// The instant shown for this run in history: when it ended, else when it started or was queued.
    var when: Double? { endedAt ?? startedAt ?? queuedAt }
}

struct TaskLastRun: Equatable {
    var runId: String
    var n: Int
    var status: String
    var endedAt: Double?
    var summary: String

    init?(_ d: [String: Any]) {
        guard let runId = d["run_id"] as? String else { return nil }
        self.runId = runId
        n = JSONish.int(d["n"]) ?? 0
        status = d["status"] as? String ?? ""
        endedAt = JSONish.double(d["ended_at"])
        summary = d["summary"] as? String ?? ""
    }
}

struct TaskView: Identifiable, Equatable {
    var id: String
    var rev: Int
    var title: String
    var description: String
    var kind: String             // "task" | "reminder"
    var schedule: TaskSchedule
    var conv: String
    var reportTo: String
    var bound: Bool
    var createdBy: String
    var createdAt: Double
    var updatedAt: Double
    var state: String            // active | paused | queued | running | waiting_limit | done
    var notBefore: Double?
    var runs: Int
    var current: TaskRun?
    var lastRun: TaskLastRun?
    var pauseReason: String
    var model: String            // "" = default; "sonnet" | "opus" | "haiku" | a full model id
    var effort: String           // "" = default; low | medium | high | xhigh | max
    var nextRun: Double?
    var scheduleText: String
    var whereText: String

    init?(_ d: [String: Any]) {
        guard let id = d["id"] as? String else { return nil }
        self.id = id
        rev = JSONish.int(d["rev"]) ?? 0
        title = d["title"] as? String ?? ""
        description = d["description"] as? String ?? ""
        kind = d["kind"] as? String ?? "task"
        schedule = TaskSchedule(d["schedule"] as? [String: Any] ?? [:])
        conv = d["conv"] as? String ?? ""
        reportTo = d["report_to"] as? String ?? ""
        bound = d["bound"] as? Bool ?? false
        createdBy = d["created_by"] as? String ?? ""
        createdAt = JSONish.double(d["created_at"]) ?? 0
        updatedAt = JSONish.double(d["updated_at"]) ?? 0
        state = d["state"] as? String ?? "active"
        notBefore = JSONish.double(d["not_before"])
        runs = JSONish.int(d["runs"]) ?? 0
        current = (d["current"] as? [String: Any]).flatMap(TaskRun.init)
        lastRun = (d["last_run"] as? [String: Any]).flatMap(TaskLastRun.init)
        pauseReason = d["pause_reason"] as? String ?? ""
        model = d["model"] as? String ?? ""     // absent on an older backend: default
        effort = d["effort"] as? String ?? ""
        nextRun = JSONish.double(d["next_run"])
        // Whole days read as days, not minutes ("Every day", "Every 2 days"); everything else is the daemon's text.
        if case .every(let n) = schedule, n % 1440 == 0 {
            scheduleText = n == 1440 ? "Every day" : "Every \(n / 1440) days"
        } else {
            scheduleText = UX.noDash(d["schedule_text"] as? String ?? "")
        }
        whereText = UX.noDash(d["where"] as? String ?? "")
    }
}

/// The schedule as the API sends and takes it (DESIGN.md 4.1): exactly one form.
enum TaskSchedule: Equatable {
    case once(String)                 // "YYYY-MM-DDTHH:MM" local
    case weekly(at: [String], days: [String])   // days empty = every day
    case every(Int)
    case unknown

    init(_ d: [String: Any]) {
        if let o = d["once"] as? String { self = .once(o) }
        else if let at = d["at"] as? [String] { self = .weekly(at: at, days: d["days"] as? [String] ?? []) }
        else if let n = JSONish.int(d["every_min"]) { self = .every(n) }
        else { self = .unknown }
    }

    var json: [String: Any] {
        switch self {
        case .once(let s): return ["once": s]
        case .weekly(let at, let days): return days.isEmpty ? ["at": at] : ["at": at, "days": days]
        case .every(let n): return ["every_min": n]
        case .unknown: return [:]
        }
    }
}

/// `task_report` notice (DESIGN.md 7.2): one Mac notification per finished unbound run.
struct TaskReport {
    var taskId: String
    var runId: String
    var title: String
    var body: String
    var status: String
    var kind: String          // "task" | "reminder": a reminder rings, a report is a silent banner
}

/// A failed op: the daemon's code and plain words (already dash-free).
struct TaskError: Error, Equatable {
    var code: String
    var message: String

    static let notConnected = TaskError(code: "disconnected", message: "Mac is still starting. Try again in a moment.")
    var isConflict: Bool { code == "task_conflict" }
    var isGone: Bool { code == "task_not_found" }
}

/// JSONSerialization numbers arrive as NSNumber; these read them without caring about the exact box.
enum JSONish {
    static func int(_ v: Any?) -> Int? {
        if let i = v as? Int { return i }
        if let d = v as? Double, d.isFinite { return Int(d) }
        return nil
    }
    static func double(_ v: Any?) -> Double? {
        if let d = v as? Double { return d }
        if let i = v as? Int { return Double(i) }
        return nil
    }
}

// MARK: - Transport: one hello'd NDJSON connection to agent.sock

/// Request/response layer over `UnixSocketClient` for agentd's framing (api §1): integer ids,
/// `op`, `{ok, data}` / `{ok: false, code, error}`, and `{"ev": ...}` notices. Callbacks on main.
final class AgentLink: @unchecked Sendable {
    typealias Reply = @MainActor @Sendable (_ ok: Bool, _ data: [String: Any], _ code: String?, _ error: String?) -> Void

    private let sock: UnixSocketClient
    private var lines = LineDecoder()          // reader thread only
    private let lock = NSLock()
    private var replies: [Int: Reply] = [:]    // guarded by lock
    private var seq = 0                        // guarded by lock

    var onNotice: (@MainActor ([String: Any]) -> Void)?
    var onConnection: (@MainActor (Bool) -> Void)?

    init(path: String) {
        sock = UnixSocketClient(path: path, name: "agent")
        sock.onConnect = { [weak self] in
            guard let self else { return }
            self.lines = LineDecoder()
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
                    for r in pending { r(false, [:], "disconnected", "disconnected") }
                    self.onConnection?(false)
                }
            }
        }
        sock.onData = { [weak self] d in self?.received(d) }
    }

    var isConnected: Bool { sock.isConnected }
    func start() { sock.start() }
    func stop() { sock.stop() }
    func reconnect() { sock.reconnect() }

    /// `{"id": n, "op": op, ...args}`. Returns false (and replies "disconnected") when not connected.
    @discardableResult
    func send(_ op: String, _ args: [String: Any] = [:], reply: Reply? = nil) -> Bool {
        let id: Int = lock.withLock {
            seq += 1
            if let reply { replies[seq] = reply }
            return seq
        }
        var obj = args
        obj["id"] = id
        obj["op"] = op
        guard sock.isConnected,
              var data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]) else {
            let r = lock.withLock { replies.removeValue(forKey: id) }
            if let r { DispatchQueue.main.async { MainActor.assumeIsolated { r(false, [:], "disconnected", "disconnected") } } }
            return false
        }
        data.append(0x0A)
        sock.send(data)
        return true
    }

    private func received(_ chunk: Data) {
        for line in lines.feed(chunk) {
            guard let obj = (try? JSONSerialization.jsonObject(with: line)) as? [String: Any] else {
                Log.warn("agent: bad JSON line (\(line.count) bytes)", "tasks")
                continue
            }
            if obj["ev"] != nil {
                let box = UncheckedBox(obj)
                DispatchQueue.main.async { MainActor.assumeIsolated { self.onNotice?(box.value) } }
            } else if let id = JSONish.int(obj["id"]) {
                let r = lock.withLock { replies.removeValue(forKey: id) }
                guard let r else { continue }
                let ok = (obj["ok"] as? Bool) ?? false
                let box = UncheckedBox((obj["data"] as? [String: Any]) ?? [:])
                let code = obj["code"] as? String
                let err = obj["error"] as? String
                DispatchQueue.main.async { MainActor.assumeIsolated { r(ok, box.value, code, err) } }
            } else if let code = obj["code"] as? String {
                // Frame-level error (id null): bad_frame, slow_client. The server closes; we reconnect.
                Log.warn("agent: frame error \(code)", "tasks")
            }
        }
    }
}

// MARK: - TasksClient (DESIGN.md section 11)

/// The app's view of agentd's task table. One persistent connection with the app token; `hello`
/// after every (re)connect; `task_list` after hello and on every `tasks_changed` with a newer rev;
/// pessimistic edits that apply only the server's returned task.
@MainActor
final class TasksClient: ObservableObject {
    /// Must equal agentd's `API_VERSION` (schema.py); a bump there touches this line too.
    static let apiVersion = "1.3"
    static let restartMessage = "Mac needs a restart to update all its parts."

    @Published private(set) var tasks: [TaskView] = []
    @Published private(set) var rev = 0
    @Published private(set) var connected = false    // hello accepted on the current connection
    @Published private(set) var loaded = false       // at least one task_list applied
    @Published private(set) var loadError: String?
    @Published private(set) var timeZoneName = TimeZone.current.identifier

    /// `task_report` notices: main.swift turns them into Mac notifications (DESIGN.md 7.2).
    var onReport: ((TaskReport) -> Void)?

    private let link: AgentLink
    private let epoch = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    private var helloBackoff: Double = 1
    private var helloWork: DispatchWorkItem?
    private var listing = false
    private var listAgain = false

    init(path: String = Paths.agentSock.path) {
        link = AgentLink(path: path)
        link.onConnection = { [weak self] up in self?.connection(up) }
        link.onNotice = { [weak self] n in self?.notice(n) }
    }

    func start() { link.start() }
    func stop() { link.stop() }

    // MARK: connection

    private func connection(_ up: Bool) {
        connected = false
        helloWork?.cancel()
        guard up else { return }
        scheduleHello(after: helloBackoff > 1 ? helloBackoff : 0)
    }

    private func scheduleHello(after delay: Double) {
        let w = DispatchWorkItem { [weak self] in MainActor.assumeIsolated { self?.hello() } }
        helloWork = w
        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: w)
    }

    private func bumpBackoff() { helloBackoff = min(helloBackoff * 2, 30) }

    private func hello() {
        guard link.isConnected else { return }
        guard let token = Self.readToken() else {
            // macsvc writes run/app-token at its start and removes it at stop: wait and retry.
            Log.info("agent: no app token yet; retrying in \(Int(helloBackoff)) s", "tasks")
            bumpBackoff()
            scheduleHello(after: helloBackoff)
            return
        }
        let args: [String: Any] = ["api": Self.apiVersion, "token": token, "client": "mac-app/1",
                                   "client_epoch": epoch, "conversations": [String]()]
        link.send("hello", args) { [weak self] ok, _, code, err in
            guard let self else { return }
            if ok {
                self.helloBackoff = 1
                self.connected = true
                Log.info("agent: hello ok", "tasks")
                self.reload()
            } else {
                // unauthorized / version_mismatch close the connection; the socket reconnects and
                // we try again with a re-read token after a growing delay (1..30 s).
                Log.warn("agent: hello failed: \(code ?? "?") \(err ?? "")", "tasks")
                self.bumpBackoff()
                if code == "version_mismatch" { self.loadError = Self.restartMessage }
            }
        }
    }

    private static func readToken() -> String? {
        guard let s = try? String(contentsOf: Paths.appToken, encoding: .utf8) else { return nil }
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? nil : t
    }

    // MARK: notices

    private func notice(_ n: [String: Any]) {
        switch n["ev"] as? String {
        case "tasks_changed":
            let r = JSONish.int(n["rev"]) ?? 0
            if r > rev { reload() }
        case "task_report":
            let report = TaskReport(taskId: n["task_id"] as? String ?? "", runId: n["run_id"] as? String ?? "",
                                    title: UX.noDash(n["title"] as? String ?? "Mac"),
                                    body: UX.noDash(n["body"] as? String ?? ""),
                                    status: n["status"] as? String ?? "",
                                    kind: n["kind"] as? String ?? "task")
            onReport?(report)
        default:
            break   // conversation events are never sent to an app client that subscribed to nothing
        }
    }

    // MARK: list

    /// Full replace from `task_list` (never patched). Coalesces: a reload during a reload runs once more after it.
    func reload() {
        guard connected else { return }
        if listing { listAgain = true; return }
        listing = true
        link.send("task_list") { [weak self] ok, data, _, err in
            guard let self else { return }
            self.listing = false
            if ok {
                self.apply(list: data)
            } else {
                self.loadError = Self.friendly(err)
            }
            if self.listAgain { self.listAgain = false; self.reload() }
        }
    }

    private func apply(list data: [String: Any]) {
        let raw = (data["tasks"] as? [[String: Any]]) ?? []
        let list = raw.compactMap(TaskView.init)
        rev = JSONish.int(data["rev"]) ?? rev
        if let tz = data["tz"] as? String, !tz.isEmpty { timeZoneName = tz }
        if list != tasks { tasks = list }
        loaded = true
        loadError = nil
    }

    /// Replace one row with the server's returned task (pessimistic edits, DESIGN.md 11).
    private func apply(task data: [String: Any]) {
        guard let t = (data["task"] as? [String: Any]).flatMap(TaskView.init) else { return }
        if let i = tasks.firstIndex(where: { $0.id == t.id }) {
            if tasks[i] != t { tasks[i] = t }
        } else {
            tasks.append(t)
        }
    }

    private func remove(id: String) {
        tasks.removeAll { $0.id == id }
    }

    // MARK: ops (completion on main; conflicts reload the list and come back as the error)

    typealias Done = (Result<[String: Any], TaskError>) -> Void

    private func call(_ op: String, _ args: [String: Any], done: @escaping Done) {
        guard connected else { done(.failure(.notConnected)); return }
        let sent = link.send(op, args) { [weak self] ok, data, code, err in
            guard let self else { return }
            if ok {
                done(.success(data))
            } else {
                let e = TaskError(code: code ?? "error", message: Self.friendly(err))
                if e.isConflict || e.isGone { self.reload() }
                if e.isGone, let id = args["task_id"] as? String { self.remove(id: id) }
                done(.failure(e))
            }
        }
        if !sent { done(.failure(.notConnected)) }
    }

    /// `model` / `effort` are sent only when set ("" = the default, which a create leaves out).
    func create(title: String, description: String, kind: String = "task", schedule: TaskSchedule,
                model: String = "", effort: String = "", done: @escaping Done) {
        var args: [String: Any] = ["title": title, "description": description, "kind": kind, "schedule": schedule.json]
        if !model.isEmpty { args["model"] = model }
        if !effort.isEmpty { args["effort"] = effort }
        call("task_create", args) { [weak self] r in
            if case .success(let d) = r { self?.apply(task: d) }
            done(r)
        }
    }

    /// `task_update`: only the given fields; `rev` guards against edits made elsewhere. An empty
    /// `model` / `effort` clears that setting back to the default.
    func update(_ task: TaskView, title: String? = nil, description: String? = nil, schedule: TaskSchedule? = nil,
                paused: Bool? = nil, model: String? = nil, effort: String? = nil, done: @escaping Done) {
        var args: [String: Any] = ["task_id": task.id, "rev": task.rev]
        if let title { args["title"] = title }
        if let description { args["description"] = description }
        if let schedule { args["schedule"] = schedule.json }
        if let paused { args["paused"] = paused }
        if let model { args["model"] = model }
        if let effort { args["effort"] = effort }
        call("task_update", args) { [weak self] r in
            if case .success(let d) = r { self?.apply(task: d) }
            done(r)
        }
    }

    /// No `rev`: delete is delete, and a row changed elsewhere must not turn it into an edit conflict.
    func delete(_ task: TaskView, done: @escaping Done) {
        call("task_delete", ["task_id": task.id]) { [weak self] r in
            if case .success = r { self?.remove(id: task.id) }
            done(r)
        }
    }

    func run(_ task: TaskView, done: @escaping Done) { taskOp("task_run", task, done: done) }
    func continueRun(_ task: TaskView, done: @escaping Done) { taskOp("task_continue", task, done: done) }
    func stop(_ task: TaskView, done: @escaping Done) { taskOp("task_stop", task, done: done) }

    private func taskOp(_ op: String, _ task: TaskView, done: @escaping Done) {
        call(op, ["task_id": task.id]) { [weak self] r in
            if case .success(let d) = r { self?.apply(task: d) }
            done(r)
        }
    }

    func history(_ task: TaskView, limit: Int, done: @escaping (Result<[TaskRun], TaskError>) -> Void) {
        call("task_history", ["task_id": task.id, "limit": limit]) { r in
            switch r {
            case .success(let d): done(.success(((d["runs"] as? [[String: Any]]) ?? []).compactMap(TaskRun.init)))
            case .failure(let e): done(.failure(e))
            }
        }
    }

    /// Plain words for people: transport strings become "Mac is still starting", daemon text loses its dashes.
    static func friendly(_ err: String?) -> String {
        guard let e = err, !e.isEmpty else { return "Something went wrong. Try again." }
        let low = e.lowercased()
        if low == "disconnected" || low.contains("not running") { return TaskError.notConnected.message }
        return UX.noDash(e)
    }
}
