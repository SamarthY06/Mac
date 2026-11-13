import AppKit
import SwiftUI

// MARK: - Presentation helpers (DESIGN.md 11, UI-NOTES.md)

/// Capsule text, symbol and tint per task state. Text and symbol always travel together.
struct TaskStateStyle {
    var text: String
    var symbol: String
    var tint: Color

    static func of(_ state: String) -> TaskStateStyle {
        switch state {
        case "queued": return .init(text: "Starting soon", symbol: "clock", tint: .accentColor)
        case "running": return .init(text: "Running", symbol: "clock.arrow.2.circlepath", tint: .accentColor)
        case "waiting_limit": return .init(text: "Waiting for the limit", symbol: "hourglass", tint: .orange)
        case "paused": return .init(text: "Paused", symbol: "pause.circle", tint: .secondary)
        case "done": return .init(text: "Done", symbol: "checkmark.circle.fill", tint: .green)
        default: return .init(text: "Active", symbol: "calendar.badge.clock", tint: .accentColor)
        }
    }

    /// Result word for a run status (line 3 and history rows).
    static func resultWord(_ status: String) -> String {
        switch status {
        case "ok": return "done"
        case "error": return "failed"
        case "limit": return "at the limit"
        case "": return "unknown"
        default: return status   // stopped, interrupted, skipped, missed are already words
        }
    }
}

enum TaskDates {
    private static var cal: Calendar { Calendar.current }

    static func time(_ t: Double) -> String {
        Date(timeIntervalSince1970: t).formatted(date: .omitted, time: .shortened)
    }

    /// "in 25 minutes", "Today at 8:00 AM", "Tomorrow at 8:00 AM", "Fri at 8:00 AM", "14 Nov at 8:00 AM".
    static func upcoming(_ t: Double, now: Date = Date()) -> String {
        let d = Date(timeIntervalSince1970: t)
        let delta = d.timeIntervalSince(now)
        if delta < 60 { return "in a moment" }
        if delta < 3600 {
            let f = RelativeDateTimeFormatter()
            f.unitsStyle = .full
            f.dateTimeStyle = .named
            return f.localizedString(for: d, relativeTo: now)
        }
        return "\(dayWord(d, now: now)) at \(time(t))"
    }

    /// "Today 8:00 AM", "Yesterday 6:30 PM", "Mon 6:30 PM", "3 Nov 6:30 PM".
    static func past(_ t: Double, now: Date = Date()) -> String {
        let d = Date(timeIntervalSince1970: t)
        return "\(dayWord(d, now: now)) \(time(t))"
    }

    /// "Mon 10 Nov, 6:30 PM" (history rows).
    static func full(_ t: Double) -> String {
        let d = Date(timeIntervalSince1970: t)
        return "\(d.formatted(.dateTime.weekday(.abbreviated).day().month(.abbreviated))), \(time(t))"
    }

    private static func dayWord(_ d: Date, now: Date) -> String {
        if cal.isDateInToday(d) { return "Today" }
        if cal.isDateInTomorrow(d) { return "Tomorrow" }
        if cal.isDateInYesterday(d) { return "Yesterday" }
        let days = cal.dateComponents([.day], from: cal.startOfDay(for: now), to: cal.startOfDay(for: d)).day ?? 99
        if abs(days) <= 6 { return d.formatted(.dateTime.weekday(.abbreviated)) }
        return d.formatted(.dateTime.day().month(.abbreviated))
    }

    static let onceFormat: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = .autoupdatingCurrent   // a zone change while the app runs must not send a stale wall time
        f.dateFormat = "yyyy-MM-dd'T'HH:mm"
        return f
    }()
}

extension String {
    /// "Tomorrow at 3:00 PM" -> "tomorrow at 3:00 PM" (mid-sentence use).
    var lowercasedFirst: String { prefix(1).lowercased() + dropFirst() }
}

/// Sort: waiting, running, queued first, then active by next run, then paused, then done by last run (newest first).
func sortedTasks(_ tasks: [TaskView]) -> [TaskView] {
    func rank(_ s: String) -> Int {
        switch s {
        case "waiting_limit": return 0
        case "running": return 1
        case "queued": return 2
        case "active": return 3
        case "paused": return 4
        case "done": return 5
        default: return 6
        }
    }
    return tasks.sorted { a, b in
        let ra = rank(a.state), rb = rank(b.state)
        if ra != rb { return ra < rb }
        if a.state == "active", a.nextRun != b.nextRun { return (a.nextRun ?? .infinity) < (b.nextRun ?? .infinity) }
        if a.state == "done", a.lastRun?.endedAt != b.lastRun?.endedAt { return (a.lastRun?.endedAt ?? 0) > (b.lastRun?.endedAt ?? 0) }
        return a.id < b.id
    }
}

// MARK: - Section

struct EditTarget: Identifiable {
    var task: TaskView?
    var id: String { task?.id ?? "new" }
}

struct TasksSettings: View {
    @ObservedObject var client: TasksClient
    @State private var editing: EditTarget?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var tasks: [TaskView] { sortedTasks(client.tasks) }
    /// "None" only when connected, loaded and empty; "unknown" (not connected, not loaded) is "Loading tasks…".
    private var showEmpty: Bool { client.connected && client.loaded && client.tasks.isEmpty && client.loadError == nil }
    private var showLoading: Bool {
        guard client.loadError != TasksClient.restartMessage else { return false }
        return !client.loaded || (!client.connected && client.tasks.isEmpty)
    }

    var body: some View {
        HStack {
            SectionTitle(text: "Tasks")
            Spacer()
            Button("New Task…") { editing = EditTarget(task: nil) }
                .controlSize(.small)
                .disabled(!client.connected)
                .accessibilityHint("Opens a form for a new scheduled task")
        }
        Card {
            if showLoading {
                Text("Loading tasks…")
                    .font(.system(size: 12)).foregroundStyle(.secondary).padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else if let e = client.loadError, client.tasks.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Label(e == TasksClient.restartMessage ? e : "Couldn’t load tasks. Try again.",
                          systemImage: "exclamationmark.circle.fill")
                        .font(.system(size: 11)).foregroundStyle(.orange)
                    if e != TasksClient.restartMessage { Button("Try Again") { client.reload() }.controlSize(.small) }
                }
                .padding(12).frame(maxWidth: .infinity, alignment: .leading)
            } else if showEmpty {
                ContentUnavailableView {
                    Label("No tasks yet", systemImage: "calendar.badge.clock")
                } description: {
                    Text("Ask Mac to do something on a schedule, for example: “Hey Mac, every weekday at 8, find new jobs that fit my resume.” Or add one here.")
                } actions: {
                    Button("New Task…") { editing = EditTarget(task: nil) }
                }
                .padding(.vertical, 8)
            } else {
                ForEach(Array(tasks.enumerated()), id: \.element.id) { i, t in
                    if i > 0 { RowDivider() }
                    TaskRow(task: t, client: client, onEdit: { editing = EditTarget(task: $0) })
                        .transition(.opacity)
                }
            }
        }
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.18), value: client.tasks)
        .sheet(item: $editing) { target in
            TaskEditSheet(client: client, existing: target.task)
        }
        Footnote(text: "Tasks run on this Mac even while you’re away. Create them here, by voice, or in WhatsApp.")
    }
}

// MARK: - Row

enum RowAction: Equatable { case run, pause, resume, cont, stop, delete }

/// A sibling of `CardRow` (same paddings, badge and fonts) with two secondary lines, a status capsule,
/// a button cluster and the History disclosure.
struct TaskRow: View {
    var task: TaskView
    @ObservedObject var client: TasksClient
    var onEdit: (TaskView) -> Void

    @State private var busy: RowAction?
    @State private var error: String?
    @State private var confirmDelete = false
    @State private var historyOpen = false
    @State private var history: [TaskRun] = []
    @State private var historyLoaded = false
    @State private var historyLoading = false
    @State private var historyError: String?
    @State private var expandedRun: String?

    private var style: TaskStateStyle { TaskStateStyle.of(task.state) }
    private var isRunning: Bool { task.state == "running" }
    private var isQueued: Bool { task.state == "queued" }
    private var isWaiting: Bool { task.state == "waiting_limit" }
    private var isPaused: Bool { task.state == "paused" }
    private var isDone: Bool { task.state == "done" }
    private var canRunNow: Bool { !(isRunning || isQueued || isWaiting) }
    private var canPauseResume: Bool { !(isRunning || isWaiting || isDone) }
    private var canStop: Bool { isRunning || isQueued || isWaiting }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .top, spacing: 12) {
                IconBadge(symbol: style.symbol, tint: style.tint)
                VStack(alignment: .leading, spacing: 3) {
                    VStack(alignment: .leading, spacing: 3) {
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Text(task.title).font(.system(size: 13)).lineLimit(2)
                                .fixedSize(horizontal: false, vertical: true)
                            Spacer(minLength: 8)
                            StatusCapsule(style: style)
                        }
                        Text(line2).font(.system(size: 11)).foregroundStyle(.secondary).lineLimit(1)
                        line3
                    }
                    .contentShape(Rectangle())
                    .onTapGesture(count: 2) { onEdit(task) }   // the text block only: not the buttons, not History rows
                    HStack(spacing: 8) {
                        Spacer(minLength: 0)
                        buttons
                        menu
                    }
                    .padding(.top, 2)
                    if let error, !error.isEmpty {
                        Label(error, systemImage: "exclamationmark.circle.fill")
                            .font(.system(size: 11)).foregroundStyle(.orange)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    DisclosureGroup(isExpanded: $historyOpen) {
                        TaskHistoryView(runs: history, loading: historyLoading, error: historyError,
                                        expanded: $expandedRun, canShowMore: historyLoaded && history.count >= 5 && history.count < 20,
                                        showMore: { loadHistory(limit: 20) })
                            .padding(.top, 4)
                    } label: {
                        Text("History").font(.system(size: 11)).foregroundStyle(.secondary)
                    }
                    .onChange(of: historyOpen) { _, open in if open, !historyLoaded, !historyLoading { loadHistory(limit: 5) } }
                    .accessibilityLabel("History of \(task.title)")
                }
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 9)
        .contentShape(Rectangle())
        .contextMenu { menuItems }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(accessibilityText)
        .alert("Delete “\(task.title)”?", isPresented: $confirmDelete) {
            Button("Cancel", role: .cancel) {}.keyboardShortcut(.defaultAction)
            Button("Delete", role: .destructive) { perform(.delete) }
        } message: {
            Text(isRunning || isQueued
                 ? "It’s running now; deleting stops it. Its run history goes too."
                 : "Its run history goes too. This can’t be undone.")
        }
        .onChange(of: task.rev) { _, _ in
            // The server has a newer row: an old error no longer describes it, and an open history may be stale.
            error = nil
            if historyOpen { loadHistory(limit: history.count > 5 ? 20 : 5) }
        }
    }

    // MARK: lines

    private var isReminder: Bool { task.kind == "reminder" }

    /// "Model: Opus · Effort: High" when either is set; nothing when both are the default.
    private var modelText: String {
        [task.model.isEmpty ? "" : "Model: \(TaskModel.label(task.model))",
         task.effort.isEmpty ? "" : "Effort: \(TaskEffort.label(task.effort))"].filter { !$0.isEmpty }.joined(separator: " · ")
    }

    private var line2: String {
        [isReminder ? "Reminder" : "", task.scheduleText, modelText, task.whereText].filter { !$0.isEmpty }.joined(separator: " · ")
    }

    private var nextText: String {
        if isPaused { return "Next: paused" }
        if isDone { return "Next: none" }
        if let n = task.nextRun { return "Next: \(TaskDates.upcoming(n))" }
        return "Next: none"
    }

    private var lastText: String {
        guard let l = task.lastRun else { return "Last: never" }
        let word = TaskStateStyle.resultWord(l.status)
        var s = "Last: \(l.endedAt.map { TaskDates.past($0) } ?? "earlier"), \(word)"
        var first = UX.noDash(l.summary.split(separator: "\n").first.map(String.init) ?? "")
        // Never "failed failed: ...": a summary that starts with the result word (or the raw status) loses it.
        for prefix in [word, l.status] where !prefix.isEmpty && first.lowercased().hasPrefix(prefix.lowercased()) {
            first = String(first.dropFirst(prefix.count)).trimmingCharacters(in: CharacterSet(charactersIn: ": ,"))
        }
        if !first.isEmpty { s += " " + first }
        return s
    }

    private var lastFailed: Bool { task.lastRun?.status == "error" }

    @ViewBuilder private var line3: some View {
        if isWaiting {
            // "It resets today at 3:00 PM." / "It resets tomorrow at 3:00 PM." / "It resets in 25 minutes."
            let reset = task.current?.resetAt.map { "It resets \(TaskDates.upcoming($0).lowercasedFirst)." } ?? ""
            Text(["Waiting for the Claude usage limit.", reset].filter { !$0.isEmpty }.joined(separator: " "))
                .font(.system(size: 11)).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        } else if isRunning {
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini)
                Text(task.current?.startedAt.map { "Running since \(TaskDates.time($0))" } ?? "Running")
                    .font(.system(size: 11)).foregroundStyle(.secondary).monospacedDigit()
            }
        } else if isQueued {
            Text("Starting soon").font(.system(size: 11)).foregroundStyle(.secondary)
        } else {
            HStack(spacing: 4) {
                Text("\(nextText) · ").font(.system(size: 11)).foregroundStyle(.secondary).monospacedDigit().lineLimit(1)
                    .layoutPriority(1)
                if lastFailed {
                    Image(systemName: "exclamationmark.circle.fill").font(.system(size: 10)).foregroundStyle(.orange)
                        .accessibilityHidden(true)
                }
                Text(lastText).font(.system(size: 11)).foregroundStyle(lastFailed ? .orange : .secondary)
                    .monospacedDigit().lineLimit(1).truncationMode(.tail)
            }
        }
    }

    private var accessibilityText: String {
        var parts = [task.title, isReminder ? "reminder" : "", task.scheduleText.lowercased(), style.text.lowercased()]
        if isWaiting, let r = task.current?.resetAt { parts.append("resets \(TaskDates.upcoming(r).lowercasedFirst)") }
        else if isRunning, let s = task.current?.startedAt { parts.append("running since \(TaskDates.time(s))") }
        else if !isQueued {
            parts.append(nextText.replacingOccurrences(of: "Next: ", with: "next run ").lowercased())
            parts.append(lastText.replacingOccurrences(of: "Last: ", with: "last run "))
        }
        return parts.filter { !$0.isEmpty }.joined(separator: ", ")
    }

    // MARK: buttons

    @ViewBuilder private var buttons: some View {
        if isWaiting {
            actionButton("Continue", .cont, prominent: true, help: "Picks up where it stopped once the limit resets.")
            actionButton("Stop", .stop, help: "Drops this run. The task stays scheduled.")
        } else if isRunning || isQueued {
            actionButton("Stop", .stop)
        } else if isPaused {
            actionButton("Resume", .resume)
        } else {
            actionButton("Run Now", .run)
            if !isDone { actionButton("Pause", .pause) }
        }
    }

    private func label(for a: RowAction) -> String {
        switch a {
        case .run: return "Run \(task.title) now"
        case .pause: return "Pause \(task.title)"
        case .resume: return "Resume \(task.title)"
        case .cont: return "Continue \(task.title)"
        case .stop: return "Stop \(task.title)"
        case .delete: return "Delete \(task.title)"
        }
    }

    private func actionButton(_ title: String, _ action: RowAction, prominent: Bool = false, help: String? = nil) -> some View {
        Button {
            perform(action)
        } label: {
            if busy == action {
                ProgressView().controlSize(.small).frame(minWidth: 40)
            } else {
                Text(title)
            }
        }
        .controlSize(.small)
        .tint(prominent ? .accentColor : nil)
        .disabled(busy != nil && busy != action)
        .help(help ?? "")
        .accessibilityLabel(label(for: action))
    }

    private var menu: some View {
        Menu {
            menuItems
        } label: {
            Image(systemName: "ellipsis.circle")
        }
        .menuStyle(.button)
        .buttonStyle(.borderless)
        .menuIndicator(.hidden)
        .controlSize(.small)
        .fixedSize()
        .disabled(busy != nil)
        .accessibilityLabel("More actions for \(task.title)")
    }

    @ViewBuilder private var menuItems: some View {
        if canRunNow { Button { perform(.run) } label: { Label("Run Now", systemImage: "play.fill") } }
        if canPauseResume {
            if isPaused { Button { perform(.resume) } label: { Label("Resume", systemImage: "play.circle") } }
            else { Button { perform(.pause) } label: { Label("Pause", systemImage: "pause.circle") } }
        }
        if isWaiting { Button { perform(.cont) } label: { Label("Continue", systemImage: "forward.fill") } }
        if canStop { Button { perform(.stop) } label: { Label("Stop", systemImage: "stop.fill") } }
        Divider()
        Button { onEdit(task) } label: { Label("Edit…", systemImage: "pencil") }
        Button { historyOpen.toggle() } label: { Label("History", systemImage: "list.bullet.clipboard") }
        Divider()
        Button(role: .destructive) { confirmDelete = true } label: { Label("Delete…", systemImage: "trash") }
    }

    // MARK: ops

    private func perform(_ a: RowAction) {
        error = nil
        busy = a
        let done: TasksClient.Done = { r in
            busy = nil
            if case .failure(let e) = r { error = e.message }
        }
        switch a {
        case .run: client.run(task, done: done)
        case .pause: client.update(task, paused: true, done: done)
        case .resume: client.update(task, paused: false, done: done)
        case .cont: client.continueRun(task, done: done)
        case .stop: client.stop(task, done: done)
        case .delete: client.delete(task, done: done)
        }
    }

    private func loadHistory(limit: Int) {
        historyLoading = true
        historyError = nil
        client.history(task, limit: limit) { r in
            historyLoading = false
            switch r {
            case .success(let runs): history = runs; historyLoaded = true
            case .failure(let e): historyError = e.message
            }
        }
    }
}

/// 10 pt medium text plus a 9 pt symbol on a tinted capsule, like the gender tag in Voice.
struct StatusCapsule: View {
    var style: TaskStateStyle
    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: style.symbol).font(.system(size: 9, weight: .semibold)).accessibilityHidden(true)
            Text(style.text).font(.system(size: 10, weight: .medium))
        }
        .padding(.horizontal, 7).padding(.vertical, 3)
        .foregroundStyle(style.tint)
        .background(Capsule().fill(style.tint.opacity(0.12)))
        .fixedSize()
        .accessibilityLabel("Status: \(style.text)")
    }
}

// MARK: - History

struct TaskHistoryView: View {
    var runs: [TaskRun]
    var loading: Bool
    var error: String?
    @Binding var expanded: String?
    var canShowMore: Bool
    var showMore: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let error {
                Label(error, systemImage: "exclamationmark.circle.fill").font(.system(size: 11)).foregroundStyle(.orange)
            } else if runs.isEmpty {
                Text(loading ? "Loading history…" : "No runs yet.").font(.system(size: 11)).foregroundStyle(.secondary)
            }
            ForEach(runs) { r in
                runRow(r)
            }
            if canShowMore || loading {
                HStack {
                    Spacer()
                    if loading, !runs.isEmpty { ProgressView().controlSize(.small) }
                    else if canShowMore { Button("Show More", action: showMore).controlSize(.small) }
                }
            }
        }
    }

    private func runRow(_ r: TaskRun) -> some View {
        let failed = r.status == "error"
        let word = TaskStateStyle.resultWord(r.status)
        // A finished run shows its report; anything else (failed, stopped, skipped, missed) shows the reason in `note`.
        let text = UX.noDash(r.status == "ok" || r.note.isEmpty ? (r.report.isEmpty ? r.note : r.report) : r.note)
        let first = text.split(separator: "\n").first.map(String.init) ?? ""
        let open = expanded == r.runId
        return VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(r.when.map(TaskDates.full) ?? "").font(.system(size: 11)).foregroundStyle(.secondary)
                    .monospacedDigit().frame(width: 128, alignment: .leading)
                HStack(spacing: 3) {
                    if failed {
                        Image(systemName: "exclamationmark.circle.fill").font(.system(size: 10)).accessibilityHidden(true)
                    }
                    Text(word).font(.system(size: 11, weight: .medium))
                }
                .foregroundStyle(failed ? Color.orange : Color.primary)
                .frame(width: 84, alignment: .leading)
                Text(first).font(.system(size: 11)).foregroundStyle(.secondary).lineLimit(open ? nil : 1)
                    .fixedSize(horizontal: false, vertical: open)
                Spacer(minLength: 0)
                if !text.isEmpty {
                    Image(systemName: open ? "chevron.down" : "chevron.right").font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.secondary).accessibilityHidden(true)
                }
            }
            if open, !text.isEmpty {
                Text(text).font(.system(size: 11)).textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 6, style: .continuous).fill(Color.primary.opacity(0.05)))
            }
        }
        .contentShape(Rectangle())
        .onTapGesture { guard !text.isEmpty else { return }; expanded = open ? nil : r.runId }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(r.when.map(TaskDates.full) ?? "") \(word). \(first)")
        .accessibilityHint(text.isEmpty ? "" : (open ? "Hides the report" : "Shows the full report"))
        .accessibilityAddTraits(text.isEmpty ? [] : .isButton)
    }
}

// MARK: - Model and effort (owner requirement; "" = default on the wire)

enum TaskModel: String, CaseIterable, Identifiable {
    case `default` = "", sonnet, opus, haiku, custom
    var id: String { rawValue }
    var title: String {
        switch self {
        case .default: return "Default"
        case .sonnet: return "Sonnet"
        case .opus: return "Opus"
        case .haiku: return "Haiku"
        case .custom: return "Custom"
        }
    }

    /// The picker choice for a wire value: a known name, "" or anything else (a full model id).
    static func choice(for wire: String) -> TaskModel {
        TaskModel(rawValue: wire).flatMap { $0 == .custom ? nil : $0 } ?? (wire.isEmpty ? .default : .custom)
    }

    /// Row text: "Opus" for a known name, the id as typed otherwise.
    static func label(_ wire: String) -> String {
        let c = choice(for: wire)
        return c == .custom ? wire : c.title
    }

    /// A custom id, as agentd accepts it (`_MODEL` in tasks.py): "claude-opus-4-1", "opus[1m]", a gateway id.
    static func isValidId(_ s: String) -> Bool {
        s.range(of: "^[a-z0-9][a-z0-9._:\\[\\]\\-]{0,99}$", options: .regularExpression) != nil
    }
}

enum TaskEffort: String, CaseIterable, Identifiable {
    case `default` = "", low, medium, high, xhigh, max
    var id: String { rawValue }
    var title: String {
        switch self {
        case .default: return "Default"
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        case .xhigh: return "Extra high"
        case .max: return "Max"
        }
    }

    static func label(_ wire: String) -> String { TaskEffort(rawValue: wire)?.title ?? wire }
}

// MARK: - Edit sheet

/// The form's working copy. Each segment keeps its own values, so switching segments loses nothing.
struct TaskDraft: Equatable {
    enum Mode: String, CaseIterable, Identifiable {
        case once, weekly, every
        var id: String { rawValue }
        var title: String {
            switch self {
            case .once: return "Once"
            case .weekly: return "Weekly"
            case .every: return "Every"
            }
        }
    }

    static let dayCodes = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]   // Calendar weekday 1...7
    static let weekdays: Set<String> = ["mon", "tue", "wed", "thu", "fri"]
    static let maxTimes = 6

    var title = ""
    var description = ""
    var mode: Mode = .once
    var onceDate: Date
    var days: Set<String> = TaskDraft.weekdays
    var times: [Date]
    var everyMin = 30
    var modelChoice: TaskModel = .default
    var customModel = ""
    var effort: TaskEffort = .default

    init(from task: TaskView?) {
        let cal = Calendar.current
        let now = Date()
        let nextHour = cal.date(bySetting: .minute, value: 0, of: now).flatMap { cal.date(byAdding: .hour, value: 1, to: $0) } ?? now
        onceDate = nextHour
        times = [cal.date(bySettingHour: 8, minute: 0, second: 0, of: now) ?? now]
        guard let task else { return }
        title = task.title
        description = task.description
        modelChoice = TaskModel.choice(for: task.model)
        if modelChoice == .custom { customModel = task.model }
        effort = TaskEffort(rawValue: task.effort) ?? .default
        switch task.schedule {
        case .once(let s):
            mode = .once
            if let d = TaskDates.onceFormat.date(from: s) { onceDate = d }
        case .weekly(let at, let days):
            mode = .weekly
            self.days = days.isEmpty ? Set(TaskDraft.dayCodes) : Set(days)
            let parsed = at.compactMap { Self.time(fromHHMM: $0) }
            if !parsed.isEmpty { times = parsed }
        case .every(let n):
            mode = .every
            everyMin = n
        case .unknown:
            break
        }
    }

    static func time(fromHHMM s: String) -> Date? {
        let p = s.split(separator: ":").compactMap { Int($0) }
        guard p.count == 2 else { return nil }
        return Calendar.current.date(bySettingHour: p[0], minute: p[1], second: 0, of: Date())
    }

    static func hhmm(_ d: Date) -> String {
        let c = Calendar.current.dateComponents([.hour, .minute], from: d)
        return String(format: "%02d:%02d", c.hour ?? 0, c.minute ?? 0)
    }

    var trimmedTitle: String { title.trimmingCharacters(in: .whitespacesAndNewlines) }
    var trimmedDescription: String { description.trimmingCharacters(in: .whitespacesAndNewlines) }

    /// The schedule as it would be sent, or nil when the segment is not valid yet.
    var schedule: TaskSchedule? {
        switch mode {
        case .once:
            return onceDate > Date() ? .once(TaskDates.onceFormat.string(from: onceDate)) : nil
        case .weekly:
            let at = Array(Set(times.map(Self.hhmm))).sorted()
            guard !days.isEmpty, !at.isEmpty else { return nil }
            let d = days.count == 7 ? [] : Self.dayCodes.filter { days.contains($0) }
            return .weekly(at: at, days: d)
        case .every:
            return (15...10080).contains(everyMin) ? .every(everyMin) : nil
        }
    }

    /// Plain words under the editor; nil when the segment is fine.
    var scheduleProblem: String? {
        switch mode {
        case .once: return onceDate > Date() ? nil : "That time has passed."
        case .weekly:
            if days.isEmpty { return "Pick at least one day." }
            if times.isEmpty { return "Add at least one time." }
            return nil
        case .every:
            if everyMin < 15 { return "At least 15 minutes." }
            if everyMin > 10080 { return "At most 10080 minutes, that is one week." }
            return nil
        }
    }

    /// "that is every 2 hours" when the minutes divide evenly.
    var everyHint: String? {
        guard (15...10080).contains(everyMin) else { return nil }
        if everyMin % 1440 == 0 { let d = everyMin / 1440; return d == 1 ? "that is every day" : "that is every \(d) days" }
        if everyMin % 60 == 0 { let h = everyMin / 60; return h == 1 ? "that is every hour" : "that is every \(h) hours" }
        return nil
    }

    var trimmedCustomModel: String { customModel.trimmingCharacters(in: .whitespacesAndNewlines) }

    /// The wire value: "" for the default, the known name, or the custom id.
    var modelWire: String { modelChoice == .custom ? trimmedCustomModel : modelChoice.rawValue }

    var modelProblem: String? {
        guard modelChoice == .custom else { return nil }
        if trimmedCustomModel.isEmpty { return "Type a model id, or pick one from the list." }
        if !TaskModel.isValidId(trimmedCustomModel) {
            return "A model id uses lower-case letters, digits, dots and hyphens, with no spaces."
        }
        return nil
    }

    var isValid: Bool {
        (1...60).contains(trimmedTitle.count) && (1...4000).contains(trimmedDescription.count) && schedule != nil
            && modelProblem == nil
    }
}

struct TaskEditSheet: View {
    @ObservedObject var client: TasksClient
    var existing: TaskView?
    @Environment(\.dismiss) private var dismiss
    @State private var draft: TaskDraft
    @State private var saving = false
    @State private var error: String?

    init(client: TasksClient, existing: TaskView?) {
        self.client = client
        self.existing = existing
        _draft = State(initialValue: TaskDraft(from: existing))
    }

    private var isNew: Bool { existing == nil }
    /// The live row for the task being edited: after a `task_conflict` the list reloads, and the next
    /// Save must carry the new `rev` and diff against the new values, or it conflicts forever.
    private var base: TaskView? { existing.flatMap { e in client.tasks.first { $0.id == e.id } } ?? existing }
    /// The row vanished from the list while the sheet was open (deleted by voice or WhatsApp).
    private var gone: Bool { existing.map { e in client.loaded && !client.tasks.contains { $0.id == e.id } } ?? false }
    private var whereText: String { base?.whereText ?? "Notifies on this Mac" }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(isNew ? "New Task" : "Edit Task").font(.system(size: 17, weight: .semibold))
                .accessibilityAddTraits(.isHeader)
            ScrollView {
                form.padding(.horizontal, 20)
            }
            .padding(.horizontal, -20)
            .frame(maxHeight: 520)
            if let error {
                Label(error, systemImage: "exclamationmark.circle.fill").font(.system(size: 11)).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            } else if gone {
                Label("This task was deleted elsewhere. Your edit can’t be saved.", systemImage: "exclamationmark.circle.fill")
                    .font(.system(size: 11)).foregroundStyle(.orange)
            }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction).disabled(saving)
                Button {
                    save()
                } label: {
                    if saving { ProgressView().controlSize(.small).frame(minWidth: 40) } else { Text("Save") }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!draft.isValid || saving || gone || !client.connected)
                .accessibilityLabel(isNew ? "Save new task" : "Save changes")
            }
        }
        .padding(20)
        .frame(width: 440)
    }

    /// The scrolling part: fields, recurrence editor and the read-only Where row. Buttons stay pinned below.
    private var form: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle(text: "Task")
            Card {
                formRow("Title") {
                    VStack(alignment: .trailing, spacing: 2) {
                        TextField("A few words", text: $draft.title)
                            .textFieldStyle(.roundedBorder)
                            .accessibilityLabel("Title")
                        if draft.trimmedTitle.count > 50 {
                            Text("\(draft.trimmedTitle.count) of 60").font(.system(size: 10)).monospacedDigit()
                                .foregroundStyle(draft.trimmedTitle.count > 60 ? Color.orange : Color.secondary)
                        }
                    }
                }
                RowDivider()
                formRow("What to do") {
                    ZStack(alignment: .topLeading) {
                        TextEditor(text: $draft.description)
                            .font(.system(size: 13))
                            .scrollContentBackground(.hidden)
                            .frame(minHeight: 72)
                            .padding(4)
                            .background(RoundedRectangle(cornerRadius: 6, style: .continuous).fill(Color.primary.opacity(0.04)))
                            .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).strokeBorder(Color.primary.opacity(0.15)))
                            .accessibilityLabel("What to do")
                        if draft.description.isEmpty {
                            Text("Tell Mac what to do, in your own words.")
                                .font(.system(size: 13)).foregroundStyle(.secondary)
                                .padding(.horizontal, 9).padding(.vertical, 4)
                                .allowsHitTesting(false)
                                .accessibilityHidden(true)
                        }
                    }
                }
            }

            SectionTitle(text: "Repeat")
            Card {
                VStack(alignment: .leading, spacing: 10) {
                    Picker("Repeat", selection: $draft.mode) {
                        ForEach(TaskDraft.Mode.allCases) { m in Text(m.title).tag(m) }
                    }
                    .pickerStyle(.segmented).labelsHidden()
                    .accessibilityLabel("Repeat")
                    scheduleEditor
                    if let p = draft.scheduleProblem {
                        Label(p, systemImage: "exclamationmark.circle.fill").font(.system(size: 11)).foregroundStyle(.orange)
                    }
                    if base?.state == "done" {
                        Text("Saving a new time schedules it again.").font(.system(size: 11)).foregroundStyle(.secondary)
                    }
                }
                .padding(12)
                RowDivider()
                formRow("Where") {
                    Text(whereText).font(.system(size: 12)).foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .accessibilityLabel("Where: \(whereText). Fixed when the task is created.")
                }
            }

            if base?.kind != "reminder" {   // a reminder shows its text; it never runs Claude
                SectionTitle(text: "Claude")
                Card {
                    formRow("Model") {
                        VStack(alignment: .leading, spacing: 6) {
                            Picker("Model", selection: $draft.modelChoice) {
                                ForEach(TaskModel.allCases) { m in Text(m.title).tag(m) }
                            }
                            .labelsHidden().frame(width: 160)
                            .accessibilityLabel("Model")
                            if draft.modelChoice == .custom {
                                TextField("claude-opus-4-1", text: $draft.customModel)
                                    .textFieldStyle(.roundedBorder).frame(width: 260)
                                    .accessibilityLabel("Custom model id")
                                if let p = draft.modelProblem {
                                    Label(p, systemImage: "exclamationmark.circle.fill").font(.system(size: 11)).foregroundStyle(.orange)
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                            }
                        }
                    }
                    RowDivider()
                    formRow("Effort") {
                        Picker("Effort", selection: $draft.effort) {
                            ForEach(TaskEffort.allCases) { e in Text(e.title).tag(e) }
                        }
                        .labelsHidden().frame(width: 160)
                        .accessibilityLabel("Effort")
                    }
                }
            }
        }
    }

    private func formRow<Content: View>(_ label: String, @ViewBuilder content: () -> Content) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text(label).font(.system(size: 12)).foregroundStyle(.secondary)
                .frame(width: 80, alignment: .trailing).padding(.top, 4)
            content()
        }
        .padding(.horizontal, 12).padding(.vertical, 9)
    }

    @ViewBuilder private var scheduleEditor: some View {
        switch draft.mode {
        case .once:
            DatePicker("Date and time", selection: $draft.onceDate, in: Date()..., displayedComponents: [.date, .hourAndMinute])
                .datePickerStyle(.stepperField)
                .font(.system(size: 12))
        case .weekly:
            weeklyEditor
        case .every:
            HStack(spacing: 6) {
                Text("Every").font(.system(size: 12))
                TextField("", value: $draft.everyMin, format: .number)
                    .textFieldStyle(.roundedBorder).frame(width: 56).multilineTextAlignment(.trailing)
                    .accessibilityLabel("Every \(draft.everyMin) minutes")
                Stepper("", value: $draft.everyMin, in: 15...10080, step: 5).labelsHidden()
                    .accessibilityLabel("Minutes")
                Text("minutes").font(.system(size: 12))
                if let h = draft.everyHint { Text(h).font(.system(size: 11)).foregroundStyle(.secondary) }
            }
        }
    }

    private var weeklyEditor: some View {
        let cal = Calendar.current
        let order = (0..<7).map { (cal.firstWeekday - 1 + $0) % 7 }
        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("Days").font(.system(size: 12)).frame(width: 40, alignment: .leading)
                HStack(spacing: 2) {
                    ForEach(order, id: \.self) { i in
                        let code = TaskDraft.dayCodes[i]
                        Toggle(cal.veryShortWeekdaySymbols[i], isOn: Binding(
                            get: { draft.days.contains(code) },
                            set: { on in if on { draft.days.insert(code) } else { draft.days.remove(code) } }))
                            .toggleStyle(.button).controlSize(.small)
                            .frame(width: 26)
                            .accessibilityLabel(cal.weekdaySymbols[i])
                    }
                }
                Spacer(minLength: 4)
                Button("Weekdays") { draft.days = TaskDraft.weekdays }.buttonStyle(.borderless).controlSize(.small)
                Button("Every day") { draft.days = Set(TaskDraft.dayCodes) }.buttonStyle(.borderless).controlSize(.small)
            }
            HStack(alignment: .top, spacing: 8) {
                Text("Times").font(.system(size: 12)).frame(width: 40, alignment: .leading).padding(.top, 3)
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(draft.times.indices, id: \.self) { i in
                        HStack(spacing: 4) {
                            DatePicker("", selection: $draft.times[i], displayedComponents: .hourAndMinute)
                                .datePickerStyle(.stepperField).labelsHidden()
                                .accessibilityLabel("Time \(i + 1)")
                            if draft.times.count > 1 {
                                Button {
                                    draft.times.remove(at: i)
                                } label: {
                                    Image(systemName: "minus.circle")
                                }
                                .buttonStyle(.borderless)
                                .accessibilityLabel("Remove time \(i + 1)")
                            }
                        }
                    }
                    if draft.times.count < TaskDraft.maxTimes {
                        Button {
                            let last = draft.times.last ?? Date()
                            draft.times.append(Calendar.current.date(byAdding: .hour, value: 1, to: last) ?? last)
                        } label: {
                            Label("Add Time", systemImage: "plus.circle")
                        }
                        .buttonStyle(.borderless).controlSize(.small)
                    }
                }
            }
        }
    }

    private func save() {
        guard let schedule = draft.schedule else { return }
        error = nil
        saving = true
        let title = draft.trimmedTitle, description = draft.trimmedDescription
        let done: TasksClient.Done = { r in
            saving = false
            switch r {
            case .success: dismiss()
            case .failure(let e):
                if e.isGone { error = "This task was deleted elsewhere. Your edit was not saved." }
                else if e.isConflict { error = "This task changed elsewhere. Check your changes and save again." }
                else if e.code == "task_limit" { error = "You can have up to 50 tasks." }
                else { error = e.message }
            }
        }
        if let t = base {
            // Only what changed against the live row; its rev guards against edits made elsewhere (task_conflict).
            let newTitle = title == t.title ? nil : title
            let newDesc = description == t.description ? nil : description
            let newSched = schedule == t.schedule ? nil : schedule
            let newModel = draft.modelWire == t.model ? nil : draft.modelWire       // "" clears to the default
            let newEffort = draft.effort.rawValue == t.effort ? nil : draft.effort.rawValue
            if newTitle == nil, newDesc == nil, newSched == nil, newModel == nil, newEffort == nil { saving = false; dismiss(); return }
            client.update(t, title: newTitle, description: newDesc, schedule: newSched, model: newModel, effort: newEffort, done: done)
        } else {
            client.create(title: title, description: description, schedule: schedule,
                          model: draft.modelWire, effort: draft.effort.rawValue, done: done)
        }
    }
}
