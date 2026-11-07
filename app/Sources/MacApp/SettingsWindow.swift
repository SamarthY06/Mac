import AppKit
import Combine
import SwiftUI
import UniformTypeIdentifiers

// MARK: - App-side preferences (things the daemon doesn't own)

enum AppPrefs {
    static let changed = Notification.Name("io.github.macagent.Mac.prefsChanged")
    static let avatarDir = Paths.home.appendingPathComponent("avatar")

    /// "auto" (only when plugged in, default) | "always" | "never" — Power.swift reads it.
    static var stayAwake: String {
        get { UserDefaults.standard.string(forKey: "stayAwake") ?? "auto" }
        set { UserDefaults.standard.set(newValue, forKey: "stayAwake"); NotificationCenter.default.post(name: changed, object: nil) }
    }

    /// "top" (default) or "bottom".
    static var hudPosition: String {
        get { UserDefaults.standard.string(forKey: "hudPosition") ?? "top" }
        set { UserDefaults.standard.set(newValue, forKey: "hudPosition"); NotificationCenter.default.post(name: changed, object: nil) }
    }

    /// ~/.mac-agent/avatar/portrait.<ext>, if the person chose a picture.
    static var avatarURL: URL? {
        (try? FileManager.default.contentsOfDirectory(at: avatarDir, includingPropertiesForKeys: nil))?
            .first { $0.deletingPathExtension().lastPathComponent == "portrait" }
    }

    static func avatarImage() -> NSImage? { avatarURL.flatMap { NSImage(contentsOf: $0) } }

    static func setAvatar(_ url: URL?) throws {
        try FileManager.default.createDirectory(at: avatarDir, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        if let old = avatarURL { try? FileManager.default.removeItem(at: old) }
        if let url {
            let ext = url.pathExtension.isEmpty ? "png" : url.pathExtension.lowercased()
            try FileManager.default.copyItem(at: url, to: avatarDir.appendingPathComponent("portrait.\(ext)"))
        }
        NotificationCenter.default.post(name: changed, object: nil)
    }
}

// MARK: - Store (daemon settings via settings_get / settings_set / voices_list)

struct Voice: Identifiable, Equatable {
    var id: String
    var name: String
    var language: String
    var langCode: String
    var gender: String
    var available: Bool
}

@MainActor
final class SettingsStore: ObservableObject {
    @Published var name = "Mac"
    @Published var wakePhrases: [String] = []
    @Published var userName = ""
    @Published var ttsVoice = ""
    @Published var ttsLang = ""
    @Published var ttsSpeed = 1.0
    @Published var idleMinutes = 10
    @Published var followupSeconds = 6.0
    @Published var resultsDir = ""
    @Published var chrome = true
    @Published var loadUserSources = true
    @Published var remoteEnabled = false
    @Published var sttLanguages: [String] = []
    @Published var voices: [Voice] = []
    @Published var errors: [String: String] = [:]
    @Published var loaded = false
    @Published var loadError: String?
    @Published var savedFlash = false
    @Published var previewing: String?

    /// True while `apply` writes daemon values into the fields (so bindings don't echo them back).
    private(set) var applying = false
    private let control: ControlClient
    private var pending: [String: DispatchWorkItem] = [:]
    private var flashWork: DispatchWorkItem?
    weak var model: AppModel?   // gets the live engine (tts.engine + clone_sample)
    private var cancellables: Set<AnyCancellable> = []

    init(control: ControlClient) { self.control = control }

    /// Reload whenever the connection comes (back) up — a load that raced a reconnect must not stick.
    func observe(_ model: AppModel) {
        self.model = model
        model.$connected.removeDuplicates().dropFirst().receive(on: RunLoop.main).sink { [weak self] up in
            MainActor.assumeIsolated { if up { self?.load() } }
        }.store(in: &cancellables)
    }

    func load() {
        loadError = nil
        let sent = control.send("settings_get") { [weak self] ok, data, err in
            guard let self else { return }
            if ok { self.apply(data); self.loaded = true; self.loadError = nil }
            else if self.model?.connected == true { self.loadError = Self.friendly(err) }
        }
        _ = sent   // not connected yet: the hero shows "Mac is starting…" and observe() reloads
        control.send("voices_list") { [weak self] ok, data, _ in
            guard ok, let self else { return }
            let raw = (data["voices"] as? [[String: Any]]) ?? (data["list"] as? [[String: Any]]) ?? []
            self.voices = raw.compactMap { v in
                guard let id = v["id"] as? String else { return nil }
                let code = (v["lang"] as? String ?? v["lang_code"] as? String ?? "").lowercased()
                return Voice(id: id, name: v["name"] as? String ?? id,
                             language: v["language"] as? String ?? Self.languageName(code),
                             langCode: code, gender: Self.genderName(v["gender"] as? String ?? ""),
                             available: v["available"] as? Bool ?? true)
            }
        }
    }

    /// Writes daemon values into the fields.
    func apply(_ d: [String: Any]) {
        applying = true
        defer { applying = false }
        if let v = d["name"] as? String { name = v }
        if let v = d["wake.phrases"] as? [String] { wakePhrases = v }
        if let v = d["user_name"] as? String { userName = v }
        if let v = d["tts.voice"] as? String { ttsVoice = v }
        if let v = d["tts.lang"] as? String { ttsLang = v }
        if let v = d["tts.speed"] as? Double { ttsSpeed = v } else if let v = d["tts.speed"] as? Int { ttsSpeed = Double(v) }
        if let v = d["brain.idle_shutdown_min"] as? Int { idleMinutes = v } else if let v = d["brain.idle_shutdown_min"] as? Double { idleMinutes = Int(v) }
        if let v = d["listen.followup_s"] as? Double { followupSeconds = v } else if let v = d["listen.followup_s"] as? Int { followupSeconds = Double(v) }
        if let v = d["results_dir"] as? String { resultsDir = v }
        if let v = d["brain.chrome"] as? Bool { chrome = v }
        if let v = d["brain.setting_sources"] as? [String] { loadUserSources = v.contains("user") }
        if let v = d["remote.enabled"] as? Bool { remoteEnabled = v }
        if let v = d["stt_languages"] as? [String] { sttLanguages = v }
        if let engine = d["tts.engine"] { model?.setVoiceEngine(engine, sample: d["clone_sample"]) }
    }

    /// "en-us" → "English (US)", "fr" → "French".
    static func languageName(_ code: String) -> String {
        guard !code.isEmpty else { return "Other" }
        let parts = code.replacingOccurrences(of: "_", with: "-").split(separator: "-").map(String.init)
        let lang = Locale.current.localizedString(forLanguageCode: parts[0])?.capitalized ?? parts[0].uppercased()
        if parts.count > 1 { return "\(lang) (\(parts[1].uppercased()))" }
        return lang
    }

    static func genderName(_ g: String) -> String {
        switch g.lowercased() {
        case "f", "female": return "Female"
        case "m", "male": return "Male"
        default: return g
        }
    }

    /// Immediate write (toggles, pickers, steppers).
    func set(_ changes: [String: Any]) {
        guard !applying else { return }
        let sent = control.send("settings_set", ["changes": changes]) { [weak self] ok, data, err in
            guard let self else { return }
            if ok {
                if let applied = data["applied"] as? [String: Any] { self.apply(applied) }
                let errs = (data["errors"] as? [String: String]) ?? [:]
                for k in changes.keys { self.errors[k] = errs[k] }
                for (k, v) in errs { self.errors[k] = v }
                if errs.isEmpty { self.flashSaved() }
            } else {
                for k in changes.keys { self.errors[k] = Self.friendly(err) }
            }
        }
        if !sent { for k in changes.keys { errors[k] = "Mac is still starting — try again in a moment." } }
    }

    /// Debounced write (text fields, sliders).
    func setDebounced(_ key: String, _ value: Any, delay: Double = 0.6) {
        guard !applying else { return }
        pending[key]?.cancel()
        let w = DispatchWorkItem { [weak self] in MainActor.assumeIsolated { self?.set([key: value]) } }
        pending[key] = w
        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: w)
    }

    func preview(_ voice: Voice) {
        previewing = voice.id
        control.send("voice_preview", ["voice": voice.id]) { [weak self] ok, _, err in
            if !ok { self?.errors["tts.voice"] = err ?? "Couldn’t play a preview." }
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) { MainActor.assumeIsolated { self?.previewing = nil } }
        }
    }

    /// Never show transport-level strings ("daemon", "disconnected") to people.
    static func friendly(_ err: String?) -> String {
        guard let e = err, !e.isEmpty else { return "Couldn’t save." }
        let low = e.lowercased()
        if low.contains("daemon") || low.contains("disconnected") || low.contains("not running") {
            return "Mac is still starting — try again in a moment."
        }
        return e
    }

    private func flashSaved() {
        savedFlash = true
        flashWork?.cancel()
        let w = DispatchWorkItem { [weak self] in MainActor.assumeIsolated { self?.savedFlash = false } }
        flashWork = w
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.8, execute: w)
    }

    /// Languages Mac can speak, from the voice list (unique, in list order).
    var languages: [(code: String, name: String)] {
        var seen: Set<String> = []
        return voices.compactMap { v in seen.insert(v.langCode).inserted ? (v.langCode, v.language) : nil }
    }

    var wakeHint: String {
        let n = name.trimmingCharacters(in: .whitespaces)
        return n.isEmpty ? "Say: “Hey Mac”" : "Say: “Hey \(n)”"
    }
}

// MARK: - Window

enum SettingsSection: String, CaseIterable, Identifiable, Hashable {
    case general, voice, appearance, connections, permissions, phone, advanced
    var id: String { rawValue }
    var title: String {
        switch self {
        case .general: return "General"
        case .voice: return "Voice"
        case .appearance: return "Appearance"
        case .connections: return "Connections"
        case .permissions: return "Permissions"
        case .phone: return "Phone"
        case .advanced: return "Advanced"
        }
    }
    var symbol: String {
        switch self {
        case .general: return "gearshape"
        case .voice: return "waveform.circle"
        case .appearance: return "person.crop.circle"
        case .connections: return "link.circle"
        case .permissions: return "lock.shield"
        case .phone: return "iphone.circle"
        case .advanced: return "gearshape.2"
        }
    }
    var filledSymbol: String { symbol + ".fill" }
}

/// Settings: 200 pt sidebar + content column with a compact hero, ~680×560 (docs/DESIGN.md "Settings").
@MainActor
final class SettingsWindow {
    struct Deps {
        var model: AppModel
        var control: ControlClient
        var perms: Permissions
        var phone: PhonePanel
        var openVoicePanel: () -> Void
        var resetConversation: () -> Void
        var voiceReset: () -> Void
    }

    private var window: NSWindow?
    private let deps: Deps
    private let store: SettingsStore
    private let nav = SettingsNav()
    private var refreshTimer: Timer?

    init(deps: Deps) {
        self.deps = deps
        self.store = SettingsStore(control: deps.control)
        self.store.observe(deps.model)
    }

    var isVisible: Bool { window?.isVisible ?? false }

    func show(_ section: SettingsSection? = nil) {
        if let section { nav.section = section }
        store.load()
        deps.perms.refresh()
        deps.model.guided?.check(deep: false)
        if window == nil {
            let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 680, height: 560),
                             styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                             backing: .buffered, defer: false)
            w.title = "Mac Settings"
            w.titlebarAppearsTransparent = true
            w.toolbarStyle = .unified
            w.isReleasedWhenClosed = false
            w.minSize = NSSize(width: 640, height: 480)
            w.setAccessibilityLabel("Mac Settings")
            let root = SettingsView(store: store, nav: nav, model: deps.model, perms: deps.perms, phone: deps.phone,
                                    openVoicePanel: deps.openVoicePanel, resetConversation: deps.resetConversation,
                                    voiceReset: deps.voiceReset)
            w.contentViewController = NSHostingController(rootView: root)
            w.setContentSize(NSSize(width: 680, height: 560))
            w.center()
            window = w
        }
        NSApp.activate(ignoringOtherApps: true)
        window?.makeKeyAndOrderFront(nil)
        deps.model.onboardingVisible = true
        refreshTimer?.invalidate()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] t in
            MainActor.assumeIsolated {
                guard let self, let w = self.window, w.isVisible else {
                    t.invalidate(); self?.deps.model.onboardingVisible = false; return
                }
                self.deps.perms.refresh()
            }
        }
    }
}

@MainActor
final class SettingsNav: ObservableObject {
    @Published var section: SettingsSection = .general
}

// MARK: - Shared card language (also used by Meet Mac)

/// Accent (or green when done) circle with an SF Symbol — the row glyph everywhere.
struct IconBadge: View {
    var symbol: String
    var tint: Color = .accentColor
    var size: CGFloat = 26
    var body: some View {
        ZStack {
            Circle().fill(tint.opacity(0.13)).frame(width: size, height: size)
            Image(systemName: symbol).font(.system(size: size * 0.46, weight: .semibold)).foregroundStyle(tint)
        }
        .accessibilityHidden(true)
    }
}

/// Rounded, softly tinted container; rows go inside, separated by `RowDivider`.
struct Card<Content: View>: View {
    @ViewBuilder var content: () -> Content
    var body: some View {
        VStack(spacing: 0) { content() }
            .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Color.primary.opacity(0.045)))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Color.primary.opacity(0.06)))
    }
}

struct RowDivider: View {
    var body: some View { Divider().padding(.leading, 50) }
}

/// icon · title / subtitle · trailing control. 13 pt body, 11 pt secondary.
struct CardRow<Trailing: View>: View {
    var symbol: String
    var tint: Color = .accentColor
    var title: String
    var subtitle: String? = nil
    var error: String? = nil
    @ViewBuilder var trailing: () -> Trailing

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 12) {
                IconBadge(symbol: symbol, tint: tint)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.system(size: 13))
                    if let subtitle, !subtitle.isEmpty {
                        Text(subtitle).font(.system(size: 11)).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 12)
                trailing()
            }
            if let error, !error.isEmpty {
                Label(error, systemImage: "exclamationmark.circle.fill")
                    .font(.system(size: 11)).foregroundStyle(.orange)
                    .padding(.leading, 38)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 9)
    }
}

struct SectionTitle: View {
    var text: String
    var body: some View {
        Text(text).font(.system(size: 13, weight: .semibold)).padding(.leading, 2)
            .accessibilityAddTraits(.isHeader)
    }
}

struct Footnote: View {
    var text: String
    var body: some View {
        Text(text).font(.system(size: 11)).foregroundStyle(.secondary).padding(.leading, 2)
            .fixedSize(horizontal: false, vertical: true)
    }
}

// MARK: - Views

struct SettingsView: View {
    @ObservedObject var store: SettingsStore
    @ObservedObject var nav: SettingsNav
    @ObservedObject var model: AppModel
    @ObservedObject var perms: Permissions
    let phone: PhonePanel
    var openVoicePanel: () -> Void
    var resetConversation: () -> Void
    var voiceReset: () -> Void

    var body: some View {
        NavigationSplitView {
            List(selection: $nav.section) {
                ForEach(SettingsSection.allCases) { s in
                    Label {
                        Text(s.title)
                    } icon: {
                        Image(systemName: nav.section == s ? s.filledSymbol : s.symbol)
                    }
                    .tag(s)
                }
            }
            .navigationSplitViewColumnWidth(min: 200, ideal: 200, max: 240)
            .listStyle(.sidebar)
        } detail: {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    SettingsHero(store: store, model: model)
                    detail
                }
                .padding(.horizontal, 24).padding(.top, 8).padding(.bottom, 24)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(nsColor: .windowBackgroundColor))
            .overlay(alignment: .topTrailing) {
                if store.savedFlash {
                    Label("Saved", systemImage: "checkmark.circle.fill")
                        .font(.footnote.weight(.medium)).foregroundStyle(.green)
                        .padding(.horizontal, 10).padding(.vertical, 5)
                        .background(Capsule().fill(Color.green.opacity(0.12)))
                        .padding(12)
                        .transition(.opacity)
                        .accessibilityLabel("Settings saved")
                }
            }
            .animation(.easeInOut(duration: 0.18), value: store.savedFlash)
            .navigationTitle(nav.section.title)
        }
        .frame(minWidth: 640, minHeight: 480)
    }

    @ViewBuilder private var detail: some View {
        switch nav.section {
        case .general: GeneralSettings(store: store)
        case .voice: VoiceSettings(store: store, model: model, openVoicePanel: openVoicePanel, voiceReset: voiceReset)
        case .appearance: AppearanceSettings()
        case .connections:
            Card { ConnectRows(model: model) }
            Footnote(text: "Everything stays on this Mac except your conversation with Claude.")
        case .permissions:
            Card { PermissionsChecklist(perms: perms, model: model, asCard: false, includeConnect: false) }
            Footnote(text: "Mac asks for each permission with a one-line why. Grants take effect right away.")
        case .phone: PhoneSettings(store: store, model: model, phone: phone)
        case .advanced: AdvancedSettings(store: store, resetConversation: resetConversation)
        }
    }
}

/// Compact identity header: portrait / app mark, Mac's name, live status with the state glyph.
struct SettingsHero: View {
    @ObservedObject var store: SettingsStore
    @ObservedObject var model: AppModel

    private var tint: Color {
        if model.off { return .secondary }
        if model.state == .error { return .orange }
        return .accentColor
    }

    var body: some View {
        HStack(spacing: 14) {
            if let portrait = AppPrefs.avatarImage() {
                Image(nsImage: portrait).resizable().interpolation(.high).scaledToFill()
                    .frame(width: 48, height: 48).clipShape(Circle())
                    .overlay(Circle().strokeBorder(Color.primary.opacity(0.1)))
                    .accessibilityHidden(true)
            } else {
                Image(nsImage: NSApp.applicationIconImage).resizable().interpolation(.high)
                    .frame(width: 48, height: 48)
                    .accessibilityHidden(true)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(store.name.isEmpty ? "Mac" : store.name).font(.system(size: 17, weight: .semibold))
                HStack(spacing: 6) {
                    Image(systemName: model.glyph).font(.system(size: 11, weight: .semibold)).foregroundStyle(tint)
                        .accessibilityHidden(true)
                    Text(model.statusLine).font(.system(size: 12)).foregroundStyle(.secondary)
                }
            }
            Spacer()
            if !model.connected {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Mac is starting…").font(.system(size: 11)).foregroundStyle(.secondary)
                }
                .accessibilityElement(children: .combine)
            }
        }
        .padding(.bottom, 2)
        .accessibilityElement(children: .combine)
    }
}

// MARK: General

struct GeneralSettings: View {
    @ObservedObject var store: SettingsStore
    @State private var launchAtLogin = UX.launchAtLogin
    @State private var loginError: String?
    @State private var stayAwake = AppPrefs.stayAwake

    var body: some View {
        SectionTitle(text: "Name")
        Card {
            CardRow(symbol: "person.wave.2", title: "Name", subtitle: store.wakeHint,
                    error: store.errors["name"] ?? store.errors["wake.phrases"]) {
                TextField("Mac", text: $store.name)
                    .textFieldStyle(.roundedBorder).frame(width: 170)
                    .onChange(of: store.name) { _, v in store.setDebounced("name", v.trimmingCharacters(in: .whitespaces)) }
                    .accessibilityLabel("Name")
                    .accessibilityHint("Changes the wake word to Hey, followed by this name")
            }
            RowDivider()
            CardRow(symbol: "person", title: "Your name", subtitle: "Mac uses it when it talks to you. Optional.",
                    error: store.errors["user_name"]) {
                TextField("", text: $store.userName)
                    .textFieldStyle(.roundedBorder).frame(width: 170)
                    .onChange(of: store.userName) { _, v in store.setDebounced("user_name", v) }
                    .accessibilityLabel("Your name")
            }
        }

        SectionTitle(text: "Language")
        Card {
            CardRow(symbol: "globe", title: "Mac speaks", error: store.errors["tts.lang"]) {
                Picker("", selection: Binding(get: { store.ttsLang }, set: { store.ttsLang = $0; store.set(["tts.lang": $0]) })) {
                    if store.languages.isEmpty { Text(store.ttsLang.isEmpty ? "English" : store.ttsLang).tag(store.ttsLang) }
                    ForEach(store.languages, id: \.code) { l in Text(l.name).tag(l.code) }
                }
                .labelsHidden().frame(width: 180)
                .accessibilityLabel("Language Mac speaks")
            }
            RowDivider()
            CardRow(symbol: "ear", title: "You speak") {
                Text("\(Self.spoken(store.sttLanguages)) — more languages soon").font(.system(size: 12)).foregroundStyle(.secondary)
            }
        }

        SectionTitle(text: "Behavior")
        Card {
            CardRow(symbol: "sunrise", title: "Launch at login", error: loginError) {
                Toggle("", isOn: $launchAtLogin).labelsHidden().toggleStyle(.switch)
                    .onChange(of: launchAtLogin) { _, on in
                        loginError = UX.setLaunchAtLogin(on)
                        if loginError != nil { launchAtLogin = UX.launchAtLogin }
                    }
                    .accessibilityLabel("Launch at login")
            }
            RowDivider()
            CardRow(symbol: "moon.zzz", title: "Idle shutdown", subtitle: "Claude goes to sleep after this long without a request.",
                    error: store.errors["brain.idle_shutdown_min"]) {
                Stepper(value: Binding(get: { store.idleMinutes }, set: { store.idleMinutes = $0; store.set(["brain.idle_shutdown_min": $0]) }),
                        in: 1...240) { Text("\(store.idleMinutes) min").font(.system(size: 12)).monospacedDigit() }
                    .accessibilityLabel("Idle shutdown, \(store.idleMinutes) minutes")
            }
            RowDivider()
            CardRow(symbol: "ellipsis.bubble", title: "Follow-up window", subtitle: "Keep talking without the wake word for this long after Mac answers.",
                    error: store.errors["listen.followup_s"]) {
                Stepper(value: Binding(get: { store.followupSeconds }, set: { store.followupSeconds = $0; store.set(["listen.followup_s": $0]) }),
                        in: 2...30, step: 1) { Text("\(Int(store.followupSeconds)) s").font(.system(size: 12)).monospacedDigit() }
                    .accessibilityLabel("Follow-up window, \(Int(store.followupSeconds)) seconds")
            }
            RowDivider()
            CardRow(symbol: "bolt", title: "Stay awake while listening",
                    subtitle: "Keeps the Mac from sleeping while Mac is listening (the display can still turn off). A closed MacBook lid still sleeps the Mac unless an external display and power are attached.") {
                Picker("", selection: $stayAwake) {
                    Text("Automatic — when plugged in").tag("auto")
                    Text("Always").tag("always")
                    Text("Never").tag("never")
                }
                .labelsHidden().frame(width: 200)
                .onChange(of: stayAwake) { _, v in AppPrefs.stayAwake = v }
                .accessibilityLabel("Stay awake while listening")
            }
        }
    }

    static func spoken(_ codes: [String]) -> String {
        let names = codes.map { Locale.current.localizedString(forLanguageCode: $0)?.capitalized ?? $0 }
        return names.isEmpty ? "English" : names.joined(separator: ", ")
    }
}

// MARK: Voice

struct VoiceSettings: View {
    @ObservedObject var store: SettingsStore
    @ObservedObject var model: AppModel
    var openVoicePanel: () -> Void
    var voiceReset: () -> Void
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var groups: [(language: String, voices: [Voice])] {
        var order: [String] = []
        var map: [String: [Voice]] = [:]
        for v in store.voices {
            if map[v.language] == nil { order.append(v.language) }
            map[v.language, default: []].append(v)
        }
        return order.map { ($0, map[$0]!) }
    }

    var body: some View {
        SectionTitle(text: "Your voice")
        Card {
            CardRow(symbol: "waveform.and.mic", title: "Now speaking with", subtitle: model.voiceEngineLabel) {
                HStack(spacing: 8) {
                    Button("Record…", action: openVoicePanel).accessibilityLabel("Record or choose a voice sample")
                    Button("Use Default", action: voiceReset).disabled(!model.voiceEngine.hasPrefix("clone"))
                }
            }
            RowDivider()
            CardRow(symbol: "gauge.with.needle", title: "Speed", error: store.errors["tts.speed"]) {
                HStack(spacing: 8) {
                    Slider(value: Binding(get: { store.ttsSpeed }, set: { store.ttsSpeed = $0; store.setDebounced("tts.speed", ($0 * 100).rounded() / 100, delay: 0.4) }),
                           in: 0.8...1.3, step: 0.05) { Text("Speed") }
                        .labelsHidden().frame(width: 160)
                        .accessibilityValue(String(format: "%.2f times", store.ttsSpeed))
                    Text(String(format: "%.2f×", store.ttsSpeed)).font(.system(size: 12)).monospacedDigit().frame(width: 42, alignment: .trailing)
                }
            }
        }
        Footnote(text: "A 15-second sample is enough to clone a voice. The sample never leaves this Mac.")

        if groups.isEmpty {
            SectionTitle(text: "Built-in voices")
            Card {
                Text(store.loaded ? "No voices reported by Mac." : "Loading voices…")
                    .font(.system(size: 12)).foregroundStyle(.secondary).padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        ForEach(groups, id: \.language) { g in
            SectionTitle(text: g.language)
            Card {
                ForEach(Array(g.voices.enumerated()), id: \.element.id) { i, v in
                    if i > 0 { Divider().padding(.leading, 40) }
                    voiceRow(v)
                }
            }
        }
        if let e = store.errors["tts.voice"] {
            Label(e, systemImage: "exclamationmark.circle.fill").font(.system(size: 11)).foregroundStyle(.orange)
        }
    }

    private func voiceRow(_ v: Voice) -> some View {
        let selected = store.ttsVoice == v.id
        return HStack(spacing: 10) {
            Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                .font(.system(size: 15))
                .foregroundStyle(selected ? Color.accentColor : Color.secondary.opacity(0.6))
                .frame(width: 18)
                .accessibilityHidden(true)
            Text(v.name).font(.system(size: 13))
            if !v.gender.isEmpty {
                Text(v.gender).font(.system(size: 10, weight: .medium))
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .foregroundStyle(.secondary)
                    .background(Capsule().fill(Color.primary.opacity(0.07)))
                    .accessibilityHidden(true)
            }
            if !v.available {
                Text("Not downloaded").font(.system(size: 11)).foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                store.preview(v)
            } label: {
                Image(systemName: store.previewing == v.id ? "speaker.wave.2.fill" : "play.fill")
                    .symbolEffect(.variableColor.iterative, options: .repeating, isActive: store.previewing == v.id && !reduceMotion)
            }
            .buttonStyle(.borderless)
            .disabled(!v.available)
            .accessibilityLabel("Preview \(v.name)")
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .contentShape(Rectangle())
        .onTapGesture {
            guard v.available else { return }
            store.ttsVoice = v.id
            store.set(["tts.voice": v.id])
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(v.name), \(v.gender)\(selected ? ", selected" : "")")
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

// MARK: Appearance

struct AppearanceSettings: View {
    @State private var portrait: NSImage? = AppPrefs.avatarImage()
    @State private var hudPosition = AppPrefs.hudPosition
    @State private var targeted = false
    @State private var error: String?

    var body: some View {
        SectionTitle(text: "Portrait")
        Card {
            HStack(alignment: .center, spacing: 16) {
                ZStack {
                    Circle().fill(Color.primary.opacity(0.06)).frame(width: 72, height: 72)
                    Circle().strokeBorder(targeted ? Color.accentColor : Color.primary.opacity(0.12),
                                          style: StrokeStyle(lineWidth: 2, dash: portrait == nil ? [5, 4] : []))
                        .frame(width: 72, height: 72)
                    if let portrait {
                        Image(nsImage: portrait).resizable().scaledToFill()
                            .frame(width: 68, height: 68).clipShape(Circle())
                    } else {
                        Image(systemName: "person.crop.circle.badge.plus")
                            .font(.system(size: 24, weight: .light)).foregroundStyle(.secondary)
                    }
                }
                .onDrop(of: [.fileURL], isTargeted: $targeted) { providers in
                    guard let p = providers.first else { return false }
                    p.loadDataRepresentation(forTypeIdentifier: UTType.fileURL.identifier) { data, _ in
                        guard let data, let url = URL(dataRepresentation: data, relativeTo: nil) else { return }
                        DispatchQueue.main.async { MainActor.assumeIsolated { use(url) } }
                    }
                    return true
                }
                .accessibilityLabel(portrait == nil ? "No portrait. Drop a photo here or choose one." : "Mac’s portrait")
                VStack(alignment: .leading, spacing: 5) {
                    Text("Mac’s portrait").font(.system(size: 13))
                    Text("Shown in the HUD, in Meet Mac and here. Drop a photo or choose one.")
                        .font(.system(size: 11)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    HStack {
                        Button("Choose Photo…", action: choose)
                        if portrait != nil { Button("Remove", action: { use(nil) }) }
                    }
                    if let error {
                        Label(error, systemImage: "exclamationmark.circle.fill").font(.system(size: 11)).foregroundStyle(.orange)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(14)
        }

        SectionTitle(text: "HUD")
        Card {
            CardRow(symbol: "rectangle.topthird.inset.filled", title: "Position",
                    subtitle: "The HUD shows what you said and what Mac is doing, on the screen your pointer is on.") {
                Picker("", selection: $hudPosition) {
                    Text("Top").tag("top")
                    Text("Bottom").tag("bottom")
                }
                .pickerStyle(.segmented).labelsHidden().frame(width: 150)
                .onChange(of: hudPosition) { _, v in AppPrefs.hudPosition = v }
                .accessibilityLabel("HUD position")
            }
        }
    }

    private func choose() {
        let p = NSOpenPanel()
        p.allowedContentTypes = [.image]
        p.allowsMultipleSelection = false
        p.message = "Choose a photo for Mac’s portrait."
        guard p.runModal() == .OK, let url = p.url else { return }
        use(url)
    }

    private func use(_ url: URL?) {
        do {
            if let url, NSImage(contentsOf: url) == nil { error = "That file isn’t an image."; return }
            try AppPrefs.setAvatar(url)
            portrait = AppPrefs.avatarImage()
            error = nil
        } catch {
            self.error = "Couldn’t save the picture: \(error.localizedDescription)"
        }
    }
}

// MARK: Phone

struct PhoneSettings: View {
    @ObservedObject var store: SettingsStore
    @ObservedObject var model: AppModel
    let phone: PhonePanel

    var body: some View {
        SectionTitle(text: "Phone")
        Card {
            CardRow(symbol: "iphone.radiowaves.left.and.right", title: "Allow talking to Mac from a phone",
                    subtitle: "Each link is private, works for 15 minutes, and asks the caller for a spoken PIN first.",
                    error: store.errors["remote.enabled"]) {
                Toggle("", isOn: Binding(get: { store.remoteEnabled }, set: { store.remoteEnabled = $0; store.set(["remote.enabled": $0]) }))
                    .labelsHidden().toggleStyle(.switch)
                    .accessibilityLabel("Allow talking to Mac from a phone")
            }
            if model.remote.active {
                RowDivider()
                CardRow(symbol: "phone.connection", title: "Session", subtitle: model.remote.banner) { EmptyView() }
            }
        }
        Card {
            PhoneView(state: phone.state, model: model, embedded: true,
                      enable: { phone.enable() }, retry: { phone.refresh() }, close: {})
                .padding(.horizontal, 12).padding(.vertical, 6)
                .frame(maxWidth: .infinity)
        }
        .onAppear { phone.refresh() }   // the daemon answers "disabled" when off → Turn On state
        .onChange(of: store.remoteEnabled) { _, on in if on { phone.refresh() } }
    }
}

// MARK: Advanced

struct AdvancedSettings: View {
    @ObservedObject var store: SettingsStore
    var resetConversation: () -> Void

    var body: some View {
        SectionTitle(text: "Claude")
        Card {
            CardRow(symbol: "globe", title: "Use Chrome for websites", error: store.errors["brain.chrome"]) {
                Toggle("", isOn: Binding(get: { store.chrome }, set: { store.chrome = $0; store.set(["brain.chrome": $0]) }))
                    .labelsHidden().toggleStyle(.switch).accessibilityLabel("Use Chrome for websites")
            }
            RowDivider()
            CardRow(symbol: "puzzlepiece.extension", title: "Load my Claude MCP servers and skills",
                    subtitle: "Everything you set up for Claude Code (~/.claude) becomes available to Mac, including tools that can act on your behalf.",
                    error: store.errors["brain.setting_sources"]) {
                Toggle("", isOn: Binding(get: { store.loadUserSources },
                                         set: { store.loadUserSources = $0; store.set(["brain.setting_sources": $0 ? ["user"] : []]) }))
                    .labelsHidden().toggleStyle(.switch).accessibilityLabel("Load my Claude MCP servers and skills")
            }
        }

        SectionTitle(text: "Files")
        Card {
            CardRow(symbol: "folder", title: "Results folder",
                    subtitle: store.resultsDir.isEmpty ? "~/Documents/Mac" : store.resultsDir,
                    error: store.errors["results_dir"]) {
                Button("Choose…", action: chooseFolder)
            }
            RowDivider()
            CardRow(symbol: "doc.text", title: "Config file and logs", subtitle: "For the curious, or when something’s off.") {
                HStack(spacing: 8) {
                    Button("Config File") {
                        if FileManager.default.fileExists(atPath: Paths.configFile.path) {
                            NSWorkspace.shared.open(Paths.configFile)
                        } else {
                            NSWorkspace.shared.open(Paths.home)
                        }
                    }
                    Button("Logs") { NSWorkspace.shared.open(Paths.logDir) }
                }
            }
        }

        SectionTitle(text: "Conversation")
        Card {
            CardRow(symbol: "arrow.counterclockwise", title: "Reset conversation",
                    subtitle: "Mac forgets the current conversation and starts fresh. Memory notes are kept.") {
                Button("Reset", action: resetConversation)
            }
        }
    }

    private func chooseFolder() {
        let p = NSOpenPanel()
        p.canChooseDirectories = true
        p.canChooseFiles = false
        p.canCreateDirectories = true
        p.message = "Where Mac saves files it makes for you."
        guard p.runModal() == .OK, let url = p.url else { return }
        store.resultsDir = url.path
        store.set(["results_dir": url.path])
    }
}
