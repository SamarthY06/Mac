import AppKit
import SwiftUI
import UniformTypeIdentifiers

// MARK: - App-side preferences (things the daemon doesn't own)

enum AppPrefs {
    static let changed = Notification.Name("io.github.macagent.Mac.prefsChanged")
    static let avatarDir = Paths.home.appendingPathComponent("avatar")

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

    init(control: ControlClient) { self.control = control }

    func load() {
        loadError = nil
        let sent = control.send("settings_get") { [weak self] ok, data, err in
            guard let self else { return }
            if ok { self.apply(data); self.loaded = true } else { self.loadError = err ?? "Couldn’t read settings." }
        }
        if !sent { loadError = "Mac isn’t connected yet. Settings will load when it is." }
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
                for k in changes.keys { self.errors[k] = err ?? "Couldn’t save." }
            }
        }
        if !sent { for k in changes.keys { errors[k] = "Mac isn’t connected yet." } }
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
    case general, voice, appearance, permissions, phone, advanced
    var id: String { rawValue }
    var title: String {
        switch self {
        case .general: return "General"
        case .voice: return "Voice"
        case .appearance: return "Appearance"
        case .permissions: return "Permissions"
        case .phone: return "Phone"
        case .advanced: return "Advanced"
        }
    }
    var symbol: String {
        switch self {
        case .general: return "gearshape"
        case .voice: return "waveform"
        case .appearance: return "person.crop.circle"
        case .permissions: return "lock.shield"
        case .phone: return "iphone"
        case .advanced: return "slider.horizontal.3"
        }
    }
}

/// Settings: sidebar + detail, ~640×520 (HIG Settings-window style).
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
        self.store.model = deps.model
    }

    var isVisible: Bool { window?.isVisible ?? false }

    func show(_ section: SettingsSection? = nil) {
        if let section { nav.section = section }
        store.load()
        deps.perms.refresh()
        if window == nil {
            let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 640, height: 520),
                             styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                             backing: .buffered, defer: false)
            w.title = "Mac Settings"
            w.titlebarAppearsTransparent = true
            w.toolbarStyle = .unified
            w.isReleasedWhenClosed = false
            w.minSize = NSSize(width: 600, height: 460)
            w.setAccessibilityLabel("Mac Settings")
            let root = SettingsView(store: store, nav: nav, model: deps.model, perms: deps.perms, phone: deps.phone,
                                    openVoicePanel: deps.openVoicePanel, resetConversation: deps.resetConversation,
                                    voiceReset: deps.voiceReset)
            w.contentViewController = NSHostingController(rootView: root)
            w.setContentSize(NSSize(width: 640, height: 520))
            w.center()
            window = w
        }
        NSApp.activate(ignoringOtherApps: true)
        window?.makeKeyAndOrderFront(nil)
        // Permission grants happen in System Settings; poll while open (and let the daemon poll faster).
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
            List(SettingsSection.allCases, selection: $nav.section) { s in
                Label(s.title, systemImage: s.symbol).tag(s)
            }
            .navigationSplitViewColumnWidth(min: 160, ideal: 170, max: 200)
            .listStyle(.sidebar)
        } detail: {
            ZStack(alignment: .topTrailing) {
                detail
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
        .frame(minWidth: 600, minHeight: 460)
    }

    @ViewBuilder private var detail: some View {
        switch nav.section {
        case .general: GeneralSettings(store: store)
        case .voice: VoiceSettings(store: store, model: model, openVoicePanel: openVoicePanel, voiceReset: voiceReset)
        case .appearance: AppearanceSettings()
        case .permissions:
            Form {
                Section {
                    PermissionsChecklist(perms: perms, model: model, asCard: false)
                } header: {
                    Text("Mac asks for each permission with a one-line why. Grants take effect right away.")
                        .font(.callout).foregroundStyle(.secondary)
                }
            }
            .formStyle(.grouped)
        case .phone: PhoneSettings(store: store, model: model, phone: phone)
        case .advanced: AdvancedSettings(store: store, resetConversation: resetConversation)
        }
    }
}

/// Inline error under a control.
private struct FieldError: View {
    var text: String?
    var body: some View {
        if let text, !text.isEmpty {
            Label(text, systemImage: "exclamationmark.circle.fill")
                .font(.footnote).foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

private struct ConnectionNote: View {
    @ObservedObject var store: SettingsStore
    var body: some View {
        if let e = store.loadError {
            Label(e, systemImage: "bolt.slash").font(.callout).foregroundStyle(.secondary)
        }
    }
}

// MARK: General

struct GeneralSettings: View {
    @ObservedObject var store: SettingsStore
    @State private var launchAtLogin = UX.launchAtLogin
    @State private var loginError: String?

    var body: some View {
        Form {
            ConnectionNote(store: store)
            Section {
                TextField("Name", text: $store.name, prompt: Text("Mac"))
                    .onChange(of: store.name) { _, v in
                        let n = v.trimmingCharacters(in: .whitespaces)
                        store.setDebounced("name", n)
                    }
                    .accessibilityHint("Changes the wake word to Hey, followed by this name")
                Text(store.wakeHint).font(.callout).foregroundStyle(.secondary)
                FieldError(text: store.errors["name"] ?? store.errors["wake.phrases"])
                TextField("Your name", text: $store.userName, prompt: Text("Optional"))
                    .onChange(of: store.userName) { _, v in store.setDebounced("user_name", v) }
                FieldError(text: store.errors["user_name"])
            } header: {
                Text("Names")
            } footer: {
                Text("Mac uses your name when it talks to you.").font(.footnote).foregroundStyle(.secondary)
            }

            Section("Language") {
                Picker("Mac speaks", selection: Binding(
                    get: { store.ttsLang },
                    set: { store.ttsLang = $0; store.set(["tts.lang": $0]) })
                ) {
                    if store.languages.isEmpty { Text(store.ttsLang.isEmpty ? "English" : store.ttsLang).tag(store.ttsLang) }
                    ForEach(store.languages, id: \.code) { l in Text(l.name).tag(l.code) }
                }
                FieldError(text: store.errors["tts.lang"])
                LabeledContent("You speak") {
                    Text("\(Self.spoken(store.sttLanguages)) — more languages soon").foregroundStyle(.secondary)
                }
            }

            Section("Behavior") {
                Toggle("Launch at login", isOn: $launchAtLogin)
                    .onChange(of: launchAtLogin) { _, on in
                        loginError = UX.setLaunchAtLogin(on)
                        if loginError != nil { launchAtLogin = UX.launchAtLogin }
                    }
                FieldError(text: loginError)
                Stepper(value: Binding(get: { store.idleMinutes }, set: { store.idleMinutes = $0; store.set(["brain.idle_shutdown_min": $0]) }),
                        in: 1...240, step: 1) {
                    LabeledContent("Idle shutdown", value: "\(store.idleMinutes) min")
                }
                .accessibilityLabel("Idle shutdown, \(store.idleMinutes) minutes")
                FieldError(text: store.errors["brain.idle_shutdown_min"])
                Stepper(value: Binding(get: { store.followupSeconds }, set: { store.followupSeconds = $0; store.set(["listen.followup_s": $0]) }),
                        in: 2...30, step: 1) {
                    LabeledContent("Follow-up window", value: "\(Int(store.followupSeconds)) s")
                }
                .accessibilityLabel("Follow-up window, \(Int(store.followupSeconds)) seconds")
                Text("After Mac answers, you can keep talking without the wake word for this long.")
                    .font(.footnote).foregroundStyle(.secondary)
                FieldError(text: store.errors["listen.followup_s"])
            }
        }
        .formStyle(.grouped)
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
        Form {
            ConnectionNote(store: store)
            Section {
                LabeledContent("Now speaking with", value: model.voiceEngineLabel)
                HStack {
                    Button("Record or Choose…", action: openVoicePanel)
                    Button("Use Default Voice", action: voiceReset).disabled(!model.voiceEngine.hasPrefix("clone"))
                }
            } header: {
                Text("Your cloned voice")
            } footer: {
                Text("A 15-second sample is enough. The sample never leaves this Mac.")
                    .font(.footnote).foregroundStyle(.secondary)
            }

            Section("Speed") {
                HStack {
                    Slider(value: Binding(get: { store.ttsSpeed }, set: { store.ttsSpeed = $0; store.setDebounced("tts.speed", ($0 * 100).rounded() / 100, delay: 0.4) }),
                           in: 0.8...1.3, step: 0.05) { Text("Speed") }
                        .accessibilityValue(String(format: "%.2f times", store.ttsSpeed))
                    Text(String(format: "%.2f×", store.ttsSpeed)).monospacedDigit().frame(width: 48, alignment: .trailing)
                }
                FieldError(text: store.errors["tts.speed"])
            }

            if groups.isEmpty {
                Section("Built-in voices") {
                    Text(store.loaded ? "No voices reported by Mac." : "Loading voices…").foregroundStyle(.secondary)
                }
            }
            ForEach(groups, id: \.language) { g in
                Section(g.language) {
                    ForEach(g.voices) { v in voiceRow(v) }
                }
            }
            FieldError(text: store.errors["tts.voice"])
        }
        .formStyle(.grouped)
    }

    private func voiceRow(_ v: Voice) -> some View {
        let selected = store.ttsVoice == v.id
        return HStack(spacing: 10) {
            Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(selected ? Color.accentColor : .secondary)
                .accessibilityHidden(true)
            Text(v.name)
            if !v.gender.isEmpty {
                Text(v.gender.capitalized).font(.caption)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(Capsule().fill(Color.primary.opacity(0.08)))
                    .accessibilityHidden(true)
            }
            if !v.available {
                Text("Not downloaded").font(.caption).foregroundStyle(.secondary)
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
        Form {
            Section {
                HStack(alignment: .center, spacing: 18) {
                    ZStack {
                        Circle().fill(Color.primary.opacity(0.06)).frame(width: 96, height: 96)
                        Circle().strokeBorder(targeted ? Color.accentColor : Color.primary.opacity(0.12),
                                              style: StrokeStyle(lineWidth: 2, dash: portrait == nil ? [5, 4] : []))
                            .frame(width: 96, height: 96)
                        if let portrait {
                            Image(nsImage: portrait).resizable().scaledToFill()
                                .frame(width: 92, height: 92).clipShape(Circle())
                        } else {
                            Image(systemName: "person.crop.circle.badge.plus")
                                .font(.system(size: 30, weight: .light)).foregroundStyle(.secondary)
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
                    .accessibilityLabel(portrait == nil ? "No picture. Drop a photo here or choose one." : "Mac’s picture")
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Picture").font(.body.weight(.medium))
                        Text("Used for Mac’s avatar — animated avatar coming soon.")
                            .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                        HStack {
                            Button("Choose Photo…", action: choose)
                            if portrait != nil { Button("Remove", action: { use(nil) }) }
                        }
                        FieldError(text: error)
                    }
                }
                .padding(.vertical, 4)
            }
            Section("HUD") {
                Picker("Position", selection: $hudPosition) {
                    Text("Top of the screen").tag("top")
                    Text("Bottom of the screen").tag("bottom")
                }
                .pickerStyle(.radioGroup)
                .onChange(of: hudPosition) { _, v in AppPrefs.hudPosition = v }
                Text("The HUD shows what you said and what Mac is doing, on the screen your pointer is on.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }

    private func choose() {
        let p = NSOpenPanel()
        p.allowedContentTypes = [.image]
        p.allowsMultipleSelection = false
        p.message = "Choose a photo for Mac’s avatar."
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
        Form {
            ConnectionNote(store: store)
            Section {
                Toggle("Allow talking to Mac from a phone", isOn: Binding(
                    get: { store.remoteEnabled },
                    set: { store.remoteEnabled = $0; store.set(["remote.enabled": $0]) }))
                FieldError(text: store.errors["remote.enabled"])
                if model.remote.active {
                    LabeledContent("Session", value: model.remote.banner)
                }
            } footer: {
                Text("Each link is private, works for 15 minutes, and asks the caller for a spoken PIN first.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            Section {
                PhoneView(state: phone.state, model: model, embedded: true,
                          enable: { phone.enable() }, retry: { phone.refresh() }, close: {})
                    .frame(maxWidth: .infinity)
            }
        }
        .formStyle(.grouped)
        .onAppear { phone.refresh() }   // the daemon answers "disabled" when off → Turn On state
        .onChange(of: store.remoteEnabled) { _, on in if on { phone.refresh() } }
    }
}

// MARK: Advanced

struct AdvancedSettings: View {
    @ObservedObject var store: SettingsStore
    var resetConversation: () -> Void

    var body: some View {
        Form {
            ConnectionNote(store: store)
            Section("Claude") {
                Toggle("Use Chrome for websites", isOn: Binding(
                    get: { store.chrome }, set: { store.chrome = $0; store.set(["brain.chrome": $0]) }))
                FieldError(text: store.errors["brain.chrome"])
                Toggle("Load my Claude MCP servers and skills", isOn: Binding(
                    get: { store.loadUserSources },
                    set: { store.loadUserSources = $0; store.set(["brain.setting_sources": $0 ? ["user"] : []]) }))
                Text("Everything you set up for Claude Code (~/.claude) becomes available to Mac, including tools that can act on your behalf.")
                    .font(.footnote).foregroundStyle(.secondary)
                FieldError(text: store.errors["brain.setting_sources"])
            }
            Section("Files") {
                LabeledContent("Results folder") {
                    HStack {
                        Text(store.resultsDir.isEmpty ? "~/Documents/Mac" : store.resultsDir)
                            .lineLimit(1).truncationMode(.middle).foregroundStyle(.secondary)
                        Button("Choose…", action: chooseFolder)
                    }
                }
                FieldError(text: store.errors["results_dir"])
                HStack {
                    Button("Open Config File") {
                        if FileManager.default.fileExists(atPath: Paths.configFile.path) {
                            NSWorkspace.shared.open(Paths.configFile)
                        } else {
                            NSWorkspace.shared.open(Paths.home)
                        }
                    }
                    Button("Open Logs") { NSWorkspace.shared.open(Paths.logDir) }
                }
            }
            Section("Conversation") {
                Button("Reset Conversation", action: resetConversation)
                Text("Mac forgets the current conversation and starts fresh. Memory notes are kept.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
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
