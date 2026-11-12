import AVFoundation
import AppKit
import ApplicationServices
import CoreGraphics
import SwiftUI
import UserNotifications

/// Permission probes + requests. All requests happen in Mac.app (the TCC responsible
/// process for the daemon child), so grants apply to macd too.
@MainActor
final class Permissions: ObservableObject {
    enum Status { case granted, denied, unknown }

    @Published var mic: Status = .unknown
    @Published var accessibility: Status = .unknown
    @Published var automation: Status = .unknown
    @Published var screen: Status = .unknown
    @Published var notifications: Status = .unknown

    var allGranted: Bool { [mic, accessibility, automation, screen].allSatisfy { $0 == .granted } }
    /// Set by AppDelegate: restart the audio engine once the microphone is granted.
    var onMicGranted: (() -> Void)?

    func refresh() {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: mic = .granted
        case .notDetermined: mic = .unknown
        default: mic = .denied
        }
        accessibility = AXIsProcessTrusted() ? .granted : .denied
        screen = CGPreflightScreenCaptureAccess() ? .granted : .denied
        automation = Self.automationStatus(ask: false)
        UNUserNotificationCenter.current().getNotificationSettings { s in
            let st: Status = switch s.authorizationStatus {
            case .authorized, .provisional: .granted
            case .notDetermined: .unknown
            default: .denied
            }
            DispatchQueue.main.async { MainActor.assumeIsolated { self.notifications = st } }
        }
    }

    func requestMic(_ done: (@MainActor (Bool) -> Void)? = nil) {
        if AVCaptureDevice.authorizationStatus(for: .audio) == .denied {
            open("x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")
            return
        }
        AVCaptureDevice.requestAccess(for: .audio) { ok in
            DispatchQueue.main.async { MainActor.assumeIsolated { self.refresh(); if ok { self.onMicGranted?() }; done?(ok) } }
        }
    }

    func requestAccessibility() {
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        let trusted = AXIsProcessTrustedWithOptions([key: true] as CFDictionary)
        if !trusted { open("x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") }
        refresh()
    }

    func requestAutomation() {
        // Launch System Events (harmless) and ask for permission to send it Apple events.
        let url = URL(fileURLWithPath: "/System/Library/CoreServices/System Events.app")
        let cfg = NSWorkspace.OpenConfiguration()
        cfg.activates = false
        NSWorkspace.shared.openApplication(at: url, configuration: cfg) { _, _ in
            DispatchQueue.global(qos: .userInitiated).async {
                let st = Self.automationStatus(ask: true)  // blocks while the prompt is up
                DispatchQueue.main.async {
                    MainActor.assumeIsolated {
                        self.automation = st
                        if st == .denied {
                            self.open("x-apple.systempreferences:com.apple.preference.security?Privacy_Automation")
                        }
                    }
                }
            }
        }
    }

    func requestScreen() {
        if !CGRequestScreenCaptureAccess() {
            open("x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")
        }
        refresh()
    }

    func requestNotifications() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, _ in
            DispatchQueue.main.async { MainActor.assumeIsolated { self.refresh() } }
        }
    }

    nonisolated static func automationStatus(ask: Bool) -> Status {
        let target = NSAppleEventDescriptor(bundleIdentifier: "com.apple.systemevents")
        guard let desc = target.aeDesc else { return .unknown }
        let st = AEDeterminePermissionToAutomateTarget(desc, typeWildCard, typeWildCard, ask)
        switch st {
        case noErr: return .granted
        case OSStatus(errAEEventNotPermitted): return .denied
        case OSStatus(errAEEventWouldRequireUserConsent): return .unknown
        default: return .unknown  // procNotFound (-600): System Events not running yet
        }
    }

    private func open(_ s: String) {
        if let u = URL(string: s) { NSWorkspace.shared.open(u) }
    }
}

/// "Meet Mac": one window: hero, connect rows + permission checklist (live), try-it, launch at login.
@MainActor
final class OnboardingWindow {
    private var window: NSWindow?
    private let perms: Permissions
    private let model: AppModel
    private var refreshTimer: Timer?
    /// Called when the person clicks Done (AppDelegate opens Settings).
    var onDone: (() -> Void)?

    init(perms: Permissions, model: AppModel) {
        self.perms = perms
        self.model = model
    }

    var isVisible: Bool { window?.isVisible ?? false }

    func show() {
        perms.refresh()
        model.onboardingVisible = true
        model.guided?.check(deep: false)
        if window == nil {
            let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 500, height: 600),
                             styleMask: [.titled, .closable, .fullSizeContentView],
                             backing: .buffered, defer: false)
            w.title = "Meet Mac"
            w.titlebarAppearsTransparent = true
            w.titleVisibility = .hidden
            w.isMovableByWindowBackground = true
            w.isReleasedWhenClosed = false
            w.setAccessibilityLabel("Meet Mac")
            let hc = NSHostingController(rootView: OnboardingView(perms: perms, model: model,
                                                                  close: { [weak w, weak self] in w?.close(); self?.onDone?() }))
            hc.sizingOptions = []  // crash fix: with .fullSizeContentView, content-driven sizing loops (safe area ↔ frame) until AppKit throws
            let size = NSSize(width: max(w.contentLayoutRect.width, hc.view.fittingSize.width), height: max(w.contentLayoutRect.height, hc.view.fittingSize.height))
            w.contentViewController = hc
            w.setContentSize(size)  // after assigning the controller, or the window collapses to 0x0
            w.center()
            window = w
        }
        NSApp.activate(ignoringOtherApps: true)
        window?.makeKeyAndOrderFront(nil)
        // Grants happen in System Settings; poll while the window is open.
        refreshTimer?.invalidate()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] t in
            MainActor.assumeIsolated {
                guard let self, let w = self.window, w.isVisible else {
                    t.invalidate(); self?.model.onboardingVisible = false; return
                }
                self.perms.refresh()
            }
        }
        UserDefaults.standard.set(true, forKey: "onboarded")
    }
}

struct OnboardingView: View {
    @ObservedObject var perms: Permissions
    @ObservedObject var model: AppModel
    var close: () -> Void
    @State private var launchAtLogin = UX.launchAtLogin
    @State private var loginError: String?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        VStack(spacing: 0) {
            hero.padding(.top, 20).padding(.bottom, 12)
            VStack(alignment: .leading, spacing: 12) {
                if model.statusItemHidden { hiddenHint }
                checklist
                tryIt
                footer
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 16)
        }
        .frame(width: 500)
        .background(Color(nsColor: .windowBackgroundColor))
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: allSet)
    }

    // MARK: hero

    private var hero: some View {
        VStack(spacing: 8) {
            if let portrait = AppPrefs.avatarImage() {
                Image(nsImage: portrait).resizable().interpolation(.high).scaledToFill()
                    .frame(width: 56, height: 56).clipShape(Circle())
                    .overlay(Circle().strokeBorder(Color.primary.opacity(0.1)))
                    .accessibilityHidden(true)
            } else {
                Image(nsImage: NSApp.applicationIconImage)
                    .resizable().interpolation(.high)
                    .frame(width: 52, height: 52)
                    .accessibilityHidden(true)
            }
            Text("Meet Mac").font(.system(size: 24, weight: .bold))
            Text("Say “Hey Mac,” then ask. Claude does the rest, right here on your Mac.")
                .font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
                .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .combine)
    }

    /// Shown when the menu bar icon is hidden by the notch / menu-bar overflow (Builder C sets it).
    private var hiddenHint: some View {
        Label {
            Text("Mac’s menu bar icon is hidden by the notch. Close a few menu bar apps, or open this window any time by double-clicking OpenValet.app.")
                .font(.callout).fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: "eye.slash").foregroundStyle(.orange).accessibilityHidden(true)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color.orange.opacity(0.10)))
    }

    // MARK: permissions

    private var allSet: Bool { perms.allGranted && (model.brainAuth == "ok" || model.brainAuth == "api_key") }

    private var checklist: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                SectionTitle(text: allSet ? "All set" : "Mac needs a few things")
                if allSet {
                    Image(systemName: "checkmark.seal.fill").foregroundStyle(.green)
                        .symbolEffect(.bounce, value: allSet)
                }
            }
            Card { PermissionsChecklist(perms: perms, model: model, asCard: true, includeConnect: true) }
        }
    }

    // MARK: try it

    private var tryIt: some View {
        let hot = model.connected && !model.off && !model.muted
        let heard = model.state.inTurn
        return HStack(spacing: 12) {
            Image(systemName: heard ? "waveform" : "waveform.circle")
                .font(.system(size: 20, weight: .medium))
                .foregroundStyle(hot ? Color.accentColor : .secondary)
                .symbolEffect(.variableColor.iterative.reversing, options: .repeating, isActive: heard && !reduceMotion)
                .frame(width: 28)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(heard ? "Heard you!" : "Try it").font(.body.weight(.medium))
                Text(heard ? model.statusLine
                     : (hot ? "Say “Hey Mac, what can you do?”" : "Once Mac is connected, say “Hey Mac, what can you do?”"))
                    .font(.callout).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(.horizontal, 12).padding(.vertical, 10)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color.accentColor.opacity(hot ? 0.10 : 0.05)))
        .accessibilityElement(children: .combine)
    }

    // MARK: footer

    private var footer: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label {
                Text("**Mute** keeps Mac running but deaf; **Turn Off** releases the mic. With AirPods, Mac keeps using the built-in mic.")
                    .font(.footnote).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } icon: {
                Image(systemName: "info.circle").foregroundStyle(.secondary).accessibilityHidden(true)
            }
            Divider()
            HStack {
                Toggle("Launch at login", isOn: $launchAtLogin)
                    .toggleStyle(.checkbox)
                    .onChange(of: launchAtLogin) { _, on in setLaunchAtLogin(on) }
                Spacer()
                if model.claudeStage != .connected {
                    Button("Later", action: close).accessibilityHint("Finish setup later from Settings")
                }
                Button("Done", action: close).keyboardShortcut(.defaultAction)
                    .disabled(model.claudeStage != .connected)
                    .accessibilityHint(model.claudeStage == .connected ? "" : "Sign in to Claude first")
            }
            if let loginError {
                Text(loginError).font(.footnote).foregroundStyle(.red)
            }
            if model.claudeStage == .connected, model.chromeStage != .connected {
                Text("Browser tasks need Chrome. Connect it above, or later in Settings.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            Text("Everything stays on this Mac except your conversation with Claude.")
                .font(.footnote).foregroundStyle(.tertiary)
        }
    }

    private func setLaunchAtLogin(_ on: Bool) {
        loginError = UX.setLaunchAtLogin(on)
        if loginError != nil { launchAtLogin = UX.launchAtLogin }
    }
}

/// The permission + prerequisite rows, shared by Meet Mac (card) and Settings › Permissions (form).
struct PermissionsChecklist: View {
    @ObservedObject var perms: Permissions
    @ObservedObject var model: AppModel
    var asCard: Bool
    var includeConnect: Bool = true
    @State private var asked: Set<String> = []   // rows whose Allow… was clicked once (UX#18)
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View { rowsView }

    private var rowsView: some View {
        VStack(spacing: 0) {
                if includeConnect {
                    ConnectRows(model: model)
                    RowDivider()
                }
                row("Microphone", "To hear “Hey Mac” and what you ask.", perms.mic, "mic.fill", request: { perms.requestMic() })
                RowDivider()
                row("Accessibility", "To click and type in apps for you.", perms.accessibility, "hand.point.up.left.fill",
                    request: perms.requestAccessibility, settings: true)
                RowDivider()
                row("Automation", "To work with Mail, Calendar, Notes and more.", perms.automation, "applescript.fill",
                    request: perms.requestAutomation, settings: true)
                RowDivider()
                row("Screen Recording", "To see your screen when you ask “what’s this?”", perms.screen, "rectangle.dashed.badge.record",
                    request: perms.requestScreen, settings: true)
                RowDivider()
                row("Notifications", "To tell you when a long task finishes. Optional.", perms.notifications, "bell.fill",
                    request: perms.requestNotifications)
            }
    }

    @ViewBuilder
    private func row(_ title: String, _ why: String, _ st: Permissions.Status, _ icon: String,
                     button: String? = nil, request: @escaping () -> Void, settings: Bool = false) -> some View {
        let btn = button ?? (settings && asked.contains(title) ? "Settings…" : "Allow…")
        HStack(spacing: 12) {
            IconBadge(symbol: icon, tint: st == .granted ? .green : .accentColor)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.system(size: 13))
                Text(why).font(.system(size: 11)).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            statusMark(st)
            Button(action: { asked.insert(title); request() }) {
                Text(btn).frame(width: 70)
            }
            .disabled(st == .granted)
            .opacity(st == .granted ? 0 : 1)
            .accessibilityLabel(btn == "Settings…" ? "Open System Settings for \(title)" : "\(btn.dropLast()) \(title)")
            .accessibilityHidden(st == .granted)
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(title): \(label(for: st)). \(why)")
    }

    private func statusMark(_ st: Permissions.Status) -> some View {
        Image(systemName: st == .granted ? "checkmark.circle.fill" : "circle")
            .font(.system(size: 16, weight: .regular))
            .foregroundStyle(st == .granted ? Color.green : Color.secondary.opacity(0.6))
            .contentTransition(reduceMotion ? .identity : .symbolEffect(.replace))
            .accessibilityHidden(true)
    }

    private func label(for st: Permissions.Status) -> String {
        switch st {
        case .granted: return "allowed"
        case .denied: return "not allowed"
        case .unknown: return "not yet asked"
        }
    }

}

/// The two rich "Connect" rows: Claude (Install → Sign in → Connected) and Chrome
/// (Install extension → Connect → Connected), shown above the macOS permissions.
struct ConnectRows: View {
    @ObservedObject var model: AppModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        VStack(spacing: 0) {
            claudeRow
            RowDivider()
            chromeRow
        }
    }

    // MARK: Claude

    private var claudeRow: some View {
        let stage = model.claudeStage
        let c = model.connect.claude
        let apiKey = model.claudeUsesApiKey
        return ConnectRow(icon: "sparkle", title: "Claude",
                          steps: ["Install", "Sign in", "Connected"], stage: stage?.rawValue,
                          status: claudeStatus(stage), badge: stage == .connected && apiKey
                              ? ("API key: browser tasks unavailable", .orange) : nil,
                          fix: stage == .connected
                              ? (apiKey ? "Browser tasks need a claude.ai sign-in; the API key covers everything else." : "")
                              : UX.noDash(c.fix)) {
            switch stage {
            case .install: Button("Install Claude Code…") { model.guided?.installClaude() }
            case .signIn: Button("Sign in…") { model.guided?.signIn() }
            case .connected: EmptyView()
            case nil: Button("Sign in…") { model.guided?.signIn() }.disabled(model.guided == nil)
            }
        } secondary: { EmptyView() } accessory: { EmptyView() }
    }

    private func claudeStatus(_ stage: ClaudeStage?) -> String {
        let c = model.connect.claude
        switch stage {
        case .install: return "Claude Code isn’t installed yet. Mac thinks with it."
        case .signIn: return c.version.isEmpty ? "Installed, not signed in yet." : "Claude Code \(c.version), not signed in yet."
        case .connected:
            let v = c.version.isEmpty ? "Claude Code" : "Claude Code \(c.version)"
            return model.claudeUsesApiKey ? "\(v) · using an API key" : "\(v) · signed in with claude.ai"
        case nil: return model.connected ? "Checking…" : "Waiting for Mac to connect…"
        }
    }

    // MARK: Chrome

    private var chromeRow: some View {
        let stage = model.chromeStage
        let ch = model.connect.chrome
        let checking = model.connect.checking
        let hint = "Open the Claude extension in Chrome and sign in with the same account."
        let daemonFix = UX.noDash(ch.fix)
        return ConnectRow(icon: "globe", title: "Claude in Chrome",
                          steps: ["Install extension", "Sign in to the extension", "Connect"], stage: stage?.rawValue,
                          status: chromeStatus(stage), badge: nil,
                          fix: stage == .connected ? "" : (stage == .connect ? (daemonFix.isEmpty ? hint : "\(hint) \(daemonFix)") : daemonFix)) {
            switch stage {
            case .install:
                Button("Get the Extension…") { model.guided?.installExtension() }
            case .connect, nil:
                // Never Terminal: Connect asks the engine for a deep check (it polls until the extension answers).
                Button {
                    model.guided?.checkConnection()
                } label: {
                    if checking {
                        HStack(spacing: 6) { ProgressView().controlSize(.small); Text("Connecting, about 10 seconds") }
                    } else {
                        Text("Connect")
                    }
                }
                .disabled(checking || !model.connected || model.guided == nil)
                .accessibilityLabel(checking ? "Connecting to the Chrome extension" : "Connect to the Chrome extension")
            case .connected:
                EmptyView()
            }
        } secondary: {
            if stage == .connected {
                Button {
                    model.guided?.checkConnection()
                } label: {
                    if checking {
                        HStack(spacing: 5) { ProgressView().controlSize(.mini); Text("Checking") }
                    } else {
                        Text("Check Connection")
                    }
                }
                .buttonStyle(.link)
                .font(.caption)
                .fixedSize()
                .disabled(checking || !model.connected || model.guided == nil)
                .accessibilityLabel("Check the Chrome connection again")
            }
        } accessory: {
            HStack(spacing: 5) {
                chip("Chrome open", ch.browserRunning)
                chip("Extension", ch.extensionInstalled)
                chip("Native host", ch.nativeHost)
            }
            .fixedSize()
        }
    }

    private func chromeStatus(_ stage: ChromeStage?) -> String {
        switch stage {
        case .install: return "For websites. Optional: get the extension for Chrome."
        case .connect: return "Extension found. Sign in to it, then connect."
        case .connected: return "Connected. Claude can browse for you."
        case nil: return model.connected ? "Checking…" : "Waiting for Mac to connect…"
        }
    }

    private func chip(_ label: String, _ ok: Bool?) -> some View {
        HStack(spacing: 3) {
            Image(systemName: ok == true ? "checkmark" : ok == false ? "xmark" : "minus")
                .font(.system(size: 8, weight: .bold))
            Text(label).font(.caption2).lineLimit(1)
        }
        .padding(.horizontal, 6).padding(.vertical, 2)
        .foregroundStyle(ok == true ? Color.green : ok == false ? Color.orange : Color.secondary)
        .background(Capsule().fill((ok == true ? Color.green : ok == false ? Color.orange : Color.secondary).opacity(0.12)))
        .accessibilityLabel("\(label): \(ok == true ? "yes" : ok == false ? "no" : "unknown")")
    }
}

/// One rich connect row: title + primary button, stepper (+ secondary action), status, badge,
/// chips, fix text. Everything horizontal is fixed-size so nothing wraps mid-word.
struct ConnectRow<Buttons: View, Secondary: View, Accessory: View>: View {
    var icon: String
    var title: String
    var steps: [String]
    var stage: Int?            // index of the current step; steps.count-1 = done; nil = unknown
    var status: String
    var badge: (String, Color)?
    var fix: String
    @ViewBuilder var buttons: () -> Buttons        // primary action, trailing on the title line
    @ViewBuilder var secondary: () -> Secondary    // small action, trailing on the stepper line
    @ViewBuilder var accessory: () -> Accessory    // chips

    private var done: Bool { stage == steps.count - 1 }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            IconBadge(symbol: icon, tint: done ? .green : .accentColor).padding(.top, 1)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(title).font(.system(size: 13, weight: .medium)).lineLimit(1)
                    Spacer(minLength: 8)
                    if done {
                        Image(systemName: "checkmark.circle.fill").font(.system(size: 16)).foregroundStyle(.green)
                            .accessibilityHidden(true)
                    }
                    buttons()
                }
                HStack(spacing: 8) {
                    stepper
                    Spacer(minLength: 8)
                    secondary()
                }
                Text(status).font(.system(size: 11)).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if let (text, color) = badge {
                    Text(text).font(.caption.weight(.medium)).fixedSize()
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .foregroundStyle(color)
                        .background(Capsule().fill(color.opacity(0.14)))
                }
                accessory()
                if !fix.isEmpty {
                    Text(fix).font(.system(size: 11)).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 12).padding(.vertical, 9)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(title): \(status)")
    }

    /// Install › Sign in › Connected ✓ with a dot per step.
    private var stepper: some View {
        HStack(spacing: 4) {
            ForEach(Array(steps.enumerated()), id: \.offset) { i, name in
                let state: Int = stage.map { s in i < s || done ? 2 : (i == s ? 1 : 0) } ?? 0
                HStack(spacing: 3) {
                    Image(systemName: state == 2 ? "checkmark.circle.fill" : state == 1 ? "circle.fill" : "circle")
                        .font(.system(size: 8))
                        .foregroundStyle(state == 2 ? Color.green : state == 1 ? Color.accentColor : Color.secondary.opacity(0.5))
                    Text(name).font(.caption).lineLimit(1)
                        .foregroundStyle(state == 1 ? Color.primary : Color.secondary)
                }
                if i < steps.count - 1 {
                    Image(systemName: "chevron.right").font(.system(size: 7, weight: .bold)).foregroundStyle(.tertiary)
                }
            }
        }
        .fixedSize()
        .accessibilityHidden(true)
    }
}
