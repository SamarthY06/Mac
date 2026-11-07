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

/// "Meet Mac": one window — hero, permission checklist (live), try-it, launch at login.
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
            let hc = NSHostingController(rootView: OnboardingView(perms: perms, model: model, close: { [weak w, weak self] in w?.close(); self?.onDone?() }))
            hc.sizingOptions = [.preferredContentSize]
            w.contentViewController = hc
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
            Text("Mac’s menu bar icon is hidden by the notch. Close a few menu bar apps, or open this window any time by double-clicking Mac.app.")
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
        GroupBox {
            PermissionsChecklist(perms: perms, model: model, asCard: true)
        } label: {
            HStack(spacing: 6) {
                Text(allSet ? "All set" : "Mac needs a few things")
                    .font(.headline)
                if allSet {
                    Image(systemName: "checkmark.seal.fill").foregroundStyle(.green)
                        .symbolEffect(.bounce, value: allSet)
                }
            }
            .padding(.bottom, 2)
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
                Button("Done", action: close).keyboardShortcut(.defaultAction)
            }
            if let loginError {
                Text(loginError).font(.footnote).foregroundStyle(.red)
            }
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
    @State private var asked: Set<String> = []   // rows whose Allow… was clicked once (UX#18)
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Group {
            if asCard {
                rowsView.padding(.vertical, -6)
            } else {
                rowsView
            }
        }
    }

    private var rowsView: some View {
        VStack(spacing: 0) {
                row("Microphone", "To hear “Hey Mac” and what you ask.", perms.mic, "mic.fill", request: { perms.requestMic() })
                Divider()
                row("Accessibility", "To click and type in apps for you.", perms.accessibility, "hand.point.up.left.fill",
                    request: perms.requestAccessibility, settings: true)
                Divider()
                row("Automation", "To work with Mail, Calendar, Notes and more.", perms.automation, "applescript.fill",
                    request: perms.requestAutomation, settings: true)
                Divider()
                row("Screen Recording", "To see your screen when you ask about it.", perms.screen, "rectangle.dashed.badge.record",
                    request: perms.requestScreen, settings: true)
                Divider()
                row("Notifications", "To tell you when a long task finishes. Optional.", perms.notifications, "bell.fill",
                    request: perms.requestNotifications)
                Divider()
                claudeRow
                Divider()
                chromeRow
            }
    }

    /// "Claude Code signed in" (UX#8): status from the daemon's `status.brain.auth`.
    private var claudeRow: some View {
        let auth = model.brainAuth
        let st: Permissions.Status = auth == "ok" || auth == "api_key" ? .granted : (auth.isEmpty ? .unknown : .denied)
        let why = auth == "missing" ? "Not installed. Mac thinks with Claude Code."
                : auth == "signed_out" ? "Not signed in. Mac uses your Claude Code sign-in."
                : auth.isEmpty && !model.connected ? "Mac uses your Claude Code sign-in. Checking…"
                : "Mac uses your Claude Code sign-in."
        return row("Claude Code", why, st, "sparkle", button: auth == "missing" ? "Install…" : "Sign in…") {
            if auth == "missing" {
                NSWorkspace.shared.open(URL(string: "https://claude.com/claude-code")!)
            } else {
                UX.openTerminal(running: "claude", name: "sign-in-to-claude")
            }
        }
    }

    /// "Chrome extension paired" (UX#8): `status.brain.chrome`. Optional.
    private var chromeRow: some View {
        let st: Permissions.Status = model.chromePaired == true ? .granted : (model.chromePaired == nil ? .unknown : .denied)
        return row("Claude in Chrome", "For websites. Optional.", st,
                   "globe", button: "Get it…") {
            NSWorkspace.shared.open(URL(string: "https://claude.ai/chrome")!)
        }
    }

    @ViewBuilder
    private func row(_ title: String, _ why: String, _ st: Permissions.Status, _ icon: String,
                     button: String? = nil, request: @escaping () -> Void, settings: Bool = false) -> some View {
        let btn = button ?? (settings && asked.contains(title) ? "Settings…" : "Allow…")
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(st == .granted ? .green : .secondary)
                .frame(width: 20)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.body.weight(.medium))
                Text(why).font(.callout).foregroundStyle(.secondary)
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
        .padding(.vertical, 5)
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
