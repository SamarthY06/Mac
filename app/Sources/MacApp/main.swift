import AVFoundation
import AppKit
import Combine
import ServiceManagement
import UserNotifications

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let model = AppModel()
    let audio = AudioEngine()
    let control = ControlClient()
    let supervisor = Supervisor()
    let perms = Permissions()
    var power: Power!
    var statusBar: StatusBar!
    var hud: HUDController!
    var hotkey: Hotkey?
    lazy var onboarding = OnboardingWindow(perms: perms, model: model)
    lazy var phone = PhonePanel(model: model)
    lazy var voice = VoicePanel(model: model, control: control)
    lazy var settings = SettingsWindow(deps: .init(
        model: model, control: control, perms: perms, phone: phone,
        openVoicePanel: { [weak self] in self?.voice.show() },
        resetConversation: { [weak self] in self?.control.send("reset") },
        voiceReset: { [weak self] in self?.control.send("voice_reset") }))

    /// The window to show for "open Mac": Meet Mac on first run, Settings afterwards.
    func showMainWindow() {
        if UserDefaults.standard.bool(forKey: "onboarded") { settings.show() } else { onboarding.show() }
    }

    func applicationDidFinishLaunching(_ note: Notification) {
        let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev"
        Log.info("Mac.app \(v) launching pid=\(ProcessInfo.processInfo.processIdentifier) MAC_HOME=\(Paths.home.path)")

        power = Power(audio: audio, control: control, model: model)
        hud = HUDController(model: model)
        statusBar = StatusBar(model: model, actions: .init(
            talk: { [weak self] in self?.talk() },
            togglePower: { [weak self] in self?.setPower(on: self?.model.off ?? false) },
            toggleMute: { [weak self] in self?.toggleMute() },
            stop: { [weak self] in self?.control.send("stop") },
            reset: { [weak self] in self?.control.send("reset") },
            voice: { [weak self] in self?.voice.show() },
            remoteLink: { [weak self] in self?.remoteLink() },
            remoteEnd: { [weak self] in self?.control.send("remote_end") },
            permissions: { [weak self] in self?.settings.show(.permissions) },
            settings: { [weak self] in self?.settings.show() },
            restartDaemon: { [weak self] in self?.supervisor.restartNow() },
            restartAudio: { [weak self] in
                self?.audio.restart(reason: "user request")
                self?.control.send("audio_reinit")
            }
        ))

        model.onStateChange = { [weak self] s in
            guard let self else { return }
            // `model.off` is the single source of truth (set by the daemon's state event, or
            // locally by setPower while disconnected); the engine follows it. setPowered is a no-op
            // when already in that state.
            self.audio.setPowered(!self.model.off)
            self.statusBar.refreshIcon()
            self.power.stateChanged(s)
        }
        control.onEvent = { [weak self] e in
            guard let self else { return }
            if (e["event"] as? String) == "notify" {
                self.notify(title: (e["title"] as? String) ?? "Mac", body: (e["body"] as? String) ?? "",
                            path: e["path"] as? String)
                return
            }
            let wasMuted = self.model.muted
            self.model.handle(event: e)
            if wasMuted != self.model.muted {
                self.audio.setMuted(self.model.muted)
            }
            self.statusBar.refreshIcon()
        }
        control.onConnection = { [weak self] up in
            guard let self else { return }
            self.model.setConnected(up)
            if up {
                self.model.setState(self.model.off ? .off : .idle)  // until the first state event arrives
                if self.model.muted { self.control.send("mute", ["value": true]) }
                if let cmd = self.pendingPowerCmd { self.sendPower(cmd) }
                self.pollStatus(force: true)
                self.power.daemonConnected()
                self.model.guided?.check(deep: false)  // static Claude/Chrome facts for Meet Mac
            }
            self.statusBar.refreshIcon()
        }
        // Daemon facts the UI needs that aren't pushed as events (brain sign-in, Chrome
        // pairing, voice availability, remote session): poll `status` every 10 s.
        Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.pollStatus() }  // rate-limited inside
        }
        audio.onStatus = { [weak self] dev, aec, warn in
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    self?.model.inputDevice = dev
                    self?.model.aec = aec
                    self?.model.audioWarning = warn
                }
            }
        }
        supervisor.onStatus = { [weak self] s in self?.model.daemonStatus = s.label }
        audio.onInputSwitched = { [weak self] text in
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    // One-time HUD line (UI clears it) + one notification per device (N5 b).
                    self?.model.inputSwitchedNote = text
                    self?.notify(title: "Mac switched your microphone", body: text, path: nil)
                }
            }
        }
        setupNotifications()
        model.guided = GuidedActions(control: control, model: model)
        // Test hook: MAC_DEBUG_GUIDED=installClaude|signIn|installExtension|connectChrome runs
        // that guided action 2 s after launch (with MAC_GUIDED_NO_OPEN=1 it only writes the script).
        if let a = Paths.env["MAC_DEBUG_GUIDED"] {
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
                MainActor.assumeIsolated {
                    switch a {
                    case "installClaude": self?.model.guided?.installClaude()
                    case "signIn": self?.model.guided?.signIn()
                    case "installExtension": self?.model.guided?.installExtension()
                    case "connectChrome": self?.model.guided?.connectChrome()
                    default: break
                    }
                }
            }
        }

        installSignalHandlers()
        power.start()
        control.start()
        supervisor.start()
        hotkey = Hotkey { [weak self] in self?.talk() }

        audio.startLink()
        // OFF survives relaunch (app-side, so the mic never blips before the daemon says so).
        if UserDefaults.standard.bool(forKey: "off") {
            model.off = true
            audio.setPowered(false)
            Log.info("starting powered off (persisted)", "power")
        }
        offPersist = model.$off.removeDuplicates().sink { UserDefaults.standard.set($0, forKey: "off") }
        // Start the engine right away (while the TCC prompt is pending, input is silent).
        // If the user grants access afterwards, restart it so real audio flows.
        audio.start()
        perms.onMicGranted = { [weak self] in self?.audio.restart(reason: "microphone permission granted") }
        let firstRun = !UserDefaults.standard.bool(forKey: "onboarded") && !Paths.flag("MAC_NO_ONBOARDING")
        // First run: the onboarding window's "Allow…" asks for the mic (UX#18, no prompt collision);
        // later launches ask right away if it is still undecided.
        if !firstRun, AVCaptureDevice.authorizationStatus(for: .audio) == .notDetermined {
            perms.requestMic()
        }
        phone.onEnable = { [weak self] in
            self?.control.send("remote_enable") { ok, _, err in
                if ok { self?.remoteLink() } else { self?.phone.show(.error(err ?? "Couldn’t turn on phone access.")) }
            }
        }
        phone.onRetry = { [weak self] in self?.remoteLink() }
        onboarding.onDone = { [weak self] in self?.settings.show(.general) }

        if firstRun {
            onboarding.show()
        }
        // Ways to find the window when the menu bar icon can't be seen (notch / overflow):
        //  - `MAC_SHOW_WINDOW=1` env or `--show` argument at launch
        //  - double-clicking Mac.app while it runs (applicationShouldHandleReopen)
        //  - distributed notification `io.github.macagent.Mac.showWindow` from any process
        //  - automatic: if the status item is hidden, notify + open the window.
        DistributedNotificationCenter.default().addObserver(
            forName: Self.showWindowNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                Log.info("show window requested via distributed notification", "ui")
                self?.showMainWindow()
            }
        }
        if Paths.flag("MAC_SHOW_WINDOW") || CommandLine.arguments.contains("--show") {
            showMainWindow()
        }
        NotificationCenter.default.addObserver(
            forName: NSApplication.didChangeScreenParametersNotification, object: nil, queue: .main
        ) { [weak self] _ in
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { MainActor.assumeIsolated { self?.checkStatusItemVisible() } }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            MainActor.assumeIsolated { self?.checkStatusItemVisible() }
        }
        // Screenshot/debug hook: MAC_DEBUG_UI=menu|onboarding|phone|voice|settings opens that UI a few seconds in.
        if let ui = Paths.env["MAC_DEBUG_UI"] {
            if let a = Paths.env["MAC_DEBUG_APPEARANCE"] {   // light|dark screenshots
                NSApp.appearance = NSAppearance(named: a == "dark" ? .darkAqua : .aqua)
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 4) { [weak self] in
                MainActor.assumeIsolated {
                    switch ui {
                    case "menu": self?.statusBar.openMenu()
                    case "onboarding": self?.onboarding.show()
                    case "phone": self?.remoteLink()
                    case "voice":
                        self?.voice.show()
                        if let f = Paths.env["MAC_DEBUG_VOICE_FILE"], !f.isEmpty { self?.voice.debugClone(file: f) }
                    case "settings":
                        self?.settings.show(SettingsSection(rawValue: Paths.env["MAC_DEBUG_SETTINGS_SECTION"] ?? "general"))
                    default: break
                    }
                }
            }
        }
    }

    /// SIGTERM/SIGINT (e.g. `killall Mac`, logout) → orderly quit so the daemon child is
    /// stopped too.
    private var signalSources: [DispatchSourceSignal] = []
    func installSignalHandlers() {
        for sig in [SIGTERM, SIGINT, SIGHUP] {
            signal(sig, SIG_IGN)
            let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            src.setEventHandler {
                Log.info("received signal \(sig); quitting")
                MainActor.assumeIsolated { NSApp.terminate(nil) }
            }
            src.resume()
            signalSources.append(src)
        }
    }

    private var offPersist: AnyCancellable?

    func applicationWillTerminate(_ note: Notification) {
        Log.info("terminating")
        control.stop()
        audio.shutdown()
        supervisor.stopSync()
    }

    /// `status` reply → AppModel (brain sign-in, Chrome pairing, voice availability, remote
    /// session). Polled every 10 s, or every 2 s while the onboarding window is open.
    private var lastStatusPoll = Date.distantPast
    func pollStatus(force: Bool = false) {
        guard control.isConnected else { return }
        let interval: TimeInterval = model.onboardingVisible ? 2 : 10
        guard force || Date().timeIntervalSince(lastStatusPoll) >= interval - 0.5 else { return }
        lastStatusPoll = Date()
        control.send("status") { [weak self] ok, data, _ in
            guard ok, let self else { return }
            // Tri-state: anything absent or unrecognised is "unknown" (never ✓ by default).
            // Accepts B2's strings (`auth`/`chrome`) and the bool form (`auth_ok`/`chrome_connected`).
            let brain = data["brain"] as? [String: Any] ?? [:]
            if let auth = brain["auth"] as? String, ["ok", "signed_out", "missing", "api_key"].contains(auth) {
                self.model.brainAuth = auth
            } else if let ok = brain["auth_ok"] as? Bool {
                self.model.brainAuth = ok ? "ok" : "signed_out"
            } else {
                self.model.brainAuth = ""
            }
            if let chrome = brain["chrome"] as? String, chrome != "unknown" {
                self.model.chromePaired = chrome == "connected"
            } else if let c = brain["chrome_connected"] as? Bool {
                self.model.chromePaired = c
            } else {
                self.model.chromePaired = nil
            }
            let voice = (data["voice"] as? Bool) ?? true
            let reason = (data["voice_reason"] as? String) ?? ""
            self.model.voiceUnavailable = voice ? nil : (reason.isEmpty ? "speech models missing" : reason)
            if let cc = data["connect"] as? [String: Any], !ConnectCheck.isPending(cc) {  // static block in status
                var c = ConnectCheck.parse(cc); c.checking = self.model.connect.checking
                if c != self.model.connect { self.model.connect = c }
            }
            if var r = data["remote"] as? [String: Any] {
                r["event"] = "remote"
                self.model.handle(event: r)  // same shape as the pushed event
            }
        }
    }

    // MARK: finding the app (notch / overflow)

    static let showWindowNotification = Notification.Name("io.github.macagent.Mac.showWindow")
    private var warnedHiddenItem = false

    /// Second launch while running (double-click in Finder/Dock): open the window.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        Log.info("reopen requested; showing the window", "ui")
        showMainWindow()
        return false
    }

    /// Is the status item actually visible? On notched Macs an overflowing menu bar hides
    /// items whose x-range falls under the notch (between the two auxiliary areas); the
    /// item's window may also be missing, invisible or fully occluded.
    func statusItemVisible() -> (visible: Bool, why: String) {
        guard let w = statusBar.buttonWindow else { return (false, "no window yet") }
        guard w.isVisible else { return (false, "window not visible") }
        guard w.occlusionState.contains(.visible) else { return (false, "window occluded") }
        let f = w.frame
        let screen = NSScreen.screens.first { $0.frame.intersects(f) } ?? NSScreen.main
        guard let screen else { return (false, "no screen") }
        if let left = screen.auxiliaryTopLeftArea, let right = screen.auxiliaryTopRightArea {
            // Notch zone = between the two auxiliary areas (screen coordinates).
            if f.midX > left.maxX && f.midX < right.minX {
                return (false, String(format: "under the notch (x %.0f, notch %.0f–%.0f)", f.midX, left.maxX, right.minX))
            }
        }
        guard screen.frame.intersects(f) else { return (false, "off screen") }
        return (true, "ok")
    }

    func checkStatusItemVisible() {
        let (visible, why) = statusItemVisible()
        Log.info("status item visible=\(visible) (\(why))", "ui")
        model.statusItemHidden = !visible
        guard !visible, !warnedHiddenItem else { return }
        warnedHiddenItem = true
        notify(title: "Mac is running",
               body: "Its menu bar icon is hidden by the notch or a full menu bar. Opening the Mac window…",
               path: nil)
        showMainWindow()
    }

    // MARK: notifications (UX#7)

    private var notificationsReady = false
    private func setupNotifications() {
        guard Bundle.main.bundleIdentifier != nil else { return }  // bare `swift run` binary
        let c = UNUserNotificationCenter.current()
        c.delegate = self
        c.getNotificationSettings { s in
            DispatchQueue.main.async { MainActor.assumeIsolated { self.notificationsReady = s.authorizationStatus != .denied } }
        }
    }

    /// Posts a user notification; a click reveals `path` in Finder (or opens a folder).
    func notify(title: String, body: String, path: String?) {
        guard notificationsReady else { Log.info("notification skipped (not authorized): \(title)", "ui"); return }
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = nil
        if let path { content.userInfo = ["path": path] }
        let req = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(req) { err in
            if let err { Log.warn("notification failed: \(err)", "ui") }
        }
    }

    // MARK: actions

    func talk() {
        guard control.isConnected else { NSSound.beep(); return }
        if model.state == .speaking || model.state == .thinking {
            control.send("stop")
        }
        control.send("listen")
    }

    /// Off switch. Works locally even when the daemon is unreachable; the command is
    /// delivered when the connection comes back.
    var pendingPowerCmd: String?
    func setPower(on: Bool) {
        model.off = !on
        model.holdOff = !on
        audio.setPowered(on)
        sendPower(on ? "on" : "off")
        // Provisional state (immediate icon/HUD feedback); the daemon's state event confirms.
        if control.isConnected { model.setState(on ? .idle : .off) } else { model.setState(on ? .unknown : .off) }
        statusBar.refreshIcon()
        Log.info("power \(on ? "on" : "off") (daemon \(control.isConnected ? "notified" : "unreachable, queued"))", "ui")
    }

    private func sendPower(_ cmd: String) {
        pendingPowerCmd = cmd
        let sent = control.send(cmd) { [weak self] ok, _, err in
            if ok { if self?.pendingPowerCmd == cmd { self?.pendingPowerCmd = nil; self?.model.holdOff = false } }
            else { Log.warn("power \(cmd) failed: \(err ?? "?")", "ui") }
        }
        if !sent { Log.info("power \(cmd) queued until the daemon connects", "ui") }
    }

    func toggleMute() {
        let m = !model.muted
        model.muted = m
        audio.setMuted(m)
        control.send("mute", ["value": m])
        statusBar.refreshIcon()
    }

    /// "Talk from Phone…": QR code panel (UX#13).
    func remoteLink() {
        phone.show(.loading)
        let sent = control.send("remote_link") { [weak self] ok, data, err in
            self?.phone.show(PhonePanel.content(ok: ok, data: data, error: err))
        }
        if !sent { phone.show(.error("Mac isn’t connected yet. Try again in a moment.")) }
    }
}

extension AppDelegate: UNUserNotificationCenterDelegate {
    nonisolated func userNotificationCenter(_ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse,
                                            withCompletionHandler done: @escaping () -> Void) {
        let path = response.notification.request.content.userInfo["path"] as? String
        DispatchQueue.main.async {
            if let path {
                let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
                var isDir: ObjCBool = false
                if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue {
                    NSWorkspace.shared.open(url)
                } else {
                    NSWorkspace.shared.activateFileViewerSelecting([url])
                }
            }
            done()
        }
    }

    nonisolated func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent n: UNNotification,
                                            withCompletionHandler done: @escaping (UNNotificationPresentationOptions) -> Void) {
        done([.banner, .list])  // show even while Mac.app is "active"
    }
}

// Ignore SIGPIPE globally (sockets also use SO_NOSIGPIPE; the child's pipes don't).
signal(SIGPIPE, SIG_IGN)

// `Mac --unregister-login-item`: used by install.sh --uninstall (SMAppService registrations
// can only be removed by the app itself).
if CommandLine.arguments.contains("--register-login-item") {
    do { try SMAppService.mainApp.register(); print("login item registered: \(SMAppService.mainApp.status.rawValue)") } catch { print("login item: \(error.localizedDescription)") }
    exit(0)
}
if CommandLine.arguments.contains("--unregister-login-item") {
    do { try SMAppService.mainApp.unregister(); print("login item removed") } catch { print("login item: \(error.localizedDescription)") }
    exit(0)
}

MainActor.assumeIsolated {
    let app = NSApplication.shared
    let delegate = AppDelegate()
    app.delegate = delegate
    app.setActivationPolicy(.accessory)
    app.run()
}
