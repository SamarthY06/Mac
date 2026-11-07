import AppKit
import Foundation

/// Result of the daemon's `connect_check` (and the static part of `status`): is this Mac
/// connected to the user's Claude Code and to Claude in Chrome? Tri-state Bools: nil =
/// not checked yet. The UI (Meet Mac) renders it; `GuidedActions` fixes it.
struct ConnectCheck: Equatable {
    struct Claude: Equatable {
        var installed: Bool?
        var version = ""
        var minOK: Bool?
        var signedIn: Bool?
        var authMethod = ""
        var fix = ""          // daemon's one-line suggestion
    }
    struct Chrome: Equatable {
        var browserRunning: Bool?
        var extensionInstalled: Bool?
        var nativeHost: Bool?
        var enabledByDefault: Bool?
        var connected: Bool?
        var connectedState = ""   // raw: "connected" | "not_connected" | "unavailable_api_key" | "unknown"
        var fix = ""
    }
    var claude = Claude()
    var chrome = Chrome()
    var ready: Bool?
    var checking = false      // a deep check is in flight
    var checkedAt: Date?

    /// Equality that ignores `checkedAt`/`checking` (what the UI should react to).
    func sameFacts(as o: ConnectCheck) -> Bool { claude == o.claude && chrome == o.chrome && ready == o.ready }

    /// Chrome is usable for browser tasks: extension + native host (we pass --chrome ourselves,
    /// so "Enabled by default" is informational only).
    var chromeReady: Bool { chrome.nativeHost == true && chrome.extensionInstalled == true }
    /// Fully connected (deep check succeeded).
    var chromeConnected: Bool { chromeReady && chrome.connected == true }

    /// `status.connect` right after daemon boot: `{"pending": true, claude: null, chrome: null}`.
    static func isPending(_ d: [String: Any]) -> Bool { (d["pending"] as? Bool) == true }

    /// Parse B2's reply `{claude: {...}, chrome: {...}, ready, checked_at}`; unknown → nil.
    /// Booleans are JSON bools except `signed_in` ("yes"|"no"|"unknown") and
    /// `chrome.connected` ("connected"|"not_connected"|"unavailable_api_key"|"unknown").
    static func parse(_ d: [String: Any]) -> ConnectCheck {
        func b(_ o: [String: Any], _ k: String) -> Bool? {
            if let v = o[k] as? Bool { return v }
            if let s = o[k] as? String {
                switch s.lowercased() {
                case "yes", "ok", "true", "connected": return true
                case "no", "false", "missing", "not_connected", "unavailable_api_key": return false
                default: return nil
                }
            }
            return nil
        }
        func s(_ o: [String: Any], _ k: String) -> String { (o[k] as? String) ?? "" }
        var c = ConnectCheck()
        let cl = d["claude"] as? [String: Any] ?? [:]
        c.claude = Claude(installed: b(cl, "installed"), version: s(cl, "version"), minOK: b(cl, "min_ok"),
                          signedIn: b(cl, "signed_in"), authMethod: s(cl, "auth_method"), fix: s(cl, "fix"))
        let ch = d["chrome"] as? [String: Any] ?? [:]
        c.chrome = Chrome(browserRunning: b(ch, "browser_running"), extensionInstalled: b(ch, "extension_installed"),
                          nativeHost: b(ch, "native_host"), enabledByDefault: b(ch, "enabled_by_default"),
                          connected: b(ch, "connected"), connectedState: s(ch, "connected"), fix: s(ch, "fix"))
        c.ready = b(d, "ready")
        c.checkedAt = Date()
        return c
    }
}

/// The "do it for me" buttons behind Meet Mac's Claude / Chrome rows. Main-actor only.
@MainActor
final class GuidedActions {
    static let claudeInstallCmd = "curl -fsSL https://claude.ai/install.sh | bash"
    static let extensionURL = URL(string: "https://chromewebstore.google.com/detail/claude/fcoeoabgfenejglbffodgkkbkcdhcgfn")!

    private let control: ControlClient
    private let model: AppModel
    private var chromePoll: Timer?
    private var chromePollStarted: Date?
    private var lastShallowReply: Date?
    private var shallowInFlight = false

    init(control: ControlClient, model: AppModel) {
        self.control = control
        self.model = model
    }

    // MARK: checks

    /// Static part (cheap): run on connect and from the status poll. `deep` is what the
    /// "Check connection" button does (daemon talks to the CLI / extension).
    func check(deep: Bool, completion: (@MainActor (ConnectCheck?) -> Void)? = nil) {
        // Shallow checks are coalesced: a caller (view onAppear, poll, …) within 1 s of the
        // last reply, or while one is in flight, gets the cached facts without a socket trip.
        if !deep {
            if shallowInFlight { completion?(model.connect); return }
            if let t = lastShallowReply, Date().timeIntervalSince(t) < 1 { completion?(model.connect); return }
            shallowInFlight = true
        }
        if deep { model.connect.checking = true }
        let sent = control.send("connect_check", ["deep": deep]) { [weak self] ok, data, err in
            guard let self else { return }
            if !deep { self.shallowInFlight = false; self.lastShallowReply = Date() }
            self.model.connect.checking = false
            guard ok else {
                Log.warn("connect_check failed: \(err ?? "?")", "connect")
                completion?(nil); return
            }
            var c = ConnectCheck.parse(data)
            c.checking = false
            // Publish only when the facts changed (a new checkedAt alone must not re-render
            // every observer, which is how a view-driven call could loop at socket speed).
            if !c.sameFacts(as: self.model.connect) { self.model.connect = c } else { self.model.connect.checkedAt = c.checkedAt }
            Log.info("connect_check deep=\(deep): claude signed_in=\(c.claude.signedIn.map(String.init) ?? "?") chrome ready=\(c.chromeReady) ready=\(c.ready.map(String.init) ?? "?")", "connect")
            completion?(c)
        }
        if !sent { shallowInFlight = false; model.connect.checking = false; completion?(nil) }
    }

    /// "Connect" / "Check connection": deep check (the daemon runs our headless `claude --chrome`,
    /// which registers the native host), then poll every 2 s until native host + extension +
    /// connected, for up to 2 minutes. No Terminal involved.
    func checkConnection() {
        check(deep: true) { [weak self] c in
            guard let self else { return }
            if let c, c.chromeConnected { return }
            self.startChromePolling(deepWhenReady: true)
        }
    }

    // MARK: guided fixes

    /// "Install Claude Code": runs the official installer in the background (no Terminal),
    /// logs its output, notifies when done, then re-checks. Sign-in still needs a Terminal.
    private var installer: Process?
    func installClaude() {
        guard installer == nil else { return }
        model.connect.checking = true
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/bash")
        p.arguments = ["-lc", Self.claudeInstallCmd]
        let out = Pipe(); p.standardOutput = out; p.standardError = out
        p.terminationHandler = { [weak self] proc in
            let text = String(decoding: out.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    guard let self else { return }
                    self.installer = nil
                    self.model.connect.checking = false
                    let ok = proc.terminationStatus == 0
                    Log.info("Claude Code installer exit \(proc.terminationStatus): \(text.suffix(400))", "connect")
                    self.onNotify?(ok ? "Claude Code installed" : "Claude Code install failed",
                                   ok ? "Next: sign in to Claude." : "See the log (Open Logs in the menu).")
                    self.check(deep: false)
                }
            }
        }
        do { try p.run(); installer = p; Log.info("running the Claude Code installer in the background", "connect") }
        catch { model.connect.checking = false; Log.error("installer failed to start: \(error)", "connect") }
    }

    /// Posts a notification (wired by AppDelegate).
    var onNotify: ((String, String) -> Void)?

    /// "Sign in": the only step that needs a Terminal (interactive browser login).
    func signIn() {
        runInTerminal(name: "sign-in-claude", title: "Sign in to Claude", lines: [
            "echo 'Signing in to Claude. A browser window opens; when you are signed in, type /exit here.'",
            "export PATH=\"$HOME/.local/bin:$PATH\"",
            "claude /login",
        ])
    }

    /// "Get the extension": Web Store page in Chrome (explicitly, if Chrome isn't the default
    /// browser), then poll until the extension shows up.
    func installExtension() {
        let url = Self.extensionURL
        let defaultBrowser = NSWorkspace.shared.urlForApplication(toOpen: url)
        let chrome = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.google.Chrome")
        if let chrome, let d = defaultBrowser, Bundle(url: d)?.bundleIdentifier != "com.google.Chrome" {
            NSWorkspace.shared.open([url], withApplicationAt: chrome, configuration: NSWorkspace.OpenConfiguration()) { _, err in
                if let err { Log.warn("open in Chrome failed: \(err)", "connect") }
            }
            Log.info("opened Web Store in Chrome explicitly (default browser is \(d.lastPathComponent))", "connect")
        } else {
            NSWorkspace.shared.open(url)
            Log.info("opened Web Store in the default browser (chrome installed: \(chrome != nil))", "connect")
        }
        startChromePolling(deepWhenReady: false)
    }

    /// Kept for API compatibility: same as the no-Terminal "Connect".
    func connectChrome() { checkConnection() }

    /// Poll `connect_check {deep:false}` every 2 s (max 2 min). After "Get the extension" we
    /// stop once the extension is installed; after "Connect" we stop once connected, and run
    /// one more deep check as soon as extension + native host are there.
    private func startChromePolling(deepWhenReady: Bool) {
        chromePoll?.invalidate()
        chromePollStarted = Date()
        var deepDone = false
        chromePoll = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] t in
            MainActor.assumeIsolated {
                guard let self else { t.invalidate(); return }
                if Date().timeIntervalSince(self.chromePollStarted ?? Date()) > 120 { t.invalidate(); self.chromePoll = nil; return }
                self.check(deep: false) { c in
                    guard let c else { return }
                    let done = deepWhenReady ? c.chromeConnected : (c.chrome.extensionInstalled == true)
                    if done { t.invalidate(); self.chromePoll = nil; Log.info("Chrome: \(deepWhenReady ? "connected" : "extension installed")", "connect"); return }
                    if deepWhenReady, c.chromeReady, !deepDone, !self.model.connect.checking {
                        deepDone = true
                        self.check(deep: true)
                    }
                }
            }
        }
    }

    // MARK: Terminal

    /// Writes a `.command` script (Terminal opens those) and launches it.
    /// `MAC_GUIDED_NO_OPEN=1` only writes the file (tests).
    @discardableResult
    func runInTerminal(name: String, title: String, lines: [String]) -> URL? {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Mac", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("\(name).command")
        var script = "#!/bin/bash\n# Generated by Mac.app: \(title)\nprintf '\\033]0;%s\\007' \"\(title)\"\nclear\n"
        script += lines.joined(separator: "\n") + "\n"
        script += "echo; echo 'Done. You can close this window and go back to Mac.'\n"
        do {
            try script.write(to: file, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: file.path)
        } catch {
            Log.error("could not write \(file.path): \(error)", "connect")
            return nil
        }
        if Paths.flag("MAC_GUIDED_NO_OPEN") { Log.info("guided action written (not opened): \(file.path)", "connect"); return file }
        // Open with Terminal explicitly (a user may have .command bound to an editor).
        if let term = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.Terminal") {
            let cfg = NSWorkspace.OpenConfiguration()
            cfg.activates = true
            NSWorkspace.shared.open([file], withApplicationAt: term, configuration: cfg) { _, err in
                if let err { Log.warn("Terminal open failed: \(err)", "connect") }
            }
        } else {
            NSWorkspace.shared.open(file)
        }
        Log.info("guided action: \(title) → \(file.lastPathComponent)", "connect")
        return file
    }
}
