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

    /// Chrome is usable for browser tasks (what the "Connect Chrome…" poll waits for).
    var chromeReady: Bool {
        chrome.nativeHost == true && chrome.extensionInstalled == true && chrome.enabledByDefault != false
    }

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

    init(control: ControlClient, model: AppModel) {
        self.control = control
        self.model = model
    }

    // MARK: checks

    /// Static part (cheap): run on connect and from the status poll. `deep` is what the
    /// "Check connection" button does (daemon talks to the CLI / extension).
    func check(deep: Bool, completion: (@MainActor (ConnectCheck?) -> Void)? = nil) {
        if deep { model.connect.checking = true }
        let sent = control.send("connect_check", ["deep": deep]) { [weak self] ok, data, err in
            guard let self else { return }
            self.model.connect.checking = false
            guard ok else {
                Log.warn("connect_check failed: \(err ?? "?")", "connect")
                completion?(nil); return
            }
            var c = ConnectCheck.parse(data)
            c.checking = false
            self.model.connect = c
            Log.info("connect_check deep=\(deep): claude signed_in=\(c.claude.signedIn.map(String.init) ?? "?") chrome ready=\(c.chromeReady) ready=\(c.ready.map(String.init) ?? "?")", "connect")
            completion?(c)
        }
        if !sent { model.connect.checking = false; completion?(nil) }
    }

    /// "Check connection" button.
    func checkConnection() { check(deep: true) }

    // MARK: guided fixes

    /// "Install Claude Code…": Terminal runs the official installer, then `claude` to sign in.
    func installClaude() {
        runInTerminal(name: "install-claude-code", title: "Install Claude Code", lines: [
            "echo 'Installing Claude Code (official installer)…'",
            Self.claudeInstallCmd,
            "export PATH=\"$HOME/.local/bin:$PATH\"",
            "echo; echo 'Now sign in. A browser window will open; when you are back here, type /exit.'",
            "claude || echo 'claude was not found on PATH — open a new Terminal window and run: claude'",
        ])
    }

    /// "Sign in…": Terminal runs `claude`.
    func signIn() {
        runInTerminal(name: "sign-in-claude", title: "Sign in to Claude", lines: [
            "echo 'Signing in to Claude Code. Follow the prompts; when you are signed in, type /exit.'",
            "export PATH=\"$HOME/.local/bin:$PATH\"",
            "claude",
        ])
    }

    /// "Install extension…": Web Store page, in Chrome explicitly when Chrome isn't the default.
    func installExtension() {
        let url = Self.extensionURL
        let defaultBrowser = NSWorkspace.shared.urlForApplication(toOpen: url)
        let chrome = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.google.Chrome")
        if let chrome, let d = defaultBrowser, Bundle(url: d)?.bundleIdentifier != "com.google.Chrome" {
            let cfg = NSWorkspace.OpenConfiguration()
            NSWorkspace.shared.open([url], withApplicationAt: chrome, configuration: cfg) { _, err in
                if let err { Log.warn("open in Chrome failed: \(err)", "connect") }
            }
            Log.info("opened Web Store in Chrome explicitly (default browser is \(d.lastPathComponent))", "connect")
        } else {
            NSWorkspace.shared.open(url)
            Log.info("opened Web Store in the default browser (chrome installed: \(chrome != nil))", "connect")
        }
        startChromePolling()
    }

    /// "Connect Chrome…": Terminal runs `claude --chrome` with instructions; the app polls
    /// until the native host + extension are in place, then re-checks deeply.
    func connectChrome() {
        runInTerminal(name: "connect-chrome", title: "Connect Claude to Chrome", lines: [
            "echo '1. Press Enter at the dialog that appears.'",
            "echo '2. Type /chrome and choose \"Enabled by default\".'",
            "echo '3. Type /exit to come back.'; echo",
            "export PATH=\"$HOME/.local/bin:$PATH\"",
            "claude --chrome",
        ])
        startChromePolling()
    }

    private func startChromePolling() {
        chromePoll?.invalidate()
        chromePollStarted = Date()
        chromePoll = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] t in
            MainActor.assumeIsolated {
                guard let self else { t.invalidate(); return }
                if Date().timeIntervalSince(self.chromePollStarted ?? Date()) > 300 { t.invalidate(); self.chromePoll = nil; return }
                self.check(deep: false) { c in
                    guard let c, c.chromeReady else { return }
                    t.invalidate(); self.chromePoll = nil
                    Log.info("Chrome extension + native host detected; deep re-check", "connect")
                    self.check(deep: true)
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
        var script = "#!/bin/bash\n# Generated by Mac.app — \(title)\nprintf '\\033]0;%s\\007' \"\(title)\"\nclear\n"
        script += lines.joined(separator: "\n") + "\n"
        script += "echo; echo 'Done — you can close this window and go back to Mac.'\n"
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
