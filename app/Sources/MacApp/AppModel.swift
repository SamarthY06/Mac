import AppKit
import Foundation
import ServiceManagement
import SwiftUI

enum AgentState: String {
    case idle, listening, thinking, speaking, confirming, followup, error
    case off      // daemon switched off ("Turn Off")
    case unknown  // daemon not connected

    /// Short status line (menu header, HUD caption). See docs/DESIGN.md "State vocabulary".
    var label: String {
        switch self {
        case .idle: return "Listening for “Hey Mac”"
        case .listening: return "Listening…"
        case .thinking: return "Thinking…"
        case .speaking: return "Speaking"
        case .confirming: return "Yes or no?"
        case .followup: return "Go on, I’m listening"
        case .error: return "Something went wrong"
        case .off: return "Mac is off"
        case .unknown: return "Connecting to Mac…"
        }
    }

    /// One SF Symbol per state (template-rendered in the menu bar, tinted in the HUD).
    var glyph: String {
        switch self {
        case .idle: return "waveform.circle"
        case .listening, .followup: return "waveform"
        case .thinking: return "ellipsis.circle"
        case .speaking: return "speaker.wave.2.fill"
        case .confirming: return "questionmark.circle"
        case .error: return "exclamationmark.triangle"
        case .off: return "power"
        case .unknown: return "waveform.badge.exclamationmark"
        }
    }

    /// Ongoing activity → continuous motion (listening ripple, thinking pulse, speaking waves).
    var isActive: Bool { [.listening, .followup, .thinking, .speaking].contains(self) }

    /// A turn is in progress (the HUD should stay up).
    var inTurn: Bool { [.listening, .thinking, .speaking, .confirming, .followup, .error].contains(self) }
}

/// A phone (LiveKit) session, ARCHITECTURE §10 / review N7.
struct RemoteSession: Equatable {
    var active = false     // a room exists (link minted) until "End Phone Session" / auto-leave
    var present = false    // the phone actually joined
    var verified = false   // the caller said the PIN
    var pin = ""           // 4 digits the caller must say first (shown while !verified)

    /// Banner text (HUD + menu).
    var banner: String {
        if !present { return "Phone link ready — waiting for your phone" }
        if !verified, !pin.isEmpty { return "Phone connected · PIN \(pin)" }
        return "Phone connected"
    }
}

/// UI state, fed by control.sock events. Main-actor only.
/// Protocol schema: daemon/src/macd/control.py docstring + orchestrator.publish calls.
@MainActor
final class AppModel: ObservableObject {
    @Published var state: AgentState = .unknown
    @Published var muted = false
    /// The single source of truth for "off" (simplicity #33): set from the daemon's state
    /// event when connected, locally (AppDelegate.setPower) when not.
    @Published var off = false
    @Published var connected = false
    @Published var userText = ""
    @Published var assistantText = ""
    @Published var activity = ""
    @Published var errorText = ""
    @Published var errorKind = ""   // auth | rate_limit | network | cli_missing | timeout | internal
    @Published var userFromPhone = false   // the current user line came from the phone (N7 mirror)
    @Published var lastTranscript = ""
    @Published var lastReply = ""
    @Published var inputDevice = "—"
    @Published var aec = false
    @Published var audioWarning: String?
    @Published var daemonStatus = ""
    @Published var remote = RemoteSession()
    // From the daemon's `status` reply (AppDelegate polls; UX#8/#9/#11):
    @Published var brainAuth = ""            // "ok" | "signed_out" | "missing" | "api_key" | "" unknown
    @Published var chromePaired: Bool?       // Claude in Chrome connected (nil = unknown)
    @Published var voiceUnavailable: String? // reason when speech models are missing/failed, else nil
    @Published var inputSwitchedNote: String?  // one-time line when the mic was switched (N5/UX#11)
    @Published var voiceReason = ""            // `voice_reason` from every state event, shown whenever non-empty
                                               // (e.g. "Phone session — local mic paused" while a phone is present)
    @Published var onboardingVisible = false   // AppDelegate polls status faster while true
    @Published var voiceEngine = ""            // "kokoro" or "clone:<sample>" (from status.voice_engine)
    @Published var statusItemHidden = false    // menu bar icon hidden by the notch / overflow (AppDelegate sets)

    var onStateChange: ((AgentState) -> Void)?
    /// While an "off" command is queued for a disconnected daemon, state events must not flip
    /// `off` back (AppDelegate sets this).
    var holdOff = false

    /// B2's shape: `status.voice_engine` is "kokoro" | "clone" and `status.clone_sample` is the
    /// sample's full path (or null); command replies also carry `voice_engine: {engine, sample}`.
    func setVoiceEngine(_ v: Any?, sample: Any? = nil) {
        var engine = ""
        var name = ""
        if let d = v as? [String: Any] {
            engine = (d["engine"] as? String ?? "").lowercased()
            name = d["sample"] as? String ?? ""
        } else if let s = v as? String {
            let parts = s.split(separator: ":", maxSplits: 1).map(String.init)
            engine = (parts.first ?? "").lowercased()
            if parts.count > 1 { name = parts[1] }
        }
        if let p = sample as? String, !p.isEmpty { name = (p as NSString).lastPathComponent }
        guard !engine.isEmpty else { return }
        voiceEngine = engine == "clone" || engine == "cloned" ? "clone:\(name)" : engine
    }

    /// "Default (Kokoro)" / "Cloned from recorded-2026-11-07.wav".
    var voiceEngineLabel: String {
        if voiceEngine.hasPrefix("clone") {
            let sample = voiceEngine.split(separator: ":", maxSplits: 1).dropFirst().first.map(String.init) ?? ""
            return sample.isEmpty ? "Cloned voice" : "Cloned from \(sample)"
        }
        return voiceEngine.isEmpty || voiceEngine == "kokoro" ? "Default (Kokoro)" : "Default (\(voiceEngine))"
    }

    // MARK: derived copy

    /// The symbol that represents the current situation, mute/off taking precedence.
    var glyph: String {
        if off { return AgentState.off.glyph }
        if muted { return "mic.slash" }
        if remote.active { return state.inTurn ? state.glyph : "iphone.radiowaves.left.and.right" }
        if voiceUnavailable != nil { return "waveform.badge.exclamationmark" }
        return state.glyph
    }

    /// While a phone is present the daemon sends `voice_reason` on every state event (B2);
    /// fallback copy if it is empty.
    var phoneSessionLine: String {
        voiceReason.isEmpty ? "Phone session — local mic paused" : voiceReason
    }

    /// One-line status for the menu header and HUD caption.
    var statusLine: String {
        if off { return AgentState.off.label }
        if muted { return "Muted — Mac can’t hear you" }
        if let why = voiceUnavailable {
            return why.isEmpty ? "Can’t hear or speak — speech models missing" : "Voice unavailable — \(why)"
        }
        if remote.active, !state.inTurn { return phoneSessionLine }
        if !voiceReason.isEmpty { return voiceReason }
        return state.label
    }

    /// Secondary status line under the header (only when something needs attention).
    var statusDetail: String? {
        if off { return "Choose Turn On to start listening again." }
        if muted { return "Choose Mute again to resume." }
        if !connected { return daemonStatus.isEmpty ? nil : daemonStatus }
        if remote.active { return state.inTurn ? phoneSessionLine : "Choose End Phone Session to resume." }
        if voiceUnavailable != nil { return "Re-run install.sh (it downloads the speech models)." }
        if brainAuth == "signed_out" { return "Not signed in to Claude — choose Sign in to Claude…" }
        if brainAuth == "missing" { return "Claude Code isn’t installed — choose Install Claude Code…" }
        if let w = audioWarning, !w.isEmpty { return w }
        if state == .error, !errorText.isEmpty { return errorText }
        return nil
    }

    /// What the person can do about the current error, keyed on the daemon's `kind`
    /// (brain/errors.py; UX#4, decision 14).
    var errorAdvice: String {
        switch errorKind {
        case "auth":
            return brainAuth == "api_key"
                ? "Add your API key, then choose Troubleshoot ▸ Restart Daemon:"
                : "Open Terminal and sign in, then ask again:"
        case "rate_limit":
            if Self.resetTime(in: errorText) != nil {
                return "Wait until then, or switch to an API key in the settings file."
            }
            return "Wait a while, or switch to an API key in the settings file."
        case "network": return "Check your internet connection and try again."
        case "cli_missing": return "Install Claude Code (claude.com/claude-code), then Troubleshoot ▸ Restart Daemon."
        case "timeout": return "Try a smaller request."
        default: return "Try again. If it keeps happening, choose Open Logs."
        }
    }

    /// A command to show (monospaced) under the advice, when there is one.
    var errorCommand: String? {
        guard errorKind == "auth" else { return nil }
        return brainAuth == "api_key" ? "macctl secret set ANTHROPIC_API_KEY" : "claude"
    }

    /// "Try again at 3 PM." → "3 PM" (the daemon already formatted the reset time).
    private static func resetTime(in text: String) -> String? {
        let re = try! NSRegularExpression(pattern: #"\bat ([0-9]{1,2}(?::[0-9]{2})? ?(?:AM|PM|am|pm))"#)
        guard let m = re.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
              let r = Range(m.range(at: 1), in: text) else { return nil }
        return String(text[r])
    }

    // MARK: updates

    func setConnected(_ c: Bool) {
        connected = c
        if !c { setState(.unknown) }
    }

    func setState(_ s: AgentState) {
        if s != .unknown, !holdOff { off = (s == .off) }
        guard s != state else { return }
        let prev = state
        state = s
        Log.info("state \(prev.rawValue) → \(s.rawValue)", "ui")
        if s == .listening && (prev == .idle || prev == .followup || prev == .unknown) {
            clearExchange()
        }
        if s == .idle || s == .followup { activity = "" }
        onStateChange?(s)
    }

    func clearExchange() {
        userText = ""; assistantText = ""; activity = ""; errorText = ""; errorKind = ""
    }

    func setMuted(_ m: Bool) {
        guard m != muted else { return }
        muted = m
        if !m { clearExchange() }  // UX#20: no stale transcript after unmute
    }

    func handle(event e: [String: Any]) {
        guard let kind = e["event"] as? String else { return }
        switch kind {
        case "state":
            if let m = e["muted"] as? Bool { setMuted(m) }
            if let v = e["voice"] as? Bool {
                let why = (e["voice_reason"] as? String) ?? ""
                voiceUnavailable = v ? nil : (why.isEmpty ? "speech models missing" : why)
                voiceReason = v ? why : ""   // a real voice problem takes precedence (shown above)
            }
            setState(AgentState(rawValue: ((e["state"] as? String) ?? "").lowercased()) ?? .unknown)
        case "transcript":
            let text = e["text"] as? String ?? ""
            let role = (e["role"] as? String ?? "user").lowercased()
            let isDelta = (e["delta"] as? Bool) == true
            if role == "user" {
                userText = isDelta ? userText + text : text
                userFromPhone = (e["remote"] as? Bool) == true
                if (e["final"] as? Bool) ?? true { lastTranscript = userText }
                assistantText = ""
            } else {
                assistantText = isDelta ? assistantText + text : text
                if !assistantText.isEmpty { lastReply = assistantText }
            }
            Log.info("transcript[\(role)] \(text.count) chars", "ui")
        case "activity":
            let k = e["kind"] as? String ?? ""
            if k == "remote" {
                applyRemote(e)
            } else if k == "voice_reset" {
                setVoiceEngine("kokoro")   // spoken "use your default voice"
            } else if let t = e["text"] as? String, !t.isEmpty {
                activity = t  // tool_start narration etc. (UX#6: keep fresh on every tool)
            } else if k == "tool_start", let tool = e["tool"] as? String {
                activity = tool
            }
        case "remote":
            applyRemote(e)
        case "error":
            errorKind = e["kind"] as? String ?? ""
            errorText = e["message"] as? String ?? "Unknown error"
        default:
            break  // metrics etc.
        }
    }

    /// `{"event":"remote","active","present","verified","pin","identity"}` (B2; also fed from the
    /// `status.remote` poll by AppDelegate), or the older activity of kind "remote" with `present`.
    private func applyRemote(_ e: [String: Any]) {
        var r = remote
        if let p = e["present"] as? Bool { r.present = p }
        if let a = e["active"] as? Bool { r.active = a } else if let p = e["present"] as? Bool { r.active = p }
        if let v = e["verified"] as? Bool { r.verified = v }
        if let p = e["pin"] { r.pin = "\(p)" }
        if !r.active { r = RemoteSession() }
        if r != remote { remote = r; Log.info("remote active=\(r.active) present=\(r.present) verified=\(r.verified)", "ui") }
    }

}

/// Shared helpers for motion, text and small system toggles.
enum UX {
    /// System “Reduce Motion” (Accessibility › Display). Continuous effects become static glyphs.
    @MainActor static var reduceMotion: Bool { NSWorkspace.shared.accessibilityDisplayShouldReduceMotion }

    static func truncate(_ s: String, _ n: Int) -> String {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines).replacingOccurrences(of: "\n", with: " ")
        return t.count <= n ? t : String(t.prefix(n - 1)).trimmingCharacters(in: .whitespaces) + "…"
    }

    @MainActor static var launchAtLogin: Bool { SMAppService.mainApp.status == .enabled }

    /// Registers/unregisters the login item; returns an error message for the UI on failure.
    @MainActor static func setLaunchAtLogin(_ on: Bool) -> String? {
        do {
            if on { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
            return nil
        } catch {
            Log.error("launch at login: \(error)", "ui")
            return "Couldn’t change Launch at Login: \(error.localizedDescription) You can also add Mac under System Settings › General › Login Items."
        }
    }

    /// Opens Terminal running `command` without needing Automation permission: Terminal runs
    /// `.command` files directly.
    static func openTerminal(running command: String, name: String) {
        let url = Paths.home.appendingPathComponent("\(name).command")
        let script = "#!/bin/zsh\nclear\necho \"Mac: running \\`\(command)\\`\"\n\(command)\n"
        do {
            try FileManager.default.createDirectory(at: Paths.home, withIntermediateDirectories: true)
            try script.write(to: url, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
            NSWorkspace.shared.open(url)
        } catch {
            Log.error("open terminal: \(error)", "ui")
            NSWorkspace.shared.open(URL(fileURLWithPath: "/System/Applications/Utilities/Terminal.app"))
        }
    }
}
