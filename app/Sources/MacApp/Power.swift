import AppKit
import Combine
import CoreGraphics
import Foundation
import IOKit.ps
import IOKit.pwr_mgt

/// Sleep/wake handling and power assertions.
///
/// CPU/battery reasoning:
/// - App Nap: macOS may throttle timers/IO of a background (LSUIElement) app it thinks is
///   idle. Our audio path is driven by CoreAudio callbacks (not napped), but the socket
///   reader thread and the supervised daemon's responsiveness matter for wake-word
///   latency, so we hold `.userInitiatedAllowingIdleSystemSleep` for the whole run. That
///   disables App Nap only; it does NOT keep the Mac awake and costs nothing when idle.
/// - While a turn is active (daemon state ≠ idle) we additionally hold
///   `.idleSystemSleepDisabled + .userInitiated + .latencyCritical`, so a long agent task
///   isn't cut off by idle sleep and timers aren't coalesced. Released as soon as the
///   daemon reports idle, so an idle Mac still sleeps normally.
/// STAY AWAKE (setting `stayAwake`, UserDefaults; UI: Settings › General › Behavior):
///   "auto" (default) = keep the Mac from idle-sleeping only while on AC power,
///   "always" = also on battery, "never" = only during a turn (the per-turn activity below).
///   Implemented as an IOPM `PreventUserIdleSystemSleep` assertion held while listening is
///   ON (not off; muted still counts as on): the DISPLAY may still sleep/lock, Mac keeps
///   hearing. A closed MacBook lid still sleeps the Mac unless an external display and
///   power are attached (clamshell mode); no app can prevent that.
@MainActor
final class Power {
    private let audio: AudioEngine
    private let control: ControlClient
    private let model: AppModel
    private var cancellables: Set<AnyCancellable> = []
    private var awakeAssertion: IOPMAssertionID = 0
    private var powerSource: CFRunLoopSource?
    private(set) var onAC = true
    private var screenLocked = false
    private var aecOffTimer: Timer?
    private var lastState: AgentState = .unknown
    private var noNap: NSObjectProtocol?
    private var turnActivity: NSObjectProtocol?
    private var observers: [NSObjectProtocol] = []

    init(audio: AudioEngine, control: ControlClient, model: AppModel) {
        self.audio = audio
        self.control = control
        self.model = model
    }

    static var stayAwakeSetting: String {
        let v = UserDefaults.standard.string(forKey: "stayAwake") ?? "auto"
        return ["auto", "always", "never"].contains(v) ? v : "auto"
    }

    func start() {
        noNap = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiatedAllowingIdleSystemSleep],
            reason: "Mac listens for its wake word")
        ProcessInfo.processInfo.disableAutomaticTermination("Mac voice agent is running")
        ProcessInfo.processInfo.disableSuddenTermination()  // we must stop the child cleanly

        let nc = NSWorkspace.shared.notificationCenter
        func on(_ name: Notification.Name, _ body: @escaping @MainActor () -> Void) {
            observers.append(nc.addObserver(forName: name, object: nil, queue: .main) { _ in
                MainActor.assumeIsolated { body() }
            })
        }
        on(NSWorkspace.willSleepNotification) { [weak self] in
            guard let self else { return }
            Log.info("system will sleep", "power")
            self.control.send("power", ["event": "will_sleep"])
            self.audio.stop(reason: "system sleep")
        }
        on(NSWorkspace.didWakeNotification) { [weak self] in
            guard let self else { return }
            Log.info("system did wake", "power")
            self.control.send("power", ["event": "did_wake"])
            // Give CoreAudio a moment to re-enumerate devices after wake.
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                self.audio.restart(reason: "system wake")
            }
        }
        on(NSWorkspace.screensDidSleepNotification) { Log.info("screens did sleep", "power") }
        on(NSWorkspace.screensDidWakeNotification) { Log.info("screens did wake", "power") }
        // Fast user switching: our session goes to the background.
        on(NSWorkspace.sessionDidResignActiveNotification) { [weak self] in self?.screenLock(true, via: "session resigned active") }
        on(NSWorkspace.sessionDidBecomeActiveNotification) { [weak self] in self?.screenLock(false, via: "session became active") }

        // Screen lock / unlock (login window over the session).
        let dnc = DistributedNotificationCenter.default()
        observers.append(dnc.addObserver(forName: Notification.Name("com.apple.screenIsLocked"), object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.screenLock(true, via: "com.apple.screenIsLocked") }
        })
        observers.append(dnc.addObserver(forName: Notification.Name("com.apple.screenIsUnlocked"), object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.screenLock(false, via: "com.apple.screenIsUnlocked") }
        })
        screenLocked = Self.sessionScreenLocked()
        model.screenLocked = screenLocked

        // Stay-awake: power source changes, setting changes, listening on/off.
        onAC = Self.providingAC()
        let ctx = Unmanaged.passUnretained(self).toOpaque()
        if let src = IOPSNotificationCreateRunLoopSource({ ctx in
            guard let ctx else { return }
            let me = Unmanaged<Power>.fromOpaque(ctx).takeUnretainedValue()
            DispatchQueue.main.async { MainActor.assumeIsolated { me.powerSourceChanged() } }
        }, ctx)?.takeRetainedValue() {
            powerSource = src
            CFRunLoopAddSource(CFRunLoopGetMain(), src, .commonModes)
        }
        NotificationCenter.default.addObserver(forName: AppPrefs.changed, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.refreshStayAwake(reason: "setting changed") }
        }
        model.$off.removeDuplicates().sink { [weak self] off in
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.refreshStayAwake(reason: off ? "turned off" : "listening on") } }
        }.store(in: &cancellables)
        refreshStayAwake(reason: "launch")
    }

    // MARK: stay awake

    private static func providingAC() -> Bool {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() as String? else { return true }
        return type == kIOPMACPowerKey  // "AC Power"
    }

    private func powerSourceChanged() {
        let ac = Self.providingAC()
        guard ac != onAC else { return }
        onAC = ac
        Log.info("power source: \(ac ? "AC" : "battery")", "power")
        refreshStayAwake(reason: ac ? "plugged in" : "on battery")
    }

    /// Decide whether to hold the PreventUserIdleSystemSleep assertion.
    func refreshStayAwake(reason: String) {
        let setting = Self.stayAwakeSetting
        let listening = !model.off
        let want: Bool
        switch setting {
        case "always": want = listening
        case "never": want = false
        default: want = listening && onAC
        }
        if want && awakeAssertion == 0 {
            var id: IOPMAssertionID = 0
            let st = IOPMAssertionCreateWithName(kIOPMAssertionTypePreventUserIdleSystemSleep as CFString,
                                                 IOPMAssertionLevel(kIOPMAssertionLevelOn),
                                                 "Mac is listening for “Hey Mac”" as CFString, &id)
            if st == kIOReturnSuccess { awakeAssertion = id } else { Log.warn("IOPM assertion failed: \(st)", "power") }
            Log.info("stay awake ON (\(reason); setting=\(setting), \(onAC ? "AC" : "battery"), display may still sleep)", "power")
        } else if !want && awakeAssertion != 0 {
            IOPMAssertionRelease(awakeAssertion)
            awakeAssertion = 0
            Log.info("stay awake OFF (\(reason); setting=\(setting), \(onAC ? "AC" : "battery"))", "power")
        }
    }

    // MARK: screen lock

    static func sessionScreenLocked() -> Bool {
        guard let d = CGSessionCopyCurrentDictionary() as? [String: Any] else { return false }
        if let b = d["CGSSessionScreenIsLocked"] as? Bool { return b }
        if let n = d["CGSSessionScreenIsLocked"] as? Int { return n != 0 }
        return false
    }

    private func screenLock(_ locked: Bool, via: String) {
        guard locked != screenLocked else { return }
        screenLocked = locked
        model.screenLocked = locked
        Log.info("screen \(locked ? "locked" : "unlocked") (\(via))", "power")
        control.send("power", ["event": locked ? "screen_locked" : "screen_unlocked"])
    }

    /// Daemon (re)connected: tell it the current lock state so it never starts out wrong.
    func daemonConnected() {
        screenLocked = Self.sessionScreenLocked()
        model.screenLocked = screenLocked
        control.send("power", ["event": screenLocked ? "screen_locked" : "screen_unlocked"])
    }

    /// Called with every daemon state change.
    func stateChanged(_ state: AgentState) {
        // Voice processing (AEC) while Mac may speak: on from `thinking` onwards, kept through
        // FOLLOWUP and for 60 s after the turn ends (people follow up), then off. Never toggled
        // at `listening` (a rebuild there would eat the words after "Hey Mac"). Each toggle is
        // an engine rebuild, so this keeps rebuilds to about one per conversation.
        switch state {
        case .thinking, .speaking, .confirming, .followup:
            aecOffTimer?.invalidate(); aecOffTimer = nil
            audio.setVoiceProcessingWanted(true)
        case .idle:
            if aecOffTimer == nil {
                // 60 s of quiet before dropping AEC (env MAC_AEC_IDLE_S overrides; e2e uses 3).
                let delay = Double(Paths.env["MAC_AEC_IDLE_S"] ?? "") ?? 60
                aecOffTimer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
                    MainActor.assumeIsolated {
                        guard let self else { return }
                        self.aecOffTimer = nil
                        if self.lastState == .idle { self.audio.setVoiceProcessingWanted(false) }
                    }
                }
            }
        case .off, .unknown, .error:
            aecOffTimer?.invalidate(); aecOffTimer = nil
            audio.setVoiceProcessingWanted(false)
        case .listening: break
        }
        lastState = state
        let active = ![.idle, .unknown, .error, .off].contains(state)
        if active, turnActivity == nil {
            turnActivity = ProcessInfo.processInfo.beginActivity(
                options: [.idleSystemSleepDisabled, .userInitiated, .latencyCritical],
                reason: "Mac is handling a request")
        } else if !active, let a = turnActivity {
            ProcessInfo.processInfo.endActivity(a)
            turnActivity = nil
        }
    }
}
