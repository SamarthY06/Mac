import AppKit
import Foundation

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
@MainActor
final class Power {
    private let audio: AudioEngine
    private let control: ControlClient
    private var noNap: NSObjectProtocol?
    private var turnActivity: NSObjectProtocol?
    private var observers: [NSObjectProtocol] = []

    init(audio: AudioEngine, control: ControlClient) {
        self.audio = audio
        self.control = control
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
    }

    /// Called with every daemon state change.
    func stateChanged(_ state: AgentState) {
        // Voice processing (AEC) only while Mac may speak: on from `thinking` onwards, off
        // again at idle. Not toggled at `listening` — a rebuild there would eat the words
        // spoken right after "Hey Mac". (Measured: VPIO alone = 18 % CPU, plain input 0.5 %.)
        switch state {
        case .thinking, .speaking, .confirming, .followup: audio.setVoiceProcessingWanted(true)
        case .idle, .off, .error, .unknown: audio.setVoiceProcessingWanted(false)
        case .listening: break
        }
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
