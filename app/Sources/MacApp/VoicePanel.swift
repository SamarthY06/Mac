import AVFoundation
import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// "Voice…": give Mac a cloned voice from a 15 s recording or an audio file.
/// Sends `voice_clone {path}` / `voice_reset`; shows the current engine from `status.voice_engine`.
@MainActor
final class VoicePanel {
    private var window: NSWindow?
    private let model: AppModel
    private let control: ControlClient
    private let state: VoiceState

    init(model: AppModel, control: ControlClient) {
        self.model = model
        self.control = control
        self.state = VoiceState(model: model, control: control)
    }

    func show() {
        state.refreshEngine()
        if window == nil {
            let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 360, height: 560),
                             styleMask: [.titled, .closable, .fullSizeContentView],
                             backing: .buffered, defer: false)
            w.title = "Voice"
            w.titlebarAppearsTransparent = true
            w.titleVisibility = .hidden
            w.isMovableByWindowBackground = true
            w.isReleasedWhenClosed = false
            w.level = .floating
            w.setAccessibilityLabel("Give Mac a voice")
            let hc = NSHostingController(rootView: VoiceView(state: state, model: model, close: { [weak w] in w?.close() }))
            hc.sizingOptions = [.preferredContentSize]
            w.contentViewController = hc
            w.center()
            window = w
        }
        NSApp.activate(ignoringOtherApps: true)
        window?.makeKeyAndOrderFront(nil)
    }

    /// Debug/screenshot hook (`MAC_DEBUG_VOICE_FILE`): preselect a sample, consent, and clone.
    func debugClone(file: String) {
        state.select(URL(fileURLWithPath: file))
        state.consent = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
            MainActor.assumeIsolated { self?.state.useThisVoice() }
        }
    }
}

/// Recorder + command state for the panel. Main-actor only.
@MainActor
final class VoiceState: NSObject, ObservableObject, AVAudioRecorderDelegate {
    enum Phase: Equatable { case idle, recording, cloning, done(String), failed(String) }

    static let maxSeconds = 15.0
    static let dir = Paths.home.appendingPathComponent("voice")

    @Published var phase: Phase = .idle
    @Published var sample: URL?
    @Published var consent = false
    @Published var level: Float = 0          // 0…1 input level while recording
    @Published var remaining = maxSeconds    // countdown while recording

    private let model: AppModel
    private let control: ControlClient
    private var recorder: AVAudioRecorder?
    private var meter: Timer?

    init(model: AppModel, control: ControlClient) {
        self.model = model
        self.control = control
        super.init()
    }

    // MARK: current engine

    /// Asks the daemon for `status` and reads `voice_engine` (the panel owns this refresh).
    func refreshEngine() {
        control.send("status") { [weak self] ok, data, _ in
            guard ok, let self else { return }
            self.model.setVoiceEngine(data["voice_engine"], sample: data["clone_sample"])
        }
    }

    // MARK: recording

    var isRecording: Bool { phase == .recording }

    func toggleRecording() {
        isRecording ? stopRecording() : startRecording()
    }

    private func startRecording() {
        do {
            try FileManager.default.createDirectory(at: Self.dir, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
        } catch {
            phase = .failed("Couldn’t create \(Self.dir.path): \(error.localizedDescription)")
            return
        }
        let stamp = Self.stamp()
        // Preferred: 24 kHz mono 16-bit WAV (what the TTS cloner wants); fallback: AAC .m4a.
        let wav = Self.dir.appendingPathComponent("recorded-\(stamp).wav")
        let wavSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 24_000, AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false,
        ]
        let m4a = Self.dir.appendingPathComponent("recorded-\(stamp).m4a")
        let m4aSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 24_000, AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
        ]
        var rec: AVAudioRecorder?
        for (url, settings) in [(wav, wavSettings), (m4a, m4aSettings)] {
            if let r = try? AVAudioRecorder(url: url, settings: settings), r.prepareToRecord() { rec = r; break }
            try? FileManager.default.removeItem(at: url)
        }
        guard let rec else {
            phase = .failed("Couldn’t start recording. Check Microphone under Permissions, or choose a file instead.")
            return
        }
        rec.delegate = self
        rec.isMeteringEnabled = true
        guard rec.record(forDuration: Self.maxSeconds) else {
            phase = .failed("Couldn’t start recording. Is another app using the microphone?")
            return
        }
        recorder = rec
        sample = nil
        remaining = Self.maxSeconds
        level = 0
        phase = .recording
        Log.info("voice: recording to \(rec.url.lastPathComponent)", "ui")
        meter = Timer.scheduledTimer(withTimeInterval: 1.0 / 30, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
    }

    private func tick() {
        guard let rec = recorder, rec.isRecording else { return }
        rec.updateMeters()
        let db = rec.averagePower(forChannel: 0)            // -160…0 dB
        let lin = max(0, min(1, (db + 50) / 50))             // -50 dB → 0, 0 dB → 1
        level = level * 0.6 + lin * 0.4                      // light smoothing
        remaining = max(0, Self.maxSeconds - rec.currentTime)
    }

    func stopRecording() {
        recorder?.stop()   // delegate finishes the state
    }

    nonisolated func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        let url = recorder.url
        DispatchQueue.main.async {
            MainActor.assumeIsolated { self.finishRecording(url: url, ok: flag) }
        }
    }

    nonisolated func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: Error?) {
        let msg = error?.localizedDescription ?? "unknown error"
        DispatchQueue.main.async {
            MainActor.assumeIsolated { self.phase = .failed("Recording failed: \(msg)") }
        }
    }

    private func finishRecording(url: URL, ok: Bool) {
        meter?.invalidate(); meter = nil
        recorder = nil
        level = 0
        let seconds = Self.maxSeconds - remaining
        guard ok, seconds >= 2 else {
            try? FileManager.default.removeItem(at: url)
            phase = .failed(ok ? "That was too short — record at least a few seconds." : "Recording didn’t finish. Try again.")
            return
        }
        sample = url
        phase = .idle
        Log.info("voice: recorded \(url.lastPathComponent) (\(Int(seconds)) s)", "ui")
    }

    // MARK: choose file

    func chooseFile() {
        let p = NSOpenPanel()
        p.allowedContentTypes = [.audio]
        p.allowsMultipleSelection = false
        p.canChooseDirectories = false
        p.message = "Choose a short, clean recording of the voice (5–30 seconds works best)."
        p.prompt = "Choose"
        guard p.runModal() == .OK, let url = p.url else { return }
        select(url)
    }

    /// Copies a chosen file into ~/.mac-agent/voice so the daemon always reads a stable path.
    func select(_ url: URL) {
        do {
            try FileManager.default.createDirectory(at: Self.dir, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
            let dest = Self.dir.appendingPathComponent("chosen-\(Self.stamp()).\(url.pathExtension.isEmpty ? "audio" : url.pathExtension)")
            if url.deletingLastPathComponent().standardizedFileURL != Self.dir.standardizedFileURL {
                try? FileManager.default.removeItem(at: dest)
                try FileManager.default.copyItem(at: url, to: dest)
                sample = dest
            } else {
                sample = url
            }
            phase = .idle
        } catch {
            phase = .failed("Couldn’t use that file: \(error.localizedDescription)")
        }
    }

    func clearSample() {
        sample = nil
        if case .done = phase { phase = .idle }
        if case .failed = phase { phase = .idle }
    }

    // MARK: commands

    var canUse: Bool { sample != nil && consent && phase != .cloning && !isRecording }

    func useThisVoice() {
        guard let sample, canUse else { return }
        phase = .cloning
        let sent = control.send("voice_clone", ["path": sample.path]) { [weak self] ok, data, err in
            guard let self else { return }
            if ok {
                self.model.setVoiceEngine(data["voice_engine"] ?? data["engine"] ?? "clone",
                                          sample: data["clone_sample"] ?? sample.path)
                self.phase = .done("Done — say “Hey Mac” to hear it.")
            } else {
                self.phase = .failed(err ?? "Couldn’t clone that voice.")
            }
        }
        if !sent { phase = .failed("Mac isn’t connected yet. Try again in a moment.") }
    }

    func useDefaultVoice() {
        phase = .cloning
        let sent = control.send("voice_reset") { [weak self] ok, data, err in
            guard let self else { return }
            if ok {
                self.model.setVoiceEngine(data["voice_engine"] ?? "kokoro")
                self.phase = .done("Back to the default voice.")
            } else {
                self.phase = .failed(err ?? "Couldn’t switch back.")
            }
        }
        if !sent { phase = .failed("Mac isn’t connected yet. Try again in a moment.") }
    }

    private static func stamp() -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd-HHmmss"
        return f.string(from: Date())
    }
}

struct VoiceView: View {
    @ObservedObject var state: VoiceState
    @ObservedObject var model: AppModel
    var close: () -> Void
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        VStack(spacing: 14) {
            hero
            recordCard
            if let s = state.sample { sampleRow(s) }
            consentRow
            statusRow
            footer
        }
        .padding(.top, 30).padding(.horizontal, 22).padding(.bottom, 18)
        .frame(width: 380)
        .background(Color(nsColor: .windowBackgroundColor))
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.18), value: state.phase)
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.18), value: state.sample)
    }

    // MARK: hero

    private var hero: some View {
        VStack(spacing: 6) {
            Image(systemName: "waveform.and.mic")
                .font(.system(size: 26, weight: .medium)).foregroundStyle(Color.accentColor)
                .accessibilityHidden(true)
            Text("Give Mac a voice").font(.title3.weight(.semibold))
            Text("Record 15 seconds, or choose a file. Mac will answer in that voice.")
                .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            Text("Now speaking with: \(model.voiceEngineLabel)")
                .font(.footnote).foregroundStyle(.tertiary)
                .accessibilityLabel("Now speaking with \(model.voiceEngineLabel)")
        }
        .accessibilityElement(children: .combine)
    }

    // MARK: record

    private var recordCard: some View {
        VStack(spacing: 10) {
            HStack(spacing: 16) {
                recordButton
                VStack(alignment: .leading, spacing: 6) {
                    Text(state.isRecording ? "Recording… speak naturally" : "Record a sample")
                        .font(.body.weight(.medium))
                    Text(state.isRecording ? countdown : "Up to 15 seconds. Read a few sentences in a quiet room.")
                        .font(.callout).foregroundStyle(.secondary)
                        .monospacedDigit()
                        .fixedSize(horizontal: false, vertical: true)
                    meterView
                }
                Spacer(minLength: 0)
            }
            HStack {
                Text("or").font(.footnote).foregroundStyle(.tertiary)
                Button("Choose File…", action: state.chooseFile)
                    .disabled(state.isRecording || state.phase == .cloning)
                    .accessibilityLabel("Choose an audio file")
                Spacer()
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Color.primary.opacity(0.05)))
    }

    private var countdown: String {
        let s = Int(ceil(state.remaining))
        return String(format: "0:%02d left", s)
    }

    private var recordButton: some View {
        Button(action: state.toggleRecording) {
            ZStack {
                Circle().fill(Color.red.opacity(state.isRecording ? 0.18 : 0.12)).frame(width: 64, height: 64)
                Circle().strokeBorder(Color.red.opacity(0.5), lineWidth: 2).frame(width: 64, height: 64)
                if state.isRecording {
                    RoundedRectangle(cornerRadius: 4, style: .continuous).fill(Color.red).frame(width: 22, height: 22)
                } else {
                    Circle().fill(Color.red).frame(width: 28, height: 28)
                }
            }
        }
        .buttonStyle(.plain)
        .disabled(state.phase == .cloning)
        .accessibilityLabel(state.isRecording ? "Stop recording" : "Start recording, up to fifteen seconds")
        .keyboardShortcut(state.isRecording ? .cancelAction : .init("r", modifiers: .command))
    }

    /// 20-segment input level meter (data-driven; no animation under Reduce Motion).
    private var meterView: some View {
        let lit = Int((state.level * 20).rounded())
        return HStack(spacing: 2) {
            ForEach(0..<20, id: \.self) { i in
                RoundedRectangle(cornerRadius: 1.5)
                    .fill(i < lit ? (i >= 17 ? Color.red : i >= 13 ? Color.orange : Color.accentColor)
                                  : Color.primary.opacity(0.1))
                    .frame(width: 8, height: 8)
            }
        }
        .opacity(state.isRecording ? 1 : 0.35)
        .animation(reduceMotion ? nil : .linear(duration: 0.05), value: lit)
        .accessibilityLabel("Input level")
        .accessibilityValue("\(Int(state.level * 100)) percent")
    }

    // MARK: sample

    private func sampleRow(_ url: URL) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "waveform").foregroundStyle(Color.accentColor).accessibilityHidden(true)
            Text(url.lastPathComponent).font(.callout).lineLimit(1).truncationMode(.middle)
            Spacer()
            Button("Clear", action: state.clearSample)
                .controlSize(.small)
                .disabled(state.phase == .cloning)
                .accessibilityLabel("Clear the selected sample")
        }
        .padding(.horizontal, 4)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Selected sample \(url.lastPathComponent)")
    }

    // MARK: consent + privacy

    private var consentRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            Toggle(isOn: $state.consent) {
                Text("This is my own voice, or I have permission to use it")
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .toggleStyle(.checkbox)
            .disabled(state.phase == .cloning)
            Label("The sample never leaves this Mac.", systemImage: "lock.fill")
                .font(.footnote).foregroundStyle(.secondary)
                .padding(.leading, 2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: status

    @ViewBuilder private var statusRow: some View {
        switch state.phase {
        case .cloning:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Cloning… about 10 seconds").font(.callout).foregroundStyle(.secondary)
            }
            .accessibilityElement(children: .combine)
        case .done(let msg):
            Label(msg, systemImage: "checkmark.circle.fill")
                .font(.callout).foregroundStyle(.green)
                .symbolEffect(.bounce, value: state.phase)
        case .failed(let msg):
            Label(msg, systemImage: "exclamationmark.triangle.fill")
                .font(.callout).foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        default:
            EmptyView()
        }
    }

    // MARK: footer

    private var footer: some View {
        HStack {
            Button("Use Default Voice", action: state.useDefaultVoice)
                .disabled(state.phase == .cloning || state.isRecording)
                .accessibilityLabel("Switch back to the default voice")
            Spacer()
            Button("Done", action: close)
            Button("Use This Voice", action: state.useThisVoice)
                .keyboardShortcut(.defaultAction)
                .disabled(!state.canUse)
                .accessibilityHint(state.consent ? "" : "Tick the consent checkbox first")
        }
    }
}
