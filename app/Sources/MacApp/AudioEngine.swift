import AVFoundation
import AudioToolbox
import CoreAudio
import Foundation
import MacProtocol

/// Owns the audio hardware: mic capture (with Voice Processing IO = AEC/NS/AGC) and TTS
/// playback, bridged to the daemon over `AudioLink`.
///
/// Concurrency: every piece of mutable state below is confined to `queue` (a serial
/// DispatchQueue). AVAudioEngine tap / completion callbacks arrive on CoreAudio threads
/// and immediately hop onto `queue`. Only `helloLock`-guarded `hello` is read elsewhere.
final class AudioEngine: @unchecked Sendable {
    static let micRate: Double = 16_000
    static let micFrameBytes = 640  // 20 ms @ 16 kHz int16 mono

    let queue = DispatchQueue(label: "mac.audio", qos: .userInteractive)
    let link: AudioLink

    // Settings
    // Settings: env overrides, or `defaults write io.github.macagent.Mac <key> -bool NO`.
    var preferVoiceProcessing = !Paths.flag("MAC_NO_AEC")
        && (UserDefaults.standard.object(forKey: "voiceProcessing") as? Bool ?? true)
    var preferBuiltInMic = !Paths.flag("MAC_NO_BUILTIN_PIN")
        && (UserDefaults.standard.object(forKey: "preferBuiltInMic") as? Bool ?? true)

    // Engine (queue-confined)
    private var engine: AVAudioEngine?
    private var player: AVAudioPlayerNode?
    private var playerFormat: AVAudioFormat?
    private var configObserver: NSObjectProtocol?
    private var generation: UInt64 = 0
    private var wantRunning = false
    private var poweredOff = false  // "Turn Off": engine released until turned on
    /// Voice processing (AEC/NS) costs ~18 % of a core by itself (Apple's BNNS model runs
    /// in-process), so it is only enabled while a turn is in progress: see
    /// `setVoiceProcessingWanted`. Idle = plain input ≈ 0.5 %.
    private var vpWanted = false
    private var vpActive = false
    /// PLAY/END that arrive while the engine is being rebuilt (≈0.3–0.5 s): replayed
    /// in order once it is up, so a filler like "Okay." sent at `thinking` isn't dropped.
    private var deferred: [AudioMessage] = []
    private var deferredBytes = 0
    private var rebuildPending = false
    private var recentRebuilds: [Date] = []
    private var vpFailedForDevice: String?
    /// N5(b): the system default input we replaced with the built-in mic (restored on
    /// off / sleep / quit), and the devices we've already told the user about.
    private var replacedDefaultInput: AudioDeviceID?
    private var notedSwitchFrom: Set<AudioDeviceID> = []

    // Mic path
    private var micConverter: AVAudioConverter?
    private var micInFormat: AVAudioFormat?
    private let micOutFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: AudioEngine.micRate,
                                             channels: 1, interleaved: true)!
    private var micAccum = Data()
    private(set) var muted = false
    private var statsTimer: DispatchSourceTimer?

    // Playback streams
    private struct Stream {
        var sampleRate: Double
        var converter: AVAudioConverter?
        var inFormat: AVAudioFormat
        var ended = false
        var pending = 0
    }
    private struct Scheduled { let token: UInt64; let stream: UInt32 }
    private var streams: [UInt32: Stream] = [:]
    private var scheduled: [Scheduled] = []
    private var nextToken: UInt64 = 1
    private var closedStreams = RecentSet(capacity: 512)  // drained or flushed: ignore late PLAY

    // HELLO (read from the socket thread)
    private let helloLock = NSLock()
    private var hello = Hello(input_device: "none", aec: false)

    /// Called on `queue` whenever the engine (re)starts: (inputDeviceName, aec).
    var onStatus: ((String, Bool, String?) -> Void)?
    /// Called on `queue` ONCE per Bluetooth device when its mic was swapped for the
    /// built-in one (text for a notification / HUD line).
    var onInputSwitched: ((String) -> Void)?

    init() {
        link = AudioLink(deliverQueue: queue)
        link.delegate = self
    }

    // MARK: lifecycle

    /// Connect to the daemon's audio socket (HELLO reports input "none" until the
    /// engine runs). Safe to call before mic permission is settled.
    func startLink() {
        link.start()
        queue.async { [self] in startStats() }
    }

    /// Start the audio hardware (no-op while powered off: `setPowered(true)` starts it).
    func start() {
        queue.async { [self] in
            guard !poweredOff else { Log.info("start skipped: powered off", "audio"); return }
            wantRunning = true
            buildAndStart(reason: "start")
        }
        link.start()
    }

    /// Stop the hardware (e.g. system going to sleep). The link stays up.
    func stop(reason: String) {
        queue.async { [self] in
            wantRunning = false
            Log.info("engine stop: \(reason)", "audio")
            flushAll(reason: "engine stop")
            teardown()
            restoreDefaultInput(reason: reason)
        }
    }

    func restart(reason: String) {
        queue.async { [self] in
            guard !poweredOff else { Log.info("restart (\(reason)) ignored: powered off", "audio"); return }
            wantRunning = true
            buildAndStart(reason: reason)
        }
    }

    /// Off switch (distinct from mute): `false` fully stops the engine and releases the
    /// input device (mic indicator goes away, no MIC frames) and announces HELLO with
    /// input "none"; `true` rebuilds the engine, which sends a fresh HELLO.
    func setPowered(_ on: Bool) {
        queue.async { [self] in
            guard poweredOff == on else { return }
            poweredOff = !on
            if on {
                wantRunning = true
                buildAndStart(reason: "turned on")
            } else {
                wantRunning = false
                Log.info("engine stop: turned off", "audio")
                flushAll(reason: "turned off")
                teardown()
                restoreDefaultInput(reason: "turned off")
                setHello(Hello(input_device: "none", aec: false), error: nil)
            }
        }
    }

    /// CPU policy: AEC is needed only while Mac may be speaking. Called by `Power` on
    /// every daemon state change; a change of mind rebuilds the engine (not at wake →
    /// listening, so the words right after "Hey Mac" are never lost).
    func setVoiceProcessingWanted(_ on: Bool) {
        queue.async { [self] in
            guard vpWanted != on else { return }
            vpWanted = on
            guard wantRunning, engine != nil, vpActive != (on && preferVoiceProcessing) else { return }
            Log.info("voice processing \(on ? "on" : "off") (state change) → rebuilding engine", "audio")
            buildAndStart(reason: on ? "AEC on for turn" : "AEC off (idle)")
        }
    }

    /// Synchronous shutdown for app termination.
    func shutdown() {
        queue.sync {
            wantRunning = false
            flushAll(reason: "shutdown")
            teardown()
            restoreDefaultInput(reason: "quit")
        }
        link.stop()
    }

    func setMuted(_ m: Bool) {
        queue.async { [self] in
            muted = m
            if let input = engine?.inputNode, input.isVoiceProcessingEnabled {
                input.isVoiceProcessingInputMuted = m
            }
            Log.info("mic muted=\(m)", "audio")
        }
    }

    private func scheduleRebuild(_ reason: String, delay: Double = 0.4) {
        guard wantRunning, !rebuildPending else { return }
        rebuildPending = true
        let now = Date()
        recentRebuilds = recentRebuilds.filter { now.timeIntervalSince($0) < 10 } + [now]
        // Storm guard: a misbehaving device can fire config changes continuously.
        let d = recentRebuilds.count > 5 ? 5.0 : delay
        queue.asyncAfter(deadline: .now() + d) { [self] in
            rebuildPending = false
            guard wantRunning else { return }
            buildAndStart(reason: reason)
        }
    }

    private func teardown() {
        generation &+= 1
        if let o = configObserver { NotificationCenter.default.removeObserver(o); configObserver = nil }
        if let eng = engine {
            eng.inputNode.removeTap(onBus: 0)
            player?.stop()
            eng.stop()
        }
        engine = nil
        player = nil
        playerFormat = nil
        vpActive = false
        micConverter = nil
        micInFormat = nil
        micAccum.removeAll(keepingCapacity: true)
    }

    private var startAttempts = 0
    private(set) var failedStarts = 0

    /// Build a fresh engine. A rebuild right after a VPIO teardown can fail with -10875
    /// (IsFormatSampleRateAndChannelCountValid: the output HW format is briefly invalid
    /// while CoreAudio reconfigures) or -10868. Strategy (never leave the mic dead):
    ///   1. retry with a FRESH engine every 250 ms, up to 8× (≈2 s);
    ///   2. then fall back to the previous configuration (VP toggled back) with the same retries;
    ///   3. only then wait 5 s and start over. Deferred PLAY/END survive all of this.
    private func buildAndStart(reason: String) {
        flushAll(reason: "engine rebuild")
        teardown()
        startAttempts = 0
        attemptStart(reason: reason, voiceProcessing: preferVoiceProcessing && vpWanted, fallbackTried: false)
    }

    private func attemptStart(reason: String, voiceProcessing wantVP: Bool, fallbackTried: Bool) {
        guard wantRunning, engine == nil else { return }
        let pinned = preferBuiltInMic && startAttempts == 0 ? pinBuiltInMicIfOnBluetooth() : nil
        let defaultName = CoreAudioDevices.defaultInput().map(CoreAudioDevices.name) ?? "none"
        let tryVP = wantVP && vpFailedForDevice != defaultName
        do {
            try startEngine(voiceProcessing: tryVP, reason: reason + (startAttempts > 0 ? " (attempt \(startAttempts + 1))" : ""), note: pinned)
            startAttempts = 0
            replayDeferred()
        } catch {
            teardown()
            startAttempts += 1
            failedStarts += 1
            Log.warn("engine start failed (vp=\(tryVP), attempt \(startAttempts)): \(error)", "audio")
            if startAttempts < 8 {
                queue.asyncAfter(deadline: .now() + 0.25) { [self] in attemptStart(reason: reason, voiceProcessing: wantVP, fallbackTried: fallbackTried) }
            } else if !fallbackTried {
                // Give up on this configuration: go back to the one that worked.
                if tryVP { vpFailedForDevice = defaultName }
                Log.warn("falling back to the previous configuration (vp=\(!wantVP))", "audio")
                startAttempts = 0
                queue.asyncAfter(deadline: .now() + 0.25) { [self] in attemptStart(reason: reason + " fallback", voiceProcessing: !wantVP, fallbackTried: true) }
            } else {
                setHello(Hello(input_device: "none", aec: false), error: "Audio engine failed: \(error.localizedDescription)")
                dropDeferred(reason: "engine failed")
                startAttempts = 0
                queue.asyncAfter(deadline: .now() + 5) { [self] in
                    if wantRunning && engine == nil { buildAndStart(reason: "retry") }
                }
            }
        }
    }

    /// N5 outcome (macOS 26, verified with a standalone VoiceProcessingIO unit and a
    /// temporary aggregate device): VPIO REJECTS any explicit device that is not the
    /// system default (kAudioOutputUnitProperty_CurrentDevice → -10851 on the input
    /// element and on the unit), and AVAudioEngine's input/output share one IO unit, so
    /// pinning a device on the input node breaks the output. The only way to keep
    /// AirPods out of HFP (telephone) mode is to make the built-in mic the system default
    /// input while Mac listens. Mitigation (DECISIONS #5 b): we remember the device we
    /// replaced, restore it on off / sleep / quit, and tell the user once per device.
    private func pinBuiltInMicIfOnBluetooth() -> String? {
        guard let cur = CoreAudioDevices.defaultInput(), CoreAudioDevices.isBluetooth(cur),
              let builtIn = CoreAudioDevices.builtInInput() else { return nil }
        let from = CoreAudioDevices.name(cur), to = CoreAudioDevices.name(builtIn)
        let st = CoreAudioDevices.setDefaultInput(builtIn)
        guard st == noErr else {
            Log.warn("could not switch default input to \(to): OSStatus \(st)", "audio")
            return "Input is the Bluetooth mic \(from) (couldn’t switch to the built-in mic)."
        }
        if replacedDefaultInput == nil { replacedDefaultInput = cur }
        Log.info("default input was Bluetooth (\(from)); switched system input to \(to) (will restore)", "audio")
        if notedSwitchFrom.insert(cur).inserted {
            onInputSwitched?("Using the built-in mic so your \(from) keep full audio quality. Your input device is restored when Mac is off, asleep or quit.")
        }
        return "Using the built-in mic instead of \(from) while Mac listens."
    }

    /// Put the user's own default input back (if it is still present).
    private func restoreDefaultInput(reason: String) {
        guard let prev = replacedDefaultInput else { return }
        replacedDefaultInput = nil
        guard CoreAudioDevices.allDevices().contains(prev) else {
            Log.info("previous default input is gone; nothing to restore (\(reason))", "audio")
            return
        }
        let st = CoreAudioDevices.setDefaultInput(prev)
        Log.info("restored system default input to \(CoreAudioDevices.name(prev)) (\(reason)): OSStatus \(st)", "audio")
    }

    private func startEngine(voiceProcessing: Bool, reason: String, note: String?) throws {
        let eng = AVAudioEngine()
        // The output side must exist before enabling VP, otherwise VPIO's
        // kAUInitialize on the output node fails with -10875 (verified on macOS 26).
        let mixer = eng.mainMixerNode
        let input = eng.inputNode
        var aec = false
        if voiceProcessing {
            try input.setVoiceProcessingEnabled(true)
            aec = input.isVoiceProcessingEnabled
            if aec {
                input.voiceProcessingOtherAudioDuckingConfiguration =
                    AVAudioVoiceProcessingOtherAudioDuckingConfiguration(enableAdvancedDucking: false, duckingLevel: .min)
                input.isVoiceProcessingInputMuted = muted
            }
        }

        let inFmt = input.outputFormat(forBus: 0)
        guard inFmt.sampleRate > 0, inFmt.channelCount > 0 else {
            throw NSError(domain: "MacAudio", code: 1, userInfo: [NSLocalizedDescriptionKey: "no input device (format \(inFmt))"])
        }

        // Playback graph
        let p = AVAudioPlayerNode()
        eng.attach(p)
        var outRate = eng.outputNode.outputFormat(forBus: 0).sampleRate
        if outRate <= 0 { outRate = 48_000 }
        let pFmt = AVAudioFormat(standardFormatWithSampleRate: outRate, channels: 1)!
        eng.connect(p, to: mixer, format: pFmt)

        // Tap with format nil (= the node's current format). Passing an explicit format
        // that no longer matches the hardware raises an ObjC exception (crash). With VP
        // the input has several channels (9 on Apple silicon); channel 0 is the processed
        // voice signal.
        let gen = generation
        input.installTap(onBus: 0, bufferSize: 1024, format: nil) { [weak self] buf, _ in
            guard let self, let ch = buf.floatChannelData else { return }
            let n = Int(buf.frameLength)
            let stride = buf.stride
            let rate = buf.format.sampleRate
            var mono = [Float](repeating: 0, count: n)
            let src = ch[0]
            if stride == 1 {
                mono.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: src, count: n) }
            } else {
                for i in 0..<n { mono[i] = src[i * stride] }
            }
            let samples = mono
            self.queue.async { self.processMic(samples, rate: rate, generation: gen) }
        }

        eng.prepare()
        do {
            try eng.start()
        } catch {
            input.removeTap(onBus: 0)
            eng.stop()
            throw error
        }

        engine = eng
        player = p
        playerFormat = pFmt
        vpActive = aec
        configObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: eng, queue: nil
        ) { [weak self] _ in
            guard let self else { return }
            self.queue.async { self.scheduleRebuild("configuration change") }
        }

        // VP (and AVAudioEngine generally) runs on an aggregate of the system defaults,
        // so the default input device is the meaningful name.
        let devID = CoreAudioDevices.defaultInput()
        let devName = devID.map(CoreAudioDevices.name) ?? "unknown"
        var warning = note
        if let d = devID, CoreAudioDevices.isBluetooth(d) {
            warning = "Input is the Bluetooth mic \(devName); audio quality may drop (HFP)."
        }
        Log.info("engine started (\(reason)): input=\(devName) aec=\(aec) inFmt=\(inFmt) out=\(outRate)Hz" +
                 (warning.map { " NOTE: \($0)" } ?? ""), "audio")
        setHello(Hello(input_device: devName, aec: aec), error: warning)
    }

    private func setHello(_ h: Hello, error: String?) {
        let changed: Bool = helloLock.withLock {
            defer { hello = h }
            return hello != h
        }
        if changed { link.sendHello(h) }
        onStatus?(h.input_device, h.aec, error)
    }

    // MARK: mic

    private func processMic(_ samples: [Float], rate: Double, generation gen: UInt64) {
        guard gen == generation, rate > 0, !samples.isEmpty else { return }
        if micInFormat?.sampleRate != rate || micConverter == nil {
            let f = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false)!
            micInFormat = f
            micConverter = AVAudioConverter(from: f, to: micOutFormat)
            if micConverter == nil { Log.error("no mic converter for \(f)", "audio") }
        }
        guard let conv = micConverter, let inFmt = micInFormat,
              let src = AVAudioPCMBuffer(pcmFormat: inFmt, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
        src.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        guard let out = convert(src, with: conv, to: micOutFormat, endOfStream: false), out.frameLength > 0 else { return }
        let bytes = Int(out.frameLength) * 2
        if muted {
            micAccum.append(Data(count: bytes))  // keep the clock running, send silence
        } else {
            micAccum.append(UnsafeBufferPointer(start: UnsafeRawPointer(out.int16ChannelData![0]).assumingMemoryBound(to: UInt8.self), count: bytes))
        }
        var off = 0
        while micAccum.count - off >= Self.micFrameBytes {
            let s = micAccum.startIndex + off
            link.sendMic(micAccum.subdata(in: s..<(s + Self.micFrameBytes)))
            off += Self.micFrameBytes
        }
        if off > 0 { micAccum.removeSubrange(micAccum.startIndex..<(micAccum.startIndex + off)) }
    }

    /// Streaming conversion. With `endOfStream == false` the converter keeps its resampler
    /// state between calls (`.noDataNow`), so consecutive chunks are seamless.
    private func convert(_ src: AVAudioPCMBuffer?, with conv: AVAudioConverter, to fmt: AVAudioFormat,
                         endOfStream: Bool) -> AVAudioPCMBuffer? {
        let inFrames = Double(src?.frameLength ?? 0)
        let ratio = fmt.sampleRate / conv.inputFormat.sampleRate
        let cap = AVAudioFrameCount(inFrames * ratio + 1024)
        guard let out = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: cap) else { return nil }
        var given = false
        var err: NSError?
        let status = conv.convert(to: out, error: &err) { _, st in
            if !given, let src {
                given = true
                st.pointee = .haveData
                return src
            }
            st.pointee = endOfStream ? .endOfStream : .noDataNow
            return nil
        }
        if status == .error {
            Log.warn("convert error: \(err?.localizedDescription ?? "?")", "audio")
            return nil
        }
        return out
    }

    private func startStats() {
        statsTimer?.cancel()
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + 10, repeating: 10)
        t.setEventHandler { [weak self] in
            guard let self else { return }
            let (sent, dropped) = self.link.takeMicCount()
            if self.link.isConnected || sent > 0 {
                Log.info(String(format: "mic: %d frames in 10 s (%.1f/s), dropped total %d, streams=%d scheduled=%d",
                                sent, Double(sent) / 10, dropped, self.streams.count, self.scheduled.count), "audio")
            }
        }
        t.resume()
        statsTimer = t
    }

    // MARK: playback

    /// True while the engine is down but expected back shortly (rebuild in progress).
    private var rebuilding: Bool { engine == nil && wantRunning && !poweredOff }

    private func deferMessage(_ m: AudioMessage, bytes: Int) {
        deferred.append(m)
        deferredBytes += bytes
        if deferredBytes > 2 * 1024 * 1024 || deferred.count > 1000 {  // ~20 s of 48 kHz audio
            dropDeferred(reason: "rebuild took too long")
        }
    }

    private func replayDeferred() {
        guard !deferred.isEmpty else { return }
        let msgs = deferred
        deferred.removeAll(); deferredBytes = 0
        Log.info("replaying \(msgs.count) PLAY/END deferred during rebuild", "audio")
        for m in msgs { audioLink(received: m) }
    }

    /// Give up on deferred audio: report the streams DRAINED so the daemon moves on.
    private func dropDeferred(reason: String) {
        guard !deferred.isEmpty else { return }
        var sids: [UInt32] = []
        for m in deferred {
            if case .play(let sid, _, _) = m, !sids.contains(sid) { sids.append(sid) }
            if case .end(let sid) = m, !sids.contains(sid) { sids.append(sid) }
        }
        deferred.removeAll(); deferredBytes = 0
        for sid in sids where !closedStreams.contains(sid) { finish(sid) }
        Log.warn("dropped deferred audio for streams \(sids) (\(reason))", "audio")
    }

    private func handlePlay(stream sid: UInt32, sampleRate: UInt32, pcm: Data) {
        guard !closedStreams.contains(sid) else { return }
        if rebuilding { deferMessage(.play(streamID: sid, sampleRate: sampleRate, pcm: pcm), bytes: pcm.count); return }
        guard let p = player, let pFmt = playerFormat, let eng = engine, eng.isRunning else {
            // No output at all (engine failed / powered off): pretend it played so the
            // daemon never waits forever.
            if streams[sid] == nil { streams[sid] = Stream(sampleRate: Double(sampleRate), converter: nil,
                                                           inFormat: pFmt0(Double(sampleRate))) }
            return
        }
        let rate = Double(max(sampleRate, 1))
        var st = streams[sid] ?? Stream(sampleRate: rate, converter: nil, inFormat: pFmt0(rate))
        if st.sampleRate != rate || (st.converter == nil && rate != pFmt.sampleRate) {
            st.sampleRate = rate
            st.inFormat = pFmt0(rate)
            st.converter = rate == pFmt.sampleRate ? nil : AVAudioConverter(from: st.inFormat, to: pFmt)
        }
        let n = pcm.count / 2
        guard n > 0, let src = AVAudioPCMBuffer(pcmFormat: st.inFormat, frameCapacity: AVAudioFrameCount(n)) else {
            streams[sid] = st
            return
        }
        src.frameLength = AVAudioFrameCount(n)
        let dst = src.floatChannelData![0]
        pcm.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            for i in 0..<n {
                let v = Int16(littleEndian: base.loadUnaligned(fromByteOffset: i * 2, as: Int16.self))
                dst[i] = Float(v) / 32768
            }
        }
        let buf: AVAudioPCMBuffer?
        if let c = st.converter { buf = convert(src, with: c, to: pFmt, endOfStream: false) } else { buf = src }
        streams[sid] = st
        if let b = buf, b.frameLength > 0 { schedule(b, stream: sid, on: p) }
    }

    private func pFmt0(_ rate: Double) -> AVAudioFormat {
        AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false)!
    }

    private func schedule(_ b: AVAudioPCMBuffer, stream sid: UInt32, on p: AVAudioPlayerNode) {
        let token = nextToken
        nextToken &+= 1
        let gen = generation
        scheduled.append(Scheduled(token: token, stream: sid))
        streams[sid]?.pending += 1
        p.scheduleBuffer(b, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            guard let self else { return }
            self.queue.async { self.bufferFinished(token: token, generation: gen) }
        }
        if !p.isPlaying { p.play() }
    }

    private func bufferFinished(token: UInt64, generation gen: UInt64) {
        guard gen == generation, let i = scheduled.firstIndex(where: { $0.token == token }) else { return }
        let sid = scheduled.remove(at: i).stream
        streams[sid]?.pending -= 1
        checkDrained(sid)
    }

    private func handleEnd(stream sid: UInt32) {
        guard !closedStreams.contains(sid) else { return }
        if rebuilding { deferMessage(.end(streamID: sid), bytes: 0); return }
        guard var st = streams[sid] else {
            // END with no audio at all: drained by definition.
            finish(sid)
            return
        }
        st.ended = true
        // Flush the resampler tail.
        if let c = st.converter, let p = player, let pFmt = playerFormat,
           let tail = convert(nil, with: c, to: pFmt, endOfStream: true), tail.frameLength > 0 {
            streams[sid] = st
            schedule(tail, stream: sid, on: p)
        } else {
            streams[sid] = st
        }
        checkDrained(sid)
    }

    private func checkDrained(_ sid: UInt32) {
        guard let st = streams[sid], st.ended, st.pending <= 0 else { return }
        finish(sid)
    }

    private func finish(_ sid: UInt32) {
        streams[sid] = nil
        closedStreams.insert(sid)
        link.sendDrained(sid)
    }

    /// FLUSH(id): the daemon's SpeechQueue plays one stream at a time, so flushing one
    /// stream == stopping playback; any other open stream is reported DRAINED too
    /// (simplicity #32: the protocol is unchanged, only the bookkeeping is smaller).
    private func handleFlush(stream sid: UInt32) {
        if sid != 0 && !closedStreams.contains(sid) && streams[sid] == nil { finish(sid) }
        flushAll(reason: sid == 0 ? "FLUSH(0)" : "FLUSH(\(sid))")
    }

    /// Stop all playback immediately and report every open stream as DRAINED.
    private func flushAll(reason: String) {
        if !deferred.isEmpty && reason != "engine rebuild" { dropDeferred(reason: reason) }
        let open = Array(streams.keys)
        scheduled.removeAll()
        player?.stop()
        for sid in open { finish(sid) }
        if !open.isEmpty { Log.info("flushed streams \(open) (\(reason))", "audio") }
    }
}

extension AudioEngine: AudioLinkDelegate {
    func audioLinkDidConnect() {
        // New daemon session: stream ids may start over.
        flushAll(reason: "audio link connected")
        closedStreams = RecentSet(capacity: 512)
    }

    func audioLinkDidDisconnect() {
        // Daemon went away: nothing will ever be flushed/ended → stop playback silently.
        scheduled.removeAll()
        player?.stop()
        streams.removeAll()
    }

    func audioLink(received m: AudioMessage) {
        switch m {
        case .play(let sid, let rate, let pcm): handlePlay(stream: sid, sampleRate: rate, pcm: pcm)
        case .end(let sid): handleEnd(stream: sid)
        case .flush(let sid): handleFlush(stream: sid)
        case .hello(let json):
            Log.info("daemon HELLO: \(String(decoding: json, as: UTF8.self))", "link")
            let v = ((try? JSONSerialization.jsonObject(with: json)) as? [String: Any])?["version"] as? Int
            if v != Hello.protocolVersion {
                Log.warn("audio protocol version mismatch: app \(Hello.protocolVersion), daemon \(v.map(String.init) ?? "missing"): update both sides", "link")
            }
        case .mic, .drained: Log.warn("unexpected \(m.frameType) from daemon", "link")
        case .unknown(let t, _): Log.warn("unknown frame type \(t)", "link")
        }
    }

    func audioLinkHello() -> Hello { helloLock.withLock { hello } }
}

/// Bounded set of recently closed stream ids.
struct RecentSet {
    private var order: [UInt32] = []
    private var set: Set<UInt32> = []
    let capacity: Int
    init(capacity: Int) { self.capacity = capacity }
    func contains(_ v: UInt32) -> Bool { set.contains(v) }
    mutating func insert(_ v: UInt32) {
        guard set.insert(v).inserted else { return }
        order.append(v)
        if order.count > capacity { set.remove(order.removeFirst()) }
    }
}
