import AppKit
import Combine
import SwiftUI

/// Floating, non-activating pill at the top-center of the screen the pointer is on.
/// Shows the state glyph, the live user transcript and the assistant's current sentence.
/// Never becomes key or main and ignores the mouse → never steals focus.
@MainActor
final class HUDController {
    private let panel: HUDPanel
    private let effect = NSVisualEffectView()
    private let host: NSHostingView<HUDView>
    private let model: AppModel
    private var hideWork: DispatchWorkItem?
    private var cancellables: Set<AnyCancellable> = []
    private var visible = false

    static let minWidth: CGFloat = 280, maxWidth: CGFloat = 560, radius: CGFloat = 16
    static let duration = 0.18

    init(model: AppModel) {
        self.model = model
        panel = HUDPanel(contentRect: NSRect(x: 0, y: 0, width: Self.minWidth, height: 56),
                         styleMask: [.nonactivatingPanel, .borderless],
                         backing: .buffered, defer: true)
        panel.isFloatingPanel = true
        panel.level = .statusBar
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]
        panel.hidesOnDeactivate = false
        panel.ignoresMouseEvents = true
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.animationBehavior = .none
        panel.alphaValue = 0
        panel.setAccessibilityLabel("Mac")
        panel.setAccessibilityRole(.group)

        effect.material = .popover
        effect.blendingMode = .behindWindow
        effect.state = .active
        effect.maskImage = Self.roundedMask(radius: Self.radius)
        panel.contentView = effect

        host = NSHostingView(rootView: HUDView(model: model, fill: true, note: nil))
        host.sizingOptions = []
        host.translatesAutoresizingMaskIntoConstraints = false
        effect.addSubview(host)
        NSLayoutConstraint.activate([
            host.leadingAnchor.constraint(equalTo: effect.leadingAnchor),
            host.trailingAnchor.constraint(equalTo: effect.trailingAnchor),
            host.topAnchor.constraint(equalTo: effect.topAnchor),
            host.bottomAnchor.constraint(equalTo: effect.bottomAnchor),
        ])

        model.$state.receive(on: RunLoop.main).sink { [weak self] s in
            MainActor.assumeIsolated { self?.stateChanged(s) }
        }.store(in: &cancellables)
        model.$muted.dropFirst().removeDuplicates().receive(on: RunLoop.main).sink { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, !self.model.off else { return }
                self.show(); self.scheduleHide(after: 2.5)
            }
        }.store(in: &cancellables)
        Publishers.MergeMany(model.$userText, model.$assistantText, model.$activity, model.$errorText)
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in MainActor.assumeIsolated { if self?.visible == true { self?.layout(animated: true) } } }
            .store(in: &cancellables)
        // Phone session banner (N7): shown for the whole session, then "Phone disconnected" briefly.
        model.$remote.removeDuplicates().dropFirst().receive(on: RunLoop.main).sink { [weak self] r in
            MainActor.assumeIsolated {
                guard let self else { return }
                if r.active { self.show() } else if self.visible { self.layout(animated: true); self.scheduleHide(after: 2.5) }
            }
        }.store(in: &cancellables)
        // Appearance settings (HUD position, portrait) changed → re-render / re-place.
        NotificationCenter.default.addObserver(forName: AppPrefs.changed, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.host.rootView = HUDView(model: self.model, fill: true, note: self.note)
                if self.visible { self.layout(animated: true) }
            }
        }
        // One-time note when the input device was switched (N5/UX#11): shown 4 s, then cleared.
        model.$inputSwitchedNote.removeDuplicates().receive(on: RunLoop.main).sink { [weak self] n in
            MainActor.assumeIsolated {
                guard let self, let n, !n.isEmpty else { return }
                self.model.inputSwitchedNote = nil
                guard !self.model.state.inTurn else { return }
                self.note = n
                self.show(); self.scheduleHide(after: 4)
            }
        }.store(in: &cancellables)
    }

    /// Transient one-line note shown in the compact variant (cleared on hide).
    private var note: String? {
        didSet { host.rootView = HUDView(model: model, fill: true, note: note) }
    }

    private func stateChanged(_ s: AgentState) {
        if Paths.flag("MAC_NO_HUD") { return }
        note = nil
        switch s {
        case .idle, .unknown:
            if visible { scheduleHide(after: model.remote.active ? 60 * 60 : 4) }
        case .off:
            show(); scheduleHide(after: 2.5)
        case .error:
            show(); scheduleHide(after: 8)
        default:
            hideWork?.cancel(); hideWork = nil
            show()
        }
    }

    // MARK: geometry

    /// The screen the pointer is on (multi-display), else the main screen.
    private var targetScreen: NSScreen? {
        let p = NSEvent.mouseLocation
        return NSScreen.screens.first { NSMouseInRect(p, $0.frame, false) } ?? NSScreen.main ?? NSScreen.screens.first
    }

    /// Measures the pill's natural size at the max width (text wraps; width shrinks to fit).
    private lazy var measure = NSHostingController(rootView: HUDView(model: model, fill: false, note: nil))

    private func targetFrame(on screen: NSScreen) -> NSRect {
        // Pass 1: natural (single-line) width → clamp. Pass 2: height at exactly that width.
        measure.rootView = HUDView(model: model, fill: false, note: note)
        let big = CGFloat.greatestFiniteMagnitude
        let natural = measure.sizeThatFits(in: NSSize(width: big, height: big))
        let w = min(max(ceil(natural.width), Self.minWidth), Self.maxWidth)
        measure.rootView = HUDView(model: model, fill: true, note: note)
        let h = max(ceil(measure.sizeThatFits(in: NSSize(width: w, height: big)).height), 44)
        let vf = screen.visibleFrame
        let y = AppPrefs.hudPosition == "bottom" ? vf.minY + 10 : vf.maxY - h - 10
        return NSRect(x: round(vf.midX - w / 2), y: y, width: w, height: h)
    }

    /// Slide direction: in from above at the top, from below at the bottom.
    private var slideOffset: CGFloat {
        UX.reduceMotion ? 0 : (AppPrefs.hudPosition == "bottom" ? -8 : 8)
    }

    private func layout(animated: Bool) {
        guard let screen = targetScreen else { return }
        let f = targetFrame(on: screen)
        guard f != panel.frame else { return }
        if animated && !UX.reduceMotion {
            NSAnimationContext.runAnimationGroup { ctx in
                ctx.duration = 0.12
                ctx.timingFunction = CAMediaTimingFunction(name: .easeOut)
                panel.animator().setFrame(f, display: true)
            }
        } else {
            panel.setFrame(f, display: true)
        }
    }

    // MARK: show / hide (0.18 s fade + 8 pt slide; fade only with Reduce Motion)

    private func show() {
        hideWork?.cancel(); hideWork = nil
        guard let screen = targetScreen else { return }
        host.rootView = HUDView(model: model, fill: true, note: note)
        host.layoutSubtreeIfNeeded()
        let f = targetFrame(on: screen)
        if visible {
            layout(animated: true)
            return
        }
        visible = true
        let slide = slideOffset
        panel.setFrame(f.offsetBy(dx: 0, dy: slide), display: true)
        panel.alphaValue = 0
        panel.orderFrontRegardless()
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = Self.duration
            ctx.timingFunction = CAMediaTimingFunction(name: .easeOut)
            panel.animator().alphaValue = 1
            panel.animator().setFrame(f, display: true)
        }
    }

    private func hide() {
        guard visible else { return }
        visible = false
        let f = panel.frame
        let slide = slideOffset
        NSAnimationContext.runAnimationGroup({ ctx in
            ctx.duration = Self.duration
            ctx.timingFunction = CAMediaTimingFunction(name: .easeIn)
            panel.animator().alphaValue = 0
            panel.animator().setFrame(f.offsetBy(dx: 0, dy: slide), display: true)
        }, completionHandler: { [weak self] in
            MainActor.assumeIsolated {
                guard let self, !self.visible else { return }
                self.panel.orderOut(nil)
                self.note = nil
            }
        })
    }

    private func scheduleHide(after s: Double) {
        hideWork?.cancel()
        let w = DispatchWorkItem { [weak self] in MainActor.assumeIsolated { self?.hide() } }
        hideWork = w
        DispatchQueue.main.asyncAfter(deadline: .now() + s, execute: w)
    }

    private static func roundedMask(radius: CGFloat) -> NSImage {
        let side = radius * 2 + 1
        let img = NSImage(size: NSSize(width: side, height: side), flipped: false) { r in
            NSColor.black.setFill()
            NSBezierPath(roundedRect: r, xRadius: radius, yRadius: radius).fill()
            return true
        }
        img.capInsets = NSEdgeInsets(top: radius, left: radius, bottom: radius, right: radius)
        img.resizingMode = .stretch
        return img
    }
}

final class HUDPanel: NSPanel {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}

struct HUDView: View {
    @ObservedObject var model: AppModel
    /// true in the panel (stretch to the panel's width); false when measuring natural size.
    var fill: Bool
    var note: String?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var compact: Bool { note != nil || model.off || model.muted || (model.remote.active && !model.state.inTurn) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if model.remote.active { banner }
            HStack(alignment: compact ? .center : .top, spacing: 10) {
                glyph
                VStack(alignment: .leading, spacing: 3) {
                    if let note {
                        Text(note).font(.callout.weight(.medium)).lineLimit(2)
                    } else if compact {
                        Text(model.statusLine).font(.callout.weight(.medium))
                    } else {
                        content
                    }
                }
                .frame(minHeight: 24, alignment: .leading)
            }
        }
        .padding(.leading, 12).padding(.trailing, 16).padding(.vertical, 10)
        .frame(maxWidth: fill ? .infinity : nil, alignment: .leading)
        .fixedSize(horizontal: false, vertical: true)
        .overlay(
            RoundedRectangle(cornerRadius: HUDController.radius, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Mac")
        .accessibilityValue(accessibilityText)
    }

    /// "Phone connected · PIN 1234" — the caller must say the PIN first (N7).
    private var banner: some View {
        HStack(spacing: 6) {
            Image(systemName: model.remote.present ? "iphone.radiowaves.left.and.right" : "iphone")
                .font(.footnote.weight(.semibold))
            Text(model.remote.banner).font(.footnote.weight(.semibold).monospacedDigit()).lineLimit(1)
            Spacer(minLength: 0)
        }
        .foregroundStyle(Color.accentColor)
        .padding(.horizontal, 10).padding(.vertical, 5)
        .background(Capsule().fill(Color.accentColor.opacity(0.12)))
        .accessibilityLabel(model.remote.banner)
    }

    @ViewBuilder private var content: some View {
        switch model.state {
        case .error:
            Text(model.errorText.isEmpty ? "Something went wrong" : model.errorText)
                .font(.body.weight(.medium)).lineLimit(3)
            Text(model.errorAdvice).font(.callout).foregroundStyle(.secondary).lineLimit(2)
            if let cmd = model.errorCommand {
                Text(cmd).font(.callout.monospaced()).textSelection(.enabled)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(RoundedRectangle(cornerRadius: 5).fill(Color.primary.opacity(0.08)))
            }
        case .confirming:
            if !model.assistantText.isEmpty {
                Text(model.assistantText).font(.body.weight(.medium)).lineLimit(3)
            }
            Text("Say “yes” to go ahead, or “no” to cancel.")
                .font(.callout).foregroundStyle(.secondary).lineLimit(2)
        default:
            if model.userText.isEmpty || model.state == .followup {
                Text(model.statusLine).font(.callout).foregroundStyle(.secondary)
                if model.screenLocked {
                    Label(AppModel.screenLockedLine, systemImage: "lock.fill")
                        .font(.footnote).foregroundStyle(.secondary).lineLimit(2)
                }
            } else if model.userFromPhone {
                Label(model.userText, systemImage: "iphone").font(.callout).foregroundStyle(.secondary).lineLimit(3)
            } else {
                Text(model.userText).font(.callout).foregroundStyle(.secondary).lineLimit(3)
            }
            if !model.assistantText.isEmpty {
                Text(model.assistantText).font(.body.weight(.medium)).lineLimit(3)
            }
            // UX#6: keep showing what Mac is doing under the last sentence while the turn runs.
            if model.state == .thinking, !model.activity.isEmpty {
                Label(model.activity, systemImage: "gearshape")
                    .labelStyle(.titleAndIcon)
                    .font(.footnote).foregroundStyle(.secondary).lineLimit(1)
            }
        }
    }

    private var glyph: some View {
        let s = model.state
        let active = !compact && !reduceMotion
        let portrait = AppPrefs.avatarImage()
        let symbol = Image(systemName: model.glyph)
            .font(.system(size: portrait == nil ? 14 : 9, weight: .semibold))
            .symbolRenderingMode(.hierarchical)
            .foregroundStyle(portrait == nil ? tint : Color.white)
            .symbolEffect(.variableColor.iterative.reversing, options: .repeating.speed(0.8),
                          isActive: active && (s == .listening || s == .followup))
            .symbolEffect(.variableColor.iterative, options: .repeating, isActive: active && s == .speaking)
            .symbolEffect(.pulse.wholeSymbol, options: .repeating.speed(0.9), isActive: active && s == .thinking)
            .contentTransition(reduceMotion ? .identity : .symbolEffect(.replace))
        return ZStack(alignment: .bottomTrailing) {
            if let portrait {
                // The person's chosen picture, with the state glyph as a small badge.
                Image(nsImage: portrait).resizable().scaledToFill()
                    .frame(width: 30, height: 30).clipShape(Circle())
                    .overlay(Circle().strokeBorder(tint.opacity(0.5), lineWidth: 1.5))
                    .saturation(compact ? 0 : 1)
                ZStack {
                    Circle().fill(tint).frame(width: 15, height: 15)
                    symbol
                }
                .offset(x: 3, y: 3)
            } else {
                Circle().fill(tint.opacity(0.14)).frame(width: 28, height: 28)
                symbol
            }
        }
        .frame(width: 30, height: 30)
        .accessibilityHidden(true)
    }

    private var tint: Color {
        if compact { return .secondary }
        switch model.state {
        case .error: return .orange
        default: return .accentColor
        }
    }

    private var accessibilityText: String {
        var parts = [note ?? model.statusLine]
        if model.remote.active { parts.insert(model.remote.banner, at: 0) }
        if !model.userText.isEmpty { parts.append("You said: \(model.userText)") }
        if !model.assistantText.isEmpty { parts.append("Mac: \(model.assistantText)") }
        if model.state == .error { parts.append(model.errorText); parts.append(model.errorAdvice) }
        return parts.joined(separator: ". ")
    }
}
