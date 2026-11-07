import AppKit
import Symbols

/// Menu bar extra: one template SF Symbol per state + a short, grouped menu.
///
/// Static states set `button.image` (the system tints template images for light/dark
/// menu bars and highlight). Ongoing states (listening, thinking, speaking) swap in an
/// `NSImageView` overlay so we can use SF Symbol effects (`variableColor`, `pulse`);
/// with Reduce Motion on, every state is a still glyph.
@MainActor
final class StatusBar: NSObject, NSMenuDelegate {
    struct Actions {
        var talk: () -> Void
        var togglePower: () -> Void
        var toggleMute: () -> Void
        var stop: () -> Void
        var reset: () -> Void
        var voice: () -> Void
        var remoteLink: () -> Void
        var remoteEnd: () -> Void
        var permissions: () -> Void
        var settings: () -> Void
        var restartDaemon: () -> Void
        var restartAudio: () -> Void
    }

    private let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
    private let model: AppModel
    private let actions: Actions
    private let overlay = PassthroughImageView()
    private var shown: (glyph: String, animated: Bool)?
    private var bounceReset: DispatchWorkItem?

    init(model: AppModel, actions: Actions) {
        self.model = model
        self.actions = actions
        super.init()
        let menu = NSMenu()
        menu.delegate = self
        menu.autoenablesItems = false
        item.menu = menu
        item.autosaveName = "io.github.macagent.Mac.status"  // stable position across launches
        item.behavior = []                                    // not removable by ⌘-drag
        if let b = item.button {
            b.toolTip = "Mac"
            b.imagePosition = .imageOnly
            overlay.imageScaling = .scaleNone
            overlay.contentTintColor = .labelColor
            overlay.isHidden = true
            overlay.translatesAutoresizingMaskIntoConstraints = false
            b.addSubview(overlay)
            NSLayoutConstraint.activate([
                overlay.centerXAnchor.constraint(equalTo: b.centerXAnchor),
                overlay.centerYAnchor.constraint(equalTo: b.centerYAnchor),
            ])
        }
        refreshIcon()
        NotificationCenter.default.addObserver(
            forName: NSWorkspace.accessibilityDisplayOptionsDidChangeNotification, object: nil, queue: .main
        ) { [weak self] _ in MainActor.assumeIsolated { self?.shown = nil; self?.refreshIcon() } }
    }

    /// Pops the menu programmatically (used by `MAC_DEBUG_UI=menu` for screenshots).
    func openMenu() { item.button?.performClick(nil) }

    /// The window that hosts the status item (nil until the menu bar has laid it out).
    /// Used by AppDelegate to detect an item hidden under the notch / menu-bar overflow.
    var buttonWindow: NSWindow? { item.button?.window }

    // MARK: icon

    func refreshIcon() {
        guard let b = item.button else { return }
        let glyph = model.glyph
        let off = model.off
        let animated = !off && !model.muted && model.state.isActive && !UX.reduceMotion
        b.alphaValue = off ? 0.45 : 1
        b.setAccessibilityLabel("Mac: \(model.statusLine)")
        b.toolTip = "Mac — \(model.statusLine)"
        if let s = shown, s.glyph == glyph, s.animated == animated { return }
        shown = (glyph, animated)
        bounceReset?.cancel()
        overlay.removeAllSymbolEffects()

        if animated {
            // Keep a transparent placeholder image so the item keeps its width/highlight.
            b.image = Self.symbol(glyph, alpha: 0)
            overlay.image = Self.symbol(glyph)
            overlay.isHidden = false
            switch model.state {
            case .listening, .followup:
                overlay.addSymbolEffect(.variableColor.iterative.reversing, options: .repeating.speed(0.8))
            case .speaking:
                overlay.addSymbolEffect(.variableColor.iterative, options: .repeating)
            default:  // thinking
                overlay.addSymbolEffect(.pulse.wholeSymbol, options: .repeating.speed(0.9))
            }
            return
        }
        overlay.isHidden = true
        b.image = Self.symbol(glyph)
        // One-shot nudge when the person needs to act (confirm) or look (error).
        if !UX.reduceMotion, !off, !model.muted, model.state == .confirming || model.state == .error {
            b.image = Self.symbol(glyph, alpha: 0)
            overlay.image = Self.symbol(glyph)
            overlay.isHidden = false
            overlay.addSymbolEffect(.bounce.up.byLayer, options: .nonRepeating)
            let w = DispatchWorkItem { [weak self] in
                MainActor.assumeIsolated {
                    guard let self else { return }
                    self.overlay.isHidden = true
                    self.item.button?.image = Self.symbol(glyph)
                }
            }
            bounceReset = w
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.9, execute: w)
        }
    }

    private static func symbol(_ name: String, alpha: CGFloat = 1) -> NSImage? {
        let cfg = NSImage.SymbolConfiguration(pointSize: 15, weight: .medium)
        guard let img = (NSImage(systemSymbolName: name, accessibilityDescription: "Mac")
                         ?? NSImage(systemSymbolName: "waveform", accessibilityDescription: "Mac"))?
            .withSymbolConfiguration(cfg) else { return nil }
        img.isTemplate = true
        guard alpha < 1 else { return img }
        let blank = NSImage(size: img.size, flipped: false) { _ in true }
        blank.isTemplate = true
        return blank
    }

    // MARK: menu

    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()

        // 1. Status
        header(menu, model.statusLine)
        if let d = model.statusDetail { info(menu, d) }
        if model.brainAuth == "signed_out" {
            action(menu, "Sign in to Claude…", symbol: "person.crop.circle.badge.exclamationmark") { [weak self] in
                self?.model.guided?.signIn()
            }
        } else if model.brainAuth == "missing" {
            action(menu, "Install Claude Code…", symbol: "arrow.down.circle") { [weak self] in
                self?.model.guided?.installClaude()
            }
        }
        if model.remote.active {
            info(menu, model.remote.banner, symbol: "iphone.radiowaves.left.and.right")
        }

        // 2. Last exchange
        if !model.lastTranscript.isEmpty || !model.lastReply.isEmpty {
            menu.addItem(.separator())
            if !model.lastTranscript.isEmpty {
                info(menu, "“\(UX.truncate(model.lastTranscript, 56))”", symbol: "person")
            }
            if !model.lastReply.isEmpty {
                info(menu, UX.truncate(model.lastReply, 56), symbol: "sparkle")
            }
        }
        menu.addItem(.separator())

        // 3. Primary actions
        let busy = model.state == .thinking || model.state == .speaking
        action(menu, "Talk to Mac", symbol: "mic", key: " ", mods: [.option], actions.talk)
            .isEnabled = model.connected && !model.muted && !model.off
        action(menu, "Stop", symbol: "stop.fill", key: ".", actions.stop).isEnabled = model.connected && busy
        if model.remote.active {
            action(menu, "End Phone Session", symbol: "iphone.slash", actions.remoteEnd)
        }
        menu.addItem(.separator())

        // 4. Toggles
        action(menu, model.off ? "Turn On" : "Turn Off", symbol: "power", actions.togglePower)
        let mute = action(menu, "Mute", symbol: "mic.slash", actions.toggleMute)
        mute.state = model.muted ? .on : .off
        mute.isEnabled = !model.off
        menu.addItem(.separator())

        // 5. Secondary
        action(menu, "New Conversation", symbol: "plus.bubble", key: "n", actions.reset).isEnabled = model.connected
        action(menu, "Voice…", symbol: "waveform.and.mic", actions.voice)
        action(menu, "Talk from Phone…", symbol: "iphone", actions.remoteLink).isEnabled = model.connected
        action(menu, "Permissions…", symbol: "lock.shield", actions.permissions)
        let login = action(menu, "Launch at Login", symbol: "sunrise") {
            if let err = UX.setLaunchAtLogin(!UX.launchAtLogin) {
                let a = NSAlert(); a.messageText = "Couldn’t change Launch at Login"; a.informativeText = err; a.runModal()
            }
        }
        login.state = UX.launchAtLogin ? .on : .off
        action(menu, "Open Logs", symbol: "doc.text") { NSWorkspace.shared.open(Paths.logDir) }
        action(menu, "Settings…", symbol: "gearshape", key: ",", actions.settings)
        let fix = NSMenuItem(title: "Troubleshoot", action: nil, keyEquivalent: "")
        fix.image = Self.menuSymbol("wrench.and.screwdriver")
        let sub = NSMenu(title: "Troubleshoot")
        sub.autoenablesItems = false
        action(sub, "Restart Audio", symbol: "speaker.wave.2", actions.restartAudio)
        action(sub, "Restart Engine", symbol: "arrow.clockwise", actions.restartDaemon)
        info(sub, "Mic: \(model.inputDevice) · \(model.aec ? "echo cancellation on" : "no echo cancellation")")
        fix.submenu = sub
        menu.addItem(fix)
        menu.addItem(.separator())

        // 6. Quit
        action(menu, "Quit Mac", key: "q") { NSApp.terminate(nil) }
    }

    private func header(_ menu: NSMenu, _ title: String) {
        let i = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        i.isEnabled = false
        i.attributedTitle = NSAttributedString(string: title, attributes: [
            .font: NSFont.menuFont(ofSize: 0).withWeight(.semibold),
            .foregroundColor: NSColor.labelColor,
        ])
        i.setAccessibilityLabel("Status: \(title)")
        menu.addItem(i)
    }

    private func info(_ menu: NSMenu, _ title: String, symbol: String? = nil) {
        let i = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        i.isEnabled = false
        if let symbol { i.image = Self.menuSymbol(symbol) }
        menu.addItem(i)
    }

    @discardableResult
    private func action(_ menu: NSMenu, _ title: String, symbol: String? = nil, key: String = "",
                        mods: NSEvent.ModifierFlags = [.command], _ block: @escaping () -> Void) -> NSMenuItem {
        let i = ClosureMenuItem(title: title, keyEquivalent: key, block: block)
        i.keyEquivalentModifierMask = mods
        if let symbol { i.image = Self.menuSymbol(symbol) }
        menu.addItem(i)
        return i
    }

    private static func menuSymbol(_ name: String) -> NSImage? {
        let img = NSImage(systemSymbolName: name, accessibilityDescription: nil)?
            .withSymbolConfiguration(.init(pointSize: 13, weight: .regular))
        img?.isTemplate = true
        return img
    }

}

/// NSMenuItem that runs a closure.
final class ClosureMenuItem: NSMenuItem {
    private let block: () -> Void
    init(title: String, keyEquivalent: String, block: @escaping () -> Void) {
        self.block = block
        super.init(title: title, action: #selector(fire), keyEquivalent: keyEquivalent)
        target = self
    }
    @available(*, unavailable) required init(coder: NSCoder) { fatalError() }
    @objc private func fire() { block() }
}

/// Image view that never intercepts clicks (lives inside the status bar button).
final class PassthroughImageView: NSImageView {
    override func hitTest(_ point: NSPoint) -> NSView? { nil }
}

private extension NSFont {
    func withWeight(_ w: NSFont.Weight) -> NSFont {
        NSFont.systemFont(ofSize: pointSize, weight: w)
    }
}
