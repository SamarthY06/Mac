import AppKit
import CoreImage
import CoreImage.CIFilterBuiltins
import SwiftUI

/// "Talk from Phone…": a small window with a QR code for the LiveKit link (UX#13).
/// When remote access is off or not configured, it explains the fix in two lines.
@MainActor
final class PhonePanel {
    enum Content: Equatable {
        case loading
        case link(String, pin: String)
        case disabled(canEnable: Bool)
        case error(String)
    }

    private var window: NSWindow?
    let state = PhoneState()
    private let model: AppModel

    init(model: AppModel) { self.model = model }
    /// Set by AppDelegate: send `remote_enable`, then ask for a link again.
    var onEnable: (() -> Void)?
    var onRetry: (() -> Void)?

    /// Ask for a fresh link (used by the Settings › Phone section).
    func refresh() { state.content = .loading; onRetry?() }
    func enable() { state.content = .loading; onEnable?() }

    func show(_ content: Content) {
        state.content = content
        if case .link(_, let pin) = content, !pin.isEmpty { state.pin = pin }
        if window == nil {
            let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 320, height: 420),
                             styleMask: [.titled, .closable, .fullSizeContentView],
                             backing: .buffered, defer: false)
            w.title = "Talk from Phone"
            w.titlebarAppearsTransparent = true
            w.titleVisibility = .hidden
            w.isMovableByWindowBackground = true
            w.isReleasedWhenClosed = false
            w.level = .floating
            let hc = NSHostingController(rootView: PhoneView(state: state, model: model, embedded: false,
                                                             enable: { [weak self] in self?.state.content = .loading; self?.onEnable?() },
                                                             retry: { [weak self] in self?.state.content = .loading; self?.onRetry?() },
                                                             close: { [weak w] in w?.close() }))
            hc.sizingOptions = []  // crash fix: with .fullSizeContentView, content-driven sizing loops (safe area ↔ frame) until AppKit throws
            w.setContentSize(NSSize(width: max(w.contentLayoutRect.width, hc.view.fittingSize.width), height: max(w.contentLayoutRect.height, hc.view.fittingSize.height)))
            w.contentViewController = hc
            w.center()
            window = w
        }
        NSApp.activate(ignoringOtherApps: true)
        window?.makeKeyAndOrderFront(nil)
    }

    /// Maps the daemon's `remote_link` reply to panel content.
    static func content(ok: Bool, data: [String: Any], error: String?) -> Content {
        if ok, let url = data["url"] as? String, !url.isEmpty {
            let pin = data["pin"].map { "\($0)" } ?? ""
            return .link(url, pin: pin == "<null>" ? "" : pin)
        }
        let e = (error ?? "").lowercased()
        if e.contains("disabled") || e.contains("not enabled") || e.contains("remote access is off") {
            let configured = (data["configured"] as? Bool) ?? !(e.contains("not configured") || e.contains("livekit"))
            return .disabled(canEnable: configured)
        }
        if e.contains("not configured") || e.contains("livekit") || e.contains("secret") || e.contains("credential") {
            return .disabled(canEnable: false)
        }
        return .error(error ?? "Couldn’t create a phone link.")
    }

    static func qrImage(_ text: String, side: CGFloat) -> NSImage? {
        let f = CIFilter.qrCodeGenerator()
        f.message = Data(text.utf8)
        f.correctionLevel = "M"
        guard let out = f.outputImage else { return nil }
        let scale = side / out.extent.width
        let scaled = out.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let rep = NSCIImageRep(ciImage: scaled)
        let img = NSImage(size: rep.size)
        img.addRepresentation(rep)
        return img
    }
}

@MainActor
final class PhoneState: ObservableObject {
    @Published var content: PhonePanel.Content = .loading
    @Published var pin = ""   // from the remote_link reply; the `remote` event may update it
}

struct PhoneView: View {
    @ObservedObject var state: PhoneState
    @ObservedObject var model: AppModel
    /// true inside Settings › Phone (no hero, no Done/Cancel buttons).
    var embedded: Bool = false
    var enable: () -> Void
    var retry: () -> Void
    var close: () -> Void
    @State private var copied = false

    var body: some View {
        VStack(spacing: 14) {
            if !embedded {
                Image(systemName: "iphone.radiowaves.left.and.right")
                    .font(.system(size: 26, weight: .medium)).foregroundStyle(Color.accentColor)
                    .accessibilityHidden(true)
                Text("Talk from your phone").font(.title3.weight(.semibold))
            }
            switch state.content {
            case .loading:
                ProgressView().controlSize(.small).padding(.vertical, 40)
                Text("Creating a private room…").font(.callout).foregroundStyle(.secondary)
            case .link(let url, _):
                if let qr = PhonePanel.qrImage(url, side: 200) {
                    Image(nsImage: qr).interpolation(.none).resizable().frame(width: 200, height: 200)
                        .padding(8).background(Color.white, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                        .accessibilityLabel("QR code for the phone link")
                }
                Text("Scan with your phone. The link works for 15 minutes.")
                    .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                let pin = model.remote.pin.isEmpty ? state.pin : model.remote.pin
                HStack(spacing: 10) {
                    Text(pin.isEmpty ? "Then say the PIN Mac shows you." : "Then say this PIN:")
                        .font(.callout.weight(.medium))
                    if !pin.isEmpty {
                        Text(pin.map(String.init).joined(separator: " "))
                            .font(.system(size: 30, weight: .semibold, design: .rounded)).monospacedDigit()
                            .foregroundStyle(Color.accentColor)
                            .accessibilityLabel("PIN \(pin.map(String.init).joined(separator: " "))")
                    }
                }
                .padding(.horizontal, 14).padding(.vertical, 8)
                .frame(maxWidth: .infinity)
                .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color.accentColor.opacity(0.10)))
                HStack {
                    Button(copied ? "Copied" : "Copy Link") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(url, forType: .string)
                        copied = true
                    }
                    if embedded { Button("New Link", action: retry) }
                    Spacer()
                    if !embedded { Button("Done", action: close).keyboardShortcut(.defaultAction) }
                }
            case .disabled(let canEnable):
                Image(systemName: "iphone.slash").font(.system(size: 36)).foregroundStyle(.secondary)
                    .padding(.vertical, 24).accessibilityHidden(true)
                if canEnable {
                    Text("Phone access is turned off.\nTurn it on to get a private link you can open anywhere.")
                        .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack {
                        if !embedded { Button("Cancel", action: close) }
                        Spacer()
                        Button("Turn On", action: enable).keyboardShortcut(.defaultAction)
                    }
                } else {
                    Text("Phone access needs a LiveKit account (free tier is fine).\nRun install.sh again and choose “Set up LiveKit”.")
                        .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                    if !embedded {
                        HStack {
                            Spacer()
                            Button("OK", action: close).keyboardShortcut(.defaultAction)
                        }
                    }
                }
            case .error(let msg):
                Image(systemName: "exclamationmark.triangle").font(.system(size: 36)).foregroundStyle(.orange)
                    .padding(.vertical, 24).accessibilityHidden(true)
                Text(msg).font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                HStack {
                    if !embedded { Button("Close", action: close) }
                    Spacer()
                    Button("Try Again", action: retry).keyboardShortcut(.defaultAction)
                }
            }
        }
        .padding(.top, embedded ? 8 : 34).padding(.horizontal, embedded ? 0 : 24).padding(.bottom, embedded ? 8 : 20)
        .frame(width: embedded ? nil : 320)
        .background(embedded ? Color.clear : Color(nsColor: .windowBackgroundColor))
    }
}
