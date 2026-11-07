import Carbon.HIToolbox
import Foundation

/// Global hotkey via Carbon `RegisterEventHotKey`.
///
/// Chosen over `NSEvent.addGlobalMonitorForEvents` because it needs no Accessibility /
/// Input Monitoring permission and actually *consumes* the key combo (a global monitor
/// can only observe). Default: ⌥Space. If another app already owns the combo,
/// registration fails and we log it (the menu item still works).
@MainActor
final class Hotkey {
    private var ref: EventHotKeyRef?
    private var handler: EventHandlerRef?
    private let action: () -> Void
    nonisolated(unsafe) private static var current: Hotkey?

    init(keyCode: UInt32 = UInt32(kVK_Space), modifiers: UInt32 = UInt32(optionKey), action: @escaping () -> Void) {
        self.action = action
        Hotkey.current = self
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), { _, _, _ in
            DispatchQueue.main.async { MainActor.assumeIsolated { Hotkey.current?.action() } }
            return noErr
        }, 1, &spec, nil, &handler)
        let id = EventHotKeyID(signature: OSType(0x4D414321 /* 'MAC!' */), id: 1)
        let st = RegisterEventHotKey(keyCode, modifiers, id, GetApplicationEventTarget(), 0, &ref)
        if st != noErr {
            Log.warn("could not register global hotkey ⌥Space (OSStatus \(st)); is another app using it?", "ui")
        } else {
            Log.info("global hotkey ⌥Space registered", "ui")
        }
    }
}
