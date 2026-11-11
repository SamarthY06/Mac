# FAQ

## Does OpenMac run on an Intel Mac?

No. OpenMac needs Apple silicon. The speech models (sherpa-onnx, Kokoro and
Parakeet) ship as Apple silicon builds and run on the CPU with Accelerate, and
the app uses Apple's voice-processing audio unit for echo cancellation. An
Intel build would also be too slow: it would break the latency budget the
design sets for replies. The installer checks for arm64 and stops if it does
not find it.

## Does OpenMac run on Windows?

No. The tools Mac uses to control your computer are macOS features:
AppleScript, the Accessibility APIs and the native host that Claude in Chrome
ships for macOS. Windows has none of these, so there is nothing for Mac to
stand on.

## Does OpenMac run on Linux?

No, for the same reason as Windows. OpenMac is built on macOS automation and
Apple's audio stack, and the installer refuses anything that is not macOS.

## Which macOS versions work?

macOS 14 (Sonoma) or newer. The installer checks the version and stops on
anything older.

## Can I use an API key instead of a Claude login?

Yes. Run `macctl secret set ANTHROPIC_API_KEY` and set `auth = "api_key"`
under `[brain]` in `~/.mac-agent/config.toml`. One limit: Claude in Chrome
only works with a claude.ai sign-in, so browser control needs the login even
if everything else uses the key.

## Does audio leave my Mac?

No. Speech recognition and speech run on your Mac, locally and free. The one
exception is the optional phone feature: if you turn it on, it relays audio
through LiveKit.
