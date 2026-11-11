# FAQ

## Does it run on an Intel Mac?

No. OpenMac needs an Apple silicon Mac, and `install.sh` stops on any other machine.
The speech models (sherpa-onnx, Kokoro, Parakeet int8) run on the CPU through Apple's
Accelerate framework, and the Python wheels they ship are built for Apple silicon. On an
Intel Mac they would be too slow to answer you in conversation time, so there is no Intel
build.

## Does it run on Windows?

No. The Swift app uses Apple's voice-processing audio unit for echo cancellation, and the
tools work by driving your apps through AppleScript and Accessibility. Windows has none of
these, and Claude in Chrome's native host is set up for macOS as shipped. A Windows port
would mean rewriting the app and most of the tools.

## Does it run on Linux?

No, for the same reasons as Windows: the audio pipeline and the app automation are built on
macOS frameworks that Linux does not have.

## Does it run on older versions of macOS?

No. OpenMac needs macOS 14 (Sonoma) or newer. The Swift app is built for macOS 14, and
`install.sh` checks the version before it installs anything. Update macOS first, then run the
installer again.

## Can I use an API key instead of a Claude login?

Yes. Run `macctl secret set ANTHROPIC_API_KEY`, then set `auth = "api_key"` under `[brain]` in
`~/.mac-agent/config.toml`. See [USAGE.md](USAGE.md#authentication) for details. Claude in
Chrome still needs you to be signed in with a Claude login, so browser tasks will not work with
an API key alone.

## Does my audio leave my Mac?

No. The wake word, speech recognition, end-of-turn detection, and voice all run on your Mac.
The one exception is the optional phone feature: if you turn it on, call audio is relayed
through LiveKit. Your requests, and what OpenMac reads to answer them, go to Anthropic as part
of your Claude session.
