# Contributing to OpenMac

Thanks for looking. This page covers how to set up a development checkout, where things live,
and what a pull request needs. Issues and feature ideas go through the templates on GitHub.

## Dev setup

You need the same things as a user: an Apple Silicon Mac on macOS 14 or newer, Claude Code
signed in, the Xcode Command Line Tools, and [uv](https://docs.astral.sh/uv/). Go is optional
(see the bridge rule below).

```bash
git clone https://github.com/SamarthY06/OpenMac.git ~/OpenMac && cd ~/OpenMac
cd daemon && uv sync && cd ..       # Python 3.12 or 3.13, pinned by uv.lock
make app                            # build + sign build/OpenMac.app
make test                           # Swift unit tests + the fast Python suite
```

The Python tests use pytest markers. The fast suite is what CI runs:

```bash
cd daemon && uv run pytest -q -m "not live and not models and not perf"
cd daemon && uv run pytest -q -m models        # needs the downloaded speech models
cd daemon && uv run pytest -q -m live          # needs a Claude login, real hardware
cd daemon && uv run pytest -q -s -m perf       # latency benchmarks, run alone
cd daemon && uv run ruff check src tests
```

`./install.sh` is the full install (models, app in `~/Applications`). You do not need it to
work on the code, but it is the fastest way to get the models.

## Where things live

| Path | What |
|---|---|
| `app/` | Swift menu bar app: audio (mic, echo cancellation, speaker), HUD, permissions, onboarding. It supervises the daemon. |
| `daemon/src/macd/` | The voice daemon: wake word, VAD, STT, Smart Turn, orchestrator, TTS, the Claude brain, the safety gate (`brain/gate.py`, `brain/hooks.py`). `macctl` is `macd/cli.py`. |
| `daemon/src/agentd/` | The agent service: sessions, per-chat profiles, the never-list, memory. The voice daemon and the WhatsApp bot are its clients. |
| `daemon/src/chatd/` | The WhatsApp bot: the runner, group context, the bridge adapter. |
| `daemon/src/macsvc/` | The background services and `scripts/mac` (`macsvc/cli.py`). |
| `bridges/whatsapp/` | The Go bridge (whatsmeow) that speaks to WhatsApp over stdio. Protocol in `docs/whatsapp/bridge-protocol.md`. |
| `scripts/` | `bundle.sh` (build the app), `services.sh`, `mac`, `e2e.sh`. |
| `docs/` | `USAGE.md`, `DESIGN.md`, `whatsapp/`, and `reviews/` (the reviews and decisions that shaped the code). |

`ARCHITECTURE.md` is the design document: concurrency model, safety model, latency budget.
Read the relevant section before changing the orchestrator or the gate.

## Running the voice daemon from the tree

```bash
cd daemon && uv run macd --audio sounddevice
```

This runs the daemon headless with the Python audio backend, without the app. Microphone and
other permissions attribute to your terminal. With the app built, `make run` opens
`build/OpenMac.app`, which starts the daemon itself. `macctl status` and `macctl logs` work
against either.

For the WhatsApp side, `scripts/services.sh install` runs a frozen copy of your last commit,
never the working tree; `scripts/services.sh update` after each commit you want to test.

## The bridge rule

**Any change under `bridges/whatsapp/` needs a `VERSION` bump and a GitHub release.** Installs
on Macs without Go do not build the bridge; they download the prebuilt binary for the
snapshot's `VERSION` from the release of that version. A bridge change without a new release
leaves those installs running the old bridge against new Python code.

## Pull requests

- Small PRs, one change each. Say what and why in the description; the template asks.
- Tests for behaviour changes. The gate (`daemon/tests/test_gate*.py`,
  `test_confirmation_gate.py`, `test_hooks*.py`) is covered by security-audit tests; a change
  there needs a test that shows the old behaviour and the new one.
- Run `make test` and `uv run ruff check src tests` before pushing. CI runs the same.
- **No message text in logs.** Logs under `~/Library/Logs/Mac/` never contain what the user
  said, what a group said, or what Mac replied. Log ids, counts and timings.
- **No secrets.** Nothing from the Keychain, `.env`, or `~/.mac-agent/` goes into the repo,
  tests, fixtures or logs.
- Nothing personal: no real group names, phone numbers or email addresses, in code, tests
  or docs. Use generic examples ("Family").
- Do not widen what the gate allows to make a task work. If a legitimate command is blocked,
  open an issue with the exact command and the gate's reason.

## Writing user-facing text

Spoken lines, HUD text, CLI output, error messages and docs follow the same rules: plain,
concrete, short. No hype words. **No em dashes**: use commas, colons or full stops. American
spelling. Error messages say what happened and what to do next, in one or two sentences.

## License

By contributing you agree that your contribution is licensed under the MIT license in
[LICENSE](LICENSE).
