# Using Mac: the details

The [README](../README.md) is the one-screen version. This page has the rest.

## What it can do

- **Works across your apps**: Chrome (your own logged-in browser), Mail, WhatsApp, Notes,
  Calendar, Reminders, Finder, Terminal. "Draft an email to Sam" → "paste it in WhatsApp" →
  "looks good, send it".
- **Sees your screen**: "what's this error?" and it looks.
- **Works with the screen off or locked**: voice, Chrome, files and app automation keep going.
- **Talks like a person**: short answers, no step-by-step chatter, one natural voice. Pick from
  54 voices in 9 languages, or clone your own voice locally.
- **Teaches**: "Hey Mac, teach me consistent hashing" turns it into a conversational tutor.
- **Remembers you**: your name, preferences and projects, in a plain `memory.md` you can read.
- **Safe by design**: anything irreversible (send, post, pay, delete, upload) is blocked by a
  hook until you say "yes" to a spoken summary of that exact action.
- **Light**: the app idles at about 1 % CPU; Claude starts when you say the wake word and
  shuts down after 10 quiet minutes. A real Off switch releases the microphone.
- **Optional**: talk to your Mac from your phone over LiveKit (QR code, spoken PIN).

The brain is **Claude** (the full Claude Code agent loop, through the Claude Agent SDK) with
your own login. Wake word (sherpa-onnx), speech recognition (Parakeet) and voice (Kokoro,
Kyutai Pocket TTS for cloning) run **locally and free**. Audio never leaves your Mac.

## Connect Claude and Chrome

Mac uses **your** Claude and **your** Chrome. The Meet Mac window walks you through both and
shows live status; here is what happens underneath.

1. **Claude**, install [Claude Code](https://code.claude.com/docs/en/quickstart) and sign in
   (`claude`, then `/login`). Mac runs your `claude` with your claude.ai plan.
   An API key works too (`macctl secret set ANTHROPIC_API_KEY`, `auth = "api_key"`), but
   **Claude in Chrome only works with a claude.ai sign-in**, per Anthropic's docs.
2. **Chrome**, install the
   [Claude in Chrome extension](https://chromewebstore.google.com/detail/claude/fcoeoabgfenejglbffodgkkbkcdhcgfn)
   (v1.0.36+), sign it in to the same claude.ai account, then run `claude --chrome` once in a
   terminal: press Enter at the dialog, type `/chrome` and choose **Enabled by default**.
   That writes Claude's native-messaging host file and pairs the extension. Mac's
   *Check connection* button verifies all of it (Chrome running · extension · native host ·
   enabled · connected). Works with Chrome, Edge, Brave and Arc.

## Using it

| Say | Mac does |
|---|---|
| "Hey Mac, what's on my calendar tomorrow?" | answers by voice |
| "Hey Mac, find me 20 product managers in Bangalore on LinkedIn" | drives Chrome, narrates progress, reads the names, saves the list to `~/Documents/Mac/`, notifies you |
| "Hey Mac, draft an email to Sam about the demo" → "paste it in WhatsApp to Sam" → "looks good, send it" | multi-app flow; the send waits for your spoken **yes** |
| "Hey Mac, teach me system design interviews" | tutor mode, in short spoken chunks |
| "stop" / "never mind" (while Mac is talking) | stops immediately |
| "Hey Mac, go to sleep" | turns listening off (menu bar → Turn On to wake) |

After Mac answers, you have a few seconds to follow up **without** the wake word.
Menu bar: Talk to Mac (⌥Space push-to-talk), Stop, Turn Off/On, Mute, New Conversation,
Talk from Phone…, Permissions…, Launch at Login, logs, settings.

Command line (`macctl` is linked into `~/.local/bin`):

```
macctl status | ask "text" [--wait] | say "text" | stop | mute on|off | listen
macctl off | on | reset | remote-link | remote-end | logs | doctor [--deep] | models
macctl voice set sample.wav | voice reset | voice status   # voice cloning (Pocket TTS)
macctl secret set LIVEKIT_URL      # secrets live in the macOS Keychain
```

## Settings

Menu bar → **Settings…** (⌘,), also shown right after onboarding:

- **General**, Mac's name (the wake word follows it: "Hey Jarvis"), your name, the language Mac
  speaks (English US/UK, Spanish, French, Hindi, Italian, Japanese, Portuguese, Chinese, you
  still speak English for now), idle shutdown, follow-up window, launch at login.
- **Voice**, 54 built-in voices with ▶ preview, speed, and your cloned voice.
- **Appearance**, your picture (Mac's portrait), HUD position.
- **Permissions**, **Phone**, **Advanced** (Chrome, your own Claude MCP servers/skills, results
  folder, config and logs).

By voice: "Hey Mac, change your name to Jarvis", "use your default voice".
CLI: `macctl settings get`, `macctl settings set tts.voice=bm_george tts.speed=1.1`, `macctl voices`,
`macctl voice preview af_sky`.

## Authentication

- **Default (`auth = "claude_cli"`)**: Mac runs your installed `claude` with your own login.
  Anthropic's Agent SDK terms say third-party products may not *offer* claude.ai login to their
  users without approval, Mac is an open-source tool you install on your own machine with your
  own account; if you distribute a build to others, read those terms.
- **API key (`auth = "api_key"`)**: `macctl secret set ANTHROPIC_API_KEY`, then set
  `auth = "api_key"` under `[brain]` in `~/.mac-agent/config.toml`.

## Configuration

`~/.mac-agent/config.toml` (created with defaults on first run; `MAC_*` env vars override).
Notable keys: `wake.phrases` (default `["hey mac", "ok mac"]`), `listen.smart_turn` (end of turn by the
Smart Turn model; when it hears an unfinished sentence Mac waits up to `listen.turn_max_silence_ms`, 2000),
`listen.silence_ms` (used without the model),
`listen.followup_s`, `tts.voice`, `tts.engine` (`kokoro` | `clone`; the *Voice…* panel sets it), `brain.idle_shutdown_min`, `brain.setting_sources`
(`["user"]` loads your own `~/.claude` MCP servers, skills and plugins into Mac; `[]` isolates),
`results_dir` (`~/Documents/Mac`), `remote.enabled`.

## Screen off, screen locked

Mac keeps listening with the display off or the screen locked (it runs in your user session, not
on the lock screen). *Settings → General → Stay awake while listening* keeps the Mac from
sleeping while plugged in. While locked: voice, Chrome, shell, files and AppleScript automation
(Mail, Calendar, Notes, Messages) all work; clicking/typing inside apps and screenshots wait
until you unlock, Mac knows the difference and never tries to unlock. A closed MacBook lid still
sleeps the Mac unless an external display is attached.

Ask "what's on my screen?" or "what's this error?" and Mac looks at your screen (Screen
Recording permission).

## Memory

Mac keeps a small `~/.mac-agent/workspace/memory.md`, your name, how you like answers, your
projects, people you mention, recurring tasks, and updates it on its own from what *you* say
(never from web pages or documents it reads). Ask "what do you remember about me?" or say
"forget that". Before a conversation is retired, Mac writes a short "recent context" note so
"what were we doing?" works after a restart.

## The background services (WhatsApp)

The WhatsApp bot (chatd) and the agent behind it (agentd) can run in the background on their own.
They start when you log in, restart by themselves if they crash, and need no open terminal.

```bash
scripts/services.sh install     # start now and at every login
scripts/services.sh status      # what is running, from which commit, and whether WhatsApp is connected
scripts/services.sh update      # switch to the commit you have checked out and restart
scripts/services.sh rollback    # go back to the commit that ran before
scripts/services.sh restart     # restart both (also brings back a bot that stopped)
scripts/services.sh uninstall   # stop and remove
```

The service runs a frozen copy of the last commit, kept in `~/.mac-agent/release/`, never the
files you are editing. Changes reach it only when you commit and run `update`. `install` and
`update` refuse to run while the repo has uncommitted changes unless you add `--allow-dirty`
(the last commit is still what runs). The last 3 copies are kept for `rollback`. Building needs
Go, because the WhatsApp helper is built from that copy. After a dependency change, run
`install.sh` before `update`.

If the bot stops for a reason a restart can't fix (its settings are wrong, its message budget is
spent, or WhatsApp is in use by another run), it stays stopped and `status` says why. The agent
keeps running. Fix the cause, then run `restart`. Logs are in `~/Library/Logs/Mac/services.log`.
They never contain message text.

macOS may ask once whether Python may use files in the folder that holds this repo (for example
Desktop or Documents). Allow it, or keep the repo outside those folders.

## How it works

See [ARCHITECTURE.md](../ARCHITECTURE.md) (design, concurrency model, safety model, latency
budget) and [docs/DESIGN.md](DESIGN.md) (UI). Short version:

```
Mac.app (Swift)  ── mic w/ echo cancellation ──►  macd (Python)
   menu bar, HUD, permissions,                    wake word → VAD → STT → orchestrator
   supervises the daemon                          → Claude (Agent SDK, --chrome) → TTS
                 ◄── speech audio ────────────────  safety hook gates outward actions
```

Everything irreversible goes through a `PreToolUse` hook that classifies each tool call
(shell commands are parsed, script files are read, the frontmost app and current browser tab
are checked). An outward action is refused until the orchestrator has heard you say "yes" to a
spoken summary; the resulting token is bound to that one action and expires in 120 s.
Prompt injection from web pages cannot bypass it, because the model never holds the token.

## Development

```bash
make app          # build + sign build/Mac.app
make test         # Swift + Python tests
cd daemon && uv run pytest -q -m "not live and not models and not perf"   # fast unit suite
cd daemon && uv run pytest -q -s -m perf                       # benchmarks, run alone (W6 latency gate)
cd daemon && uv run pytest -q -m models                       # needs downloaded models
cd daemon && uv run pytest -q -m live                         # needs claude login / LiveKit creds
uv run macd --audio sounddevice   # headless daemon (permissions attribute to your terminal)
```

Reviews that shaped the code (architecture, code quality, simplicity, UX) are in `docs/reviews/`.

## The mac command

`scripts/mac` links WhatsApp and manages the groups Mac answers in. Group names match
WhatsApp's, ignoring case and extra spaces.

```
scripts/mac whatsapp link                      show a QR code, link, pick groups
scripts/mac chats [list]                       the groups Mac has settings for
scripts/mac chats add "<group>"                answer in a group
scripts/mac chats on|off "<group>"             turn a group on or off
scripts/mac chats allow|deny "<group>" <what>  for example: deny "Family" Chrome, the web
scripts/mac chats remove "<group>"             forget a group's settings
scripts/mac status                             same as scripts/services.sh status
```

What a group may use: reading files, writing files, Chrome, the web, the desktop, sending
files, sending messages for you, reading group messages. New groups start with everything on.

`link` stops the background services while it runs, because WhatsApp's data allows one user at
a time, and starts them again afterwards. It saves your chat list to
`~/.mac-agent/chat/pairing.json`. "Timetable" is kept there so it stays blocked, but it is never
offered. The `chats` commands need the services running (`scripts/services.sh install`); they
talk to Mac's agent with a token the services make at each start
(`~/.mac-agent/run/app-token`, readable only by you).

## License

MIT, see [LICENSE](../LICENSE). Speech models carry their own licenses (sherpa-onnx: Apache-2.0;
Parakeet: CC-BY-4.0; Kokoro: Apache-2.0; Silero VAD: MIT; Pipecat Smart Turn v3.2: BSD-2-Clause; Kyutai Pocket TTS weights: CC-BY-4.0;
only clone voices you own or have permission to use).
