# Hey Mac

**Stop touching your Mac. Just talk to it.**

Say *"Hey Mac, find three trending repos, put them in a note and send the list to my team on
WhatsApp"* and it does it, across apps, while you sit back. It answers in a human voice, tells
you what it found, and asks before anything leaves your machine. It also works from your
WhatsApp groups: start a message with "Mac" and it does the job.

<!-- DEMO -->

## Setup

**You need:** an Apple Silicon Mac (macOS 14+), [Claude Code](https://code.claude.com/docs/en/quickstart)
installed and signed in (`claude`, then `/login`), [uv](https://docs.astral.sh/uv/) (the installer
gets it if missing) and, for WhatsApp, [Go](https://go.dev/dl/) (`brew install go`).

```bash
git clone https://github.com/SamarthY06/Mac.git && cd Mac
./install.sh
```

The installer sets up Python, downloads the speech models (~700 MB), builds `Mac.app` and opens
it. The **Meet Mac** window walks you through the permissions and pairing Chrome. Then say
*"Hey Mac, what can you do?"*

**Optional keys**, to talk to Mac from your phone over LiveKit or to use an Anthropic API key
instead of your Claude login: store each one in the Keychain with `macctl secret set <NAME>`
(the names are in `.env.example`). A `.env` file in the repo also works for the app; git
ignores it.

## Voice

| Say | Mac does |
|---|---|
| "Hey Mac, what's on my calendar tomorrow?" | answers by voice |
| "Hey Mac, find 20 product managers in Bangalore on LinkedIn" | drives your Chrome, saves the list to `~/Documents/Mac/` |
| "Hey Mac, draft an email to Sam" → "send it" | the send waits for your spoken **yes** |
| "stop" | stops at once |

Menu bar: push-to-talk (⌥Space), Stop, Turn Off, Settings. Command line: `macctl`.

## WhatsApp

Mac links to your own WhatsApp as a linked device, like WhatsApp Web.

```bash
scripts/mac whatsapp link          # scan the QR code with your phone, then pick groups
scripts/services.sh install        # run it in the background, at every login
scripts/mac chats add "Family"     # add another group later
```

**In a group**, Mac acts only on your own messages that start with "Mac" ("Mac, find me 20
jobs"). Reply to a message with your request and Mac sees that message. Mac can tag people.
It reads the group's last 50 messages, other members' included, so it can follow the
conversation. Those messages go to Claude as context and are never treated as instructions.

**Turning things off:**

```bash
scripts/mac chats                              # each group and what it may use
scripts/mac chats deny "Family" reading group messages, the web
scripts/mac chats off "Family"                 # Mac ignores the group
scripts/mac chats remove "Family"              # forget it
```

You can also send "Mac, disallow reading group messages" in the group, or say "stop" to stop
a task.

## Safety

- Anything irreversible (send, post, pay, delete, upload) waits for your yes to a summary of
  that exact action.
- Mac never reads or writes the "Timetable" group. That never-list wins over every setting.
- In WhatsApp, only your own messages are requests. Everyone else's are context.
- The safety check is a rule-based filter, not a sandbox. Add only groups whose members you
  trust, and deny writing files where you don't need it.

## Updates

```bash
git pull && ./install.sh           # the app and the Python environment
scripts/services.sh update         # the background services, from the commit you have checked out
scripts/services.sh rollback       # back to the commit that ran before
scripts/mac status                 # what is running, and whether WhatsApp is connected
```

The services run a frozen copy of a commit, so editing the repo never changes what runs.

## Your data

Everything lives on your Mac, never in git: settings, memory and the WhatsApp link in
`~/.mac-agent/`, secrets in the Keychain, logs in `~/Library/Logs/Mac/` (no message text).
Audio never leaves your Mac. Your requests and what Mac reads for them go to Anthropic as part
of your Claude session.

## Uninstall

```bash
scripts/services.sh uninstall      # stop the background services
./install.sh --uninstall           # remove the app, settings and (if you say so) data
```

To unlink WhatsApp, remove this Mac under Settings ▸ Linked devices in WhatsApp on your phone.

## More

[docs/USAGE.md](docs/USAGE.md) has the details: every command, settings, Chrome, authentication,
the services, memory and development. Design: [ARCHITECTURE.md](ARCHITECTURE.md). License: MIT
([LICENSE](LICENSE)); the speech models carry their own licenses (see docs/USAGE.md).
