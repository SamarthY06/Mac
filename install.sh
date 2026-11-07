#!/usr/bin/env bash
# Mac — installer. Idempotent: safe to re-run after `git pull`.
#
#   ./install.sh               interactive install
#   ./install.sh --yes         non-interactive (assume "yes" to installs, skip optional prompts)
#   ./install.sh --no-models   skip speech model download
#   ./install.sh --no-open     don't launch the app at the end
#   ./install.sh --app-only    only rebuild + reinstall Mac.app (skip uv, models, LiveKit)
#   ./install.sh --uninstall   remove Mac.app, settings and (optionally) data
#
# Env: SIGN_IDENTITY (Developer ID for signing), LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET
#      (stored into the Keychain when given, also with --yes).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DAEMON_DIR="$ROOT/daemon"
APP_DEST="$HOME/Applications/Mac.app"
BUNDLE_ID="io.github.macagent.Mac"
KEYCHAIN_SERVICE="mac-agent"

YES=0; MODELS=1; UNINSTALL=0; OPEN=1; APP_ONLY=0
for a in "$@"; do
  case "$a" in
    -y|--yes) YES=1 ;;
    --no-models) MODELS=0 ;;
    --no-open) OPEN=0 ;;
    --uninstall) UNINSTALL=1 ;;
    --app-only) APP_ONLY=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

if [[ -t 1 ]]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; C=$'\033[36m'; O=$'\033[0m'; else B= G= Y= R= C= O=; fi
step() { echo; echo "${B}${C}==>${O}${B} $*${O}"; }
ok()   { echo "  ${G}✓${O} $*"; }
warn() { echo "  ${Y}!${O} $*"; }
die()  { echo "  ${R}✗ $*${O}" >&2; exit 1; }
ask()  {  # ask "question" default(y|n) → 0 for yes
  local q="$1" def="${2:-n}" ans
  if [[ $YES == 1 ]]; then [[ $def == y ]]; return; fi
  if [[ ! -t 0 ]]; then [[ $def == y ]]; return; fi
  read -r -p "  $q [$([[ $def == y ]] && echo Y/n || echo y/N)] " ans || true
  ans="${ans:-$def}"; [[ $ans =~ ^[Yy] ]]
}

quit_app() {
  if pgrep -f "Mac.app/Contents/MacOS/Mac" >/dev/null 2>&1; then
    pkill -TERM -f "Mac.app/Contents/MacOS/Mac" || true   # app stops its daemon on SIGTERM
    for _ in $(seq 1 40); do pgrep -f "Mac.app/Contents/MacOS/Mac" >/dev/null || break; sleep 0.25; done
  fi
}

# ------------------------------------------------------------------------- uninstall
if [[ $UNINSTALL == 1 ]]; then
  step "Uninstalling Mac"
  quit_app; ok "app stopped"
  if [[ -x "$APP_DEST/Contents/MacOS/Mac" ]]; then
    "$APP_DEST/Contents/MacOS/Mac" --unregister-login-item >/dev/null 2>&1 && ok "launch-at-login registration removed" || true
  fi
  rm -rf "$APP_DEST" && ok "removed $APP_DEST"
  [[ -L "$HOME/.local/bin/macctl" ]] && rm -f "$HOME/.local/bin/macctl" && ok "removed ~/.local/bin/macctl"
  defaults delete "$BUNDLE_ID" >/dev/null 2>&1 && ok "removed app preferences" || true
  tccutil reset All "$BUNDLE_ID" >/dev/null 2>&1 && ok "reset privacy permissions for $BUNDLE_ID" || true
  if ask "Delete LiveKit/API secrets from the Keychain (service \"$KEYCHAIN_SERVICE\")?" n; then
    for acct in LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET ANTHROPIC_API_KEY; do
      security delete-generic-password -s "$KEYCHAIN_SERVICE" -a "$acct" >/dev/null 2>&1 || true
    done
    ok "secrets removed"
  fi
  if ask "Delete ~/.mac-agent (config, models, memory, workspace)?" n; then
    rm -rf "$HOME/.mac-agent" && ok "removed ~/.mac-agent"
  fi
  if ask "Delete the local code-signing keychain (\"Mac Agent Local\")?" n; then
    security delete-keychain "$HOME/Library/Keychains/mac-agent-signing.keychain-db" >/dev/null 2>&1 || true
    rm -rf "$HOME/.mac-agent/signing"; ok "removed signing identity"
  fi
  echo; echo "${G}Mac has been uninstalled.${O} Logs remain in ~/Library/Logs/Mac."
  exit 0
fi

echo "${B}Mac — an always-on voice agent for macOS, powered by Claude${O}"

# ------------------------------------------------------------------------- system
step "Checking your system"
[[ $(uname -s) == Darwin ]] || die "Mac only runs on macOS."
[[ $(uname -m) == arm64 ]] || die "Mac requires an Apple silicon Mac (arm64); found $(uname -m)."
MACOS="$(sw_vers -productVersion)"
[[ ${MACOS%%.*} -ge 14 ]] || die "macOS 14 (Sonoma) or newer is required; you have $MACOS."
ok "macOS $MACOS on Apple silicon"
if ! xcrun --find swift >/dev/null 2>&1; then
  die "Swift toolchain not found. Install the Command Line Tools: xcode-select --install (then re-run)."
fi
ok "Swift $(xcrun swift --version 2>/dev/null | sed -n 's/.*Swift version \([0-9.]*\).*/\1/p' | head -1)"

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if [[ $APP_ONLY == 0 ]]; then
# ------------------------------------------------------------------------- uv
step "Python environment (uv)"
if ! command -v uv >/dev/null 2>&1; then
  if ask "uv (the Python package manager from Astral) is not installed. Install it now?" y; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  else
    die "uv is required: https://docs.astral.sh/uv/"
  fi
fi
ok "uv $(uv --version | awk '{print $2}')"
(cd "$DAEMON_DIR" && uv sync --quiet) || die "uv sync failed in $DAEMON_DIR"
ok "daemon dependencies installed ($DAEMON_DIR/.venv)"
mkdir -p "$HOME/.local/bin"
ln -sf "$DAEMON_DIR/.venv/bin/macctl" "$HOME/.local/bin/macctl"
ok "macctl linked to ~/.local/bin/macctl"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) warn "add ~/.local/bin to your PATH to use macctl from any terminal" ;; esac

# ------------------------------------------------------------------------- models
step "Speech models"
MODELS_OK=0
if [[ $MODELS == 0 ]]; then
  warn "skipped (--no-models). Mac can't hear or speak until you run: macctl models"
elif (cd "$DAEMON_DIR" && uv run --quiet python -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("macd.speech.models") else 1)') 2>/dev/null; then
  echo "  downloading speech models (~700 MB, one time; this is the slow step) …"
  if (cd "$DAEMON_DIR" && uv run python -m macd.speech.models); then
    ok "models ready"; MODELS_OK=1
  else
    warn "model download failed; re-run ./install.sh or: macctl models"
  fi
else
  warn "model downloader (macd.speech.models) not available in this checkout yet; later: macctl models"
fi

fi  # APP_ONLY

# ------------------------------------------------------------------------- claude
step "Claude Code"
# Official native installer (code.claude.com/docs/en/setup); brew/npm are alternatives.
CLAUDE_INSTALL_CMD='curl -fsSL https://claude.ai/install.sh | bash'
find_claude() {
  CLAUDE="$(command -v claude || true)"
  [[ -z $CLAUDE && -x $HOME/.local/bin/claude ]] && CLAUDE="$HOME/.local/bin/claude"
  return 0
}
claude_signed_in() {  # sets STATUS; 0 = signed in
  STATUS="$(perl -e 'alarm 20; exec @ARGV' "$CLAUDE" auth status 2>/dev/null || true)"
  grep -q '"loggedIn": *true' <<<"$STATUS"
}
find_claude
if [[ -z $CLAUDE ]]; then
  warn "The 'claude' CLI was not found. Mac uses your Claude Code sign-in."
  echo "    Install it with:  ${B}$CLAUDE_INSTALL_CMD${O}"
  echo "    (alternatives: brew install --cask claude-code  ·  npm install -g @anthropic-ai/claude-code)"
  if [[ $YES == 0 ]] && ask "Install Claude Code now?" n; then
    bash -c "$CLAUDE_INSTALL_CMD" || warn "the installer reported an error"
    find_claude
    if [[ -n $CLAUDE ]]; then ok "installed $CLAUDE"
    else warn "'claude' still isn't on PATH — open a new terminal and re-run ./install.sh"; fi
  else
    warn "Skipped. Install it, run 'claude' once to sign in, then re-run ./install.sh (or choose Restart Daemon in Mac's menu)."
  fi
fi
if [[ -n $CLAUDE ]]; then
  ok "claude $("$CLAUDE" --version 2>/dev/null | awk '{print $1}') at $CLAUDE"
  if claude_signed_in; then
    ok "signed in ($(sed -n 's/.*"authMethod": *"\([^"]*\)".*/\1/p' <<<"$STATUS"))"
  elif [[ -n $STATUS ]]; then
    warn "Claude Code is installed but not signed in."
    if [[ $YES == 0 ]] && ask "Sign in now? (opens 'claude' here — follow the prompts, then type /exit)" y; then
      "$CLAUDE" || true
      if claude_signed_in; then ok "signed in"
      else warn "still not signed in — run 'claude' later, then choose Restart Daemon in Mac's menu"; fi
    else
      warn "Run 'claude' in a terminal and sign in, then choose Restart Daemon in Mac's menu."
    fi
  else
    warn "Couldn't check the sign-in. Make sure running 'claude' works and you're signed in."
  fi
fi
warn "For browser tasks, install the Claude in Chrome extension and connect it to Claude Code."
[[ ${MAC_INSTALL_STOP_AFTER:-} == claude ]] && exit 0  # internal: script-path tests

if [[ $APP_ONLY == 0 ]]; then
# ------------------------------------------------------------------------- LiveKit
step "Talk from your phone (optional, LiveKit)"
have_secret() { security find-generic-password -s "$KEYCHAIN_SERVICE" -a "$1" >/dev/null 2>&1; }
store_secret() { security add-generic-password -U -s "$KEYCHAIN_SERVICE" -a "$1" -w "$2" >/dev/null; }
if [[ -n ${LIVEKIT_URL:-} && -n ${LIVEKIT_API_KEY:-} && -n ${LIVEKIT_API_SECRET:-} ]]; then
  store_secret LIVEKIT_URL "$LIVEKIT_URL"; store_secret LIVEKIT_API_KEY "$LIVEKIT_API_KEY"
  store_secret LIVEKIT_API_SECRET "$LIVEKIT_API_SECRET"
  ok "LiveKit credentials from the environment saved to the Keychain"
elif have_secret LIVEKIT_URL; then
  ok "LiveKit already configured (Keychain service \"$KEYCHAIN_SERVICE\")"
elif [[ $YES == 0 && -t 0 ]] && ask "Set up LiveKit now? (free account at https://cloud.livekit.io)" n; then
  read -r -p "  LiveKit URL (wss://….livekit.cloud): " lk_url
  read -r -p "  API key: " lk_key
  read -r -s -p "  API secret: " lk_secret; echo
  if [[ -n $lk_url && -n $lk_key && -n $lk_secret ]]; then
    store_secret LIVEKIT_URL "$lk_url"; store_secret LIVEKIT_API_KEY "$lk_key"; store_secret LIVEKIT_API_SECRET "$lk_secret"
    ok "saved to the Keychain. Enable it with [remote] enabled = true in ~/.mac-agent/config.toml"
  else
    warn "incomplete, skipped"
  fi
else
  ok "skipped (re-run ./install.sh to set it up later)"
fi

fi  # APP_ONLY

# ------------------------------------------------------------------------- app
step "Building Mac.app"
"$ROOT/scripts/bundle.sh"
quit_app
mkdir -p "$HOME/Applications"
rm -rf "$APP_DEST"
ditto "$ROOT/build/Mac.app" "$APP_DEST"
ok "installed $APP_DEST"
defaults write "$BUNDLE_ID" daemonDir "$DAEMON_DIR"
ok "daemonDir = $DAEMON_DIR"
if codesign -dv "$APP_DEST" 2>&1 | grep -q "Signature=adhoc"; then
  warn "Mac.app is ad-hoc signed: macOS will ask for permissions again after every rebuild."
fi

# First-minute collision: don't launch (and prompt for permissions) while the models are
# still missing — the app would show "voice unavailable" right away.
if [[ $OPEN == 1 && ( $MODELS_OK == 1 || $MODELS == 0 || $APP_ONLY == 1 ) ]]; then
  open "$APP_DEST"
  ok "Mac is running — look for the waveform icon in the menu bar"
elif [[ $OPEN == 1 ]]; then
  warn "not opening Mac.app because the speech models are missing; run 'macctl models', then: open ~/Applications/Mac.app"
fi

echo
echo "${B}Next steps${O}"
echo "  1. In the Permissions window, allow, in order: Microphone, Accessibility, Automation,"
echo "     Screen Recording, and (optional) Notifications. Menu bar icon → Permissions… reopens it."
echo "  2. Say ${B}“Hey Mac, what can you do?”${O}  — or press ${B}⌥Space${O} to talk."
echo "  3. Settings: ~/.mac-agent/config.toml   Logs: ~/Library/Logs/Mac/   CLI: macctl status"
echo "  Optional: menu bar icon → Launch at login."
