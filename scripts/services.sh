#!/bin/bash
# Run Mac's background services (agentd and the WhatsApp bot, chatd) as a launchd LaunchAgent.
# They start at login, restart after a crash and need no open terminal.
#
#   scripts/services.sh install     write the LaunchAgent and start it
#   scripts/services.sh uninstall   stop it and remove the LaunchAgent
#   scripts/services.sh status      say what is running
#   scripts/services.sh restart     restart everything (also brings back a stopped bot)
set -euo pipefail

LABEL="com.mac-agent.services"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/daemon/.venv/bin/python"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGS="$HOME/Library/Logs/Mac"
STATUS="${MAC_HOME:-$HOME/.mac-agent}/run/services.json"
DOMAIN="gui/$(id -u)"

xml() { sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' <<<"$1"; }

loaded() { launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; }

install() {
  if [[ ! -x "$PY" ]]; then
    echo "The Python environment is missing ($PY). Run install.sh first." >&2
    exit 1
  fi
  mkdir -p "$HOME/Library/LaunchAgents" "$LOGS"
  cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$(xml "$PY")</string>
    <string>-m</string>
    <string>macsvc</string>
  </array>
  <key>WorkingDirectory</key><string>$(xml "$REPO/daemon")</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>$(xml "$HOME/.local/bin"):/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key><false/>
  </dict>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$(xml "$LOGS/services.launchd.log")</string>
  <key>StandardErrorPath</key><string>$(xml "$LOGS/services.launchd.log")</string>
</dict>
</plist>
EOF
  plutil -lint "$PLIST" >/dev/null
  if loaded; then
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  fi
  launchctl bootstrap "$DOMAIN" "$PLIST"
  echo "Installed. Mac's services start now and at every login."
  echo "Logs: $LOGS/services.log"
}

uninstall() {
  if loaded; then
    launchctl bootout "$DOMAIN/$LABEL" || true
  fi
  rm -f "$PLIST"
  echo "Uninstalled. Mac's services are stopped and won't start at login."
}

restart() {
  if ! loaded; then
    echo "The services aren't installed. Run: scripts/services.sh install" >&2
    exit 1
  fi
  launchctl kickstart -k "$DOMAIN/$LABEL"
  echo "Restarted."
}

status() {
  if loaded; then
    echo "Installed: yes (starts at login)."
  elif [[ -f "$PLIST" ]]; then
    echo "Installed: the LaunchAgent file exists but isn't loaded. Run: scripts/services.sh install"
  else
    echo "Installed: no. Run: scripts/services.sh install"
  fi
  if [[ ! -f "$STATUS" ]]; then
    echo "No status yet: the services have not run."
    return
  fi
  "$PY" - "$STATUS" <<'EOF'
import json, os, sys, time

path = sys.argv[1]
try:
    s = json.load(open(path))
except (OSError, ValueError):
    print("The status file can't be read.")
    sys.exit(0)

def ago(t):
    d = max(0, int(time.time() - (t or 0)))
    if d < 90:
        return f"{d} seconds"
    if d < 5400:
        return f"{d // 60} minutes"
    if d < 172800:
        return f"{d // 3600} hours"
    return f"{d // 86400} days"

try:
    os.kill(int(s.get("pid") or 0), 0)
    up = not s.get("stopping")
except (OSError, ValueError):
    up = False
if not up:
    print(f"Supervisor: not running (last update {ago(s.get('updated_at'))} ago).")
else:
    print(f"Supervisor: running for {ago(s.get('started_at'))}.")
names = {"agentd": "Agent (agentd)", "chatd": "WhatsApp bot (chatd)"}
for name, c in (s.get("children") or {}).items():
    label = names.get(name, name)
    state = c.get("state", "unknown")
    restarts = c.get("restarts") or 0
    extra = f", restarted {restarts} time{'s' if restarts != 1 else ''}" if restarts else ""
    if state == "running" and up:
        print(f"{label}: running for {ago(c.get('since'))}{extra}.")
    else:
        print(f"{label}: {state}. {c.get('why', '')}".rstrip())
wa = s.get("whatsapp")
if up and wa:
    detail = wa.get("detail") or ""
    print(f"WhatsApp: {wa.get('state', 'unknown')}." + (f" {detail}" if detail else ""))
EOF
}

case "${1:-}" in
  install) install ;;
  uninstall) uninstall ;;
  status) status ;;
  restart) restart ;;
  *) echo "Usage: $0 install|uninstall|status|restart" >&2; exit 2 ;;
esac
