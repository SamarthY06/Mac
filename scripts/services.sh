#!/bin/bash
# Run Mac's background services (agentd and the WhatsApp bot, chatd) as a launchd LaunchAgent.
# They start at login, restart after a crash and need no open terminal.
#
# The service never runs the working tree. install and update take a frozen snapshot of a commit
# (git archive of HEAD) into ~/.mac-agent/release/<sha>, build the WhatsApp bridge from it, copy
# the bridge to ~/.mac-agent/bin/wa-bridge, and point the LaunchAgent at that snapshot's code.
# Edits in the repo change nothing until the next update.
#
#   scripts/services.sh install  [--allow-dirty]   snapshot HEAD, write the LaunchAgent, start it
#   scripts/services.sh update   [--allow-dirty]   snapshot HEAD, switch to it, restart
#   scripts/services.sh rollback                   switch back to the previous snapshot, restart
#   scripts/services.sh uninstall                  stop and remove the LaunchAgent (keeps snapshots)
#   scripts/services.sh status                     what is running, and from which commit
#   scripts/services.sh restart                    restart (also brings back a stopped bot)
#
# The repo's Python environment (daemon/.venv) supplies the dependencies; the snapshot supplies
# Mac's own code. After a dependency change, run install.sh before update.
set -euo pipefail

LABEL="com.mac-agent.services"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/daemon/.venv/bin/python"
MAC_DIR="${MAC_HOME:-$HOME/.mac-agent}"
RELEASES="$MAC_DIR/release"
BRIDGE="$MAC_DIR/bin/wa-bridge"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGS="$HOME/Library/Logs/Mac"
STATUS="$MAC_DIR/run/services.json"
KEEP=3
LAUNCHCTL="${LAUNCHCTL:-launchctl}"
DOMAIN="gui/$(id -u)"

xml() { sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' <<<"$1"; }
die() { echo "$*" >&2; exit 1; }

loaded() { "$LAUNCHCTL" print "$DOMAIN/$LABEL" >/dev/null 2>&1; }
SHA_RE='^[0-9a-f]{40}$'

# One run at a time: a lock dir holding the owner's pid (a dead owner's lock is taken over).
LOCK="$RELEASES/.lock"
take_lock() {
  mkdir -p "$RELEASES"
  if ! mkdir "$LOCK" 2>/dev/null; then
    local pid
    pid="$(cat "$LOCK/pid" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      die "Another services.sh run is in progress. Try again when it has finished."
    fi
    rm -rf "$LOCK"
    mkdir "$LOCK" 2>/dev/null || die "Another services.sh run is in progress. Try again when it has finished."
  fi
  echo $$ >"$LOCK/pid"
  trap 'rm -rf "$LOCK"' EXIT
}

# Delete one snapshot or staging dir, and only if it really is one: a <40-hex sha> or .new.<x>
# directory sitting directly in $RELEASES (resolved, so no symlink or .. leads elsewhere).
remove_release() {
  local name="$1" root real
  [[ "$name" =~ $SHA_RE || "$name" =~ ^\.new\.[A-Za-z0-9]+$ ]] || return 0
  [[ -d "$RELEASES/$name" && ! -L "$RELEASES/$name" ]] || return 0
  root="$(cd "$RELEASES" && pwd -P)"
  real="$(cd "$RELEASES/$name" && pwd -P)"
  [[ "$real" == "$root/$name" ]] || return 0
  rm -rf "$real"
}

current_sha() { [[ -L "$RELEASES/current" ]] && basename "$(readlink "$RELEASES/current")" || true; }

check_tree() {  # $1: --allow-dirty or empty
  git -C "$REPO" rev-parse --verify HEAD >/dev/null 2>&1 || die "This repo has no commit to run."
  if [[ "${1:-}" != "--allow-dirty" && -n "$(git -C "$REPO" status --porcelain)" ]]; then
    die "The repo has uncommitted changes, and the service runs the last commit only.
Commit them first, or run again with --allow-dirty to use the last commit anyway."
  fi
  [[ -x "$PY" ]] || die "The Python environment is missing ($PY). Run install.sh first."
}

# Snapshot HEAD into $RELEASES/<sha> with a built bridge. Prints the sha.
snapshot() {
  local sha dir tmp
  sha="$(git -C "$REPO" rev-parse HEAD)"
  dir="$RELEASES/$sha"
  if [[ ! -f "$dir/.complete" ]]; then
    mkdir -p "$RELEASES"
    tmp="$(mktemp -d "$RELEASES/.new.XXXXXX")"
    git -C "$REPO" archive --format=tar HEAD | tar -x -C "$tmp"
    if ! make -s -C "$tmp/bridges/whatsapp" build >&2; then
      rm -rf "$tmp"
      die "The WhatsApp bridge did not build from commit ${sha:0:12}. Is Go installed?"
    fi
    [[ -x "$tmp/bridges/whatsapp/bin/wa-bridge" ]] || { rm -rf "$tmp"; die "The bridge build made no binary."; }
    touch "$tmp/.complete"
    rm -rf "$dir"
    mv "$tmp" "$dir"
  fi
  make_newest "$sha"
  echo "$sha"
}

write_plist() {  # $1: snapshot dir
  local src="$1/daemon/src" extra=""
  [[ -n "${MAC_HOME:-}" ]] && extra="<key>MAC_HOME</key><string>$(xml "$MAC_HOME")</string>"
  mkdir -p "$HOME/Library/LaunchAgents" "$LOGS"
  cat >"$PLIST.tmp" <<EOF
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
  <key>WorkingDirectory</key><string>$(xml "$src")</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONPATH</key><string>$(xml "$src")</string>
    <key>PATH</key><string>$(xml "$HOME/.local/bin"):/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    $extra
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key><false/>
  </dict>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ExitTimeOut</key><integer>40</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$(xml "$LOGS/services.launchd.log")</string>
  <key>StandardErrorPath</key><string>$(xml "$LOGS/services.launchd.log")</string>
</dict>
</plist>
EOF
  plutil -lint "$PLIST.tmp" >/dev/null
  mv "$PLIST.tmp" "$PLIST"
}

stop_agent() {  # bootout, then wait until launchd has really let go (up to 60 s)
  loaded || return 0
  "$LAUNCHCTL" bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
  local i
  for i in $(seq 1 120); do
    loaded || return 0
    sleep 0.5
  done
  die "The services did not stop. Try again in a minute."
}

# Point everything at snapshot $1 and (re)load the LaunchAgent.
switch_to() {
  local sha="$1" dir="$RELEASES/$1"
  [[ -f "$dir/.complete" ]] || die "Snapshot ${sha:0:12} is missing."
  stop_agent
  mkdir -p "$(dirname "$BRIDGE")" || return 1
  install -m 0755 "$dir/bridges/whatsapp/bin/wa-bridge" "$BRIDGE.tmp" || return 1
  mv -f "$BRIDGE.tmp" "$BRIDGE" || return 1
  ln -sfn "$sha" "$RELEASES/current" || return 1
  write_plist "$dir" || return 1
  "$LAUNCHCTL" bootstrap "$DOMAIN" "$PLIST" >&2 || return 1
}

make_newest() {  # $1: sha; history is newest last
  { grep -vx "$1" "$RELEASES/history" 2>/dev/null || true; echo "$1"; } >"$RELEASES/history.tmp"
  mv "$RELEASES/history.tmp" "$RELEASES/history"
}

# Switch to $1; if launchd won't start it, go back to $2 (when there is one) and exit 1.
switch_or_back() {
  local new="$1" back="${2:-}"
  switch_to "$new" && return 0
  if [[ -n "$back" && "$back" != "$new" && -f "$RELEASES/$back/.complete" ]]; then
    echo "Commit ${new:0:12} did not start, so Mac went back to commit ${back:0:12}." >&2
    switch_to "$back" || die "Commit ${back:0:12} did not start either. Mac's services are stopped. See $LOGS/services.launchd.log"
    make_newest "$back"
    echo "Mac's services are running commit ${back:0:12} again." >&2
    exit 1
  fi
  die "Commit ${new:0:12} did not start. Mac's services are stopped. See $LOGS/services.launchd.log"
}

prune() {  # keep the newest $KEEP snapshots in history and the current one; runs under the lock
  local cur keep_list name path
  cur="$(current_sha)"
  keep_list="$( (grep -E "$SHA_RE" "$RELEASES/history" || true) | tail -n "$KEEP")"
  for path in "$RELEASES"/* "$RELEASES"/.new.*; do
    name="${path##*/}"
    [[ -e "$path" && "$name" != "$cur" && "$name" != current ]] || continue
    if [[ "$name" =~ $SHA_RE ]] && grep -qx "$name" <<<"$keep_list"; then
      continue
    fi
    remove_release "$name"  # old snapshots, and staging dirs a crashed run left (we hold the lock)
  done
  { grep -E "$SHA_RE" "$RELEASES/history" | while read -r name; do
      [[ -f "$RELEASES/$name/.complete" ]] && echo "$name"
    done || true; } >"$RELEASES/history.tmp"
  mv "$RELEASES/history.tmp" "$RELEASES/history"
}

deploy_cmd() {  # $1: install or update, $2: --allow-dirty or empty
  if [[ "$1" == update ]]; then
    [[ -f "$PLIST" ]] || die "The services aren't installed. Run: scripts/services.sh install"
  fi
  check_tree "$2"
  take_lock
  local sha old
  old="$(current_sha)"
  sha="$(snapshot)"
  switch_or_back "$sha" "$old"
  prune
  if [[ "$1" == install ]]; then
    echo "Installed commit ${sha:0:12}. Mac's services start now and at every login."
    echo "Logs: $LOGS/services.log"
  elif [[ "$sha" == "$old" ]]; then
    echo "Already on commit ${sha:0:12}. Restarted."
  elif [[ -z "$old" ]]; then
    echo "Now on commit ${sha:0:12}. Restarted."
  else
    echo "Updated from ${old:0:12} to ${sha:0:12}. Restarted. To go back: scripts/services.sh rollback"
  fi
}

rollback_cmd() {
  [[ -f "$PLIST" ]] || die "The services aren't installed. Run: scripts/services.sh install"
  take_lock
  local cur prev
  cur="$(current_sha)"
  prev="$( (grep -E "$SHA_RE" "$RELEASES/history" 2>/dev/null || true) | grep -vx "$cur" | tail -n 1 || true)"
  [[ -n "$prev" && -f "$RELEASES/$prev/.complete" ]] || die "There is no earlier snapshot to go back to."
  switch_or_back "$prev" "$cur"
  make_newest "$prev"  # the newest now, so a second rollback goes forward again
  echo "Rolled back from ${cur:0:12} to ${prev:0:12}. Restarted."
}

uninstall_cmd() {
  take_lock
  stop_agent
  rm -f "$PLIST"
  echo "Uninstalled. Mac's services are stopped and won't start at login."
  echo "Snapshots are kept in $RELEASES."
}

restart_cmd() {
  loaded || die "The services aren't installed. Run: scripts/services.sh install"
  "$LAUNCHCTL" kickstart -k "$DOMAIN/$LABEL"
  echo "Restarted."
}

status_cmd() {
  if loaded; then
    echo "Installed: yes (starts at login)."
  elif [[ -f "$PLIST" ]]; then
    echo "Installed: the LaunchAgent file exists but isn't loaded. Run: scripts/services.sh install"
  else
    echo "Installed: no. Run: scripts/services.sh install"
  fi
  local cur
  cur="$(current_sha)"
  if [[ -n "$cur" ]]; then
    echo "Running commit: ${cur:0:12} ($(git -C "$REPO" log -1 --format=%s "$cur" 2>/dev/null || echo "not in this repo"))."
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

flag="${2:-}"
if [[ -n "$flag" && ( "$flag" != "--allow-dirty" || ! "${1:-}" =~ ^(install|update)$ ) ]] || [[ $# -gt 2 ]]; then
  die "Unknown option: ${*:2}"
fi
case "${1:-}" in
  install | update) deploy_cmd "$1" "$flag" ;;
  rollback) rollback_cmd ;;
  uninstall) uninstall_cmd ;;
  status) status_cmd ;;
  restart) restart_cmd ;;
  *) die "Usage: $0 install|update [--allow-dirty] | rollback | uninstall | status | restart" ;;
esac
