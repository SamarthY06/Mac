#!/usr/bin/env bash
# Sign OpenValet.app.
#
#   scripts/sign.sh path/to/OpenValet.app
#
# Order of preference:
#   1. $SIGN_IDENTITY (e.g. "Developer ID Application: …") from the default keychains,
#      with hardened runtime + timestamp (for notarized releases).
#   2. A persistent local self-signed identity "Mac Agent Local". It lives in a
#      DEDICATED keychain (~/Library/Keychains/mac-agent-signing.keychain-db) whose
#      random password is stored in ~/.mac-agent/signing/keychain.pass (0600). Because we
#      know that password we can unlock it and set the key partition list without ever
#      prompting the user (adding it to the login keychain would need the login
#      password). The certificate is reused across rebuilds, so the app's designated
#      requirement (identifier + leaf cert hash) is stable and macOS privacy (TCC)
#      grants survive rebuilds.
#   3. Ad-hoc ("-") — works, but TCC keys grants to the cdhash, so permissions re-prompt
#      after every rebuild.
#
# Env: SIGN_IDENTITY, MAC_SIGN_ADHOC=1 (force ad-hoc), MAC_SIGN_QUIET=1.
set -euo pipefail

APP="${1:?usage: sign.sh path/to/Mac.app}"
IDENTITY_NAME="Mac Agent Local"
SIGN_DIR="$HOME/.mac-agent/signing"
KC="$HOME/Library/Keychains/mac-agent-signing.keychain-db"
PASS_FILE="$SIGN_DIR/keychain.pass"
ENTITLEMENTS="$(cd "$(dirname "$0")" && pwd)/Mac.entitlements"

c_yellow=$'\033[33m'; c_green=$'\033[32m'; c_off=$'\033[0m'
say()  { echo "${c_green}[sign]${c_off} $*"; }
warn() { echo "${c_yellow}[sign] WARNING:${c_off} $*" >&2; }

OPENSSL=/usr/bin/openssl   # LibreSSL: produces PKCS#12 that `security import` accepts
[[ -x $OPENSSL ]] || OPENSSL="$(command -v openssl)"

ensure_local_identity() {
  mkdir -p "$SIGN_DIR"; chmod 700 "$SIGN_DIR"
  if [[ ! -f $PASS_FILE ]]; then
    (umask 077; LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 32 >"$PASS_FILE")
  fi
  local pw; pw="$(cat "$PASS_FILE")"
  if [[ ! -f $KC ]]; then
    security create-keychain -p "$pw" "$KC" >/dev/null
    security set-keychain-settings "$KC"   # no auto-lock timeout
  fi
  security unlock-keychain -p "$pw" "$KC" >/dev/null 2>&1 || return 1
  # Keep it in the user search list (codesign/TCC tooling find it there; harmless).
  if ! security list-keychains -d user | grep -q "mac-agent-signing"; then
    # shellcheck disable=SC2046
    security list-keychains -d user -s $(security list-keychains -d user | tr -d '"') "$KC"
  fi
  if security find-certificate -c "$IDENTITY_NAME" "$KC" >/dev/null 2>&1; then
    return 0
  fi
  say "creating self-signed code-signing identity \"$IDENTITY_NAME\" (one time)"
  local tmp; tmp="$(mktemp -d)"
  cat >"$tmp/cert.cnf" <<CNF
[req]
distinguished_name = dn
x509_extensions = ext
prompt = no
[dn]
CN = $IDENTITY_NAME
O = Mac contributors
[ext]
basicConstraints = critical,CA:false
keyUsage = critical,digitalSignature
extendedKeyUsage = critical,codeSigning
subjectKeyIdentifier = hash
CNF
  $OPENSSL req -x509 -newkey rsa:2048 -nodes -days 3650 -config "$tmp/cert.cnf" \
    -keyout "$tmp/key.pem" -out "$tmp/cert.pem" >/dev/null 2>&1
  $OPENSSL pkcs12 -export -inkey "$tmp/key.pem" -in "$tmp/cert.pem" \
    -name "$IDENTITY_NAME" -passout pass:macagent -out "$tmp/id.p12" >/dev/null 2>&1
  security import "$tmp/id.p12" -k "$KC" -P macagent -T /usr/bin/codesign >/dev/null
  security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$pw" "$KC" >/dev/null 2>&1 || true
  cp "$tmp/cert.pem" "$SIGN_DIR/cert.pem"
  rm -rf "$tmp"
}

sign_with() {  # identity, extra args...
  local id="$1"; shift
  codesign --force --options runtime --entitlements "$ENTITLEMENTS" \
    --identifier io.github.macagent.Mac -s "$id" "$@" "$APP"
}

if [[ -n ${SIGN_IDENTITY:-} ]]; then
  say "signing with \$SIGN_IDENTITY=$SIGN_IDENTITY"
  sign_with "$SIGN_IDENTITY" --timestamp
  codesign --verify --strict "$APP"
  exit 0
fi

if [[ ${MAC_SIGN_ADHOC:-0} != 1 ]] && ensure_local_identity; then
  # Bounded: never let a keychain dialog hang an automated run.
  if perl -e 'alarm 60; exec @ARGV' codesign --force --options runtime \
       --entitlements "$ENTITLEMENTS" --identifier io.github.macagent.Mac \
       --keychain "$KC" -s "$IDENTITY_NAME" "$APP" 2>"$SIGN_DIR/last-error.log"; then
    if codesign --verify "$APP" 2>/dev/null; then
      say "signed with local identity \"$IDENTITY_NAME\" (stable across rebuilds)"
      exit 0
    fi
  fi
  warn "local identity signing failed: $(tr '\n' ' ' <"$SIGN_DIR/last-error.log")"
fi

sign_with - 
warn "signed AD-HOC. macOS will re-ask for Microphone/Accessibility/Automation"
warn "permissions after every rebuild. Set SIGN_IDENTITY or fix the local identity."
