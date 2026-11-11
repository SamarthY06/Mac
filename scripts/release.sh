#!/usr/bin/env bash
# Publish an OpenMac release: tag v$(cat VERSION), build the WhatsApp bridge for Apple silicon and
# attach it with its checksum. There is no app download: install.sh builds OpenMac.app locally.
#
#   scripts/release.sh                 tag, build, create the GitHub release
#   scripts/release.sh --draft         same, as a draft release
#   scripts/release.sh --build-only    only build build/release/v<version>/ (no tag, no push)
#
# Installs without Go download wa-bridge-darwin-arm64 for the snapshot's VERSION (scripts/services.sh),
# so any change under bridges/whatsapp needs a VERSION bump and a new release.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRAFT=0; BUILD_ONLY=0
for a in "$@"; do
  case "$a" in
    --draft) DRAFT=1 ;;
    --build-only) BUILD_ONLY=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done
die() { echo "release: $*" >&2; exit 1; }

VER="$(tr -d '[:space:]' <"$ROOT/VERSION")"
[[ "$VER" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.]+)?$ ]] || die "VERSION '$VER' is not a semantic version"
TAG="v$VER"
ASSET="wa-bridge-darwin-arm64"
DIST="$ROOT/build/release/$TAG"

grep -q "^version = \"$VER\"" "$ROOT/daemon/pyproject.toml" || die "daemon/pyproject.toml does not say version = \"$VER\""
grep -q "\"$VER\"" "$ROOT/daemon/src/macd/__init__.py" || die "daemon/src/macd/__init__.py does not say $VER"
command -v go >/dev/null 2>&1 || die "Go is required to build the bridge (brew install go)"

if [[ $BUILD_ONLY == 0 ]]; then
  command -v gh >/dev/null 2>&1 || die "gh is required (brew install gh)"
  gh auth status >/dev/null 2>&1 || die "gh is not signed in (gh auth login)"
  [[ -z "$(git -C "$ROOT" status --porcelain)" ]] || die "the working tree has uncommitted changes"
  git -C "$ROOT" fetch -q origin
  git -C "$ROOT" merge-base --is-ancestor HEAD origin/main || die "HEAD is not on origin/main; push first"
fi

echo "==> building $ASSET for $TAG"
rm -rf "$DIST"; mkdir -p "$DIST"
(cd "$ROOT/bridges/whatsapp" && CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -trimpath -ldflags="-s -w" -o "$DIST/$ASSET" .)
file "$DIST/$ASSET" | grep -q arm64 || die "the bridge binary is not arm64"
# The bridge refuses to start without its arguments (exit 2); any clean non-zero exit means it runs.
set +e; "$DIST/$ASSET" </dev/null >/dev/null 2>&1; rc=$?; set -e
[[ $rc -ne 0 && $rc -lt 126 ]] || die "the bridge binary did not run as expected (exit $rc)"
(cd "$DIST" && shasum -a 256 "$ASSET" >SHA256SUMS)
echo "    $(cat "$DIST/SHA256SUMS")"
[[ $BUILD_ONLY == 1 ]] && { echo "built $DIST"; exit 0; }

echo "==> tag $TAG"
if git -C "$ROOT" rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  [[ "$(git -C "$ROOT" rev-parse "$TAG^{commit}")" == "$(git -C "$ROOT" rev-parse HEAD)" ]] || die "tag $TAG exists and is not HEAD"
else
  git -C "$ROOT" tag -a "$TAG" -m "OpenMac $VER"
fi
git -C "$ROOT" push -q origin "$TAG"
gh release view "$TAG" >/dev/null 2>&1 && die "release $TAG already exists: bump VERSION, or gh release delete $TAG"

echo "==> release notes"
NOTES="$DIST/notes.md"
{
  echo "# OpenMac $VER"
  echo
  echo "Install or update:"
  echo
  echo '```bash'
  echo "curl -fsSL https://raw.githubusercontent.com/SamarthY06/OpenMac/main/install.sh | bash"
  echo '```'
  echo
  echo "## Changes"
  echo
  section=""
  [[ -f "$ROOT/CHANGELOG.md" ]] && section="$(awk -v v="$VER" '
      /^## /{ if (on) exit; on = ($0 ~ ("^## \\[?v?" v "([^0-9.]|$)")); next } on' "$ROOT/CHANGELOG.md")"
  if [[ -n "${section//[[:space:]]/}" ]]; then
    printf '%s\n' "$section"
  else
    prev="$(git -C "$ROOT" describe --tags --abbrev=0 "$TAG^" 2>/dev/null || true)"
    git -C "$ROOT" log --format='- %s' ${prev:+"$prev..$TAG"}
  fi
  echo
  echo "## Files"
  echo
  echo "- \`$ASSET\`: the WhatsApp bridge. The installer downloads it when Go is not installed and checks it against \`SHA256SUMS\` from this release (a transfer check; both files come from the same place). With Go installed the bridge is built from source instead. There is no app download: \`install.sh\` builds OpenMac.app on your Mac."
} >"$NOTES"

echo "==> gh release create $TAG"
args=(release create "$TAG" "$DIST/$ASSET" "$DIST/SHA256SUMS" --title "OpenMac $VER" --notes-file "$NOTES" --verify-tag)
[[ $DRAFT == 1 ]] && args+=(--draft)
gh "${args[@]}"
