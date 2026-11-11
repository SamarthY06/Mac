#!/usr/bin/env bash
# Build OpenMac.app:  swift build -c release → build/OpenMac.app → sign (scripts/sign.sh)
#   scripts/bundle.sh [--debug]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG=release
[[ ${1:-} == --debug ]] && CONFIG=debug
VERSION="${MAC_VERSION:-$(tr -d '[:space:]' <"$ROOT/VERSION")}"
BUILD_NUM="${MAC_BUILD:-$(date +%Y%m%d%H%M)}"
OUT="$ROOT/build"
APP="$OUT/OpenMac.app"

g=$'\033[32m'; o=$'\033[0m'
step() { echo "${g}==>${o} $*"; }

step "swift build -c $CONFIG"
(cd "$ROOT/app" && swift build -c "$CONFIG" --product MacApp)
BIN="$(cd "$ROOT/app" && swift build -c "$CONFIG" --show-bin-path)/MacApp"

step "assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/Mac"

# Icon (cached; regenerate when the script changes).
ICNS="$OUT/AppIcon.icns"
if [[ ! -f $ICNS || "$ROOT/scripts/make_icon.swift" -nt $ICNS ]]; then
  step "rendering icon"
  rm -rf "$OUT/AppIcon.iconset"
  swift "$ROOT/scripts/make_icon.swift" "$OUT/AppIcon.iconset" >/dev/null
  iconutil -c icns "$OUT/AppIcon.iconset" -o "$ICNS"
  rm -rf "$OUT/AppIcon.iconset"
fi
cp "$ICNS" "$APP/Contents/Resources/AppIcon.icns"

cat >"$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key><string>en</string>
	<key>CFBundleExecutable</key><string>Mac</string>
	<key>CFBundleIdentifier</key><string>io.github.macagent.Mac</string>
	<key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
	<key>CFBundleName</key><string>OpenMac</string>
	<key>CFBundleDisplayName</key><string>OpenMac</string>
	<key>CFBundlePackageType</key><string>APPL</string>
	<key>CFBundleShortVersionString</key><string>$VERSION</string>
	<key>CFBundleVersion</key><string>$BUILD_NUM</string>
	<key>CFBundleIconFile</key><string>AppIcon</string>
	<key>LSMinimumSystemVersion</key><string>14.0</string>
	<key>LSUIElement</key><true/>
	<key>LSApplicationCategoryType</key><string>public.app-category.productivity</string>
	<key>NSHighResolutionCapable</key><true/>
	<key>NSPrincipalClass</key><string>NSApplication</string>
	<key>NSHumanReadableCopyright</key><string>MIT License. Copyright © OpenMac contributors.</string>
	<key>NSMicrophoneUsageDescription</key>
	<string>Mac listens for “Hey Mac” and your spoken requests. Audio is processed on this Mac.</string>
	<key>NSAppleEventsUsageDescription</key>
	<string>Mac controls apps like Mail, Calendar, Notes and System Events to carry out your requests.</string>
</dict>
</plist>
PLIST
plutil -lint "$APP/Contents/Info.plist" >/dev/null
printf 'APPL????' >"$APP/Contents/PkgInfo"

step "signing"
"$ROOT/scripts/sign.sh" "$APP"
step "built $APP"
