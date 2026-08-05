#!/usr/bin/env bash
# packaging/build-and-notarize.sh - build, sign, notarize, and package Tessera Studio.
#
# Produces a notarized .pkg installer signed with the user's Developer ID,
# installable via `installer -pkg TesseraStudio-X.Y.Z.pkg -target /` or by
# double-clicking in Finder.
#
# Usage:
#   ./packaging/build-and-notarize.sh [version] [build-number]
#
# Requirements (one-time setup):
#   1. A "Developer ID Application" identity in your keychain (Xcode does
#      this for you if you've signed in with your Apple ID and downloaded
#      a Developer ID certificate).
#   2. A "Developer ID Installer" identity in your keychain (download from
#      https://developer.apple.com/account/resources/certificates/list).
#   3. A notarytool credential profile:
#        xcrun notarytool store-credentials tessera-notary
#      (uses an App Store Connect API key, NOT your Apple ID password.
#       Generate the key at https://appstoreconnect.apple.com/access/api.)
#
# Environment overrides (all optional):
#   DEVELOPER_ID_APP       "Developer ID Application: Name (TEAMID)"
#   DEVELOPER_ID_INSTALLER "Developer ID Installer: Name (TEAMID)"
#   DEVELOPMENT_TEAM       "TEAMID"
#   NOTARY_PROFILE         "tessera-notary"
#   BUNDLE_ID              "com.tessera.studio.mac"
#   SKIP_NOTARIZATION      "1" to build + sign + package but skip notarytool
#   OUTPUT_DIR             "dist" (relative to repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
STUDIO_DIR="$REPO_ROOT/TesseraStudio"
ARTIFACTS_DIR="$STUDIO_DIR/artifacts"
BUILD_DIR="$STUDIO_DIR/build-release"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/dist}"

# Version + build number. Defaults to 1.0.0 / 1 (the values the project ships
# with); pass them as args to bump.
VERSION="${1:-1.0.0}"
BUILD_NUMBER="${2:-1}"
BUNDLE_ID="${BUNDLE_ID:-com.tessera.studio.mac}"
NOTARY_PROFILE="${NOTARY_PROFILE:-tessera-notary}"
SKIP_NOTARIZATION="${SKIP_NOTARIZATION:-}"

# --- Auto-detect signing identities from the keychain if not set ---
if [ -z "${DEVELOPER_ID_APP:-}" ]; then
    DEVELOPER_ID_APP="$(security find-identity -v -p codesigning 2>/dev/null \
        | awk -F'"' '/Developer ID Application/ {print $2; exit}')"
fi
if [ -z "${DEVELOPER_ID_INSTALLER:-}" ]; then
    DEVELOPER_ID_INSTALLER="$(security find-identity -v -p codesigning 2>/dev/null \
        | awk -F'"' '/Developer ID Installer/ {print $2; exit}')"
fi
if [ -z "${DEVELOPMENT_TEAM:-}" ] && [ -n "${DEVELOPER_ID_APP:-}" ]; then
    # Extract the team ID from the parens at the end of the identity name
    DEVELOPMENT_TEAM="$(echo "$DEVELOPER_ID_APP" | awk -F'[()]' '{print $2}')"
fi

# --- Pre-flight ---
echo "=== tessera-studio packaging ==="
echo "  version:    $VERSION"
echo "  build:      $BUILD_NUMBER"
echo "  bundle id:  $BUNDLE_ID"
echo "  output:     $OUTPUT_DIR"
echo "  signing:    ${DEVELOPER_ID_APP:-<not found>}"
echo "  installer:  ${DEVELOPER_ID_INSTALLER:-<not found>}"
echo "  team:       ${DEVELOPMENT_TEAM:-<not found>}"
if [ -n "$SKIP_NOTARIZATION" ]; then
    echo "  notary:     SKIPPED (SKIP_NOTARIZATION=1)"
else
    echo "  notary:     profile=$NOTARY_PROFILE"
fi
echo ""

missing=()
warnings=()
if [ -z "$DEVELOPER_ID_APP" ]; then
    missing+=("Developer ID Application identity (download from developer.apple.com or import via Xcode)")
fi
if [ -z "$DEVELOPER_ID_INSTALLER" ]; then
    # The Developer ID Installer identity is optional. The .pkg wrapper
    # itself won't be Installer-signed without it, but the embedded .app
    # is still Developer-ID-signed + notarized, so Gatekeeper accepts the
    # install. Warn loudly but don't fail.
    warnings+=("Developer ID Installer identity NOT FOUND - the .pkg will NOT be Installer-signed. Download from developer.apple.com to add the package-level seal.")
fi
if [ -z "$DEVELOPMENT_TEAM" ]; then
    missing+=("development team (could not be inferred from the identity; set DEVELOPMENT_TEAM)")
fi
if [ -z "$SKIP_NOTARIZATION" ] && command -v xcrun >/dev/null 2>&1; then
    if ! xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null 2>&1; then
        missing+=("notarytool profile '$NOTARY_PROFILE' (run: xcrun notarytool store-credentials $NOTARY_PROFILE)")
    fi
fi

if [ ${#missing[@]} -ne 0 ]; then
    echo "Pre-flight failed. Missing prerequisites:"
    for m in "${missing[@]}"; do
        echo "  - $m"
    done
    echo ""
    echo "Resolve the above, or run with SKIP_NOTARIZATION=1 to skip the notarial step."
    exit 1
fi

if [ ${#warnings[@]} -ne 0 ]; then
    echo "Pre-flight warnings (non-fatal):"
    for w in "${warnings[@]}"; do
        echo "  - $w"
    done
    echo ""
fi

# --- Step 1: xcframework (build if missing) ---
if [ ! -d "$ARTIFACTS_DIR/tessera.xcframework" ]; then
    echo "=== building xcframework ==="
    "$STUDIO_DIR/scripts/build-xcframework.sh"
fi

# --- Step 2: xcodebuild Release ---
echo "=== xcodebuild Release ==="
rm -rf "$BUILD_DIR"
xcodebuild -project "$STUDIO_DIR/TesseraStudio.xcodeproj" \
    -scheme TesseraStudioMac \
    -configuration Release \
    -derivedDataPath "$BUILD_DIR" \
    -destination 'generic/platform=macOS' \
    MARKETING_VERSION="$VERSION" \
    CURRENT_PROJECT_VERSION="$BUILD_NUMBER" \
    PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
    CODE_SIGN_IDENTITY="$DEVELOPER_ID_APP" \
    CODE_SIGN_STYLE=Manual \
    DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM" \
    ENABLE_HARDENED_RUNTIME=YES \
    OTHER_CODE_SIGN_FLAGS="--timestamp" \
    clean build 2>&1 | tail -20

APP_PATH="$BUILD_DIR/Build/Products/Release/TesseraStudioMac.app"
if [ ! -d "$APP_PATH" ]; then
    echo "xcodebuild did not produce $APP_PATH"
    exit 1
fi

# --- Step 3: notarize the .app (required before the pkg, so the embedded
#     app is already notarized when the pkg installer unpacks it) ---
if [ -z "$SKIP_NOTARIZATION" ]; then
    echo "=== notarytool submit (.app) ==="
    xcrun notarytool submit "$APP_PATH" \
        --keychain-profile "$NOTARY_PROFILE" \
        --wait

    echo "=== stapler staple (.app) ==="
    xcrun stapler staple "$APP_PATH"
fi

# --- Step 4: build the .pkg ---
echo "=== pkgbuild ==="
mkdir -p "$OUTPUT_DIR"

# Stage the .app at /Applications inside a payload root. pkgbuild installs
# this at the absolute --install-location, so the result lands in
# /Applications/TesseraStudio.app.
PAYLOAD_DIR="$BUILD_DIR/pkg-payload"
rm -rf "$PAYLOAD_DIR"
mkdir -p "$PAYLOAD_DIR/Applications"
cp -R "$APP_PATH" "$PAYLOAD_DIR/Applications/TesseraStudio.app"

COMPONENT_PKG="$BUILD_DIR/TesseraStudio-component.pkg"
pkgbuild --root "$PAYLOAD_DIR" \
    --identifier "$BUNDLE_ID" \
    --version "$VERSION" \
    --install-location "/" \
    "$COMPONENT_PKG"

FINAL_PKG="$OUTPUT_DIR/TesseraStudio-$VERSION.pkg"
echo "=== productbuild ==="
if [ -n "$DEVELOPER_ID_INSTALLER" ]; then
    productbuild --package "$COMPONENT_PKG" \
        --sign "$DEVELOPER_ID_INSTALLER" \
        "$FINAL_PKG"
else
    echo "  (no Developer ID Installer identity; producing an unsigned .pkg)"
    productbuild --package "$COMPONENT_PKG" \
        "$FINAL_PKG"
fi

# --- Step 5: notarize the .pkg (separate from the embedded .app's
#     notarization - the .pkg seal needs its own notary ticket too) ---
if [ -z "$SKIP_NOTARIZATION" ]; then
    echo "=== notarytool submit (.pkg) ==="
    xcrun notarytool submit "$FINAL_PKG" \
        --keychain-profile "$NOTARY_PROFILE" \
        --wait

    echo "=== stapler staple (.pkg) ==="
    xcrun stapler staple "$FINAL_PKG"
fi

echo ""
echo "=== done ==="
echo "  $FINAL_PKG"
ls -lh "$FINAL_PKG"
