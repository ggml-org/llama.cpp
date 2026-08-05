#!/usr/bin/env bash
# build-xcframework.sh - build the Tessera engine into tessera.xcframework.
#
# Produces TesseraStudio/artifacts/tessera.xcframework containing a merged
# arm64-apple-macosx static library with the full Tessera C++ engine and
# the C FFI entry points declared in tessera_ffi.h.
#
# Usage:
#   ./TesseraStudio/scripts/build-xcframework.sh
#
# Requirements: Xcode command-line tools (cmake, xcodebuild, libtool).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDIO_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$STUDIO_DIR")"

BUILD_DIR="$REPO_ROOT/build-ffi"
ARTIFACTS_DIR="$STUDIO_DIR/artifacts"
FRAMEWORK_STAGE="$BUILD_DIR/framework-stage"
FW="$FRAMEWORK_STAGE/tessera.framework"

NCPU="$(sysctl -n hw.ncpu)"

echo "=== tessera xcframework build ==="
echo "  repo:      $REPO_ROOT"
echo "  build:     $BUILD_DIR"
echo "  artifacts: $ARTIFACTS_DIR"
echo ""

# -----------------------------------------------------------------------
# 1. Configure (static libs only, arm64 macOS, minimal tool surface)
# -----------------------------------------------------------------------
echo "--- configure ---"
mkdir -p "$BUILD_DIR"
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DLLAMA_BUILD_COMMON=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_UI=OFF \
    > "$BUILD_DIR/configure.log" 2>&1 || {
        echo "configure failed; see $BUILD_DIR/configure.log"
        tail -30 "$BUILD_DIR/configure.log"
        exit 1
    }

# -----------------------------------------------------------------------
# 2. Build the tessera-ffi target (pulls in all engine dependencies)
# -----------------------------------------------------------------------
echo "--- build tessera-ffi (-j$NCPU) ---"
cmake --build "$BUILD_DIR" --target tessera-ffi -j "$NCPU" \
    > "$BUILD_DIR/build.log" 2>&1 || {
        echo "build failed; see $BUILD_DIR/build.log"
        tail -40 "$BUILD_DIR/build.log"
        exit 1
    }

# -----------------------------------------------------------------------
# 3. Merge all static archives into one fat .a via libtool
# -----------------------------------------------------------------------
echo "--- merge static archives ---"
MERGED="$BUILD_DIR/libtessera-merged.a"

# collect every .a produced by the build
LIBS=()
while IFS= read -r -d '' lib; do
    LIBS+=("$lib")
done < <(find "$BUILD_DIR" -name '*.a' -print0)

if [ ${#LIBS[@]} -eq 0 ]; then
    echo "error: no .a files found in $BUILD_DIR"
    exit 1
fi

echo "  merging ${#LIBS[@]} archives"
libtool -static -o "$MERGED" "${LIBS[@]}" 2>/dev/null
echo "  merged -> $MERGED ($(du -h "$MERGED" | cut -f1))"

# -----------------------------------------------------------------------
# 4. Assemble the framework bundle
# -----------------------------------------------------------------------
echo "--- assemble framework bundle ---"
rm -rf "$FW"
mkdir -p "$FW/Versions/A/Headers" "$FW/Versions/A/Modules" "$FW/Versions/A/Resources"

cp "$STUDIO_DIR/Sources/CTesseraFFI/include/tessera_ffi.h" "$FW/Versions/A/Headers/tessera_ffi.h"
cp "$MERGED" "$FW/Versions/A/tessera"

cat > "$FW/Versions/A/Modules/module.modulemap" << 'EOF'
framework module tessera {
    header "tessera_ffi.h"
    export *
}
EOF

# The framework's Info.plist is required for `xcodebuild`'s
# `builtin-validateFramework` step when the xcframework is embedded
# into the app (CFBundlePackageType=FMWK tells the linker this is a
# framework, not a regular bundle; CFBundleExecutable matches the
# static-archive name written above). The xcframework's own
# top-level Info.plist is the xcframework manifest, not this.
cat > "$FW/Versions/A/Resources/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>tessera</string>
    <key>CFBundleIdentifier</key>
    <string>com.tessera.engine</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>tessera</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>MacOSX</string>
    </array>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
</dict>
</plist>
EOF

# macOS frameworks use a versioned bundle layout (Versions/A/...) and
# expose the current version via Versions/Current. The top-level entries
# (Headers, Modules, Resources, tessera) are symlinks into Versions/Current
# so consumers and the embed step both find them at the canonical path.
ln -sfn A "$FW/Versions/Current"
ln -sfn Versions/Current/Headers "$FW/Headers"
ln -sfn Versions/Current/Modules "$FW/Modules"
ln -sfn Versions/Current/Resources "$FW/Resources"
ln -sfn Versions/Current/tessera "$FW/tessera"

echo "  framework -> $FW"

# -----------------------------------------------------------------------
# 5. Package as .xcframework
# -----------------------------------------------------------------------
echo "--- create xcframework ---"
rm -rf "$ARTIFACTS_DIR/tessera.xcframework"
mkdir -p "$ARTIFACTS_DIR"

xcodebuild -create-xcframework \
    -framework "$FW" \
    -output "$ARTIFACTS_DIR/tessera.xcframework" \
    > "$BUILD_DIR/xcframework.log" 2>&1 || {
        echo "xcframework creation failed; see $BUILD_DIR/xcframework.log"
        cat "$BUILD_DIR/xcframework.log"
        exit 1
    }

echo ""
echo "=== done ==="
echo "  $ARTIFACTS_DIR/tessera.xcframework"
