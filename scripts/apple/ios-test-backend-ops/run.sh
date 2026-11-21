#!/usr/bin/env bash
# Run test-backend-ops on a connected iPhone/iPad (iOS cannot run command-line tools, so it is wrapped in a small app).
#
# usage: DEVELOPMENT_TEAM=<team id> scripts/apple/ios-test-backend-ops/run.sh [test-backend-ops arguments]
#   default arguments: -b ANE
#   e.g.  run.sh -b ANE -o MUL_MAT
#         run.sh -b MTL0
#
# environment:
#   DEVELOPMENT_TEAM  Apple development team id used for automatic signing (required, see Xcode > Settings > Accounts)
#   DEVICE            device name or identifier (default: the only paired device, from `xcrun devicectl list devices`)
#   BUNDLE_ID         bundle id of the test app (default: com.example.ggml-test-backend-ops)
#   BUILD_DIR         build directory (default: build-ios-test-backend-ops)
#
# requirements: Xcode 15+ signed in to an Apple account (Xcode > Settings > Accounts; automatic signing creates the
# provisioning profile for BUNDLE_ID), the device paired with this Mac, Developer Mode enabled on the device.
# note: every ANE program is compiled by iOS and kept in its compile cache (~2.5x the program weights, not removable by
# the app). The full ANE test suite compiles thousands of small programs; keep some free storage on the device.
set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$DIR/../../.." && pwd)
BUILD_DIR=${BUILD_DIR:-$ROOT/build-ios-test-backend-ops}
BUNDLE_ID=${BUNDLE_ID:-com.example.ggml-test-backend-ops}

if [ -z "${DEVELOPMENT_TEAM:-}" ]; then
    echo "set DEVELOPMENT_TEAM to your Apple development team id" >&2
    exit 1
fi

if [ -z "${DEVICE:-}" ]; then
    tmp=$(mktemp)
    xcrun devicectl list devices --json-output "$tmp" >/dev/null
    DEVICE=$(python3 - "$tmp" <<'EOF'
import json, sys
devs = json.load(open(sys.argv[1]))["result"]["devices"]
ok = [d for d in devs if d.get("connectionProperties", {}).get("pairingState") == "paired"
      and d.get("hardwareProperties", {}).get("platform") == "iOS"]
print(ok[0]["identifier"] if len(ok) == 1 else "")
EOF
)
    rm -f "$tmp"
    if [ -z "$DEVICE" ]; then
        echo "set DEVICE: no single paired iOS device found (xcrun devicectl list devices)" >&2
        exit 1
    fi
fi

ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
    ARGS=(-b ANE)
fi

cmake -S "$DIR" -B "$BUILD_DIR" -G Xcode \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 \
    -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM" \
    -DBUNDLE_ID="$BUNDLE_ID" \
    -DGGML_ANE=ON >/dev/null
cmake --build "$BUILD_DIR" --config Release --target test-backend-ops-app -- -allowProvisioningUpdates -quiet

APP=$(find "$BUILD_DIR" -path "*Release-iphoneos/ggml-test-backend-ops.app" -maxdepth 4 | head -1)
xcrun devicectl device install app --device "$DEVICE" "$APP" >/dev/null
echo "running on $DEVICE: test-backend-ops ${ARGS[*]}"
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing --console "$BUNDLE_ID" "${ARGS[@]}"
