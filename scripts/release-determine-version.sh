#!/bin/bash
# Determine the release version for llama.cpp.
# Usage: release-determine-version.sh [version]
#   version: optional explicit version (e.g. v0.1.0); reads CMakeLists.txt if omitted
# Writes "version=<ver>" to $GITHUB_OUTPUT when running in GitHub Actions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${1:-}" ]]; then
    VERSION="$1"
else
    MAJOR=$(grep "set(LLAMA_VERSION_MAJOR" "$REPO_ROOT/CMakeLists.txt" | grep -oP '\d+')
    MINOR=$(grep "set(LLAMA_VERSION_MINOR" "$REPO_ROOT/CMakeLists.txt" | grep -oP '\d+')
    PATCH=$(grep "set(LLAMA_VERSION_PATCH" "$REPO_ROOT/CMakeLists.txt" | grep -oP '\d+')
    VERSION="v${MAJOR}.${MINOR}.${PATCH}"
fi

echo "Determined version: ${VERSION}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "version=${VERSION}" >> "$GITHUB_OUTPUT"
fi
