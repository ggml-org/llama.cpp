#!/bin/bash
# Check that the vendored ggml/ matches the corresponding upstream ggml release tag.
# Usage: release-check-ggml.sh [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

MAJOR=$(grep "set(GGML_VERSION_MAJOR" "$REPO_ROOT/ggml/CMakeLists.txt" | grep -oP '\d+')
MINOR=$(grep "set(GGML_VERSION_MINOR" "$REPO_ROOT/ggml/CMakeLists.txt" | grep -oP '\d+')
PATCH=$(grep "set(GGML_VERSION_PATCH" "$REPO_ROOT/ggml/CMakeLists.txt" | grep -oP '\d+')
GGML_VERSION="v${MAJOR}.${MINOR}.${PATCH}"
echo "Local ggml version: ${GGML_VERSION}"

if ! git clone --depth 1 --branch "${GGML_VERSION}" https://github.com/ggml-org/ggml.git upstream-ggml 2>/dev/null; then
    echo "Warning: tag ${GGML_VERSION} not found in upstream ggml - skipping comparison"
    exit 0
fi

echo "Comparing local ggml/ src and include with upstream ${GGML_VERSION}..."
DIFF=$(diff -rq \
    "$REPO_ROOT/ggml/src" upstream-ggml/src \
    2>&1 || true)
DIFF+=$(diff -rq \
    "$REPO_ROOT/ggml/include" upstream-ggml/include \
    2>&1 || true)
DIFF+=$(diff \
    "$REPO_ROOT/ggml/CMakeLists.txt" upstream-ggml/CMakeLists.txt \
    2>&1 || true)

rm -rf upstream-ggml

if [[ -n "$DIFF" ]]; then
    echo "local ggml/ differs from upstream ${GGML_VERSION}:"
    echo "$DIFF"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "Warning: would abort release due to ggml mismatch (dry run, continuing)."
    else
        echo "Error: ggml must match upstream before making a release."
        exit 1
    fi
else
    echo "local ggml/ matches upstream ${GGML_VERSION}"
fi
