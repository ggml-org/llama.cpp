#!/bin/bash
# Check API/ABI compatibility between the previous release tag and current HEAD.
#
# Finds the most recent vX.Y.Z tag, checks it out in a temporary git worktree,
# builds both versions with shared libs enabled, and uses check-apiabi-compat.sh
# to compare the results.
#
# Exit codes:
#   0: compatible, or check was skipped
#   1: backwards-incompatible changes found, or build failed
#
# Options:
#   --tag <version>: compare against this tag instead of the latest release
#
# Environment:
#   SKIP_APIABI_CHECK:    set to 1 or true to skip
#   APIABI_COMPARE_TAG:   equivalent to --tag (used by CI)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    echo "Usage: $0 [--tag <version>]" >&2
    echo "  --tag <version>  Compare against this release tag (default: latest)" >&2
}

COMPARE_TAG="${APIABI_COMPARE_TAG:-}"
while [[ "$#" -gt 0 ]]; do
    case "$1" in
    --tag)
        if [[ -z "${2:-}" ]]; then usage; exit 1; fi
        COMPARE_TAG="$2"
        shift 2
        ;;
    --tag=*)
        COMPARE_TAG="${1#*=}"
        shift
        ;;
    -h | --help)
        usage; exit 0
        ;;
    *)
        usage; exit 1
        ;;
    esac
done

if [[ "${SKIP_APIABI_CHECK:-}" == "1" || "${SKIP_APIABI_CHECK:-}" == "true" ]]; then
    echo "SKIP_APIABI_CHECK is set - skipping API/ABI compatibility check"
    exit 0
fi

if ! command -v abi-compliance-checker >/dev/null 2>&1 || ! command -v abidw >/dev/null 2>&1; then
    echo "Warning: abi-compliance-checker or abigail-tools not installed - skipping API/ABI check"
    exit 0
fi

discover_libs() {
    local build_dir="$1"
    local libs=()
    for dir in "$build_dir/src" "$build_dir/bin"; do
        [[ -d "$dir" ]] || continue
        for f in "$dir"/lib*.so; do
            [[ -f "$f" ]] && libs+=("$(basename "$f" .so)")
        done
    done
    echo "${libs[@]}"
}

if [[ -n "${COMPARE_TAG}" ]]; then
    PREV_TAG="${COMPARE_TAG}"
    if ! git -C "$REPO_ROOT" rev-parse --verify "${PREV_TAG}^{}" >/dev/null 2>&1; then
        echo "Error: tag '${PREV_TAG}' not found in repository." >&2
        exit 1
    fi
else
    PREV_TAG=$(git -C "$REPO_ROOT" tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -n 1 || true)
    if [[ -z "${PREV_TAG}" ]]; then
        echo "Warning: no previous release tag found - skipping API/ABI check"
        exit 0
    fi
fi
OLD_VERSION="${PREV_TAG#v}"
OLD_MAJOR="${OLD_VERSION%%.*}"
OLD_MINOR="${OLD_VERSION#*.}"; OLD_MINOR="${OLD_MINOR%%.*}"

NEW_MAJOR=$(grep "set(LLAMA_VERSION_MAJOR" "$REPO_ROOT/CMakeLists.txt" | sed 's/.*MAJOR \([0-9]*\).*/\1/')
NEW_MINOR=$(grep "set(LLAMA_VERSION_MINOR" "$REPO_ROOT/CMakeLists.txt" | sed 's/.*MINOR \([0-9]*\).*/\1/')

if [[ "$NEW_MAJOR" -gt "$OLD_MAJOR" ]]; then
    echo "Major version increment ($OLD_MAJOR -> $NEW_MAJOR): API/ABI breaking changes are expected, skipping compatibility check."
    exit 0
fi

CHECK_FLAGS=()
if [[ "$NEW_MINOR" -eq "$OLD_MINOR" ]]; then
    echo "Patch version bump detected: checking for any API/ABI changes (a minor bump is required if any are found)..."
    CHECK_FLAGS+=(--strict)
else
    echo "Minor version bump detected: checking for backwards-incompatible API/ABI changes..."
fi

echo "Checking API/ABI compatibility against ${PREV_TAG}..."

WORKTREE_DIR=$(mktemp -d)
BUILD_OLD=$(mktemp -d)
BUILD_NEW=$(mktemp -d)

cleanup() {
    git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
    rm -rf "$WORKTREE_DIR" "$BUILD_OLD" "$BUILD_NEW"
}
trap cleanup EXIT

git -C "$REPO_ROOT" worktree add "$WORKTREE_DIR" "$PREV_TAG"

cmake -S "$WORKTREE_DIR" -B "$BUILD_OLD" -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD_OLD" --parallel "$(nproc)"
OLD_LIBS=($(discover_libs "$BUILD_OLD"))
echo "Libraries found in old build: ${OLD_LIBS[*]}"

cmake -S "$REPO_ROOT"    -B "$BUILD_NEW" -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD_NEW" --parallel "$(nproc)"
NEW_LIBS=($(discover_libs "$BUILD_NEW"))
echo "Libraries found in new build: ${NEW_LIBS[*]}"

(cd "$WORKTREE_DIR" && "$SCRIPT_DIR/check-apiabi-compat.sh" --include-path ggml/include --generate "$BUILD_OLD" "${OLD_LIBS[@]}")
(cd "$REPO_ROOT"    && "$SCRIPT_DIR/check-apiabi-compat.sh" --include-path ggml/include --generate "$BUILD_NEW" "${NEW_LIBS[@]}")
(cd "$REPO_ROOT"    && "$SCRIPT_DIR/check-apiabi-compat.sh" "${CHECK_FLAGS[@]}" --check "$BUILD_OLD" "$BUILD_NEW")
