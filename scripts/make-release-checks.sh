#!/bin/bash
# Run all pre-release checks and determine the release version.
#
# Usage: make-release-checks.sh [--dry-run]
#   --dry-run: warn on failures instead of aborting
#
# Env (when running in GitHub Actions):
#   GITHUB_OUTPUT
#   RELEASE_BRANCH: when set, HEAD must belong to origin/RELEASE_BRANCH and must
#     not be older than 3 days from the branch HEAD (skipped when unset)
#   REQUIRE_DOCKER: set to "false" to skip the container image check and the
#     release re-tagging of the nightly images in make-release.yml (default: true)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY_RUN=false
CHECKS_PASSED=true
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

MAJOR=$(grep "set(LLAMA_VERSION_MAJOR" "$REPO_ROOT/CMakeLists.txt" | grep -oP '\d+')
MINOR=$(grep "set(LLAMA_VERSION_MINOR" "$REPO_ROOT/CMakeLists.txt" | grep -oP '\d+')
PATCH=$(grep "set(LLAMA_VERSION_PATCH" "$REPO_ROOT/CMakeLists.txt" | grep -oP '\d+')
VERSION="v${MAJOR}.${MINOR}.${PATCH}"
echo "Determined version: ${VERSION}"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "version=${VERSION}" >> "$GITHUB_OUTPUT"
fi

SHA=$(git rev-parse HEAD)

echo "Checking that commit ${SHA} belongs to the release branch..."
if [[ -z "${RELEASE_BRANCH:-}" ]]; then
    echo "Warning: RELEASE_BRANCH not set - skipping commit check (local run)"
else
    TIP="origin/${RELEASE_BRANCH}"
    COMMIT_ERR=""
    if ! git rev-parse --verify "${TIP}" >/dev/null 2>&1; then
        COMMIT_ERR="branch ${RELEASE_BRANCH} not found on remote"
    elif ! git merge-base --is-ancestor "${SHA}" "${TIP}"; then
        COMMIT_ERR="commit ${SHA} is not part of branch ${RELEASE_BRANCH}"
    else
        COMMIT_TS=$(git show -s --format=%ct "${SHA}")
        TIP_TS=$(git show -s --format=%ct "${TIP}")
        AGE_DAYS=$(( (TIP_TS - COMMIT_TS) / 86400 ))
        if (( TIP_TS - COMMIT_TS > 3 * 86400 )); then
            COMMIT_ERR="commit ${SHA} is ${AGE_DAYS} day(s) older than the HEAD of ${RELEASE_BRANCH} (max: 3)"
        fi
    fi
    if [[ -n "${COMMIT_ERR}" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "Warning: ${COMMIT_ERR} (dry run, continuing)."
            CHECKS_PASSED=false
        else
            echo "Error: ${COMMIT_ERR}"
            exit 1
        fi
    else
        echo "Commit ${SHA} is on branch ${RELEASE_BRANCH} and within 3 days of its HEAD - OK"
    fi
fi

echo "Checking that tag ${VERSION} does not already exist..."
if git ls-remote --tags origin "${VERSION}" | grep -q "${VERSION}"; then
    echo "Error: tag ${VERSION} already exists on remote"
    exit 1
fi
echo "Tag ${VERSION} does not exist on remote - OK"

# NOTE: two upstream release gates were removed here rather than reworked:
# (1) a byte-for-byte diff of ggml/src, ggml/include and ggml/CMakeLists.txt
#     against the matching ggml-org/ggml tag, and (2) a lookup of
#     .github/workflows/release.yml CI runs for the release commit.
# Both assume upstream's topology, which this fork intentionally does not
# have: ggml/ carries the TurboQuant+ codec and SYCL changes plus 9+ deleted
# backends (so it never matches upstream byte-for-byte), and release.yml was
# deleted when backends were pruned (afe22f08c), so no commit since then can
# ever have a run recorded against that workflow path - the check was an
# unconditional, permanent block, not a real gate. See CLAUDE.md.
#
# The upstream ghcr.io container-image check is also dropped: its variants
# include cuda/rocm/musa images this fork never builds, and it depends on a
# docker workflow that was deleted along with the rest of .github/.

MAJOR=$(grep "set(GGML_VERSION_MAJOR" "$REPO_ROOT/ggml/CMakeLists.txt" | grep -oP '\d+')
MINOR=$(grep "set(GGML_VERSION_MINOR" "$REPO_ROOT/ggml/CMakeLists.txt" | grep -oP '\d+')
PATCH=$(grep "set(GGML_VERSION_PATCH" "$REPO_ROOT/ggml/CMakeLists.txt" | grep -oP '\d+')
GGML_VERSION="v${MAJOR}.${MINOR}.${PATCH}"
echo "Local ggml version: ${GGML_VERSION}"

echo "Checking API/ABI compatibility..."
set +e
bash "$SCRIPT_DIR/check-release-apiabi.sh"
APIABI_RESULT=$?
set -e
if [[ $APIABI_RESULT -ne 0 ]]; then
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "Warning: API/ABI check found backwards-incompatible changes (dry run, continuing)."
        CHECKS_PASSED=false
    else
        echo "Error: API/ABI check found backwards-incompatible changes."
        exit 1
    fi
else
    echo "API/ABI compatibility check passed - OK"
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "checks_passed=${CHECKS_PASSED}" >> "$GITHUB_OUTPUT"
fi
