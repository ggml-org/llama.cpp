#!/bin/bash
# Run all pre-release checks and determine the release version.
#
# Usage: make-release-checks.sh [--dry-run]
#   --dry-run: warn on failures instead of aborting
#
# Env (when running in GitHub Actions): GH_TOKEN, GITHUB_REPOSITORY, GITHUB_OUTPUT
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

MAJOR=$(grep "set(LLAMA_VERSION_MAJOR" "$REPO_ROOT/CMakeLists.txt" | grep -oP '\d+')
MINOR=$(grep "set(LLAMA_VERSION_MINOR" "$REPO_ROOT/CMakeLists.txt" | grep -oP '\d+')
PATCH=$(grep "set(LLAMA_VERSION_PATCH" "$REPO_ROOT/CMakeLists.txt" | grep -oP '\d+')
VERSION="v${MAJOR}.${MINOR}.${PATCH}"
echo "Determined version: ${VERSION}"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "version=${VERSION}" >> "$GITHUB_OUTPUT"
fi

echo "Checking that tag ${VERSION} does not already exist..."
if git ls-remote --tags origin "${VERSION}" | grep -q "${VERSION}"; then
    echo "Error: tag ${VERSION} already exists on remote"
    exit 1
fi
echo "Tag ${VERSION} does not exist on remote - OK"

SHA=$(git rev-parse HEAD)
echo "Checking release.yml status for commit ${SHA}..."
if [[ -z "${GITHUB_REPOSITORY:-}" ]]; then
    echo "Warning: GITHUB_REPOSITORY not set - skipping CI check (local run)"
else
    RUNS=$(gh api "repos/${GITHUB_REPOSITORY}/actions/workflows/release.yml/runs" \
        --jq "[.workflow_runs[] | select(.head_sha == \"${SHA}\" and .conclusion == \"success\")] | length")
    if [[ "$RUNS" -eq 0 ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "Warning: no successful release.yml run found for HEAD (${SHA}) (dry run, continuing)."
        else
            echo "Error: no successful release.yml run found for HEAD (${SHA})"
            echo "The nightly build must complete successfully before making a release."
            exit 1
        fi
    else
        echo "Found successful release.yml run for HEAD."
    fi
fi

MAJOR=$(grep "set(GGML_VERSION_MAJOR" "$REPO_ROOT/ggml/CMakeLists.txt" | grep -oP '\d+')
MINOR=$(grep "set(GGML_VERSION_MINOR" "$REPO_ROOT/ggml/CMakeLists.txt" | grep -oP '\d+')
PATCH=$(grep "set(GGML_VERSION_PATCH" "$REPO_ROOT/ggml/CMakeLists.txt" | grep -oP '\d+')
GGML_VERSION="v${MAJOR}.${MINOR}.${PATCH}"
echo "Local ggml version: ${GGML_VERSION}"

if ! git clone --depth 1 --branch "${GGML_VERSION}" https://github.com/ggml-org/ggml.git upstream-ggml 2>/dev/null; then
    echo "Warning: tag ${GGML_VERSION} not found in upstream ggml - skipping comparison"
else
    echo "Comparing local ggml/ src and include with upstream ${GGML_VERSION}..."
    DIFF=$(diff -rq "$REPO_ROOT/ggml/src"          upstream-ggml/src          2>&1 || true)
    DIFF+=$(diff -rq "$REPO_ROOT/ggml/include"     upstream-ggml/include      2>&1 || true)
    DIFF+=$(diff     "$REPO_ROOT/ggml/CMakeLists.txt" upstream-ggml/CMakeLists.txt 2>&1 || true)
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
fi
