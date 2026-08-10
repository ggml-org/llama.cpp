#!/bin/bash
# Check that release.yml completed successfully for the current HEAD commit.
# Usage: release-check-ci.sh [--dry-run]
# Env: GH_TOKEN, GITHUB_REPOSITORY
set -euo pipefail

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

SHA=$(git rev-parse HEAD)
echo "Checking release.yml status for commit ${SHA}..."

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
