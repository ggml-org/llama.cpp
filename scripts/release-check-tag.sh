#!/bin/bash
# Check that a release tag does not already exist on the remote.
# Usage: release-check-tag.sh <version>
#   version: the tag to check (e.g. v0.1.0)
set -euo pipefail

if [[ -z "${1:-}" ]]; then
    echo "Error: version argument is required"
    exit 1
fi

VERSION="$1"

echo "Checking that tag ${VERSION} does not already exist..."
if git ls-remote --tags origin "${VERSION}" | grep -q "${VERSION}"; then
    echo "Error: tag ${VERSION} already exists on remote"
    exit 1
fi

echo "Tag ${VERSION} does not exist on remote - OK"
