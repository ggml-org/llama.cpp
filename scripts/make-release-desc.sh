#!/bin/bash
# Generate the description of a release: the previous release version, the
# change log and the link to the nightly release corresponding to the commit being released.
#
# Usage: make-release-desc.sh <version>
#   <version>: current release version (v<maj>.<min>.<pat>, the leading v is optional)
#
# The previous version is the highest plain semver tag (v<maj>.<min>.<pat>)
# strictly below <version>. The change log lists all commits between the
# previous version tag and the release commit, one line per commit.
#
# The release commit is the commit <version> points at when the tag exists,
# HEAD otherwise. The nightly release is the b* tag pointing at that commit
# (release.yml tags the same commit); the link is only generated when that
# tag exists.
#
# Env (when running in GitHub Actions):
#   GITHUB_OUTPUT: previous_tag, changelog_title, changelog and nightly are written here
#   GITHUB_REPOSITORY: owner/repo, used to build the nightly release URL (skipped when unset)
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $(basename "$0") <version>"
    exit 1
fi
VERSION="$1"

# Accept the version with or without the leading v, reject anything else
if [[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    VERSION="v${VERSION}"
elif [[ ! "${VERSION}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: invalid version '${VERSION}' (expected v<maj>.<min>.<pat>)"
    exit 1
fi

# Make sure all remote tags are available locally (skipped on local runs without origin)
if ! git fetch --tags origin 2>/dev/null; then
    echo "Warning: could not fetch tags from origin (local run?)"
fi

# Release commit: the commit <version> points at when the tag exists, HEAD otherwise.
if ! RELEASE_COMMIT="$(git rev-parse -q --verify "refs/tags/${VERSION}^{commit}" 2>/dev/null)"; then
    RELEASE_COMMIT="$(git rev-parse HEAD)"
fi

echo "Release commit: $(git rev-parse --short "${RELEASE_COMMIT}")"

PREV="$( { git tag --list; echo "${VERSION}"; } \
    | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' \
    | sort -V \
    | awk -v cur="${VERSION}" '$0 == cur { exit } { prev = $0 } END { print prev }')"

if [[ -n "${PREV}" ]]; then
    CHANGELOG="$(git log --oneline "${PREV}..${RELEASE_COMMIT}")"
    CHANGELOG_TITLE="Change log since ${PREV}"
else
    CHANGELOG="(no previous release tag found)"
    CHANGELOG_TITLE="Change log"
fi

# Nightly release: the b* tag pointing at the release commit (|| true: no match is not an error)
NIGHTLY_TAG="$(git tag --points-at "${RELEASE_COMMIT}" | grep -E '(^|-)b[0-9]+(-[0-9a-f]{7})?$' | head -n 1 || true)"

NIGHTLY=""
if [[ -n "${NIGHTLY_TAG}" ]]; then
    if [[ -n "${GITHUB_REPOSITORY:-}" ]]; then
        NIGHTLY_URL="https://github.com/${GITHUB_REPOSITORY}/releases/tag/${NIGHTLY_TAG}"
        NIGHTLY="**Nightly build:** [${NIGHTLY_TAG}](${NIGHTLY_URL})"
        echo "Nightly release: ${NIGHTLY_URL}"
    fi
else
    echo "No nightly release found for commit $(git rev-parse --short "${RELEASE_COMMIT}")"
fi

echo "Previous version: ${PREV:-none}"
echo "${CHANGELOG}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
        echo "previous_tag=${PREV}"
        echo "changelog_title=${CHANGELOG_TITLE}"
        echo "nightly=${NIGHTLY}"
        echo "changelog<<CHANGELOG_EOF"
        echo "${CHANGELOG}"
        echo "CHANGELOG_EOF"
    } >> "${GITHUB_OUTPUT}"
fi
