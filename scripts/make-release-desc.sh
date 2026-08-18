#!/bin/bash
# Generate the description of a release: the previous release version, the
# change log and the link to the nightly release corresponding to the commit being released.
#
# Usage: make-release-desc.sh <version>
#   <version>: current release version (e.g. v0.1.1)
#
# The previous version is the highest plain semver tag (v<maj>.<min>.<pat>)
# strictly below <version>. The change log lists all commits between the
# previous version tag and HEAD, one line per commit.
#
# The nightly release is resolved the same way as the get-tag-name action in
# release.yml (b<commit-count> on master, <branch>-b<commit-count>-<hash7>
# otherwise); the link is only generated when that release tag exists.
#
# Env (when running in GitHub Actions):
#   GITHUB_OUTPUT: previous_tag, changelog_title, changelog and nightly are written here
#   GITHUB_REPOSITORY: owner/repo, used to build the nightly release URL (skipped when unset)
#   RELEASE_BRANCH: branch the release commit belongs to (defaults to master)
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $(basename "$0") <version>"
    exit 1
fi
VERSION="$1"

# Make sure all remote tags are available locally (skipped on local runs without origin)
if ! git fetch --tags origin 2>/dev/null; then
    echo "Warning: could not fetch tags from origin (local run?)"
fi

PREV="$( { git tag --list; echo "${VERSION}"; } \
    | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' \
    | sort -V \
    | awk -v cur="${VERSION}" '$0 == cur { exit } { prev = $0 } END { print prev }')"

if [[ -n "${PREV}" ]]; then
    CHANGELOG="$(git log --oneline "${PREV}..HEAD")"
    CHANGELOG_TITLE="Change log since ${PREV}"
else
    CHANGELOG="(no previous release tag found)"
    CHANGELOG_TITLE="Change log"
fi

# Nightly release corresponding to HEAD
BRANCH="${RELEASE_BRANCH:-master}"
BUILD_NUMBER="$(git rev-list --count HEAD)"
SHORT_HASH="$(git rev-parse --short=7 HEAD)"
if [[ "${BRANCH}" == "master" ]]; then
    NIGHTLY_TAG="b${BUILD_NUMBER}"
else
    SAFE_NAME="$(echo "${BRANCH}" | tr '/' '-')"
    NIGHTLY_TAG="${SAFE_NAME}-b${BUILD_NUMBER}-${SHORT_HASH}"
fi

NIGHTLY=""
if [[ -n "${GITHUB_REPOSITORY:-}" ]] && git rev-parse -q --verify "refs/tags/${NIGHTLY_TAG}" >/dev/null 2>&1; then
    NIGHTLY_URL="https://github.com/${GITHUB_REPOSITORY}/releases/tag/${NIGHTLY_TAG}"
    NIGHTLY="**Nightly build:** [${NIGHTLY_TAG}](${NIGHTLY_URL})"
    echo "Nightly release: ${NIGHTLY_URL}"
else
    echo "No nightly release found for tag ${NIGHTLY_TAG}"
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
