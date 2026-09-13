#!/bin/bash
#
# Release preparation script for llama.cpp.
#
# Bumps the version in CMakeLists.txt on a release candidate branch.
# The branch should then be pushed and a PR created, reviewed, and
# merged. After the PR is merged and the build-cpu workflow has
# completed successfully, the release is finalized by the make-release
# workflow (.github/workflows/make-release.yml), which creates the tag.
#
# Usage:
#   ./scripts/release.sh [major|minor|patch] [--dry-run]
#
# Example:
#   $ ./scripts/release.sh minor
#
# The script:
# 1. Creates a release candidate branch (llama-rc-v<major>.<minor>.<patch>)
# 2. Bumps the version in CMakeLists.txt
# 3. Commits the version bump
#

set -e

if [ ! -f "CMakeLists.txt" ] || [ ! -d "scripts" ]; then
    echo "Error: Must be run from llama.cpp root directory"
    exit 1
fi

# Parse command line arguments
VERSION_TYPE=""
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        major|minor|patch)
            VERSION_TYPE="$arg"
            ;;
        *)
            echo "Error: Unknown argument '$arg'"
            echo "Usage: $0 [major|minor|patch] [--dry-run]"
            exit 1
            ;;
    esac
done

# Default to patch if no version type specified
VERSION_TYPE="${VERSION_TYPE:-patch}"

# Common validation functions
check_git_status() {
    # Check for uncommitted changes (skip in dry-run)
    if [ "$DRY_RUN" = false ] && ! git diff-index --quiet HEAD --; then
        echo "Error: You have uncommitted changes. Please commit or stash them first."
        exit 1
    fi
}

check_master_branch() {
    # Ensure we're on master branch
    CURRENT_BRANCH=$(git branch --show-current)
    if [ "$CURRENT_BRANCH" != "master" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "[dry run] Warning: Not on master branch (currently on: $CURRENT_BRANCH). Continuing with dry-run..."
            echo ""
        else
            echo "Error: Must be on master branch. Currently on: $CURRENT_BRANCH"
            exit 1
        fi
    fi
}

check_master_up_to_date() {
    # Check if we have the latest from master (skip in dry-run)
    if [ "$DRY_RUN" = false ]; then
        echo "Checking if local master is up-to-date with remote..."
        git fetch origin master
        LOCAL=$(git rev-parse HEAD)
        REMOTE=$(git rev-parse origin/master)

        if [ "$LOCAL" != "$REMOTE" ]; then
            echo "Error: Your local master branch is not up-to-date with origin/master."
            echo "Please run 'git pull origin master' first."
            exit 1
        fi
        echo "✓ Local master is up-to-date with remote"
        echo ""
    elif [ "$(git branch --show-current)" = "master" ]; then
        echo "[dry run] Warning: Dry-run mode - not checking if master is up-to-date with remote"
        echo ""
    fi
}

# In-place sed that works on both GNU (Linux) and BSD (macOS) sed
sed_inplace() {
    if sed --version >/dev/null 2>&1; then
        sed -i "$@"
    else
        sed -i '' "$@"
    fi
}

prepare_release() {
    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] Preparing release (no changes will be made)"
    else
        echo "Starting release preparation..."
    fi
    echo ""

    check_git_status
    check_master_branch
    check_master_up_to_date

    # Extract current version from CMakeLists.txt
    echo "Step 1: Reading current version..."
    MAJOR=$(grep "set(LLAMA_VERSION_MAJOR" CMakeLists.txt | sed 's/.*MAJOR \([0-9]*\).*/\1/')
    MINOR=$(grep "set(LLAMA_VERSION_MINOR" CMakeLists.txt | sed 's/.*MINOR \([0-9]*\).*/\1/')
    PATCH=$(grep "set(LLAMA_VERSION_PATCH" CMakeLists.txt | sed 's/.*PATCH \([0-9]*\).*/\1/')

    echo "Current version: $MAJOR.$MINOR.$PATCH"

    # Calculate new version
    case $VERSION_TYPE in
        major)
            NEW_MAJOR=$((MAJOR + 1))
            NEW_MINOR=0
            NEW_PATCH=0
            ;;
        minor)
            NEW_MAJOR=$MAJOR
            NEW_MINOR=$((MINOR + 1))
            NEW_PATCH=0
            ;;
        patch)
            NEW_MAJOR=$MAJOR
            NEW_MINOR=$MINOR
            NEW_PATCH=$((PATCH + 1))
            ;;
    esac

    NEW_VERSION="$NEW_MAJOR.$NEW_MINOR.$NEW_PATCH"
    RC_BRANCH="llama-rc-v$NEW_VERSION"
    echo "New release version: $NEW_VERSION"
    echo "Release candidate branch: $RC_BRANCH"
    echo ""

    # Create release candidate branch
    echo "Step 2: Creating release candidate branch..."
    if [ "$DRY_RUN" = true ]; then
        echo "  [dry-run] Would create branch: $RC_BRANCH"
    else
        git checkout -b "$RC_BRANCH"
        echo "✓ Created and switched to branch: $RC_BRANCH"
    fi
    echo ""

    # Update CMakeLists.txt for release
    echo "Step 3: Updating version in CMakeLists.txt..."
    if [ "$DRY_RUN" = true ]; then
        echo "  [dry-run] Would update LLAMA_VERSION_MAJOR to $NEW_MAJOR"
        echo "  [dry-run] Would update LLAMA_VERSION_MINOR to $NEW_MINOR"
        echo "  [dry-run] Would update LLAMA_VERSION_PATCH to $NEW_PATCH"
    else
        sed_inplace -e "s/set(LLAMA_VERSION_MAJOR [0-9]*)/set(LLAMA_VERSION_MAJOR $NEW_MAJOR)/" CMakeLists.txt
        sed_inplace -e "s/set(LLAMA_VERSION_MINOR [0-9]*)/set(LLAMA_VERSION_MINOR $NEW_MINOR)/" CMakeLists.txt
        sed_inplace -e "s/set(LLAMA_VERSION_PATCH [0-9]*)/set(LLAMA_VERSION_PATCH $NEW_PATCH)/" CMakeLists.txt
    fi
    echo ""

    # Commit version bump
    echo "Step 4: Committing version bump..."
    if [ "$DRY_RUN" = true ]; then
        echo "  [dry-run] Would commit: 'llama.cpp : bump version to $NEW_VERSION'"
    else
        git add CMakeLists.txt
        git commit -m "llama.cpp : bump version to $NEW_VERSION"
    fi
    echo ""

    echo ""
    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] Summary (no changes were made):"
        echo "  • Would have created branch: $RC_BRANCH"
        echo "  • Would have updated version to: $NEW_VERSION"
    else
        echo "Release preparation completed!"
        echo "Summary:"
        echo "  • Created branch: $RC_BRANCH"
        echo "  • Updated version to: $NEW_VERSION"
        echo ""
        echo "Next steps:"
        echo "  • Push branch to remote: git push origin $RC_BRANCH"
        echo "  • Create a Pull Request from $RC_BRANCH to master"
        echo "  • After the PR is merged and the build-cpu workflow has passed,"
        echo "    create the release with the make-release workflow"
        echo "    (.github/workflows/make-release.yml)"
    fi
}

prepare_release
