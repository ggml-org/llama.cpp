#!/usr/bin/env bash

# initialize a new worktree from a branch name:
#
# - creates a new branch from current HEAD
# - creates a new worktree in a parent folder, suffixed with the branch name
#
# sample usage:
#   ./scripts/wc2wt.sh gg/new-feature-foo-bar
#   ./scripts/wc2wt.sh gg/new-feature-foo-bar opencode
#   ./scripts/wc2wt.sh gg/new-feature-foo-bar "cmake -B build && cmake --build build"
#   ./scripts/wc2wt.sh gg/new-feature-foo-bar "bash -l"

function usage() {
    echo "usage: $0 <branch_name> [cmd]"
    exit 1
}

# check we are in the right directory
if [[ ! -f "scripts/wc2wt.sh" ]]; then
    echo "error: this script must be run from the root of the repository"
    exit 1
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
fi

BRANCH=$1

if [[ -z "$BRANCH" ]]; then
    echo "error: branch name must not be empty"
    exit 1
fi

dir=$(basename $(pwd))
# sanitize branch name for directory name (replace / with -)
dir_suffix=$(echo "$BRANCH" | tr '/' '-')

git worktree add -b "$BRANCH" "../$dir-$dir_suffix" HEAD

og_path=$(pwd)
wt_path=$(cd "../$dir-$dir_suffix" && pwd)

echo "git worktree created in $wt_path"

cd "$wt_path"

# pi agent setup in the worktree
if [[ -f "$og_path/.pi/SYSTEM.md" && ! -f ".pi/SYSTEM.md" ]]; then
    mkdir -p .pi
    ln -sfn "$og_path/.pi/SYSTEM.md" .pi/SYSTEM.md
fi

if [[ $# -eq 2 ]]; then
    echo "executing: $2"
    eval "$2"
fi
