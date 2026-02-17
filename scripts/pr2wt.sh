#!/usr/bin/env bash

# intialize a new worktree from a PR number:
#
# - creates a new remote using the fork's clone URL
# - creates a local branch tracking the remote branch
# - creates a new worktree in a parent folder, suffixed with "-pr-$PR"
#
# sample usage:
#   ./scripts/pr2wt.sh 12345
#   ./scripts/pr2wt.sh 12345 opencode
#   ./scripts/pr2wt.sh 12345 "cmake -B build && cmake --build build"
#   ./scripts/pr2wt.sh 12345 "bash -l"

function usage() {
    echo "usage: $0 <pr_number> [cmd]"
    exit 1
}

# check we are in the right directory
if [[ ! -f "scripts/pr2wt.sh" ]]; then
    echo "error: this script must be run from the root of the repository"
    exit 1
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
fi

PR=$1
[[ "$PR" =~ ^[0-9]+$ ]] || { echo "error: PR number must be numeric"; exit 1; }

url_origin=$(git config --get remote.upstream.url 2>/dev/null) || \
url_origin=$(git config --get remote.origin.url) || {
    echo "error: no remote named 'upstream' or 'origin' in this repository"
    exit 1
}

# Extract org/repo from either https or ssh format.
if [[ $url_origin =~ ^git@ ]]; then
    org_repo=$(echo $url_origin | cut -d: -f2)
else
    org_repo=$(echo $url_origin | cut -d/ -f4-)
fi
org_repo=${org_repo%.git}

echo "org/repo: $org_repo"

meta=$(curl -sSLf -H "Accept: application/vnd.github+json" "https://api.github.com/repos/$org_repo/pulls/$PR")

url_remote=$(echo "$meta" | jq -r '.head.repo.clone_url')
head_ref=$(echo "$meta" | jq -r '.head.ref')

echo "url:      $url_remote"
echo "head_ref: $head_ref"

url_remote_cur=$(git config --get "remote.pr/$PR.url" 2>/dev/null || true)

if [[ "$url_remote_cur" != "$url_remote" ]]; then
    git remote rm  pr/$PR 2> /dev/null
    git remote add pr/$PR "$url_remote"
fi

git fetch "pr/$PR" "$head_ref"

dir=$(basename $(pwd))

git branch -D pr/$PR 2> /dev/null
git worktree add -b pr/$PR ../$dir-pr-$PR pr/$PR/$head_ref 2> /dev/null

wt_path=$(cd ../$dir-pr-$PR && pwd)

echo "git worktree created in $wt_path"

cd $wt_path
git branch --set-upstream-to=pr/$PR/$head_ref
git pull   --ff-only || {
    echo "error: failed to pull pr/$PR"
    exit 1
}

if [[ $# -eq 2 ]]; then
    echo "executing: $2"
    eval "$2"
fi
