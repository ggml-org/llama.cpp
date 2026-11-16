#!/usr/bin/env bash
# List GitHub Actions workflow runs that are not finished yet, grouped by status.

set -euo pipefail

usage() {
    echo "usage: $(basename "$0") [-r owner/repo] [-s status]... [-b branch] [-w regex] [-t regex] [-f regex] [--kill [-y]]"
    echo "  -r, --repo     repository (default: current)"
    echo "  -s, --status   run status to include, repeatable (default: the not-finished ones)"
    echo "  -b, --branch   branch to fetch runs for; exact match, handled by the API"
    echo "  -w, --workflow case-insensitive regex matched against the workflow name (client-side)"
    echo "  -t, --title    case-insensitive regex matched against the run title (client-side)"
    echo "  -f, --filter   case-insensitive regex matched against the branch name (client-side)"
    echo "  --kill         cancel the selected runs (requires -b, -w, -t or -f)"
    echo "  -y, --yes      skip the confirmation prompt of --kill"
}

repo=""
branch=""
workflow=""
title=""
filter=""
kill=false
assume_yes=false
statuses=()
need_value() { echo "missing value for $1" >&2; exit 1; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--repo)     [[ $# -ge 2 ]] || need_value "$1"; repo="$2"; shift 2 ;;
        -s|--status)   [[ $# -ge 2 ]] || need_value "$1"; statuses+=("$2"); shift 2 ;;
        -b|--branch)   [[ $# -ge 2 ]] || need_value "$1"; branch="$2"; shift 2 ;;
        -w|--workflow) [[ $# -ge 2 ]] || need_value "$1"; workflow="$2"; shift 2 ;;
        -t|--title)    [[ $# -ge 2 ]] || need_value "$1"; title="$2"; shift 2 ;;
        -f|--filter)   [[ $# -ge 2 ]] || need_value "$1"; filter="$2"; shift 2 ;;
        --kill)        kill=true; shift ;;
        -y|--yes)      assume_yes=true; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

# one request per status: the API does not accept multiple statuses at once
[[ ${#statuses[@]} -gt 0 ]] || statuses=(in_progress queued pending waiting requested)

[[ -n "$repo" ]] || repo=$(gh repo view --json nameWithOwner -q .nameWithOwner)

query=("repos/$repo/actions/runs" -f per_page=100)
[[ -z "$branch" ]] || query+=(-f branch="$branch")

selected=$(
    for status in "${statuses[@]}"; do
        gh api --paginate --method GET "${query[@]}" -f status="$status" -q '.workflow_runs[]'
    done | jq -sr --arg branch "$branch" --arg workflow "$workflow" --arg title "$title" --arg filter "$filter" '
        unique_by(.id)
        | map(
            .age = ((now - (.run_started_at // .created_at | fromdate)) / 60 | floor)
            | select(
                ($branch == "" or .head_branch == $branch) and
                ($workflow == "" or (.name | test($workflow; "i"))) and
                ($title == "" or (.display_title | test($title; "i"))) and
                ($filter == "" or (.head_branch | test($filter; "i")))
            )
        )
        | sort_by(.status, -.age)'
)

tsv=$(jq -r '.[] | [ .status, (.age | tostring) + "m", .name, .head_branch, .display_title, .html_url ] | @tsv' <<< "$selected")

if [[ -z "$tsv" ]]; then
    echo "no active runs in $repo"
    exit 0
fi

printf 'STATUS\tAGE\tWORKFLOW\tBRANCH\tTITLE\tURL\n%s\n' "$tsv" | column -t -s $'\t'
echo
printf '%s:' "$repo"
for status in "${statuses[@]}"; do
    printf ' %s=%s' "$status" "$(awk -F'\t' -v s="$status" '$1 == s' <<< "$tsv" | wc -l | tr -d ' ')"
done
echo

if [[ "$kill" == false ]]; then
    exit 0
fi

if [[ -z "$branch$workflow$title$filter" ]]; then
    echo "refusing to cancel everything: pass -b, -w, -t or -f to select the runs to cancel" >&2
    exit 1
fi

count=$(wc -l <<< "$tsv" | tr -d ' ')
if [[ "$assume_yes" == false ]]; then
    printf 'cancel %s run(s)? [y/N] ' "$count"
    read -r reply </dev/tty || { echo; echo "aborted: no terminal to confirm on" >&2; exit 1; }
    [[ "$reply" == [yY]* ]] || { echo "aborted"; exit 1; }
fi

jq -r '.[] | [ .id, .status ] | @tsv' <<< "$selected" |
while IFS=$'\t' read -r id status; do
    gh run cancel "$id" --repo "$repo" || echo "failed to cancel run $id (status: $status)"
done
