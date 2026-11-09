#!/bin/bash
# Delete GitHub Actions caches matching a key prefix, oldest first.
#
# Usage: ccache-clear.sh --key KEY [--older DURATION] [--min N] [--dry-run]
#   --key:       cache key prefix to match and delete (without the ccache- prefix)
#   --older:     only delete caches created more than DURATION ago (e.g. 5m, 1h, 1d);
#                by default all matching caches are deleted
#   --min:       stop deleting if fewer than N caches would remain (default: 0)
#   --dry-run:   only print the caches that would be deleted, without deleting them
#
# Env (when running in GitHub Actions):
#   GH_TOKEN:          token for the gh CLI
#   GITHUB_REPOSITORY: owner/repo of the caches to manage
set -euo pipefail

KEY=""
OLDER=""
MIN=0
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --key)     [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }; KEY="$2"; shift 2 ;;
        --older)   [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }; OLDER="$2"; shift 2 ;;
        --min)     [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }; MIN="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

command -v gh >/dev/null 2>&1 || { echo "Error: GitHub CLI (gh) is required" >&2; exit 1; }
[[ -n "${GITHUB_REPOSITORY:-}" ]] || { echo "Error: GITHUB_REPOSITORY not set" >&2; exit 1; }
[[ -n "$KEY" ]] || { echo "Error: --key is required" >&2; exit 1; }
[[ "$MIN" =~ ^[0-9]+$ ]] || { echo "Invalid min value: $MIN" >&2; exit 1; }

# Convert a duration (e.g. 90m, 1h, 1d, plain seconds) to seconds
to_seconds() {
  local val="$1"
  [[ "$val" =~ ^[0-9]+$ ]] && { echo "$val"; return 0; }
  local num="${val%?}" unit="${val: -1}" mult
  [[ "$num" =~ ^[0-9]+$ ]] || return 1
  case "$unit" in
    s) mult=1 ;;
    m) mult=60 ;;
    h) mult=3600 ;;
    d) mult=86400 ;;
    *) return 1 ;;
  esac
  echo $((num * mult))
}

# Convert an ISO-8601 UTC timestamp (e.g. 2026-08-23T16:51:23.313693Z) to epoch seconds
to_epoch() {
  local val="$1" out
  # GNU date (e.g. Linux)
  if out=$(date -d "$val" +%s 2>/dev/null) && [[ "$out" =~ ^[0-9]+$ ]]; then
    echo "$out"
    return 0
  fi
  # BSD date (e.g. macOS); fractional seconds are not needed, TZ forces UTC
  out=$(TZ=UTC date -j -f "%Y-%m-%dT%H:%M:%S" "${val:0:19}" +%s 2>/dev/null) || return 1
  [[ "$out" =~ ^[0-9]+$ ]] || return 1
  echo "$out"
}

CACHES=$(gh cache list --repo "$GITHUB_REPOSITORY" --key "ccache-$KEY" --json id,key,createdAt --jq '.[] | [.createdAt, .id, .key] | @tsv' | LC_ALL=C sort)
if [[ -z "$CACHES" ]]; then
    echo "No caches found with key prefix: $KEY"
    exit 0
fi

TOTAL=$(( $(wc -l <<< "$CACHES") ))

echo "Found $TOTAL cache(s) with key prefix: $KEY (oldest first):"
while IFS=$'\t' read -r CREATED ID CACHE_KEY; do
    printf '  %s  %s  %s\n' "$CREATED" "$ID" "$CACHE_KEY"
done <<< "$CACHES"

CUTOFF=""
if [[ -n "$OLDER" ]]; then
    OLDER_SECONDS=$(to_seconds "$OLDER") || { echo "Invalid older value: $OLDER (expected e.g. 90m, 1h, 1d)" >&2; exit 1; }
    CUTOFF=$(( $(date +%s) - OLDER_SECONDS ))
fi

# Caches are sorted oldest first
DELETED=0
while IFS=$'\t' read -r CREATED ID CACHE_KEY; do
    if [[ -n "$CUTOFF" ]]; then
        CREATED_SECONDS=$(to_epoch "$CREATED") || { echo "Failed to parse date: $CREATED" >&2; exit 1; }
        if [[ "$CREATED_SECONDS" -ge "$CUTOFF" ]]; then
            echo "Rest are not older than $OLDER, stopping"
            break
        fi
    fi
    if (( TOTAL - DELETED - 1 < MIN )); then
        echo "Keeping at least $MIN cache(s), stopping"
        break
    fi
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "Would delete cache: $ID ($CACHE_KEY)"
    else
        echo "Deleting cache: $ID ($CACHE_KEY)"
        gh cache delete --repo "$GITHUB_REPOSITORY" "$ID"
    fi
    DELETED=$((DELETED + 1))
done <<< "$CACHES"
