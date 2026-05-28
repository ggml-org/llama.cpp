#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPPLINT_BIN="${CPPLINT_BIN:-cpplint}"

usage() {
    cat <<'USAGE'
Usage:
  user_scripts/cpplint-check.sh [--changed|--all] [-- cpplint args...]

Run cpplint on C/C++ files after cpplint-style filtering.

Modes:
  --changed   Use files changed in the current Git working tree:
              unstaged files, staged files, and untracked files.
  --all       Use all files under the current repository directory.

Examples:
  user_scripts/cpplint-check.sh
  user_scripts/cpplint-check.sh --all

Environment:
  CPPLINT_BIN  cpplint executable. Default: cpplint.
USAGE
}

is_cpplint_file() {
    case "$1" in
        *.c|*.cc|*.cpp|*.cxx|*.h|*.hh|*.hpp|*.hxx)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

is_excluded_path() {
    case "$1" in
        .git/*|build*|vendor/*|tools/ui/node_modules/*|examples/*|tests/* \
        |common/jinja/value.h \
        |common/jinja/string.h \
        |ggml/src/ggml-common.h \
        )
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

filter_cpplint_files() {
    while IFS= read -r file; do
        file="${file#./}"
        if [[ -f "$file" ]] && ! is_excluded_path "$file" && is_cpplint_file "$file"; then
            printf '%s\n' "$file"
        fi
    done
}

get_all_files() {
    find . -type f -print
}

get_changed_files() {
    {
        git diff --name-only --diff-filter=ACMRTUXB
        git diff --cached --name-only --diff-filter=ACMRTUXB
        git ls-files --others --exclude-standard
    } | sort -u
}

get_cpplint_check_files() {
    case "$MODE" in
        --changed)
            get_changed_files | filter_cpplint_files
            ;;
        --all)
            get_all_files | filter_cpplint_files | sort -u
            ;;
        *)
            echo "Unknown mode: $MODE" >&2
            usage >&2
            exit 2
            ;;
    esac
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

MODE="--changed"
CPPLINT_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --changed|--all)
            MODE="$1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            CPPLINT_ARGS+=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cd "$ROOT_DIR"

mapfile -t FILES < <(get_cpplint_check_files)

if [[ "${#FILES[@]}" -eq 0 ]]; then
    exit 0
fi

echo "Running $CPPLINT_BIN on ${#FILES[@]} files..."
"$CPPLINT_BIN" "${CPPLINT_ARGS[@]}" "${FILES[@]}"
