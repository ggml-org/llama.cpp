#!/usr/bin/env bash
#
# alphaevolve-run.sh - control panel for a multi-instance alphaevolve run.
#
# WHAT THIS DOES:
#   - Generates the exact Agent invocations for the init / join / finalize phases.
#   - Watches the shared run directory and reports live status as agents work.
#   - Prints cleanup commands once finalize is done.
#
# WHAT THIS DOES NOT DO:
#   - It cannot launch ZCode Agent tool calls itself. The Agent tool is a
#     session-internal call, not a shell command. You fire the printed
#     invocations in separate ZCode sessions; that parallelism is the point.
#   - It never commits, pushes, runs gh, or creates branches. Pure observation.
#
# USAGE:
#   scripts/alphaevolve-run.sh init "<goal>"            print the init command
#   scripts/alphaevolve-run.sh layout <run> [N]         print N join commands + finalize
#   scripts/alphaevolve-run.sh watch <run>              tail changes.md + status until done
#   scripts/alphaevolve-run.sh status <run>             one-shot status snapshot
#   scripts/alphaevolve-run.sh cleanup <run>            print worktree-remove commands
#
# A "run" is the slug the alphaevolve init phase chose (it writes
# .zcode/alphaevolve/<run>/spec.md). Find runs with: scripts/alphaevolve-run.sh list

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
AE_DIR="$ROOT/.zcode/alphaevolve"

die()  { printf 'alphaevolve-run: %s\n' "$*" >&2; exit 1; }

need_run_dir() {
    local run="$1"
    [[ -d "$AE_DIR/$run" ]] || {
        printf 'alphaevolve-run: run %s not found at %s/%s\n' "'$run'" "$AE_DIR" "$run" >&2
        if [[ -d "$AE_DIR" ]] && [[ -n "$(ls -A "$AE_DIR" 2>/dev/null)" ]]; then
            printf 'existing runs:\n' >&2
            list_runs >&2
        else
            printf 'no runs exist yet under %s\n' "$AE_DIR" >&2
        fi
        exit 1
    }
}

list_runs() {
    [[ -d "$AE_DIR" ]] || return 0
    local d
    for d in "$AE_DIR"/*/; do
        [[ -d "$d" ]] || continue
        local name; name="$(basename "$d")"
        local spec="$d/spec.md"
        local status="(no spec.md)"
        if [[ -f "$spec" ]]; then
            if [[ -f "$d/best.md" ]]; then
                status="finalized"
            elif [[ -f "$d/gene-ledger.json" ]]; then
                status="running"
            else
                status="bootstrapped"
            fi
        fi
        printf '  %s\t%s\n' "$name" "$status"
    done
}

# ---- init --------------------------------------------------------------------
cmd_init() {
    local goal="${1:-}"
    [[ -n "$goal" ]] || die "usage: $0 init \"<goal>\""
    printf 'Fire this in a ZCode session:\n\n'
    printf '  Agent(alphaevolve, "init: %s")\n\n' "$goal"
    printf 'After it exits, it will have written .zcode/alphaevolve/<run>/spec.md\n'
    printf 'with the run slug. Then run:\n\n'
    printf '  %s layout <run> 4\n' "$0"
}

# ---- layout: print the join + finalize commands ------------------------------
cmd_layout() {
    local run="${1:-}"
    local n="${2:-4}"
    [[ -n "$run" ]] || die "usage: $0 layout <run> [N]"
    need_run_dir "$run"

    # How many seeds did init produce? Use that to sanity-check N.
    local seeds=""
    if [[ -f "$AE_DIR/$run/gene-ledger.json" ]]; then
        seeds="$(grep -o '"gene_id"' "$AE_DIR/$run/gene-ledger.json" | wc -l | tr -d ' ')"
    fi
    local agents=(north south east west)  # up to 4 named agents; extend if you run more
    local i
    printf 'Run: %s   seeds available: %s   joiners: %s\n\n' "$run" "${seeds:-?}" "$n"
    if [[ -n "$seeds" && "$seeds" -lt "$n" ]]; then
        printf 'NOTE: fewer seeds (%s) than joiners (%s). Some agents will claim 0 genes.\n\n' "$seeds" "$n"
    fi

    printf 'Fire these %s invocations in SEPARATE ZCode sessions, in parallel:\n\n' "$n"
    for (( i=0; i<n; i++ )); do
        local id="${agents[$((i % ${#agents[@]}))]}"
        printf '  Agent(alphaevolve, "join %s as %s genes 1")\n' "$run" "$id"
    done
    printf '\nWait for all joiners to exit (their genes freeze, promote, and purge).\n'
    printf 'Then fire finalize in any session:\n\n'
    printf '  Agent(alphaevolve, "finalize %s")\n\n' "$run"
    printf 'Watch progress in another terminal:\n\n'
    printf '  %s watch %s\n' "$0" "$run"
}

# ---- status: one-shot snapshot ----------------------------------------------
# Count regex matches in a file. Returns 0 (not empty) when no matches and never
# fails, so it composes safely under `set -e` / `pipefail`. The explicit
# `return 0` is essential: grep exits 1 on no-match, pipefail propagates it, and
# without this the assignment `var=$(count ...)` would abort the script.
count() {
    local pat="$1" file="$2"
    grep -o -- "$pat" "$file" 2>/dev/null | wc -l | tr -d ' '
    return 0
}

cmd_status() {
    local run="${1:-}"
    [[ -n "$run" ]] || die "usage: $0 status <run>"
    need_run_dir "$run"
    local rd="$AE_DIR/$run"

    printf 'run: %s\n' "$run"
    if [[ -f "$rd/spec.md" ]]; then
        printf 'baseline_sha: %s\n' \
            "$(sed -n 's/^baseline_sha:[[:space:]]*//p' "$rd/spec.md" | head -1)"
    fi

    if [[ ! -f "$rd/gene-ledger.json" ]]; then
        printf 'state: no gene-ledger yet (init still running?)\n'
        return
    fi

    local total live frozen promoted purged integrated noncomp
    total="$(count '"gene_id"'              "$rd/gene-ledger.json")"
    live="$(count '"status": *"live"'        "$rd/gene-ledger.json")"
    frozen="$(count '"status": *"frozen"'    "$rd/gene-ledger.json")"
    promoted="$(count '"status": *"promoted"' "$rd/gene-ledger.json")"
    purged="$(count '"status": *"purged"'    "$rd/gene-ledger.json")"
    integrated="$(count '"stacked_on_main": *true'  "$rd/gene-ledger.json")"
    noncomp="$(count '"stacked_on_main": *false' "$rd/gene-ledger.json")"
    printf 'genes: total=%s  live=%s  frozen=%s  promoted=%s  purged=%s\n' \
        "$total" "$live" "$frozen" "$promoted" "$purged"
    printf 'integration/main: stacked=%s  not-stacked=%s\n' "$integrated" "$noncomp"

    if [[ -f "$rd/changes.md" ]]; then
        local evals; evals="$(count 'verdict=' "$rd/changes.md")"
        printf 'candidates evaluated: %s\n' "$evals"
    fi
    if [[ -f "$rd/best.md" ]]; then
        printf 'best.md present (finalize has run)\n'
    fi

    # Champion branches in the integration repo.
    local integ="$rd/integration"
    if [[ -d "$integ/.git" ]] || git -C "$integ" rev-parse --git-dir >/dev/null 2>&1; then
        printf 'champion branches:\n'
        git -C "$integ" for-each-ref --format='  %(refname:short)' 'refs/heads/champions/*' 2>/dev/null || true
        printf 'integration/main log (last 5):\n'
        git -C "$integ" log --oneline -5 integration/main 2>/dev/null | sed 's/^/  /' || true
    fi
}

# ---- watch: tail changes.md and refresh status until best.md appears ---------
cmd_watch() {
    local run="${1:-}"
    [[ -n "$run" ]] || die "usage: $0 watch <run>"
    need_run_dir "$run"
    local rd="$AE_DIR/$run"
    local changes="$rd/changes.md"

    printf 'Watching %s\n' "$run"
    printf 'Press Ctrl-C to stop watching (the run keeps going in its sessions).\n\n'

    # Initial snapshot.
    cmd_status "$run"
    printf '\n--- changes.md (live) ---\n'

    if [[ -f "$changes" ]]; then
        tail -n 50 "$changes"
    else
        printf '(no changes.md yet - init may still be researching)\n'
    fi

    # Tail forever; refresh status every 30s. Stop when best.md appears.
    local since=0
    while [[ ! -f "$rd/best.md" ]]; do
        # Append any new lines from changes.md.
        if [[ -f "$changes" ]]; then
            local total; total="$(wc -l < "$changes" | tr -d ' ')"
            if (( total > since )); then
                tail -n "+$((since + 1))" "$changes"
                since="$total"
            fi
        fi
        sleep 5
        # Periodic status refresh on stderr so it doesn't break the log stream.
        if (( RANDOM % 6 == 0 )); then
            printf '\n--- status snapshot ---\n' >&2
            cmd_status "$run" >&2 || true
        fi
    done

    printf '\n--- finalize complete (best.md present) ---\n'
    cmd_status "$run"
    printf '\nTo clean up:\n  %s cleanup %s\n' "$0" "$run"
}

# ---- cleanup: print (do not run) removal commands ----------------------------
cmd_cleanup() {
    local run="${1:-}"
    [[ -n "$run" ]] || die "usage: $0 cleanup <run>"
    need_run_dir "$run"
    local rd="$AE_DIR/$run"

    printf '# Review before running any of these. The champion branches and\n'
    printf '# integration/patches/ hold the durable record; once you are done\n'
    printf '# with a run you can remove the worktrees and the run directory.\n\n'

    # If finalize ran, point at the review branch on the main repo first.
    if git rev-parse --verify "evolve-review/$run" >/dev/null 2>&1; then
        printf '# review branch on the main repo (merge, cherry-pick, or reset as you like):\n'
        printf '  git log --oneline %%s..evolve-review/%s\n' "$run"
        printf '  git diff %%s...evolve-review/%s   # %%s = the baseline you branched from\n\n' "$run"
    fi

    if git -C "$rd/integration" rev-parse --git-dir >/dev/null 2>&1; then
        printf '# remove the integration worktree and its repo:\n'
        printf '  git worktree remove --force "%s/integration"\n' "$rd"
        printf '  git -C "%s/integration" branch -D integration/main 2>/dev/null || true\n' "$rd"
        # Champion branches live in the integration repo's object store; once the
        # worktree is removed they are unreachable and will be GC'd eventually.
        printf '  # champion branches (champions/<gene-id>) become unreachable after the\n'
        printf '  # worktree is removed; run "git -C <integration-repo> gc" if you keep it.\n\n'
    fi

    # Any lingering gene worktrees an agent forgot to purge.
    local gw
    if [[ -d "$rd/worktrees" ]]; then
        printf '# remove any lingering gene worktrees:\n'
        for gw in "$rd"/worktrees/*/; do
            [[ -d "$gw" ]] || continue
            local name; name="$(basename "$gw")"
            [[ "$name" == *.claim-* ]] && continue   # claim-marker dirs, not worktrees
            # Only attempt worktree removal for actual git worktrees.
            if git -C "$gw" rev-parse --git-dir >/dev/null 2>&1; then
                printf '  git worktree remove --force "%s"\n' "$gw"
            fi
        done
        printf '\n'
    fi

    printf '# finally, remove the run directory (keeps the rest of .zcode/alphaevolve):\n'
    printf '  rm -rf "%s"\n' "$rd"
    printf '\n# or list remaining runs:\n  %s list\n' "$0"
}

# ---- list --------------------------------------------------------------------
cmd_list() {
    [[ -d "$AE_DIR" ]] || { printf 'no runs at %s\n' "$AE_DIR"; exit 0; }
    printf 'runs under %s:\n' "$AE_DIR"
    list_runs
}

# ---- dispatch ----------------------------------------------------------------
sub="${1:-}"
shift || true
case "$sub" in
    init)    cmd_init "$@" ;;
    layout)  cmd_layout "$@" ;;
    status)  cmd_status "$@" ;;
    watch)   cmd_watch "$@" ;;
    cleanup) cmd_cleanup "$@" ;;
    list)    cmd_list "$@" ;;
    ""|-h|--help|help)
        sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
        printf '\nsubcommands: init, layout, status, watch, cleanup, list\n'
        ;;
    *) die "unknown subcommand '$sub' (try: $0 --help)" ;;
esac
