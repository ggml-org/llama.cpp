#!/usr/bin/env bash
# run_track.sh — Outer loop: invoke AI agent to optimize a CUDA kernel track.
#
# Usage: run_track.sh <track-name> --experiments <N> [--quick-until-improvement] [--max-hours <H>]

set -uo pipefail
# Note: NOT using set -e — we handle errors explicitly to keep the loop running

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Parse args ----
TRACK="${1:?Usage: run_track.sh <track-name> --experiments <N>}"
shift
EXPERIMENTS=10
QUICK_UNTIL_IMPROVEMENT=false
MAX_HOURS=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiments) EXPERIMENTS="$2"; shift 2 ;;
        --quick-until-improvement) QUICK_UNTIL_IMPROVEMENT=true; shift ;;
        --max-hours) MAX_HOURS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

TRACK_DIR="$SCRIPT_DIR/$TRACK"
if [[ ! -f "$TRACK_DIR/program.md" ]]; then
    echo "Error: $TRACK_DIR/program.md not found" >&2
    exit 1
fi

TARGET_FILE=$(jq -r '.target_file' "$TRACK_DIR/baseline.json")
HISTORY_FILE="$TRACK_DIR/history.jsonl"
BASELINE_FILE="$TRACK_DIR/baseline.json"
START_TIME=$(date +%s)
MAX_SECONDS=$((MAX_HOURS * 3600))

# ---- Create experiment branch ----
BRANCH_NAME="autoresearch/$TRACK/$(date +%Y%m%d-%H%M%S)"
cd "$REPO_DIR"
git checkout -b "$BRANCH_NAME" 2>/dev/null || true
BASELINE_SHA=$(git rev-parse HEAD)
echo ">>> Branch: $BRANCH_NAME (base: ${BASELINE_SHA:0:8})" >&2

# ---- Establish baseline ----
echo ">>> Establishing baseline..." >&2
BASELINE_RESULT=$("$SCRIPT_DIR/run_experiment.sh" "$TRACK")
echo "$BASELINE_RESULT" | jq . >&2

BASELINE_STATUS=$(echo "$BASELINE_RESULT" | jq -r '.status')
if [[ "$BASELINE_STATUS" != "success" ]]; then
    echo "Error: baseline benchmark failed: $BASELINE_STATUS" >&2
    exit 1
fi

# Update baseline with current measurements
BASELINE_TG=$(echo "$BASELINE_RESULT" | jq '.tg128')
BASELINE_PPL=$(echo "$BASELINE_RESULT" | jq '.ppl // empty')
jq --argjson tg "$BASELINE_TG" '.tg128 = $tg' "$BASELINE_FILE" > "$BASELINE_FILE.tmp" && mv "$BASELINE_FILE.tmp" "$BASELINE_FILE"
if [[ -n "${BASELINE_PPL:-}" ]]; then
    jq --argjson ppl "$BASELINE_PPL" '.ppl = $ppl' "$BASELINE_FILE" > "$BASELINE_FILE.tmp" && mv "$BASELINE_FILE.tmp" "$BASELINE_FILE"
fi

echo ">>> Baseline: ${BASELINE_TG} t/s" >&2

# ---- Counters ----
CONSECUTIVE_FAILURES=0
CONSECUTIVE_NO_IMPROVEMENT=0
TOTAL_KEPT=0
TOTAL_REVERTED=0
BEST_TG=$BASELINE_TG

# ---- Experiment loop ----
for i in $(seq 1 "$EXPERIMENTS"); do
    ELAPSED=$(( $(date +%s) - START_TIME ))
    if (( ELAPSED > MAX_SECONDS )); then
        echo ">>> Time limit reached (${MAX_HOURS}h). Stopping." >&2
        break
    fi

    echo "" >&2
    echo "================================================================" >&2
    echo ">>> Experiment $i / $EXPERIMENTS" >&2
    echo "================================================================" >&2

    # Determine if we should skip PPL this round
    QUICK_FLAG=""
    if [[ "$QUICK_UNTIL_IMPROVEMENT" == "true" ]]; then
        QUICK_FLAG="--quick"
    fi
    # Every 10th experiment always runs PPL
    if (( i % 10 == 0 )); then
        QUICK_FLAG=""
    fi

    # Build the agent prompt
    LAST_RESULT=""
    if [[ -f "$HISTORY_FILE" ]]; then
        LAST_RESULT=$(tail -1 "$HISTORY_FILE" 2>/dev/null || echo "")
    fi

    STALL_HINT=""
    if (( CONSECUTIVE_NO_IMPROVEMENT >= 5 )); then
        STALL_HINT="IMPORTANT: The last $CONSECUTIVE_NO_IMPROVEMENT experiments showed no improvement. Try a fundamentally different approach — different algorithm, different memory access pattern, different thread mapping."
        CONSECUTIVE_NO_IMPROVEMENT=0
    fi

    AGENT_PROMPT="You are optimizing a CUDA kernel. Read the program file and make ONE modification to improve performance.

Read: $TRACK_DIR/program.md
Read: $REPO_DIR/$TARGET_FILE

$(if [[ -n "$LAST_RESULT" ]]; then echo "Last experiment result: $LAST_RESULT"; fi)
$(if [[ -n "$STALL_HINT" ]]; then echo "$STALL_HINT"; fi)

Current best: ${BEST_TG} t/s (baseline: ${BASELINE_TG} t/s)

Rules:
- Modify ONLY: $TARGET_FILE
- Make exactly ONE conceptual change per experiment
- Do NOT run any builds, benchmarks, or git commands
- After editing, briefly describe what you changed and why (one line to stderr)"

    # Invoke the agent
    echo ">>> Invoking agent..." >&2
    claude -p "$AGENT_PROMPT" \
        --allowedTools Read,Edit \
        --max-turns 20 \
        2>&1 | tee /dev/stderr | tail -1 > /dev/null || true

    # Check if the file was actually modified
    if ! git diff --quiet -- "$TARGET_FILE" 2>/dev/null; then
        # Check no OTHER source files were modified (ignore config/untracked)
        OTHER_CHANGES=$(git diff --name-only -- '*.cu' '*.cuh' '*.cpp' '*.h' '*.c' | grep -vF "$TARGET_FILE" | head -5)
        if [[ -n "$OTHER_CHANGES" ]]; then
            echo ">>> SAFETY: Agent modified non-target files: $OTHER_CHANGES — reverting all" >&2
            git checkout -- .
            RESULT='{"status": "safety_revert", "error": "modified non-target files"}'
        else
            # Run the experiment
            echo ">>> Running experiment..." >&2
            RESULT=$("$SCRIPT_DIR/run_experiment.sh" "$TRACK" $QUICK_FLAG)
            echo "$RESULT" | jq . >&2
        fi
    else
        echo ">>> Agent made no changes. Skipping." >&2
        RESULT='{"status": "no_change"}'
    fi

    STATUS=$(echo "$RESULT" | jq -r '.status')
    TG=$(echo "$RESULT" | jq -r '.tg128 // 0')

    # Decide: keep or revert
    KEPT=false
    if [[ "$STATUS" == "success" ]]; then
        # Check if it's actually faster
        IMPROVEMENT=$(echo "$TG > $BEST_TG" | bc -l 2>/dev/null || echo 0)
        if [[ "$IMPROVEMENT" == "1" ]]; then
            # Speed improvement found. If we were in quick mode, validate PPL now.
            if [[ -n "$QUICK_FLAG" && "$QUICK_UNTIL_IMPROVEMENT" == "true" ]]; then
                echo ">>> Speed improvement found (+$(echo "$RESULT" | jq -r '.delta_pct')). Validating PPL..." >&2
                PPL_RESULT=$("$SCRIPT_DIR/run_experiment.sh" "$TRACK")
                PPL_STATUS=$(echo "$PPL_RESULT" | jq -r '.status')
                if [[ "$PPL_STATUS" == "ppl_regression" ]]; then
                    echo ">>> PPL regression — reverting despite speed gain" >&2
                    git checkout -- "$TARGET_FILE"
                    KEPT=false
                    STATUS="ppl_regression"
                    RESULT="$PPL_RESULT"
                else
                    KEPT=true
                fi
            else
                KEPT=true
            fi

            if [[ "$KEPT" == "true" ]]; then
                BEST_TG="$TG"
                git add "$TARGET_FILE"
                SUMMARY=$(git diff --cached --stat | head -1)
                git commit -m "autoresearch($TRACK): +$(echo "$RESULT" | jq -r '.delta_pct') tg128 (experiment $i)" --no-verify
                # Update baseline
                jq --argjson tg "$TG" '.tg128 = $tg' "$BASELINE_FILE" > "$BASELINE_FILE.tmp" && mv "$BASELINE_FILE.tmp" "$BASELINE_FILE"
                PPL_VAL=$(echo "$RESULT" | jq '.ppl // empty')
                if [[ -n "${PPL_VAL:-}" ]]; then
                    jq --argjson ppl "$PPL_VAL" '.ppl = $ppl' "$BASELINE_FILE" > "$BASELINE_FILE.tmp" && mv "$BASELINE_FILE.tmp" "$BASELINE_FILE"
                fi
                TOTAL_KEPT=$((TOTAL_KEPT + 1))
                CONSECUTIVE_NO_IMPROVEMENT=0
                CONSECUTIVE_FAILURES=0
                echo ">>> KEPT: ${TG} t/s (+$(echo "$RESULT" | jq -r '.delta_pct'))" >&2
            fi
        else
            echo ">>> No improvement (${TG} vs best ${BEST_TG}). Reverting." >&2
            git checkout -- "$TARGET_FILE"
            CONSECUTIVE_NO_IMPROVEMENT=$((CONSECUTIVE_NO_IMPROVEMENT + 1))
        fi
    elif [[ "$STATUS" == "build_failed" ]]; then
        echo ">>> Build failed. Reverting." >&2
        git checkout -- "$TARGET_FILE"
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
    elif [[ "$STATUS" == "runtime_crash" ]]; then
        echo ">>> Runtime crash. Reverting." >&2
        git checkout -- "$TARGET_FILE"
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
    else
        echo ">>> Status: $STATUS. Reverting." >&2
        git checkout -- "$TARGET_FILE"
    fi

    if [[ "$KEPT" == "false" ]]; then
        TOTAL_REVERTED=$((TOTAL_REVERTED + 1))
    fi

    # Log to history
    TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "$RESULT" | jq -c --arg exp "$i" --arg ts "$TIMESTAMP" --argjson kept "$KEPT" \
        '. + {experiment: ($exp|tonumber), timestamp: $ts, kept: $kept}' >> "$HISTORY_FILE"

    # Safety: too many consecutive failures
    if (( CONSECUTIVE_FAILURES >= 3 )); then
        echo ">>> WARNING: 3 consecutive failures. Pausing for review." >&2
        echo ">>> Last error: $(echo "$RESULT" | jq -r '.error // .status')" >&2
        CONSECUTIVE_FAILURES=0
        # Don't exit — just reset counter and let the stall hint kick in
    fi

    # GPU cooldown
    sleep 5
done

# ---- Summary ----
echo "" >&2
echo "================================================================" >&2
echo ">>> AUTORESEARCH COMPLETE" >&2
echo ">>> Track: $TRACK" >&2
echo ">>> Experiments: $((TOTAL_KEPT + TOTAL_REVERTED))" >&2
echo ">>> Kept: $TOTAL_KEPT" >&2
echo ">>> Reverted: $TOTAL_REVERTED" >&2
echo ">>> Baseline: ${BASELINE_TG} t/s → Best: ${BEST_TG} t/s" >&2
if [[ "$BEST_TG" != "$BASELINE_TG" ]]; then
    TOTAL_GAIN=$(echo "scale=1; ($BEST_TG - $BASELINE_TG) / $BASELINE_TG * 100" | bc -l 2>/dev/null || echo "?")
    echo ">>> Total improvement: +${TOTAL_GAIN}%" >&2
fi
echo ">>> Branch: $BRANCH_NAME" >&2
echo ">>> History: $HISTORY_FILE" >&2
echo "================================================================" >&2
