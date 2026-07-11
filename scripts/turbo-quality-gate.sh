#!/bin/bash
# TurboQuant quality + speed gate - run BEFORE pushing any changes.
#
# Stages (each emits a deterministic PASS | SKIP (reason) | FAIL (exit=N, log=<path>) line):
#   0  kernel correctness (CPU-vs-SYCL harness)
#   1  perplexity (Turbo KV is FA-only - `-fa off` is a configuration error in strict mode)
#   2  context scaling ratio (Turbo vs q8_0 prefill t/s)
#
# Usage:
#   bash scripts/turbo-quality-gate.sh
#   TURBO_QUALITY_STRICT=1 bash scripts/turbo-quality-gate.sh  # nonzero on any FAIL/SKIP/XFAIL/XPASS
#
# Env vars (sensible script-relative defaults; override to redirect):
#   LLAMA            path to llama.cpp bin dir    (default: $SCRIPT_DIR/../build-port/bin)
#   CORRECTNESS_BIN  path to test-sycl-turbo-correctness (default: $SCRIPT_DIR/../build-port/bin/test-sycl-turbo-correctness)
#                    set to "skip" to bypass the correctness stage (non-strict only)
#   MODEL            path to GGUF                 (no default - strict mode rejects without it)
#   WIKI             path to wikitext-2 test.raw  (no default - strict mode rejects without it)
#   TURBO_QUALITY_STRICT=1   reject non-`1` truthy values - only `1` enables strict
#                            (anything else is treated as `0` / non-strict)
#
# Exit codes:
#   non-strict mode:  0 = green (SKIP allowed), 1 = FAIL/XPASS detected
#   strict mode:      0 = all PASS, 1 = any FAIL, 2 = forbidden SKIP, 124 = timeout
#

set +e  # per-stage functions capture and classify; no implicit abort

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_PORT_BIN="$REPO_ROOT/build-port/bin"

LLAMA=${LLAMA:-$BUILD_PORT_BIN}
CORRECTNESS_BIN=${CORRECTNESS_BIN:-$BUILD_PORT_BIN/test-sycl-turbo-correctness}

# Validate strict mode value: only "1" enables strict. Anything else is treated as 0.
case "${TURBO_QUALITY_STRICT:-0}" in
  1) STRICT=1 ;;
  0|"") STRICT=0 ;;
  *) echo "REJECT: TURBO_QUALITY_STRICT='${TURBO_QUALITY_STRICT}' is not '1'; treating as non-strict (set TURBO_QUALITY_STRICT=1 to enable strict)" >&2; STRICT=0 ;;
esac

# Per-stage tempdir for captured stdout/stderr.
STAGE_LOG_DIR="$(mktemp -d -t turbo-gate.XXXXXX)"
trap 'if [ "${PRESERVE_LOGS:-0}" = "1" ]; then printf "  [preserved logs at %s]\n" "$STAGE_LOG_DIR" >&2; else rm -rf "$STAGE_LOG_DIR"; fi' EXIT

FAIL_COUNT=0
SKIP_COUNT=0
TIMEOUT_COUNT=0
FAIL_MESSAGES=""

# emit_summary: deterministic one-line per stage
emit_summary() {
  local stage="$1" status="$2" log="$3" reason="$4"
  case "$status" in
    PASS)   printf '  PASS | %s\n' "$stage" ;;
    FAIL)   printf '  FAIL | %s (reason=%s, log=%s)\n' "$stage" "$reason" "$log" ;;
    SKIP)   printf '  SKIP | %s (%s)\n' "$stage" "$reason" ;;
    XPASS)  printf '  XPASS | %s (unexpected pass, promote to GATE) - log=%s\n' "$stage" "$log" ;;
    124)    printf '  TIMEOUT | %s (exit=124, log=%s)\n' "$stage" "$log" ;;
    *)      printf '  UNKNOWN | %s (status=%s)\n' "$stage" "$status" ;;
  esac
}

# stage_correctness - always uses -fa on. In strict mode, also runs a
# second pass with LLAMA_TEST_TURBO_FA=1 to exercise the turbo-FA path.
stage_correctness() {
  local stage_label="0.1 correctness (LLAMA_TEST_TURBO_FA=0)"
  local log="$STAGE_LOG_DIR/correctness-a.log"

  if [ "$CORRECTNESS_BIN" = "skip" ]; then
    if [ "$STRICT" = "1" ]; then
      FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label}: CORRECTNESS_BIN=skip is forbidden in strict mode"
      FAIL_COUNT=$((FAIL_COUNT+1))
      emit_summary "$stage_label" "FAIL" "-" "CORRECTNESS_BIN=skip forbidden in strict"
    else
      SKIP_COUNT=$((SKIP_COUNT+1))
      emit_summary "$stage_label" "SKIP" "-" "CORRECTNESS_BIN=skip (non-strict bypass)"
    fi
    return
  fi

  if [ ! -x "$CORRECTNESS_BIN" ]; then
    FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label}: binary missing or not executable at $CORRECTNESS_BIN"
    FAIL_COUNT=$((FAIL_COUNT+1))
    emit_summary "$stage_label" "FAIL" "-" "binary missing at $CORRECTNESS_BIN"
    return
  fi

  if timeout 180 env ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}" "$CORRECTNESS_BIN" >"$log" 2>&1; then
    grep -qE '^== summary: 0 GATE-FAIL,' "$log" || {
      FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label}: harness did not report 0 GATE-FAIL"
      FAIL_COUNT=$((FAIL_COUNT+1))
      emit_summary "$stage_label" "FAIL" "$log" "harness GATE-FAIL non-zero or missing summary"
      return
    }
    emit_summary "$stage_label" "PASS" "$log" ""
  else
    local rc=$?
    if [ "$rc" = "124" ]; then
      TIMEOUT_COUNT=$((TIMEOUT_COUNT+1))
      emit_summary "$stage_label" "124" "$log" ""
    else
      FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label}: harness exited $rc"
      FAIL_COUNT=$((FAIL_COUNT+1))
      emit_summary "$stage_label" "FAIL" "$log" "harness exited $rc"
    fi
  fi

  if [ "$STRICT" = "1" ]; then
    local stage_label2="0.2 correctness (LLAMA_TEST_TURBO_FA=1)"
    local log2="$STAGE_LOG_DIR/correctness-b.log"
    if timeout 180 env ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}" \
         LLAMA_TEST_TURBO_FA=1 "$CORRECTNESS_BIN" >"$log2" 2>&1; then
      grep -qE '^== summary: 0 GATE-FAIL,' "$log2" || {
        FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label2}: harness did not report 0 GATE-FAIL"
        FAIL_COUNT=$((FAIL_COUNT+1))
        emit_summary "$stage_label2" "FAIL" "$log2" "harness GATE-FAIL non-zero or missing summary"
        return
      }
      emit_summary "$stage_label2" "PASS" "$log2" ""
    else
      local rc2=$?
      if [ "$rc2" = "124" ]; then
        TIMEOUT_COUNT=$((TIMEOUT_COUNT+1))
        emit_summary "$stage_label2" "124" "$log2" ""
      else
        FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label2}: harness exited $rc2"
        FAIL_COUNT=$((FAIL_COUNT+1))
        emit_summary "$stage_label2" "FAIL" "$log2" "harness exited $rc2"
      fi
    fi
  fi
}

# validate_numeric <label> <value> - sets METRIC_VALID
validate_numeric() {
  local label="$1" val="$2"
  if [ -z "$val" ] || ! printf '%s' "$val" | grep -qE '^[0-9]+(\.[0-9]+)?$'; then
    echo "    [warn] $label missing or non-numeric: '$val'" >&2
    METRIC_VALID=0
    return
  fi
  METRIC_VALID=1
}

# stage_ppl - perplexity check. -fa on is mandatory.
stage_ppl() {
  local stage_label="1 perplexity (turbo3 vs q8_0, -fa on)"
  local log_t="$STAGE_LOG_DIR/ppl-turbo.log"
  local log_q="$STAGE_LOG_DIR/ppl-q8.log"
  local rc_t rc_q ppl_turbo_valid ppl_q8_valid ppl_limit

  if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    if [ "$STRICT" = "1" ]; then
      FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label}: MODEL env var unset or file not found"
      FAIL_COUNT=$((FAIL_COUNT+1))
      emit_summary "$stage_label" "FAIL" "-" "MODEL unset or not found"
    else
      SKIP_COUNT=$((SKIP_COUNT+1))
      emit_summary "$stage_label" "SKIP" "-" "MODEL unset (non-strict)"
    fi
    return
  fi

  if [ -z "$WIKI" ] || [ ! -f "$WIKI" ]; then
    if [ "$STRICT" = "1" ]; then
      FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label}: WIKI env var unset or file not found"
      FAIL_COUNT=$((FAIL_COUNT+1))
      emit_summary "$stage_label" "FAIL" "-" "WIKI unset or not found"
    else
      SKIP_COUNT=$((SKIP_COUNT+1))
      emit_summary "$stage_label" "SKIP" "-" "WIKI unset (non-strict; no auto-download by design)"
    fi
    return
  fi

  timeout 600 "$LLAMA/llama-perplexity" -m "$MODEL" -f "$WIKI" -c 512 \
    -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 >"$log_t" 2>&1
  rc_t=$?
  timeout 600 "$LLAMA/llama-perplexity" -m "$MODEL" -f "$WIKI" -c 512 \
    -ctk q8_0 -ctv q8_0 -fa on --chunks 8 -ngl 99 >"$log_q" 2>&1
  rc_q=$?

  if [ "$rc_t" = "124" ] || [ "$rc_q" = "124" ]; then
    PRESERVE_LOGS=1
    TIMEOUT_COUNT=$((TIMEOUT_COUNT+1))
    FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label}: timeout (turbo3=$rc_t, q8_0=$rc_q; logs=$log_t + $log_q)"
    emit_summary "$stage_label" "124" "$log_t + $log_q" ""
    return
  fi
  if [ "$rc_t" != "0" ] || [ "$rc_q" != "0" ]; then
    PRESERVE_LOGS=1
    FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label}: run failed (turbo3=$rc_t, q8_0=$rc_q; logs=$log_t + $log_q)"
    FAIL_COUNT=$((FAIL_COUNT+1))
    emit_summary "$stage_label" "FAIL" "$log_t + $log_q" "turbo3=$rc_t, q8_0=$rc_q"
    return
  fi

  PPL_TURBO=$(grep "Final" "$log_t" | grep -oE 'PPL = [0-9.]+' | grep -oE '[0-9.]+' | tail -1)
  PPL_Q8=$(grep "Final" "$log_q" | grep -oE 'PPL = [0-9.]+' | grep -oE '[0-9.]+' | tail -1)
  METRIC_VALID=0
  validate_numeric "PPL_TURBO" "$PPL_TURBO"
  ppl_turbo_valid=$METRIC_VALID
  METRIC_VALID=0
  validate_numeric "PPL_Q8" "$PPL_Q8"
  ppl_q8_valid=$METRIC_VALID
  if [ "$ppl_turbo_valid" != "1" ] || [ "$ppl_q8_valid" != "1" ]; then
    PRESERVE_LOGS=1
    FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label}: final PPL missing or non-numeric (logs=$log_t + $log_q)"
    FAIL_COUNT=$((FAIL_COUNT+1))
    emit_summary "$stage_label" "FAIL" "$log_t + $log_q" "final PPL missing or non-numeric"
    return
  fi

  ppl_limit=$(awk -v q="$PPL_Q8" 'BEGIN { printf "%.6f", q * 1.05 }')
  echo "    turbo3 PPL: $PPL_TURBO, q8_0 PPL: $PPL_Q8, maximum: $ppl_limit" >&2
  echo "    logs: $log_t + $log_q" >&2
  if awk -v t="$PPL_TURBO" -v q="$PPL_Q8" 'BEGIN { exit !(t <= q * 1.05) }'; then
    emit_summary "$stage_label" "PASS" "$log_t + $log_q" ""
  else
    PRESERVE_LOGS=1
    FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label}: turbo3 PPL $PPL_TURBO exceeds 105% of q8_0 PPL $PPL_Q8 (limit=$ppl_limit; logs=$log_t + $log_q)"
    FAIL_COUNT=$((FAIL_COUNT+1))
    emit_summary "$stage_label" "FAIL" "$log_t + $log_q" "turbo3 PPL $PPL_TURBO > limit $ppl_limit (q8_0 $PPL_Q8)"
  fi
}

# stage_scaling - context scaling ratio > 0.95
stage_scaling() {
  local stage_label="2 context-scaling ratio"
  local log_t="$STAGE_LOG_DIR/scaling-turbo.log"
  local log_q="$STAGE_LOG_DIR/scaling-q8.log"
  local rc_t rc_q ratio

  if [ -z "$MODEL" ] || [ ! -f "$MODEL" ] || [ -z "$WIKI" ] || [ ! -f "$WIKI" ]; then
    if [ "$STRICT" = "1" ]; then
      FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label}: MODEL or WIKI unset"
      FAIL_COUNT=$((FAIL_COUNT+1))
      emit_summary "$stage_label" "FAIL" "-" "MODEL or WIKI unset"
    else
      SKIP_COUNT=$((SKIP_COUNT+1))
      emit_summary "$stage_label" "SKIP" "-" "MODEL or WIKI unset (non-strict)"
    fi
    return
  fi

  timeout 600 "$LLAMA/llama-perplexity" -m "$MODEL" -f "$WIKI" -c 4096 \
    -ctk turbo3 -ctv turbo3 -fa on --chunks 4 -ngl 99 >"$log_t" 2>&1
  rc_t=$?
  timeout 600 "$LLAMA/llama-perplexity" -m "$MODEL" -f "$WIKI" -c 4096 \
    -ctk q8_0 -ctv q8_0 -fa on --chunks 4 -ngl 99 >"$log_q" 2>&1
  rc_q=$?

  if [ "$rc_t" = "124" ] || [ "$rc_q" = "124" ]; then
    PRESERVE_LOGS=1
    TIMEOUT_COUNT=$((TIMEOUT_COUNT+1))
    FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label}: timeout (turbo3=$rc_t, q8_0=$rc_q; logs=$log_t + $log_q)"
    emit_summary "$stage_label" "124" "$log_t + $log_q" ""
    return
  fi
  if [ "$rc_t" != "0" ] || [ "$rc_q" != "0" ]; then
    PRESERVE_LOGS=1
    FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label}: run failed (turbo3=$rc_t, q8_0=$rc_q; logs=$log_t + $log_q)"
    FAIL_COUNT=$((FAIL_COUNT+1))
    emit_summary "$stage_label" "FAIL" "$log_t + $log_q" "turbo3=$rc_t, q8_0=$rc_q"
    return
  fi

  TURBO_TPS=$(grep "prompt eval" "$log_t" | grep -oE '[0-9.]+ tokens per second' | grep -oE '[0-9.]+' | tail -1)
  Q8_TPS=$(grep "prompt eval" "$log_q" | grep -oE '[0-9.]+ tokens per second' | grep -oE '[0-9.]+' | tail -1)

  METRIC_VALID=0
  validate_numeric "TURBO_TPS" "$TURBO_TPS"
  local tps_turbo_valid=$METRIC_VALID
  METRIC_VALID=0
  validate_numeric "Q8_TPS" "$Q8_TPS"
  local tps_q8_valid=$METRIC_VALID
  if [ "$tps_turbo_valid" != "1" ] || [ "$tps_q8_valid" != "1" ] || \
     ! awk -v q="$Q8_TPS" 'BEGIN { exit !(q > 0) }'; then
    PRESERVE_LOGS=1
    FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label}: prefill t/s missing, non-numeric, or q8_0 is zero (logs=$log_t + $log_q)"
    FAIL_COUNT=$((FAIL_COUNT+1))
    emit_summary "$stage_label" "FAIL" "$log_t + $log_q" "invalid prefill t/s (turbo3='$TURBO_TPS', q8_0='$Q8_TPS')"
    return
  fi

  ratio=$(awk -v t="$TURBO_TPS" -v q="$Q8_TPS" 'BEGIN { printf "%.6f", t / q }')
  echo "    turbo3: $TURBO_TPS t/s, q8_0: $Q8_TPS t/s, ratio: $ratio (required > 0.95)" >&2
  echo "    logs: $log_t + $log_q" >&2
  if awk -v t="$TURBO_TPS" -v q="$Q8_TPS" 'BEGIN { exit !(t / q > 0.95) }'; then
    emit_summary "$stage_label" "PASS" "$log_t + $log_q" ""
  else
    PRESERVE_LOGS=1
    FAIL_MESSAGES="$FAIL_MESSAGES\n  - ${stage_label}: turbo/q8 throughput ratio $ratio is not > 0.95 (turbo3=$TURBO_TPS, q8_0=$Q8_TPS; logs=$log_t + $log_q)"
    FAIL_COUNT=$((FAIL_COUNT+1))
    emit_summary "$stage_label" "FAIL" "$log_t + $log_q" "ratio $ratio <= 0.95 (turbo3=$TURBO_TPS, q8_0=$Q8_TPS)"
  fi
}

echo "========================================"
echo "  TurboQuant Quality + Speed Gate"
echo "  mode: $([ "$STRICT" = "1" ] && echo strict || echo non-strict)"
echo "========================================"

stage_correctness
stage_ppl
stage_scaling

echo "========================================"
echo "  Summary"
echo "    stages seen:    3 (correctness, ppl, scaling)"
echo "    failures:      $FAIL_COUNT"
echo "    skips:         $SKIP_COUNT"
echo "    timeouts:      $TIMEOUT_COUNT"
[ -n "$FAIL_MESSAGES" ] && printf '  issues:%b\n' "$FAIL_MESSAGES"
echo "========================================"

# Exit code: strict mode exits nonzero on any FAIL, any forbidden SKIP,
PRESERVE_LOGS=0
if [ "$FAIL_COUNT" -gt 0 ]; then
  PRESERVE_LOGS=1
  [ "$TIMEOUT_COUNT" -gt 0 ] && exit 124
  exit 1
fi
if [ "$TIMEOUT_COUNT" -gt 0 ]; then PRESERVE_LOGS=1; exit 124; fi
if [ "$STRICT" = "1" ] && [ "$SKIP_COUNT" -gt 0 ]; then
  PRESERVE_LOGS=1  # strict SKIP usually means a config bug worth diffing
  exit 2
fi
exit 0
