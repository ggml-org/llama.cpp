#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE=ae580510f000811b2f29eed3d436d1d372876dea
EVIDENCE=/home/edwin/models/qwen38-27b-q4s8/checkpoint-fix-final-validation-20260816
OUT=${QWEN38_VERIFY_OUT:-/home/edwin/models/qwen38-27b-q4s8/checkpoint-fix-final-monitor-latest}
CYCLES=${QWEN38_VERIFY_CYCLES:-3}
PORT=${QWEN38_VERIFY_PORT:-18176}
JOBS=${QWEN38_VERIFY_JOBS:-16}
REGEX='^(test-generate-models|test-recurrent-state-rollback|test-save-load-state|test-arg-parser|test-model-resolution|test-opt|test-tensor-split|test-meta-split|test-model-load-cancel|test-autorelease|test-backend-sampler|test-thread-safety)$'

fail() {
    echo "FINAL_VERIFICATION=FAIL: $*" >&2
    exit 1
}

cd "$ROOT"
OUT=$(realpath -m -- "$OUT") || fail "cannot resolve output path"
case "$OUT" in
    /|"$HOME"|"$ROOT"|"$EVIDENCE") fail "unsafe output path: $OUT" ;;
esac
[[ $(git rev-parse --show-toplevel) == "$ROOT" ]] || fail "wrong worktree"
[[ -z $(git status --porcelain) ]] || fail "worktree is not clean"
[[ $(git rev-parse HEAD^) == "$BASE" ]] || fail "HEAD is not a single reviewed commit on the validated base"
git diff --check "$BASE" HEAD

mapfile -t changed < <(git diff --name-only "$BASE" HEAD | LC_ALL=C sort)
expected=(
    scripts/reproduce-qwen38-checkpoint-crash.py
    scripts/reproduce-qwen38-checkpoint-crash.sh
    scripts/reproduce-qwen38-long-checkpoint-rewind.py
    scripts/verify-qwen38-checkpoint-fix.sh
    src/llama-context.cpp
    tests/test-recurrent-state-rollback.cpp
)
[[ ${changed[*]} == "${expected[*]}" ]] || {
    printf 'unexpected changed paths:\n' >&2
    printf '  %s\n' "${changed[@]}" >&2
    fail "reviewed path set changed"
}
[[ $(git diff --numstat "$BASE" HEAD -- src/llama-context.cpp) == $'5\t0\tsrc/llama-context.cpp' ]] ||
    fail "production diff is not the reviewed five-line addition"
if git diff "$BASE" HEAD -- src/llama-context.cpp | grep -Eq 'hipSetDevice|GGML_CUDA_DISABLE_GRAPHS|GGML_HIP_SAFE_STATE_IO'; then
    fail "forbidden device/feature workaround found in production diff"
fi

rm -rf "$OUT"
mkdir -p "$OUT"
{
    printf 'ROOT=%s\nBASE=%s\nHEAD=%s\n' "$ROOT" "$BASE" "$(git rev-parse HEAD)"
    printf 'EVIDENCE=%s\nOUT=%s\nCYCLES=%s\nPORT=%s\nJOBS=%s\n' "$EVIDENCE" "$OUT" "$CYCLES" "$PORT" "$JOBS"
    printf 'PATH=%s\nHOME=%s\nLC_ALL=%s\n' "$PATH" "$HOME" "${LC_ALL:-}"
    git status --short
    git diff --stat "$BASE" HEAD
} > "$OUT/provenance.txt"

grep -Fqx 'GGML_HIP:BOOL=ON' build/CMakeCache.txt || fail "preserved build is not ROCm-enabled"
grep -Fqx 'LLAMA_BUILD_TESTS:BOOL=ON' build/CMakeCache.txt || fail "preserved build does not enable tests"
grep -Fqx 'CMAKE_BUILD_TYPE:STRING=Release' build/CMakeCache.txt || fail "preserved build is not Release"

cmake -S . -B build > "$OUT/configure.log" 2>&1
cmake --build build -j "$JOBS" --target \
    llama-server test-recurrent-state-rollback test-save-load-state \
    test-arg-parser test-model-resolution test-opt test-tensor-split test-meta-split \
    test-model-load-cancel test-autorelease test-backend-sampler test-thread-safety \
    > "$OUT/build.log" 2>&1

(
    cd build
    ctest --output-on-failure -R "$REGEX"
) > "$OUT/ctest.log" 2>&1

env LC_ALL=C "$EVIDENCE/verify-evidence.sh" > "$OUT/evidence.log" 2>&1

./scripts/reproduce-qwen38-checkpoint-crash.sh \
    --server "$ROOT/build/bin/llama-server" \
    --out "$OUT/stress" \
    --port "$PORT" \
    --cycles "$CYCLES" \
    > "$OUT/stress.stdout.log" 2> "$OUT/stress.stderr.log"

expected_restores=$((5 * CYCLES - 1))
grep -Fqx 'READY=1' "$OUT/stress/status.txt"
grep -Fqx 'CLIENT_RC=0' "$OUT/stress/status.txt"
grep -Fqx "COMPLETED_CYCLES=$CYCLES" "$OUT/stress/status.txt"
grep -Fqx 'SERVER_ALIVE_AFTER=1' "$OUT/stress/status.txt"
grep -Fqx 'SERVER_RC=0' "$OUT/stress/status.txt"
grep -Fqx "RESTORES=$expected_restores" "$OUT/stress/status.txt"
grep -Fqx "EXPECTED_RESTORES=$expected_restores" "$OUT/stress/status.txt"
grep -Fqx 'SCHED_RESERVES=3' "$OUT/stress/status.txt"
grep -Fqx 'ERROR_LINES=0' "$OUT/stress/status.txt"
[[ ! -s "$OUT/stress/processes-after.txt" ]] || fail "GPU workload remained after stress"
[[ -z $(git status --porcelain) ]] || fail "verification changed tracked or untracked worktree state"

{
    printf 'CONFIGURE=PASS\nBUILD=PASS\nCTEST=PASS\nEVIDENCE=PASS\n'
    printf 'STRESS_CYCLES=%s\nSTRESS_REQUESTS=%s\nSTRESS_RESTORES=%s\n' \
        "$CYCLES" "$((5 * CYCLES))" "$expected_restores"
    printf 'SCHED_RESERVES=3\nERROR_LINES=0\nTEARDOWN=PASS\nFINAL_VERIFICATION=PASS\n'
} | tee "$OUT/status.txt"

(
    cd "$OUT"
    find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS
)
sha256sum "$OUT/SHA256SUMS"
