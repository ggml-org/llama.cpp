#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRIVER="$REPO_ROOT/scripts/validate-dense-turbo4-capacity.sh"
SOURCE_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
TEST_ROOT="$(mktemp -d -t p45a-contract.XXXXXX)"
trap 'rm -rf "$TEST_ROOT"' EXIT

PASS_COUNT=0

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  printf 'PASS: %s\n' "$1"
}

make_fixture() {
  local name="$1"
  FIXTURE="$TEST_ROOT/$name"
  FAKE_BIN="$FIXTURE/fake-bin"
  BUILD_BIN="$FIXTURE/build/bin"
  CALL_DIR="$FIXTURE/calls"
  MODEL_DIR="$FIXTURE/models with space"
  mkdir -p "$FAKE_BIN" "$BUILD_BIN" "$CALL_DIR" "$MODEL_DIR"

  MISTRAL_MODEL="$MODEL_DIR/mistral.gguf"
  LLAMA31_MODEL="$MODEL_DIR/llama31.gguf"
  WIKI="$MODEL_DIR/wiki.test.raw"
  FAKE_LIB="$FIXTURE/build/libggml-sycl.so"
  printf 'mistral\n' >"$MODEL_DIR/mistral-target.gguf"
  ln -s "$MODEL_DIR/mistral-target.gguf" "$MISTRAL_MODEL"
  printf 'llama31\n' >"$LLAMA31_MODEL"
  printf 'corpus\n' >"$WIKI"
  printf 'sycl\n' >"$FAKE_LIB"

  cat >"$FAKE_BIN/ldd" <<'EOF'
#!/bin/bash
printf 'libggml-sycl.so => %s (0x00000000)\n' "$FAKE_LIB"
EOF

  cat >"$FAKE_BIN/fuser" <<'EOF'
#!/bin/bash
count_file="$CALL_DIR/fuser"
count=0
[ ! -f "$count_file" ] || count="$(cat "$count_file")"
printf '%s\n' "$((count + 1))" >"$count_file"
if [ "${FAKE_CASE:-}" = occupied ]; then
  printf '                     USER        PID ACCESS COMMAND\n' >&2
  printf '/dev/dri/renderD128: test       4242 F...m holder\n' >&2
  exit 0
fi
exit 1
EOF

  cat >"$FAKE_BIN/sycl-ls" <<'EOF'
#!/bin/bash
printf '[level_zero:gpu:0] Fake Intel Arc A770\n'
EOF

  cat >"$BUILD_BIN/test-sycl-turbo-correctness" <<'EOF'
#!/bin/bash
if [ "${1:-}" = --version ]; then
  printf 'fake-correctness 1\n'
  exit 0
fi
printf '1\n' >"$CALL_DIR/harness"
if [ "${FAKE_CASE:-}" = harness_fail ]; then
  printf '== summary: 1 GATE-FAIL, 0 XFAIL ==\n'
  exit 0
fi
printf '== summary: 0 GATE-FAIL, 0 XFAIL ==\n'
EOF

  cat >"$BUILD_BIN/llama-perplexity" <<'EOF'
#!/bin/bash
if [ "${1:-}" = --version ]; then
  printf 'fake-perplexity 1\n'
  exit 0
fi
count_file="$CALL_DIR/perplexity"
count=0
[ ! -f "$count_file" ] || count="$(cat "$count_file")"
printf '%s\n' "$((count + 1))" >"$count_file"

kv=''
chunks=''
ctx=''
verbose=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    -ctk) kv="$2"; shift 2 ;;
    --chunks) chunks="$2"; shift 2 ;;
    -c) ctx="$2"; shift 2 ;;
    -v) verbose=1; shift ;;
    *) shift ;;
  esac
done

if [ "$chunks" = 8 ] && [ "${FAKE_CASE:-}" = short_nan ] && [ "$kv" = turbo4 ]; then
  printf 'chunk 1: PPL = NaN\nFinal estimate: PPL = NaN\n'
  exit 0
fi

if [ "$ctx" = 16384 ] || [ "$verbose" -eq 1 ]; then
  k_type="$kv"
  v_type="$kv"
  if [ "${FAKE_CASE:-}" = bad_capacity_type ] && [ "$kv" = turbo4 ]; then
    k_type=q8_0
  fi
  if [ "$kv" = q8_0 ]; then
    k_mib=100.0
    v_mib=100.0
  else
    k_mib=50.0
    v_mib=50.0
    if [ "${FAKE_CASE:-}" = bad_capacity_ratio ]; then
      k_mib=55.0
      v_mib=55.0
    fi
  fi
  printf 'llama_kv_cache: size = 2.00 MiB (dry run), K (f16):  1.00 MiB, V (f16):  1.00 MiB\n'
  printf 'llama_kv_cache: size = 16384, K (%s):  %s MiB, V (%s):  %s MiB\n' "$k_type" "$k_mib" "$v_type" "$v_mib"
  printf 'Final estimate: PPL = 7.0\n'
  exit 0
fi

case "$kv" in
  f16) ppl=7.00 ;;
  q8_0) ppl=7.01 ;;
  q4_0) ppl=7.50 ;;
  turbo4) ppl=7.40 ;;
  *) exit 64 ;;
esac
if [ "$chunks" = 564 ] && [ "${FAKE_CASE:-}" = full_order ] && [ "$kv" = turbo4 ]; then
  ppl=7.60
fi
for ((i = 1; i <= chunks; i++)); do
  chunk_ppl="$ppl"
  if [ "$chunks" = 8 ] && [ "${FAKE_CASE:-}" = short_drift ] && [ "$kv" = turbo4 ] && [ "$i" -eq 4 ]; then
    chunk_ppl=200.0
  fi
  printf '[%s]%s,' "$i" "$chunk_ppl"
done
printf '\nFinal estimate: PPL = %s\n' "$ppl"
EOF

  cat >"$BUILD_BIN/llama-bench" <<'EOF'
#!/bin/bash
if [ "${1:-}" = --version ]; then
  printf 'fake-bench 1\n'
  exit 0
fi
count_file="$CALL_DIR/bench"
count=0
[ ! -f "$count_file" ] || count="$(cat "$count_file")"
printf '%s\n' "$((count + 1))" >"$count_file"
kv=''
while [ "$#" -gt 0 ]; do
  case "$1" in
    -ctk) kv="$2"; shift 2 ;;
    *) shift ;;
  esac
done
case "$kv" in
  q4_0) tps=100.0 ;;
  turbo4)
    if [ "${FAKE_CASE:-}" = decode_regression ]; then tps=95.0; else tps=101.0; fi
    ;;
  *) exit 64 ;;
esac
printf '{"avg_ts":%s}\n' "$tps"
EOF

  chmod +x "$FAKE_BIN/ldd" "$FAKE_BIN/fuser" "$FAKE_BIN/sycl-ls" \
    "$BUILD_BIN/test-sycl-turbo-correctness" "$BUILD_BIN/llama-perplexity" "$BUILD_BIN/llama-bench"
  CORRECTNESS_BIN="$BUILD_BIN/test-sycl-turbo-correctness"
  export FIXTURE FAKE_BIN BUILD_BIN CALL_DIR MODEL_DIR MISTRAL_MODEL LLAMA31_MODEL WIKI FAKE_LIB CORRECTNESS_BIN
}

run_driver() {
  local case_name="$1" out_dir="$2"
  shift 2
  env \
    PATH="$FAKE_BIN:$PATH" \
    FAKE_CASE="$case_name" \
    CALL_DIR="$CALL_DIR" \
    FAKE_LIB="$FAKE_LIB" \
    P45A_RUN_TIMEOUT=10 \
    BUILD_BIN="$BUILD_BIN" \
    CORRECTNESS_BIN="$CORRECTNESS_BIN" \
    MISTRAL_MODEL="$MISTRAL_MODEL" \
    LLAMA31_MODEL="$LLAMA31_MODEL" \
    WIKI="$WIKI" \
    OUT_DIR="$out_dir" \
    SOURCE_SHA="$SOURCE_SHA" \
    "$@" \
    "$DRIVER" >"$FIXTURE/driver.stdout" 2>"$FIXTURE/driver.stderr"
}

assert_status() {
  local out_dir="$1" expected_code="$2" expected_status="$3" actual_code="$4"
  [ "$actual_code" -eq "$expected_code" ] || fail "expected exit $expected_code, got $actual_code for $expected_status"
  jq -e --arg status "$expected_status" '.status == $status' "$out_dir/verdict.json" >/dev/null \
    || fail "expected status $expected_status"
  [ "$(cat "$out_dir/EXIT")" -eq "$expected_code" ] || fail "EXIT does not match $expected_code"
}

make_fixture missing-input
out="$FIXTURE/out"
set +e
env PATH="$FAKE_BIN:$PATH" BUILD_BIN= OUT_DIR="$out" SOURCE_SHA="$SOURCE_SHA" "$DRIVER" >/dev/null 2>&1
rc=$?
set -e
assert_status "$out" 2 ERROR "$rc"
[ ! -e "$CALL_DIR/fuser" ] || fail "missing-input case touched GPU occupancy"
[ ! -e "$CALL_DIR/harness" ] || fail "missing-input case invoked harness"
pass "missing required input returns ERROR=2 before GPU work"

make_fixture occupied
out="$FIXTURE/out"
set +e
run_driver occupied "$out"
rc=$?
set -e
assert_status "$out" 3 PARK "$rc"
[ ! -e "$CALL_DIR/bench" ] || fail "occupied case invoked a timing leg"
pass "occupied render node returns PARK=3 before timing"

make_fixture harness-fail
out="$FIXTURE/out"
set +e
run_driver harness_fail "$out"
rc=$?
set -e
assert_status "$out" 1 KILL "$rc"
[ ! -e "$CALL_DIR/perplexity" ] || fail "harness failure invoked perplexity"
jq -e '.gates.provenance and (.gates.harness | not)' "$out/verdict.json" >/dev/null \
  || fail "harness failure gate state is wrong"
pass "nonzero harness GATE-FAIL returns KILL=1"

make_fixture short-nan
out="$FIXTURE/out"
set +e
run_driver short_nan "$out"
rc=$?
set -e
assert_status "$out" 1 KILL "$rc"
[ "$(cat "$CALL_DIR/perplexity")" -lt 8 ] || fail "short NaN reached full PPL matrix"
jq -e '.gates.harness and (.gates.short_ppl | not)' "$out/verdict.json" >/dev/null \
  || fail "short NaN gate state is wrong"
pass "short NaN returns KILL before full PPL"

make_fixture short-drift
out="$FIXTURE/out"
set +e
run_driver short_drift "$out"
rc=$?
set -e
assert_status "$out" 1 KILL "$rc"
jq -e '.gates.harness and (.gates.short_ppl | not)' "$out/verdict.json" >/dev/null \
  || fail "intermediate drift gate state is wrong"
grep -Fq 'short PPL exponential drift' "$out/verdict.json" \
  || fail "intermediate drift did not produce the decisive reason"
pass "finite final PPL with intermediate exponential drift returns KILL"

make_fixture full-order
out="$FIXTURE/out"
set +e
run_driver full_order "$out"
rc=$?
set -e
assert_status "$out" 1 KILL "$rc"
jq -e '.gates.short_ppl and (.gates.full_ppl | not)' "$out/verdict.json" >/dev/null \
  || fail "full ordering gate state is wrong"
pass "turbo4 not lower than q4_0 returns KILL"

make_fixture capacity-type
out="$FIXTURE/out"
set +e
run_driver bad_capacity_type "$out"
rc=$?
set -e
assert_status "$out" 1 KILL "$rc"
jq -e '.gates.full_ppl and (.gates.capacity | not)' "$out/verdict.json" >/dev/null \
  || fail "bad capacity type gate state is wrong"

make_fixture capacity-ratio
out_ratio="$FIXTURE/out"
set +e
run_driver bad_capacity_ratio "$out_ratio"
rc=$?
set -e
assert_status "$out_ratio" 1 KILL "$rc"
pass "effective types and sub-1.90 capacity ratio return KILL"

make_fixture decode-regression
out="$FIXTURE/out"
set +e
run_driver decode_regression "$out"
rc=$?
set -e
assert_status "$out" 1 KILL "$rc"
jq -e '.gates.capacity and (.gates.decode | not)' "$out/verdict.json" >/dev/null \
  || fail "decode regression gate state is wrong"
pass "decode lower confidence bound below -2 returns KILL"

make_fixture success
out="$FIXTURE/out"
set +e
run_driver success "$out" DEVICE_SELECTOR=test-selector
rc=$?
set -e
assert_status "$out" 0 GO "$rc"
jq -e '
  ([.gates[]] | all) and
  .models.mistral.status == "PASS" and
  .models.llama31.status == "PASS" and
  .models.mistral.ppl.turbo4 < .models.mistral.ppl.q4_0 and
  .models.llama31.capacity.q8_over_turbo4 >= 1.90 and
  .models.mistral.decode.lower95_pct >= -2.0
' "$out/verdict.json" >/dev/null || fail "GO verdict contract is incomplete"
jq -e '.models.mistral.bytes == 8' "$out/manifest.json" >/dev/null \
  || fail "manifest byte count did not follow the model symlink"
jq -e --arg expected_host "$(hostname)" --arg expected_device "test-selector" '
  .host == $expected_host and
  .device_selector == $expected_device
' "$out/manifest.json" >/dev/null \
  || fail "manifest host/device_selector are not runtime-captured (got host=\(.host), device=\(.device_selector))"

required=(
  manifest.json ppl.json capacity.json bench.json verdict.json EXIT harness.log commands.txt
  version-llama-perplexity.txt version-llama-bench.txt version-correctness.txt
  ldd-llama-perplexity.txt ldd-llama-bench.txt ldd-correctness.txt
  oneapi-env.txt device-list.txt sha256.txt
)
for artifact in "${required[@]}"; do
  [ -s "$out/$artifact" ] || fail "required GO artifact is missing or empty: $artifact"
done
for model in mistral llama31; do
  for kv in f16 q8_0 q4_0 turbo4; do
    [ -s "$out/short-${model}-${kv}.log" ] || fail "missing short log for $model/$kv"
    [ -s "$out/full-${model}-${kv}.log" ] || fail "missing full log for $model/$kv"
  done
  for kv in q8_0 turbo4; do
    [ -s "$out/capacity-${model}-${kv}.log" ] || fail "missing capacity log for $model/$kv"
  done
done
grep -Fq 'ONEAPI_DEVICE_SELECTOR=test-selector' "$out/commands.txt" \
  || fail "commands.txt did not preserve shell-escaped environment"
grep -Fq '\ ' "$out/commands.txt" \
  || fail "commands.txt does not demonstrate shell escaping"
pass "complete fixture returns GO=0 with all durable artifacts"

[ "$PASS_COUNT" -eq 9 ] || fail "expected 9 contract cases, observed $PASS_COUNT"
printf '== summary: %s contract cases passed ==\n' "$PASS_COUNT"
