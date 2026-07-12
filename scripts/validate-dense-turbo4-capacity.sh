#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEVICE_SELECTOR="${DEVICE_SELECTOR:-level_zero:0}"
RUN_TIMEOUT="${P45A_RUN_TIMEOUT:-7200}"

GATE_PROVENANCE=false
GATE_HARNESS=false
GATE_SHORT_PPL=false
GATE_FULL_PPL=false
GATE_CAPACITY=false
GATE_DECODE=false
PPL_JSON='{}'
CAPACITY_JSON='{}'
BENCH_JSON='{}'

PRELIMINARY_JSON='{
  "source": "build-p45a on mergerfs with /usr/bin/cc as C compiler",
  "binding": false,
  "llama31": {
    "f16": 7.5431,
    "q8_0": 7.5454,
    "q4_0": 7.7704,
    "turbo4": 7.6602
  },
  "excluded": "Qwen3 auto-asymmetric run changed effective turbo4 K to q8_0"
}'


write_verdict() {
  local status="$1" reason="$2"
  local mistral llama31
  mistral="$(jq -n \
    --arg status "$(if [ "$status" = GO ]; then printf PASS; else printf NOT_RUN; fi)" \
    --argjson ppl "$(jq -c '.mistral // {}' <<<"$PPL_JSON")" \
    --argjson capacity "$(jq -c '.mistral // {}' <<<"$CAPACITY_JSON")" \
    --argjson decode "$(jq -c '.mistral // {}' <<<"$BENCH_JSON")" \
    '{status:$status,ppl:$ppl,capacity:$capacity,decode:$decode}')"
  llama31="$(jq -n \
    --arg status "$(if [ "$status" = GO ]; then printf PASS; else printf NOT_RUN; fi)" \
    --argjson ppl "$(jq -c '.llama31 // {}' <<<"$PPL_JSON")" \
    --argjson capacity "$(jq -c '.llama31 // {}' <<<"$CAPACITY_JSON")" \
    --argjson decode "$(jq -c '.llama31 // {}' <<<"$BENCH_JSON")" \
    '{status:$status,ppl:$ppl,capacity:$capacity,decode:$decode}')"

  jq -n \
    --arg source_sha "${SOURCE_SHA:-}" \
    --arg status "$status" \
    --arg reason "$reason" \
    --argjson provenance "$GATE_PROVENANCE" \
    --argjson harness "$GATE_HARNESS" \
    --argjson short_ppl "$GATE_SHORT_PPL" \
    --argjson full_ppl "$GATE_FULL_PPL" \
    --argjson capacity "$GATE_CAPACITY" \
    --argjson decode "$GATE_DECODE" \
    --argjson mistral "$mistral" \
    --argjson llama31 "$llama31" \
    --argjson preliminary "$PRELIMINARY_JSON" \
    '{
      schema_version:1,
      source_sha:$source_sha,
      status:$status,
      reason:$reason,
      gates:{
        provenance:$provenance,
        harness:$harness,
        short_ppl:$short_ppl,
        full_ppl:$full_ppl,
        capacity:$capacity,
        decode:$decode
      },
      models:{mistral:$mistral,llama31:$llama31},
      preliminary_nonbinding:$preliminary
    }' >"$OUT_DIR/verdict.json"
}

finish() {
  local status="$1" reason="$2" code="$3"
  write_verdict "$status" "$reason"
  printf '%s\n' "$code" >"$OUT_DIR/EXIT"
  exit "$code"
}

fail_config() {
  local message="$1"
  if [ -n "${OUT_DIR:-}" ] && mkdir -p "$OUT_DIR" 2>/dev/null; then
    finish ERROR "$message" 2
  fi
  printf 'ERROR: %s\n' "$message" >&2
  exit 2
}

park() {
  finish PARK "$1" 3
}

kill() {
  finish KILL "$1" 1
}

go() {
  finish GO "all binding P4.5a gates passed" 0
}

append_command() {
  local first=1 arg
  for arg in "$@"; do
    if [ "$first" -eq 0 ]; then printf ' ' >>"$OUT_DIR/commands.txt"; fi
    printf '%q' "$arg" >>"$OUT_DIR/commands.txt"
    first=0
  done
  printf '\n' >>"$OUT_DIR/commands.txt"
}

run_logged() {
  local stage="$1" log="$2"
  shift 2
  append_command env "ONEAPI_DEVICE_SELECTOR=$DEVICE_SELECTOR" timeout "$RUN_TIMEOUT" "$@"
  env ONEAPI_DEVICE_SELECTOR="$DEVICE_SELECTOR" timeout "$RUN_TIMEOUT" "$@" >"$log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    return "$rc"
  fi
  if grep -Eq 'DeviceLost|UR_RESULT_ERROR_DEVICE_LOST' "$log"; then
    printf 'invalid output in stage %s\n' "$stage" >>"$log"
    return 65
  fi
  return 0
}

require_empty_gpu() {
  local stage="$1"
  local record="$OUT_DIR/${stage}.fuser"
  append_command fuser -v /dev/dri/renderD128
  fuser -v /dev/dri/renderD128 >"$record" 2>&1
  local rc=$?
  if [ "$rc" -eq 1 ] && [ ! -s "$record" ]; then
    return 0
  fi
  return 1
}

extract_ppl() {
  local log="$1" matches value
  matches="$(grep -Eo 'Final estimate: PPL = [0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?' "$log" || true)"
  if [ "$(printf '%s\n' "$matches" | grep -c .)" -ne 1 ]; then
    return 1
  fi
  value="${matches##*= }"
  jq -en --arg value "$value" '$value | tonumber | select(isfinite)' 2>/dev/null
}

extract_chunk_ppl() {
  local log="$1" expected="$2"
  grep -Eo '\[[0-9]+\][+-]?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?' "$log" \
    | jq -eRsc --argjson expected "$expected" '
        split("\n")
        | map(select(length > 0)
            | capture("^\\[(?<index>[0-9]+)\\](?<value>[+-]?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)$")
            | {index:(.index | tonumber), value:(.value | tonumber)})
        | . as $rows
        | select(($rows | length) == $expected)
        | select(($rows | map(.index)) == [range(1; $expected + 1)])
        | $rows
        | map(.value)
      '
}

extract_kv() {
  local log="$1" line k_type v_type k_mib v_mib
  line="$(grep -E 'llama_kv_cache: size = .* K \([^)]*\): .* MiB, V \([^)]*\): .* MiB' "$log" | tail -n 1 || true)"
  [ -n "$line" ] || return 1
  k_type="$(sed -E 's/.* K \(([^)]*)\):.*/\1/' <<<"$line")"
  v_type="$(sed -E 's/.* V \(([^)]*)\):.*/\1/' <<<"$line")"
  k_mib="$(sed -E 's/.* K \([^)]*\):[[:space:]]+([0-9.]+) MiB,.*/\1/' <<<"$line")"
  v_mib="$(sed -E 's/.* V \([^)]*\):[[:space:]]+([0-9.]+) MiB.*/\1/' <<<"$line")"
  jq -en --arg k_type "$k_type" --arg v_type "$v_type" --arg k "$k_mib" --arg v "$v_mib" '
    ($k | tonumber) as $km | ($v | tonumber) as $vm |
    {k_type:$k_type,v_type:$v_type,k_mib:$km,v_mib:$vm,total_mib:($km+$vm)}'
}

extract_tps() {
  local jsonl="$1"
  jq -e -s '
    select(length == 1) | .[0].avg_ts |
    select(type == "number" and isfinite and . > 0)
  ' "$jsonl"
}

require_regular_nonempty() {
  local name="$1" path="$2"
  [ -f "$path" ] && [ -s "$path" ] || fail_config "$name must be a regular nonempty file: $path"
}

for required in BUILD_BIN CORRECTNESS_BIN MISTRAL_MODEL LLAMA31_MODEL WIKI OUT_DIR SOURCE_SHA; do
  [ -n "${!required:-}" ] || fail_config "missing required environment variable: $required"
done

if [ -e "$OUT_DIR" ] && [ ! -d "$OUT_DIR" ]; then
  fail_config "OUT_DIR exists and is not a directory: $OUT_DIR"
fi
mkdir -p "$OUT_DIR" 2>/dev/null || fail_config "cannot create OUT_DIR: $OUT_DIR"
if [ -n "$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  fail_config "OUT_DIR must be new or empty: $OUT_DIR"
fi
: >"$OUT_DIR/commands.txt"

[ -x "$BUILD_BIN/llama-perplexity" ] || fail_config "llama-perplexity is not executable: $BUILD_BIN/llama-perplexity"
[ -x "$BUILD_BIN/llama-bench" ] || fail_config "llama-bench is not executable: $BUILD_BIN/llama-bench"
[ -x "$CORRECTNESS_BIN" ] || fail_config "correctness binary is not executable: $CORRECTNESS_BIN"
require_regular_nonempty MISTRAL_MODEL "$MISTRAL_MODEL"
require_regular_nonempty LLAMA31_MODEL "$LLAMA31_MODEL"
require_regular_nonempty WIKI "$WIKI"

for command_name in git sha256sum ldd fuser jq; do
  command -v "$command_name" >/dev/null 2>&1 || fail_config "required command is unavailable: $command_name"
done

[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || fail_config "SOURCE_SHA must be 40 lowercase hexadecimal characters"
cd "$REPO_ROOT" || fail_config "cannot enter repository root: $REPO_ROOT"
actual_sha="$(git rev-parse HEAD 2>/dev/null)" || fail_config "cannot resolve repository HEAD"
[ "$actual_sha" = "$SOURCE_SHA" ] || fail_config "SOURCE_SHA does not match HEAD: expected $actual_sha"
git diff --quiet || fail_config "tracked worktree changes are present"
git diff --cached --quiet || fail_config "tracked index changes are present"

ldd "$BUILD_BIN/llama-perplexity" >"$OUT_DIR/ldd-llama-perplexity.txt" 2>&1 || fail_config "ldd failed for llama-perplexity"
ldd "$BUILD_BIN/llama-bench" >"$OUT_DIR/ldd-llama-bench.txt" 2>&1 || fail_config "ldd failed for llama-bench"
ldd "$CORRECTNESS_BIN" >"$OUT_DIR/ldd-correctness.txt" 2>&1 || fail_config "ldd failed for correctness binary"
lib_sycl="$(sed -nE 's/^[[:space:]]*libggml-sycl\.so[^=]*=>[[:space:]]*([^[:space:]]+).*/\1/p' "$OUT_DIR/ldd-llama-perplexity.txt")"
[ -n "$lib_sycl" ] || fail_config "ldd did not resolve libggml-sycl.so"
lib_sycl="$(realpath "$lib_sycl" 2>/dev/null)" || fail_config "cannot resolve libggml-sycl.so path"
build_root="$(realpath "$BUILD_BIN/.." 2>/dev/null)" || fail_config "cannot resolve build root"
case "$lib_sycl" in
  "$build_root"/*) ;;
  *) fail_config "libggml-sycl.so resolves outside the commit-scoped build: $lib_sycl" ;;
esac
require_regular_nonempty libggml-sycl.so "$lib_sycl"

untracked_json="$(git ls-files --others --exclude-standard | jq -Rsc 'split("\n") | map(select(length > 0))')"

"$BUILD_BIN/llama-perplexity" --version >"$OUT_DIR/version-llama-perplexity.txt" 2>&1 || true
"$BUILD_BIN/llama-bench" --version >"$OUT_DIR/version-llama-bench.txt" 2>&1 || true
"$CORRECTNESS_BIN" --version >"$OUT_DIR/version-correctness.txt" 2>&1 || true
if ! env | grep -E '^(CMPLR_ROOT|DPCPP_ROOT|INTEL|MKLROOT|ONEAPI|SYCL|TBBROOT)=' | sort >"$OUT_DIR/oneapi-env.txt"; then
  printf 'no matching oneAPI environment variables\n' >"$OUT_DIR/oneapi-env.txt"
fi
if command -v sycl-ls >/dev/null 2>&1; then
  sycl-ls >"$OUT_DIR/device-list.txt" 2>&1 || true
else
  printf 'sycl-ls unavailable\n' >"$OUT_DIR/device-list.txt"
fi

sha256sum \
  "$BUILD_BIN/llama-perplexity" "$BUILD_BIN/llama-bench" "$CORRECTNESS_BIN" "$lib_sycl" \
  "$MISTRAL_MODEL" "$LLAMA31_MODEL" "$WIKI" >"$OUT_DIR/sha256.txt"

jq -n \
  --arg source_sha "$SOURCE_SHA" \
  --argjson untracked "$untracked_json" \
  --arg perplexity_path "$(realpath "$BUILD_BIN/llama-perplexity")" \
  --arg perplexity_sha "$(sha256sum "$BUILD_BIN/llama-perplexity" | cut -d' ' -f1)" \
  --arg bench_path "$(realpath "$BUILD_BIN/llama-bench")" \
  --arg bench_sha "$(sha256sum "$BUILD_BIN/llama-bench" | cut -d' ' -f1)" \
  --arg correctness_path "$(realpath "$CORRECTNESS_BIN")" \
  --arg correctness_sha "$(sha256sum "$CORRECTNESS_BIN" | cut -d' ' -f1)" \
  --arg sycl_path "$lib_sycl" \
  --arg sycl_sha "$(sha256sum "$lib_sycl" | cut -d' ' -f1)" \
  --arg mistral_path "$(realpath "$MISTRAL_MODEL")" \
  --argjson mistral_bytes "$(stat -Lc %s "$MISTRAL_MODEL")" \
  --arg mistral_sha "$(sha256sum "$MISTRAL_MODEL" | cut -d' ' -f1)" \
  --arg llama31_path "$(realpath "$LLAMA31_MODEL")" \
  --argjson llama31_bytes "$(stat -Lc %s "$LLAMA31_MODEL")" \
  --arg llama31_sha "$(sha256sum "$LLAMA31_MODEL" | cut -d' ' -f1)" \
  --arg corpus_path "$(realpath "$WIKI")" \
  --argjson corpus_bytes "$(stat -Lc %s "$WIKI")" \
  --arg corpus_sha "$(sha256sum "$WIKI" | cut -d' ' -f1)" \
  --arg device_selector "$DEVICE_SELECTOR" \
  --arg host "$(hostname)" \
   '{
    schema_version:1,
    source_sha:$source_sha,
    tracked_source_clean:true,
    untracked_files:$untracked,
    build_mode:"JIT",
    sycl_device_arch:"",
    device_selector:$device_selector,
    host:$host,
    artifacts:{
      llama_perplexity:{path:$perplexity_path,sha256:$perplexity_sha},
      llama_bench:{path:$bench_path,sha256:$bench_sha},
      correctness:{path:$correctness_path,sha256:$correctness_sha},
      libggml_sycl:{path:$sycl_path,sha256:$sycl_sha}
    },
    models:{
      mistral:{path:$mistral_path,bytes:$mistral_bytes,sha256:$mistral_sha},
      llama31:{path:$llama31_path,bytes:$llama31_bytes,sha256:$llama31_sha}
    },
    corpus:{path:$corpus_path,bytes:$corpus_bytes,sha256:$corpus_sha}
  }' >"$OUT_DIR/manifest.json"
GATE_PROVENANCE=true

harness_log="$OUT_DIR/harness.log"
append_command env "ONEAPI_DEVICE_SELECTOR=$DEVICE_SELECTOR" LLAMA_TEST_TURBO_FA=1 timeout "$RUN_TIMEOUT" "$CORRECTNESS_BIN"
env ONEAPI_DEVICE_SELECTOR="$DEVICE_SELECTOR" LLAMA_TEST_TURBO_FA=1 timeout "$RUN_TIMEOUT" "$CORRECTNESS_BIN" >"$harness_log" 2>&1
harness_rc=$?
[ "$harness_rc" -eq 0 ] || kill "correctness harness exited $harness_rc"
grep -q '^== summary: 0 GATE-FAIL' "$harness_log" || kill "correctness harness did not report 0 GATE-FAIL"
if grep -Eqi '(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)|DeviceLost|UR_RESULT_ERROR_DEVICE_LOST' "$harness_log"; then
  kill "correctness harness emitted invalid numeric or device-lost output"
fi
GATE_HARNESS=true

model_names=(mistral llama31)
model_paths=("$MISTRAL_MODEL" "$LLAMA31_MODEL")
kv_types=(f16 q8_0 q4_0 turbo4)

for model_index in 0 1; do
  model_name="${model_names[$model_index]}"
  model_path="${model_paths[$model_index]}"
  f16_chunks='[]'
  turbo_chunks='[]'
  for kv in "${kv_types[@]}"; do
    log="$OUT_DIR/short-${model_name}-${kv}.log"
    if ! run_logged "short-${model_name}-${kv}" "$log" \
      "$BUILD_BIN/llama-perplexity" -m "$model_path" -ngl 99 -fa on -ctk "$kv" -ctv "$kv" \
      -c 512 -b 512 -ub 512 -f "$WIKI" --chunks 8 --no-warmup --no-mmap; then
      kill "short PPL failed for $model_name/$kv"
    fi
    extract_ppl "$log" >/dev/null || kill "short PPL final estimate missing for $model_name/$kv"
    chunks="$(extract_chunk_ppl "$log" 8)" || kill "short PPL chunk series invalid for $model_name/$kv"
    if [ "$kv" = f16 ]; then f16_chunks="$chunks"; fi
    if [ "$kv" = turbo4 ]; then turbo_chunks="$chunks"; fi
  done
  jq -en --argjson f16 "$f16_chunks" --argjson turbo "$turbo_chunks" '
    [range(0; ([($f16|length),($turbo|length)]|min)) as $i |
      select($turbo[$i] > 100 and $turbo[$i] > 10 * $f16[$i])] | length == 0
  ' >/dev/null || kill "short PPL exponential drift for $model_name/turbo4"
done
GATE_SHORT_PPL=true

for model_index in 0 1; do
  model_name="${model_names[$model_index]}"
  model_path="${model_paths[$model_index]}"
  model_ppl='{}'
  f16_chunks='[]'
  turbo_chunks='[]'
  for kv in "${kv_types[@]}"; do
    log="$OUT_DIR/full-${model_name}-${kv}.log"
    if ! run_logged "full-${model_name}-${kv}" "$log" \
      "$BUILD_BIN/llama-perplexity" -m "$model_path" -ngl 99 -fa on -ctk "$kv" -ctv "$kv" \
      -c 512 -b 512 -ub 512 -f "$WIKI" --chunks 564 --no-warmup --no-mmap; then
      kill "full PPL failed for $model_name/$kv"
    fi
    ppl="$(extract_ppl "$log")" || kill "full PPL final estimate missing for $model_name/$kv"
    model_ppl="$(jq --arg kv "$kv" --argjson ppl "$ppl" '. + {($kv):$ppl}' <<<"$model_ppl")"
    chunks="$(extract_chunk_ppl "$log" 564)" || kill "full PPL chunk series invalid for $model_name/$kv"
    if [ "$kv" = f16 ]; then f16_chunks="$chunks"; fi
    if [ "$kv" = turbo4 ]; then turbo_chunks="$chunks"; fi
  done
  jq -en --argjson p "$model_ppl" '$p.q8_0 <= $p.f16 * 1.01' >/dev/null \
    || kill "q8_0 PPL exceeds f16 by more than 1% for $model_name"
  jq -en --argjson p "$model_ppl" '$p.turbo4 < $p.q4_0' >/dev/null \
    || kill "turbo4 PPL is not lower than q4_0 for $model_name"
  jq -en --argjson f16 "$f16_chunks" --argjson turbo "$turbo_chunks" '
    [range(0; ([($f16|length),($turbo|length)]|min)) as $i |
      select($turbo[$i] > 100 and $turbo[$i] > 10 * $f16[$i])] | length == 0
  ' >/dev/null || kill "full PPL exponential drift for $model_name/turbo4"
  model_ppl="$(jq '
    . as $p | . + {
      deltas:{
        vs_f16:{absolute:(.turbo4-.f16),percentage:(100*(.turbo4/.f16-1))},
        vs_q8_0:{absolute:(.turbo4-.q8_0),percentage:(100*(.turbo4/.q8_0-1))},
        vs_q4_0:{absolute:(.turbo4-.q4_0),percentage:(100*(.turbo4/.q4_0-1))}
      }
    }' <<<"$model_ppl")"
  PPL_JSON="$(jq --arg model "$model_name" --argjson value "$model_ppl" '. + {($model):$value}' <<<"$PPL_JSON")"
done
printf '%s\n' "$PPL_JSON" | jq . >"$OUT_DIR/ppl.json"
GATE_FULL_PPL=true

for model_index in 0 1; do
  model_name="${model_names[$model_index]}"
  model_path="${model_paths[$model_index]}"
  model_capacity='{}'
  for kv in q8_0 turbo4; do
    log="$OUT_DIR/capacity-${model_name}-${kv}.log"
    if ! run_logged "capacity-${model_name}-${kv}" "$log" \
      "$BUILD_BIN/llama-perplexity" -v -m "$model_path" -ngl 99 -fa on -ctk "$kv" -ctv "$kv" \
      -c 16384 -b 512 -ub 512 -f "$WIKI" --chunks 1 --no-warmup --no-mmap; then
      kill "capacity run failed for $model_name/$kv"
    fi
    allocation="$(extract_kv "$log")" || kill "KV allocation metric missing for $model_name/$kv"
    model_capacity="$(jq --arg kv "$kv" --argjson value "$allocation" '. + {($kv):$value}' <<<"$model_capacity")"
  done
  jq -en --argjson c "$model_capacity" '$c.q8_0.k_type == "q8_0" and $c.q8_0.v_type == "q8_0"' >/dev/null \
    || kill "q8_0 effective KV types are not q8_0/q8_0 for $model_name"
  jq -en --argjson c "$model_capacity" '$c.turbo4.k_type == "turbo4" and $c.turbo4.v_type == "turbo4"' >/dev/null \
    || kill "turbo4 effective KV types are not turbo4/turbo4 for $model_name"
  ratio="$(jq -en --argjson c "$model_capacity" '$c.q8_0.total_mib / $c.turbo4.total_mib')"
  jq -en --argjson ratio "$ratio" '$ratio >= 1.90' >/dev/null \
    || kill "q8_0/turbo4 capacity ratio is below 1.90 for $model_name"
  model_capacity="$(jq --argjson ratio "$ratio" '. + {q8_over_turbo4:$ratio}' <<<"$model_capacity")"
  CAPACITY_JSON="$(jq --arg model "$model_name" --argjson value "$model_capacity" '. + {($model):$value}' <<<"$CAPACITY_JSON")"
done
printf '%s\n' "$CAPACITY_JSON" | jq . >"$OUT_DIR/capacity.json"
GATE_CAPACITY=true

run_bench_leg() {
  local model_name="$1" model_path="$2" label="$3" kv="$4"
  local log="$OUT_DIR/bench-${model_name}-${label}-${kv}.jsonl"
  require_empty_gpu "bench-${model_name}-${label}-${kv}" || park "render node occupancy prevents sole-tenancy timing for $model_name/$label/$kv"
  if ! run_logged "bench-${model_name}-${label}-${kv}" "$log" \
    "$BUILD_BIN/llama-bench" -m "$model_path" -ngl 99 -fa on -ctk "$kv" -ctv "$kv" \
    -p 0 -n 128 -d 16384 -r 1 --no-warmup -o jsonl; then
    kill "decode benchmark failed for $model_name/$label/$kv"
  fi
  extract_tps "$log" || kill "decode avg_ts metric missing for $model_name/$label/$kv"
}

for model_index in 0 1; do
  model_name="${model_names[$model_index]}"
  model_path="${model_paths[$model_index]}"
  run_bench_leg "$model_name" "$model_path" warmup q4_0 >/dev/null
  run_bench_leg "$model_name" "$model_path" warmup turbo4 >/dev/null
  pairs='[]'
  for pair in 1 2 3 4 5 6 7; do
    if [ $((pair % 2)) -eq 1 ]; then order=(q4_0 turbo4); else order=(turbo4 q4_0); fi
    q4_tps=''
    turbo_tps=''
    for kv in "${order[@]}"; do
      tps="$(run_bench_leg "$model_name" "$model_path" "pair-${pair}" "$kv")" || exit $?
      if [ "$kv" = q4_0 ]; then q4_tps="$tps"; else turbo_tps="$tps"; fi
    done
    delta="$(jq -en --argjson turbo "$turbo_tps" --argjson q4 "$q4_tps" '100 * ($turbo / $q4 - 1)')"
    order_json="$(printf '%s\n' "${order[@]}" | jq -Rsc 'split("\n") | map(select(length > 0))')"
    pairs="$(jq --argjson pair "$pair" --argjson order "$order_json" --argjson q4 "$q4_tps" --argjson turbo "$turbo_tps" --argjson delta "$delta" \
      '. + [{pair:$pair,order:$order,q4_0_tps:$q4,turbo4_tps:$turbo,delta_pct:$delta}]' <<<"$pairs")"
  done
  stats="$(jq -en --argjson pairs "$pairs" '
    ($pairs | map(.delta_pct)) as $d |
    ($d | add / length) as $mean |
    (($d | map((. - $mean) * (. - $mean)) | add / (length - 1)) | sqrt) as $sd |
    {
      pairs:$pairs,
      mean_pct:$mean,
      sample_stdev_pct:$sd,
      critical_value:2.447,
      lower95_pct:($mean - 2.447 * $sd / (7 | sqrt))
    }')"
  jq -en --argjson stats "$stats" '$stats.lower95_pct >= -2.0' >/dev/null \
    || kill "decode lower 95% confidence bound is below -2.0% for $model_name"
  BENCH_JSON="$(jq --arg model "$model_name" --argjson value "$stats" '. + {($model):$value}' <<<"$BENCH_JSON")"
done
printf '%s\n' "$BENCH_JSON" | jq . >"$OUT_DIR/bench.json"
GATE_DECODE=true

go
