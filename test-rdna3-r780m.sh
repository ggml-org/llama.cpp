#!/usr/bin/env bash
# test-rdna3-r780m.sh — test-replay protocol for the RDNA3 iGPU matmul tune.
# Reference-branch tool, NOT for merge. See TESTING-RDNA3-780M.md.
#
# Usage:
#   ./test-rdna3-r780m.sh <path-to-model.gguf> [tune-branch] [base-commit]
# Defaults:
#   tune-branch = ggml-vk-rdna3-igpu-ltile
#   base-commit = parent of tune-branch tip   (i.e. upstream state under test)
#
# Steps: fresh builds of base + tune -> greedy byte-identical text check ->
# llama-bench pp/tg for both -> logs under results/<timestamp>/.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODEL="${1:?usage: $0 <model.gguf> [tune-branch] [base-commit]}"
TUNE_BRANCH="${2:-ggml-vk-rdna3-igpu-ltile}"
BASE_COMMIT="${3:-${TUNE_BRANCH}~1}"
TS="$(date +%Y%m%d-%H%M%S)"
RESULTS="results/r780m-$TS"
mkdir -p "$RESULTS"

NGL=99
CTX=4096
ARGS="-m ${MODEL} -ngl ${NGL} -c ${CTX} --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 -temp 0 -n 60"
PROMPT="Continue in exactly two short sentences: The morning mist always hung lowest over the village of Oakhaven. Pip, the small green dragon, lived alone in his cave at the top of the peak."

log() { echo "== $*"; }

log "fetching refs"
git fetch

log "build BASE = ${BASE_COMMIT}"
git checkout -q -B base-replay "$BASE_COMMIT"
cmake -B build-base -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF
cmake --build build-base --config Release -j

log "build TUNE = ${TUNE_BRANCH}"
git checkout -q "$TUNE_BRANCH"
cmake -B build-tune -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF
cmake --build build-tune --config Release -j
ls -l build-base/bin/llama-cli build-tune/bin/llama-cli

log "correctness: greedy (-temp 0) must match"
build-base/bin/llama-cli $ARGS -p "$PROMPT" 2>&1 | tee "$RESULTS/out-base.log" >/dev/null
build-tune/bin/llama-cli $ARGS -p "$PROMPT" 2>&1 | tee "$RESULTS/out-tune.log" >/dev/null
if diff <(sed -n '/\[Prompt/,$p' "$RESULTS/out-base.log") \
        <(sed -n '/\[Prompt/,$p' "$RESULTS/out-tune.log") > "$RESULTS/text-diff.txt"; then
  log "TEXT: byte-identical incl. timing"
elif grep -qE '^-?\[? ?Prompt|t/s' "$RESULTS/text-diff.txt"; then
  log "TEXT: identical except timing lines (OK)"
else
  log "TEXT MISMATCH — see $RESULTS/text-diff.txt"
fi

log "speed: llama-bench -p 512 -n 64 -r 3"
BENCH_ARGS=(--fa on --cache-type-k q8_0 --cache-type-v q8_0 -p 512 -n 64 -r 3 -m "$MODEL" -ngl "$NGL")
./build-base/bin/llama-bench "${BENCH_ARGS[@]}" 2>&1 | tee "$RESULTS/bench-base.log"
./build-tune/bin/llama-bench "${BENCH_ARGS[@]}" 2>&1 | tee "$RESULTS/bench-tune.log"

log "context"
{
  echo "base-tree : $(git rev-parse base-replay) ($(git log -1 --format=%s base-replay))"
  echo "tune-tree : $(git rev-parse "$TUNE_BRANCH") ($(git log -1 --format=%s "$TUNE_BRANCH"))"
  command -v vulkaninfo >/dev/null && vulkaninfo --summary || echo "vulkaninfo: not available"
  grep -h 'build  :' "$RESULTS"/out-*.log
} | tee "$RESULTS/context.txt"

log "done -> $RESULTS/"
echo
echo "PASS criteria (reference): pp512 base ~249 t/s, tune ~401 t/s (+60%), tg64 flat, text identical."
echo "Upload: git checkout -b results-$TS && git add $RESULTS && git commit -m 'test results $TS: R780M' && git push -f <fork-url> HEAD"
