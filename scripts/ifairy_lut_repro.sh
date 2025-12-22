#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BIN_TEST="${BIN_TEST:-${ROOT}/build-rel/bin/test-ifairy}"
BIN_CLI="${BIN_CLI:-${ROOT}/build-rel/bin/llama-cli}"
BIN_BENCH="${BIN_BENCH:-${ROOT}/build-rel/bin/llama-bench}"
MODEL="${MODEL:-${ROOT}/models/Fairy-plus-minus-i-700M/ifairy.gguf}"

THREADS="${THREADS:-4}"
N_PROMPT="${N_PROMPT:-128}"
N_GEN="${N_GEN:-256}"
REPS="${REPS:-1}"
NO_WARMUP="${NO_WARMUP:-1}"
DEVICE="${DEVICE:-none}"
PROMPT="${PROMPT:-I believe life is}"
KERNEL="${KERNEL:-auto}"
LAYOUT="${LAYOUT:-}"

if [[ ! -x "${BIN_TEST}" ]]; then
  echo "missing BIN_TEST=${BIN_TEST} (build first)" >&2
  exit 1
fi
if [[ ! -x "${BIN_CLI}" ]]; then
  echo "missing BIN_CLI=${BIN_CLI} (build first)" >&2
  exit 1
fi
if [[ ! -x "${BIN_BENCH}" ]]; then
  echo "missing BIN_BENCH=${BIN_BENCH} (build first)" >&2
  exit 1
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "missing MODEL=${MODEL} (set MODEL=/path/to/ifairy.gguf)" >&2
  exit 1
fi

COMMON_ENV=(GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0)
if [[ -n "${KERNEL}" ]]; then
  COMMON_ENV+=(GGML_IFAIRY_LUT_KERNEL="${KERNEL}")
fi
if [[ -n "${LAYOUT}" ]]; then
  COMMON_ENV+=(GGML_IFAIRY_LUT_LAYOUT="${LAYOUT}")
fi

echo "== test-ifairy =="
"${BIN_TEST}"
echo
echo "== test-ifairy (strict) =="
GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_VALIDATE_STRICT=1 "${BIN_TEST}"

echo
echo "== llama-cli sanity =="
env "${COMMON_ENV[@]}" "${BIN_CLI}" -m "${MODEL}" --gpu-layers 0 -t "${THREADS}" \
  -p "${PROMPT}" -n 64 -no-cnv

echo
echo "== llama-bench =="
BENCH_LOG="${BENCH_LOG:-${TMPDIR:-/tmp}/ifairy_bench.$(date +%Y%m%dT%H%M%S).jsonl}"
COMMON_ARGS=( -m "${MODEL}" --threads "${THREADS}" --n-prompt "${N_PROMPT}" --n-gen "${N_GEN}"
  -ngl 0 --device "${DEVICE}" --repetitions "${REPS}" -o jsonl )
if [[ "${NO_WARMUP}" == "1" ]]; then
  COMMON_ARGS+=( --no-warmup )
fi
env "${COMMON_ENV[@]}" "${BIN_BENCH}" "${COMMON_ARGS[@]}" | tee "${BENCH_LOG}"
echo "bench log: ${BENCH_LOG}"
