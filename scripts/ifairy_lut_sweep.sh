#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-${ROOT}/build-rel/bin/llama-bench}"
MODEL="${MODEL:-${ROOT}/models/Fairy-plus-minus-i-700M/ifairy.gguf}"

THREADS="${THREADS:-4}"
N_PROMPT="${N_PROMPT:-8}"
N_GEN="${N_GEN:-8}"
TEST_MODE="${TEST_MODE:-gen}"
REPS="${REPS:-1}"
NO_WARMUP="${NO_WARMUP:-1}"
DEVICE="${DEVICE:-none}"

BK_LIST="${BK_LIST:-0 2}"
BM_LIST="${BM_LIST:-64}"
FULLACC_LIST="${FULLACC_LIST:-0 1}"

if [[ ! -x "${BIN}" ]]; then
  echo "missing BIN=${BIN} (build first: cmake --build build-rel --target llama-bench ...)" >&2
  exit 1
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "missing MODEL=${MODEL}" >&2
  exit 1
fi

case "${TEST_MODE}" in
  gen)
    TEST_PROMPT=0
    TEST_GEN="${N_GEN}"
    TEST_ARGS=( --n-prompt 0 --n-gen "${N_GEN}" )
    ;;
  prompt)
    TEST_PROMPT="${N_PROMPT}"
    TEST_GEN=0
    TEST_ARGS=( --n-prompt "${N_PROMPT}" --n-gen 0 )
    ;;
  pg)
    TEST_PROMPT="${N_PROMPT}"
    TEST_GEN="${N_GEN}"
    TEST_ARGS=( -pg "${N_PROMPT},${N_GEN}" )
    ;;
  *)
    echo "invalid TEST_MODE=${TEST_MODE} (expected: gen|prompt|pg)" >&2
    exit 1
    ;;
 esac

COMMON=( -m "${MODEL}" --threads "${THREADS}" -ngl 0 --device "${DEVICE}" --repetitions "${REPS}" -o jsonl )
if [[ "${NO_WARMUP}" == "1" ]]; then
  COMMON+=( --no-warmup )
fi

TMP="${TMPDIR:-/tmp}/ifairy_lut_sweep.$(date +%s).csv"
echo "bk_blocks,bm,fullacc,tok_per_s" > "${TMP}"

extract_tok_s() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

target_prompt = int(sys.argv[1])
target_gen = int(sys.argv[2])
val = None
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        data = json.loads(line)
    except Exception:
        continue
    if data.get("n_prompt") == target_prompt and data.get("n_gen") == target_gen:
        val = data.get("avg_ts")
        break
if val is None:
    sys.exit(1)
print(val)
PY
}

run_case() {
  local bk="$1"
  local bm="$2"
  local fullacc="$3"

  local -a envs=(GGML_IFAIRY_LUT=1)
  if [[ "${bk}" != "0" ]]; then
    envs+=(GGML_IFAIRY_LUT_BK_BLOCKS="${bk}" GGML_IFAIRY_LUT_BM="${bm}" GGML_IFAIRY_LUT_FULLACC="${fullacc}")
  fi

  local tok_s
  tok_s="$(env "${envs[@]}" "${BIN}" "${COMMON[@]}" "${TEST_ARGS[@]}" 2>/dev/null | extract_tok_s "${TEST_PROMPT}" "${TEST_GEN}" || true)"
  if [[ -z "${tok_s}" ]]; then
    tok_s="nan"
  fi
  echo "${bk},${bm},${fullacc},${tok_s}" >> "${TMP}"
}

for bk in ${BK_LIST}; do
  if [[ "${bk}" == "0" ]]; then
    run_case 0 0 0
    continue
  fi
  for bm in ${BM_LIST}; do
    for fullacc in ${FULLACC_LIST}; do
      run_case "${bk}" "${bm}" "${fullacc}"
    done
  done
done

echo
echo "results: ${TMP}"
echo
python3 - <<'PY' "${TMP}"
import csv
import math
import sys

path = sys.argv[1]
rows = []
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        try:
            v = float(r["tok_per_s"])
        except Exception:
            v = float("nan")
        rows.append((v, r))
rows.sort(key=lambda x: (-math.inf if math.isnan(x[0]) else -x[0]))
for v, r in rows:
    print(f'bk={r["bk_blocks"]:>2} bm={r["bm"]:>3} fullacc={r["fullacc"]} tok/s={r["tok_per_s"]}')
PY
