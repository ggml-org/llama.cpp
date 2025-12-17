#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-${ROOT}/build-rel/bin/llama-cli}"
MODEL="${MODEL:-${ROOT}/models/Fairy-plus-minus-i-700M/ifairy.gguf}"
THREADS="${THREADS:-4}"
TOKENS="${TOKENS:-256}"
PROMPT="${PROMPT:-I believe life is}"
GPU_LAYERS="${GPU_LAYERS:-0}"

BK_LIST="${BK_LIST:-0 2}"
BM_LIST="${BM_LIST:-64}"
FULLACC_LIST="${FULLACC_LIST:-0 1}"

COMMON=( -m "${MODEL}" --gpu-layers "${GPU_LAYERS}" -t "${THREADS}" -b 1 --seed 1 -p "${PROMPT}" -n "${TOKENS}" -no-cnv )

if [[ ! -x "${BIN}" ]]; then
  echo "missing BIN=${BIN} (build first: cmake --build build-rel --config Release ...)" >&2
  exit 1
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "missing MODEL=${MODEL}" >&2
  exit 1
fi

TMP="${TMPDIR:-/tmp}/ifairy_lut_sweep.$(date +%s).csv"
echo "bk_blocks,bm,fullacc,tok_per_s" > "${TMP}"

extract_tok_s() {
  awk '
    /eval time/ && !/prompt eval time/ {
      if (match($0, /[0-9.]+[[:space:]]+tokens per second/)) {
        s = substr($0, RSTART, RLENGTH)
        gsub(/[[:space:]]+tokens per second/, "", s)
        val = s
      }
      if (match($0, /[0-9.]+[[:space:]]+tok\/s/)) {
        s = substr($0, RSTART, RLENGTH)
        gsub(/[[:space:]]+tok\/s/, "", s)
        val = s
      }
    }
    END { if (val != "") print val }
  '
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
  tok_s="$(env "${envs[@]}" "${BIN}" "${COMMON[@]}" 2>&1 | extract_tok_s || true)"
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
import csv, math, sys
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
