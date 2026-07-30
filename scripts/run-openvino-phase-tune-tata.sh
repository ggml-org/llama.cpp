#!/usr/bin/env bash
# Phase-tune benchmark (CPU vs GPU.0) for hybrid crossover study on tata-like hosts.
set -eo pipefail

MODEL="${1:-$HOME/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
REPS="${REPS:-1}"
PP="${PP:-1024}"
TG="${TG:-2048}"
if [[ -z "${GGML_OPENVINO_PHASE_TUNE_OUTPUT_DIR:-}" ]]; then
  OUT="/tmp/ov_phase_tune_tata_${PP}_${TG}"
else
  OUT="$GGML_OPENVINO_PHASE_TUNE_OUTPUT_DIR"
fi
BENCH="${BENCH:-$HOME/src/llama/build_oneapi_master_openvino_RelWithDebInfo/bin/llama-bench}"

if [[ ! -x "$BENCH" ]]; then
  echo "error: llama-bench not found: $BENCH" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "error: model not found: $MODEL" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$HOME/intel/openvino/setupvars.sh"

mkdir -p "$OUT"
export GGML_OPENVINO_CACHE_DIR="${GGML_OPENVINO_CACHE_DIR:-/tmp/ov_phase_tune_cache}"
export GGML_OPENVINO_PHASE_TUNE=1
export GGML_OPENVINO_PHASE_TUNE_DEVICES="${GGML_OPENVINO_PHASE_TUNE_DEVICES:-CPU,GPU.0}"
export GGML_OPENVINO_PHASE_TUNE_OUTPUT_DIR="$OUT"
export GGML_OPENVINO_PHASE_SPLIT=1
export GGML_OPENVINO_PREFILL_DEVICE=CPU
export GGML_OPENVINO_DECODE_DEVICE=GPU.0
export GGML_OPENVINO_STATEFUL_EXECUTION=0

echo "Running phase tune -pg ${PP},${TG} (reps=$REPS) -> $OUT"
echo "Note: tune mode runs 3x infer per step; tg=${TG} can take a long time (use --progress)."

# -p 0 -n 0: skip default pp512/tg128 tests (otherwise llama-bench runs those too).
"$BENCH" -m "$MODEL" -r "$REPS" --no-warmup --progress -p 0 -n 0 -pg "${PP},${TG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/plot-openvino-phase-tune.py" "$OUT" --device0 CPU --device1 GPU.0 --out "$OUT"
echo "Plots: $OUT/phase_tune_pp.png $OUT/phase_tune_tg.png"
