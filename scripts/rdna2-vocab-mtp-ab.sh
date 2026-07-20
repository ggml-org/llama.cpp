#!/usr/bin/env bash
# Balanced mirrored vs vocabulary-parallel MTP server A/B on four RDNA2 GPUs.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-$ROOT/build}
ROCM_PATH=${ROCM_PATH:-/opt/rocm/core-7.14}
MODEL=${MODEL:-$HOME/models/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf}
OUT_DIR=${OUT_DIR:-$HOME/llama-jobs/vocab-mtp-ab-$(date +%Y%m%d-%H%M%S)}
TOKENS=${TOKENS:-512}
PARALLEL=${PARALLEL:-1}
PORT=${PORT:-8081}

mkdir -p "$OUT_DIR"
[ -x "$BUILD_DIR/bin/llama-server" ] || { echo "missing llama-server: $BUILD_DIR/bin/llama-server" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "missing model: $MODEL" >&2; exit 2; }

export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-10.3.0}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
export GGML_CUDA_ALLREDUCE=nccl

cat >"$OUT_DIR/client.sh" <<'CLIENT'
#!/usr/bin/env bash
set -euo pipefail
curl -fsS --max-time 120 "$LLAMA_SERVER_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed technical explanation of lock-free queues, including correctness and memory ordering.\"}],\"max_tokens\":${TOKENS},\"temperature\":0.6,\"seed\":1234}" \
  >"$LLAMA_JOB_DIR/response.json"
CLIENT
chmod +x "$OUT_DIR/client.sh"
export TOKENS

run_mode() {
    local mode=$1 label=$2 run_dir="$OUT_DIR/$label-$mode"
    unset GGML_TP_VOCAB_OUTPUT GGML_TP_SHARDED_OUTPUT
    if [ "$mode" = vocab ]; then export GGML_TP_VOCAB_OUTPUT=1; fi
    echo "=== label=$label mode=$mode ===" | tee -a "$OUT_DIR/summary.log"
    LLAMA_JOB_DIR="$run_dir" "$ROOT/scripts/rdna2-server-job.sh" --port "$PORT" --ready-timeout 180 \
      --server "$BUILD_DIR/bin/llama-server" -m "$MODEL" -ngl all --split-mode tensor --tensor-split 1,1,1,1 \
      --flash-attn on --ctx-size 8192 --cache-type-k f16 --cache-type-v f16 --batch-size 2048 --ubatch-size 256 \
      --parallel "$PARALLEL" --host 127.0.0.1 --port "$PORT" --temp 0.6 --spec-type draft-mtp --spec-draft-ngl all \
      --spec-draft-n-max 3 --spec-draft-type-k f16 --spec-draft-type-v f16 --spec-draft-p-min 0.0 \
      --spec-draft-p-split 0.10 --spec-draft-backend-sampling --jinja \
      --client "$OUT_DIR/client.sh" | tee -a "$OUT_DIR/summary.log"
    grep -E "eval time|draft acceptance" "$run_dir/server.log" | tee -a "$OUT_DIR/summary.log"
}

# A/B, B/A, A/B controls load order and thermal drift.
run_mode off   ab1
run_mode vocab ab1
run_mode vocab ba2
run_mode off   ba2
run_mode off   ab3
run_mode vocab ab3

echo "MTP A/B artifacts: $OUT_DIR"