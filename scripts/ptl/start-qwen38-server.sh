#!/usr/bin/env bash
# Qwen3.8-Flash-Next Q2_0 + MTP on Intel Panther Lake (Arc B390 iGPU), llama.cpp SYCL.
# Verified setup: 30.1 tok/s decode over Agents4PLC (strict pass 66.7 %), prefill ~225 tok/s at 34k context.
# No root needed.
#
#   MODEL_DIR=/path/to/models HOST=0.0.0.0 PORT=8601 scripts/ptl/start-qwen38-server.sh
#
# MODEL_DIR must contain:
#   Q2_0_hcq8/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf (+ -00002-of-00002.gguf), see README step 3
#   mtp/mtp-Qwen3.8-Flash-Next-Q4_0.gguf
# NEW=0 runs the 7 Nov setup instead (original Q2_0 files in Q2_0/, 27.7 tok/s).
# Stop the server with: kill <pid>. Wait >= 30 s before starting it again.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODEL_DIR=${MODEL_DIR:-$HOME/models}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8601}
CTX=${CTX:-65536}
ALIAS=${ALIAS:-qwen38-flash-next-q2_0-mtp}
CPUS=${CPUS:-0-3}          # P-cores on PTL; the host thread must not land on the slow E-cores
ONEAPI=${ONEAPI:-/opt/intel/oneapi/setvars.sh}
BIN=$ROOT/build/bin/llama-server
NEW=${NEW:-1}
MSUB=Q2_0_hcq8; [ "$NEW" = 1 ] || MSUB=Q2_0
MODEL=$MODEL_DIR/$MSUB/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
MTP=$MODEL_DIR/mtp/mtp-Qwen3.8-Flash-Next-Q4_0.gguf

for f in "$BIN" "$MODEL" "$MTP"; do
    [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done
if pgrep -x llama-server > /dev/null; then
    echo "llama-server is already running (only one GPU job at a time):"
    pgrep -xa llama-server | cut -c1-120
    exit 1
fi

set +u
# shellcheck disable=SC1090
source "$ONEAPI" > /dev/null 2>&1
set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:0 UR_LOADER_USE_LEVEL_ZERO_V2=0 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
# GGML_SYCL_Q2_0_FAST=1       faster Q2_0 MoE expert kernel
# GGML_SYCL_MMID_SMALL_BATCH=8 MTP verify (up to 8 tokens) uses the fused one-token expert path per token
# LLAMA_DIO_BOUNCE=1          with -lm dio: keep direct IO for the model load (no page cache, no memory pressure)
export GGML_SYCL_Q2_0_FAST=1 GGML_SYCL_MMID_SMALL_BATCH=8 LLAMA_DIO_BOUNCE=1
if [ "$NEW" = 1 ]; then
    # GGML_SYCL_BF16_FAST=1          BF16 mat-vec for the remaining BF16 tensors (1-8 tokens)
    # GGML_SYCL_TOP_K_MASK=1         fused sparse-attention mask
    # GGML_SYCL_XMX_Q2_0=1, GGML_SYCL_DYNAMIC_PRECISION=BF16   prefill: Q2_0 experts on the XMX units (bf16, f32 acc.)
    # GGML_SYCL_DNNL_FPMATH=bf16     prefill: dense oneDNN GEMMs on the XMX units
    # GGML_SYCL_SPARSE_FA=1, GGML_SYCL_SPARSE_FA_ROWS=8, GGML_SYCL_FA_SKIP_MASKED=1
    #                                decode: attention only over the cells the indexer picked, also for MTP verify
    export GGML_SYCL_BF16_FAST=1 GGML_SYCL_TOP_K_MASK=1
    export GGML_SYCL_XMX_Q2_0=1 GGML_SYCL_DYNAMIC_PRECISION=BF16 GGML_SYCL_DNNL_FPMATH=bf16
    export GGML_SYCL_SPARSE_FA=1 GGML_SYCL_SPARSE_FA_ROWS=8 GGML_SYCL_FA_SKIP_MASKED=1
fi
# SYCL graphs stay off (GGML_SYCL_ENABLE_GRAPH unset): with MTP no graph is replayed and the cache grows per request.
# -cram 0: no host prompt cache (it grew 120-230 MB per request); the slot still reuses the prefix of the last request.

# The model is read with direct IO, so cached copies of the model files only take memory; with little free memory
# at launch the GPU buffer gets fragmented and decode runs up to 3x slower. Evict them first (no root needed).
python3 -c 'import os,sys
for f in sys.argv[1:]:
    fd=os.open(f,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)' "$(dirname "$MODEL")"/*.gguf "$MTP"

LOG=${LOG:-$ROOT/qwen38-server-$(date +%Y%m%d-%H%M).log}
setsid nohup taskset -c "$CPUS" "$BIN" -m "$MODEL" -lm dio -lzm on -ngl 99 -fa on \
    --jinja --reasoning off --no-reasoning-preserve \
    -c "$CTX" -np 1 -b 4096 -ub 2048 \
    -md "$MTP" -ngld 99 --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-p-min 0.5 \
    -cram 0 --alias "$ALIAS" --host "$HOST" --port "$PORT" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "pid $PID, log $LOG"
for _ in $(seq 1 100); do
    grep -q "listening on" "$LOG" && break
    kill -0 $PID 2> /dev/null || { echo "server exited:"; tail -20 "$LOG"; exit 1; }
    sleep 3
done
grep -E "listening on" "$LOG" || { echo "not ready after 300 s, see $LOG"; exit 1; }
