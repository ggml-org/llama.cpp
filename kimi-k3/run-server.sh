#!/usr/bin/env bash
# Run Kimi-K3 (IQ1_S-XS 540 GB by default, or MXFP4 1.5 TB) with routed experts streamed from NVMe, dense layers 0..GPU_LAYERS-1 on the GPU.
#
#   ./run-server.sh [extra llama-server args]
#
# env:
#   MODEL       first shard of the GGUF          (default: the IQ1_S-XS model, /mnt/k3/kimi-k3-iq1/...-00001-of-00034.gguf;
#               the MXFP4 model /mnt/k3/kimi-k3/MXFP4_MOE/...-00001-of-00031.gguf is the accuracy-preserving one)
#   PORT        listen port                      (default: 18090)
#   GPU_LAYERS  layers whose dense weights go to CUDA0, from layer 0 (default: IQ1 model file 11 with
#               OFFLOAD=1 / 19 with OFFLOAD=0, 8 otherwise; measured on the 12 GB card: MXFP4 10 layers left
#               ~20 MB free; IQ1 19 layers leave ~500 MB without offload, with it 12 leave ~490 MB and 16 OOM)
#   CTX         context size                     (default: 8192)
#   THREADS     CPU threads, physical cores      (default: 14)
#   UBATCH      prompt batch size (-b = -ub); every batch reads nearly all experts once (default: 2048)
#   PREDICT     next-layer expert prediction: 0 off, 1 observe (statistics only), 2 prefetch (default: 2)
#   OFFLOAD     1: prompt batches run the expert matmuls on the GPU (each layer's expert tensors are uploaded,
#               streamed from disk ahead of the copy); needs ~3 expert tensors of GPU memory, so only the IQ1
#               model fits (default: 1 for an IQ1 model file, 0 otherwise)
#   PREDICT_STATS  where the learned prediction state is kept across restarts
#               (default: ${XDG_STATE_HOME:-~/.local/state}/llama.cpp/<model file name>.predict.bin, one per model)
#
# THREADS should stay at the physical core count: the prediction worker runs on a spare hyperthread
#
# memory: the dense weights of the CPU layers (~49 GiB) are loaded into pinned host memory (-lm none), so expert
# streaming cannot evict them; the routed experts stay on disk (--lazy-mode on) and are read per token through
# the page cache with GGML_CPU_MOE_PREFETCH=1. --no-warmup: the warmup run would route all 896 experts per layer.
#
# expected decode speed on 4x Gen3 990 RAID0 + 160 GB RAM: ~0.1-0.15 tok/s, bound by ~26 GB of experts per token
# at the ~4 GB/s random-read ceiling of these DRAM-less drives

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$HERE/../build/bin/llama-server}"
MODEL="${MODEL:-/mnt/k3/kimi-k3-iq1/Kimi-K3-abliterated-IQ1_S-XS-00001-of-00034.gguf}"
PORT="${PORT:-18090}"
if [[ "$(basename "$MODEL")" == *IQ1* ]]; then
    OFFLOAD="${OFFLOAD:-1}"
    # the offload staging buffer takes ~2.3 GB more GPU memory: 11 dense layers with it (12 left < 500 MiB), 19 without
    if [[ "$OFFLOAD" == 1 ]]; then
        GPU_LAYERS="${GPU_LAYERS:-11}"
    else
        GPU_LAYERS="${GPU_LAYERS:-19}"
    fi
else
    GPU_LAYERS="${GPU_LAYERS:-8}"
    OFFLOAD="${OFFLOAD:-0}"
fi
CTX="${CTX:-8192}"
THREADS="${THREADS:-14}"
UBATCH="${UBATCH:-2048}"
PREDICT="${PREDICT:-2}"
PREDICT_STATS="${PREDICT_STATS:-${XDG_STATE_HOME:-$HOME/.local/state}/llama.cpp/$(basename "$MODEL" .gguf).predict.bin}"
N_LAYER=93

die() { echo "run-server.sh: $*" >&2; exit 1; }

# flags that defeat streaming: loading/locking/validating everything touches all 1.5 TB, op offload copies
# whole 5 GB expert tensors to the GPU, and load/lazy/placement settings are fixed by this script.
# the user's args are passed BEFORE the fixed ones, so for single-value options the script's value wins anyway;
# the check below catches the options that accumulate or only switch on, and gives a clear error for the rest.
# llama-server treats '_' and '-' alike in long options, so names are normalized before matching.
for a in "$@"; do
    n="${a%%=*}"
    [[ "$n" == --* ]] && n="${n//_/-}"
    case "$n" in
        --check-tensors)
            die "refusing '$a': it reads the whole 1.5 TB model" ;;
        --op-offload|--no-op-offload)
            die "refusing '$a': set OFFLOAD=0/1 instead (it must fit the GPU memory budget of this script)" ;;
        -lm|--load-mode|-lzm|--lazy-mode|-fit|--fit|-ot|--override-tensor|-ngl|--n-gpu-layers|--gpu-layers|-dev|--device|-ts|--tensor-split|--rpc|--no-host)
            die "refusing '$a': load mode, lazy mode and tensor placement are fixed by this script (use GPU_LAYERS)" ;;
        -m|--model|-c|--ctx-size|-t|--threads|--port|-b|--batch-size|-ub|--ubatch-size)
            die "refusing '$a': set it with the MODEL / CTX / THREADS / PORT / UBATCH environment variables" ;;
        -fa|--flash-attn|-np|--parallel|--host)
            die "refusing '$a': fixed by this script" ;;
    esac
done

# any setting can also come from the environment (LLAMA_ARG_*): start from a clean slate
for v in $(compgen -e | grep '^LLAMA_ARG_' || true); do
    unset "$v"
done
# these would silently turn the pinned dense weights into pageable memory, or offload single-token decode
unset GGML_CUDA_NO_PINNED GGML_CUDA_ENABLE_UNIFIED_MEMORY GGML_OP_OFFLOAD_MIN_BATCH

# ... and from llama.cpp config files, which are applied before the environment and the arguments
for f in /etc/llama.cpp/config.ini "${XDG_CONFIG_HOME:-$HOME/.config}/llama.cpp/config.ini"; do
    [[ -e "$f" ]] && die "llama.cpp config file $f could override this script's settings, move it away first"
done

[[ -x "$BIN" ]]   || die "llama-server not found at $BIN (build the worktree first)"
[[ -f "$MODEL" ]] || die "model not found: $MODEL"
[[ "$GPU_LAYERS" =~ ^(0|[1-9][0-9]*)$ && "$GPU_LAYERS" -le "$N_LAYER" ]] || die "GPU_LAYERS must be 0..$N_LAYER"
[[ "$UBATCH" =~ ^[1-9][0-9]*$ && "$UBATCH" -ge 32 && "$UBATCH" -le 8192 ]] || die "UBATCH must be 32..8192"
[[ "$PREDICT" =~ ^[012]$ ]] || die "PREDICT must be 0, 1 or 2"
[[ "$OFFLOAD" =~ ^[01]$ ]] || die "OFFLOAD must be 0 or 1"

# another llama process competes for the same 160 GB of RAM and page cache
if others="$(pgrep -a -x 'llama-[a-z0-9-]+' || true)"; [[ -n "$others" ]]; then
    die "another llama process is running, stop it first:
$others"
fi
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
    die "port $PORT is already in use"
fi

# dense weights of layers >= GPU_LAYERS stay on the CPU; experts always on the CPU (lazy, streamed)
cpu_layers="$(seq -s '|' "$GPU_LAYERS" $((N_LAYER - 1)))"
OT='_exps\.=CPU'
if [[ -n "$cpu_layers" ]]; then
    OT="$OT,blk\.($cpu_layers)\.=CPU"
fi

echo "run-server.sh: GPU dense layers 0..$((GPU_LAYERS - 1)), ctx $CTX, threads $THREADS, port $PORT"
echo "run-server.sh: expected decode ~0.1-0.15 tok/s; loading reads ~58 GB of dense weights first"

export GGML_CPU_MOE_PREFETCH=1
if [[ "$PREDICT" != 0 ]]; then
    export LLAMA_MOE_PREDICT="$PREDICT" LLAMA_MOE_PREDICT_STATS="$PREDICT_STATS"
else
    unset LLAMA_MOE_PREDICT LLAMA_MOE_PREDICT_STATS
fi
# prompt batches (>= 32 tokens) use nearly every expert, so uploading each layer's expert tensors to the GPU and
# running the matmuls there is far faster than the CPU; single-token decode always stays on the CPU
if [[ "$OFFLOAD" == 1 ]]; then
    OFFLOAD_FLAG=""
else
    OFFLOAD_FLAG="--no-op-offload"
fi
exec "$BIN" "$@" \
    -m "$MODEL" \
    -lm none --lazy-mode on $OFFLOAD_FLAG -fit off --no-warmup \
    -ngl 999 -ot "$OT" \
    -t "$THREADS" -fa on -c "$CTX" -b "$UBATCH" -ub "$UBATCH" -np 1 \
    --host 127.0.0.1 --port "$PORT"
