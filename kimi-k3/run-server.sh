#!/usr/bin/env bash
# Run Kimi-K3 (IQ1_S-XS 540 GB by default, or MXFP4 1.5 TB) with routed experts streamed from NVMe, dense layers 0..GPU_LAYERS-1 on the GPU.
#
#   ./run-server.sh [extra llama-server args]
#
# env:
#   MODEL       first shard of the GGUF          (default: the IQ1_S-XS model, /mnt/k3/kimi-k3-iq1/...-00001-of-00034.gguf;
#               the MXFP4 model /mnt/k3/kimi-k3/MXFP4_MOE/...-00001-of-00031.gguf is the accuracy-preserving one)
#   PORT        listen port                      (default: 18090)
#   HOST        listen address                   (default: 0.0.0.0 = every interface, reachable from the local
#               network; 127.0.0.1 = this machine only). There is no API key by default: anyone who can reach the
#               port, or any web page open in a browser on the network (CORS allows all origins), can use the
#               server; set API_KEY to require one
#   API_KEY     require this key on API requests (the web UI asks for it once)
#   CTX         context size in tokens           (default: 262144). Picks one of two profiles:
#                 CTX > 8192:  KV_TYPE q8_0, GPU_LAYERS 0, output head on the CPU, CACHE_RAM 32768
#                 CTX <= 8192: KV_TYPE f16, GPU_LAYERS 11 (IQ1, OFFLOAD=1) / 19 (IQ1, OFFLOAD=0) / 8 (other),
#                              output head on the GPU, the server's prompt cache size
#   GPU_LAYERS  layers whose dense weights go to CUDA0, from layer 0 (default: see CTX). Measured on the 12 GB card
#               at CTX 8192: MXFP4 10 layers left ~20 MB free; IQ1 19 layers leave ~500 MB without offload, with it
#               12 leave ~490 MB and 16 OOM. At CTX 262144 the 0-layer default leaves ~800 MB idle; each dense layer
#               takes ~345 MiB, so more than 0 is unmeasured there ("failed to allocate compute pp buffers" = too many)
#   KV_TYPE     attention cache type, f16 or q8_0 (default: see CTX; passed as both -ctk and -ctv, the model needs
#               them equal). The cache holds 24 layers x 576 values per token: 3.6 GiB at 262144 tokens in q8_0;
#               f16 is refused above CTX 131072 (6.75 GiB at 262144 does not fit)
#   CACHE_RAM   host memory for the server's prompt cache in MiB (--cache-ram; default: 32768 when CTX > 8192, else
#               the server's 8192). One long chat needs room for its cache (~4.1 GB at 256k) plus its context
#               checkpoints (443 MiB each, up to 32 = ~14 GB); a smaller limit drops long chats when the UI switches
#               conversations, and returning to one then reprocesses it from the start; 0 turns the cache off.
#               The active chat's own checkpoints (up to ~14 GB) come on top; both take host RAM away from the
#               page cache that expert streaming uses
#   THREADS     CPU threads                      (default: the number of physical cores)
#   NUMA        1: on a machine with several NUMA nodes (sockets), each node's threads compute their own part of
#               every matrix, and that part of the weights is kept in the node's memory: the dense weights are split
#               at load (in ordinary, not pinned, host memory), the experts when they are read into the page cache
#               (--numa distribute --no-host --no-repack). 0: one memory pool, as on a single socket
#               (default: 1 if the machine has more than one NUMA node)
#   UBATCH      prompt batch size (-b = -ub); every batch reads nearly all experts once, so a larger batch means fewer
#               passes over the 545 GB of experts (default: 4096 with a GPU of 15 GB or more, else 2048; at CTX 262144
#               4096 uses 13.3 of the 16.3 GB of an RTX 5070 Ti)
#   PREDICT     next-layer expert prediction: 0 off, 1 observe (statistics only), 2 prefetch (default: 2)
#   OFFLOAD     1: prompt batches run the expert matmuls on the GPU (each layer's expert tensors are uploaded,
#               streamed from disk ahead of the copy); needs ~3 expert tensors of GPU memory, so only the IQ1
#               model fits (default: 1 for an IQ1 model file, 0 otherwise)
#   PREDICT_STATS  where the learned prediction state is kept across restarts
#               (default: ${XDG_STATE_HOME:-~/.local/state}/llama.cpp/<model file name>.predict.bin, one per model)
#   CHECKPOINT_FILL  1: a prompt batch is not ended early at a user message just to place a checkpoint there when the
#               rest of the prompt fits in the batch (--checkpoint-fill-batch): every batch costs a pass over the experts
#               (~65-95 s), so this saves one pass on the first request and on every chat turn whose history the
#               template re-renders; the checkpoint 4 tokens before the end of each prompt is still made (default: 1)
#   CHECKPOINTS context checkpoints per slot (llama-server --ctx-checkpoints; default: the server's). The server
#               splits a prompt at the user-turn start to create one, which costs a second pass over the experts
#               (~1.6k-token prompt: 108 s with, 76 s without). 0 is right for single-turn requests; keep the
#               default when conversations continue across requests (the checkpoint is what lets the next turn
#               reuse the processed prefix)
#   UI_CONFIG   web UI defaults for a browser's first visit (--ui-config; default: '{"displayPerformance":true}',
#               the live performance panel; browsers that already saved settings tick Display > Display performance once)
#   UI_PATH     serve the web UI from this directory (--path) instead of the one built into the binary, e.g. after
#               `cd tools/ui && LLAMA_UI_OUT_DIR=/some/dir npm run build` (no relink); skips the UI build checks below
#
# THREADS should stay at the physical core count: the prediction worker runs on a spare hyperthread
# NUMA=1 works best with kernel.numa_balancing=0 (the kernel otherwise keeps sampling and moving the dense weights)
#
# memory: the dense weights of the CPU layers (~33 GB at the long-context default) are read once at load with direct
# I/O (-lm dio) into host memory that expert streaming cannot evict: pinned with NUMA=0, split over the nodes with
# NUMA=1; the routed experts stay on disk (--lazy-mode on) and are read per token through the page cache with
# GGML_CPU_MOE_PREFETCH=1. --no-warmup: the warmup run would route all 896 experts per layer.
#
# measured 2026-11-03 (IQ1 model, 2x E5-2699 v4, 448 GB RAM, RTX 5070 Ti 16 GB, CTX 262144, 2.6k-token document
# prompt + 128 tokens, page cache warm): prompt 74 s, decode ~0.82-0.87 tok/s; a chat turn that adds ~50 tokens
# ~27 s of prompt. The first request after a cold page cache decodes slower (~0.5 tok/s) while the experts are read.
# Before these defaults (14 threads, no NUMA split, 2048-token batches): prompt ~145 s, decode ~0.67 tok/s.
#
# long context (CTX 262144): the attention cache must stay on the GPU (in host RAM the CPU would do the attention,
# +1-2 s per token near a full context), so the dense layers and the output head move to the CPU to make room:
# expected ~+0.09 s per token. Near a full context the GPU also reads the whole cache per token (estimated ~20-25 ms)
# and each prompt batch adds an estimated ~15-25 s of attention to its ~70-100 s of expert streaming.
# A fresh 256k-token prompt therefore takes hours; a chat that grows turn by turn only processes each new turn

set -euo pipefail

die() { echo "run-server.sh: $*" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$HERE/../build/bin/llama-server}"
MODEL="${MODEL:-/mnt/k3/kimi-k3-iq1/Kimi-K3-abliterated-IQ1_S-XS-00001-of-00034.gguf}"
PORT="${PORT:-18090}"
HOST="${HOST:-0.0.0.0}"
API_KEY="${API_KEY:-}"
CTX="${CTX:-262144}"
[[ "$CTX" =~ ^[1-9][0-9]{0,7}$ ]] || die "CTX must be a positive number of tokens"
# small contexts keep the previously measured layout; larger ones need the GPU memory for the attention cache
if [[ "$CTX" -le 8192 ]]; then
    LONG_CTX=0
else
    LONG_CTX=1
fi
GPU_LAYERS_SET="${GPU_LAYERS:+1}"
if [[ "$(basename "$MODEL")" == *IQ1* ]]; then
    OFFLOAD="${OFFLOAD:-1}"
    # the offload staging buffer takes ~2.3 GB more GPU memory: 11 dense layers with it (12 left < 500 MiB), 19 without
    if [[ "$LONG_CTX" == 1 ]]; then
        GPU_LAYERS="${GPU_LAYERS:-0}"
    elif [[ "$OFFLOAD" == 1 ]]; then
        GPU_LAYERS="${GPU_LAYERS:-11}"
    else
        GPU_LAYERS="${GPU_LAYERS:-19}"
    fi
else
    if [[ "$LONG_CTX" == 1 ]]; then
        GPU_LAYERS="${GPU_LAYERS:-0}"
    else
        GPU_LAYERS="${GPU_LAYERS:-8}"
    fi
    OFFLOAD="${OFFLOAD:-0}"
fi
if [[ "$LONG_CTX" == 1 ]]; then
    KV_TYPE="${KV_TYPE:-q8_0}"
    CACHE_RAM="${CACHE_RAM:-32768}"
else
    KV_TYPE="${KV_TYPE:-f16}"
    CACHE_RAM="${CACHE_RAM:-}"
fi
# physical cores: unique (socket, core) pairs
CORES="$(lscpu -p=SOCKET,CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
THREADS="${THREADS:-${CORES:-14}}"
NODES="$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)"
if [[ "${NODES:-1}" -gt 1 ]]; then
    NUMA="${NUMA:-1}"
else
    NUMA="${NUMA:-0}"
fi
VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
if [[ "${VRAM_MIB:-0}" =~ ^[0-9]+$ && "${VRAM_MIB:-0}" -ge 15000 ]]; then
    UBATCH="${UBATCH:-4096}"
else
    UBATCH="${UBATCH:-2048}"
fi
CHECKPOINT_FILL="${CHECKPOINT_FILL:-1}"
PREDICT="${PREDICT:-2}"
PREDICT_STATS="${PREDICT_STATS:-${XDG_STATE_HOME:-$HOME/.local/state}/llama.cpp/$(basename "$MODEL" .gguf).predict.bin}"
CHECKPOINTS="${CHECKPOINTS:-}"
UI_CONFIG_DEFAULT='{"displayPerformance":true}'
UI_CONFIG="${UI_CONFIG:-$UI_CONFIG_DEFAULT}"
UI_PATH="${UI_PATH:-}"
N_LAYER=93

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
        -lm|--load-mode|-lzm|--lazy-mode|-fit|--fit|-ot|--override-tensor|-ngl|--n-gpu-layers|--gpu-layers|-dev|--device|-ts|--tensor-split|--rpc|--no-host|-nr|--no-repack|--repack)
            die "refusing '$a': load mode, lazy mode and tensor placement are fixed by this script (use GPU_LAYERS)" ;;
        --numa)
            die "refusing '$a': set it with the NUMA environment variable" ;;
        -m|--model|-c|--ctx-size|-t|--threads|--port|-b|--batch-size|-ub|--ubatch-size)
            die "refusing '$a': set it with the MODEL / CTX / THREADS / PORT / UBATCH environment variables" ;;
        -fa|--flash-attn|-np|--parallel|--host|--api-key|--api-key-file)
            die "refusing '$a': fixed by this script" ;;
        -ctxcp|--ctx-checkpoints|--swa-checkpoints|--checkpoint-fill-batch)
            die "refusing '$a': set it with the CHECKPOINTS environment variable" ;;
        -ctk|--cache-type-k|-ctv|--cache-type-v)
            die "refusing '$a': set it with the KV_TYPE environment variable" ;;
        -cram|--cache-ram)
            die "refusing '$a': set it with the CACHE_RAM environment variable" ;;
        -nkvo|--no-kv-offload|-kvo|--kv-offload)
            die "refusing '$a': the attention cache must stay on the GPU" ;;
        --ui-config|--webui-config|--ui-config-file|--webui-config-file|--path)
            die "refusing '$a': set it with the UI_CONFIG / UI_PATH environment variables" ;;
    esac
done

# any setting can also come from the environment (LLAMA_ARG_*): start from a clean slate
for v in $(compgen -e | grep '^LLAMA_ARG_' || true); do
    unset "$v"
done
# these would silently turn the pinned dense weights into pageable memory, or offload single-token decode
unset GGML_CUDA_NO_PINNED GGML_CUDA_ENABLE_UNIFIED_MEMORY GGML_OP_OFFLOAD_MIN_BATCH
# llama-server adds LLAMA_API_KEY to the keys given on the command line: only API_KEY may set one
unset LLAMA_API_KEY

# ... and from llama.cpp config files, which are applied before the environment and the arguments
for f in /etc/llama.cpp/config.ini "${XDG_CONFIG_HOME:-$HOME/.config}/llama.cpp/config.ini"; do
    [[ -e "$f" ]] && die "llama.cpp config file $f could override this script's settings, move it away first"
done

[[ -x "$BIN" ]]   || die "llama-server not found at $BIN (build the worktree first)"
[[ -f "$MODEL" ]] || die "model not found: $MODEL"
[[ "$GPU_LAYERS" =~ ^(0|[1-9][0-9]?)$ && "$GPU_LAYERS" -le "$N_LAYER" ]] || die "GPU_LAYERS must be 0..$N_LAYER"
[[ "$UBATCH" =~ ^[1-9][0-9]{0,3}$ && "$UBATCH" -ge 32 && "$UBATCH" -le 8192 ]] || die "UBATCH must be 32..8192"
[[ "$PREDICT" =~ ^[012]$ ]] || die "PREDICT must be 0, 1 or 2"
[[ "$OFFLOAD" =~ ^[01]$ ]] || die "OFFLOAD must be 0 or 1"
[[ "$NUMA" =~ ^[01]$ ]] || die "NUMA must be 0 or 1"
[[ "$CHECKPOINT_FILL" =~ ^[01]$ ]] || die "CHECKPOINT_FILL must be 0 or 1"
[[ "$THREADS" =~ ^[1-9][0-9]{0,3}$ ]] || die "THREADS must be a positive number"
[[ -z "$CHECKPOINTS" || "$CHECKPOINTS" =~ ^(0|[1-9][0-9]{0,5})$ ]] || die "CHECKPOINTS must be a non-negative integer"
[[ "$KV_TYPE" == f16 || "$KV_TYPE" == q8_0 ]] || die "KV_TYPE must be f16 or q8_0"
[[ "$KV_TYPE" != f16 || "$CTX" -le 131072 ]] || die "KV_TYPE=f16 does not fit the GPU above CTX 131072, use q8_0"
[[ -z "$CACHE_RAM" || "$CACHE_RAM" =~ ^(0|[1-9][0-9]{0,6})$ ]] || die "CACHE_RAM must be a number of MiB"
if [[ "$LONG_CTX" == 1 && "$GPU_LAYERS_SET" == 1 && "$GPU_LAYERS" -gt 0 ]]; then
    echo "run-server.sh: warning: GPU_LAYERS=$GPU_LAYERS with CTX $CTX is unmeasured: each dense layer takes ~345 MiB of GPU memory and CTX 262144 leaves ~800 MiB with 0; 'failed to allocate compute pp buffers' means lower it" >&2
fi

[[ "$HOST" =~ ^[0-9A-Za-z.:_-]+$ && "$HOST" != -* ]] || die "HOST must be an address or host name"
# the key goes to the server through a pipe (--api-key-file), not the command line every local user can read; the
# file form is one key per line with # comments, so a key must not contain a line break or start with #
[[ "$API_KEY" != *$'\n'* && "$API_KEY" != \#* ]] || die "API_KEY must not contain a line break or start with #"
KEY_FLAG=()
if [[ -n "$API_KEY" ]]; then
    # fd 3 stays open across exec (a <(...) pipe would be closed before exec runs)
    exec 3<<<"$API_KEY"
    KEY_FLAG=(--api-key-file /dev/fd/3)
fi
if [[ "$HOST" != 127.* && "$HOST" != localhost && "$HOST" != ::1 && -z "$API_KEY" ]]; then
    echo "run-server.sh: listening on $HOST:$PORT without an API key: anyone on the network (and any web page open in a browser there, CORS allows all origins) can use this server; set API_KEY to require one" >&2
fi

# the web UI: the embedded one must be built from tools/ui (-DLLAMA_BUILD_UI=ON), not the downloaded release bundle

BUILD_DIR="$(cd "$(dirname "$BIN")/.." && pwd)"
UI_FLAG=()
if [[ -n "$UI_PATH" ]]; then
    [[ -f "$UI_PATH/index.html" ]] || die "UI_PATH=$UI_PATH has no index.html"
    UI_FLAG=(--path "$UI_PATH")
else
    # the UI is linked into the server library when it is shared, else into the binary
    UI_LIB="$(dirname "$BIN")/libllama-server-impl.so"
    [[ -f "$UI_LIB" ]] || UI_LIB="$BIN"
    if [[ -e "$BUILD_DIR/tools/ui/.ui-stamp" ]]; then
        die "the embedded web UI is the downloaded release bundle (no performance panel): reconfigure with -DLLAMA_BUILD_UI=ON and rebuild, or set UI_PATH"
    fi
    if [[ "$BUILD_DIR/tools/ui/ui.cpp" -nt "$UI_LIB" ]]; then
        die "the web UI was regenerated after $(basename "$UI_LIB") was linked: rebuild llama-server first"
    fi
    if [[ -f "$HERE/../tools/ui/dist/index.html" ]]; then
        echo "run-server.sh: warning: tools/ui/dist exists and is embedded instead of the UI built from source" >&2
    fi
fi

# another llama process competes for the same RAM and page cache
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
# long contexts: the ~0.9 GB output head goes to the CPU too (read once per token, ~17 ms; prompt batches only
# compute the last token's logits, so op offload never uploads it)
if [[ "$LONG_CTX" == 1 ]]; then
    OT="$OT,^output\.=CPU"
    HEAD_ON=CPU
else
    HEAD_ON=GPU
fi
if [[ "$GPU_LAYERS" == 0 ]]; then
    gpu_desc="GPU dense layers: none"
else
    gpu_desc="GPU dense layers 0..$((GPU_LAYERS - 1))"
fi
CACHE_RAM_FLAG=()
cache_desc="server default"
if [[ -n "$CACHE_RAM" ]]; then
    CACHE_RAM_FLAG=(--cache-ram "$CACHE_RAM")
    cache_desc="$CACHE_RAM MiB"
fi

echo "run-server.sh: $gpu_desc, output head on $HEAD_ON, ctx $CTX, KV $KV_TYPE, prompt cache $cache_desc, threads $THREADS, NUMA $NUMA ($NODES nodes), port $PORT"
echo "run-server.sh: loading reads ~35 GB of dense weights first (direct I/O, not kept in the page cache)"

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
# NUMA: split rows per node; the split dense weights must be in ordinary host memory (pinned memory is placed when
# it is allocated), and repacked weights would be computed with another row split
NUMA_FLAG=()
if [[ "$NUMA" == 1 ]]; then
    NUMA_FLAG=(--numa distribute --no-host --no-repack)
fi
CHECKPOINTS_FLAG=()
if [[ -n "$CHECKPOINTS" ]]; then
    CHECKPOINTS_FLAG=(--ctx-checkpoints "$CHECKPOINTS")
fi
if [[ "$CHECKPOINT_FILL" == 1 ]]; then
    CHECKPOINTS_FLAG+=(--checkpoint-fill-batch)
fi
unset API_KEY
exec "$BIN" "$@" \
    -m "$MODEL" \
    -lm dio --lazy-mode on $OFFLOAD_FLAG -fit off --no-warmup "${CHECKPOINTS_FLAG[@]}" "${NUMA_FLAG[@]}" \
    -ngl 999 -ot "$OT" \
    -t "$THREADS" -fa on -c "$CTX" -ctk "$KV_TYPE" -ctv "$KV_TYPE" "${CACHE_RAM_FLAG[@]}" \
    -b "$UBATCH" -ub "$UBATCH" -np 1 \
    --ui-config "$UI_CONFIG" "${UI_FLAG[@]}" \
    "${KEY_FLAG[@]}" --host "$HOST" --port "$PORT"
