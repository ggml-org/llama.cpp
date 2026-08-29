#!/usr/bin/env bash
# Launch the verified Qwen3.8-27B target on dynamically discovered matching GPUs.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-gfx1100-unified}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-27B-Q4-AutoRound-Code-GGUF}"
MODEL="${MODEL:-$MODEL_DIR/Qwen3.8-27B-Q4_0-AutoRound-Code.gguf}"
MMPROJ="${MMPROJ:-$MODEL_DIR/mmproj-model.gguf}"
BUNDLE="${BUNDLE:-$BUILD_DIR/bin/spec-sidecar-mtp}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm/core-10.0}"
PROFILE="${PROFILE:-safe}"
CTX_SIZE="${CTX_SIZE:-262144}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
REQUIRE_GPUS="${REQUIRE_GPUS:-2}"
SPLIT_MODE="${SPLIT_MODE:-tensor}"
KV_TYPE="${KV_TYPE:-}"
USE_SIDECAR=1
USE_GFX1100_ADD_RMS_FUSION=0
DRY_RUN=0
EXTRA_ARGS=()

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
usage() {
    cat <<'EOF'
Usage: scripts/run-qwen38-rdna-unified.sh [options] [-- extra llama-server args]

  --build-dir PATH     native architecture build (default build-gfx1100-unified)
  --model PATH         verified target GGUF
  --mmproj PATH        verified multimodal projector
  --bundle PATH        prepared spec-sidecar-mtp bundle
  --profile NAME       safe | experimental (default safe)
  --split-mode MODE    tensor | layer (default tensor)
  --kv-type TYPE       cache type (default f16 for tensor, q8_0 for layer)
  --ctx-size N         target context (default 262144)
  --host HOST          listen address (default 0.0.0.0)
  --port PORT          listen port (default 8080)
  --no-sidecar         target-only smoke mode
  --gfx1100-add-rms-fusion
                       opt in to the validated, default-off Add+RMSNorm fusion
  --dry-run            validate and print the command without executing it

GPU ordinals are derived at runtime from AMD SMI after validating the complete
RX 7900 XT identity tuple; PCI addresses and ordinals are never hard-coded.
Unknown devices are logged and skipped.
EOF
}

while (($#)); do
    case "$1" in
        --build-dir) [[ $# -ge 2 ]] || fail "$1 requires a value"; BUILD_DIR=$2; shift 2 ;;
        --model)     [[ $# -ge 2 ]] || fail "$1 requires a value"; MODEL=$2; shift 2 ;;
        --mmproj)    [[ $# -ge 2 ]] || fail "$1 requires a value"; MMPROJ=$2; shift 2 ;;
        --bundle)    [[ $# -ge 2 ]] || fail "$1 requires a value"; BUNDLE=$2; shift 2 ;;
        --profile)    [[ $# -ge 2 ]] || fail "$1 requires a value"; PROFILE=$2; shift 2 ;;
        --split-mode) [[ $# -ge 2 ]] || fail "$1 requires a value"; SPLIT_MODE=$2; shift 2 ;;
        --kv-type)    [[ $# -ge 2 ]] || fail "$1 requires a value"; KV_TYPE=$2; shift 2 ;;
        --ctx-size)   [[ $# -ge 2 ]] || fail "$1 requires a value"; CTX_SIZE=$2; shift 2 ;;
        --host)      [[ $# -ge 2 ]] || fail "$1 requires a value"; HOST=$2; shift 2 ;;
        --port)      [[ $# -ge 2 ]] || fail "$1 requires a value"; PORT=$2; shift 2 ;;
        --no-sidecar) USE_SIDECAR=0; shift ;;
        --gfx1100-add-rms-fusion) USE_GFX1100_ADD_RMS_FUSION=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown argument: $1 (put llama-server arguments after --)" ;;
    esac
done
case "$PROFILE" in safe|experimental) ;; *) fail "invalid profile: $PROFILE" ;; esac
case "$SPLIT_MODE" in tensor|layer) ;; *) fail "invalid split mode: $SPLIT_MODE" ;; esac
if [[ -z $KV_TYPE ]]; then
    [[ $SPLIT_MODE == tensor ]] && KV_TYPE=f16 || KV_TYPE=q8_0
fi
if [[ $SPLIT_MODE == tensor && $KV_TYPE != f16 && $KV_TYPE != bf16 ]]; then
    fail "tensor mode requires f16 or bf16 KV; got $KV_TYPE"
fi
[[ $REQUIRE_GPUS =~ ^[1-9][0-9]*$ ]] || fail "REQUIRE_GPUS must be positive"
[[ $CTX_SIZE =~ ^[1-9][0-9]*$ ]] || fail "CTX_SIZE must be a positive integer"
[[ -x $ROCM_PATH/bin/amd-smi ]] || fail "AMD SMI not found under $ROCM_PATH"
SERVER=$BUILD_DIR/bin/llama-server
[[ -x $SERVER ]] || fail "server not found: $SERVER"
[[ -f $MODEL && -f $MMPROJ ]] || fail "model/projector file missing"
SOURCE_FILE=$(dirname "$MODEL")/SOURCE.txt
[[ -f $SOURCE_FILE ]] || fail "SOURCE.txt verification evidence is missing"
grep -q 'revision=04a41723de3622e56bb499676ebaaacaa430f345' "$SOURCE_FILE" ||
    fail "model revision evidence does not match the pinned revision"
HASH_FILE=$(dirname "$MODEL")/SHA256SUMS
[[ -f $HASH_FILE ]] || fail "model SHA256SUMS evidence is missing"
grep -q '^6f02e53c762a4a29a795a2346704c07f35c8a8ae7b74967aa1c0fda6bf047100  Qwen3.8-27B-Q4_0-AutoRound-Code.gguf$' "$HASH_FILE" ||
    fail "target SHA-256 evidence is missing"
grep -q '^9da757136cb044abdf552334c56f2dcb63839799ea54c705ba4bcee807abdad2  mmproj-model.gguf$' "$HASH_FILE" ||
    fail "projector SHA-256 evidence is missing"

identity_json=$(mktemp)
trap 'rm -f "$identity_json"' EXIT
"$ROCM_PATH/bin/amd-smi" static --gpu all --asic --bus --vram --json > "$identity_json"
mapfile -t gpu_rows < <(python3 - "$identity_json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    rows = json.load(f).get("gpu_data", [])
for row in rows:
    a, b, v = row.get("asic", {}), row.get("bus", {}), row.get("vram", {})
    ident = (
        str(a.get("vendor_id", "")).lower() == "0x1002" and
        str(a.get("device_id", "")).lower() == "0x744c" and
        str(a.get("subvendor_id", "")).lower() == "0x1002" and
        str(a.get("subsystem_id", "")).lower() == "0x1002" and
        str(a.get("rev_id", "")).lower() == "0xcc" and
        a.get("target_graphics_version") == "gfx1100" and
        int(a.get("num_compute_units", -1)) == 84 and
        int(v.get("size", {}).get("value", -1)) == 20464 and
        v.get("size", {}).get("unit") == "MB"
    )
    if ident:
        print(f"{int(row['gpu'])}|{b.get('bdf', '?')}|{a['target_graphics_version']}")
    else:
        print(f"SKIP unknown GPU index={row.get('gpu')} bdf={b.get('bdf')} asic={a}", file=sys.stderr)
PY
)
((${#gpu_rows[@]} == REQUIRE_GPUS)) || fail "expected $REQUIRE_GPUS matching GPUs, found ${#gpu_rows[@]}"

indices=() devices=() splits=() arches=()
for row in "${gpu_rows[@]}"; do
    IFS='|' read -r idx bdf arch <<< "$row"
    indices+=("$idx"); devices+=("ROCm$idx"); splits+=(1); arches+=("$arch")
    printf 'selected GPU index=%s bdf=%s arch=%s (identity matched)\n' "$idx" "$bdf" "$arch" >&2
done
[[ $(printf '%s\n' "${arches[@]}" | sort -u | wc -l) -eq 1 ]] || fail "mixed GPU architectures are not supported by one native build"
ARCH=${arches[0]}
DEVICES=$(IFS=,; echo "${devices[*]}")
TENSOR_SPLIT=$(IFS=,; echo "${splits[*]}")
DRAFT_DEVICE=${devices[0]}
MMPROJ_DEVICE=${devices[${#devices[@]}-1]}

export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$BUILD_DIR/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
list=$($SERVER --list-devices 2>&1)
for dev in "${devices[@]}"; do grep -q "^[[:space:]]*$dev:" <<< "$list" || fail "$dev is absent from llama-server --list-devices"; done

export HSA_NO_SCRATCH_RECLAIM=1
export GGML_HIP_SAFE_STATE_IO=1
if [[ $ARCH == gfx1100 ]]; then
    [[ -z ${HSA_OVERRIDE_GFX_VERSION:-} ]] || fail "HSA_OVERRIDE_GFX_VERSION must be unset for native gfx1100"
    unset HSA_OVERRIDE_GFX_VERSION
    unset GGML_HIP_RDNA2_AUTO GGML_HIP_GFX1030_NATIVE GGML_HIP_GFX1030_Q8_1_FUSION
    export GGML_HIP_RDNA3_ADD_RMS_NORM_FUSION=$USE_GFX1100_ADD_RMS_FUSION
    case "$PROFILE" in
        safe)         export GGML_HIP_RDNA3_GDN_CHUNKED=0 ;;
        experimental) export GGML_HIP_RDNA3_GDN_CHUNKED=1 ;;
    esac
elif [[ $ARCH == gfx1030 ]]; then
    export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
    export GGML_HIP_RDNA2_AUTO="${GGML_HIP_RDNA2_AUTO:-1}"
    unset GGML_HIP_RDNA3_GDN_CHUNKED GGML_HIP_RDNA3_ADD_RMS_NORM_FUSION
else
    fail "unsupported architecture: $ARCH"
fi

MMPROJ_MODE=gpu
if [[ $SPLIT_MODE == tensor ]] && ((CTX_SIZE > 131072)); then
    # F16 KV at maximum context leaves insufficient VRAM for the 1.76 GiB
    # projector on either card. Keep multimodal support, but execute it on CPU.
    MMPROJ_MODE=cpu
fi

cmd=(
    "$SERVER"
    --model "$MODEL"
    --mmproj "$MMPROJ" --mmproj-device "$MMPROJ_DEVICE"
    --alias Qwen3.8-27B-Q4-AutoRound-Code
    --host "$HOST" --port "$PORT"
    --device "$DEVICES"
    --split-mode "$SPLIT_MODE" --tensor-split "$TENSOR_SPLIT" --main-gpu "${indices[0]}"
    --n-gpu-layers 999 --fit off
    --ctx-size "$CTX_SIZE"
    --batch-size 8192 --ubatch-size 4096
    --flash-attn on --cache-type-k "$KV_TYPE" --cache-type-v "$KV_TYPE"
    --parallel 1 --no-context-shift --ctx-checkpoints 0
    --cache-ram 0 --no-cache-idle-slots
)
if [[ $MMPROJ_MODE == cpu ]]; then
    cmd+=(--no-mmproj-offload)
fi

if ((USE_SIDECAR)); then
    [[ -f $BUILD_DIR/bin/spec_hip_sidecar.so ]] || fail "MTP sidecar library missing"
    [[ -d $BUNDLE && -f $BUNDLE/drafter_manifest.json && -f $BUNDLE/drafter_weights.bin && -f $BUNDLE/draft_head_ids.bin ]] ||
        fail "prepared MTP bundle is missing or incomplete: $BUNDLE"
    export SPEC_SIDECAR=1
    export LLAMA_SPEC_HIP_SIDECAR="$BUILD_DIR/bin/spec_hip_sidecar.so"
    export LLAMA_SPEC_HIP_WEIGHTS="$BUNDLE"
    export LLAMA_DRAFT_HEAD_IDS="$BUNDLE/draft_head_ids.bin"
    sidecar_max_pos=$CTX_SIZE
    ((sidecar_max_pos > 131072)) && sidecar_max_pos=131072
    export LLAMA_SPEC_HIP_MAX_POS="$sidecar_max_pos"
    cmd+=(
        --spec-type draft-mtp,ngram-map-k4v
        --spec-ngram-map-k4v-size-n 12
        --spec-ngram-map-k4v-size-m 48
        --spec-draft-n-max 3 --spec-draft-p-min 0
        --device-draft "$DRAFT_DEVICE"
        --spec-draft-ubatch-size 4096
    )
else
    unset SPEC_SIDECAR LLAMA_SPEC_HIP_SIDECAR LLAMA_SPEC_HIP_WEIGHTS LLAMA_DRAFT_HEAD_IDS LLAMA_SPEC_HIP_MAX_POS
    cmd+=(--spec-type none)
fi
cmd+=("${EXTRA_ARGS[@]}")

printf 'profile=%s arch=%s devices=%s split_mode=%s tensor_split=%s kv=%s mmproj=%s sidecar=%s add_rms_fusion=%s\n' \
    "$PROFILE" "$ARCH" "$DEVICES" "$SPLIT_MODE" "$TENSOR_SPLIT" "$KV_TYPE" "$MMPROJ_MODE" "$USE_SIDECAR" "$USE_GFX1100_ADD_RMS_FUSION" >&2
if ((DRY_RUN)); then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
exec "${cmd[@]}"
