#!/usr/bin/env bash
# Predeclared, unprofiled DSV4 raw-TG screen for RCCL algorithm/protocol candidates.
set -Eeuo pipefail

ROOT_DIR=${DSV4_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RUN_TG="$ROOT_DIR/scripts/dsv4-rocm/run-tg.sh"
[[ -x "$RUN_TG" ]] || { echo "ERROR: run-tg.sh not executable: $RUN_TG" >&2; exit 2; }

usage() {
    cat <<'USAGE'
Usage: scripts/dsv4-rocm/screen-rccl-tg.sh auto|tree-ll|ring-ll [--dry-run]

Runs the unchanged target-only raw-TG contract at 16K/32K/64K with one model
load, six raw repetitions, the predeclared first repetition discarded, and no
profiler. Environment controls are process-scoped and recorded by run-tg.sh.

Candidates:
  auto     RCCL tuning model; NCCL_ALGO/NCCL_PROTO unset (control)
  tree-ll  NCCL_ALGO=Tree, NCCL_PROTO=LL
  ring-ll  NCCL_ALGO=Ring, NCCL_PROTO=LL

Channel forcing is intentionally excluded: the installed RCCL 2.30.4 binary
reports NCCL_MIN_NCHANNELS is ignored for fewer than eight GPUs. Use tree-ll
first; run auto only if it is plausibly >=3% above the accepted historical
baseline, then ring-ll only if tree-ll fails the matched gate.
USAGE
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
candidate=$1
shift
case "$candidate" in
    auto|tree-ll|ring-ll) ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown RCCL candidate: $candidate" >&2; usage >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "ERROR: only --dry-run may follow the candidate" >&2; exit 2; }
[[ $# -eq 0 || $1 == --dry-run ]] || { echo "ERROR: unknown argument: $1" >&2; exit 2; }

# Fail closed on inherited communication tuning. Candidate identity means the
# complete NCCL/RCCL environment below, not merely ALGO/PROTO layered onto an
# unknown shell configuration.
contaminants=()
while IFS='=' read -r name _; do
    case "$name" in NCCL_*|RCCL_*) contaminants+=("$name") ;; esac
done < <(env)
if [[ ${#contaminants[@]} -ne 0 ]]; then
    printf 'ERROR: inherited communication environment is forbidden:' >&2
    printf ' %s' "${contaminants[@]}" >&2
    printf '\n' >&2
    exit 2
fi

case "$candidate" in
    auto) ;;
    tree-ll) export NCCL_ALGO=Tree NCCL_PROTO=LL ;;
    ring-ll) export NCCL_ALGO=Ring NCCL_PROTO=LL ;;
esac

# Force the exact accepted raw-TG contract rather than accepting inherited
# overrides. Timeout/output-root variables remain operational controls only.
export DSV4_RCCL_CANDIDATE="$candidate"
export DSV4_TG_MODE=performance
export DSV4_TG_PROFILE=none
export DSV4_TG_DEPTHS=16384,32768,65536
export DSV4_TG_N_GEN=32
export DSV4_TG_REPS=6
export DSV4_TG_DISCARD_FIRST=1
export DSV4_TG_STABILITY_LIMIT=0.03
export DSV4_TG_DEPTH_STATE_API=context
export DSV4_HASH_MODE=full
export DSV4_LABEL=raw-tg-rccl-screen-$candidate
export DSV4_MODEL=/home/edwin/models/DeepSeek-V4-Flash-0731-GGUF/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf
export DSV4_BENCH="$ROOT_DIR/build/bin/llama-bench"
export DSV4_BATCH=512 DSV4_UBATCH=256 DSV4_THREADS=12
export DSV4_TENSOR_SPLIT=1/1/1/1 DSV4_CACHE_TYPE_K=f16 DSV4_CACHE_TYPE_V=f16 DSV4_LOAD_MODE=mmap
export DSV4_REQUIRE_ACCEPTED_STACK=1 DSV4_ALLOW_BUSY_GPUS=0
export GGML_HIP_RDNA2_MMQ_J=16 GGML_HIP_RDNA2_HC_MIXES=1 GGML_HIP_RDNA2_LID_SUBWAVE=4
export GGML_CUDA_ALLREDUCE=nccl GGML_CUDA_P2P=1 GGML_HIP_GRAPHS=1
export HSA_NO_SCRATCH_RECLAIM=1 HSA_OVERRIDE_GFX_VERSION=10.3.0
# INFO is restricted to communicator setup/tuning and occurs outside accepted TG.
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ENV,TUNING

exec "$RUN_TG" "$@"