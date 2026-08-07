#!/usr/bin/env bash
# Build and compare stock-map S8 candidates, then select a Pareto-best recipe.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  optimize-qwen35-s8.sh --input PATH --bf16 PATH --out-root PATH \
    --code-kld-base PATH --wiki-kld-base PATH [options]

The input BF16 GGUF must already exist. Candidates are built from that source;
no quantized model is ever requantized.

Options:
  --repo PATH              llama.cpp source tree
  --build-dir PATH         build directory
  --python PATH            Python interpreter
  --stages LIST            comma-separated: stock,fixed,native,auto (default: all)
  --stock-q4-0 PATH        optional historical stock map; omit for BF16-derived base
  --imatrix PATH           optional imatrix for every candidate
  --threads N              quantizer threads (default: physical core count)
  --objective NAME         mtp|decode|prompt|balanced (default: balanced)
  --quality-tolerance PCT  allowed KLD increase over stock (default: 0)
  --auto-q8-fraction PCT  auto-stage final Q8 byte fraction (default: 25)
  --auto-max-tensor-mib N auto-stage maximum promoted tensor size (default: unlimited)
  --skip-build             evaluate existing candidate files only
  --skip-kld               skip KLD evaluation
  --skip-bench             skip V620 benchmarking
  --force                  replace candidate plans/output files
  -h, --help               show this help
EOF
}

die() { echo "error: $*" >&2; exit 1; }
warn() { echo "warning: $*" >&2; }

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
BUILD_DIR="$REPO/build"
PYTHON=python3
INPUT=
BF16=
OUT_ROOT=
STOCK_Q40=
CODE_KLD=
WIKI_KLD=
IMATRIX=
STAGES=stock,fixed,native
THREADS=
OBJECTIVE=balanced
QUALITY_TOLERANCE=0
AUTO_Q8_FRACTION=25
AUTO_MAX_TENSOR_MIB=0
SKIP_BUILD=0
SKIP_KLD=0
SKIP_BENCH=0
FORCE=0

while (($#)); do
    case "$1" in
        --input)             [[ $# -ge 2 ]] || die "--input needs a path"; INPUT=$2; shift 2 ;;
        --bf16)              [[ $# -ge 2 ]] || die "--bf16 needs a path"; BF16=$2; shift 2 ;;
        --out-root)          [[ $# -ge 2 ]] || die "--out-root needs a path"; OUT_ROOT=$2; shift 2 ;;
        --stock-q4-0)        [[ $# -ge 2 ]] || die "--stock-q4-0 needs a path"; STOCK_Q40=$2; shift 2 ;;
        --code-kld-base)     [[ $# -ge 2 ]] || die "--code-kld-base needs a path"; CODE_KLD=$2; shift 2 ;;
        --wiki-kld-base)     [[ $# -ge 2 ]] || die "--wiki-kld-base needs a path"; WIKI_KLD=$2; shift 2 ;;
        --repo)              [[ $# -ge 2 ]] || die "--repo needs a path"; REPO=$2; shift 2 ;;
        --build-dir)         [[ $# -ge 2 ]] || die "--build-dir needs a path"; BUILD_DIR=$2; shift 2 ;;
        --python)            [[ $# -ge 2 ]] || die "--python needs a path"; PYTHON=$2; shift 2 ;;
        --stages)            [[ $# -ge 2 ]] || die "--stages needs a list"; STAGES=$2; shift 2 ;;
        --imatrix)           [[ $# -ge 2 ]] || die "--imatrix needs a path"; IMATRIX=$2; shift 2 ;;
        --threads)           [[ $# -ge 2 ]] || die "--threads needs a number"; THREADS=$2; shift 2 ;;
        --objective)         [[ $# -ge 2 ]] || die "--objective needs a name"; OBJECTIVE=$2; shift 2 ;;
        --quality-tolerance) [[ $# -ge 2 ]] || die "--quality-tolerance needs a percentage"; QUALITY_TOLERANCE=$2; shift 2 ;;
        --auto-q8-fraction) [[ $# -ge 2 ]] || die "--auto-q8-fraction needs a percentage"; AUTO_Q8_FRACTION=$2; shift 2 ;;
        --auto-max-tensor-mib) [[ $# -ge 2 ]] || die "--auto-max-tensor-mib needs a size"; AUTO_MAX_TENSOR_MIB=$2; shift 2 ;;
        --skip-build)        SKIP_BUILD=1; shift ;;
        --skip-kld)          SKIP_KLD=1; shift ;;
        --skip-bench)        SKIP_BENCH=1; shift ;;
        --force)             FORCE=1; shift ;;
        -h|--help)           usage; exit 0 ;;
        *)                   die "unknown option: $1 (use --help)" ;;
    esac
done

[[ -n "$INPUT" && -n "$BF16" && -n "$OUT_ROOT" ]] || { usage >&2; die "--input, --bf16, and --out-root are required"; }
[[ "$OBJECTIVE" == mtp || "$OBJECTIVE" == decode || "$OBJECTIVE" == prompt || "$OBJECTIVE" == balanced ]] || die "invalid objective: $OBJECTIVE"
[[ "$QUALITY_TOLERANCE" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid quality tolerance: $QUALITY_TOLERANCE"
[[ "$AUTO_Q8_FRACTION" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid auto Q8 fraction: $AUTO_Q8_FRACTION"
[[ "$AUTO_MAX_TENSOR_MIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid auto max tensor size: $AUTO_MAX_TENSOR_MIB"
INPUT=$(readlink -f -- "$INPUT")
BF16=$(readlink -f -- "$BF16")
OUT_ROOT=$(readlink -m -- "$OUT_ROOT")
if [[ -n "$STOCK_Q40" ]]; then STOCK_Q40=$(readlink -f -- "$STOCK_Q40"); fi
REPO=$(readlink -f -- "$REPO")
BUILD_DIR=$(readlink -f -- "$BUILD_DIR")
[[ -e "$INPUT" ]] || die "input does not exist: $INPUT"
[[ -f "$BF16" ]] || die "BF16 source does not exist: $BF16"
[[ -z "$STOCK_Q40" || -f "$STOCK_Q40" ]] || die "stock Q4_0 map does not exist: $STOCK_Q40"
[[ -f "$CODE_KLD" ]] || { [[ "$SKIP_KLD" -eq 1 ]] || die "code KLD base does not exist: $CODE_KLD"; }
[[ -f "$WIKI_KLD" ]] || { [[ "$SKIP_KLD" -eq 1 ]] || die "wiki KLD base does not exist: $WIKI_KLD"; }
BUILDER="$REPO/scripts/build-qwen35-q4-0-s8.sh"
QUANT="$BUILD_DIR/bin/llama-quantize"
PPL="$BUILD_DIR/bin/llama-perplexity"
BENCH="$BUILD_DIR/bin/llama-bench"
SERVER="$BUILD_DIR/bin/llama-server"
[[ -x "$BUILDER" ]] || die "missing builder: $BUILDER"
[[ "$SKIP_KLD" -eq 1 || -x "$PPL" ]] || die "missing llama-perplexity: $PPL"
[[ "$SKIP_BENCH" -eq 1 || -x "$BENCH" ]] || die "missing llama-bench: $BENCH"
[[ "$SKIP_BENCH" -eq 1 || -x "$SERVER" ]] || die "missing llama-server: $SERVER"
if [[ -z "$THREADS" ]]; then
    THREADS=$(lscpu -p=Core 2>/dev/null | awk '!/^#/ && NF { print $1 }' | sort -u | wc -l)
    ((THREADS > 0)) || THREADS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
fi
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "invalid thread count: $THREADS"
mkdir -p -- "$OUT_ROOT"

IFS=, read -r -a STAGE_LIST <<< "$STAGES"
for stage in "${STAGE_LIST[@]}"; do
    [[ "$stage" == stock || "$stage" == fixed || "$stage" == native || "$stage" == auto ]] || die "invalid stage in list: $stage"
done

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export GGML_CUDA_ALLREDUCE="${GGML_CUDA_ALLREDUCE:-nccl}"
export GGML_CUDA_P2P="${GGML_CUDA_P2P:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-PXB}"
export GGML_TP_SHARDED_OUTPUT="${GGML_TP_SHARDED_OUTPUT:-1}"
unset GGML_HIP_Q4_0_DOT8

if [[ -z "$IMATRIX" && -f "$HOME/models/qwen35-imatrix/imatrix_unsloth.gguf_file" ]]; then
    IMATRIX="$HOME/models/qwen35-imatrix/imatrix_unsloth.gguf_file"
fi
if [[ -n "$IMATRIX" ]]; then
    IMATRIX=$(readlink -f -- "$IMATRIX")
    [[ -f "$IMATRIX" ]] || die "imatrix does not exist: $IMATRIX"
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
    for stage in "${STAGE_LIST[@]}"; do
        out="$OUT_ROOT/$stage"
        args=(--python "$PYTHON" --input "$INPUT" --bf16 "$BF16" --out-dir "$out"
              --stage "$stage" --skip-convert --skip-mmproj --threads "$THREADS")
        [[ -n "$STOCK_Q40" ]] && args+=(--stock-q4-0 "$STOCK_Q40")
        [[ -n "$IMATRIX" ]] && args+=(--imatrix "$IMATRIX")
        [[ "$stage" == auto ]] && args+=(--auto-q8-fraction "$AUTO_Q8_FRACTION")
        [[ "$stage" == auto && "$AUTO_MAX_TENSOR_MIB" != 0 ]] && args+=(--auto-max-tensor-mib "$AUTO_MAX_TENSOR_MIB")
        [[ "$FORCE" -eq 1 ]] && args+=(--force)
        echo "[build] $stage"
        "$BUILDER" "${args[@]}" >"$OUT_ROOT/$stage-build.log" 2>&1 || {
            tail -80 "$OUT_ROOT/$stage-build.log" >&2
            die "candidate build failed: $stage"
        }
    done
fi

run_kld() {
    local label=$1 model=$2 base=$3 log=$4
    timeout 1800 "$PPL" -m "$model" --kl-divergence-base "$base" --kl-divergence \
        -c 512 -b 512 --chunks 20 -ngl all -sm layer -ts 1/1/1/1 -fa on >"$log" 2>&1
}

run_bench() {
    local label=$1 model=$2
    "$BENCH" -m "$model" -p 4096 -n 0 -r 3 -b 2048 -ub 256 -ngl 999 \
        -sm layer -ts 1/1/1/1 -fa on -ctk f16 -ctv f16 -o jsonl >"$OUT_ROOT/$label-pp.jsonl" 2>"$OUT_ROOT/$label-pp.log"
    "$BENCH" -m "$model" -p 0 -n 512 -r 3 -b 2048 -ub 256 -ngl 999 \
        -sm layer -ts 1/1/1/1 -fa on -ctk f16 -ctv f16 -o jsonl >"$OUT_ROOT/$label-tg.jsonl" 2>"$OUT_ROOT/$label-tg.log"
}

run_mtp() {
    local label=$1 model=$2 port=$3
    local slog="$OUT_ROOT/$label-mtp.log" response="$OUT_ROOT/$label-mtp.json"
    "$SERVER" -m "$model" --host 127.0.0.1 --port "$port" -np 1 -c 4096 -b 2048 -ub 256 \
        -ngl all -sm layer -ts 1/1/1/1 -fa on --spec-type draft-mtp --spec-draft-n-max 3 \
        --no-webui -v >"$slog" 2>&1 &
    local pid=$!
    local ready=0
    for _ in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ready=1; break; fi
        sleep 1
    done
    if [[ "$ready" -ne 1 ]]; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        die "MTP server failed to start for $label"
    fi
    curl -fsS "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
        --data '{"prompt":"Explain how a GPU kernel performs quantized matrix multiplication and why memory bandwidth matters.","n_predict":512,"temperature":0,"seed":123,"cache_prompt":false,"stream":false}' >"$response"
    sleep 2
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

for stage in "${STAGE_LIST[@]}"; do
    model="$OUT_ROOT/$stage/Qwen3.6-35B-A3B-MTP-Q4_0-S8-$stage.gguf"
    [[ -f "$model" ]] || die "candidate GGUF missing: $model"
    if [[ "$SKIP_KLD" -eq 0 ]]; then
        echo "[kld] $stage"
        run_kld "$stage-code" "$model" "$CODE_KLD" "$OUT_ROOT/$stage-code-kld.log"
        run_kld "$stage-wiki" "$model" "$WIKI_KLD" "$OUT_ROOT/$stage-wiki-kld.log"
    fi
    if [[ "$SKIP_BENCH" -eq 0 ]]; then
        echo "[bench] $stage"
        run_bench "$stage" "$model"
        run_mtp "$stage" "$model" 18081
    fi
done

"$PYTHON" - "$OUT_ROOT" "$STAGES" "$OBJECTIVE" "$QUALITY_TOLERANCE" <<'PY'
import json
import math
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
stages = sys.argv[2].split(",")
objective = sys.argv[3]
tolerance = float(sys.argv[4]) / 100.0
rows = []
for stage in stages:
    def metric(path, pattern):
        text = path.read_text(errors="replace") if path.exists() else ""
        m = re.findall(pattern, text)
        return float(m[-1]) if m else float("nan")
    code = metric(root / f"{stage}-code-kld.log", r"Mean\s+KLD:\s+([0-9.]+)")
    wiki = metric(root / f"{stage}-wiki-kld.log", r"Mean\s+KLD:\s+([0-9.]+)")
    pp = tg = mtp = acc = float("nan")
    p = root / f"{stage}-pp.jsonl"
    if p.exists():
        vals = [json.loads(line)["avg_ts"] for line in p.read_text().splitlines() if line.strip()]
        pp = statistics.mean(vals) if vals else float("nan")
    p = root / f"{stage}-tg.jsonl"
    if p.exists():
        vals = [json.loads(line)["avg_ts"] for line in p.read_text().splitlines() if line.strip()]
        tg = statistics.mean(vals) if vals else float("nan")
    p = root / f"{stage}-mtp.json"
    if p.exists():
        data = json.loads(p.read_text())
        mtp = data.get("timings", {}).get("predicted_per_second", float("nan"))
    log = root / f"{stage}-mtp.log"
    if log.exists():
        m = re.findall(r"draft acceptance =\s*([0-9.]+)", log.read_text(errors="replace"))
        acc = float(m[-1]) if m else float("nan")
    rows.append({"stage": stage, "code_kld": code, "wiki_kld": wiki, "quality": max(code, wiki), "pp": pp, "tg": tg, "mtp": mtp, "acceptance": acc})

print("stage\tcode_kld\twiki_kld\tpp4096\ttg512\tmtp_tg512\tacceptance")
for r in rows:
    print("{stage}\t{code_kld:.6f}\t{wiki_kld:.6f}\t{pp:.3f}\t{tg:.3f}\t{mtp:.3f}\t{acceptance:.5f}".format(**r))
base = next((r for r in rows if r["stage"] == "stock"), rows[0])
eligible = [r for r in rows if math.isfinite(r["quality"]) and r["quality"] <= base["quality"] * (1 + tolerance)]
if not eligible:
    eligible = rows

def score(r):
    vals = {k: r[k] / base[k] for k in ("pp", "tg", "mtp") if math.isfinite(r[k]) and math.isfinite(base[k]) and base[k] > 0}
    if objective == "prompt": return vals.get("pp", -float("inf"))
    if objective == "decode": return vals.get("tg", -float("inf"))
    if objective == "mtp": return vals.get("mtp", -float("inf"))
    return math.prod(vals.values()) ** (1 / len(vals)) if vals else -float("inf")
chosen = max(eligible, key=score)
print(f"recommended={chosen['stage']} objective={objective} quality_floor={base['quality']*(1+tolerance):.6f}")
PY