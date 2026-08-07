#!/usr/bin/env bash
# Build a Qwen3.6-35B-A3B S8 recipe from either an existing stock Q4_0
# tensor map or a fresh BF16 Q4_0 dry-run. The fixed stage promotes base
# Q4_0 tensors selected by the Q4_K_M dry-run. The native stage additionally
# maps base Q5_0/Q4_1/Q6_K tensors to Q8_0 for V620-friendly inference.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  build-qwen35-q4-0-s8.sh --input PATH [options]

Input:
  PATH                         raw HF model directory, or an existing BF16 GGUF

Options:
  --repo PATH                  llama.cpp source tree (default: script parent)
  --build-dir PATH             build directory (default: REPO/build)
  --python PATH                Python interpreter (default: python3)
  --out-dir PATH               output directory (default: input parent)
  --bf16 PATH                  BF16 intermediate path
  --output PATH                final GGUF path
  --mmproj-output PATH         Q8_0 vision projector path
  --stock-q4-0 PATH            optional existing stock Q4_0 GGUF type map;
                               omit to derive the base map from a BF16 Q4_0 dry-run
  --stage stock|fixed|native|auto
                               stock=base map only (default: fixed)
                               fixed=base map + Q4_K_M-selected Q8 upgrades
                               native=also maps Q5_0/Q4_1/Q6_K to Q8_0
                               auto=BF16/imatrix-ranked Q4_0 -> Q8_0 upgrades
  --imatrix PATH               imatrix path (default: auto-detect $HOME/models/qwen35-imatrix/...)
  --no-imatrix                 disable imatrix use and force plain RTN
  --threads N                  quantizer threads (default: physical core count)
  --auto-q8-fraction PCT      auto-stage maximum final Q8_0 byte fraction (default: 25)
  --auto-max-tensor-mib N    auto-stage only promotes tensors no larger than N MiB (default: unlimited)
  --keep-bf16                  retain the BF16 intermediate (default: retain)
  --remove-bf16                remove BF16 only after successful quantization
  --skip-mmproj                do not create the vision projector
  --skip-convert               raw input must already have --bf16 present
  --plan-only                  convert/inspect/print the plan, do not quantize
  --allow-large-q8             continue when Q8_0 exceeds 50% of planned quantized bytes
                               (native/auto may require this after inspection)
  --force                      allow replacing existing output/plan files
  -h, --help                   show this help

Example:
  scripts/build-qwen35-q4-0-s8.sh \
    --input ~/models/Qwen3.6-35B-A3B-raw \
    --out-dir ~/models/qwen35-s8 \
    --threads 24
EOF
}

die() { echo "error: $*" >&2; exit 1; }
warn() { echo "warning: $*" >&2; }

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
BUILD_DIR="$REPO/build"
PYTHON=python3
INPUT=
OUT_DIR=
BF16=
FINAL=
MMPROJ=
STOCK_Q40="${S8_STOCK_Q4_0:-}"
STAGE=fixed
AUTO_Q8_FRACTION=25
AUTO_MAX_TENSOR_MIB=0
IMATRIX="${S8_IMATRIX_PATH:-${HOME}/models/qwen35-imatrix/imatrix_unsloth.gguf_file}"
USE_IMATRIX=1
THREADS=
KEEP_BF16=1
REMOVE_BF16=0
SKIP_MMPROJ=0
SKIP_CONVERT=0
PLAN_ONLY=0
ALLOW_LARGE_Q8=0
FORCE=0

while (($#)); do
    case "$1" in
        --input)          [[ $# -ge 2 ]] || die "--input needs a path"; INPUT=$2; shift 2 ;;
        --repo)           [[ $# -ge 2 ]] || die "--repo needs a path"; REPO=$2; shift 2 ;;
        --build-dir)      [[ $# -ge 2 ]] || die "--build-dir needs a path"; BUILD_DIR=$2; shift 2 ;;
        --python)         [[ $# -ge 2 ]] || die "--python needs a path"; PYTHON=$2; shift 2 ;;
        --out-dir)        [[ $# -ge 2 ]] || die "--out-dir needs a path"; OUT_DIR=$2; shift 2 ;;
        --bf16)           [[ $# -ge 2 ]] || die "--bf16 needs a path"; BF16=$2; shift 2 ;;
        --output)         [[ $# -ge 2 ]] || die "--output needs a path"; FINAL=$2; shift 2 ;;
        --mmproj-output)  [[ $# -ge 2 ]] || die "--mmproj-output needs a path"; MMPROJ=$2; shift 2 ;;
        --stock-q4-0)     [[ $# -ge 2 ]] || die "--stock-q4-0 needs a path"; STOCK_Q40=$2; shift 2 ;;
        --stage)          [[ $# -ge 2 ]] || die "--stage needs stock, fixed, native, or auto"; STAGE=$2; shift 2 ;;
        --imatrix)       [[ $# -ge 2 ]] || die "--imatrix needs a path"; IMATRIX=$2; USE_IMATRIX=1; shift 2 ;;
        --no-imatrix)    IMATRIX=; USE_IMATRIX=0; shift ;;
        --threads)        [[ $# -ge 2 ]] || die "--threads needs a number"; THREADS=$2; shift 2 ;;
        --auto-q8-fraction) [[ $# -ge 2 ]] || die "--auto-q8-fraction needs a percentage"; AUTO_Q8_FRACTION=$2; shift 2 ;;
        --auto-max-tensor-mib) [[ $# -ge 2 ]] || die "--auto-max-tensor-mib needs a size"; AUTO_MAX_TENSOR_MIB=$2; shift 2 ;;
        --keep-bf16)      KEEP_BF16=1; REMOVE_BF16=0; shift ;;
        --remove-bf16)    KEEP_BF16=0; REMOVE_BF16=1; shift ;;
        --skip-mmproj)    SKIP_MMPROJ=1; shift ;;
        --skip-convert)   SKIP_CONVERT=1; shift ;;
        --plan-only)      PLAN_ONLY=1; shift ;;
        --allow-large-q8) ALLOW_LARGE_Q8=1; shift ;;
        --force)          FORCE=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                die "unknown option: $1 (use --help)" ;;
    esac
done

[[ -n "$INPUT" ]] || { usage >&2; die "--input is required"; }
INPUT=$(readlink -f -- "$INPUT")
[[ -e "$INPUT" ]] || die "input does not exist: $INPUT"
REPO=$(readlink -f -- "$REPO")
BUILD_DIR=$(readlink -f -- "$BUILD_DIR")
if [[ -n "$STOCK_Q40" ]]; then
    STOCK_Q40=$(readlink -f -- "$STOCK_Q40")
    [[ -f "$STOCK_Q40" ]] || die "missing stock Q4_0 type-map GGUF: $STOCK_Q40"
fi
[[ "$STAGE" == stock || "$STAGE" == fixed || "$STAGE" == native || "$STAGE" == auto ]] || die "invalid stage: $STAGE (use stock, fixed, native, or auto)"
[[ "$AUTO_Q8_FRACTION" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid auto Q8 fraction: $AUTO_Q8_FRACTION"
[[ "$AUTO_MAX_TENSOR_MIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid auto max tensor size: $AUTO_MAX_TENSOR_MIB"
awk "BEGIN { exit !($AUTO_Q8_FRACTION <= 100) }" || die "auto Q8 fraction must be <= 100"
"$PYTHON" -V >/dev/null 2>&1 || die "cannot run Python interpreter: $PYTHON"
QUANT="$BUILD_DIR/bin/llama-quantize"
[[ -x "$QUANT" ]] || die "missing executable: $QUANT (build llama-quantize first)"
CONVERTER="$REPO/convert_hf_to_gguf.py"
[[ -f "$CONVERTER" ]] || die "missing converter: $CONVERTER"
RANKER="$REPO/scripts/rank-q4-0-q8-promotions.py"
if [[ "$STAGE" == auto ]]; then
    [[ -f "$RANKER" ]] || die "missing auto ranker: $RANKER"
fi

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR=$(dirname -- "$INPUT")
fi
OUT_DIR=$(readlink -m -- "$OUT_DIR")
mkdir -p -- "$OUT_DIR"

[[ -n "$BF16" ]] || BF16="$OUT_DIR/Qwen3.6-35B-A3B-MTP-BF16.gguf"
if [[ -z "$FINAL" ]]; then
    FINAL="$OUT_DIR/Qwen3.6-35B-A3B-MTP-Q4_0-S8-$STAGE.gguf"
fi
[[ -n "$MMPROJ" ]] || MMPROJ="$OUT_DIR/mmproj-Qwen3.6-35B-A3B-Q8_0.gguf"
BF16=$(readlink -m -- "$BF16")
FINAL=$(readlink -m -- "$FINAL")
MMPROJ=$(readlink -m -- "$MMPROJ")
if [[ "$USE_IMATRIX" -eq 1 && -n "$IMATRIX" ]]; then
    IMATRIX=$(readlink -m -- "$IMATRIX")
    if [[ -f "$IMATRIX" ]]; then
        IMATRIX=$(readlink -f -- "$IMATRIX")
    else
        warn "imatrix not found; continuing with plain RTN: $IMATRIX"
        IMATRIX=
    fi
fi
WORK="$OUT_DIR/q4_0-s8-$STAGE-plan"
BASE_LOG="$WORK/q4_0-base-dry-run.log"
AUTO_LOG="$WORK/auto-q8-ranking.log"
AUTO_PLAN="$WORK/auto-q8-promotions.txt"
PLAN_LOG="$WORK/q4_k_m-dry-run.log"
PLAN_TSV="$WORK/tensor-plan.tsv"
OVERRIDES="$WORK/tensor-overrides.txt"
SUMMARY="$WORK/summary.txt"
VALIDATION="$WORK/final-types.txt"
mkdir -p -- "$WORK"

for path in "$FINAL" "$BASE_LOG" "$AUTO_LOG" "$AUTO_PLAN" "$PLAN_LOG" "$PLAN_TSV" "$OVERRIDES" "$SUMMARY"; do
    if [[ -e "$path" && "$FORCE" -ne 1 ]]; then
        die "output already exists: $path (use --force to replace)"
    fi
done

if [[ -z "$THREADS" ]]; then
    THREADS=$(lscpu -p=Core 2>/dev/null | awk '!/^#/ && NF { print $1 }' | sort -u | wc -l)
    ((THREADS > 0)) || THREADS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
fi
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "invalid thread count: $THREADS"

if [[ -d "$INPUT" ]]; then
    if [[ "$SKIP_CONVERT" -eq 1 ]]; then
        [[ -f "$BF16" ]] || die "--skip-convert requires existing BF16 GGUF: $BF16"
    else
        if [[ -e "$BF16" && "$FORCE" -ne 1 ]]; then
            die "BF16 output already exists: $BF16 (use --force or --skip-convert)"
        fi
        echo "[1/5] converting raw HF model to BF16 GGUF"
        "$PYTHON" "$CONVERTER" "$INPUT" --outtype bf16 --outfile "$BF16"
    fi
    if [[ "$SKIP_MMPROJ" -eq 0 && ! -e "$MMPROJ" ]]; then
        echo "[2/5] converting vision projector directly to Q8_0"
        "$PYTHON" "$CONVERTER" "$INPUT" --mmproj --outtype q8_0 --outfile "$MMPROJ"
    elif [[ "$SKIP_MMPROJ" -eq 0 ]]; then
        echo "[2/5] keeping existing projector: $MMPROJ"
    fi
else
    [[ "$INPUT" == *.gguf ]] || die "file input must be a BF16 GGUF: $INPUT"
    BF16="$INPUT"
    echo "[1/5] using existing GGUF as the BF16 source: $BF16"
    if [[ "$SKIP_MMPROJ" -eq 0 ]]; then
        warn "a file input cannot create a projector; use a raw HF directory for --mmproj"
    fi
fi
[[ -f "$BF16" ]] || die "BF16 source was not created: $BF16"

if [[ -n "$STOCK_Q40" ]]; then
    echo "[3/6] using the supplied stock Q4_0 tensor map"
else
    echo "[3/6] deriving the base Q4_0 tensor map from BF16"
    base_args=(--dry-run "$BF16" Q4_0)
    if [[ "$USE_IMATRIX" -eq 1 && -n "$IMATRIX" ]]; then
        base_args=(--imatrix "$IMATRIX" "${base_args[@]}")
    fi
    "$QUANT" "${base_args[@]}" >"$BASE_LOG" 2>&1
fi

echo "[4/6] deriving the Q4_K_M sensitivity plan"
"$QUANT" --dry-run "$BF16" Q4_K_M >"$PLAN_LOG" 2>&1
if [[ "$STAGE" == auto ]]; then
    echo "[4/6] ranking BF16 Q4_0 -> Q8_0 promotions"
    rank_args=(--bf16 "$BF16" --q8-fraction "$AUTO_Q8_FRACTION" --protect-low-bandwidth --output "$AUTO_PLAN")
    [[ "$AUTO_MAX_TENSOR_MIB" != 0 ]] && rank_args+=(--max-tensor-mib "$AUTO_MAX_TENSOR_MIB")
    if [[ -n "$STOCK_Q40" ]]; then
        rank_args+=(--base-map "$STOCK_Q40")
    else
        rank_args+=(--base-log "$BASE_LOG")
    fi
    if [[ "$USE_IMATRIX" -eq 1 && -n "$IMATRIX" ]]; then
        rank_args+=(--imatrix "$IMATRIX")
    fi
    PYTHONPATH="$REPO/gguf-py${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$RANKER" "${rank_args[@]}" >"$AUTO_LOG" 2>&1 || {
        cat "$AUTO_LOG" >&2
        die "automatic Q8 ranking failed"
    }
    cat "$AUTO_LOG"
fi

PYTHONPATH="$REPO/gguf-py${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - "$PLAN_LOG" "$PLAN_TSV" "$OVERRIDES" "$SUMMARY" "$ALLOW_LARGE_Q8" "$STOCK_Q40" "$STAGE" "$BASE_LOG" "$AUTO_PLAN" <<'PY'
import math
import re
import sys
from collections import Counter
from pathlib import Path

from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType

log_path, plan_path, overrides_path, summary_path = map(Path, sys.argv[1:5])
allow_large_q8 = sys.argv[5] == "1"
stock_path = Path(sys.argv[6]) if sys.argv[6] else None
stage = sys.argv[7]
base_log_path = Path(sys.argv[8])
auto_plan_path = Path(sys.argv[9])
line_re = re.compile(
    r"^\[\s*\d+\s*/\s*\d+\]\s+(\S+)\s+-\s+\[([^\]]+)\],\s+"
    r"type\s*=\s*(\S+),\s+size\s*=\s*([0-9.]+)\s+MiB"
    r"(?:\s+->\s+[0-9.]+\s+MiB\s+\(([^)]+)\))?"
)
quant_ref = {"q4_k", "q5_k", "q6_k", "q8_0", "q5_0", "q4_0"}
protected = {"q5_k", "q6_k", "q8_0"}
stock_allowed = {"f32", "f16", "bf16", "q4_0", "q4_1", "q5_0", "q6_k", "q8_0"}
fixed_allowed = {"f32", "f16", "bf16", "q4_0", "q4_1", "q5_0", "q6_k", "q8_0"}
native_allowed = {"f32", "q4_0", "q8_0"}
block_bytes = {"q4_0": 18, "q4_1": 20, "q5_0": 22, "q6_k": 210, "q8_0": 34}
float_bytes = {"f16": 2, "bf16": 2, "f32": 4}

def type_name(value):
    return GGMLQuantizationType(int(value)).name.lower()

if stock_path is not None:
    stock_reader = GGUFReader(str(stock_path))
    stock_types = {tensor.name: type_name(tensor.tensor_type) for tensor in stock_reader.tensors}
    base_description = str(stock_path)
else:
    stock_types = {}
    for raw in base_log_path.read_text(errors="replace").splitlines():
        m = line_re.match(raw)
        if not m:
            continue
        name, _shape_text, source_type, _source_mib, final_type = m.groups()
        stock_types[name] = (final_type or source_type).lower()
    base_description = f"derived from {base_log_path}"
if not stock_types:
    raise SystemExit(f"base Q4_0 map is empty: {base_description}")
auto_promotions = set()
if stage == "auto" and auto_plan_path.exists():
    for raw in auto_plan_path.read_text(errors="replace").splitlines():
        if not raw or raw.startswith("#"):
            continue
        pattern = raw.split("\t", 1)[0]
        if pattern.startswith("^") and pattern.endswith("$=Q8_0"):
            auto_promotions.add(pattern[1:-len("$=Q8_0")])
        elif "=Q8_0" in pattern:
            auto_promotions.add(pattern.split("=", 1)[0])

rows = []
overrides = []
ref_counts = Counter()
stock_counts = Counter()
final_counts = Counter()
component_counts = Counter()
bytes_by_final = Counter()
q8_bytes = 0
quant_bytes = 0
mtp_seen = set()
promotions = 0

for raw in log_path.read_text(errors="replace").splitlines():
    m = line_re.match(raw)
    if not m:
        continue
    name, shape_text, source_type, source_mib, ref_type = m.groups()
    source_type = source_type.lower()
    if source_type in quant_ref:
        raise SystemExit(
            f"input appears already quantized ({source_type} at {name}); "
            "provide the raw HF directory or a BF16 GGUF, never requantize for S8"
        )
    stock_type = stock_types.get(name)
    if stock_type is None:
        raise SystemExit(f"base Q4_0 map is missing tensor: {name}")
    if stock_path is None and stock_type == "q4_0" and re.search(r"\.ssm_(?:alpha|beta)\.weight$", name):
        stock_type = "f32"
    if stock_type not in stock_allowed:
        raise SystemExit(f"unsupported stock tensor type {stock_type} at {name}")
    ref_type = ref_type.lower() if ref_type else "preserve"
    dims = [int(x) for x in re.findall(r"\d+", shape_text)]
    if not dims:
        raise SystemExit(f"cannot parse shape for {name}: {shape_text!r}")
    ncols = dims[0]
    is_mtp = ".nextn." in name.lower() or name.lower().startswith("nextn.")
    component = "MTP" if is_mtp else "language"
    component_counts[component] += 1
    ref_counts[ref_type] += 1
    stock_counts[stock_type] += 1

    # Start from the exact stock Q4_0 recipe. The Q4_K_M oracle is used only
    # to identify the 29 stock Q4_0 tensors worth promoting to Q8_0.
    final_type = stock_type
    reason = "stock-map"
    if stage in {"fixed", "native"} and stock_type == "q4_0" and ref_type in protected:
        final_type = "q8_0"
        reason = f"base-Q4_0-to-Q8_{ref_type.upper()}"
        promotions += 1
    if stage == "auto" and stock_type == "q4_0" and name in auto_promotions:
        final_type = "q8_0"
        reason = "BF16-error-ranked-Q4_0-to-Q8_0"
        promotions += 1
    if stock_type == "bf16":
        # llama-quantize emits these stock MTP entries as F32 from the BF16
        # source; this is a precision-preserving upgrade, not a downgrade.
        final_type = "f32"
        reason = "BF16-to-F32-preserve"
    if stage == "native":
        if stock_type in {"q5_0", "q4_1", "q6_k"}:
            final_type = "q8_0"
            reason = f"V620-{stock_type.upper()}-to-Q8_0"
    elif stage not in {"stock", "fixed", "auto"}:
        raise SystemExit(f"unsupported stage: {stage}")

    if final_type not in (native_allowed if stage == "native" else fixed_allowed):
        raise SystemExit(f"stage {stage} cannot emit {final_type} at {name}")
    if final_type != "q4_0":
        overrides.append(f"^{re.escape(name)}$={final_type.upper()}")
    if is_mtp and final_type in block_bytes:
        mtp_seen.add(name)

    ne = math.prod(dims)
    if final_type in block_bytes:
        if ncols % (256 if final_type == "q6_k" else 32):
            raise SystemExit(f"incompatible {final_type} shape for {name}: {shape_text}")
        block = 256 if final_type == "q6_k" else 32
        estimate = (ne // ncols) * (ncols // block) * block_bytes[final_type]
        quant_bytes += estimate
        if final_type == "q8_0":
            q8_bytes += estimate
    elif final_type in float_bytes:
        estimate = ne * float_bytes[final_type]
    else:
        raise SystemExit(f"cannot estimate final type {final_type} at {name}")
    bytes_by_final[final_type] += estimate
    final_counts[final_type] += 1
    rows.append((name, "x".join(map(str, dims)), component, source_type, stock_type, ref_type, final_type, reason, str(int(estimate))))

if not rows:
    raise SystemExit("the Q4_K_M dry-run produced no tensor records")
if stock_path is not None and stage in {"fixed", "native"}:
    expected_promotions = 29
    if promotions != expected_promotions:
        raise SystemExit(f"expected exactly {expected_promotions} base Q4_0 -> Q8_0 promotions, found {promotions}")
elif stage == "stock" and promotions != 0:
    raise SystemExit(f"stock stage unexpectedly promoted {promotions} tensors")
if not mtp_seen:
    raise SystemExit("no quantizable .nextn. MTP tensors were found; refusing to build")
plan_path.write_text(
    "name\tshape\tcomponent\tsource\tbase_type\tq4_k_m_reference\ts8_target\treason\testimated_bytes\n"
    + "\n".join("\t".join(r) for r in rows) + "\n"
)
overrides_path.write_text("\n".join(overrides) + ("\n" if overrides else ""))

with summary_path.open("w") as f:
    f.write(f"Stage: {stage}\n")
    f.write(f"Base Q4_0 map: {base_description}\n")
    f.write("Base tensor counts:\n")
    for k, v in sorted(stock_counts.items()): f.write(f"  {k}: {v}\n")
    f.write(f"\nQ4_K_M reference counts:\n")
    for k, v in sorted(ref_counts.items()): f.write(f"  {k}: {v}\n")
    f.write(f"\nBase Q4_0 -> Q8_0 promotions: {promotions}\n")
    f.write("\nPlanned final counts:\n")
    for k, v in sorted(final_counts.items()): f.write(f"  {k}: {v}\n")
    f.write("\nComponents:\n")
    for k, v in sorted(component_counts.items()): f.write(f"  {k}: {v}\n")
    f.write("\nEstimated output bytes by final type:\n")
    for k, v in sorted(bytes_by_final.items()): f.write(f"  {k}: {int(v)} ({v / 1024**3:.3f} GiB)\n")
    q8_fraction = q8_bytes / max(quant_bytes, 1)
    f.write(f"\nQ8_0 fraction of quantized bytes: {q8_fraction:.2%}\n")
    f.write(f"MTP tensors seen: {len(mtp_seen)}\n")

q8_fraction = q8_bytes / max(quant_bytes, 1)
if q8_fraction > 0.50 and not allow_large_q8:
    raise SystemExit(
        f"Q8_0 estimate is unexpectedly large: {q8_fraction:.1%} of quantized bytes; "
        "inspect the saved plan or rerun with --allow-large-q8"
    )
PY

cat "$SUMMARY"
echo "Full tensor plan: $PLAN_TSV"
cat "$PLAN_TSV"
echo "Explicit overrides: $OVERRIDES"
cat "$OVERRIDES"

if [[ "$PLAN_ONLY" -eq 1 ]]; then
    echo "plan-only requested; no final GGUF was written"
    exit 0
fi

if [[ -e "$FINAL" && "$FORCE" -ne 1 ]]; then
    die "final output already exists: $FINAL (use --force)"
fi

echo "[5/6] quantizing $STAGE recipe from the derived/supplied base map using $THREADS threads"
QUANT_ARGS=(--pure --tensor-type-file "$OVERRIDES")
if [[ -n "$IMATRIX" ]]; then
    echo "using imatrix: $IMATRIX"
    QUANT_ARGS+=(--imatrix "$IMATRIX")
fi
"$QUANT" "${QUANT_ARGS[@]}" "$BF16" "$FINAL" Q4_0 "$THREADS"
[[ -s "$FINAL" ]] || die "quantizer did not create a non-empty final file"

echo "[6/6] validating metadata and tensor types"
if [[ -x "$BUILD_DIR/bin/llama-gguf" ]]; then
    "$BUILD_DIR/bin/llama-gguf" "$FINAL" r n >"$VALIDATION" 2>&1 || warn "llama-gguf metadata read failed; see $VALIDATION"
    if ! grep -q 'nextn_predict_layers' "$VALIDATION"; then
        warn "final GGUF does not visibly contain nextn_predict_layers; inspect $VALIDATION"
    fi
fi

if PYTHONPATH="$REPO/gguf-py${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - "$FINAL" "$STAGE" <<'PY'
from collections import Counter
import sys
try:
    from gguf import GGUFReader
except Exception as exc:
    print(f"type validation skipped: {exc}")
    raise SystemExit(0)
reader = GGUFReader(sys.argv[1])
stage = sys.argv[2]
counts = Counter(t.tensor_type.name.lower() for t in reader.tensors)
print("Final tensor type counts:")
for key in sorted(counts):
    print(f"  {key}: {counts[key]}")
allowed = ({"f32", "q4_0", "q8_0"} if stage == "native" else
           {"f32", "bf16", "q4_0", "q4_1", "q5_0", "q6_k", "q8_0"})
unexpected = {key: value for key, value in counts.items() if key not in allowed}
if unexpected:
    raise SystemExit(f"unexpected {stage} tensor types: {unexpected}")
PY
then :; else die "final tensor type validation failed"; fi

if [[ "$REMOVE_BF16" -eq 1 ]]; then
    if [[ "$BF16" != "$FINAL" && -f "$BF16" ]]; then
        rm -f -- "$BF16"
        echo "removed BF16 intermediate: $BF16"
    fi
else
    echo "retaining BF16 intermediate: $BF16"
fi

echo
echo "S8 build complete"
echo "  final:   $FINAL ($(du -h "$FINAL" | awk '{print $1}'))"
echo "  plan:    $PLAN_TSV"
echo "  summary: $SUMMARY"
[[ "$SKIP_MMPROJ" -eq 1 ]] || echo "  mmproj:  $MMPROJ"