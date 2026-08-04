#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 LLAMA_BENCH MODEL OUT_DIR [llama-bench arguments...]" >&2
    exit 2
fi

bench=$1
model=$2
out_dir=$3
shift 3
extra_args=("$@")

if [[ ! -x "$bench" ]]; then
    echo "llama-bench is not executable: $bench" >&2
    exit 2
fi
if [[ ! -f "$model" ]]; then
    echo "model does not exist: $model" >&2
    exit 2
fi

read -r -a ubatches <<< "${PREFILL_UBATCHES:-512 1024 2048 4096 8192}"
threads=${PREFILL_THREADS:-25}
repetitions=${PREFILL_REPETITIONS:-5}
tokens=${PREFILL_TOKENS:-8192}
python_bin=${PYTHON:-python3}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-moe-ubatch"
mkdir -p "$run_dir"
nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
printf 'case\tubatch\tenvironment\n' > "$run_dir/manifest.tsv"

cases=(
    'disabled|GGML_CUDA_MOE_MMQ_DISABLE=1'
    'persistent|'
    'persistent-w13-epilogue-quant|GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused-quant'
)

for entry in "${cases[@]}"; do
    label=${entry%%|*}
    environment=${entry#*|}
    env_args=()
    if [[ -n $environment ]]; then
        read -r -a env_args <<< "$environment"
    fi
    for ubatch in "${ubatches[@]}"; do
        printf '%s\t%s\t%s\n' "$label" "$ubatch" "$environment" >> "$run_dir/manifest.tsv"
        env GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 "${env_args[@]}" "$bench" \
            -m "$model" \
            -p "$tokens" \
            -n 0 \
            -r "$repetitions" \
            -t "$threads" \
            -ngl 999 \
            -b 8192 \
            -ub "$ubatch" \
            -fa on \
            -o jsonl \
            "${extra_args[@]}" \
            > "$run_dir/$label-ub$ubatch.jsonl" \
            2> "$run_dir/$label-ub$ubatch.stderr"
    done
done

"$python_bin" - "$run_dir" > "$run_dir/summary.md" <<'PY'
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
rows = []
for path in sorted(run_dir.glob("*.jsonl")):
    match = re.fullmatch(r"(.+)-ub(\d+)", path.stem)
    if match is None:
        continue
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for value in values:
        rows.append((match.group(1), int(match.group(2)), value))

baseline = {(ubatch, int(value["n_prompt"])): float(value["avg_ts"])
            for case, ubatch, value in rows if case == "disabled"}
print("| case | ubatch | tokens | latency ms | tok/s | vs disabled |")
print("|---|---:|---:|---:|---:|---:|")
for case, ubatch, value in rows:
    tokens = int(value["n_prompt"])
    throughput = float(value["avg_ts"])
    speedup = throughput / baseline.get((ubatch, tokens), throughput)
    print(f"| {case} | {ubatch} | {tokens} | {float(value['avg_ns']) / 1e6:.3f} | "
          f"{throughput:.1f} | {speedup:.3f}x |")
PY

printf '%s\n' "$run_dir"
