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

threads=${PREFILL_THREADS:-25}
repetitions=${PREFILL_REPETITIONS:-3}
read -r -a prompts <<< "${ATTENTION_PROMPTS:-512 2048 8192}"
read -r -a ubatches <<< "${ATTENTION_UBATCHES:-2048 8192}"
selected=,${ATTENTION_CASES:-all},
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-attention-stages"
mkdir -p "$run_dir"

nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
printf 'label\tprompt\tubatch\tenvironment\tresult\n' > "$run_dir/manifest.tsv"

cases=(
    'baseline|'
    'direct-causal|LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1'
    'q-rope|LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1 LLAMA_CUDA_FATTN_Q_ROPE=1'
    'causal-tiles|LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1 LLAMA_CUDA_FATTN_Q_ROPE=1 GGML_CUDA_FATTN_CAUSAL_TILES=1'
    'sm120-causal|LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1 LLAMA_CUDA_FATTN_Q_ROPE=1 GGML_CUDA_FATTN_SM120_CAUSAL_SCHEDULE=1'
)

run_selected() {
    local label=$1
    [[ "$selected" == ',all,' || "$selected" == *",$label,"* ]]
}

for entry in "${cases[@]}"; do
    label=${entry%%|*}
    environment=${entry#*|}
    if ! run_selected "$label"; then
        continue
    fi
    read -r -a env_args <<< "$environment"

    for prompt in "${prompts[@]}"; do
        for ubatch in "${ubatches[@]}"; do
            if ((ubatch > prompt)); then
                continue
            fi
            result="$label-pp$prompt-ub$ubatch.jsonl"
            env GGML_CUDA_DISABLE_GRAPHS=1 LLAMA_KQ_MASK_CONTIGUOUS_LOG=1 LLAMA_CUDA_FATTN_DIRECT_CAUSAL_LOG=1 GGML_CUDA_FATTN_LOG_CONFIG=1 \
                "${env_args[@]}" "$bench" \
                -m "$model" \
                -p "$prompt" \
                -n 0 \
                -r "$repetitions" \
                -t "$threads" \
                -ngl 999 \
                -b 8192 \
                -ub "$ubatch" \
                -fa on \
                -o jsonl \
                "${extra_args[@]}" \
                > "$run_dir/$result" \
                2> "$run_dir/$label-pp$prompt-ub$ubatch.stderr"
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$label" "$prompt" "$ubatch" "$environment" "$result" >> "$run_dir/manifest.tsv"
        done
    done
done

for label in direct-causal q-rope causal-tiles sm120-causal; do
    if run_selected "$label" && ! grep -q 'FlashAttention: direct-causal=1' "$run_dir/$label-"*.stderr; then
        echo "$label did not select direct causal attention" >&2
        exit 1
    fi
done
for label in q-rope causal-tiles sm120-causal; do
    if run_selected "$label" && ! grep -q 'FlashAttention: q-rope=1' "$run_dir/$label-"*.stderr; then
        echo "$label did not select Q RoPE fusion" >&2
        exit 1
    fi
done
if run_selected sm120-causal &&
        ! grep -q 'FlashAttention: sm120-causal=1' "$run_dir/sm120-causal-"*.stderr; then
    echo "sm120-causal did not select the SM120 schedule" >&2
    exit 1
fi

printf '%s\n' "$run_dir"
