#!/usr/bin/env bash
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-./build/bin/llama-cli}"
MODEL="${MODEL:-$HOME/dev/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf}"
PROMPTS_DIR="${PROMPTS_DIR:-moe_exp/prompts_v2}"
RUNS_DIR="${RUNS_DIR:-moe_exp/runs_v2}"

N_PREDICT="${N_PREDICT:-512}"
CTX="${CTX:-4096}"
NGL="${NGL:-999}"
SEEDS="${SEEDS:-1 2}"

mkdir -p "$RUNS_DIR"

commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

for prompt_path in "$PROMPTS_DIR"/*.txt; do
    prompt_name="$(basename "$prompt_path" .txt)"

    for seed in $SEEDS; do
        run_name="${prompt_name}_seed${seed}"
        out_dir="$RUNS_DIR/$run_name"
        mkdir -p "$out_dir"

        trace_path="$out_dir/moe_trace.bin"
        stdout_path="$out_dir/stdout.txt"
        stderr_path="$out_dir/stderr.txt"
        meta_path="$out_dir/metadata.txt"

        echo "== running $run_name =="

        cat > "$meta_path" <<EOF
run_name=$run_name
prompt_name=$prompt_name
prompt_path=$prompt_path
model=$MODEL
llama_bin=$LLAMA_BIN
llama_commit=$commit
n_predict=$N_PREDICT
ctx=$CTX
ngl=$NGL
seed=$seed
trace_path=$trace_path
EOF

        LLAMA_MOE_TRACE="$trace_path" \
        "$LLAMA_BIN" \
            -m "$MODEL" \
            -ngl "$NGL" \
            -c "$CTX" \
            -n "$N_PREDICT" \
            --seed "$seed" \
            --single-turn \
            -p "$(cat "$prompt_path")" \
            > "$stdout_path" \
            2> "$stderr_path"

        if [[ ! -s "$trace_path" ]]; then
            echo "ERROR: empty trace: $trace_path" >&2
            exit 1
        fi

        ls -lh "$trace_path"
    done
done