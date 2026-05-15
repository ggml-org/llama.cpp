#!/usr/bin/env bash
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-./build/bin/llama-cli}"
MODEL="${MODEL:-$HOME/dev/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf}"
PROMPTS_DIR="${PROMPTS_DIR:-moe_exp/prompts}"
RUNS_DIR="${RUNS_DIR:-moe_exp/runs}"

N_PREDICT="${N_PREDICT:-256}"
CTX="${CTX:-4096}"
NGL="${NGL:-999}"
SEED="${SEED:-1}"

mkdir -p "$RUNS_DIR"

commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

for prompt_path in "$PROMPTS_DIR"/*.txt; do
    name="$(basename "$prompt_path" .txt)"
    out_dir="$RUNS_DIR/$name"
    mkdir -p "$out_dir"

    trace_path="$out_dir/moe_trace.bin"
    stdout_path="$out_dir/stdout.txt"
    stderr_path="$out_dir/stderr.txt"
    meta_path="$out_dir/metadata.txt"

    echo "== running $name =="

    cat > "$meta_path" <<EOF
name=$name
prompt_path=$prompt_path
model=$MODEL
llama_bin=$LLAMA_BIN
llama_commit=$commit
n_predict=$N_PREDICT
ctx=$CTX
ngl=$NGL
seed=$SEED
trace_path=$trace_path
EOF

    LLAMA_MOE_TRACE="$trace_path" \
    "$LLAMA_BIN" \
        -m "$MODEL" \
        -ngl "$NGL" \
        -c "$CTX" \
        -n "$N_PREDICT" \
        --seed "$SEED" \
        -p "$(cat "$prompt_path")" \
        > "$stdout_path" \
        2> "$stderr_path"

    ls -lh "$trace_path"
done