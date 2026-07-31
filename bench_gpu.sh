#!/bin/bash
MODEL="/mnt/p5/models/Qwen3.5-35B-A3B-abliterated-Q4_K_M.gguf"
PROMPT="Explain quantum physics in simple terms for a 10 year old."

run_bench() {
    NGL=$1
    NCMOE=$2
    CTX=$3
    echo "=========================================="
    echo "Testing: -ngl $NGL --n-cpu-moe $NCMOE -c $CTX"
    echo "=========================================="
    
    pkill -9 llama-cli 2>/dev/null || true
    sleep 1
    
    out=$(RECURRENT_D=12 ./build/bin/llama-cli -m "$MODEL" -p "$PROMPT" -n 32 -ngl $NGL --n-cpu-moe $NCMOE -c $CTX -t 12 2>&1)
    
    echo "$out" | grep -E "eval time|llama_model_load:|allocating" | tail -n 10
    echo ""
}

echo "=== BENCHMARKING GPU OFFLOAD CONFIGURATIONS ==="
run_bench 12 64 8192
run_bench 20 64 8192
run_bench 28 64 8192
run_bench 20 40 8192
run_bench 24 48 8192
