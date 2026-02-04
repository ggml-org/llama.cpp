#!/bin/bash
cat << 'EOF'

   ██╗     ██╗      █████╗ ███╗   ███╗ █████╗    ██████╗██████╗ ██████╗
   ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗  ██╔════╝██╔══██╗██╔══██╗
   ██║     ██║     ███████║██╔████╔██║███████║  ██║     ██████╔╝██████╔╝
   ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║  ██║     ██╔═══╝ ██╔═══╝
   ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║  ╚██████╗██║     ██║
   ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═════╝╚═╝     ╚═╝
            ██████╗ ███████╗██╗  ██╗ █████╗  ██████╗  ██████╗
           ██╔════╝ ██╔════╝╚██╗██╔╝██╔══██╗██╔═████╗██╔════╝
           ██║  ███╗█████╗   ╚███╔╝ ╚██████║██║██╔██║███████╗
           ██║   ██║██╔══╝   ██╔██╗  ╚═══██║████╔╝██║██╔═══██╗
           ╚██████╔╝██║     ██╔╝ ██╗ █████╔╝╚██████╔╝╚██████╔╝
            ╚═════╝ ╚═╝     ╚═╝  ╚═╝ ╚════╝  ╚═════╝  ╚═════╝


EOF

export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0
MODEL_PATH="/media/iacoppbk/80F42C9BF42C96061/llms/Qwen3-VL-30B-A3B-Thinking-Q4_1.gguf"

LOG_FILE="bench_results.md"


BENCH_PARAMS=(
    -m "$MODEL_PATH"       # Model path
    -ngl 99                # Number of GPU layers (all on GPU)
    -t $(nproc)            # Number of CPU threads
    -fa 1                  # Flash attention (1=on, 0=off)
    -ctk q8_0              # KV cache key type (q8_0 quantization)
    -ctv f16               # KV cache value type (f16 precision)
    --main-gpu 0           # Main GPU device ID
    --progress             # Show progress during benchmark
    -r 1                   # Number of repetitions
    -b 2048                # Batch size
    -ub 2048               # Micro-batch size
    #-d 8192               # Context size 
)


#BENCH_TESTS="-p 0 -n 2048"
#BENCH_TESTS="-p 2048 -n 0"
BENCH_TESTS="-p 512 -n 1,128,2048"

echo "=== Benchmark ==="
echo "Model: $(basename "$MODEL_PATH")"
echo ""

cd "$(dirname "$0")"
[ ! -f "./build/bin/llama-bench" ] && echo "Error: llama-bench not found" && exit 1

./build/bin/llama-bench "${BENCH_PARAMS[@]}" $BENCH_TESTS "$@" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "Output saved to: $LOG_FILE"
