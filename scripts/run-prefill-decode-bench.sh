#!/bin/bash

# run-prefill-decode-bench.sh
# Simple wrapper script to run prefill-decode benchmarks

set -e

# Default parameters
# Check if we're on a Jetson platform

if command -v jetson_release >/dev/null 2>&1 && jetson_release >/dev/null 2>&1; then
    #> Jetson platform
    MODEL="${MODEL:-/datasets/gguf/Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf}"
else
    #> Apple platform (default)
    MODEL="${MODEL:-/Volumes/zijiessd/gguf/Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf}"
fi

THREADS="${THREADS:-12}"
REPETITIONS="${REPETITIONS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-bench_results}"
GEN_TOKENS="${GEN_TOKENS:-128}"
# Define context depths to test
DEPTHS="${DEPTHS:-1024,2048,4096}"
# Define KV cache types to test
KV_CACHE_TYPES="${KV_CACHE_TYPES:-f16,q8_0,q4_0}"
# Flag for forced alignment
FORCED_ALIGNMENT="${FORCED_ALIGNMENT:-1}"
# Prompt length
N_PROMPT="${N_PROMPT:-1024}"
# Number of GPU layers
NUM_GPU_LAYERS="${NUM_GPU_LAYERS:-0}"
# Flag to skip data processing
SKIP_ANALYSIS="${SKIP_ANALYSIS:-false}"

# Display help information
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Run prefill-decode benchmarks for CPU and GPU backends with different prefill depths and KV cache types."
    echo
    echo "Options:"
    echo "  -m, --model PATH        Path to the model (default: $MODEL)"
    echo "  -t, --threads N         Number of threads to use (default: $THREADS)"
    echo "  -r, --repetitions N     Number of benchmark repetitions (default: $REPETITIONS)"
    echo "  -o, --output-dir DIR    Directory to save results (default: $OUTPUT_DIR)"
    echo "  -g, --gen-tokens N      Number of tokens to generate (default: $GEN_TOKENS)"
    echo "  -d, --depths LIST       Comma-separated list of prefill depths to test (default: $DEPTHS)"
    echo "  -k, --kv-cache-types LIST  Comma-separated list of KV cache types to test (default: $KV_CACHE_TYPES)"
    echo "                          Allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1"
    echo "  -p, --n-prompt N        Prompt length in tokens (default: $N_PROMPT)"
    echo "  -f, --forced-alignment N   Force KV cache alignment (default: $FORCED_ALIGNMENT)"
    echo "  --skip-analysis         Skip data analysis step (default: $SKIP_ANALYSIS)"
    echo "  -h, --help              Show this help message"
    echo
    echo "Example:"
    echo "  $0 --model models/7B/ggml-model-q4_0.gguf --threads 16 --kv-cache-types f16,q4_0,q8_0"
    echo
}

# Parse command line arguments
while [ $# -gt 0 ]; do
    case "$1" in
    -m | --model)
        MODEL="$2"
        shift 2
        ;;
    -t | --threads)
        THREADS="$2"
        shift 2
        ;;
    -r | --repetitions)
        REPETITIONS="$2"
        shift 2
        ;;
    -o | --output-dir)
        OUTPUT_DIR="$2"
        shift 2
        ;;
    -g | --gen-tokens)
        GEN_TOKENS="$2"
        shift 2
        ;;
    -d | --depths)
        DEPTHS="$2"
        shift 2
        ;;
    -k | --kv-cache-types)
        KV_CACHE_TYPES="$2"
        shift 2
        ;;
    -p | --n-prompt)
        N_PROMPT="$2"
        shift 2
        ;;
    -f | --forced-alignment)
        FORCED_ALIGNMENT="$2"
        shift 2
        ;;
    -ngl | --num-gpu-layers)
        NUM_GPU_LAYERS="$2"
        shift 2
        ;;
    --skip-analysis)
        SKIP_ANALYSIS=true
        shift
        ;;
    -h | --help)
        show_help
        exit 0
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
    esac
done

# 检查Python依赖
check_python_deps() {
    python -c "import pandas" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Warning: pandas is not installed. Data analysis will be skipped."
        echo "To install pandas, run: pip install pandas"
        return 1
    fi
    return 0
}

# Extract model name for folder creation
MODEL_BASENAME=$(basename "$MODEL")
MODEL_NAME="${MODEL_BASENAME%.*}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create model-specific output directory
MODEL_OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}/${MODEL_NAME}"
echo "Creating model directory: $MODEL_OUTPUT_DIR"

# Clean/create the model-specific directory
rm -rf "$MODEL_OUTPUT_DIR"
mkdir -p "$MODEL_OUTPUT_DIR"

# Generate timestamp for unique filenames
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Using timestamp: $TIMESTAMP"

# Run benchmarks
echo "=== Starting Prefill-Decode Benchmarks ==="
echo "Model: $MODEL"
echo "Threads: $THREADS"
echo "Repetitions: $REPETITIONS"
echo "Output directory: $MODEL_OUTPUT_DIR"
echo "Generate tokens: $GEN_TOKENS"
echo "Testing depths: $DEPTHS"
echo "Testing KV cache types: $KV_CACHE_TYPES"
echo "Prompt length: $N_PROMPT"
echo

# Convert depths string to array
IFS=',' read -r -a DEPTHS_ARRAY <<<"$DEPTHS"

# Convert KV cache types string to array
IFS=',' read -r -a KV_CACHE_TYPES_ARRAY <<<"$KV_CACHE_TYPES"

# Build path to llama-bench
LLAMA_BENCH="${REPO_ROOT}/build/bin/llama-bench"
if [ ! -f "$LLAMA_BENCH" ]; then
    echo "Error: llama-bench not found at $LLAMA_BENCH"
    echo "Please build llama.cpp first with 'make llama-bench'"
    exit 1
fi

# Run benchmarks for each KV cache type
for KV_TYPE in "${KV_CACHE_TYPES_ARRAY[@]}"; do
    echo "=== Testing KV cache type: $KV_TYPE ==="
    # Create benchmark file for this KV cache type
    BENCHMARK_FILE="${MODEL_OUTPUT_DIR}/prefill_decode_${KV_TYPE}_${TIMESTAMP}.csv"
    
    # Run the benchmark with all depths at once for this KV cache type
    echo "Testing KV cache type $KV_TYPE with prefill depths: $DEPTHS"
    
    echo "Running benchmark with the following parameters:"
    echo "  Model: $MODEL"
    echo "  Threads: $THREADS"
    echo "  Repetitions: $REPETITIONS"
    echo "  Depths: $DEPTHS"
    echo "  Generate tokens: $GEN_TOKENS"
    echo "  Prompt length: $N_PROMPT"
    echo "  Forced alignment: $FORCED_ALIGNMENT"
    echo "  KV cache type: $KV_TYPE"
    echo "  Number of GPU layers: $NUM_GPU_LAYERS"
    echo "  Output format: csv"
    echo "  Output file: $BENCHMARK_FILE"
    echo
    
    "$LLAMA_BENCH" \
        -m "$MODEL" \
        -t "$THREADS" \
        -r "$REPETITIONS" \
        -d "$DEPTHS" \
        -n "$GEN_TOKENS" \
        -p "$N_PROMPT" \
        -fa "$FORCED_ALIGNMENT" \
        -ctk "$KV_TYPE" \
        -ctv "$KV_TYPE" \
        -ngl "$NUM_GPU_LAYERS" \
        -o "csv" >> "$BENCHMARK_FILE"
done

echo "=== Benchmark Complete ==="
echo "Results saved to $MODEL_OUTPUT_DIR as CSV files:"
ls -la "$MODEL_OUTPUT_DIR"/prefill_decode_*_${TIMESTAMP}.csv

# 运行分析脚本
if [ "$SKIP_ANALYSIS" = "false" ]; then
    echo "=== Running Data Analysis ==="
    ANALYSIS_SCRIPT="${SCRIPT_DIR}/analyze_benchmark_results.py"
    
    if [ -f "$ANALYSIS_SCRIPT" ]; then
        if check_python_deps; then
            echo "Running data analysis using $ANALYSIS_SCRIPT"
            python "$ANALYSIS_SCRIPT" --dir "$MODEL_OUTPUT_DIR"
            
            if [ $? -eq 0 ]; then
                echo "=== Data Analysis Complete ==="
                echo "Generated analysis files:"
                echo "  ${MODEL_OUTPUT_DIR}/prefill_performance_pivot.csv"
                echo "  ${MODEL_OUTPUT_DIR}/prefill_by_depth_pivot.csv"
                echo "  ${MODEL_OUTPUT_DIR}/decode_performance_pivot.csv"
                echo "  ${MODEL_OUTPUT_DIR}/decode_by_depth_pivot.csv"
            else
                echo "ERROR: Data analysis failed"
            fi
        else
            echo "Skipping data analysis due to missing Python dependencies."
        fi
    else
        echo "Warning: Analysis script not found at $ANALYSIS_SCRIPT"
        echo "Please make sure scripts/analyze_benchmark_results.py exists."
    fi
else
    echo "Skipping data analysis as requested."
fi

echo "=== All Operations Complete ==="
