#!/bin/bash

# run-breakdown.sh
# Script to run operator breakdown profiling with different prefill depths

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
OUTPUT_DIR="${OUTPUT_DIR:-breakdown_results}"
GEN_TOKENS="${GEN_TOKENS:-16}"
# Define context depths to test (1k, 2k, 4k, 8k, 16k, 32k, 64k)
DEPTHS="${DEPTHS:-1024,2048,4096,8192,16384,32768,65536}"
# Flag for forced alignment
FORCED_ALIGNMENT="${FORCED_ALIGNMENT:-1}"
# Prompt length (0 means use the depth as prompt length)
N_PROMPT="${N_PROMPT:-0}"
# Number of GPU layers
NUM_GPU_LAYERS="${NUM_GPU_LAYERS:-0}"

# Display help information
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Run operator breakdown profiling for different prefill depths."
    echo
    echo "Options:"
    echo "  -m, --model PATH        Path to the model (default: $MODEL)"
    echo "  -t, --threads N         Number of threads to use (default: $THREADS)"
    echo "  -o, --output-dir DIR    Directory to save results (default: $OUTPUT_DIR)"
    echo "  -g, --gen-tokens N      Number of tokens to generate (default: $GEN_TOKENS)"
    echo "  -d, --depths LIST       Comma-separated list of prefill depths to test (default: $DEPTHS)"
    echo "  -p, --n-prompt N        Prompt length in tokens (default: $N_PROMPT, 0 means use depth as prompt length)"
    echo "  -f, --forced-alignment N   Force KV cache alignment (default: $FORCED_ALIGNMENT)"
    echo "  -ngl, --num-gpu-layers N   Number of GPU layers (default: $NUM_GPU_LAYERS)"
    echo "  -h, --help              Show this help message"
    echo
    echo "Example:"
    echo "  $0 --model models/7B/ggml-model-q4_0.gguf --threads 16 --depths 1024,2048,4096"
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

# Convert depths string to array
IFS=',' read -r -a DEPTHS_ARRAY <<<"$DEPTHS"

# Build path to llama-bench
LLAMA_BENCH="${REPO_ROOT}/build-arm64/bin/llama-bench"
if [ ! -f "$LLAMA_BENCH" ]; then
    echo "Error: llama-bench not found at $LLAMA_BENCH"
    echo "Please build llama.cpp first with 'make llama-bench'"
    exit 1
fi

echo "=== Starting Operator Breakdown Profiling ==="
echo "Model: $MODEL"
echo "Threads: $THREADS"
echo "Output directory: $MODEL_OUTPUT_DIR"
echo "Generate tokens: $GEN_TOKENS"
echo "Testing depths: $DEPTHS"
echo "Prompt length: $N_PROMPT (0 means use depth value)"
echo "Forced alignment: $FORCED_ALIGNMENT"
echo "Number of GPU layers: $NUM_GPU_LAYERS"
echo

# Run benchmarks for each depth
for DEPTH in "${DEPTHS_ARRAY[@]}"; do
    echo "=== Testing depth: $DEPTH ==="
    
    # Create results file for this depth
    RESULTS_FILE="${MODEL_OUTPUT_DIR}/breakdown_${DEPTH}.csv"
    
    # Set prompt length equal to depth if N_PROMPT is 0
    PROMPT_LENGTH="$N_PROMPT"
    if [ "$PROMPT_LENGTH" -eq 0 ]; then
        PROMPT_LENGTH="$DEPTH"
    fi
    
    echo "Running profile with the following parameters:"
    echo "  Model: $MODEL"
    echo "  Threads: $THREADS"
    echo "  Depth: $DEPTH"
    echo "  Generate tokens: $GEN_TOKENS"
    echo "  Prompt length: $PROMPT_LENGTH"
    echo "  Forced alignment: $FORCED_ALIGNMENT"
    echo "  Number of GPU layers: $NUM_GPU_LAYERS"
    echo "  Output file: $RESULTS_FILE"
    echo
    
    # Set GGML_GRAPH_PROFILE to output file and run llama-bench for a single depth
    # We're using GGML_GRAPH_PROFILE to capture operator breakdown
    echo "Running command: GGML_GRAPH_PROFILE=$RESULTS_FILE \"$LLAMA_BENCH\" -m \"$MODEL\" -t \"$THREADS\" -r 1 -d \"$DEPTH\" -n \"$GEN_TOKENS\" -p \"$PROMPT_LENGTH\" -fa \"$FORCED_ALIGNMENT\" -ngl \"$NUM_GPU_LAYERS\""
    
    GGML_GRAPH_PROFILE=$RESULTS_FILE "$LLAMA_BENCH" \
        -m "$MODEL" \
        -t "$THREADS" \
        -r 1 \
        -d "$DEPTH" \
        -n "$GEN_TOKENS" \
        -p 0 \
        -fa "$FORCED_ALIGNMENT" \
        -ngl "$NUM_GPU_LAYERS"
    
    echo "Profile for depth $DEPTH saved to $RESULTS_FILE"
done

echo "=== Profiling Complete ==="
echo "Results saved to $MODEL_OUTPUT_DIR as CSV files:"
ls -la "$MODEL_OUTPUT_DIR"/breakdown_*_${TIMESTAMP}.csv

echo "=== All Operations Complete ===" 