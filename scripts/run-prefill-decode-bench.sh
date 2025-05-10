#!/bin/bash

# run-prefill-decode-bench.sh
# Simple wrapper script to run prefill-decode benchmarks

set -e

# Default parameters
MODEL="${MODEL:-/Volumes/zijiessd/gguf/Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf}"
THREADS="${THREADS:-12}"
REPETITIONS="${REPETITIONS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-bench_results}"
GEN_TOKENS="${GEN_TOKENS:-128}"
# Define context depths to test
DEPTHS="${DEPTHS:-1024,2048,4096}"

# Display help information
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Run prefill-decode benchmarks for CPU and GPU backends with different prefill depths."
    echo
    echo "Options:"
    echo "  -m, --model PATH        Path to the model (default: $MODEL)"
    echo "  -t, --threads N         Number of threads to use (default: $THREADS)"
    echo "  -r, --repetitions N     Number of benchmark repetitions (default: $REPETITIONS)"
    echo "  -o, --output-dir DIR    Directory to save results (default: $OUTPUT_DIR)"
    echo "  -g, --gen-tokens N      Number of tokens to generate (default: $GEN_TOKENS)"
    echo "  -d, --depths LIST       Comma-separated list of prefill depths to test (default: $DEPTHS)"
    echo "  -h, --help              Show this help message"
    echo
    echo "Example:"
    echo "  $0 --model models/7B/ggml-model-q4_0.gguf --threads 16 --repetitions 5"
    echo
}

# Parse command line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        -r|--repetitions)
            REPETITIONS="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -g|--gen-tokens)
            GEN_TOKENS="$2"
            shift 2
            ;;
        -d|--depths)
            DEPTHS="$2"
            shift 2
            ;;
        -h|--help)
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

# Run benchmarks
echo "=== Starting Prefill-Decode Benchmarks ==="
echo "Model: $MODEL"
echo "Threads: $THREADS"
echo "Repetitions: $REPETITIONS"
echo "Output directory: $MODEL_OUTPUT_DIR"
echo "Generate tokens: $GEN_TOKENS"
echo "Testing depths: $DEPTHS"
echo

# Convert depths string to array
IFS=',' read -r -a DEPTHS_ARRAY <<< "$DEPTHS"

# Build path to llama-bench
LLAMA_BENCH="${REPO_ROOT}/build/bin/llama-bench"
if [ ! -f "$LLAMA_BENCH" ]; then
    echo "Error: llama-bench not found at $LLAMA_BENCH"
    echo "Please build llama.cpp first with 'make llama-bench'"
    exit 1
fi

# Create header for CPU benchmark results
CPU_BENCHMARK_FILE="${MODEL_OUTPUT_DIR}/prefill_decode_CPU_${TIMESTAMP}.md"
echo "# Prefill-Decode Benchmark for CPU - $(date)" > "$CPU_BENCHMARK_FILE"
echo "Model: $MODEL" >> "$CPU_BENCHMARK_FILE"
echo "Generate tokens: $GEN_TOKENS" >> "$CPU_BENCHMARK_FILE"
echo "Threads: $THREADS" >> "$CPU_BENCHMARK_FILE"
echo "Repetitions: $REPETITIONS" >> "$CPU_BENCHMARK_FILE"
echo "Timestamp: $TIMESTAMP" >> "$CPU_BENCHMARK_FILE"
echo "" >> "$CPU_BENCHMARK_FILE"

# Run CPU benchmarks for each depth
for DEPTH in "${DEPTHS_ARRAY[@]}"; do
    echo "Testing CPU with prefill depth: $DEPTH"
    
    # Add section header
    echo "## Prefill depth: $DEPTH tokens" >> "$CPU_BENCHMARK_FILE"
    
    # Run the benchmark and append results
    "$LLAMA_BENCH" \
        -m "$MODEL" \
        -t "$THREADS" \
        -r "$REPETITIONS" \
        -p "$DEPTH" \
        -n "$GEN_TOKENS" \
        -o "md" >> "$CPU_BENCHMARK_FILE"
    
    # Add build info
    git_hash=$(cd "$REPO_ROOT" && git rev-parse --short HEAD)
    build_number=$(cd "$REPO_ROOT" && git rev-list --count HEAD)
    echo "" >> "$CPU_BENCHMARK_FILE"
    echo "build: $git_hash ($build_number)" >> "$CPU_BENCHMARK_FILE"
    echo "" >> "$CPU_BENCHMARK_FILE"
done

echo "=== Benchmark Complete ==="
echo "Results saved to $MODEL_OUTPUT_DIR as Markdown files:"
ls -la "$MODEL_OUTPUT_DIR"/prefill_decode_*_${TIMESTAMP}.md

# Run the extraction script to generate CSV
echo "=== Generating CSV Summary ==="
if [ -f "$SCRIPT_DIR/extract_bench_results.py" ]; then
    python "$SCRIPT_DIR/extract_bench_results.py" --dir "$MODEL_OUTPUT_DIR" --output "$MODEL_OUTPUT_DIR/${MODEL_NAME}_summary.csv"
    echo "Summary CSV generated at: $MODEL_OUTPUT_DIR/${MODEL_NAME}_summary.csv"
else
    echo "Warning: extract_bench_results.py not found in $SCRIPT_DIR"
fi 