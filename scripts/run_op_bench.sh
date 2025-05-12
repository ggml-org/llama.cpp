#!/bin/bash

# run-flash-attn-bench.sh
# Wrapper script to run flash attention benchmarks

set -e

# Default parameters
OUTPUT_DIR="${OUTPUT_DIR:-bench_results}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# Test different head sizes
HEAD_SIZES="${HEAD_SIZES:-64,128}"
# Test different context lengths
KV_LENGTHS="${KV_LENGTHS:-4096,8192,16384}"
# Test different grouped-query factors
NR_VALUES="${NR_VALUES:-1,4}"
# Test different quantization types
QUANT_TYPES="${QUANT_TYPES:-f16,q8_0,q4_0}"
# Skip analysis step
SKIP_ANALYSIS="${SKIP_ANALYSIS:-false}"

# Display help information
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Run flash attention benchmarks for CPU backend with different head sizes and KV lengths."
    echo
    echo "Options:"
    echo "  -o, --output-dir DIR    Directory to save results (default: $OUTPUT_DIR)"
    echo "  -h, --head-sizes LIST   Comma-separated list of head sizes to test (default: $HEAD_SIZES)"
    echo "  -k, --kv-lengths LIST   Comma-separated list of KV lengths to test (default: $KV_LENGTHS)"
    echo "  -n, --nr-values LIST    Comma-separated list of nr values to test (default: $NR_VALUES)"
    echo "  -q, --quant-types LIST  Comma-separated list of quantization types to test (default: $QUANT_TYPES)"
    echo "  --skip-analysis         Skip data analysis step (default: $SKIP_ANALYSIS)"
    echo "  --help                  Show this help message"
    echo
    echo "Example:"
    echo "  $0 --head-sizes 64,128 --kv-lengths 4096,8192"
    echo
}

# Parse command line arguments
while [ $# -gt 0 ]; do
    case "$1" in
    -o | --output-dir)
        OUTPUT_DIR="$2"
        shift 2
        ;;
    --help)
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

# Check Python dependencies
check_python_deps() {
    python -c "import pandas" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Warning: pandas is not installed. Data analysis will be skipped."
        echo "To install pandas, run: pip install pandas"
        return 1
    fi
    return 0
}

# Create output directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

# Generate timestamp for unique filenames
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Using timestamp: $TIMESTAMP"

# Clean up non-directory files in the benchmark directory if it exists
if [ -d "$BENCH_DIR" ]; then
    echo "Cleaning up non-directory files in $BENCH_DIR"
    find "$BENCH_DIR" -type f -maxdepth 1 -delete
fi


echo "Creating benchmark directory: $BENCH_DIR"
mkdir -p "$BENCH_DIR"

# Path to test-backend-ops executable
TEST_BACKEND_OPS="${REPO_ROOT}/build/bin/test-backend-ops"
if [ ! -f "$TEST_BACKEND_OPS" ]; then
    echo "Error: test-backend-ops not found at $TEST_BACKEND_OPS"
    echo "Please build llama.cpp first with 'make test-backend-ops'"
    exit 1
fi

# Create unique filename for the benchmark results
BENCHMARK_FILE="${BENCH_DIR}/flash_attn_bench_${TIMESTAMP}.txt"
BENCHMARK_CSV_FILE="${BENCH_DIR}/flash_attn_bench_${TIMESTAMP}.csv"

# Run benchmarks
"$TEST_BACKEND_OPS" perf -o FLASH_ATTN_EXT -b CPU > "$BENCHMARK_FILE"

echo "=== Benchmark Complete ==="
echo "Results saved to $BENCHMARK_FILE"

# Run analysis script if available
if [ "$SKIP_ANALYSIS" = "false" ]; then
    echo "=== Running Data Analysis ==="
    ANALYSIS_SCRIPT="${SCRIPT_DIR}/summary_flash_attn.py"
    
    if [ -f "$ANALYSIS_SCRIPT" ]; then
        if check_python_deps; then
            echo "Running data analysis using $ANALYSIS_SCRIPT"
            python "$ANALYSIS_SCRIPT" --input "$BENCHMARK_FILE" --csv "$BENCHMARK_CSV_FILE"
            
            if [ $? -eq 0 ]; then
                echo "=== Data Analysis Complete ==="
                echo "Analysis results saved to the output directory"
            else
                echo "ERROR: Data analysis failed"
            fi
        else
            echo "Skipping data analysis due to missing Python dependencies."
        fi
    else
        echo "Warning: Analysis script not found at $ANALYSIS_SCRIPT"
        echo "Please make sure scripts/summary_flash_attn.py exists."
    fi
else
    echo "Skipping data analysis as requested."
fi

echo "=== All Operations Complete ==="