#!/bin/bash
#
# Run llama-bench with AMD MI50 ROCm support and GFX906 optimizations
# Built for gfx906 architecture - matches SCRIPT_launch_server configuration
#

# Set ROCm environment variables for MI50 ONLY (optimal configuration)
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0           # ONLY MI50 (Device 0)
export CUDA_VISIBLE_DEVICES=0          # Additional CUDA compatibility
export ROCR_VISIBLE_DEVICES=0          # ROCr runtime device selection
export GGML_BACKEND_HIP=1
export HCC_AMDGPU_TARGET=gfx906

# Path to your model file - update this to your actual model path
MODEL_PATH="/home/iacopo/Downloads/Qwen3-30B-A3B-Thinking-2507-Q4_0.gguf"
# MODEL_PATH="/home/iacopo/Downloads/Qwen_Qwen3-4B-Instruct-2507-Q4_0.gguf"

# Default benchmark parameters (matching server configuration)
BENCH_PARAMS=(
    -m "$MODEL_PATH"
    -ngl 99                    # Offload all layers to GPU
    -b 1024                    # Batch size (matches server)
    -t $(nproc)                # Use all CPU threads
    -fa 1                     # Enable flash attention (GFX906 optimized
    -ctk q8_0                  # q8_0 quantized K cache (matches server)
    -ctv q8_0                  # q8_0 quantized V cache (matches server)
    -d 512                   
    --main-gpu 0               # Force MI50 as main GPU
    --progress                 # Show progress indicators
)

# Benchmark configurations
QUICK_TEST="-p 512 -n 128"                           # Quick test
STANDARD_TEST="-p 512,1024,2048,4096 -n 128"    # Standard comprehensive test
PROMPT_FOCUS="-p 512,1024,2048,4096,8192 -n 64"     # Focus on prompt processing
GENERATION_FOCUS="-p 512 -n 128,256,512,1024"       # Focus on text generation
EXTENSIVE_TEST="-p 512,1024,2048,4096,8192 -n 128,256,512"  # Extensive testing

# Function to display usage
usage() {
    echo "Usage: $0 [test_type] [additional_llama-bench_args...]"
    echo ""
    echo "Test types:"
    echo "  quick       - Quick test (512 prompt, 128 generation)"
    echo "  standard    - Standard test (multiple prompt sizes, 2 gen sizes) [DEFAULT]"
    echo "  prompt      - Focus on prompt processing (up to 8K prompts)"
    echo "  generation  - Focus on text generation (multiple lengths)"
    echo "  extensive   - Extensive testing (all combinations)"
    echo "  custom      - Use your own parameters (provide as additional args)"
    echo ""
    echo "Examples:"
    echo "  $0                          # Run standard benchmark"
    echo "  $0 quick                    # Run quick benchmark" 
    echo "  $0 prompt                   # Test prompt processing"
    echo "  $0 custom -p 1024 -n 256    # Custom benchmark"
    echo ""
    echo "Model path: $MODEL_PATH"
    echo "Output format: markdown (default), add -o csv for CSV output"
}

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at: $MODEL_PATH"
    echo "Please update MODEL_PATH in this script or ensure the model exists."
    exit 1
fi

# Parse command line arguments
TEST_TYPE="${1:-standard}"
shift  # Remove first argument, rest will be passed to llama-bench

case "$TEST_TYPE" in
    "help"|"-h"|"--help")
        usage
        exit 0
        ;;
    "quick")
        TEST_PARAMS="$QUICK_TEST"
        echo "=== Running Quick Benchmark ==="
        ;;
    "standard")
        TEST_PARAMS="$STANDARD_TEST"
        echo "=== Running Standard Benchmark ==="
        ;;
    "prompt")
        TEST_PARAMS="$PROMPT_FOCUS"
        echo "=== Running Prompt Processing Focused Benchmark ==="
        ;;
    "generation")
        TEST_PARAMS="$GENERATION_FOCUS"
        echo "=== Running Text Generation Focused Benchmark ==="
        ;;
    "extensive")
        TEST_PARAMS="$EXTENSIVE_TEST"
        echo "=== Running Extensive Benchmark (this will take a while) ==="
        ;;
    "custom")
        TEST_PARAMS=""
        echo "=== Running Custom Benchmark ==="
        echo "Custom parameters: $@"
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        usage
        exit 1
        ;;
esac

# Display system info
echo "Model: $(basename "$MODEL_PATH")"
echo "GPU: MI50 (gfx906) - Device 0 only"
echo "Flash Attention: ENABLED (GFX906 optimized)"
echo "KV Cache: q8_0 quantized"
echo ""

# Display GPU info
echo "=== ROCm GPU Information ==="
rocm-smi --showproductname --showtemp --showmeminfo --showuse --showpower
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if llama-bench exists
if [ ! -f "./build/bin/llama-bench" ]; then
    echo "Error: llama-bench not found. Please compile the project first:"
    echo "  ./SCRIPT_compile_MI50.sh"
    exit 1
fi

# Run the benchmark
echo "=== Starting llama-bench with GFX906 Flash Attention ==="
echo "Command: ./build/bin/llama-bench ${BENCH_PARAMS[*]} $TEST_PARAMS $@"
echo ""

./build/bin/llama-bench "${BENCH_PARAMS[@]}" $TEST_PARAMS "$@"

BENCH_EXIT_CODE=$?

echo ""
echo "=== Benchmark Complete ==="
if [ $BENCH_EXIT_CODE -eq 0 ]; then
    echo "✓ Benchmark completed successfully"
    echo ""
    echo "Tip: Add '-o csv' to get CSV output for analysis"
    echo "Tip: Add '-r 10' to run more repetitions for better accuracy"
else
    echo "✗ Benchmark failed with exit code: $BENCH_EXIT_CODE"
fi

exit $BENCH_EXIT_CODE
