#!/bin/bash
#
# Launch llama.cpp server with AMD MI50 ROCm support
# Built for gfx906 architecture
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

PARAMS=(
    -m "$MODEL_PATH"
    -ngl 99                    # Offload all layers to GPU
    -c 32000                    # Context size
    -np 1                      # Parallel requests
    -t $(nproc)                # Use all CPU threads
    --port 8090                # Server port
    --host 0.0.0.0            # Listen on all interfaces
    #--mlock                    # Lock model in memory
    #--no-mmap                  # Don't use memory mapping
    -b 512                       # Batch size
    #--cont-batching            # Enable continuous batching
    --flash-attn on              # Enable flash attention
    --cache-type-k q8_0        # q8_0 quantized K cache (50% memory savings)
    --cache-type-v q8_0        # q8_0 quantized V cache (50% memory savings)
    --main-gpu 0               # Force MI50 as main GPU
    --device "ROCm0"           # Explicit ROCm device
    # --no-warmup                # Skip warmup for consistent profiling
)

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at: $MODEL_PATH"
    echo "Usage: $0 [model_path] [additional_args...]"
    echo ""
    echo "Example: $0 ./models/llama-2-7b-chat.q4_0.gguf --ctx-size 8192"
    exit 1
fi

# Display GPU info
echo "=== ROCm GPU Information ==="
rocm-smi --showproductname --showtemp --showmeminfo --showuse --showpower
echo ""

# Launch llama.cpp server
echo "=== Launching llama.cpp server with MI50 optimization ==="
echo "Model: $MODEL_PATH"
echo "GPU: MI50 (gfx906)"
echo "Server will be available at: http://localhost:8080"
echo "Parameters: ${PARAMS[*]} ${@:2}"
echo ""

cd "$(dirname "$0")"
./build/bin/llama-server "${PARAMS[@]}" "${@:2}"
