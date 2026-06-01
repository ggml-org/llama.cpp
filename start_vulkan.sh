#!/bin/bash

# Vulkan Configuration
# Ensure we use the Strix Halo (8060S) which should be device 0 or 1
# You can check with `vulkaninfo` but usually it just works.
#export GGML_VK_VISIBLE_DEVICES=1

# Model Path
MODEL="./models/UD-Q8_K_XL/Qwen3-Next-80B-A3B-Thinking-UD-Q8_K_XL-00001-of-00002.gguf"
MODEL="./models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf"

echo "Starting Strix Halo (Vulkan Backend)..."
#  --ctx-size 32768 \

# Note: We are using the bin/llama-server from the NEW build-vulkan folder
GGML_VK_VISIBLE_DEVICES=0 ./build-vulkan/bin/llama-server \
  -m "$MODEL" \
  --n-gpu-layers 99 \
  --threads 16 \
  --ctx-size 8196 \
  --batch-size 512 \
  --host 0.0.0.0
