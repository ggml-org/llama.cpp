#!/bin/bash

# Configuration for Strix Halo iGPU Only when using an eGPU as primary
# We force Vulkan to ONLY see Device 1 (The 8060S iGPU)

# Model Path
#MODEL="./models/Qwen3-Next-80B-A3B-Thinking-Q4_K_M.gguf"
#MODEL="./models/UD-Q8_K_XL/Qwen3-Next-80B-A3B-Thinking-UD-Q8_K_XL-00001-of-00002.gguf"
#MODEL="./models/models/InternVL2_5-26B/model-00001-of-00011.safetensors"
#MODEL="./models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf"
MODEL="./models/Nemotron3-nano/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf"

# Performance Flags
# -ngl 99:    Put 100% of layers on the iGPU.
# --threads 16: Physical Core count (Best for latency).
# --ctx-size 32768: Massive context window (You have the RAM for it!).
# -no-cnv:    Disable conversation mode (Optional, remove if you want chat interface).

# If using eGPU
#HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-server \
#If using iGPU
#HIP_VISIBLE_DEVICES=1 HSA_OVERRIDE_GFX_VERSION=11.5.1 ./build-rocm/bin/llama-server \
#HSA_XNACK=1 HIP_VISIBLE_DEVICES=1 HSA_OVERRIDE_GFX_VERSION=11.5.1 
#HSA_ENABLE_SDMA=0 HIP_VISIBLE_DEVICES=1 HSA_OVERRIDE_GFX_VERSION=11.0.0 
HIP_VISIBLE_DEVICES=1 ./build-rocm/bin/llama-server \
  -m "$MODEL" \
  --n-gpu-layers 99 \
  --threads 16 \
  --ctx-size 8196 \
  -ctk q8_0 \
  -ctv q8_0 \
  --batch-size 512 \
  --temp 0.6 \
  --min-p 0.05 \
  --host 0.0.0.0 \
  --no-mmap
