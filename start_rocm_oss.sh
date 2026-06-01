#!/bin/bash

# ==============================================================================
# HYBRID CONFIG: R9700 (Device 0) + Strix Halo (Device 1)
# ==============================================================================

export HSA_ENABLE_SDMA=0

# Path to the first split file
MODEL="./models/openai_gpt-oss-120b-Q4_K_M/openai_gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"

# Make both GPUs visible
export HIP_VISIBLE_DEVICES=0,1

echo "Starting Server..."
echo " - Model: $MODEL"
echo " - Split: Layer Mode (Required for eGPU)"
echo " - Main GPU: Device 1 (Strix Halo, 128GB)"

# NOTE: Do not add comments inside the command block below!
./build-rocm/bin/llama-server \
  -m "$MODEL" \
  --host 0.0.0.0 \
  --port 8080 \
  --threads 16 \
  --n-gpu-layers 999 \
  --no-mmap \
  --split-mode layer \
  --main-gpu 1 \
  --tensor-split 1,4 \
  --ctx-size 32768 \
  --batch-size 512 \
  -ctk q8_0 \
  -ctv q8_0 \
  --temp 0.6 \
  --min-p 0.05
