#!/bin/bash

set -e

echo "=== llama.cpp GFX906 Benchmark Script ==="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Update submodules
echo -e "${GREEN}[1/4] Updating submodules...${NC}"
git submodule update --init --recursive

# Step 2: Build the project with GFX906 support
echo -e "${GREEN}[2/4] Building llama.cpp with GFX906 support...${NC}"
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Step 3: Check for model and download if needed
MODEL_DIR="models"
MODEL_FILE="llama-2-7b.Q4_0.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
MODEL_URL="https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf"

echo -e "${GREEN}[3/4] Checking for model...${NC}"

# Create models directory if it doesn't exist
mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_PATH}" ]; then
    echo -e "${YELLOW}Model already exists at ${MODEL_PATH}${NC}"
else
    echo -e "${YELLOW}Model not found. Downloading from Hugging Face...${NC}"
    echo "This may take a while (~3.8 GB file)..."
    
    # Download with wget (with progress bar)
    if command -v wget &> /dev/null; then
        wget -c "${MODEL_URL}" -O "${MODEL_PATH}" --show-progress
    # Fallback to curl if wget is not available
    elif command -v curl &> /dev/null; then
        curl -L "${MODEL_URL}" -o "${MODEL_PATH}" --progress-bar
    else
        echo -e "${RED}Error: Neither wget nor curl found. Please install one of them.${NC}"
        exit 1
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Model downloaded successfully!${NC}"
    else
        echo -e "${RED}Error downloading model. Please check your internet connection.${NC}"
        exit 1
    fi
fi

# Step 4: Run benchmark
echo -e "${GREEN}[4/4] Running benchmark...${NC}"
echo "Configuration:"
echo "  - Model: ${MODEL_PATH}"
echo "  - GPU layers: 99 (full offload)"
echo "  - Flash attention: disabled"
echo "  - Prompt sizes: 512, 1024"
echo "  - Generation sizes: 128, 256"
echo

# Check if the binary exists
if [ ! -f "build/bin/llama-bench" ]; then
    echo -e "${RED}Error: llama-bench binary not found. Build may have failed.${NC}"
    exit 1
fi

# Run the benchmark
./build/bin/llama-bench -m "${MODEL_PATH}" -ngl 99 -fa 0 -p 512,1024 -n 128,256

echo
echo -e "${GREEN}=== Benchmark Complete ===${NC}"