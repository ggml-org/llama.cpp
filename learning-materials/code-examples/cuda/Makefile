# Makefile for CUDA Examples
# ==========================
#
# This Makefile compiles the CUDA examples for the llama.cpp learning materials.
# It provides various build targets and architectures.
#
# Usage:
#   make              - Build all examples with default settings
#   make clean        - Remove all built files
#   make vector_add   - Build only the vector addition example
#   make matmul       - Build only the matrix multiplication example
#   make quantized    - Build only the quantized matmul example
#   make run-all      - Build and run all examples
#   make arch=sm_86   - Build for specific GPU architecture (e.g., RTX 30xx)
#
# GPU Architectures (use arch=<value>):
#   sm_75  - Turing (RTX 20xx, T4)
#   sm_80  - Ampere (A100)
#   sm_86  - Ampere (RTX 30xx)
#   sm_89  - Ada Lovelace (RTX 40xx)
#   sm_90  - Hopper (H100)
#
# Example: Build for RTX 3090
#   make arch=sm_86 opt=3

# Compiler
NVCC := nvcc

# Detect GPU architecture if not specified
# You can override this with: make arch=sm_86
ifndef arch
    arch := sm_75  # Default to Turing for compatibility
endif

# Optimization level (0, 1, 2, 3)
opt ?= 3

# Compiler flags
NVCC_FLAGS := -arch=$(arch) -O$(opt)

# Additional flags for better warnings and standards
NVCC_FLAGS += --std=c++11
NVCC_FLAGS += -Xcompiler -Wall

# Debug flags (uncomment for debugging)
# NVCC_FLAGS += -g -G -lineinfo

# Source files
SRC_VECTOR := 01-vector-add.cu
SRC_MATMUL := 02-matrix-multiply.cu
SRC_QUANT  := 03-quantized-matmul.cu

# Output binaries
BIN_VECTOR := vector_add
BIN_MATMUL := matmul
BIN_QUANT  := quantized_matmul

# All binaries
BINS := $(BIN_VECTOR) $(BIN_MATMUL) $(BIN_QUANT)

# Default target: build all examples
.PHONY: all
all: $(BINS)
	@echo ""
	@echo "======================================"
	@echo "All CUDA examples built successfully!"
	@echo "======================================"
	@echo ""
	@echo "Run the examples:"
	@echo "  ./$(BIN_VECTOR)    - Vector addition (basics)"
	@echo "  ./$(BIN_MATMUL)    - Matrix multiplication (shared memory)"
	@echo "  ./$(BIN_QUANT)     - Quantized matmul (INT8 operations)"
	@echo ""
	@echo "Architecture: $(arch)"
	@echo "Optimization: -O$(opt)"
	@echo ""

# Build individual examples
$(BIN_VECTOR): $(SRC_VECTOR)
	@echo "Building $@ (architecture: $(arch), optimization: -O$(opt))..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "✓ $@ built successfully"
	@echo ""

$(BIN_MATMUL): $(SRC_MATMUL)
	@echo "Building $@ (architecture: $(arch), optimization: -O$(opt))..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "✓ $@ built successfully"
	@echo ""

$(BIN_QUANT): $(SRC_QUANT)
	@echo "Building $@ (architecture: $(arch), optimization: -O$(opt))..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "✓ $@ built successfully"
	@echo ""

# Convenience targets
.PHONY: vector_add matmul quantized
vector_add: $(BIN_VECTOR)
matmul: $(BIN_MATMUL)
quantized: $(BIN_QUANT)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(BINS)
	@echo "✓ Clean complete"

# Run all examples
.PHONY: run-all
run-all: all
	@echo ""
	@echo "======================================"
	@echo "Running all examples..."
	@echo "======================================"
	@echo ""
	@echo ">>> Running Vector Addition Example <<<"
	@echo "========================================"
	./$(BIN_VECTOR)
	@echo ""
	@echo ""
	@echo ">>> Running Matrix Multiplication Example <<<"
	@echo "=============================================="
	./$(BIN_MATMUL)
	@echo ""
	@echo ""
	@echo ">>> Running Quantized MatMul Example <<<"
	@echo "========================================="
	./$(BIN_QUANT)
	@echo ""
	@echo "======================================"
	@echo "All examples completed!"
	@echo "======================================"

# Run individual examples
.PHONY: run-vector run-matmul run-quantized
run-vector: $(BIN_VECTOR)
	@echo "Running Vector Addition Example..."
	./$(BIN_VECTOR)

run-matmul: $(BIN_MATMUL)
	@echo "Running Matrix Multiplication Example..."
	./$(BIN_MATMUL)

run-quantized: $(BIN_QUANT)
	@echo "Running Quantized MatMul Example..."
	./$(BIN_QUANT)

# Help target
.PHONY: help
help:
	@echo "CUDA Examples Makefile"
	@echo "======================"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build all examples (default)"
	@echo "  vector_add   - Build vector addition example"
	@echo "  matmul       - Build matrix multiplication example"
	@echo "  quantized    - Build quantized matmul example"
	@echo "  clean        - Remove all built files"
	@echo "  run-all      - Build and run all examples"
	@echo "  run-vector   - Build and run vector addition"
	@echo "  run-matmul   - Build and run matrix multiplication"
	@echo "  run-quantized - Build and run quantized matmul"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Options:"
	@echo "  arch=<sm_XX> - Specify GPU architecture (default: sm_75)"
	@echo "  opt=<0-3>    - Optimization level (default: 3)"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build all with defaults"
	@echo "  make arch=sm_86         # Build for RTX 30xx series"
	@echo "  make arch=sm_89 opt=3   # Build for RTX 40xx with -O3"
	@echo "  make clean all          # Clean and rebuild"
	@echo "  make run-all            # Build and run everything"
	@echo ""
	@echo "GPU Architectures:"
	@echo "  sm_75 - Turing (RTX 20xx, T4)"
	@echo "  sm_80 - Ampere (A100)"
	@echo "  sm_86 - Ampere (RTX 30xx)"
	@echo "  sm_89 - Ada Lovelace (RTX 40xx)"
	@echo "  sm_90 - Hopper (H100)"
	@echo ""
	@echo "Check your GPU compute capability:"
	@echo "  nvidia-smi --query-gpu=compute_cap --format=csv"
	@echo ""

# Info target - show build configuration
.PHONY: info
info:
	@echo "Build Configuration"
	@echo "==================="
	@echo "Compiler:      $(NVCC)"
	@echo "Architecture:  $(arch)"
	@echo "Optimization:  -O$(opt)"
	@echo "Flags:         $(NVCC_FLAGS)"
	@echo ""
	@echo "Source Files:"
	@echo "  $(SRC_VECTOR)"
	@echo "  $(SRC_MATMUL)"
	@echo "  $(SRC_QUANT)"
	@echo ""
	@echo "Output Binaries:"
	@echo "  $(BIN_VECTOR)"
	@echo "  $(BIN_MATMUL)"
	@echo "  $(BIN_QUANT)"
	@echo ""

# Check if NVCC is available
.PHONY: check
check:
	@echo "Checking CUDA installation..."
	@which nvcc > /dev/null 2>&1 && echo "✓ nvcc found: $$(which nvcc)" || \
		(echo "✗ nvcc not found. Please install CUDA toolkit." && exit 1)
	@nvcc --version | head -n 4
	@echo ""
	@echo "Checking for NVIDIA GPU..."
	@which nvidia-smi > /dev/null 2>&1 && nvidia-smi --query-gpu=name,compute_cap --format=csv || \
		echo "✗ nvidia-smi not found or no GPU detected"
	@echo ""

# Advanced: Build with profiling enabled
.PHONY: profile
profile: NVCC_FLAGS += -lineinfo
profile: all
	@echo ""
	@echo "Built with profiling support."
	@echo "Use with: nsys profile ./$(BIN_VECTOR)"
	@echo "      or: ncu --set full ./$(BIN_MATMUL)"

# Advanced: Build with debug symbols
.PHONY: debug
debug: opt = 0
debug: NVCC_FLAGS += -g -G
debug: all
	@echo ""
	@echo "Built with debug symbols."
	@echo "Use with: cuda-gdb ./$(BIN_VECTOR)"
