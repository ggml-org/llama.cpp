# Sparse Ternary FMA Kernel - Build Configuration
# Copyright 2025 HyperFold Technologies UK Ltd
# Author: Maurice Wilson
# License: Apache 2.0

# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -fPIC
CFLAGS_AVX512 = $(CFLAGS) -mavx512f

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BENCHMARK_DIR = benchmark
BUILD_DIR = build
LIB_DIR = lib
BIN_DIR = bin

# Files
SOURCES = $(SRC_DIR)/sparse_ternary_fma.c
HEADERS = $(INCLUDE_DIR)/sparse_ternary_fma.h
BENCHMARK_SRC = $(BENCHMARK_DIR)/benchmark.c

# Output files
LIB_STATIC = $(LIB_DIR)/libsparsetfma.a
LIB_SHARED = $(LIB_DIR)/libsparsetfma.so
BENCHMARK_BIN = $(BIN_DIR)/benchmark

# Object files
OBJECTS = $(BUILD_DIR)/sparse_ternary_fma.o
BENCHMARK_OBJ = $(BUILD_DIR)/benchmark.o

# Default target
.PHONY: all
all: $(LIB_STATIC) $(LIB_SHARED) $(BENCHMARK_BIN)

# Create directories
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(LIB_DIR):
	@mkdir -p $(LIB_DIR)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Compile library source
$(BUILD_DIR)/sparse_ternary_fma.o: $(SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS_AVX512) -I$(INCLUDE_DIR) -c $(SOURCES) -o $@

# Compile benchmark
$(BUILD_DIR)/benchmark.o: $(BENCHMARK_SRC) $(HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS_AVX512) -I$(INCLUDE_DIR) -c $(BENCHMARK_SRC) -o $@

# Create static library
$(LIB_STATIC): $(OBJECTS) | $(LIB_DIR)
	ar rcs $@ $(OBJECTS)
	@echo "✓ Static library created: $@"

# Create shared library
$(LIB_SHARED): $(OBJECTS) | $(LIB_DIR)
	$(CC) -shared -o $@ $(OBJECTS)
	@echo "✓ Shared library created: $@"

# Build benchmark executable
$(BENCHMARK_BIN): $(BENCHMARK_OBJ) $(OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS_AVX512) -o $@ $(BENCHMARK_OBJ) $(OBJECTS) -lm
	@echo "✓ Benchmark executable created: $@"

# Run benchmark
.PHONY: benchmark
benchmark: $(BENCHMARK_BIN)
	@echo ""
	@echo "Running benchmark..."
	@echo ""
	@$(BENCHMARK_BIN)

# Clean build artifacts
.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR) $(LIB_DIR) $(BIN_DIR)
	@echo "✓ Build artifacts cleaned"

# Install library and headers
.PHONY: install
install: all
	@mkdir -p /usr/local/lib /usr/local/include
	@cp $(LIB_STATIC) /usr/local/lib/
	@cp $(LIB_SHARED) /usr/local/lib/
	@cp $(HEADERS) /usr/local/include/
	@ldconfig
	@echo "✓ Library installed to /usr/local/lib"
	@echo "✓ Headers installed to /usr/local/include"

# Uninstall library and headers
.PHONY: uninstall
uninstall:
	@rm -f /usr/local/lib/libsparsetfma.a
	@rm -f /usr/local/lib/libsparsetfma.so
	@rm -f /usr/local/include/sparse_ternary_fma.h
	@ldconfig
	@echo "✓ Library uninstalled"

# Display help
.PHONY: help
help:
	@echo "Sparse Ternary FMA Kernel - Build Targets"
	@echo ""
	@echo "  make              - Build library and benchmark"
	@echo "  make benchmark    - Run performance benchmarks"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make install      - Install library system-wide"
	@echo "  make uninstall    - Uninstall library"
	@echo "  make help         - Display this help message"
	@echo ""

.PHONY: info
info:
	@echo "Build Configuration:"
	@echo "  Compiler: $(CC)"
	@echo "  CFLAGS: $(CFLAGS)"
	@echo "  AVX-512 Support: Enabled"
	@echo ""
	@echo "Output Directories:"
	@echo "  Libraries: $(LIB_DIR)/"
	@echo "  Executables: $(BIN_DIR)/"
	@echo "  Objects: $(BUILD_DIR)/"
	@echo ""
