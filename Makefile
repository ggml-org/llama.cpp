# =============================================================================
# llama.cpp Tsavorite build orchestration
#
# Targets:
#   make env          — generate .env from build.yml (auto-run when needed)
#   make all          — build FPGA target
#   make clean        — remove llama.cpp build directories
#   make clean-all    — remove build directories + python venv
#   make package      — package FPGA bundle (requires a prior FPGA build)
#
# Variables (override on command line):
#   BUILD_TYPE=debug|release|debug-tmu|debug-tmu-detail  (default: debug)
#
# Examples:
#   make all
#   make all BUILD_TYPE=release
#   make clean
# =============================================================================

SHELL        := /bin/bash
.ONESHELL:
# Failures print the exact failing command before Make reports "Error N".
# The ERR trap fires on any non-zero exit inside the recipe, printing
# the command that failed so there is no need to re-run outside make.
.SHELLFLAGS  := -euo pipefail -c

ROOT         := $(shell git rev-parse --show-toplevel 2>/dev/null)
BUILD_YML    := $(ROOT)/build.yml
ENV_FILE     := $(ROOT)/.env
BUILD_SCRIPT := $(ROOT)/tsi-pkg-build.sh
GEN_ENV      := $(ROOT)/gen-env.py

# Default build type — override on the command line: make all BUILD_TYPE=release
BUILD_TYPE ?= debug

# =============================================================================
# .env generation
#
# gen-env.py reads build.yml and writes .env with all resolved paths.
# .env is gitignored and machine-specific. It is auto-regenerated whenever
# build.yml is newer than .env.
# =============================================================================

.PHONY: env
env: $(ENV_FILE)

$(ENV_FILE): $(BUILD_YML) $(GEN_ENV)
	python3 "$(GEN_ENV)" "$(BUILD_YML)" "$(ENV_FILE)"

# =============================================================================
# Build targets
#
# Each target sources .env and invokes tsi-pkg-build.sh as a plain executable.
# The ERR trap ensures any failure prints the exact failing command before
# Make exits, making the root cause visible without re-running outside make.
# =============================================================================

.PHONY: all
all: $(ENV_FILE)
	trap 'echo "[MAKE] FAILED: $$BASH_COMMAND" >&2' ERR
	echo "[MAKE] build started: BUILD_TYPE=$(BUILD_TYPE)"
	source "$(ENV_FILE)"
	SDK_VERSION="$$SDK_VERSION" \
	MLIR_COMPILER_DIR="$$MLIR_COMPILER_DIR" \
	TOOLBOX_DIR="$$TOOLBOX_DIR" \
	RUNTIME_DIR="$$RUNTIME_DIR" \
	CC="$$CC" \
	CXX="$$CXX" \
	LD_LIBRARY_PATH="$$LD_LIBRARY_PATH" \
	bash "$(BUILD_SCRIPT)" $(BUILD_TYPE) build-fpga
	echo "[MAKE] build complete"

.PHONY: clean
clean: $(ENV_FILE)
	trap 'echo "[MAKE] FAILED: $$BASH_COMMAND" >&2' ERR
	echo "[MAKE] clean"
	source "$(ENV_FILE)"
	SDK_VERSION="$$SDK_VERSION" bash "$(BUILD_SCRIPT)" clean

.PHONY: clean-all
clean-all: $(ENV_FILE)
	trap 'echo "[MAKE] FAILED: $$BASH_COMMAND" >&2' ERR
	echo "[MAKE] clean-all"
	source "$(ENV_FILE)"
	SDK_VERSION="$$SDK_VERSION" bash "$(BUILD_SCRIPT)" clean-all

.PHONY: package
package: $(ENV_FILE)
	trap 'echo "[MAKE] FAILED: $$BASH_COMMAND" >&2' ERR
	echo "[MAKE] package"
	source "$(ENV_FILE)"
	SDK_VERSION="$$SDK_VERSION" \
	MLIR_COMPILER_DIR="$$MLIR_COMPILER_DIR" \
	TOOLBOX_DIR="$$TOOLBOX_DIR" \
	RUNTIME_DIR="$$RUNTIME_DIR" \
	bash "$(BUILD_SCRIPT)" package

.PHONY: help
help:
	echo "Targets:"
	echo "  make all          - build FPGA target (default BUILD_TYPE=debug)"
	echo "  make clean        - remove llama.cpp build directories"
	echo "  make clean-all    - clean + remove python venv"
	echo "  make package      - package FPGA bundle (requires prior build)"
	echo "  make env          - regenerate .env from build.yml"
	echo ""
	echo "Variables:"
	echo "  BUILD_TYPE=debug|release|debug-tmu|debug-tmu-detail  (default: debug)"