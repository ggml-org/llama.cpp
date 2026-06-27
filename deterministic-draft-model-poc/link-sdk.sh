#!/bin/bash
# link-sdk.sh -- Link external SDK artifacts into the PoC project
#
# Run this after building llama.cpp with -DDETERMINISTIC_SPEC_ENABLED=ON
# It symlinks the distributed header and shared library into the PoC lib/ dir.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

POC_LIB="$SCRIPT_DIR/lib"
EXTERNAL_INCLUDE="$REPO_ROOT/external/include"
EXTERNAL_LIB="$REPO_ROOT/external/lib"

# Relative path from lib/ back to the repo root, so the symlinks stay valid
# across clones/copies of the repo (not tied to this machine's absolute path).
REL_TO_ROOT="$(realpath --relative-to="$POC_LIB" "$REPO_ROOT")"

mkdir -p "$POC_LIB"

# Link the SDK headers (relative symlinks so they work across clones)
ln -sf "$REL_TO_ROOT/external/include/deterministic_draft_plugin.h" "$POC_LIB/deterministic_draft_plugin.h"
echo "Linked: lib/deterministic_draft_plugin.h -> $EXTERNAL_INCLUDE/deterministic_draft_plugin.h"

ln -sf "$REL_TO_ROOT/external/include/deterministic_draft_capabilities.h" "$POC_LIB/deterministic_draft_capabilities.h"
echo "Linked: lib/deterministic_draft_capabilities.h -> $EXTERNAL_INCLUDE/deterministic_draft_capabilities.h"

ln -sf "$REL_TO_ROOT/external/include/llama_deterministic_draft.h" "$POC_LIB/llama_deterministic_draft.h"
echo "Linked: lib/llama_deterministic_draft.h -> $EXTERNAL_INCLUDE/llama_deterministic_draft.h"

# Link the spec shared library
if [ -f "$EXTERNAL_LIB/libdeterministic_draft_spec.so" ]; then
    ln -sf "$REL_TO_ROOT/external/lib/libdeterministic_draft_spec.so" "$POC_LIB/libdeterministic_draft_spec.so"
    echo "Linked: lib/libdeterministic_draft_spec.so -> $EXTERNAL_LIB/libdeterministic_draft_spec.so"
else
    echo "Warning: libdeterministic_draft_spec.so not found in $EXTERNAL_LIB"
    echo "  Build llama.cpp with -DDETERMINISTIC_SPEC_ENABLED=ON first"
fi

echo "Done. PoC project is ready to build."
echo "  cd $SCRIPT_DIR"
echo "  cmake -B build && cmake --build build -j\$(nproc)"
