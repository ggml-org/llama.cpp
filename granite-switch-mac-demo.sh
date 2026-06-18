#!/usr/bin/env bash
# granite-switch-mac-demo.sh
#
# End-to-end Granite Switch demo on an Apple-Silicon Mac (Metal GPU backend).
# Builds llama.cpp from this branch, converts the composed checkpoint to GGUF,
# and runs the crisp mid-sequence adapter-switch demos verified on Vela:
#   - answerability : <|answerability|>  -> "unanswerable"   (base would just answer)
#   - query_rewrite : <|query_rewrite|>  -> {"rewritten_question": ...}  (base would answer)
#
# Each demo runs the SAME prompt twice, differing only by a control token placed
# mid-sequence (right before the assistant turn). The outputs differ structurally,
# which is the switch firing per-token.
#
# Usage:
#   ./granite-switch-mac-demo.sh            # build + convert (first run) then demo
#   ./granite-switch-mac-demo.sh demo       # skip build/convert, just run the demos
#   GGUF=/path/to/gs-f16.gguf ./granite-switch-mac-demo.sh demo   # use an existing GGUF
#
# Requirements: Xcode CLT (`xcode-select --install`), cmake (`brew install cmake`),
# python3, and ~16GB+ unified memory for the f16 model (8.4 GB on disk).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

SRC_MODEL="${SRC_MODEL:-ibm-granite/granite-switch-4.1-3b-preview}"
GGUF="${GGUF:-$REPO_DIR/gs-f16.gguf}"
BIN="$REPO_DIR/build/bin/llama-completion"
NGL="${NGL:-99}"     # offload all layers to the Metal GPU; set NGL=0 to force CPU

# ---------------------------------------------------------------------------
build() {
  echo "=== Build llama.cpp (Metal is on by default on macOS) ==="
  if [ ! -x "$BIN" ]; then
    cmake -S . -B build -DLLAMA_CURL=OFF
    cmake --build build -j --target llama llama-completion llama-tokenize
  else
    echo "  (build/bin/llama-completion already present — skipping)"
  fi
}

convert() {
  if [ -f "$GGUF" ]; then
    echo "=== GGUF already present at $GGUF — skipping convert ==="
    return
  fi
  echo "=== Set up python deps for the converter ==="
  python3 -m venv .venv-convert
  # shellcheck disable=SC1091
  . .venv-convert/bin/activate
  pip install --upgrade pip
  pip install numpy torch transformers safetensors sentencepiece "huggingface_hub[cli]"

  echo "=== Download $SRC_MODEL (~8 GB) ==="
  SRC_DIR="$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$SRC_MODEL'))")"

  echo "=== Convert HF -> GGUF (f16) ==="
  PYTHONPATH=gguf-py python convert_hf_to_gguf.py "$SRC_DIR" --outfile "$GGUF" --outtype f16
  deactivate
}

# run NAME PROMPT N
run() {
  local name="$1" prompt="$2" n="${3:-16}"
  echo "=== $name ==="
  "$BIN" -m "$GGUF" -ngl "$NGL" --temp 0 -n "$n" -p "$prompt" 2>/dev/null
  echo; echo "-----"
}

demo() {
  [ -x "$BIN" ] || { echo "FATAL: $BIN not built — run without 'demo' first"; exit 1; }
  [ -f "$GGUF" ] || { echo "FATAL: $GGUF missing — run without 'demo' first"; exit 1; }

  echo
  echo "############ DEMO 1: answerability (<|answerability|>, id 100356) ############"
  echo "# Document is about the Eiffel Tower; question asks Australia's capital."
  echo "# Base ANSWERS the question; adapter judges it UNANSWERABLE from the doc."
  local ANS_CTX='<|start_of_role|>user<|end_of_role|>Document: The Eiffel Tower is in Paris. Question: What is the capital of Australia?<|end_of_role|><|end_of_text|><|start_of_role|>assistant<|end_of_role|>'
  run "answerability OFF (base, no control)"           "${ANS_CTX}" 12
  run "answerability ON  (<|answerability|> mid-seq)"  "${ANS_CTX}<|answerability|>" 12

  echo
  echo "############ DEMO 2: query_rewrite (<|query_rewrite|>, id 100353) ############"
  echo "# Follow-up 'What other movies has HE made?' — base answers it;"
  echo "# adapter REWRITES it into a standalone query (resolves 'he')."
  local QR_CTX='<|start_of_role|>user<|end_of_role|>Who directed Inception?<|end_of_text|><|start_of_role|>assistant<|end_of_role|>Christopher Nolan directed Inception.<|end_of_text|><|start_of_role|>user<|end_of_role|>What other movies has he made?<|end_of_text|><|start_of_role|>assistant<|end_of_role|>'
  run "query_rewrite OFF (base, no control)"           "${QR_CTX}" 24
  run "query_rewrite ON  (<|query_rewrite|> mid-seq)"  "${QR_CTX}<|query_rewrite|>" 24

  echo "=== DONE. In each pair, only the mid-sequence control token differs. ==="
}

# ---------------------------------------------------------------------------
case "${1:-all}" in
  demo)    demo ;;
  build)   build ;;
  convert) convert ;;
  all)     build; convert; demo ;;
  *) echo "usage: $0 [all|build|convert|demo]"; exit 1 ;;
esac
