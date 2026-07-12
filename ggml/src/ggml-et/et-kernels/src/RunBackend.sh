#!/usr/bin/env bash
set -euo pipefail

LOG="llama_bench_$(date +%Y%m%d_%H%M%S).log"

{
  echo "===== START ====="
  date
  hostname
  uname -a
  echo "Command:"
  echo "./build/bin/llama-bench -m ../../models/Llama-3.2-1B-Instruct-Q8_0.gguf -fa 0 -p 32,64,128,256,512 -n 32,64,128,256,512"
  echo "================="

  ./build/bin/llama-bench \
    -m ../../models/Llama-3.2-1B-Instruct-Q8_0.gguf \
    -fa 0 \
    -p 32,64,128,256,512 \
    -n 32,64,128,256,512

  echo "===== END ====="
  date
} 2>&1 | tee "$LOG"
