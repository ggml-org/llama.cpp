#!/bin/bash

set -e

cd "$(dirname "$0")/.." || exit

# Specify the model you want to use here:
MODEL="${MODEL:-./models/llama-2-13b-chat.ggmlv3.q5_K_M.bin}"
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-./prompts/chat-system.txt}

# Adjust to the number of CPU cores you want to use.
N_THREAD="${N_THREAD:-12}"

# Note: you can also override the generation options by specifying them on the command line:
GEN_OPTIONS="${GEN_OPTIONS:---ctx_size 4096 --batch-size 1024}"


# shellcheck disable=SC2086 # Intended splitting of GEN_OPTIONS
./llama-server $GEN_OPTIONS \
  --model "$MODEL" \
  --threads "$N_THREAD" \
  --rope-freq-scale 1.0 \
  "$@"

# I used this to test the model with mps, but omitted it from the general purpose. If you want to use it, just specify it on the command line.
# -ngl 1 \
