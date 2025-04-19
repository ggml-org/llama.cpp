#!/bin/sh

# for convenience, we have this script to ease the development process

# make sure we are in the right directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT="$SCRIPT_DIR/../.."
cd $PROJECT_ROOT

./build/bin/llama-phi4mm-cli \
  -m ../models/Phi-4-multimodal-instruct/model.gguf \
  --mmproj ../models/Phi-4-multimodal-instruct/mmproj.gguf \
  --lora ../models/Phi-4-multimodal-instruct/vision_lora.gguf \
  --image ../models/bliss.png
