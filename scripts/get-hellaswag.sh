#!/usr/bin/env bash

wget https://raw.githubusercontent.com/klosax/hellaswag_text_data/main/hellaswag_val_full.txt

echo "Usage:"
echo ""
echo "  ./llama-perplexity -m model.gguf -f hellaswag_val_full.txt --hellaswag [--hellaswag-tasks N] [other params]"
echo ""

exit 0
