#!/bin/bash

wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip

echo "Usage:"
echo ""
echo "  ./llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw [other params]"
echo ""

exit 0
