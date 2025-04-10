#!/bin/bash

wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip

echo "Usage:"
echo ""
echo "  ./llama-perplexity -m model.gguf -f wiki.test.raw [other params]"
echo ""

exit 0
