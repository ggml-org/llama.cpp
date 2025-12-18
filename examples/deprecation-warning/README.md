# Migration notice for binary filenames

> [!IMPORTANT]
[2024 Jun 12] Binaries have been renamed w/ a `llama-` prefix. `main` is now `llama-cli`, `server` is `llama-server`, etc (https://github.com/ggerganov/llama.cpp/pull/7809)

This migration was important, but it is a breaking change that may not always be immediately obvious to users.

Please update all scripts and workflows to use the new binary names.

| Old Filename | New Filename |
| ---- | ---- |
| main | llama-cli |
| server | llama-server |
| llama-bench | llama-bench |
| embedding | llama-embedding |
| quantize | llama-quantize |
| tokenize | llama-tokenize |
| export-lora | llama-export-lora |
| libllava.a | libllava.a |
| baby-llama | llama-baby-llama |
| batched | llama-batched |
| batched-bench | llama-batched-bench |
| benchmark-matmult | llama-benchmark-matmult |
| convert-llama2c-to-ggml | llama-convert-llama2c-to-ggml |
| eval-callback | llama-eval-callback |
| gbnf-validator | llama-gbnf-validator |
| gguf | llama-gguf |
| gguf-split | llama-gguf-split |
| gritlm | llama-gritlm |
| imatrix | llama-imatrix |
| infill | llama-infill |
| llava-cli | llama-llava-cli |
| lookahead | llama-lookahead |
| lookup | llama-lookup |
| lookup-create | llama-lookup-create |
| lookup-merge | llama-lookup-merge |
| lookup-stats | llama-lookup-stats |
| parallel | llama-parallel |
| passkey | llama-passkey |
| perplexity | llama-perplexity |
| q8dot | llama-q8dot |
| quantize-stats | llama-quantize-stats |
| retrieval | llama-retrieval |
| save-load-state | llama-save-load-state |
| simple | llama-simple |
| speculative | llama-speculative |
| vdot | llama-vdot |
| tests/test-c.o | tests/test-c.o |

