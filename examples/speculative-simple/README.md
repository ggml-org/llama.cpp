# llama.cpp/examples/speculative-simple

Demonstration of basic greedy speculative decoding

```bash
./bin/llama-speculative-simple \
    -m  ../models/qwen2.5-32b-coder-instruct/ggml-model-q8_0.gguf \
    -md ../models/qwen2.5-1.5b-coder-instruct/ggml-model-q4_0.gguf \
    -f test.txt -c 0 -ngl 99 --color on \
    --sampling-seq k --top-k 1 -fa on --temp 0.0 \
    -ngld 99 --spec-draft-n-max 16 --spec-draft-n-draft-min 5 --draft-p-min 0.9
```
