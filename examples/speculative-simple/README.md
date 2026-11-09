# llama.cpp/examples/speculative-simple

Demonstration of basic greedy speculative decoding

```bash
# spec-type draft-simple
./bin/llama-speculative-simple \
        -hf  ggml-org/Qwen3-8B-Base-GGUF:Q8_0 \
        -hfd ggml-org/Qwen3-0.6B-Base-GGUF \
        -p "Here is a quick sort implementation in C++. Just code, no comments:\n\n#include" \
        --spec-type draft-simple --spec-draft-n-max 7 -ngld 99 --color on \
        -n 256 --temp 0 --top-k 1 --seed 42 -ngl 99 -lv 4

# spec-type draft-mtp
./bin/llama-speculative-simple \
        -hf ggml-org/Qwen3.6-27B-GGUF:Q8_0 \
        -p "Here is a quick sort implementation in C++. Just code, no comments:\n\n#include" \
        --spec-type draft-mtp --spec-draft-n-max 3 -ngld 99 --color on \
        -n 256 --temp 0 --top-k 1 --seed 42 -ngl 99 -lv 4

# spec-type draft-mtp (with shared KV cache)
# note: this model needs a <s> token at the start to somewhat work without the chat template
./bin/llama-speculative-simple \
        -hf ggml-org/Gemma-4-31B-it-GGUF:Q8_0 \
        -p "<s>Here is a quick sort implementation in C++. Just code, no comments:\n\n#include" \
        --spec-type draft-mtp --spec-draft-n-max 3 -ngld 99 --color on \
        -n 256 --temp 0 --top-k 1 --seed 42 -ngl 99 -lv 4

# spec-type draft-eagle3
./bin/llama-speculative-simple \
        -hf ggml-org/gpt-oss-20b-GGUF \
        -p "Here is a quick sort implementation in C++. Just code, no comments:\n\n#include" \
        --spec-type draft-eagle3 --spec-draft-n-max 3 -ngld 99 --color on \
        -n 256 --temp 0 --top-k 1 --seed 42 -ngl 99 -lv 4

# spec-type draft-dflash
./bin/llama-speculative-simple \
        -hf ggml-org/Qwen3-8B-GGUF \
        -p "Here is a quick sort implementation in C++. Just code, no comments:\n\n#include" \
        --spec-type draft-dflash --spec-draft-n-max 7 -ngld 99 --color on \
        -n 256 --temp 0 --top-k 1 --seed 42 -ngl 99 -lv 4

# spec-type draft-dspark
./bin/llama-speculative-simple \
        -hf ggml-org/Qwen3-8B-GGUF \
        -p "Here is a quick sort implementation in C++. Just code, no comments:\n\n#include" \
        --spec-type draft-dspark --spec-draft-n-max 7 -ngld 99 --color on \
        -n 256 --temp 0 --top-k 1 --seed 42 -ngl 99 -lv 4
```
