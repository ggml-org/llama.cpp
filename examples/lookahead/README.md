# llama.cpp/examples/lookahead

Demonstration of lookahead decoding technique:

https://lmsys.org/blog/2023-11-21-lookahead-decoding/

More info: https://github.com/ggml-org/llama.cpp/pull/4207

Sample command:

```bash
llama-lookahead -hf ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF -p "// network server implemented in C\n// author: Peter Hacker\n\n#include" -e -ngl 99 -t 4 -n 512 -c 4096 -kvu
```
