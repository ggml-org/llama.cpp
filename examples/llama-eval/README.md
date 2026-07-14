# llama-eval

Simple evaluation tool for llama.cpp with support for multiple datasets.

For a full description, usage examples, and sample results, see:

- [PR 21152](https://github.com/ggml-org/llama.cpp/pull/21152)

## Quick start

```bash
# Single server
python3 llama-eval.py \
  --server http://localhost:8033 \
  --model my-model \
  --dataset gsm8k --n_cases 100 \
  --grader-type regex --threads 32

# Multiple servers (comma-separated URLs and thread counts)
python3 llama-eval.py \
  --server http://server1:8033,http://server2:8033 \
  --server-name server1,server2 \
  --threads 16,16 \
  --dataset aime2025 --n_cases 240 \
  --grader-type regex
```
