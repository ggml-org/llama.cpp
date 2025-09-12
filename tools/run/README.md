# llama.cpp/example/run

The purpose of this example is to demonstrate a minimal usage of llama.cpp for running models.

```bash
llama-run -hf llama.cpp/example/run
```

```bash
Usage: llama-run [server-options]

This tool starts a llama-server process and provides an interactive chat interface.
All options except --port are passed through to llama-server.

Common options:
  -h, --help                  Show this help
  -m,    --model FNAME        model path (default: `models/$filename` with filename from `--hf-file`
                              or `--model-url` if set, otherwise models/7B/ggml-model-f16.gguf)
  -hf,   -hfr, --hf-repo      <user>/<model>[:quant]
                              Hugging Face model repository; quant is optional, case-insensitive,
                              default to Q4_K_M, or falls back to the first file in the repo if
                              Q4_K_M doesn't exist.
                              mmproj is also downloaded automatically if available. to disable, add
                              --no-mmproj
                              example: unsloth/phi-4-GGUF:q4_k_m
                              (default: unused)
  -c, --ctx-size N            Context size
  -n, --predict N             Number of tokens to predict
  -t, --threads N             Number of threads

For all server options, run: llama-server --help
```
