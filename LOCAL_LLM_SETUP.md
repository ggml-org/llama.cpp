# Local LLM quick setup (llama.cpp)

This repository can run local LLMs with `llama-cli`.
No cleanup/deletion is required.

## 1) Prepare a GGUF model

- Place a model file in `models/`, for example: `models/model.gguf`
- This repo already contains one model file at the repository root:
  - `gemma-4-E4B-it-Q4_K_M.gguf`

## 2) Run the helper script

```bash
chmod +x run-local.sh
./run-local.sh
```

If `build/bin/llama-cli` does not exist yet, the script builds it automatically.

## 3) Optional: specify a model path

```bash
./run-local.sh models/your-model.gguf
```

## 4) Useful runtime tuning

- `-ngl 999`: offload layers to Metal GPU when possible
- `-c 4096`: context length
- `--temp 0.7`: generation temperature

Extra args can be passed through:

```bash
./run-local.sh models/your-model.gguf -t 8
```
