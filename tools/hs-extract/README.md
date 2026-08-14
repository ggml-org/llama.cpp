# hs-extract

Single-prompt hidden state extraction tool for llama.cpp models.

Extracts residual stream activations from a single prompt and outputs them as JSON.

## Usage

```bash
llama-hs-extract -m model.gguf -p "Hello, world!"
```

## Options

- `-m, --model MODEL` - Path to GGUF model file (required)
- `-p, --prompt TEXT` - Input prompt text
- `-f, --file FILE` - Read prompt from file
- `-l, --layers LIST` - Comma-separated layer indices or `all` (default: all)
- `--raw` - Interpret prompt as comma-separated token IDs
- `--no-bos` - Don't add BOS token
- `-t, --threads N` - Number of CPU threads (default: 4)
- `-ngl, --n-gpu-layers N` - Number of layers to offload to GPU (default: 0)
- `--output FILE` - Output JSON file (default: stdout)

## Examples

Extract all layers:
```bash
llama-hs-extract -m model.gguf -p "The quick brown fox"
```

Extract specific layers:
```bash
llama-hs-extract -m model.gguf -p "Hello" -l 0,5,10
```

Read prompt from file:
```bash
llama-hs-extract -m model.gguf -f prompt.txt --output result.json
```

GPU acceleration:
```bash
llama-hs-extract -m model.gguf -p "Test" -ngl 99 --output output.json
```

## Output Format

JSON output structure:
```json
{
  "n_tokens": 4,
  "n_embd": 2048,
  "n_layers": 2,
  "layers": [
    {"layer": 0, "values": [0.123, -0.456, ...]},
    {"layer": 5, "values": [0.234, -0.567, ...]}
  ]
}
```

Each layer entry contains a flat array of `n_tokens * n_embd` float values
(row-major: token 0 dims 0..n_embd-1, then token 1, etc.).

## Notes

- Calls `llama_synchronize()` after decode to ensure CUDA writes complete before reading
- Supports both CPU and GPU inference via `-ngl` flag
- For batch processing, use [hs-extract-batch](../hs-extract-batch/) instead
