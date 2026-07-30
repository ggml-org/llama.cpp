# llama-tweak

Offline backend tuning for llama.cpp (Intel-focused MVP: OpenVINO, SYCL, Vulkan).  
`llama-tweak record` benchmarks a fixed matrix of backends and writes a JSON cache.  
At runtime, `--device tweak` (CLI, llama-bench, or `LLAMA_TWEAK=1`) loads the cache and picks the fastest plan for the requested prefill/decode sizes.

Build with a multi-backend tree (for example SYCL + Vulkan + OpenVINO), then:

```bash
cmake --build <build-dir> --target llama-tweak llama-bench llama-cli
```

## Cache file location

Priority (same pattern as other llama.cpp paths):

1. `llama-tweak record --output /path/to/cache.json` (or `-o`)
2. `--tweak-cache` / `LLAMA_TWEAK_CACHE` (CLI and inference)
3. Default: `./llama-tweak-<model-stem>.json` in the **current working directory**

The cache stores a model fingerprint (size + mtime). Re-run `record` after the GGUF changes.

## Record (tuning)

```bash
cd /path/where/you/want/cache
llama-tweak record -m /path/model.gguf \
  --pp 128,512 \
  --tg 128 \
  --runs 3
```

- **`--pp`**: comma-separated prefill sizes (each tested with the same `--tg`).
- **`--tg`**: single decode length (default `128`).
- **`--runs`**: separate in-process `llama-bench` calls per backend (default `3`); results store mean and stddev of `avg_ts`.

Optional env (same as inference):

- `LLAMA_TWEAK_CACHE` – cache path
- `LL_OPENVINO_IGPU_DEVICE` – OpenVINO iGPU name (default `GPU.0`)

## Explain (inspect cache)

```bash
llama-tweak explain -m /path/model.gguf --pp 512 --tg 128
```

Uses nearest cached `pp` / `tg` if the exact pair was not recorded.  
Runtime uses the same resolution via `LLAMA_TWEAK_PP` / `LLAMA_TWEAK_TG` (defaults `512` / `128`).

## Verify with llama-bench

After `record`, run one benchmark with the tuned device:

```bash
llama-bench -m /path/model.gguf -pg 512,128 -p 0 -n 0 -r 3 -o jsonl --device tweak
```

On stderr you should see a line like:

```text
llama-tweak: pp=512 tg=128 -> cache pp=512 tg=128 <tag> (<ggml_device>, <expected> tok/s expected)
```

Compare JSONL `avg_ts` to the expected value (same order of magnitude; small drift between runs is normal).

## Inference (llama-cli / server)

```bash
export LLAMA_TWEAK_PP=512
export LLAMA_TWEAK_TG=128
llama-cli -m /path/model.gguf --device tweak ...
```

Or set `LLAMA_TWEAK=1` and configure devices via the cache (still requires `--device tweak` or tweak routing flag).

OpenVINO phase-split plans set `GGML_OPENVINO_PHASE_SPLIT`, prefill/decode devices, and stateful flags before the backend initializes.

## JSON schema (version 1)

Top-level: `schema_version`, `model_path`, `model_fingerprint`, `entries[]`.

Each entry: `tag`, `pp`, `tg`, `backend_kind`, `ggml_device`, backend-specific fields, `mean_tps`, `stddev_tps`, `runs`.

Selection rule today: for requested `(pp, tg)`, pick nearest cached `pp`, then nearest `tg`, then the entry with highest `mean_tps`.

See [TODO.md](TODO.md) for planned metrics, backend filters, and vendor hooks.
