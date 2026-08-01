# DSv4 GGUF validation harness

`tests/test-dsv4-validation.sh` is a model-dependent smoke/regression harness.
It does not download or commit model weights. Set `DSV4_MODEL` to a real DSv4
GGUF and build `llama-server` first:

```bash
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --target llama-server --parallel
DSV4_MODEL=/models/dsv4.gguf tests/test-dsv4-validation.sh
```

The harness starts a fresh server for each mode and uses deterministic greedy
decoding (`seed=123`, temperature `0`). It checks:

* the same prompt and continuation in the layer split mode (the reference);
* tensor split mode with `--tensor-split` (default `1,1,1,1` for the four-V620
  validation host);
* `--flash-attn auto` (override with `DSV4_FLASH_ATTN=on` or `off`);
* a repeated continuation request with server prompt caching and
  `--cache-reuse`, requiring a positive `timings.cache_n`;
* deterministic `first`, continuation, and replay responses matching between
  reference and tensor modes.

Override `DSV4_TENSOR_SPLIT` for a different device layout, for example
`DSV4_TENSOR_SPLIT=3,1`. `DSV4_REFERENCE_SPLIT` can be changed when a non-layer
reference is needed. `DSV4_PARALLEL` controls the server's `--parallel` slot
count and defaults to `1`.

Speculative decoding is opt-in and requires a compatible draft GGUF. Supplying
`DSV4_DRAFT_MODEL` adds `--spec-draft-model` and uses `--spec-type draft-mtp`
unless `DSV4_SPEC_TYPE` overrides it. `DSV4_DRAFT_N_MAX`, when supplied, adds
`--spec-draft-n-max`. If either draft setting is supplied without an existing
`DSV4_DRAFT_MODEL`, the harness stops with an explicit validation error. Draft
runs use the same deterministic requests and compare all three responses
between reference and tensor modes.

The script uses only flags present in the current `build/bin/llama-server
--help`; it is intentionally not registered as a default CTest because it
requires external model weights and a suitable GPU setup.

## Long-context validation

Set `DSV4_CTX_SIZE` to exercise a longer context. `DSV4_N_PREDICT` is an
optional override for generated tokens per request (the default is `8`):

```bash
DSV4_MODEL=/models/dsv4.gguf \
DSV4_CTX_SIZE=32768 DSV4_N_PREDICT=128 \
tests/test-dsv4-validation.sh
```

## Optional commands

These are manual follow-ups, not required by the harness. The current server
supports parallel slots; start it with two slots and send requests to distinct
slots (the API body uses the existing `id_slot` field):

```bash
build/bin/llama-server -m "$DSV4_MODEL" --parallel 2 --port 18080
curl -sS http://127.0.0.1:18080/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"first prompt","id_slot":0,"n_predict":8,"seed":123,"temperature":0}'
curl -sS http://127.0.0.1:18080/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"second prompt","id_slot":1,"n_predict":8,"seed":123,"temperature":0}'
```

Use `--help` on the built binary to inspect device-specific options. The
speculative flags used by the harness are `--spec-draft-model`, `--spec-type`,
and `--spec-draft-n-max`; do not use removed legacy names such as `--draft` or
`--draft-max`.
