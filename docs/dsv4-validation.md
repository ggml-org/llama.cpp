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
* tensor split mode with `--tensor-split` (default `1,1`);
* `--flash-attn auto` (override with `DSV4_FLASH_ATTN=on` or `off`);
* a repeated continuation request with server prompt caching and
  `--cache-reuse`, requiring a positive `timings.cache_n`.

The default tensor proportions assume two visible devices. Override
`DSV4_TENSOR_SPLIT` for the devices in the validation host, for example
`DSV4_TENSOR_SPLIT=3,1`. `DSV4_REFERENCE_SPLIT` can be changed when a
non-layer reference is needed. The script uses only flags present in the
current CLI/server help; it is intentionally not registered as a default CTest
because it requires external model weights and a suitable GPU setup.

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

Speculative MTP decoding is also exposed by the existing server/CLI flags, but
requires a compatible MTP sidecar GGUF (not supplied here):

```bash
build/bin/llama-server -m "$DSV4_MODEL" \
  --spec-draft-model /models/dsv4-mtp.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --parallel 1 --port 18080
```

Use `--help` on the built binary to inspect device-specific options. Do not
use `--draft` or `--draft-max`; those legacy names are removed in this tree.