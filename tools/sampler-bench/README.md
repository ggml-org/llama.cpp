# Sampler grammar benchmark

`llama-sampler-bench` compares CPU grammar + greedy sampling with the same chain on a ggml backend. It loads only the vocabulary from a GGUF and replays synthetic logits through the sampler. No transformer evaluation or KV cache is involved, so these are sampler-only token rates, not model generation rates.

From the repository root, using an existing configured build:

```sh
cmake -S . -B build
cmake --build build --target llama-sampler-bench -j 8
build/bin/llama-sampler-bench -m /devel/models/qwen35-0.8b-q8_0-pure.gguf
```

Acquire the shared `gpu` hardware queue before running on a GPU and release it immediately afterwards. On this machine GPU execution also needs sandbox escalation. No GPU is needed to generate the schema and grammar:

```sh
build/bin/llama-sampler-bench --generate-only \
    --write-schema tools/test-grammar-tools.schema.json \
    --write-grammar tools/test-grammar-tools.gbnf
```

The standalone target requires `LLAMA_BUILD_TOOLS=ON` and `LLAMA_BUILD_COMMON=ON`. For a new CPU-only build, configure with `-DGGML_HIP=OFF -DGGML_VULKAN=OFF -DGGML_CUDA=OFF`. Use the same backend options as your existing build for GPU comparisons. After the dependencies are built, recompiling the benchmark requires one C++ translation unit and a link.

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `-m PATH` | required for timing | GGUF vocabulary source; weights are not loaded |
| `-n N` | `512` | Measured sampler steps per repetition and path |
| `--repetitions N` | `1` | Number of repetitions; CPU/backend order alternates |
| `--device NAME` | `gpu` | First GPU, `cpu`, or an exact ggml device name |
| `--tools PATH` | `tools/test-grammar-tools.json` | Tool definitions and replay examples |
| `--write-schema PATH` | none | Save the derived JSON Schema |
| `--write-grammar PATH` | none | Save the derived GBNF |
| `--generate-only` | off | Generate files without loading a vocabulary or backend |

Run from the repository root or provide `--tools` explicitly. CMake also copies the fixture to `build/tools/sampler-bench/test-grammar-tools.json`. Unsupported devices or incomplete backend chains fail with a nonzero exit status; there is no CPU fallback under the backend label. With `--device cpu`, the comparison is the host sampler against a ggml CPU sampling graph.

For a longer comparison with variability and alternating order, use `-n 1024 --repetitions 3`. With one repetition the printed standard deviation is zero because variability was not measured.

## Fixture and validation

`tools/test-grammar-tools.json` contains 15 OpenAI-style function definitions with descriptions, required and optional parameters, enums, arrays, booleans, and a string-valued environment object. The definitions are adapted from Syndatis file-system, shell, task-list, and structured source tools. `code_search` is a representative project search definition rather than an exact Syndatis export. No tools are executed.

The program derives an array-of-tool-calls schema with a `oneOf` branch per tool, fixes each tool name with `const`, and rejects extra call/argument fields. It converts this schema with `json_schema_to_grammar(schema, true)`, forcing GBNF even in llguidance builds. Regenerate the checked-in `.schema.json` and `.gbnf` files after changing the definitions.

One trace contains all 15 example calls and an EOS token. Examples exercise optional fields, enum and array values, escapes, and UTF-8. Before timing, an independent CPU grammar pass checks every trace token and records one forbidden token at every position. During the benchmark the expected token has logit 10, the forbidden token has logit 20, and all other tokens have logit 0. Every sampled token must match the expected token, so a missing grammar mask fails the run. This checks the replayed states, not every possible input or all sampler combinations.

Both paths warm up for one complete trace. Each measured repetition starts from the grammar root, repeats the trace as needed, and checks every sampled token. If `-n` is shorter than the printed `trace_tokens`, the measured portion covers only that prefix; warmup still covers the complete trace. At trace boundaries grammar reset is included in the timing.

## Timing boundaries

CPU timing includes constructing full-vocabulary candidates from the synthetic logits, grammar filtering, greedy selection, and accepting the selected token. Backend timing includes grammar mask preparation, mask upload through the scheduler, graph execution (TOKEN_MASK + ARGMAX), synchronous selected-token readback, and host grammar acceptance. The graph and its allocations are reused for the entire run. CPU backend graph execution uses one thread.

Synthetic backend logits are initialized once; changing the two nonzero logits and clearing them after a step happens outside the timer. This models logits already resident on the backend. CPU candidate construction is inside the timer because the host sampler consumes candidate structures rather than the model's float logits. Model loading, schema conversion, preflight validation, graph setup, warmup, correctness comparisons, and output printing are outside the reported sampling time. Times are accumulated per step with a monotonic clock.

Output includes elapsed seconds, tok/s, microseconds per token, arithmetic mean tok/s, sample standard deviation, range, and the backend/CPU mean-rate ratio. The replay uses artificial scores to hold grammar states constant across paths; it does not reproduce a model's score distribution, stochastic sampling, server integration, or end-to-end throughput. This focused sampler benchmark is separate from the standard `llama-bench` matrix.

# Matched-policy benchmark

`llama-sampler-bench` above always masks the full vocabulary before selection, which is the backend grammar-first workload. The server's host path normally samples first and checks one token, so comparing those two numbers mixes policies. `llama-sampler-bench-matched` runs host and backend sampling under the same policy, with the same grammar, logits, chain and rejection schedule, and reports every step of each path.

```sh
cmake --build build --target llama-sampler-bench-matched -j 8
build/bin/llama-sampler-bench-matched -m /devel/models/qwen35-0.8b-q8_0-pure.gguf -n 1024 --repetitions 3 --jsonl out.jsonl
```

## Policies and paths

Two policies, each run by three paths. Tables compare paths only within one policy.

- `sample-then-check` (`stc`, server default): select with the chain, check that one token against the grammar, and only on rejection mask the full vocabulary and select again.
- `grammar-first` (`gf`): mask the full vocabulary, then select.

Paths:

- `cpu-scan`: the host sampler as `common_sampler_sample` runs it without backend sampling. The grammar sampler masks by scanning all candidates (`llama_grammar_apply_impl`).
- `cpu-trie`: the same host flow, but full masks come from `llama_grammar_get_allowed_tokens`, the trie + mask cache + scan fallback the backend uses. This separates the algorithmic gain of the trie from the gain of running on the backend. Before timing, the trie mask is checked against the scan mask at every replayed grammar state.
- `backend`: ggml sampling graphs on `--device`.

## Timed components

Every component is inside the per-step time. `retry_*` components run only on rejected steps; the tables average them over all steps.

| Path | Policy | Components |
| --- | --- | --- |
| host | stc | `logits_d2h` (copy the model's logits out of the backend buffer, as `llama_decode` does), `cand_init` (n_vocab candidates), `chain`, `validate` (one-token grammar check); on rejection `retry_cand_init`, `retry_mask_prep` (+ `retry_mask_apply` for trie), `retry_chain`; `accept` (grammar + chain) |
| host | gf | `logits_d2h`, `cand_init`, `mask_prep` (+ `mask_apply` for trie), `chain`, `accept` |
| backend | stc | `retain_copy` (device copy of the logits, as `llama_context::retain_logits`), `upload` (chain inputs), `graph` (chain, no grammar), `readback` (token), `validate`; on rejection the `llama_context::sample_retry` sequence: `retry_state` (copy the grammar into the retry sampler), `retry_mask_prep`, `retry_logits_copy` (retained logits into the retry graph), `retry_upload` (mask), `retry_graph` (TOKEN_MASK + chain), `retry_readback`; `accept` |
| backend | gf | `mask_prep` (`llama_sampler_grammar_prepare_rows`), `upload` (mask through the scheduler, slot cache), `graph` (TOKEN_MASK + chain), `readback`, `accept` |

All paths also have `eval` (the simulated model evaluation from `--eval-us`, a busy wait) and `reset` (grammar reset when the replay wraps around). The decode graph reads the model logits from a device tensor directly. The retry graph owns its input and runs without a scheduler, like `sample_retry`. Host candidate construction is not a separate backend step: the backend path reads back a single token.

`--logits-copy off` removes `logits_d2h` from the host paths, modelling logits that are already in host memory. By default the backend builds masks on the host (`mask_source=host`). `--device-mask on` allows the GPU grammar prototype, if the backend exposes `ggml_backend_grammar_compute`; this sets `LLAMA_GRAMMAR_GPU_MASK`, which `llama_grammar_gpu_init` reads.

## Inputs

Synthetic mode (default): every token gets a fixed pseudo-random logit in [-3, 3]. The expected trace token gets 15. For each rate in `--rates` (default `0,0.01,0.1,0.5,1`), a seeded schedule (`--seed`) picks the steps on which a grammar-forbidden token gets 20. On those steps sample-then-check must reject and retry. Every path and policy sees the same rows. The run fails if a selected token differs from the expected token, or if a sample-then-check path retries on a different set of steps than the schedule.

Real-logit replay: `--dump-logits FILE` loads the model (`-ngl`, direct I/O), builds a ChatML prompt with the tool list, runs the trace teacher-forced and saves the logits that predict each trace token. A Qwen3.5-0.8B file is about 420 MB. `--replay FILE` replays those rows unmodified. The retry count of a sample-then-check path is the actual rejection rate. Paths accept the trace token (teacher forcing), so grammar states match the saved logits. Token differences are counted rather than fatal: `mismatch` is against `cpu-scan` of the same policy; "selected != trace token" means the model would have chosen a different continuation. A cross-policy check compares the `stc` and `gf` selections, which must agree under greedy selection.

The default chain is top-k 40, top-p 0.95, min-p 0.05, greedy (`--top-k 0`, `--top-p 1`, `--min-p 0` remove samplers). Greedy keeps host and backend selections comparable token for token.

## Output

For each rate and policy: mean us/token, the SD of repetition means, per-step p50/p99, tok/s, retries, token mismatches, the backend ratio, and a per-component breakdown. With `--eval-us`, a "sampling share" row gives the fraction of the step not spent in simulated evaluation. For `cpu-trie` the mask cache hits and misses are printed. `--jsonl` writes one record per step (components, proposal, selected token, forced and retried flags) and one summary record per cell.

## Caveats

- The backend sampling graph runs separately from any model graph, so each step pays its own launch and synchronization. In the server, sampling is part of the decode graph. This favours the host paths. With `--eval-us`, the GPU also idles between launches and down-clocks, which inflates the standalone graph time further.
- `sample_retry` copies the grammar into its retry sampler (`retry_state`). Grammar clones share the program and the mask cache with their source, so the copy is cheap and backend retries hit the same cache as host retries. The cache also survives grammar resets, so masks computed in one trace cycle (or server request) are reused in the next.
- Proposal-state cloning in the server's grammar adapter is not modelled; for the stateless samplers used here it is cheap.
- Results are sampler-only rates plus an optional simulated evaluation, not end-to-end generation rates. They do not replace the standard `llama-bench` matrix.
