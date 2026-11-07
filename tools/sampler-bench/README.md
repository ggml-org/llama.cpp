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
