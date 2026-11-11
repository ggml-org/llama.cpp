# Flint native tool-call grammar

With `LLAMA_LLGUIDANCE=ON` and `--jinja`, `llama-server` recognizes Flint's embedded chat template and applies its tool grammar directly to `/v1/chat/completions`. No Python proxy or separate `tokenizer.json` is needed. Token IDs come from the loaded GGUF vocabulary. SSE streaming uses the native server and PEG parser.

## Build and run

This branch is based on public llama.cpp `3b4d6ab3f79651935fba1dd93732f29022a7f5f5` (2026-11-09), which already includes Flint model loading and template escapes. No separate model-support or template patch is required.

Rust/Cargo is required at build time for the existing llguidance dependency. This change does not add a Python runtime dependency.

```sh
cmake -S . -B build-flint -DLLAMA_LLGUIDANCE=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-flint --target llama-server test-chat test-grammar-llguidance -j 8
build-flint/bin/llama-server -m /path/to/flint.gguf --alias flint --jinja \
    --host 127.0.0.1 --port 8080 -ngl all -c 8192 -np 1
```

Add the usual backend options for the target platform, such as `-DGGML_CUDA=ON`. The local qualification used Metal on macOS; CUDA and Windows are not validated by these results.

Use the normal endpoint and request fields:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' -d '{
      "model": "flint",
      "messages": [{"role": "user", "content": "Read hello.py"}],
      "tools": [{"type": "function", "function": {
        "name": "view", "parameters": {
          "type": "object", "properties": {"path": {"type": "string"}},
          "required": ["path"], "additionalProperties": false
        }
      }}],
      "tool_choice": "required", "parallel_tool_calls": false,
      "stream": true, "max_tokens": 1024
    }'
```

## Behavior

- `auto` allows text followed by zero or more calls. `required` requires at least one call. `parallel_tool_calls: false` allows at most one call.
- A named choice (`{"type":"function","function":{"name":"view"}}`) permits exactly one call of that tool or its native alias. An unknown name returns HTTP 400. `none` removes tools from the prompt and prohibits tool calls.
- `read` aliases to the first offered name in `view`, `read_file`. `write` aliases to the first offered name in `create`, `write_to_file`, `write_file`. An explicitly offered `read` or `write` takes precedence. Responses, including streamed tool-name deltas, use the offered name.
- Declared properties appear in schema order, at most once. Required properties are enforced regardless of the tool's `strict` flag. Plain string arguments remain raw text; other values use llguidance JSON schemas, including nested objects/arrays and local definitions.
- Optional reasoning precedes text and calls. The grammar includes the assistant prefix consumed by llama.cpp before sampling and explicitly includes `<|observation|>`, `<|user|>`, and `<|end_of_text|>` so a completed turn can terminate.
- The grammar is eager rather than triggered after the first tool marker. Builds without llguidance retain the existing automatic parser behavior; they do not get this port's constraints.

The XML schema mapping intentionally preserves the Python builder's limits. It is not a complete JSON Schema validator: declared properties exclude additional keys even when `additionalProperties` allows them; schemas without properties use the original free-text key/value fallback, which can repeat keys. Root combinators and constraints beyond that mapping are not implemented. Plain strings select `const`, `enum`, `pattern`, or length bounds in that order instead of intersecting those constraints. Protocol-marker exclusion follows the Python rules. Tool validity does not establish that a file exists, that an agent/shell ID is real, or that the model chose the right tool.

The Flint handler rejects assistant continuation/prefill and `response_format` JSON schemas. These were not supported by the reference evaluation proxy. Templates with other protocols, including GLM, continue to use their existing handlers.

## Copilot background-shell filtering

To match Albert's optional `kit+alias-only+noshell` proxy variant, set `"chat_template_kwargs": {"flint_noshell": true}` on the chat request. The default is false. When enabled, `read_bash`, `write_bash`, `stop_bash`, and `list_bash` are removed from both the rendered tool list and grammar while the conversation has no background-shell evidence. A named choice for a hidden tool returns HTTP 400.

The C++ implementation follows the Python state heuristic: a tool result containing `<command with shellId: NAME is still running ...>` marks that shell running; `<shellId: NAME completed ...>` removes it. Multiple shells are tracked separately. The proxy returns immediately for any historical `bash` call with `mode: "async"`; this port preserves that conservative behavior, so session tools remain available even if a later result reports that async shell completed. This option neither validates shell IDs nor replaces the client's shell registry.

## Source and qualification

The port follows Albert Zhang's `flint_lark.py`, `kit+alias-only`, and optional `+noshell` behavior from [Miles PR #1233](https://github.com/recursive-org/miles/pull/1233), based on [PR #1172](https://github.com/recursive-org/miles/pull/1172). GitHub access was restored and the remote PR head verified on 2026-11-03: `13aff7f82d3d98298c28ea05ba1f70459833b16f`. Its `flint_lark.py` is byte-identical to the earlier inspected copy; the additional implementation change is background-shell filtering. Named-choice and `none` handling follow the qualified handoff at `69e910275`.

Initial qualification base llama.cpp revision: `418f905a4efe910d2a75636bbe65d48eca38790a`. Existing llguidance pin: `d795912fedc7d393de740177ea9ea761e7905774` (v1.0.1).

The adapter now rejects grammar compilation errors rather than returning an unconstrained sampler. Its reset recreates the matcher because the pinned llguidance version can fail to roll back explicit special tokens. Sampler-chain allocation occurs after grammar initialization so rejected grammars do not leak a chain. The generic llguidance suite covers the reset path as well as the Flint tests.

```sh
build-flint/bin/test-chat
build-flint/bin/test-grammar-llguidance models/ggml-vocab-llama-bpe.gguf
build-flint/bin/test-grammar-llguidance /path/to/flint.gguf --flint
```

The last command only loads the GGUF vocabulary. It checks valid and invalid tool turns, required/optional fields, duplicate keys, aliases, named/required/none choices, parallel-call limits, length bounds, patterns, references, nested JSON, markers, termination, assistant-prefix consumption, and invalid grammar compilation. Chat tests feed responses one character at a time, including `read` versus `read_bash`, escaped strings and empty strings.

Local end-to-end qualification uses the cached Flint INT4 GGUF on Apple M4 Pro/Metal, 8192 context, one slot, temperature 0 and a 1024-token output cap. Eleven successful generation checks cover plain chat, file-read, a supplied tool-result round trip, named choice, escaped edits, enum/integer arguments, one Microsoft first-call snapshot, three SSE variants, and `none`. Further checks force named and required calls against conflicting user text, preserve `length` on truncated SSE output, and return HTTP 400 for an unknown named choice or invalid grammar. Twenty-five HTTP checks pass in total, including nine shell-history cases compared with the pinned Python implementation; server logs are checked for unexpected matcher errors. The test does not execute generated tools. It does not establish Microsoft task success, Windows compatibility or a performance improvement.

The template fixture `models/templates/RecursiveSI-flint.jinja` is copied from the cached GGUF's embedded template; the server uses the GGUF template by default. GGUF revision: `9c47151036bc2c34331142dda04488bceab2bc9f`; the previously recorded checksum is `43c07baccdba42eb7528a9630ee15699b1942c297291dc1b39a061f508ee1a4b`.

## Qualification scope

### Public-base validation (2026-11-09)

On upstream `3b4d6ab3f79651935fba1dd93732f29022a7f5f5`, the migrated server and test targets build. All four CTest suites (`test-chat`, `test-grammar-llguidance`, `test-chat-auto-parser`, and `test-chat-peg-parser`) pass with llguidance enabled. The chat suite also passes in a separate build without llguidance. The grammar suite passes against the real Flint GGUF vocabulary.

All 25 live HTTP smoke checks pass on this base, including SSE, tool aliases, named/required/none choices, truncation, invalid grammar rejection, and shell-history filtering. This uses M4 Pro/Metal, Flint INT4 Q4_1, 16,384 context and one slot. Tools were not executed. Logs and raw HTTP captures are retained locally in the ignored `build-flint/qualification/` directory. The full Microsoft replay comparison was not rerun on this base.

### Earlier qualification

The earlier end-to-end checks above ran on upstream `3de52ac38818978d6da8f386eda43a856e53714a` on 2026-11-04. All 25 HTTP smoke checks passed. A separate seed-101 replay of seven supplied failing-turn snapshots returned HTTP 200 without structural/schema failures in both nonstreaming and SSE modes. Six of seven responses in each mode passed interface checks; the remaining response attempted to create an existing scaffold. Tools were not executed. These results do not establish Windows/CUDA compatibility or task-level acceptance and are not results for the November 9 base.

## Python golden-test coverage

The fixed corpus in [`tests/flint-golden.h`](../../tests/flint-golden.h) is ported from Miles [#1233](https://github.com/recursive-org/miles/pull/1233), head `13aff7f82d3d98298c28ea05ba1f70459833b16f`, including the grammar tests inherited from #1172. PR #123 is an unrelated training recipe. Expected outputs are frozen from the Python tests, not generated from C++ output.

| Python check | Native regression coverage |
|---|---|
| Six accepted and twelve rejected turn examples, both strict modes | `test_flint_python_golden`: nested/nullable schemas, 512/513-character boundary, required fields, list types, padding, and exact tool/key spelling |
| Ordinary-token spellings of four protocol markers | Token-mask rejection checks in text, reasoning, raw strings and JSON strings; both strict modes |
| Required/named choices and no parallel calls | Accepted/rejected fixed turns, including missing, wrong and repeated calls |
| Missing parameter schema and native aliases | Free-form arguments, `read` versus `read_bash`, and explicitly offered `read` precedence |
| Marker regex | Safe partial prefixes and rejected complete/overlapping markers against the full Flint marker set; the Python helper's isolated three-marker subset is not a separate C++ API |
| Fixed stream equals whole response at chunk sizes 1, 3, 2/5/7, and whole turn | Parser delta reconstruction plus native OpenAI serializer goldens, including raw argument-fragment reconstruction, IDs, role, finish reason and usage |
| Three token-limit/cut-call examples | Fixed prefix inputs with `length` finish, preserved reasoning/content, no protocol leakage, and stream/whole reconstruction |
| Shell availability | Existing nine history cases, including synchronous completion, async starts, multiple shells, and disabled filtering |

The normal `test-grammar-llguidance` CTest run adds Flint protocol tokens to a temporary copy of the bundled vocabulary, then removes it. The ordinary-token checks therefore run in CI without downloading Flint weights. The `--flint` run repeats the goldens with the real GGUF vocabulary. `test-chat` runs the fixed parser and native response-serializer goldens when llguidance is enabled; the build without llguidance remains covered by its existing chat suite.

The parser comparison normalizes argument JSON formatting and outer content whitespace because the Python proxy strips that whitespace. Native stream/whole comparisons preserve content and argument fragments exactly. The proxy buffers and drops incomplete calls at the token limit; native OpenAI streaming may expose partial calls and retains `finish_reason: length`. This native contract is tested rather than replacing it with proxy buffering. HTTP `[DONE]` transport framing remains covered by the live HTTP smoke suite.

SGLang-specific legacy/unknown structural-tag dispatch tests are not applicable to the native API. The SGLang backend's JSON/EBNF wrappers are outside this tool-grammar port: Flint's native `response_format` remains explicitly rejected (with a regression assertion), and user-supplied grammars are not wrapped in optional reasoning. These are documented scope differences, not claimed as ported features.

```sh
ctest --test-dir build-flint -R '^(test-chat|test-grammar-llguidance)$' --output-on-failure
build-flint/bin/test-grammar-llguidance /path/to/flint.gguf --flint
```
