# TurboAnchorKV experiment

`llama-turboanchorkv` is an offline research and evaluation tool for a two-tier KV representation derived from AnchorKV and TurboQuant+. It can analyze one layer, pack and reconstruct real cache state for continuation PPL and logit drift, and compare baseline versus compressed greedy generation. It does not reduce runtime memory because the compact attention kernel does not exist yet.

The evaluator is currently built on non-Windows platforms. It intentionally uses internal cache inspection interfaces whose symbols are not exported by the Windows shared-library build.

Tracking issue: [#332](https://github.com/TheTom/llama-cpp-turboquant/issues/332)

## Algorithm

The default analysis configuration retains the final 32 positions as bf16 anchors, distributes the remaining `S/128` anchor slots uniformly over the old prompt, assigns K and V independently to their nearest anchors, ranks residuals by squared norm, and encodes selected residuals with the existing block-128 TurboQuant codec. The packed representation stores bf16 anchors and coefficients, uint16 assignments, uint32 O(1) residual-slot maps, and contiguous TurboQuant residual rows. Its byte budget includes every packed array. The dense evaluator can use turbo2, turbo3, turbo4, or f16 residuals.

The tool keeps the paper-style attention anchor selection, observation-output utility ranking, random controls, and post-RoPE projection as ablation modes. Dense evaluation reconstructs the existing f16 cache in place and therefore measures quality without measuring compact-cache memory or decode speed.

## Build and model-free test

```sh
cmake -S . -B "$HOME/build-turboanchorkv" \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOME/build-turboanchorkv" --target llama-turboanchorkv test-turbo-quant -j 12
"$HOME/build-turboanchorkv/bin/llama-turboanchorkv" --self-test
ctest --test-dir "$HOME/build-turboanchorkv" -R '^test-turboanchorkv$' --output-on-failure
```

This recipe builds the CPU evaluator. For GPU evaluation, follow the supported
backend build instructions and use the resulting binary in the commands below.

The self-test checks synthetic RoPE recovery, deterministic uniform anchor placement, residual-norm ordering, residual codec quality ordering, partial residual-slot lookup, and exact serialized byte accounting. It needs no model.

## Implementation lineage

The packed storage and O(1) residual-slot design were adapted from [Giveen's AnchorKV CPU reference](https://github.com/giveen/llama-cpp-turboquant/tree/feature/anchor-kv-cpu-reference), especially the [initial storage format](https://github.com/giveen/llama-cpp-turboquant/commit/875a1f370), [precomputed slot indices](https://github.com/giveen/llama-cpp-turboquant/commit/db851cea8), and the [slot-order bug fix](https://github.com/giveen/llama-cpp-turboquant/commit/6ac311a72). TurboAnchorKV substitutes the ablated uniform-anchor, residual-norm, and existing TurboQuant codec choices. The reference branch's cache lifecycle, shared scratch, graph operator, and CUDA path are not included here.

## Modes

Set `TURBOANCHORKV_MODE` to one of:

- `analyze` (default): capture one layer and sweep 5x, 10x, and 20x simulated budgets using exact observation-attention error.
- `ppl`: save a prefix state, evaluate an unmodified suffix, restore the prefix, round-trip all physical full-attention KV layers, then report continuation PPL, exact mean KL, and top-1 agreement.
- `generate`: generate once from the unmodified prefix, restore it, round-trip the cache, generate again, and report both outputs plus token-prefix and expected-answer checks.

The dense modes currently support a plain or iSWA non-transposed f16/bf16 cache. They use post-RoPE cache tensors directly or recover the exact per-position transform from two configurable Gemma 4 layers for pre-RoPE projection.

## Layer analysis

```sh
TURBOANCHORKV_LAYER=20 \
TURBOANCHORKV_ANCHOR_MODE=uniform \
TURBOANCHORKV_RANK_MODE=norm \
TURBOANCHORKV_K_SPACE=pre \
./build/bin/llama-turboanchorkv \
  -m model.gguf -f prompt.txt -c 10240 -ngl 99 \
  -b 2048 -ub 512 -fa on -ctk f16 -ctv f16 --no-perf
```

## Continuation PPL and logit drift

Use the repository Wikitext helper, then cap the experiment to a context-sized prefix and suffix:

```sh
(cd build && ../scripts/get-wikitext-2.sh)

TURBOANCHORKV_MODE=ppl \
TURBOANCHORKV_MAX_TOKENS=9000 \
TURBOANCHORKV_PPL_TOKENS=512 \
TURBOANCHORKV_RATIO=5 \
TURBOANCHORKV_RESIDUAL_TYPE=turbo4 \
TURBOANCHORKV_K_SPACE=pre \
./build/bin/llama-turboanchorkv \
  -m model.gguf -f build/wikitext-2-raw/wiki.test.raw \
  -c 10240 -ngl 99 -b 2048 -ub 512 \
  -fa on -ctk f16 -ctv f16 --no-perf
```

Run the identity control before interpreting a new model or backend:

```sh
TURBOANCHORKV_MODE=ppl \
TURBOANCHORKV_DENSE_MODE=identity \
TURBOANCHORKV_MAX_TOKENS=9000 \
TURBOANCHORKV_PPL_TOKENS=128 \
./build/bin/llama-turboanchorkv \
  -m model.gguf -f build/wikitext-2-raw/wiki.test.raw \
  -c 10240 -ngl 99 -b 2048 -ub 512 \
  -fa on -ctk f16 -ctv f16 --no-perf
```

Expect identity to report zero PPL delta, zero KL, and 1.0 top-1 agreement. Any deviation invalidates compressed results.

## Needle retrieval

Generate deterministic prompts at several depths:

```sh
python3 examples/turboanchorkv/make-needle-prompt.py \
  docs/function-calling.md build/needle-50.txt --depth 0.5

TURBOANCHORKV_MODE=generate \
TURBOANCHORKV_RATIO=5 \
TURBOANCHORKV_RESIDUAL_TYPE=turbo4 \
TURBOANCHORKV_K_SPACE=pre \
TURBOANCHORKV_GENERATE_TOKENS=24 \
TURBOANCHORKV_EXPECT=TURBO-ANCHOR-7391 \
./build/bin/llama-turboanchorkv \
  -m model.gguf -f build/needle-50.txt \
  -c 10240 -ngl 99 -b 2048 -ub 512 \
  -fa on -ctk f16 -ctv f16 --no-perf
```

Repeat with `--depth 0.1` and `--depth 0.9`. A retrieval pass requires `expected_match=yes`; `exact_match=yes` is a stronger result.

## Controls

- `TURBOANCHORKV_LAYER` (default `0`)
- `TURBOANCHORKV_ROPE_LAYER_SECONDARY` (default `5` for the second Gemma 4 head dimension)
- `TURBOANCHORKV_WINDOW` (default `32`)
- `TURBOANCHORKV_ANCHOR_MODE` (`uniform`, `attention`, or `random`; default `uniform`)
- `TURBOANCHORKV_RANK_MODE` (`norm`, `utility`, or `random`; default `norm`)
- `TURBOANCHORKV_K_SPACE` (`pre` or `post`; default `pre`)
- `TURBOANCHORKV_RESIDUAL_TYPE` (`turbo2`, `turbo3`, `turbo4`, or `f16`; default `turbo2`)
- `TURBOANCHORKV_RATIO` (default `10`)
- `TURBOANCHORKV_PPL_TOKENS` (default `128`)
- `TURBOANCHORKV_GENERATE_TOKENS` (default `64`)
- `TURBOANCHORKV_EXPECT` (optional substring required in compressed output)
- `TURBOANCHORKV_MAX_TOKENS` (optional token cap after tokenization)
- `TURBOANCHORKV_DENSE_MODE` (`compress` or `identity`; default `compress`)
- `TURBOANCHORKV_KEEP_EDGE_LAYERS` (physical edge layers left exact; default `0`)
- `TURBOANCHORKV_SEED` (default `42`)
- `TURBOANCHORKV_Q_NAME`, `TURBOANCHORKV_K_PRE_NAME`, `TURBOANCHORKV_K_POST_NAME`, and `TURBOANCHORKV_V_NAME` override captured tensor names.

## Initial M5 Max results

Gemma 4 12B Q4_K_XL, Metal, f16 KV, about 8.5K prefix tokens:

| Test | Simulated ratio | Result |
|---|---:|---|
| Dense identity control | 1.0x | 0 PPL delta, 0 KL, 1.0 top-1 agreement |
| Docs continuation, packed turbo2 | 10.0x | +58.49% PPL, KL 0.685, 78.9% top-1 |
| Docs continuation, packed turbo4 | 5.0x | +10.70% PPL over 512 tokens, KL 0.894, 78.7% top-1 |
| Wikitext-2 excerpt, packed turbo4 | 5.0x | +23.86% PPL over 512 tokens, KL 1.166, 66.2% top-1 |
| Needle at 10%, 50%, and 90%, turbo4 | 5.0x | Exact baseline match and correct retrieval at all depths |
| Needle at 10%, turbo2 | 10.0x | Exact baseline match and correct retrieval |

The retrieval result is promising, but the PPL and logit drift fail a production-parity gate. Do not treat the one-layer attention proxy or needle pass as general quality evidence. A compact cache and Metal attention kernel should not be built until a configuration passes multi-model downstream gates.
