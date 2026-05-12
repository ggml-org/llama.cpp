# Speculative Decoding

llama.cpp supports speculative decoding, a technique that can significantly accelerate token generation by predicting multiple tokens ahead of the main model.

[Speculative decoding](https://en.wikipedia.org/wiki/Transformer_(deep_learning)#Speculative_decoding) leverages the fact that computing n tokens in a batch (as in prompt processing) is more efficient than computing n sequentially (as in response generation). By generating draft tokens quickly and then verifying them with the target model in a single batch, this approach can achieve substantial speedups when the draft predictions are frequently correct.

## Implementations

The `llama-server` application supports several implementations of speculative decoding. An implementation with draft model can be mixed with an implementation without draft model.

### Draft Model (`draft`)

A much smaller model (called the _draft model_) generates drafts.
A draft model is the most used approach in speculative decoding.

### n-gram Cache (`ngram-cache`)

An n-gram is a sequence of n tokens. The n-gram cache implementation maintains statistics about short n-gram sequences.
A draft is computed using probabilities derived from these statistics. External statistics can also be loaded from files for improved accuracy.

See:

- #5479, #6828, #6848

### n-gram Map (`ngram-simple`, `ngram-map-*`)

These implementations search the token history for patterns and use matching sequences as draft candidates.
They require no additional model but rely on patterns that have already appeared in the generated text.
An example to use this approach can be the rewriting of source code by a LLM.

#### n-gram Map (`ngram-simple`)

This implementation looks for the last n-gram in history that matches the current n-gram and creates a draft using the m tokens following the matched n-gram. It is the simplest self-speculative approach with minimal overhead.

```
llama-server [...] --spec-type ngram-simple --spec-draft-n-max 64
```

#### n-gram Map Key (`ngram-map-k`)

This implementation looks for the current n-gram of size n (called the _key_) in the token history. If the key n-gram is followed by the same m tokens (called the _mgram_) multiple times, it creates a draft using these m tokens. This approach requires a minimum number of occurrences (argument `--spec-ngram-map-k-min-hits`, default is 1) before generating drafts.

The number of accepted tokens is stored for each used n-gram.

**Example:**
```
llama-server [...] --spec-type ngram-map-k --spec-draft-n-max 64
```

#### n-gram Map Key-4-Values (`ngram-map-k4v`)

This experimental implementation looks for the current n-gram of size n (called the _key_) in the token history. For each key, up to four _values_ (n-grams of size m, called _mgrams_) are tracked. An internal statistic counts the occurrences of each mgram after the key n-gram. If one mgram is significantly more frequent than the others, it is used as the draft.

The number of accepted tokens is stored for each used n-gram.

**Example:** Server options to be used if there are a lot of longer repetitions.
```
llama-server [...] --spec-type ngram-map-k4v --spec-ngram-map-k4v-size-n 8 --spec-ngram-map-k4v-size-m 8 --spec-ngram-map-k4v-min-hits 2 --spec-draft-n-max 64
```

### n-gram Mod (`ngram-mod`)

Add basic ngram hasher for speculative decoding:

- For each ngram, compute a hash using LCG
- For each computed hash, store the next token
- During speculation, iteratively compute the rolling hash of the last n tokens and pick the next token from the storage

Some characteristics:

- Lightweight (~16 MB)
- Constant memory and complexity
- Can generate variable draft lengths (i.e. m is not fixed)

Currently, a single hash pool is shared across all server slots, so different requests can benefit from each other.

**Sample usage:**

```
# notes:
# - small `n` are not recommended
# - MoEs require long drafts
# - dense models: can reduce `--spec-ngram-mod-n-min` and `--spec-ngram-mod-n-max`

llama-server ... --spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
```

Applications:

- Iterating over a block of text/code (e.g. in llama.vim)
- Reasoning models (when they have to repeat their thinking in the final answer)
- Summarization

Example Video:

- See #19164

### Differences between ngram-simple, ngram-map and ngram-mod

- ngram-simple looks for a previous matching n-gram and inserts the following m-gram.
- ngram-map-k looks for a previous matching n-gram and inserts the following m-gram but uses an internal hash-map of n-grams in the current context window.
- ngram-mod uses a hash pool which is shared across all server slots. The hash pool is a map from n-gram hash to the next token (not the next m-gram as in ngram-map).

## Command-Line Options

If a draft model is combined with a draftless decoding the draftless decoding has higher precedence.

### General Speculative Parameters

```
--spec-type [none|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]
                                        type of speculative decoding to use when no draft model is provided
                                        (default: none)
                                        (env: LLAMA_ARG_SPEC_TYPE)
--spec-default                          use default speculative decoding
```

### Draft Model Parameters

```
--spec-draft-model, -md, --model-draft  FNAME
                                        draft model for speculative decoding (default: unused)
                                        (env: LLAMA_ARG_SPEC_DRAFT_MODEL)
--spec-draft-hf, -hfd, -hfrd, --hf-repo-draft  <user>/<model>[:quant]
                                        HuggingFace repository for the draft model
--spec-draft-n-max                      N
                                        number of tokens to draft for speculative decoding (default: 16)
                                        (env: LLAMA_ARG_SPEC_DRAFT_N_MAX)
--spec-draft-n-min                      N
                                        minimum number of draft tokens to use for speculative decoding (default: 0)
                                        (env: LLAMA_ARG_SPEC_DRAFT_N_MIN)
--spec-draft-p-split, --draft-p-split   P
                                        speculative decoding split probability (default: 0.10)
                                        (env: LLAMA_ARG_SPEC_DRAFT_P_SPLIT)
--spec-draft-p-min, --draft-p-min       P
                                        minimum speculative decoding probability (greedy) (default: 0.75)
                                        (env: LLAMA_ARG_SPEC_DRAFT_P_MIN)
--spec-draft-ctx-size, -cd, --ctx-size-draft  N
                                        size of the prompt context for the draft model (default: 0, 0 = loaded from model)
                                        (env: LLAMA_ARG_SPEC_DRAFT_CTX_SIZE)
--spec-draft-ngl, -ngld, --gpu-layers-draft, --n-gpu-layers-draft  N
                                        max. number of draft model layers to store in VRAM, either an exact number, 'auto', or 'all' (default: auto)
                                        (env: LLAMA_ARG_N_GPU_LAYERS_DRAFT)
--spec-draft-device, -devd, --device-draft  <dev1,dev2,..>
                                        comma-separated list of devices to use for offloading the draft model
--spec-draft-replace, --spec-replace    TARGET  DRAFT
                                        translate the string in TARGET into DRAFT if the draft model and main model are not compatible
```

### n-gram Mod Parameters

```
--spec-ngram-mod-n-match                N
                                        ngram-mod lookup length (default: 24)
--spec-ngram-mod-n-min                  N
                                        minimum number of ngram tokens to use for ngram-based speculative decoding (default: 48)
--spec-ngram-mod-n-max                  N
                                        maximum number of ngram tokens to use for ngram-based speculative decoding (default: 64)
```

### n-gram Simple Parameters

```
--spec-ngram-simple-size-n              N
                                        ngram size N for ngram-simple speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-simple-size-m              N
                                        ngram size M for ngram-simple speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-simple-min-hits            N
                                        minimum hits for ngram-simple speculative decoding (default: 1)
```

### n-gram Map Key Parameters

```
--spec-ngram-map-k-size-n               N
                                        ngram size N for ngram-map-k speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-map-k-size-m               N
                                        ngram size M for ngram-map-k speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-map-k-min-hits             N
                                        minimum hits for ngram-map-k speculative decoding (default: 1)
```

### n-gram Map Key-4-Values Parameters

```
--spec-ngram-map-k4v-size-n             N
                                        ngram size N for ngram-map-k4v speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-map-k4v-size-m             N
                                        ngram size M for ngram-map-k4v speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-map-k4v-min-hits           N
                                        minimum hits for ngram-map-k4v speculative decoding (default: 1)
```

### `--spec-type TYPE`

Specifies a type of speculative decoding without draft model.

| Type | Description |
|------|-------------|
| `none` | No speculative decoding (default) |
| `ngram-cache` | Use n-gram cache lookup |
| `ngram-simple` | Use simple n-gram pattern matching |
| `ngram-map-k` | Use n-gram pattern matching with n-gram-keys |
| `ngram-map-k4v` | Use n-gram pattern matching with n-gram-keys and up to four m-gram values (experimental) |
| `ngram-mod` | Use basic ngram hasher for speculative decoding with shared pool |

**Example:** Server-instance used to refactor source code.
```bash
./llama-server [...] --spec-type ngram-simple
```

### `--spec-ngram-*-size-n N`

Sets the size N of the lookup n-gram for n-gram map based speculative decoding.
The n-gram size N determines how many tokens in a row to look back when searching for matching patterns.

Each n-gram implementation has its own parameter:

- `--spec-ngram-simple-size-n` for `ngram-simple`
- `--spec-ngram-map-k-size-n` for `ngram-map-k`
- `--spec-ngram-map-k4v-size-n` for `ngram-map-k4v`
- `--spec-ngram-mod-n-match` for `ngram-mod`

### `--spec-ngram-*-size-m M`

Sets the size M of the draft m-gram for n-gram map based speculative decoding.
The m-gram size determines how many tokens to draft when a match is found.
Larger values can provide more speedup but may reduce acceptance rate.

Each n-gram implementation has its own parameter:

- `--spec-ngram-simple-size-m` for `ngram-simple`
- `--spec-ngram-map-k-size-m` for `ngram-map-k`
- `--spec-ngram-map-k4v-size-m` for `ngram-map-k4v`

### `--spec-ngram-*-min-hits H`

This option defines how often a key has to appear in the token history to be used as a draft (default is 1).

Each n-gram implementation has its own parameter:

- `--spec-ngram-simple-min-hits` for `ngram-simple`
- `--spec-ngram-map-k-min-hits` for `ngram-map-k`
- `--spec-ngram-map-k4v-min-hits` for `ngram-map-k4v`

## Statistics
Each speculative decoding implementation prints statistics.

```
draft acceptance rate = 0.57576 (  171 accepted /   297 generated)
statistics ngram_simple: #calls = 15, #gen drafts = 5, #acc drafts = 5, #gen tokens = 187, #acc tokens = 73
statistics draft: #calls = 10, #gen drafts = 10, #acc drafts = 10, #gen tokens = 110, #acc tokens = 98
```

```
draft acceptance rate = 0.70312 (   90 accepted /   128 generated)
statistics ngram_mod: #calls = 810, #gen drafts = 15, #acc drafts = 15, #gen tokens = 960, #acc tokens = 730, dur(b,g,a) = 0.149, 0.347, 0.005 ms
```

```
statistics ngram_map_k: #calls(b,g,a) = 6 1690 26, #gen drafts = 26, #acc drafts = 26, #gen tokens = 1248, #acc tokens = 968, dur(b,g,a) = 2.234, 1.427, 0.016 ms
```


- `#calls(b,g,a)`: number of calls of begin (new prompt), generation and accumulation of this implementations
- `#gen drafts`: number of drafts generated by this implementation
- `#acc drafts`: number of drafts accepted (partially) by the main model
- `#gen tokens`: number of tokens generated by this implementation (including rejected tokens)
- `#acc tokens`: number of tokens accepted by the main model
- `dur(b,g,a): durations of begin (new prompt), generation and accumulation (process acceptance).
