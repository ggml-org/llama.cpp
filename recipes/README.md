# Quantization Recipes

A recipe file defines how each tensor category should be quantized. Recipes are an alternative to the built-in ftype-based logic in `llama_tensor_get_type_impl`.

## Usage

```bash
llama-quantize --recipe Q4_K_M model-f32.gguf model-q4km.gguf Q4_K
llama-quantize --recipe /path/to/custom.recipe model-f32.gguf model-out.gguf Q4_K
```

If the argument is not a file path, the tool looks for `recipes/<name>.recipe`.

## File Format

```
# Comments start with #

name    Q4_K_M
default Q4_K

[CATEGORY]
  conditions : TYPE
```

### Header

- `name` — recipe name (used for logging)
- `default` — fallback quantization type when no rule matches

### Categories

Each `[CATEGORY]` block contains rules for a group of tensors. Category names are case-insensitive.

| Category           | Tensors matched                              |
|--------------------|----------------------------------------------|
| `OUTPUT`           | output.weight (and tied token embeddings)     |
| `TOKEN_EMBD`       | token_embd.weight                            |
| `ATTENTION_V`      | attention value weights                      |
| `ATTENTION_K`      | attention key weights                        |
| `ATTENTION_Q`      | attention query weights                      |
| `ATTENTION_QKV`    | fused QKV weights                            |
| `ATTENTION_KV_B`   | fused KV bias weights                        |
| `ATTENTION_WV`     | combined: matches V, QKV, and KV_B           |
| `ATTENTION_OUTPUT` | attention output projection                  |
| `FFN_UP`           | feed-forward up projection                   |
| `FFN_GATE`         | feed-forward gate projection                 |
| `FFN_DOWN`         | feed-forward down projection                 |

These categories match those found in `llama-quant.h`

Tensors not covered by any category block use the `default` type.

### Rules

Rules are evaluated top-to-bottom — **last match wins**. Write general defaults first, then specific overrides after. Each rule has the form:

```
  conditions : TYPE
```

where `TYPE` is a ggml type like `Q4_K`, `Q6_K`, `Q8_0`, etc.

Use `*` for an unconditional rule (always matches):

```
  *          : Q6_K
```

Multiple conditions can be comma-separated. All conditions in a rule must be true (AND logic):

```
  arch=falcon, layer<1/16 : Q6_K
```

## Conditions

### `more_bits`

A built-in heuristic that allocates more bits to the first 1/8, last 1/8, and every third layer in between. Uses **category-relative** counts (e.g., position among all ATTENTION_V tensors). When used inside an `[ATTENTION_WV]` block, uses the combined counter across all attn_v-like tensors (V, QKV, KV_B).

```
  more_bits : Q6_K
```

### `layer`

Absolute layer position, parsed from the tensor name (`blk.%d.`). Values are fractions of total layer count.

```
  layer<1/8              : Q6_K    # first 1/8 of layers
  layer>=7/8             : Q6_K    # last 1/8 of layers
```

`layer=more_bits` applies the same heuristic as `more_bits` but using absolute layer position instead of category-relative position. This is important for FFN tensors in MoE models where the category counter includes per-expert tensors.

```
  layer=more_bits        : Q6_K
```

### `category`

Position within the category's own tensor count. Same fraction syntax as `layer`, but relative to how many tensors of this category exist.

```
  category<1/4           : Q5_K    # first quarter of this category's tensors
```

`category=more_bits` is equivalent to the bare `more_bits` keyword.

### `index`

Absolute sequential counter within the category (0-based).

```
  index<2                : Q5_K    # first two tensors of this category
```

### `arch`

Match model architecture. Case-insensitive.

```
  arch=falcon            : Q8_0
  arch!=falcon           : Q5_K
```

### `n_expert`

Number of experts in the model (1 for dense models).

```
  n_expert>=8            : Q8_0
```

### `n_gqa`

Number of grouped query attention heads.

```
  n_gqa>=4               : Q4_K
```

### `model_type`

Match the model's size designation (e.g., `7B`, `13B`, `70B`).

```
  model_type=70B         : Q5_K
```

### `has_imatrix` / `!has_imatrix`

Whether an importance matrix was provided.

```
  has_imatrix            : Q6_K
  !has_imatrix           : Q3_K
```

## Comparison Operators

| Operator | Meaning                |
|----------|------------------------|
| `=`      | equal                  |
| `!=`     | not equal              |
| `<`      | less than              |
| `<=`     | less than or equal     |
| `>`      | greater than           |
| `>=`     | greater than or equal  |

## Complete Example

```
name    Q4_K_M
default Q4_K

[OUTPUT]
  *                              : Q6_K
  arch=falcon                    : Q8_0

[ATTENTION_WV]
  model_type=70B                 : Q5_K
  more_bits                      : Q6_K
  n_expert=8                     : Q8_0

[ATTENTION_K]
  n_expert=8                     : Q8_0

[FFN_DOWN]
  layer=more_bits                : Q6_K
  arch=falcon, layer<1/16        : Q6_K
  arch=falcon, layer=more_bits   : Q5_K

[ATTENTION_OUTPUT]
  n_expert=8                     : Q5_K
```

Reading the `[ATTENTION_WV]` block: start with the default (Q4_K). If the model is 70B, override to Q5_K. If this tensor gets more bits (using the combined ATTENTION_WV counter), override to Q6_K. If there are exactly 8 experts, override to Q8_0. The last matching rule wins.
