---
name: slice-model
description: Build a small test "slice" of a HF model (fewer layers/experts) by downloading only the needed tensors via HTTP range requests. Use when the user needs a small version of a big model to test conversion or inference.
---

# Slice a HF model for testing

[slice_model.py](slice_model.py) is a **reference** implementation. Read it first: its docstring and `--help` explain how it works and what it does. Its heuristics only cover common layouts, and new models often have tensor names, shapes or config keys it does not handle. Do not run it blindly. If the model needs special handling, copy it to `tmp/slice_<model>.py` and adapt the copy, or write a new script with the same approach.

## Steps

1. Ask the user for the repo, what the slice must exercise (e.g. a MoE layer, each layer type of a hybrid model, MTP, the vision encoder), the size budget and the output dir.
2. Inspect the model: run the script with `--inspect`, read `config.json`, and read the HF modeling code if anything is unclear. Check the plan against the pitfalls below.
3. Build the slice, then read the config patches in the log and check each one.
4. Verify: load the slice in transformers (no missing/unexpected weight warnings, no NaN logits), then convert it with `convert_hf_to_gguf.py` and run it. For numerical checks, see `examples/model-conversion/`.

## Pitfalls

- Layer counts in nested or unknown config keys, or several towers with the same layer count.
- Hybrid layer patterns (`layer_types`, `full_attention_interval`, ...): keep at least one layer of each type, and check the patched config still matches the kept layers.
- Dense leading layers (`first_k_dense_replace`) and MTP/nextn layers numbered after the regular layers.
- Every per-expert tensor must be sliced to the same expert count: experts, router weight and biases. Tensors with the expert dim not at dim 0 are not contiguous.
- Routing constraints: top-k, expert groups, shared experts, zero-compute experts.
- Quantized weights (FP8, MXFP4): slice the scale tensors in the same way as their weight.
- Sizes derived in modeling code instead of stored in the config: follow the exact formula if you shrink them.
- Slicing the vocab without the tokenizer breaks conversion.
