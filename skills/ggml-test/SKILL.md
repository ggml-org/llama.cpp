---
name: ggml-test
description: Test a single ggml op or a small piece of a cgraph in isolation, by compiling a minimal standalone ggml program and comparing it against a reference implementation (usually PyTorch). Use when porting a model component (e.g. from HF transformers) and you want to validate it before wiring it into the full graph, or when debugging a specific op/subgraph (custom RoPE variant, interpolation, pooling, an isolated sub-block like AltUp) without running the whole model.
---

# Test a ggml op / cgraph fragment in isolation

Porting a whole model end-to-end and only then comparing final logits makes it hard to find *which* op is wrong. This skill validates one component at a time: write a tiny PyTorch reference with randomly-initialized tensors, dump its inputs/weights/output to GGUF, write a minimal standalone ggml program that runs the same op sequence, and compare the two outputs numerically. This is a scratch/throwaway harness, not a permanent test -- it lives in `tmp/` (gitignored) and gets deleted once the component is confirmed correct and merged into the real graph-building code.

Good times to reach for this:
- A new architecture has a non-standard sub-component developed/ported independently (e.g. an AltUp block, a custom mixer/state-space component) before it's wired into the full model graph.
- Validating one op's semantics against a reference before trusting it inside a much bigger graph (e.g. a multi-dimensional RoPE variant, an interpolation/resize mode, a custom pooling or normalization).
- Debugging a numerical mismatch in `clip.cpp` / `mtmd/models/*.cpp` / `src/models/*.cpp` where the whole-model comparison (see `examples/model-conversion/README.md`) shows a mismatch but doesn't say where.

This complements, it does not replace, the full logits-verification workflow in `examples/model-conversion/README.md` and the `add-new-model` skill -- use this to isolate a failure or de-risk a component *before* or *while* doing that full-model verification.

## Workflow

### 0. Confirm there's a real prerequisite

You need: (a) the reference implementation's source (usually a HF `transformers` `modeling_*.py` file -- fetch the specific class/method, don't rely on memory of what it does), and (b) the ggml code under test already written (even if unverified), so you know exactly which op sequence to transcribe. If the ggml side doesn't exist yet, write it first (following the `add-new-model` skill's conventions), then come back here to validate it.

### 1. Write a minimal PyTorch reference

Only implement the component under test, not the surrounding model. Use small, randomly-initialized tensors (`torch.manual_seed(...)` for reproducibility) with the smallest shapes that still exercise the interesting behavior (e.g. a handful of patches/tokens, small hidden size) -- this keeps compile/run iteration fast and makes mismatches easy to eyeball. Save every tensor the ggml side will need as an input (weights, indices, etc.) plus the expected output, using the helper in `scripts/gguf_io.py`:

```python
import sys
sys.path.insert(0, "skills/ggml-test/scripts")
from gguf_io import save_tensors

save_tensors("tmp/<component>_in.gguf", {
    "some_weight": weight_tensor.numpy().astype(np.float32),
    "some_indices": idx_tensor.numpy().astype(np.int32),
})
save_tensors("tmp/<component>_ref.gguf", {
    "out": expected_output.numpy().astype(np.float32),
})
```

Notes:
- `save_tensors` preserves each array's numpy shape/dtype as-is; gguf-py reverses the axis order under the hood to match ggml's `ne[]` convention (numpy's last axis becomes ggml's `ne[0]`), same as regular model conversion. You don't need to transpose anything by hand.
- Use `int32` for any tensor that becomes a ggml `I32` index tensor (e.g. positions fed to `ggml_get_rows`/`ggml_rope_ext`) -- ggml has no int64 op support for these.
- Split "inputs" and "reference output" into separate files (or just separate tensor names in one file) so the ggml program only has to read what it needs to build the graph, and the comparison step only has to read what it needs to check.

### 2. Write the minimal ggml program

Put it in `tmp/` (gitignored). Structure:

1. Load the input GGUF with `gguf_init_from_file(path, {no_alloc=false, ctx=&ctx_data})`. This gives you a `ggml_context` whose tensors already have their data loaded in plain CPU memory (`ggml_get_tensor(ctx_data, name)`) -- no backend buffer juggling needed for a CPU-only test.
2. Build a *second*, `no_alloc=true` context for the graph, and **copy-paste the actual op sequence from the real source file into it** -- don't reimplement/paraphrase it. Swap struct-member references (`model.foo`, `ctx0`, `hparams.bar`) for local variables holding the same values, but keep the `ggml_*` calls themselves verbatim. This is the entire point: you're testing the exact code that will ship, not a restatement of it.
3. `ggml_backend_cpu_init()` + `ggml_gallocr_new(ggml_backend_cpu_buffer_type())` + `ggml_gallocr_alloc_graph(...)`, then `ggml_backend_tensor_set(...)` the graph's input tensors from the data loaded in step 1, then `ggml_backend_graph_compute(...)`.
4. Dump the result tensor to a GGUF file with `gguf_add_tensor` + `gguf_write_to_file` so the comparison step can read it back.

See `references/gemma4v_pos_embd_example.md` for a complete, verified worked example covering all four steps.

### 3. Compile and run

```bash
skills/ggml-test/scripts/build_and_run.sh tmp/test_<component>.cpp tmp/<component>_in.gguf tmp/<component>_out.gguf
```

This links directly against the already-built `libggml*` in the project's CMake build dir (default `build/`, override with `GGML_TEST_BUILD_DIR`) -- no need to add a target to the project's `CMakeLists.txt` for a throwaway test. Requires the project to have been built at least once already.

### 4. Compare

```bash
python3 skills/ggml-test/scripts/compare_tensors.py tmp/<component>_ref.gguf tmp/<component>_out.gguf
```

Reports, per tensor name, shape, max abs diff, mean abs diff, relative L2 diff, and a pass/fail against `--rtol`/`--atol` (defaults `1e-3`/`1e-4` -- loosen for f16/bf16 or tighten for a pure-f32 op with no reduction). A shape mismatch is reported explicitly since it usually means a transpose/reshape assumption is wrong, not a numerical issue.

### 5. Iterate, then clean up

Fix the ggml side (or discover the PyTorch reference itself was wrong -- re-check against the HF source), re-run steps 3-4 until it passes. Once confirmed, port the validated op sequence into the real graph-building file if it wasn't already there, and delete the scratch files under `tmp/` (they're gitignored, but delete them anyway so they don't linger and get mistaken for still-relevant scratch work).

## Common pitfalls

- Forgetting `ggml_set_input()`/`ggml_set_output()` on the tensors you need to set/read -- without these the graph allocator is free to reuse/overwrite their memory.
- Testing with shapes so small that a bug that depends on stride/alignment (e.g. an off-by-one in a view offset, a wrong `nb[]` in `ggml_view_*`) doesn't get exercised. Prefer shapes where every dimension has a different size, so a transposed axis or swapped stride shows up as a shape or numeric mismatch instead of silently working.
- Comparing f16/bf16 ops with the same tight tolerance as f32 -- loosen `--rtol`/`--atol` accordingly, or cast both sides to f32 before comparing if you only care about the op logic and not quantization-induced error.
- Not resetting `torch.manual_seed(...)` -- without it the reference becomes non-reproducible across runs, which makes it impossible to tell whether a fix actually changed anything.
