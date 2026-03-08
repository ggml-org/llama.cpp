---
name: ggml-op-development
description: >
  Step-by-step checklist for adding a new GGML op to llama.cpp, covering all files
  that must be modified across the enum, name/symbol tables, CPU backend, Vulkan backend,
  and testing. Use this skill whenever implementing a new fused kernel, recurrence op,
  or any custom GGML operation. Prevents the "too many initializers" and static_assert
  bugs that come from missing or misordered entries.
---

# Adding a New GGML Op to llama.cpp

## Pre-flight

- Decide on the op name: `GGML_OP_<YOUR_OP>` (e.g., `GGML_OP_DELTA_NET_RECURRENCE`)
- Identify which backends will support it (CPU is mandatory, Vulkan/CUDA/Metal optional)
- Count the number of input tensors (max 10 via `src[0]`..`src[9]`)
- Design the output tensor shape (combined output+state pattern if recurrent)

## Checklist: 8 Files Minimum

### 1. `ggml/include/ggml.h` - Enum + Function Declaration

- [ ] Add `GGML_OP_<YOUR_OP>` to the `enum ggml_op` list
- [ ] **Position matters**: Insert in a logical group (e.g., after `GGML_OP_GATED_LINEAR_ATTN` for attention-like ops). The position in the enum determines the array index in the name/symbol tables.
- [ ] Add the public API function declaration:
```c
GGML_API struct ggml_tensor * ggml_your_op(
        struct ggml_context * ctx,
        struct ggml_tensor  * input1,
        struct ggml_tensor  * input2,
        ...);
```

### 2. `ggml/src/ggml.c` - Name Table + Symbol Table + Implementation

This is where most bugs happen. Three separate arrays must be updated:

- [ ] **GGML_OP_NAME table** (~line 944): Add `"YOUR_OP"` string at the **same position** as the enum entry. Count from the top to verify index alignment.
- [ ] **GGML_OP_SYMBOL table** (~line 1055): Add `"your_op(args)"` string at the **same position**. This table is parallel to the name table.
- [ ] **static_assert**: Update both `GGML_OP_COUNT == N` assertions (after each table) from N to N+1.
- [ ] **Function implementation**: Add the `ggml_your_op()` function that creates the result tensor, sets `result->op`, assigns `result->src[0..N]`.

**CRITICAL BUG PREVENTION**: The name and symbol tables are `[GGML_OP_COUNT]` arrays. If you add the string at the wrong position (e.g., at the end instead of matching the enum order), you get "too many initializers" or worse, silent misalignment where op names don't match their ops.

**Verification technique**: After editing, count the entries before your new one in both the enum and both tables. The counts must match.

### 3. `ggml/src/ggml-cpu/ops.h` - CPU Forward Declaration

- [ ] Add: `void ggml_compute_forward_your_op(const struct ggml_compute_params * params, struct ggml_tensor * dst);`

### 4. `ggml/src/ggml-cpu/ops.cpp` - CPU Implementation

- [ ] Implement the full computation in C/C++
- [ ] This is the reference implementation that Vulkan/CUDA results are compared against
- [ ] Parallelize over heads/sequences using `ggml_compute_params` thread info
- [ ] Place near similar ops (e.g., after `ggml_compute_forward_gla` for attention ops)

### 5. `ggml/src/ggml-cpu/ggml-cpu.c` - CPU Dispatch

Two locations:

- [ ] **Forward dispatch** (in `ggml_compute_forward`): Add `case GGML_OP_YOUR_OP:` calling your forward function
- [ ] **Thread count** (in `ggml_cpu_get_n_tasks`): Add to the appropriate group:
  - `n_tasks = n_threads` for parallelizable ops (most ops)
  - `n_tasks = 1` for ops that can't parallelize

### 6. `ggml/src/ggml-vulkan/vulkan-shaders/your_shader.comp` - GLSL Shader (if Vulkan)

- [ ] Write the compute shader
- [ ] Define push constants struct matching the C++ side
- [ ] Use `layout(std430, binding = N)` for each buffer
- [ ] Set appropriate `local_size_x` (128 is common, match to problem dimension)

### 7. `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` - Shader Registration

- [ ] Add: `string_to_spv("your_shader_f32", "your_shader.comp", {});`
- [ ] If using specialization constants: `string_to_spv("your_shader_f32", "your_shader.comp", {128, 64});`
- [ ] **Must rebuild shader-gen separately** (it's an ExternalProject):
```
# Build shader-gen
cd build-win/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-build
ninja -j 16
# Then run shader-gen or let main build do it
```

### 8. `ggml/src/ggml-vulkan/ggml-vulkan.cpp` - Vulkan Backend (6 locations)

- [ ] **Pipeline declaration** (~line 814): `vk_pipeline pipeline_your_op_f32;` in `vk_device_struct`
- [ ] **Push constants struct** (~line 1075): Define matching the shader's push constants
- [ ] **Pipeline creation** (~line 4500): Call `ggml_vk_create_pipeline(device, device->pipeline_your_op_f32, "your_shader_f32", ...)`
  - Parameters: name, spv_len, spv_data, "main", num_bindings, push_constant_size, wg_denoms, specialization_constants, align, disable_robustness, require_full_subgroups
- [ ] **Pipeline selection** in `ggml_vk_op_get_pipeline` (~line 9039): Add `case GGML_OP_YOUR_OP:` returning the pipeline
- [ ] **Dispatch function**: Create `ggml_vk_your_op()` that sets up buffers, push constants, and calls `ggml_vk_dispatch_pipeline`
- [ ] **Graph build** in `ggml_vk_build_graph` (~line 12597): Add `case GGML_OP_YOUR_OP:` calling your dispatch
- [ ] **supports_op** (~line 13500): Return `true` for your op (with type/shape constraints)

## Post-Implementation

- [ ] Build and fix any static_assert or "too many initializers" errors
- [ ] Run CPU test first (quick sanity check)
- [ ] Run Vulkan test and compare output against CPU reference
- [ ] Add backend-ops test (see `backend-ops-testing` skill)
- [ ] Benchmark to verify no regression on unrelated models
- [ ] Update memory files with the new op details

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Entry at wrong position in name/symbol table | "too many initializers" or silent name mismatch | Count entries from top, must match enum order |
| Forgot to update static_assert | `GGML_OP_COUNT != N` compile error | Update both assertions (after name table AND symbol table) |
| Forgot shader-gen rebuild | Old shader used, new shader ignored | Rebuild shader-gen ExternalProject separately |
| Push constant struct mismatch | GPU garbage output or crash | Verify C++ struct layout matches GLSL push_constant block exactly |
| Wrong binding count in create_pipeline | Assertion failure at runtime | Count bindings in shader, must match parameter_count arg |
| Forgot supports_op | Op silently falls back to CPU | Add case in Vulkan supports_op function |
