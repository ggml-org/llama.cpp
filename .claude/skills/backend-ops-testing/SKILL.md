---
name: backend-ops-testing
description: >
  Guide for writing and running backend-ops tests for GGML operations in llama.cpp.
  Covers the test framework, how to add new op tests, numerical tolerance, CPU vs GPU
  comparison, and running specific test subsets. Use when adding new GGML ops, verifying
  Vulkan/CUDA shader correctness, or debugging numerical mismatches between backends.
---

# Backend-Ops Testing for GGML Operations

## Overview

The backend-ops test framework (`tests/test-backend-ops.cpp`) verifies that GPU backend
implementations produce the same results as the CPU reference implementation. Every new
GGML op should have a backend-ops test.

## Running Tests

### Full test suite
```bash
./bin/test-backend-ops
```

### Vulkan backend only
```bash
./bin/test-backend-ops -b Vulkan0
```

### Specific op test
```bash
./bin/test-backend-ops -b Vulkan0 -o DELTA_NET_RECURRENCE
```

### List available ops
```bash
./bin/test-backend-ops -l
```

### With verbose output (shows individual test cases)
```bash
./bin/test-backend-ops -b Vulkan0 -o YOUR_OP -v
```

## Adding a New Op Test

### 1. Find the test registration section

In `tests/test-backend-ops.cpp`, find where similar ops are tested. SSM/attention ops
are typically near `test_ssm_scan`, `test_gated_linear_attn`, etc.

### 2. Create a test struct

```cpp
struct test_delta_net_recurrence : public test_case {
    const ggml_type type;
    const int64_t S;      // State dimension
    const int64_t H;      // Number of heads
    const int64_t n_seqs; // Batch size

    std::string vars() override {
        return VARS_TO_STR4(type, S, H, n_seqs);
    }

    test_delta_net_recurrence(ggml_type type = GGML_TYPE_F32,
                              int64_t S = 128, int64_t H = 4, int64_t n_seqs = 1)
        : type(type), S(S), H(H), n_seqs(n_seqs) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // Create input tensors matching the op's expected shapes
        ggml_tensor * state = ggml_new_tensor_4d(ctx, type, S, S, H, n_seqs);
        ggml_tensor * q     = ggml_new_tensor_3d(ctx, type, S, H, n_seqs);
        ggml_tensor * k     = ggml_new_tensor_3d(ctx, type, S, H, n_seqs);
        ggml_tensor * v     = ggml_new_tensor_3d(ctx, type, S, H, n_seqs);
        ggml_tensor * gate  = ggml_new_tensor_2d(ctx, type, 1, H * n_seqs);
        ggml_tensor * beta  = ggml_new_tensor_2d(ctx, type, 1, H * n_seqs);

        // Call the op
        ggml_tensor * out = ggml_delta_net_recurrence(ctx, q, k, v, gate, beta, state);
        return out;
    }
};
```

### 3. Register test cases

In the `get_test_cases()` function, add test instances with various parameters:

```cpp
// Delta-Net recurrence
test_cases.emplace_back(new test_delta_net_recurrence(GGML_TYPE_F32, 128, 4, 1));
test_cases.emplace_back(new test_delta_net_recurrence(GGML_TYPE_F32, 128, 8, 1));
test_cases.emplace_back(new test_delta_net_recurrence(GGML_TYPE_F32, 128, 4, 2));
test_cases.emplace_back(new test_delta_net_recurrence(GGML_TYPE_F32, 64, 16, 1));
```

Test dimensions to cover:
- Typical production sizes (S=128, H=8-32)
- Edge cases (S=1, H=1, n_seqs=1)
- Batch sizes (n_seqs=1, 2, 4)
- Multiple state dimensions if the shader specializes (e.g., d128 vs d256)

### 4. Set tolerance if needed

Override `max_nmse_err()` for ops with accumulated floating point error:

```cpp
double max_nmse_err() override {
    return 5e-4;  // Allow slightly higher error for ops with many FP accumulations
}
```

Default is `1e-4` which is fine for simple ops. Recurrence ops with many multiply-add
chains may need `5e-4` to `1e-3`.

## Building the Test Binary

```bash
# Add test-backend-ops to your ninja build targets
ninja -j 16 bin/test-backend-ops.exe
```

Or in the build batch file:
```bat
ninja -j 16 bin/llama-bench.exe bin/llama-cli.exe bin/test-backend-ops.exe
```

## Test Framework Internals

### How it works

1. Creates the compute graph on CPU backend
2. Runs the graph on CPU to get reference output
3. Creates the same graph on the target backend (e.g., Vulkan)
4. Runs the graph on the target backend
5. Compares outputs using NMSE (Normalized Mean Squared Error)
6. Reports PASS/FAIL with the error magnitude

### Input initialization

Test inputs are filled with random data by default. The framework handles:
- Random seed management for reproducibility
- Proper tensor allocation on both backends
- Buffer transfer between CPU and GPU

### What NMSE measures

```
NMSE = sum((cpu[i] - gpu[i])^2) / sum(cpu[i]^2)
```

A value of `1e-6` means the GPU output differs from CPU by 0.0001% of the signal energy.

## Debugging Test Failures

### Numerical mismatch (NMSE too high)

1. Run with `-v` to see which specific test case fails
2. Check for:
   - Accumulation order differences (GPU may sum in different order)
   - exp() or log() precision differences between CPU and GPU
   - Subnormal handling differences
   - Integer overflow in index calculations
3. Try increasing tolerance in `max_nmse_err()`
4. Compare a few specific values by adding debug prints to both CPU and GPU paths

### Op not found

- Check that `supports_op` returns `true` in the backend
- Verify the op is in the test registration (get_test_cases)
- Make sure the binary was rebuilt after adding the test

### Crash during test

- Usually a buffer binding mismatch or push constant size mismatch
- Check `parameter_count` in `ggml_vk_create_pipeline` matches shader bindings
- Check `push_constant_size` matches the shader's push constant block
- Verify workgroup dispatch dimensions are correct (not zero, not exceeding limits)

## Example: Verifying a New Op End-to-End

```bash
# 1. Build everything including test binary
cmd.exe //c "C:\Users\fabia\build_all.bat"
ninja -j 16 bin/test-backend-ops.exe

# 2. Run just your op on Vulkan
./bin/test-backend-ops.exe -b Vulkan0 -o DELTA_NET_RECURRENCE -v

# 3. If it passes, run the full suite to check for regressions
./bin/test-backend-ops.exe -b Vulkan0

# 4. Run the actual model to verify end-to-end
./bin/llama-bench.exe -m /path/to/model.gguf -ngl 99 -p 0 -n 32 -r 1
```
