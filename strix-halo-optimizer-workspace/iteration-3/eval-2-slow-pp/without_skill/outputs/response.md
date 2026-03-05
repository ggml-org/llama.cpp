# Diagnosing Slow Prompt Processing on AMD Strix Halo

Your symptom -- 350 t/s prompt processing (pp) but ~48 t/s token generation (tg) on a 7B model -- points to a classic configuration problem: **prompt processing is running on the CPU instead of being offloaded to the GPU, or the CPU backend is not using optimal SIMD/threading settings**. Token generation is memory-bandwidth-bound and less affected, which is why tg looks normal while pp is severely bottlenecked.

Here is a systematic diagnosis and set of fixes.

---

## 1. Offload All Layers to the Radeon 890M iGPU (Most Likely Cause)

The single biggest factor for prompt processing speed is whether the matrix multiplications are running on the GPU. A 7B Q4 model is roughly 3.5-4.5 GiB, which fits easily in Strix Halo's shared LPDDR5X memory when using the Vulkan backend.

**Check:** Look at your startup log for a line like `llm_load_tensors: offloaded X/Y layers to GPU`. If X is 0 or low, that is your problem.

**Fix:** Pass `-ngl 99` (or `--gpu-layers 99`) to offload all layers to the iGPU:

```bash
./llama-bench -m your-model.gguf -ngl 99 -fa on
```

Or when running `llama-cli` / `llama-server`:

```bash
./llama-cli -m your-model.gguf -ngl 99 -fa on -p "your prompt"
```

With full GPU offload on a Vulkan-enabled build, prompt processing on a 7B Q4_K_M model should reach **1500-3000+ t/s** rather than 350 t/s.

---

## 2. Ensure You Built with Vulkan Support

Strix Halo's Radeon 890M (RDNA 3.5) iGPU is best accessed via the Vulkan backend in llama.cpp. If you built without `-DGGML_VULKAN=ON`, then `-ngl` has no GPU backend to offload to, and everything runs on the CPU.

**Check:** Look at the startup log for `ggml_vulkan: Using AMD Radeon ...` or similar. If you only see `CPU` as the backend, Vulkan is not active.

**Fix:** Rebuild with Vulkan enabled:

```bash
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)
```

Make sure the Vulkan SDK / runtime is installed:
```bash
# Debian/Ubuntu
sudo apt-get install libvulkan-dev glslc

# Verify
vulkaninfo | head -20
```

---

## 3. Increase Batch Size (`-b` and `-ub`)

Prompt processing works by evaluating the entire prompt in batches. The default logical batch size is 2048 and the physical ubatch size is 512. If you or your application has overridden these to small values, pp speed drops dramatically.

**Fix:** Use large batch sizes explicitly:

```bash
./llama-cli -m model.gguf -ngl 99 -b 2048 -ub 512 -p "your prompt"
```

For benchmarking, try even larger ubatch sizes to see the peak throughput:

```bash
./llama-bench -m model.gguf -ngl 99 -b 2048 -ub 512,1024,2048
```

As shown in the llama-bench documentation, increasing batch size from 128 to 1024 can nearly double pp throughput.

---

## 4. Enable Flash Attention

Flash attention reduces memory traffic and can significantly speed up prompt processing, especially for longer prompts.

**Fix:**

```bash
./llama-cli -m model.gguf -ngl 99 -fa on
```

Or equivalently: `--flash-attn on`

---

## 5. CPU-Only Scenario: Fix Thread Count

If you are intentionally running CPU-only (no `-ngl`), then 350 t/s on a Zen 5 chip with 12 cores is low but plausible if the thread count is wrong.

**How llama.cpp picks threads by default:** On Linux x86_64, the default thread count is `hardware_concurrency() / 2`. Strix Halo (Ryzen AI 300 series) typically has 12 cores / 24 threads, so the default would be 12 threads. This is correct for Zen 5 since it does not have efficiency cores (the Intel hybrid-core detection in the code does not apply to AMD).

**However**, if your system reports an unexpected number (e.g., some cores are offline, or you are running inside a VM/container), the auto-detection may pick too few threads.

**Fix:** Explicitly set threads, especially batch threads:

```bash
./llama-cli -m model.gguf -t 12 -tb 12 -p "your prompt"
```

The `-tb` (`--threads-batch`) flag controls threads used during prompt processing specifically. If unset, it defaults to the same as `-t`.

---

## 6. Verify AVX-512 Is Being Used

Strix Halo's Zen 5 cores support AVX-512. If the build does not use AVX-512 (e.g., you used a pre-built binary compiled for a generic target), prompt processing on CPU will be significantly slower.

**Check:** Look at the startup log for the `system_info` line. It should show `AVX512 = 1`.

**Fix:** Build with native optimizations:

```bash
cmake -B build -DGGML_NATIVE=ON -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)
```

`-DGGML_NATIVE=ON` compiles with `-march=native`, enabling AVX-512 and all Zen 5-specific instruction sets.

---

## 7. NUMA Considerations

Strix Halo is a monolithic die (single CCX for most SKUs), so NUMA is generally not an issue. However, if your system for some reason exposes multiple NUMA nodes (e.g., the NPU or iGPU memory appears as a separate node), you can try:

```bash
./llama-cli -m model.gguf --numa distribute -ngl 99
```

---

## Summary: Most Likely Fix

Given that your tg speed (48 t/s) is reasonable for a 7B model on CPU but your pp speed (350 t/s) is very low, the most likely scenario is:

1. **You are running CPU-only without GPU offload.** A 7B model on CPU with 12 Zen 5 threads at AVX-512 might achieve 300-500 t/s pp, which matches your 350 t/s observation. Adding `-ngl 99` with a Vulkan build should increase pp to 1500-3000+ t/s.

2. **If you already have `-ngl 99` set**, then the Vulkan backend may not have built or loaded correctly. Check the logs for Vulkan initialization messages.

**Recommended command for best Strix Halo performance:**

```bash
# Build
cmake -B build -DGGML_VULKAN=ON -DGGML_NATIVE=ON
cmake --build build --config Release -j$(nproc)

# Run
./build/bin/llama-cli \
    -m your-7B-model-Q4_K_M.gguf \
    -ngl 99 \
    -fa on \
    -b 2048 \
    -ub 512 \
    -t 12 \
    -tb 12 \
    -p "your prompt here"
```

**Benchmark to verify:**

```bash
./build/bin/llama-bench \
    -m your-7B-model-Q4_K_M.gguf \
    -ngl 0,99 \
    -fa 0,1 \
    -t 12
```

This will compare CPU-only vs full GPU offload, with and without flash attention, so you can see the exact impact of each setting.
