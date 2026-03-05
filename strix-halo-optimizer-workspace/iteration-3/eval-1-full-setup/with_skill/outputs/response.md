# Running Llama 3.1 70B on AMD Ryzen AI Max+ 395 with llama.cpp

Your Framework Desktop with the Ryzen AI Max+ 395 and 128GB of unified LPDDR5X memory is well suited for running Llama 3.1 70B. The key is that this chip has a Radeon 890M integrated GPU (RDNA 3.5, 40 CUs) that shares the full 128GB memory pool with the CPU -- there is no separate VRAM, so the entire model can be offloaded to the GPU without PCIe transfer bottlenecks. Below is the complete setup.

---

## Step 1: Increase the TTM Memory Limit (Critical)

By default, the Linux kernel restricts how much system memory the GPU can access. Without this fix, you will get out-of-memory errors loading a 70B model even though you have 128GB of RAM.

Create the file `/etc/modprobe.d/increase_amd_memory.conf`:

```
options ttm pages_limit=25600000
options ttm page_pool_size=25600000
```

Then apply and reboot:

```bash
sudo update-initramfs -u -k all
sudo reboot
```

After reboot, verify:

```bash
cat /sys/module/ttm/parameters/pages_limit
# Should show: 25600000
```

With 4KB pages, this gives the GPU access to approximately 100GB of memory, which is enough for a 70B Q4_K_M model (roughly 40GB for weights plus KV cache and overhead).

---

## Step 2: Verify ROCm and GPU Detection

Since you already have ROCm 6.3 installed:

```bash
rocminfo | grep -i "Name:" | grep gfx
# Should show: gfx1150 (or gfx1151)
```

Make sure your user is in the required groups:

```bash
sudo usermod -a -G video,render $USER
# Log out and back in if needed
```

**Important:** Do NOT set `HSA_OVERRIDE_GFX_VERSION`. Older guides recommended this, but llama.cpp now has native RDNA 3.5 support. The override forces generic kernels that are slower than the native gfx1150 code paths.

---

## Step 3: Build llama.cpp

Use the optimized build configuration for Strix Halo:

```bash
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx1150" \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="-O3"

cmake --build build --config Release -j $(nproc)
```

What each flag does:

| Flag | Purpose |
|------|---------|
| `GGML_HIP=ON` | Enables the ROCm/HIP GPU backend |
| `GPU_TARGETS="gfx1150"` | Compiles kernels specifically for RDNA 3.5 instead of a bloated multi-architecture build. Cuts compile time from 30+ minutes to under 10 and produces better-optimized code. |
| `GGML_HIP_GRAPHS=ON` | Enables HIP graph capture to reduce kernel dispatch overhead |
| `GGML_CUDA_FORCE_MMQ=ON` | Forces quantized matrix multiplication kernels, which are often faster than hipBLAS on RDNA 3.5 for quantized models |
| `CMAKE_HIP_FLAGS="-O3"` | Maximum compiler optimization |

After building, verify GPU detection:

```bash
./build/bin/llama-bench --list-devices
# Should show: AMD Radeon Graphics (gfx1150 or gfx1151)
```

---

## Step 4: Download the Model

For the best balance of quality and speed on your hardware, use a Q4_K_M quantization of Llama 3.1 70B. This uses about 4.8 bits per weight and the total model size will be roughly 40GB, fitting comfortably in your 128GB unified memory.

If you want maximum token generation speed and can tolerate slightly lower quality, Q4_0 is the fastest quantization format.

---

## Step 5: Run with Optimal Settings

### Set the Critical Environment Variable

```bash
export ROCBLAS_USE_HIPBLASLT=1
```

This is essential. The default rocBLAS kernels for gfx1150 are poorly optimized and prompt processing will be 2-3x slower without this. Add it to your `~/.bashrc` so it persists across sessions.

### Run Inference

```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-cli -m llama-3.1-70b.Q4_K_M.gguf \
  -ngl 99 \
  --no-mmap \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 512 \
  -t 4
```

### Explanation of Each Runtime Flag

| Flag | Purpose |
|------|---------|
| `-ngl 99` | Offload all layers to the GPU. On unified memory, there is zero PCIe transfer penalty, so full GPU offload gives the best token generation speed. |
| `--no-mmap` | **Critical for large models.** Without this, HIP's `hipMemcpy()` must lock/unlock mmap pages, causing extreme slowdown during model loading. The 70B model is large enough to trigger this. |
| `-fa on` | Enables flash attention, which is WMMA-accelerated on RDNA 3.5. |
| `-ctk q8_0 -ctv q8_0` | Quantizes the KV cache to 8-bit, halving its memory footprint with negligible quality impact. This is important for a 70B model where you want headroom for longer contexts. |
| `-b 2048 -ub 512` | Batch sizes for prompt processing. These are good defaults for Strix Halo. |
| `-t 4` | Keep CPU threads low. Since CPU and GPU share the ~256 GB/s memory bus, high CPU thread counts compete with the GPU for bandwidth and actually hurt performance. 2-4 threads is optimal for GPU-primary inference. |

### If You Want to Run as a Server

```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-server \
  -m llama-3.1-70b.Q4_K_M.gguf \
  -ngl 99 \
  --no-mmap \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -c 8192 \
  -np 4 \
  -t 4 \
  --host 0.0.0.0 \
  --port 8080
```

---

## Step 6: Benchmark to Validate Performance

Run a benchmark sweep to confirm everything is working correctly:

```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-bench -m llama-3.1-70b.Q4_K_M.gguf \
  -ngl 99 \
  --no-mmap \
  -t 4 \
  -p 512 -n 128 \
  -r 3
```

For a 70B Q4_K_M model on this hardware, token generation speed will be constrained by the shared ~256 GB/s memory bandwidth. Since the model is roughly 40GB, the theoretical maximum is around `256 / 40 = ~6.4 tokens/s`, with real-world results typically at 60-70% of theoretical, so expect roughly 4-5 tokens/s for token generation. Prompt processing should be significantly faster with hipBLASLt enabled.

If you want to compare CPU-only vs GPU offloading (though full GPU offload should win for this model):

```bash
./build/bin/llama-bench -m llama-3.1-70b.Q4_K_M.gguf \
  -ngl 0,99 \
  --no-mmap \
  -t 4,16 \
  -p 512 -n 128 \
  -r 3 \
  -o csv > bench_results.csv
```

---

## Summary of Key Points

1. **Increase TTM limits** before anything else, or the GPU cannot access enough memory for a 70B model.
2. **Always set `ROCBLAS_USE_HIPBLASLT=1`** for 2-3x better prompt processing speed.
3. **Build with `GPU_TARGETS="gfx1150"`** for faster compilation and better-optimized kernels.
4. **Use `-ngl 99`** to fully offload to the GPU -- unified memory means no transfer penalty.
5. **Always use `--no-mmap`** for models this large to avoid HIP page-locking overhead.
6. **Keep CPU threads low (`-t 4`)** during GPU inference to avoid memory bandwidth contention.
7. **Do not use `HSA_OVERRIDE_GFX_VERSION`** -- native gfx1150 support is available and faster.
8. **Use Q4_K_M quantization** for the best quality/speed tradeoff, or Q4_0 for maximum speed.
