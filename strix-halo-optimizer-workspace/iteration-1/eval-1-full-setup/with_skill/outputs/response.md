# Running Llama 3.1 70B on Framework Desktop (AMD Ryzen AI Max+ 395) with llama.cpp

Great hardware choice — the Ryzen AI Max+ 395 (Strix Halo) with 128GB unified memory is well-suited for running Llama 3.1 70B. Here's everything you need to get maximum token generation speed.

---

## Step 1: Critical System Configuration (Do This Before Building)

Before anything else, you need to increase the Linux kernel's TTM memory limits. Without this, the GPU can only access a small fraction of your 128GB RAM, and a 70B model will hit out-of-memory errors even though you have plenty of free RAM.

Create the configuration file:

```bash
sudo nano /etc/modprobe.d/increase_amd_memory.conf
```

Add these two lines:

```
options ttm pages_limit=25600000
options ttm page_pool_size=25600000
```

Apply and reboot:

```bash
sudo update-initramfs -u -k all
sudo reboot
```

After rebooting, verify it took effect:

```bash
cat /sys/module/ttm/parameters/pages_limit
# Should show: 25600000
```

With 4KB pages, this allows ~100GB of GPU-accessible memory, which is sufficient for Llama 3.1 70B in Q4 quantization (~35-40GB).

---

## Step 2: Verify Your GPU Architecture

First, confirm what GPU architecture identifier your system reports:

```bash
rocminfo | grep -i "Name:" | grep gfx
```

It should show `gfx1150` or `gfx1151`. You'll need this for the build step. Most Ryzen AI Max+ 395 systems report `gfx1150`.

Also verify your user has GPU access:

```bash
groups | grep -E "video|render"
```

If not, add yourself:

```bash
sudo usermod -a -G video,render $USER
# Log out and back in
```

---

## Step 3: Build llama.cpp

Clone the repository if you haven't already:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

Use the recommended optimized build command for Strix Halo:

```bash
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx1150" \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="-O3"

cmake --build build --config Release -j $(nproc)
```

**Key build flags explained:**

| Flag | Why It Matters |
|------|----------------|
| `DGGML_HIP=ON` | Enables the ROCm/HIP GPU backend |
| `GPU_TARGETS="gfx1150"` | Compiles kernels specifically for your RDNA 3.5 iGPU — reduces compile time from 30+ minutes to under 10 and produces better-optimized code than a generic multi-arch build |
| `DGGML_HIP_GRAPHS=ON` | Reduces GPU kernel dispatch overhead |
| `DGGML_CUDA_FORCE_MMQ=ON` | Forces quantized matmul kernels instead of hipBLAS — often faster on RDNA 3.5 for quantized models like Q4_K_M |
| `DGGML_NATIVE=ON` | Auto-detects and enables all CPU features (AVX-512, BF16) on your Zen 5 |
| `DCMAKE_HIP_FLAGS="-O3"` | Full optimization for HIP kernels |

If `rocminfo` showed `gfx1151` instead of `gfx1150`, change `GPU_TARGETS="gfx1150"` to `GPU_TARGETS="gfx1151"`. If you're unsure, you can build for both: `GPU_TARGETS="gfx1150;gfx1151"` (slightly longer compile time).

After building, verify the GPU is detected:

```bash
./build/bin/llama-bench --list-devices
# Should show: AMD Radeon Graphics (gfx1150 or gfx1151)
```

**Important**: Do not use `HSA_OVERRIDE_GFX_VERSION`. Older guides recommended this to make Strix Halo impersonate a different GPU, but llama.cpp now has native RDNA 3.5 support. Using the override forces generic kernels that are slower than the native gfx1150 code paths.

---

## Step 4: Get a Quantized Model

For Llama 3.1 70B on this hardware, **Q4_K_M** offers the best balance of quality and speed. It fits comfortably in your 128GB and runs well on the iGPU.

Download from Hugging Face (example using huggingface-cli):

```bash
pip install huggingface_hub
huggingface-cli download bartowski/Meta-Llama-3.1-70B-Instruct-GGUF \
  --include "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf" \
  --local-dir ./models
```

The Q4_K_M file for 70B is approximately 40GB.

---

## Step 5: Set the Critical Environment Variable

This is the single most impactful setting for prompt processing speed. The default rocBLAS kernels for gfx1150 are poorly optimized — this forces hipBLASLt instead, giving a 2-3x speedup on prompt processing:

```bash
export ROCBLAS_USE_HIPBLASLT=1
```

**Real benchmark impact** (7B Q4_0 as reference):
- Without: pp512 = ~348 tokens/s
- With hipBLASLt: pp512 = ~882 tokens/s (~2.5x improvement)
- Token generation speed is not affected by this setting

Add it to your `.bashrc` so it's always set:

```bash
echo 'export ROCBLAS_USE_HIPBLASLT=1' >> ~/.bashrc
```

---

## Step 6: Run Llama 3.1 70B

Here is the recommended command for maximum token generation speed:

```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-cli \
  -m ./models/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  --no-mmap \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 512 \
  -t 4 \
  -p "Your prompt here"
```

**Settings explained:**

| Setting | Value | Reason |
|---------|-------|--------|
| `-ngl 99` | All layers on GPU | Full GPU offload is optimal. Unified memory means zero PCIe transfer penalty for partial offloading, but full GPU offload still wins for throughput |
| `--no-mmap` | Disabled | **Critical for 70B.** HIP's `hipMemcpy()` must lock/unlock mmap'd pages, causing severe slowdown for models larger than ~64GB. This flag loads the model directly into RAM instead |
| `-fa on` | Flash attention | WMMA-accelerated on RDNA 3.5, saves memory and speeds up attention |
| `-ctk q8_0 -ctv q8_0` | Q8 KV cache | Reduces KV cache memory footprint with negligible quality impact, freeing bandwidth and memory for the model weights |
| `-b 2048 -ub 512` | Batch sizes | Good default for Strix Halo with ample unified memory |
| `-t 4` | 4 CPU threads | Keep this low. CPU and GPU share the ~256 GB/s memory bus — too many CPU threads compete with the GPU for bandwidth and slow token generation |

---

## What Speed to Expect

Token generation for Llama 3.1 70B Q4_K_M on Strix Halo typically lands in the range of **6-12 tokens/s** depending on context length and system load. This is memory-bandwidth bound: the iGPU must stream ~40GB of model weights through ~256 GB/s of shared bandwidth.

For reference, smaller models are faster:
- 7B Q4_0: ~45-50 tokens/s token generation
- 13B Q4_K_M: ~25-30 tokens/s
- 70B Q4_K_M: ~6-12 tokens/s

Prompt processing (ingesting your prompt) will be significantly faster thanks to the `ROCBLAS_USE_HIPBLASLT=1` setting.

---

## Optional: Use llama-server Instead

If you want an OpenAI-compatible API server (useful for connecting frontends like Open WebUI):

```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-server \
  -m ./models/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  --no-mmap \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -c 8192 \
  -t 4 \
  --host 0.0.0.0 \
  --port 8080
```

Then access it at `http://localhost:8080`.

---

## Benchmark to Verify

After getting things running, benchmark to confirm you're getting expected performance:

```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-bench \
  -m ./models/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  --no-mmap \
  -ctk q8_0 -ctv q8_0 \
  -p 512 -n 128 \
  -t 4 \
  -r 3
```

Look at the `tg` (token generation) column — that's your sustained generation speed.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM error loading 70B model | TTM limits not set — follow Step 1, reboot required |
| Very slow prompt processing | Make sure `ROCBLAS_USE_HIPBLASLT=1` is exported |
| Model loading takes many minutes or hangs | Add `--no-mmap` — essential for models >64GB |
| GPU not detected | Check `rocminfo`, verify user is in `video` and `render` groups |
| `rocm-smi` shows only 512MB or 4GB VRAM | Normal for APUs — Strix Halo uses unified memory. Reported VRAM is just the BIOS-reserved portion. The full 128GB is accessible after TTM configuration |
| Slow compilation (30+ minutes) | Ensure `GPU_TARGETS="gfx1150"` is set, avoiding multi-arch build |
| Performance suddenly drops | Try toggling `GGML_CUDA_FORCE_MMQ` in your build, or check for known regressions in recent commits |

---

## Summary Checklist

- [ ] Set TTM memory limits in `/etc/modprobe.d/increase_amd_memory.conf` and rebooted
- [ ] Verified GPU shows as `gfx1150` or `gfx1151` via `rocminfo`
- [ ] Built with `DGGML_HIP=ON`, `GPU_TARGETS="gfx1150"`, `DGGML_CUDA_FORCE_MMQ=ON`, `DGGML_NATIVE=ON`
- [ ] Set `export ROCBLAS_USE_HIPBLASLT=1` (added to `.bashrc`)
- [ ] Running inference with `-ngl 99 --no-mmap -fa on -ctk q8_0 -ctv q8_0 -t 4`
- [ ] Did NOT set `HSA_OVERRIDE_GFX_VERSION`
