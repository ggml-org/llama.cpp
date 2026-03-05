# Vulkan vs HIP on AMD Strix Halo for llama.cpp

## Short Answer

Yes, you can use Vulkan with llama.cpp on Strix Halo, and mesa drivers are sufficient -- you do not need the Vulkan SDK or ROCm. However, there are known issues with the Vulkan backend on this hardware, and the ROCm/HIP backend is significantly better optimized for Strix Halo. If avoiding ROCm is your priority, Vulkan will work for many models, but expect lower performance and potential model loading failures.

## Building with Vulkan (No ROCm Required)

With mesa drivers already installed, building is straightforward:

```bash
# Ensure Vulkan development packages are installed
sudo apt-get install libvulkan-dev glslc

# Verify Vulkan detects your GPU
vulkaninfo | grep -i "deviceName"
# Should show: AMD Radeon Graphics (or similar for the 890M)

# Build llama.cpp with Vulkan
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
```

That is all you need. No ROCm installation, no SDK downloads, no kernel module configuration. The mesa RADV Vulkan driver supports RDNA 3.5 hardware natively.

## Running with Vulkan

```bash
./build/bin/llama-cli -m model.gguf \
  -ngl 99 \
  -fa on \
  -t 4 \
  -b 2048 -ub 512
```

Note: with Vulkan you do not need `--no-mmap` (the mmap slowdown is specific to HIP's `hipMemcpy()` page-locking behavior). You also do not need `ROCBLAS_USE_HIPBLASLT=1` since that is a rocBLAS-specific fix.

## Where Vulkan Falls Short on Strix Halo

### 1. Known Model Loading Failures

There is a known issue ([Issue #18741](https://github.com/ggml-org/llama.cpp/issues/18741)) where the Vulkan backend fails to load certain models on Strix Halo with a "failed to load model" error. This is a reliability problem that may or may not affect the specific models you want to run.

### 2. No RDNA 3.5-Specific Optimizations in Vulkan

The Vulkan backend in llama.cpp identifies AMD GPUs by architecture class (GCN, RDNA1, RDNA2, RDNA3) but does not have a separate RDNA 3.5 category. Your Strix Halo GPU (gfx1150) is detected as `AMD_RDNA3` in the Vulkan backend. This means it gets RDNA 3 code paths, which work but are not tuned for RDNA 3.5 specifics.

In contrast, the HIP backend has explicit `GGML_CUDA_CC_RDNA3_5` handling and gfx1150-targeted kernel compilation, meaning kernels are compiled specifically for your hardware.

### 3. No hipBLASLt Equivalent

One of the biggest HIP performance wins on Strix Halo is `ROCBLAS_USE_HIPBLASLT=1`, which improves prompt processing by approximately 2.5x (e.g., 348 t/s to 882 t/s for 7B Q4_0 on pp512). There is no equivalent optimization path in the Vulkan backend. Vulkan uses its own compute shader-based matrix multiplication, which is generic across vendors.

### 4. No HIP Graphs Equivalent

The HIP backend supports `GGML_HIP_GRAPHS=ON`, which captures kernel dispatch sequences into graphs for reduced dispatch overhead. Vulkan has no equivalent optimization in the current llama.cpp implementation.

### 5. No WMMA/rocWMMA Flash Attention

The HIP backend can use `GGML_HIP_ROCWMMA_FATTN=ON` for hardware-accelerated flash attention via WMMA instructions on RDNA 3.5. The Vulkan backend does have flash attention support (scalar and cooperative matrix paths), but it uses generic Vulkan compute shaders rather than architecture-specific WMMA instructions.

## Performance Comparison Summary

| Metric | HIP (ROCm) | Vulkan (mesa) |
|--------|-----------|---------------|
| **Prompt processing (7B Q4_0, pp512)** | ~880 t/s (with hipBLASLt) | Lower (no hipBLASLt equivalent) |
| **Token generation (7B Q4_0, tg128)** | ~48 t/s | Comparable (bandwidth-bound) |
| **Kernel optimization for gfx1150** | Native, architecture-specific | Generic RDNA3 code paths |
| **Flash attention** | WMMA-accelerated (rocWMMA) | Compute shader based |
| **Model loading reliability** | Stable | Known failures ([#18741](https://github.com/ggml-org/llama.cpp/issues/18741)) |
| **Large model loading (>64GB)** | Requires `--no-mmap` | No mmap issues |
| **Install complexity** | ROCm 6.1+ required (~5-10 GB) | Mesa drivers only |
| **TTM kernel config needed** | Yes (for models >8GB) | Yes (same GPU memory limits apply) |

Token generation speed is largely memory-bandwidth-bound (reading model weights from memory), so Vulkan and HIP should perform similarly for that metric since both are limited by the same ~256 GB/s LPDDR5X bus. The main HIP advantage is in prompt processing (compute-bound matrix multiplications) where hipBLASLt and architecture-specific kernels make a large difference.

## TTM Configuration Still Required for Vulkan

Even with Vulkan, you still need the TTM kernel parameter fix if you want to load models larger than a few GB onto the GPU. The TTM limit restricts how much system memory the GPU can access regardless of the backend.

Create `/etc/modprobe.d/increase_amd_memory.conf`:
```
options ttm pages_limit=25600000
options ttm page_pool_size=25600000
```

Then apply:
```bash
sudo update-initramfs -u -k all
sudo reboot
```

## Recommendations

**If you want to avoid ROCm entirely and are running smaller models (7B-13B):**
Vulkan with mesa is a reasonable choice. You get GPU acceleration with zero additional software installation. Token generation speed will be comparable to HIP. Prompt processing will be slower, but for interactive use with short prompts, this may be acceptable.

**If you are running larger models or need maximum throughput:**
ROCm/HIP is strongly recommended. The 2.5x prompt processing improvement from hipBLASLt alone justifies the installation overhead. The architecture-specific kernel compilation (`GPU_TARGETS=gfx1150`) and WMMA flash attention provide additional advantages that Vulkan cannot match on this hardware.

**If you hit the Vulkan model loading failure:**
This is a known issue with no current workaround other than switching to HIP. If you encounter it, you will need ROCm.

**A middle ground -- CPU-only with AVX-512:**
If you truly want zero GPU backend complexity, Strix Halo's Zen 5 cores with AVX-512 provide decent CPU-only inference. Build with:
```bash
cmake -B build -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)

./build/bin/llama-cli -m model.gguf -ngl 0 -t $(nproc)
```
This avoids all GPU driver considerations but will be slower than either Vulkan or HIP for token generation.
