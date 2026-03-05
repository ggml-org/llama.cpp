# Using Vulkan Instead of ROCm/HIP on Strix Halo for llama.cpp

Yes, you can absolutely use Vulkan with llama.cpp on your Strix Halo system, and with Mesa (RADV) drivers already installed, it is the lowest-friction path to GPU acceleration. Below is a detailed comparison and practical guidance.

---

## Building with Vulkan

Since you already have Mesa drivers, you just need the Vulkan development headers and the `glslc` shader compiler. On Debian/Ubuntu:

```bash
sudo apt-get install libvulkan-dev glslc
```

Then build llama.cpp:

```bash
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)
```

Verify your GPU is detected:

```bash
vulkaninfo | grep -i "deviceName"
# Should show something like: AMD Radeon 890M (RADV GFX1150)

./build/bin/llama-cli -m YOUR_MODEL.gguf -p "test" -ngl 99
# Output should include: ggml_vulkan: Using AMD Radeon ... | uma: 1
```

The `uma: 1` in the output confirms the Vulkan backend recognizes your iGPU as a unified memory architecture device, which is important for efficient operation.

---

## How Vulkan Compares to HIP on Strix Halo

### Architecture Detection

The Vulkan backend classifies Strix Halo's Radeon 890M (RDNA 3.5, gfx1150/gfx1151) as `AMD_RDNA3` in its architecture detection logic. The classification is based on the GPU reporting support for `integerDotProduct4x8BitPackedMixedSignednessAccelerated`, which RDNA 3 and later all support. This means Strix Halo gets the same tuned shader paths as discrete RDNA 3 cards (RX 7000 series).

The HIP backend, by contrast, has a distinct `GGML_CUDA_CC_RDNA3_5` compute capability class specifically for gfx1150/gfx1151, enabling more fine-grained optimizations for this exact silicon.

### Feature Support Comparison

| Feature | Vulkan (RADV/Mesa) | HIP (ROCm) |
|---|---|---|
| **Install footprint** | ~50-100 MB (Mesa + Vulkan headers) | ~5-15 GB (full ROCm stack) |
| **Driver requirement** | Mesa RADV (already installed) | ROCm 6.1+ with hipBLAS, rocBLAS |
| **Unified memory (UMA)** | Auto-detected for iGPUs; avoids unnecessary copies | Requires `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` env var |
| **Cooperative matrix (tensor ops)** | KHR_cooperative_matrix via RADV -- enabled for all AMD GPUs with open-source drivers | WMMA intrinsics, specifically tuned for RDNA 3/3.5 |
| **Flash attention** | Vulkan flash attention with coopmat1 path on RDNA3-class hardware | WMMA-accelerated flash attention (`-fa on`), with optional rocWMMA for enhanced performance |
| **Integer dot product** | Supported (accelerated 4x8-bit packed dot products) | Supported via native ISA |
| **Quantized matmul (MMQ)** | Vulkan compute shaders with warptile tuning for AMD + coopmat | HIP MMQ kernels with RDNA 3.5-specific tuning (e.g., extended batch size thresholds for IQ2 quants) |
| **hipBLAS/rocBLAS** | N/A | Available but default rocBLAS kernels for gfx1150 are poorly optimized; requires `ROCBLAS_USE_HIPBLASLT=1` workaround |
| **Subgroup (wave) size** | Wave32/Wave64 (RADV supports both; backend uses subgroup size control) | Wave64 default via HIP compiler |
| **Build time** | Fast (~2-5 minutes, shader compilation at build time) | Slow (~10-30+ minutes depending on GPU_TARGETS) |
| **BFloat16** | Supported if VK_KHR_shader_bfloat16 is available in driver | Supported natively |

### Performance Expectations

**Where HIP has an advantage:**

- **Prompt processing (batch workloads):** HIP with properly configured rocBLAS (`ROCBLAS_USE_HIPBLASLT=1`) can achieve higher throughput on large matrix multiplications during prompt processing. The hipBLASLt library has kernels tuned for gfx1150.
- **Flash attention:** The HIP flash attention path uses WMMA (Wave Matrix Multiply-Accumulate) intrinsics that are well-optimized for RDNA 3.5. The rocWMMA-enhanced path (`-DGGML_HIP_ROCWMMA_FATTN=ON`) can provide further gains.
- **Fine-grained arch tuning:** HIP distinguishes RDNA 3.0 from RDNA 3.5 in kernel selection (e.g., different MMQ batch thresholds), while Vulkan groups them together as RDNA3.

**Where Vulkan holds its own or wins:**

- **Token generation (single-token inference):** For autoregressive generation (the common interactive use case), the performance gap is smaller. Matrix-vector operations are more memory-bandwidth-bound, and both backends can saturate the LPDDR5X bandwidth similarly.
- **UMA handling:** Vulkan's UMA support is more transparent. It auto-detects the iGPU and avoids unnecessary buffer copies. The HIP path requires manually setting environment variables.
- **Ease of use and maintenance:** No ROCm version compatibility issues, no multi-GB SDK to manage, no `GPU_TARGETS` to configure correctly.
- **Cooperative matrix support:** With RADV, `VK_KHR_cooperative_matrix` is enabled for all AMD GPUs (including RDNA 3.5). The Vulkan backend has AMD-specific coopmat warptile tuning that benefits matrix multiplication workloads.

**Rough performance ballpark (based on community reports and architecture analysis):**

- Token generation: Vulkan typically achieves 80-95% of HIP performance on iGPUs
- Prompt processing: Vulkan may be 60-80% of well-tuned HIP (with `ROCBLAS_USE_HIPBLASLT=1`)
- For interactive chat use cases where token generation is the bottleneck, this difference is often negligible

### Known Caveats with Vulkan on Strix Halo

1. **Shader compilation on first run:** The first launch may be slow as Vulkan shaders are compiled and cached. Subsequent runs use the cached shaders.
2. **RADV driver version matters:** Newer Mesa versions have better RDNA 3.5 support. Use Mesa 24.1+ if possible for full cooperative matrix and compute shader support.
3. **Memory limits:** While UMA is auto-detected, the Vulkan backend may report a smaller VRAM budget than the total system RAM. This is a driver-reported limit, not a hard physical limit. You can use `GGML_VK_FORCE_MAX_BUFFER_SIZE` to override if needed.

---

## Recommended Vulkan Runtime Flags

```bash
./build/bin/llama-cli \
    -m model.gguf \
    -ngl 99 \
    -fa \
    -ctk q8_0 -ctv q8_0 \
    -c 8192 \
    --no-mmap
```

- **`-ngl 99`**: Offload all layers to the iGPU. With unified memory, there is no PCIe transfer penalty.
- **`-fa`**: Enable flash attention, which uses the coopmat1 path on RDNA3-class hardware.
- **`-ctk q8_0 -ctv q8_0`**: Quantize the KV cache to save memory and improve bandwidth utilization.
- **`--no-mmap`**: Avoids memory-mapped I/O; allocates buffers directly. Can improve performance on UMA systems.

---

## Summary

Vulkan with Mesa/RADV is a perfectly viable and practical choice for Strix Halo. You trade some peak prompt processing performance for dramatically simpler setup, smaller disk footprint, and zero ROCm dependency management. For interactive use cases (chatbots, code completion) where token generation speed matters most, the performance difference is modest. If you later find you need every last token/second for batch processing workloads, you can always add ROCm -- llama.cpp even supports building both backends simultaneously (`-DGGML_VULKAN=ON -DGGML_HIP=ON`) and selecting at runtime with `--device`.
