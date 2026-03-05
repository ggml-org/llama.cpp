# Using Vulkan vs HIP/ROCm for llama.cpp on AMD Strix Halo

## Short Answer

Yes, you can absolutely use Vulkan with llama.cpp on Strix Halo, and it works with mesa drivers out of the box. However, HIP/ROCm will deliver better performance on this hardware if you are willing to install it. Below is a detailed comparison.

## Vulkan on Strix Halo: How to Build and Use

Building llama.cpp with Vulkan support is straightforward. You only need the Vulkan development libraries and a GLSL compiler:

```bash
# Install dependencies (Debian/Ubuntu)
sudo apt-get install libvulkan-dev glslc

# Verify your GPU is detected
vulkaninfo

# Build llama.cpp with Vulkan
cmake -S . -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

With mesa's RADV driver (which supports RDNA 3.5 / gfx1150), Vulkan should detect your Strix Halo iGPU automatically. The Vulkan backend recognizes the GPU as an integrated (UMA) device and adjusts memory allocation accordingly, preferring host-visible device-local memory.

## Vulkan vs HIP: Performance Comparison

### Where HIP/ROCm Wins

1. **Prompt processing (prefill) throughput**: HIP with hipBLASLt (`ROCBLAS_USE_HIPBLASLT=1`) provides significantly better matrix multiplication performance on gfx1150. The standard rocBLAS kernels are mediocre on Strix Halo, but the hipBLASLt path can roughly double prompt processing speed (e.g., from ~350 t/s to ~880 t/s on a 7B model).

2. **Flash Attention**: The HIP backend has mature flash attention support with rocWMMA on RDNA3+ architectures, offering better memory efficiency for long-context inference.

3. **Quantized matrix multiplication (MMQ)**: HIP can use MFMA (Matrix Fused Multiply-Add) instructions on CDNA and specialized paths for RDNA, giving it an edge for quantized model inference.

4. **Overall maturity on AMD**: The HIP/CUDA backend in llama.cpp has been more heavily optimized for AMD GPUs, with specific tuning for different AMD architectures.

### Where Vulkan is Competitive

1. **Token generation speed**: For autoregressive decoding (generating one token at a time), the performance gap narrows considerably since this is more memory-bandwidth-bound than compute-bound. On a UMA architecture like Strix Halo, both backends are accessing the same physical memory.

2. **Cooperative matrix support**: The Vulkan backend does support `VK_KHR_cooperative_matrix` for accelerated matrix operations on supported hardware. Whether RADV exposes this extension for gfx1150 depends on your mesa version.

3. **Small model inference**: For smaller models (7B and under) where you are not compute-bound, the difference between Vulkan and HIP is less dramatic.

### Estimated Performance Gap

As a rough guide, expect Vulkan to deliver approximately 60-80% of HIP/ROCm performance for prompt processing on Strix Halo. Token generation performance may be closer to 85-95% of HIP. These numbers depend heavily on model size, quantization, and context length.

## Vulkan: Known Limitations on Strix Halo

- **Architecture detection**: The Vulkan backend classifies RDNA 3.5 (Strix Halo) as `AMD_RDNA3` since it checks for integer dot product support to distinguish RDNA3+ from RDNA2. This means it gets RDNA3-class optimizations, which is generally appropriate.

- **UMA memory handling**: The backend detects Strix Halo as an integrated GPU (`uma = true`) and adjusts buffer allocation. However, unlike the HIP path where you can tune the TTM `pages_limit` to unlock more system memory for the GPU, Vulkan memory management is handled by the driver (RADV) and you have less control.

- **No equivalent to hipBLASLt**: There is no Vulkan equivalent to the `ROCBLAS_USE_HIPBLASLT=1` optimization that dramatically improves prompt processing on gfx1150. This is one of the biggest performance differentiators.

## Recommendation

**If you want simplicity and zero ROCm overhead**, Vulkan is a perfectly viable choice. It will give you good performance for interactive use (chatting with models), especially for token generation. Build with `-DGGML_VULKAN=ON` and you are ready to go.

**If you want maximum performance**, particularly for prompt processing, batch inference, or large context workloads, HIP/ROCm is worth the installation overhead. The `ROCBLAS_USE_HIPBLASLT=1` environment variable alone can provide a significant speedup that Vulkan cannot match on this hardware.

**A middle-ground option**: You can build llama.cpp with both backends enabled (`-DGGML_VULKAN=ON -DGGML_HIP=ON`) and switch between them at runtime using the `--device` flag. This lets you start with Vulkan immediately and add HIP later if you decide you need the extra performance.

## Quick Start with Vulkan

```bash
# Build
cmake -S . -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run (offload all layers to GPU)
./build/bin/llama-cli -m model.gguf -ngl 99 -p "Hello, world"

# Check detected device
./build/bin/llama-cli --list-devices
```

For Strix Halo with its unified memory architecture, use `-ngl 99` to offload all layers to the GPU regardless of backend choice.
