# Using Vulkan Instead of ROCm on Strix Halo

Yes, you can absolutely use Vulkan for llama.cpp on your Strix Halo system. Since you already have Mesa drivers installed, you likely have everything you need -- no additional software installation required. This is one of Vulkan's main advantages over the ROCm/HIP stack.

## Building with Vulkan

The build is straightforward:

```bash
cmake -S . -B build \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j $(nproc)
```

That's it. No `GPU_TARGETS`, no ROCm SDK, no multi-gigabyte dependency chain. Vulkan uses runtime SPIR-V shaders rather than precompiled ISA, so it automatically works with your RDNA 3.5 hardware without any architecture-specific flags.

## Running Inference

```bash
./build/bin/llama-cli -m model.gguf \
  -ngl 99 \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -t 4
```

The Vulkan backend will auto-detect your Radeon 890M via the AMD vendor ID (0x1002). It queries `VkPhysicalDeviceSubgroupProperties` at runtime and adapts shader dispatch for RDNA 3.5's wave32/wave64 subgroup sizes. Matmul shaders use subgroup operations (`subgroupAdd`, `subgroupShuffle`) for efficient reductions.

For memory, Vulkan sees both device-local memory (the BIOS-allocated VRAM portion) and host-visible memory. On a unified memory APU like Strix Halo, both map to the same physical LPDDR5X, but device-local allocations may get priority bandwidth. The backend prefers device-local for model weights.

## Vulkan vs HIP: Honest Comparison

Here is where you should set expectations. HIP/ROCm does have real performance advantages on Strix Halo:

| Factor | HIP/ROCm | Vulkan |
|--------|----------|--------|
| **Prompt processing** | Excellent -- ~880 t/s on 7B Q4 with hipBLASLt | Good, but no hipBLASLt equivalent (expect closer to ~350 t/s range) |
| **Token generation** | ~48 t/s (7B Q4) | Comparable to HIP |
| **Graph capture** | HIP graphs reduce kernel dispatch overhead | Not available |
| **Flash attention** | WMMA-accelerated via rocWMMA | Shader-based (functional but not hardware-accelerated the same way) |
| **Setup complexity** | Requires ROCm 6.1+ (multi-GB install) | Just needs Vulkan drivers (Mesa, already installed) |
| **Build time** | Can be 10-30 minutes depending on GPU targets | Fast -- standard C++ compilation |
| **Stability** | Mature on Strix Halo (with TTM + hipBLASLt fixes) | Known model loading failures on some configurations ([#18741](https://github.com/ggml-org/llama.cpp/issues/18741)) |

### Where HIP Wins

The biggest gap is **prompt processing**. The hipBLASLt kernel library (`ROCBLAS_USE_HIPBLASLT=1`) provides a roughly 2.5x speedup for prompt processing on gfx1150/gfx1151. Vulkan has no equivalent optimization -- it relies on its own SPIR-V matmul shaders in `ggml/src/ggml-vulkan/vulkan-shaders/`. This means if your workload involves processing long prompts (RAG, long documents, large system prompts), you will notice a significant difference.

HIP graphs also reduce kernel dispatch overhead for token generation, which Vulkan cannot replicate.

### Where Vulkan Wins

- **Zero setup friction**: Mesa Vulkan drivers are already on your system. No ROCm packages to install, manage, or update.
- **Build simplicity**: A Vulkan build compiles in minutes. HIP builds targeting gfx1150 take around 10 minutes; multi-arch builds can exceed 30 minutes.
- **Portability**: The same Vulkan build works across AMD, NVIDIA, and Intel GPUs.
- **Lighter footprint**: ROCm is a large software stack. If you are on a laptop and care about disk space or system cleanliness, Vulkan is significantly lighter.

### Where They Are Similar

**Token generation speed is comparable.** Token generation is memory-bandwidth bound (reading model weights from memory), and both backends are limited by the same ~256 GB/s LPDDR5X bus. The compute kernels matter less here -- what matters is how fast you can stream weights through the GPU. Both HIP and Vulkan achieve similar throughput for this workload.

## Known Vulkan Issue to Watch For

There is a known issue ([#18741](https://github.com/ggml-org/llama.cpp/issues/18741)) where the Vulkan backend can fail to load models on some Strix Halo configurations. If you hit "failed to load model" errors, this is a known problem. There is no universal workaround within Vulkan itself -- this is one of the reasons HIP is considered the more reliable backend on this hardware.

## Important: TTM Limits Still Apply

Even with Vulkan, you should configure the kernel TTM limits if you plan to run models larger than ~8GB. Without this, the GPU can only access a fraction of your system RAM regardless of which backend you use.

Create `/etc/modprobe.d/increase_amd_memory.conf`:
```
options ttm pages_limit=25600000
options ttm page_pool_size=25600000
```

Then:
```bash
sudo update-initramfs -u -k all
sudo reboot
```

## Bottom Line

**Vulkan is a perfectly viable choice** for your use case, especially if you want to avoid the weight of the ROCm stack. For interactive chat and moderate-length prompts, you will get good token generation performance that is comparable to HIP. The main trade-off is prompt processing speed -- if you regularly feed in long contexts (thousands of tokens), you will notice HIP's hipBLASLt advantage.

If your typical workflow is conversational (short prompts, lots of token generation), Vulkan will serve you well. If you later find you need faster prompt processing and decide to install ROCm, you can always switch -- the two backends are not mutually exclusive, and you can even build llama.cpp with both enabled.
