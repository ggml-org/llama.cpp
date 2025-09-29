# llama.cpp-gfx906: AMD MI50/MI60/Vega7 fork

This fork is specifically optimized for AMD GFX906 architecture (MI50, MI60, Vega VII) . The aim of this fork is to maximize prompt-processing and inference on a single card. Compatability is now tested on Qwen3 30B-A3B Thinking 2507 (Q4_0) and Qwen3 4B Instruct 2507 (Q4_0).

---

## Key Features of b6615 - forked

- **Replaced bpermute instructions with swizzle** (AMD native warp reductions, main contribution)

---

## Target Hardware & Models

### Supported GPUs
- **AMD MI50** (Vega 20) (only one actually tested)
- **AMD MI60** (Vega 20) 
- **AMD Vega VII** (Vega 20)

### Supported Models
- **All the llamacpp supported models**
- Tested extensively with **Qwen3-30B-A3B** (Q4_0, Q4_1)

### Performance comparison -- lama bench 
- did not use the -d because long prompt processing make gpu to reach 80C and throttle, making the comparison difficult
- all models tested with:
| ---------- | --- | ------- | ------- | ------ | ------ | -- |
| backend    | ngl | threads | n_batch | type_k | type_v | fa |
| ROCm       |  99 |      12 |    1024 |   q8_0 |   q8_0 |  1 |
| ---------- | --- | ------- | ------- | ------ | ------ | -- |

| **normal:**                    |       size |     params |		test |  		t/s |
| ------------------------------ | ---------: | ---------: | --------------: | -------------------: |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B |	       pp512 |       1768.68 ± 0.86 |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B |	      pp1024 |       1728.56 ± 0.33 |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B |	      pp2048 |       1636.15 ± 0.57 |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B |	      pp4096 |       1469.47 ± 1.09 |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B |	       tg128 |        116.76 ± 0.02 |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B |	       tg256 |        115.45 ± 1.11 |


| **swizzle:**                   |       size |     params |		test |  		t/s |
| ------------------------------ | ---------: | ---------: | --------------: | -------------------: |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B |	       pp512 |       1777.11 ± 0.65 |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B |	      pp1024 |       1734.32 ± 0.24 |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B |	      pp2048 |       1643.62 ± 0.25 |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B |	      pp4096 |       1479.31 ± 0.17 |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B |	       tg128 |        116.94 ± 0.04 |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B |	       tg256 |        116.66 ± 0.04 |


| **normal:**                    |       size |     params |		test |  		t/s |
| ------------------------------ | ---------: | ---------: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_0          |  16.18 GiB |    30.53 B |	       pp512 |       1269.93 ± 9.69 |
| qwen3moe 30B.A3B Q4_0          |  16.18 GiB |    30.53 B |	      pp1024 |       1255.27 ± 6.57 |
| qwen3moe 30B.A3B Q4_0          |  16.18 GiB |    30.53 B |	      pp2048 |       1196.97 ± 2.63 |
| qwen3moe 30B.A3B Q4_0          |  16.18 GiB |    30.53 B |	      pp4096 |       1081.50 ± 1.17 |
| qwen3moe 30B.A3B Q4_0          |  16.18 GiB |    30.53 B |	       tg128 |         92.84 ± 0.10 |
| qwen3moe 30B.A3B Q4_0          |  16.18 GiB |    30.53 B |	       tg256 |         92.69 ± 0.05 |


| **swizzle:**                   |       size |     params |		test |  		t/s |
| ------------------------------ | ---------: | ---------: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_0          |  16.18 GiB |    30.53 B |	       pp512 |       1272.79 ± 7.94 |
| qwen3moe 30B.A3B Q4_0          |  16.18 GiB |    30.53 B |	      pp1024 |       1257.33 ± 6.35 |
| qwen3moe 30B.A3B Q4_0          |  16.18 GiB |    30.53 B |	      pp2048 |       1200.32 ± 2.16 |
| qwen3moe 30B.A3B Q4_0          |  16.18 GiB |    30.53 B |	      pp4096 |       1087.70 ± 1.32 |
| qwen3moe 30B.A3B Q4_0          |  16.18 GiB |    30.53 B |	       tg128 |         93.41 ± 0.09 |
| qwen3moe 30B.A3B Q4_0          |  16.18 GiB |    30.53 B |	       tg256 |         93.27 ± 0.05 |

### Performance comparison -- prompt: write a 1000 words story
- tried some times to get same token count from both benches, however the slight speed increase is visible.

|**normal:**                                                                                         |
| ---------------------------------------------------------------------------------------------------|
|prompt eval time =      61.27 ms /    15 tokens (    4.08 ms per token,   244.83 tokens per second) |
|       eval time =   27459.54 ms /  2238 tokens (   12.27 ms per token,    81.50 tokens per second) |
|      total time =   27520.80 ms /  2253 tokens                                                     |

|**swizzle:**                                                                                        |
| ---------------------------------------------------------------------------------------------------|
|prompt eval time =      60.72 ms /    15 tokens (    4.05 ms per token,   247.03 tokens per second) |
|       eval time =   26540.24 ms /  2240 tokens (   11.85 ms per token,    84.40 tokens per second) |
|      total time =   26600.97 ms /  2255 tokens                                                     |



## Quick Start

### Prerequisites

- **ROCm 7.0.1** (tested version - other versions may work)
- **CMake 3.21+**
- **HIP compiler toolchain**
- **AMD GFX906 GPU** (MI50/MI60/Vega VII)
- **UBUNTU 24.04** (should work with other systems, not tested)

### System Dependencies

```bash
# Ubuntu
sudo apt update
sudo apt install cmake build-essential

# Install ROCm 7.0.1 following AMD's official guide
# Tensile library for gfx906 must be imported to use this ROCM version

# Verify ROCm installation
/opt/rocm/bin/rocm-smi
```

### Build Instructions

#### 1. Clone the repository

```bash
git clone https://github.com/iacopPBK/llama.cpp-gfx906.git
cd llama.cpp-gfx906
```

#### 2. Compile using the provided script

```bash
chmod +x SCRIPT_compile_MI50.sh
./SCRIPT_compile_MI50.sh
```

The compilation script automatically:
- Sets GFX906-specific compiler flags
- Enables HIP backend with GFX906 optimizations  
- Builds with flash attention support
- Links against ROCm libraries (rocBLAS, hipBLAS)

#### 3. Launch the server

```bash
# Edit SCRIPT_launch_server_MI50.sh to set your model path
vim SCRIPT_launch_server_MI50.sh

# Launch server with FA and KV quantizations
./SCRIPT_launch_server_MI50.sh
```

### Environment Variables

The optimized build sets these automatically:

```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0  
export ROCR_VISIBLE_DEVICES=0
export GGML_BACKEND_HIP=1
export HCC_AMDGPU_TARGET=gfx906
```

---

## Build Configuration

The build enables these optimizations:

- `GGML_HIP=ON` - Enable HIP backend
- `GGML_HIP_GFX906_OPTIMIZED=ON` - GFX906-specific optimizations
- `CMAKE_HIP_ARCHITECTURES=gfx906` - Target GFX906 architecture
- Flash attention with F16 precision (hardcoded)

---

*Built with care for the AMD GFX906 community ❤️‍🔥 x 1000*
