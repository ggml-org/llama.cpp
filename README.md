# llama.cpp-gfx906: AMD MI50/MI60/Vega7 fork

This fork is specifically optimized for AMD GFX906 architecture (MI50, MI60, Vega VII) . The aim of this fork is to maximize prompt-processing and inference on a single card. Compatability is now tested on Qwen3 30B-A3B Thinking 2507 (Q4_0) and Qwen3 4B Instruct 2507 (Q4_0).

---

## Acknowledgments
**Special thanks to [skyne98](https://github.com/skyne98/ggml-gfx906)** for the foundational work, of course to the whole **[ggml-org](https://github.com/ggml-org/llama.cpp)** open source community, and to all the https://discord.gg/sgjdAU9eRC people for the efforts on gfx906 optimization.

**The fork is now based on llama.cpp build 051b3382 **

---

## Key Features of fattn-vec-f16.cu - forked

- **Replaced bpermute instructions with swizzle** (AMD native warp reductions, main contribution)
- **V vectors caching** (another nice bump in speed)  

---

## Target Hardware & Models

### Supported GPUs
- **AMD MI50** (Vega 20) (only one actually tested)
- **AMD MI60** (Vega 20) 
- **AMD Vega VII** (Vega 20)

### Supported Models
- **All the llamacpp supported models**
- Tested extensively with **Qwen3-30B-A3B** (Q4_0, Q4_1) and **Qwen3-4B** (Q4_0, Q4_1)

---

## Performance Improvements wrt vanilla llamacpp:

### Qwen3 4B Q4_0 Model
| Metric | Standard Depth | Improvement | Depth 1024 | Improvement |
|--------|---------------|-------------|------------|-------------|
| **Token Gen (tg128)** | 84.86 → 104.15 t/s | **+22.7%** | 72.70 → 84.57 t/s | **+16.3%** |
| **Token Gen (tg256)** | 82.51 → 102.79 t/s | **+24.6%** | 68.87 → 81.21 t/s | **+17.9%** |

### Qwen3moe 30B.A3B Q4_0 Model  
| Metric | Standard Depth | Improvement | Depth 1024 | Improvement |
|--------|---------------|-------------|------------|-------------|
| **Token Gen (tg128)** | 66.54 → 76.01 t/s | **+14.2%** | 55.94 → 66.25 t/s | **+18.4%** |
| **Token Gen (tg256)** | 66.47 → 75.91 t/s | **+14.2%** | 54.50 → 66.11 t/s | **+21.3%** |

---

### Qwen3 4B Q4_0

### Standard Depth Configuration

| Test | Vanilla llamacpp | gfx906-v2 Fork | Δ Change | % Improvement |
|------|-----------------|----------------|----------|---------------|
| **pp512** | 1793.78 t/s | 1797.05 t/s | +3.27 | +0.2% |
| **pp1024** | 1750.35 t/s | 1755.67 t/s | +5.32 | +0.3% |
| **pp2048** | 1653.97 t/s | 1661.81 t/s | +7.84 | +0.5% |
| **pp4096** | 1424.06 t/s | 1496.66 t/s | +72.60 | +5.1% |
| **tg128** | 84.86 t/s | **104.15 t/s** | +19.29 | **+22.7%** |
| **tg256** | 82.51 t/s | **102.79 t/s** | +20.28 | **+24.6%** |

### Depth 1024 Configuration

| Test | Vanilla llamacpp | gfx906-v2 Fork | Δ Change | % Improvement |
|------|-----------------|----------------|----------|---------------|
| **pp512** | 1590.85 t/s | 1574.58 t/s | -16.27 | -1.0% |
| **pp1024** | 1559.56 t/s | 1556.18 t/s | -3.38 | -0.2% |
| **pp2048** | 1482.26 t/s | 1488.85 t/s | +6.59 | +0.4% |
| **pp4096** | 1261.73 t/s | 1318.91 t/s | +57.18 | +4.5% |
| **tg128** | 72.70 t/s | **84.57 t/s** | +11.87 | **+16.3%** |
| **tg256** | 68.87 t/s | **81.21 t/s** | +12.34 | **+17.9%** |

---

### Qwen3moe 30B.A3B Q4_0

### Standard Depth Configuration

| Test | Vanilla llamacpp | gfx906-v2 Fork | Δ Change | % Improvement |
|------|-----------------|----------------|----------|---------------|
| **pp512** | 1286.98 t/s | 1284.54 t/s | -2.44 | -0.2% |
| **pp1024** | 1268.94 t/s | 1265.53 t/s | -3.41 | -0.3% |
| **pp2048** | 1207.67 t/s | 1207.41 t/s | -0.26 | 0.0% |
| **pp4096** | 1091.78 t/s | 1095.08 t/s | +3.30 | +0.3% |
| **tg128** | 66.54 t/s | **76.01 t/s** | +9.47 | **+14.2%** |
| **tg256** | 66.47 t/s | **75.91 t/s** | +9.44 | **+14.2%** |

### Depth 1024 Configuration

| Test | Vanilla llamacpp | gfx906-v2 Fork | Δ Change | % Improvement |
|------|-----------------|----------------|----------|---------------|
| **pp512** | 1173.63 t/s | 1174.89 t/s | +1.26 | +0.1% |
| **pp1024** | 1145.44 t/s | 1146.86 t/s | +1.42 | +0.1% |
| **pp2048** | 1053.87 t/s | 1096.24 t/s | +42.37 | +4.0% |
| **pp4096** | 864.03 t/s | 983.24 t/s | +119.21 | +13.8% |
| **tg128** | 55.94 t/s | **66.25 t/s** | +10.31 | **+18.4%** |
| **tg256** | 54.50 t/s | **66.11 t/s** | +11.61 | **+21.3%** |

---

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
