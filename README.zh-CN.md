<div align="center">

# Summer.cpp

### 在 NVIDIA GPU、DRAM 与 SSD 之间运行超出显存容量的 GGUF 模型

**分层内存执行 · 分片 GGUF 支持 · CUDA 本地推理**

[日本語](README.md) · [English](README.en.md) · **简体中文** · [繁體中文](README.zh-TW.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md)

[![Platform](https://img.shields.io/badge/platform-Linux-111827?logo=linux&logoColor=white)](#环境要求)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?logo=nvidia&logoColor=white)](#环境要求)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-00599C?logo=cplusplus&logoColor=white)](#手动构建)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#许可证)

</div>

> [!IMPORTANT]
> 当前最稳定的组合是 **VRAM + DRAM**。在 GTX 1660 SUPER 上，已验证 `--vram-mib 3800 --dram-mib 6500` 且 SSD 分配为 0 MiB 的配置。Turing GPU 上的选择性 SSD 流式加载仍属实验功能。

## 概述

Summer.cpp 是 llama.cpp 的一个 fork，加入了分层内存后端和专用可执行文件 `llama-tiered`。大型 GGUF 张量可根据用途与预算放置到不同内存层级。

| 层级 | 存储位置 | 用途 |
|---|---|---|
| VRAM | CUDA 设备内存 | 高频使用的 dense weight、embedding 与热点张量 |
| DRAM | CUDA 映射的主机内存 | 通过 zero-copy 或 mapped pinned copy 访问无法放入显存的权重 |
| SSD | 基于文件的 GGUF 映射 | 按需暂存路由器选中的 MoE expert，并将热点 expert 保留在自适应缓存中 |

```text
GGUF file
   │ mmap
   ▼
Placement planner
   ├── VRAM: resident CUDA allocations
   ├── DRAM: mapped host memory / pinned copy fallback
   └── SSD : selected MoE slabs -> adaptive VRAM cache -> reusable scratch
```

## 已验证配置

| 项目 | 数值 |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 SUPER |
| Compute capability | 7.5 |
| 模型 | Qwen3.6-35B-A3B-UD-IQ1_M.gguf |
| GGUF 大小 | 约 9.35 GiB |
| VRAM 预算 | 3800 MiB |
| DRAM 预算 | 6500 MiB |
| SSD 放置 | 0 MiB |
| 加载时间 | 约 5.65 秒 |
| Prompt 处理 | 约 31.7 tokens/s |
| Token 生成 | 约 27.7 tokens/s |

这些数据仅来自一次短 prompt 测试。模型、上下文、采样器、CPU、PCIe、驱动和后台负载都会影响结果。

## 环境要求

- Linux，推荐 Ubuntu 22.04 或 24.04
- NVIDIA GPU 与驱动
- CUDA Toolkit 和 `nvcc`
- CMake 与支持 C++17 的编译器
- Python 3.10 或更高版本
- GGUF 模型
- 足够的 SSD 空间
- 使用 DRAM 层时需要足够的系统内存

对于 GTX 1660 SUPER 和约 9.35 GiB 的模型，建议系统内存至少为 16 GiB。DRAM fallback 可能会在文件 mmap 之外额外分配 mapped pinned copy。

```bash
nvidia-smi
nvcc --version
cmake --version
python3 --version
```

## 快速安装

### 1. 安装依赖

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  python3
```

如果尚未安装 CUDA Toolkit，请选择与当前 NVIDIA 驱动和 GPU 兼容的版本。

### 2. 克隆仓库

```bash
git clone https://github.com/vnlpscale/Summer.cpp.git "$HOME/Summer.cpp"
cd "$HOME/Summer.cpp"
```

### 3. 构建并安装

```bash
bash scripts/install-summer.sh
```

安装脚本会应用 Turing/GTX 16 的 DRAM fallback 补丁，以 Release 模式构建 CUDA 版 `llama-tiered`，安装到 `~/.local/bin`，删除旧的 SummerCLI 命令，并创建 `~/models`。

GTX 1660 SUPER 可显式指定：

```bash
CUDA_ARCH=75 FORCE_MMQ=ON bash scripts/install-summer.sh
```

对于无需强制 MMQ 的 Tensor Core GPU：

```bash
FORCE_MMQ=OFF bash scripts/install-summer.sh
```

### 4. 配置 PATH

```bash
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
source "$HOME/.bashrc"
hash -r
command -v llama-tiered
```

### 5. 放置模型

```bash
mkdir -p "$HOME/models"
cp /path/to/model.gguf "$HOME/models/"
```

使用 split GGUF 时，请将所有分片放在同一目录。

### 6. 运行

```bash
llama-tiered \
  -m "$HOME/models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  -n 128 \
  "请介绍一下你自己。"
```

CLI 会将 llama.cpp 的块状标志和 Summer.cpp 标题输出到标准错误，因此标准输出中的生成文本仍可安全用于管道处理。

## 手动构建

```bash
cd "$HOME/Summer.cpp"
python3 scripts/apply-tiered-dram-pinned-fallback.py
python3 scripts/apply-tiered-dram-matmul-staging.py
python3 scripts/apply-tiered-no-prompt-echo.py
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_BACKEND_DL=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build build --target llama-tiered -j"$(nproc)"
install -Dm755 build/bin/llama-tiered "$HOME/.local/bin/llama-tiered"
```

## 自适应 Expert 缓存

当大型 MoE 模型的部分权重位于 SSD 时，可使用 `--cache-mib`。缓存容量包含在 `--vram-mib` 中，会自动从常驻权重预算扣除。

```bash
llama-tiered \
  -m "$HOME/models/Laguna-S-2.1-UD-IQ1_S.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  --cache-mib 1024 \
  -n 128 \
  "你好"
```

缓存会在 single-row decode 时根据路由历史选择热点 expert，并在 multi-row prompt batch 中自动绕过以避免污染。结束日志会显示命中率、H2D/D2D 流量、admission 与 eviction 数量。

## 内存预算调优

GTX 1660 SUPER 的建议起点：

```text
--vram-mib 3800 --dram-mib 6500
```

- 出现 GPU allocation error 时降低 `--vram-mib`。
- 张量被放到 SSD 时提高 `--dram-mib`。
- 系统内存不足时使用更小或量化程度更高的 GGUF。
- GPU 同时负责桌面显示时保留更多显存。

优先稳定性时，应使用启动日志中显示 `SSD 0.00 MiB` 的配置。

## 故障排查

- 缺少 `build/bin/llama-tiered`：重新运行 `bash scripts/install-summer.sh`。
- Turing/GTX 16 上出现 `invalid argument`：应用 DRAM pinned fallback 后重新构建。
- `tensor_state layout did not match expected source`：恢复 `ggml/src/ggml-cuda/tiered.cu` 后重新运行安装脚本。
- `operation not supported`：删除 `build` 并使用最新补丁重建。
- CUDA 非法内存访问：更新并重建，使用短 prompt 配合 `compute-sanitizer --tool memcheck`。
- `summer: command not found`：当前命令为 `llama-tiered`，并确认 `~/.local/bin` 位于 `PATH`。

## SSD 流式加载状态

SSD 层将 stacked MoE expert tensor 保留在 GGUF mmap 中，并在 `MUL_MAT_ID` 时仅把被选中的 expert slab 传入可复用的 VRAM scratch。当前限制包括：源页面可能常驻 page-locked RAM、scratch 按最大 stacked expert tensor 分配、传输与计算尚未重叠、缓存仅支持 single-row decode、共享同一模型的多个 context 会串行执行 graph，以及每种 GPU 架构都需要实机验证。

GTX 1660 SUPER 与 Laguna-S-2.1 IQ1_S 已测试 1、16、128 个生成 token，并通过 `compute-sanitizer` 验证。其他 GPU 或模型请从短生成开始测试。

## Library API

```cpp
#include "llama-tiered.h"

llama_model_params model_params = llama_model_default_params();
llama_tiered_memory_params memory = llama_tiered_memory_default_params();
memory.vram_budget_bytes = 3800ull * 1024 * 1024;
memory.dram_budget_bytes = 6500ull * 1024 * 1024;

llama_tiered_model * owner = llama_tiered_model_load_from_file(
        "model.gguf", model_params, memory);
if (!owner) {
    fprintf(stderr, "tiered load failed: %s\n", llama_tiered_last_error());
    return 1;
}
llama_model * model = llama_tiered_model_get_model(owner);
// 在 owner 存活期间创建并使用 llama_context。
llama_tiered_model_free(owner);
```

`llama_tiered_model_get_model()` 返回的是借用指针，请勿直接传给 `llama_model_free()`。

## 上游与许可证

Summer.cpp 基于 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)，采用与上游相同的 MIT License。详见 [LICENSE](LICENSE)。
