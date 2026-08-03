<div align="center">

# Summer.cpp

### 在 NVIDIA GPU、DRAM 與 SSD 之間執行超出顯示記憶體容量的 GGUF 模型

**分層記憶體執行 · 分割 GGUF 支援 · CUDA 本機推論**

[日本語](README.md) · [English](README.en.md) · [简体中文](README.zh-CN.md) · **繁體中文** · [한국어](README.ko.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md)

[![Platform](https://img.shields.io/badge/platform-Linux-111827?logo=linux&logoColor=white)](#環境需求)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?logo=nvidia&logoColor=white)](#環境需求)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-00599C?logo=cplusplus&logoColor=white)](#手動建置)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#授權)

</div>

> [!IMPORTANT]
> 目前最穩定的組合是 **VRAM + DRAM**。在 GTX 1660 SUPER 上，已驗證 `--vram-mib 3800 --dram-mib 6500` 且 SSD 配置為 0 MiB 的設定。Turing GPU 上的選擇性 SSD 串流仍屬實驗功能。

## 概述

Summer.cpp 是 llama.cpp 的 fork，加入分層記憶體後端與專用執行檔 `llama-tiered`。大型 GGUF tensor 可依用途與記憶體預算配置到不同層級。

| 層級 | 位置 | 用途 |
|---|---|---|
| VRAM | CUDA device memory | 常用 dense weight、embedding 與 hot tensor |
| DRAM | CUDA mapped host memory | 以 zero-copy 或 mapped pinned copy 存取無法放入 VRAM 的權重 |
| SSD | file-backed GGUF mapping | 視需要載入路由器選中的 MoE expert，並在自適應 cache 中保留 hot expert |

```text
GGUF file
   │ mmap
   ▼
Placement planner
   ├── VRAM: resident CUDA allocations
   ├── DRAM: mapped host memory / pinned copy fallback
   └── SSD : selected MoE slabs -> adaptive VRAM cache -> reusable scratch
```

## 已驗證設定

| 項目 | 數值 |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 SUPER |
| Compute capability | 7.5 |
| 模型 | Qwen3.6-35B-A3B-UD-IQ1_M.gguf |
| GGUF 大小 | 約 9.35 GiB |
| VRAM 預算 | 3800 MiB |
| DRAM 預算 | 6500 MiB |
| SSD 配置 | 0 MiB |
| 載入時間 | 約 5.65 秒 |
| Prompt 處理 | 約 31.7 tokens/s |
| Token 生成 | 約 27.7 tokens/s |

以上數字是短 prompt 的單次測量。模型、context、sampler、CPU、PCIe、driver 與背景負載都會影響結果。

## 環境需求

- Linux，建議 Ubuntu 22.04 或 24.04
- NVIDIA GPU 與驅動程式
- CUDA Toolkit 與 `nvcc`
- CMake 與支援 C++17 的編譯器
- Python 3.10 以上
- GGUF 模型
- 足夠的 SSD 空間
- 使用 DRAM tier 時需要足夠的系統記憶體

GTX 1660 SUPER 搭配約 9.35 GiB 模型時，建議至少 16 GiB 系統記憶體。DRAM fallback 可能會在 file mmap 之外另行配置 mapped pinned copy。

```bash
nvidia-smi
nvcc --version
cmake --version
python3 --version
```

## 快速安裝

### 1. 安裝相依套件

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  python3
```

若尚未安裝 CUDA Toolkit，請選擇與目前 NVIDIA driver 與 GPU 相容的版本。

### 2. Clone

```bash
git clone https://github.com/vnlpscale/Summer.cpp.git "$HOME/Summer.cpp"
cd "$HOME/Summer.cpp"
```

### 3. 建置與安裝

```bash
bash scripts/install-summer.sh
```

安裝程式會套用 Turing/GTX 16 的 DRAM fallback patch，以 Release 模式建置 CUDA 版 `llama-tiered`，安裝到 `~/.local/bin`，移除舊 SummerCLI 指令，並建立 `~/models`。

GTX 1660 SUPER 可明確指定：

```bash
CUDA_ARCH=75 FORCE_MMQ=ON bash scripts/install-summer.sh
```

對不需要強制 MMQ 的 Tensor Core GPU：

```bash
FORCE_MMQ=OFF bash scripts/install-summer.sh
```

### 4. 設定 PATH

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

使用 split GGUF 時，請將全部分片放在同一目錄。

### 6. 執行

```bash
llama-tiered \
  -m "$HOME/models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  -n 128 \
  "請介紹你自己。"
```

CLI 會把 llama.cpp 的區塊標誌與 Summer.cpp 標題輸出至標準錯誤，因此標準輸出的生成文字仍適合透過 pipe 處理。

## 手動建置

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

## 自適應 Expert Cache

大型 MoE 模型有權重放在 SSD 時，可使用 `--cache-mib`。cache 容量包含在 `--vram-mib` 內，並會自動從 resident weight 預算扣除。

```bash
llama-tiered \
  -m "$HOME/models/Laguna-S-2.1-UD-IQ1_S.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  --cache-mib 1024 \
  -n 128 \
  "你好"
```

cache 會在 single-row decode 時依 router 歷史選取 hot expert，並在 multi-row prompt batch 中自動 bypass 以避免污染。結束 log 會顯示命中率、H2D/D2D 流量、admission 與 eviction 數量。

## 記憶體預算調整

GTX 1660 SUPER 的建議起點：

```text
--vram-mib 3800 --dram-mib 6500
```

- 發生 GPU allocation error 時降低 `--vram-mib`。
- tensor 被配置到 SSD 時提高 `--dram-mib`。
- 系統記憶體不足時使用更小或量化程度更高的 GGUF。
- GPU 同時負責桌面顯示時保留更多 VRAM。

以穩定性為優先時，請使用啟動 log 顯示 `SSD 0.00 MiB` 的設定。

## 疑難排解

- 缺少 `build/bin/llama-tiered`：重新執行 `bash scripts/install-summer.sh`。
- Turing/GTX 16 出現 `invalid argument`：套用 DRAM pinned fallback 後重新建置。
- `tensor_state layout did not match expected source`：還原 `ggml/src/ggml-cuda/tiered.cu` 後重新執行安裝程式。
- `operation not supported`：刪除 `build`，並以最新 patch 重建。
- CUDA illegal memory access：更新並重建，使用短 prompt 搭配 `compute-sanitizer --tool memcheck`。
- `summer: command not found`：目前支援的指令是 `llama-tiered`，並確認 `~/.local/bin` 在 `PATH` 中。

## SSD 串流狀態

SSD tier 會把 stacked MoE expert tensor 保留在 GGUF mmap 中，並在 `MUL_MAT_ID` 時只把選中的 expert slab 傳送到可重用的 VRAM scratch。現有限制包括：來源 page 可能常駐 page-locked RAM、scratch 依最大 stacked expert tensor 配置、傳輸與運算尚未重疊、cache 僅支援 single-row decode、共用同一模型的多個 context 會序列化 graph 執行，以及每種 GPU architecture 都需要實機驗證。

GTX 1660 SUPER 與 Laguna-S-2.1 IQ1_S 已測試 1、16、128 個生成 token，並通過 `compute-sanitizer` 驗證。其他 GPU 或模型請從短生成開始。

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
// 在 owner 存活期間建立並使用 llama_context。
llama_tiered_model_free(owner);
```

`llama_tiered_model_get_model()` 回傳的是 borrowed pointer，請勿直接傳給 `llama_model_free()`。

## 上游與授權

Summer.cpp 以 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 為基礎，採用與上游相同的 MIT License。詳見 [LICENSE](LICENSE)。
