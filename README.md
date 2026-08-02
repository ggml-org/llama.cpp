<div align="center">

# Summer.cpp

### VRAMを超えるGGUFモデルをNVIDIA GPU・DRAM・SSDで動かす llama.cpp fork

**階層メモリ実行基盤・GGUF分割対応・CUDA向けローカル推論**

[![Platform](https://img.shields.io/badge/platform-Linux-111827?logo=linux&logoColor=white)](#必要環境)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?logo=nvidia&logoColor=white)](#必要環境)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-00599C?logo=cplusplus&logoColor=white)](#手動ビルド)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#ライセンス)

</div>

> [!IMPORTANT]
> 現在もっとも安定している構成は **VRAM + DRAM** です。GTX 1660 SUPERでは `--vram-mib 3800 --dram-mib 6500` を使い、SSD配置を0 MiBにした構成で実機動作を確認しています。Turing世代のSSD selective streamingはまだ実験的です。

## 概要

Summer.cppは、llama.cppへtiered-memory backendと専用実行ファイル`llama-tiered`を追加したforkです。

大きなGGUFモデルのtensorを、用途と予算に応じて次のメモリ階層へ配置します。

| Tier | 配置先 | 用途 |
|---|---|---|
| VRAM | CUDA device memory | 頻繁に使うdense weight、embedding、hot tensor |
| DRAM | CUDA mapped host memory | VRAMに収まらないweightをzero-copyまたはmapped pinned copyで参照 |
| SSD | CUDA Virtual Memory Management | routerが選択したMoE expert weightを必要時にstage |

```text
GGUF file
   │ mmap
   ▼
Placement planner
   ├── VRAM: resident CUDA allocations
   ├── DRAM: mapped host memory / pinned copy fallback
   └── SSD : temporary CUDA VMM mappings for MoE experts
```

## 実機確認済み構成

| 項目 | 確認値 |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 SUPER |
| Compute capability | 7.5 |
| Model | Qwen3.6-35B-A3B-UD-IQ1_M.gguf |
| GGUF size | 約9.35 GiB |
| VRAM budget | 3800 MiB |
| DRAM budget | 6500 MiB |
| SSD placement | 0 MiB |
| Load time | 約5.65秒 |
| Prompt processing | 約31.7 tokens/s |
| Token generation | 約27.7 tokens/s |

性能値は短いpromptでの一例です。model、context、sampler、CPU、PCIe、driver、background workloadにより変化します。

## 必要環境

- Linux。Ubuntu 22.04または24.04を推奨
- NVIDIA GPU
- NVIDIA Driver
- CUDA Toolkitと`nvcc`
- CMake
- C++17対応コンパイラ
- Python 3.10以上
- GGUFモデル
- モデルを保持できるSSD容量
- DRAM tierを使う場合は十分なsystem RAM

GTX 1660 SUPERと約9.35 GiBのモデルでは、OSや他processを含めて16 GiB以上のsystem RAMを推奨します。DRAM fallbackはfile mmapとは別にmapped pinned copyを確保する場合があります。

確認:

```bash
nvidia-smi
nvcc --version
cmake --version
python3 --version
```

## 最短インストール

### 1. 依存パッケージ

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  python3
```

CUDA Toolkitが未導入の場合は、使用中のNVIDIA DriverとGPUに適合するCUDA Toolkitを導入してください。

### 2. Clone

```bash
git clone https://github.com/vnlpscale/Summer.cpp.git "$HOME/Summer.cpp"
cd "$HOME/Summer.cpp"
```

### 3. Buildとinstall

インストーラは次を実行します。

1. GTX 16/Turing向けDRAM mapped pinned fallbackを適用
2. DRAM matmul stagingとprompt echo抑制patchを適用
3. CUDA版`llama-tiered`をRelease build
4. `llama-tiered`を`~/.local/bin`へinstall
5. 旧SummerCLIコマンドを削除
6. `~/models`を作成

```bash
bash scripts/install-summer.sh
```

GPU architectureは`nvidia-smi`から自動検出します。GTX 1660 SUPERでは`75`になります。明示する場合:

```bash
CUDA_ARCH=75 FORCE_MMQ=ON bash scripts/install-summer.sh
```

Tensor Core対応GPUでMMQの強制が不要な場合:

```bash
FORCE_MMQ=OFF bash scripts/install-summer.sh
```

### 4. PATH

```bash
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"

source "$HOME/.bashrc"
hash -r
command -v llama-tiered
```

### 5. モデル配置

```bash
mkdir -p "$HOME/models"
cp /path/to/model.gguf "$HOME/models/"
```

split GGUFを使う場合は、同じdirectoryへ全partを置いてください。

```text
~/models/
├── Qwen3.6-35B-A3B-UD-IQ1_M.gguf
└── large-model/
    ├── model-00001-of-00003.gguf
    ├── model-00002-of-00003.gguf
    └── model-00003-of-00003.gguf
```

### 6. 実行

```bash
llama-tiered \
  -m "$HOME/models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  -n 128 \
  "こんにちは。自己紹介してください。"
```

## 手動ビルド

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
```

生成物:

```text
build/bin/llama-tiered
```

user-local install:

```bash
install -Dm755 build/bin/llama-tiered "$HOME/.local/bin/llama-tiered"
```

## llama-tieredの使用

build directoryから直接実行する場合:

```bash
cd "$HOME/Summer.cpp"

./build/bin/llama-tiered \
  -m "$HOME/models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  -n 128 \
  "こんにちは。自己紹介してください。"
```

llama.cpp/CUDAのlogを非表示にする場合:

```bash
llama-tiered \
  -m "$HOME/models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  -n 128 \
  "こんにちは" \
  2>/dev/null
```

`2>/dev/null`はerrorも非表示にします。問題調査時には外してください。

## Memory budget調整

GTX 1660 SUPERでの開始値:

```text
--vram-mib 3800 --dram-mib 6500
```

調整方針:

- GPU allocation errorが出る場合は`--vram-mib`を下げる
- SSDへtensorが配置される場合は`--dram-mib`を増やす
- system RAMが不足する場合は、より小さいGGUFまたは量子化を使う
- desktop表示にもGPUを使う場合はVRAM reserveを十分に残す

起動logでは次を確認します。

```text
tiered weights: VRAM ... MiB, DRAM ... MiB, SSD 0.00 MiB (0 streamed tensors)
```

安定性を優先する場合、SSDが`0.00 MiB`になる設定を使用してください。

## Update

```bash
cd "$HOME/Summer.cpp"
git pull --ff-only
bash scripts/install-summer.sh
```

local patchが競合する場合:

```bash
cd "$HOME/Summer.cpp"
git status --short
git restore ggml/src/ggml-cuda/tiered.cu
bash scripts/install-summer.sh
```

## Troubleshooting

### `build/bin/llama-tiered: No such file or directory`

buildが完了していません。

```bash
cd "$HOME/Summer.cpp"
bash scripts/install-summer.sh
```

確認:

```bash
ls -lh build/bin/llama-tiered
```

### `llama_model_load: ... invalid argument`

Turing/GTX 16環境では、read-only GGUF mmapの`cudaHostRegisterMapped`が失敗する場合があります。DRAM pinned fallbackを適用して再buildします。

```bash
cd "$HOME/Summer.cpp"
python3 scripts/apply-tiered-dram-pinned-fallback.py
cmake --build build --target llama-tiered -j"$(nproc)"
```

### `tensor_state layout did not match expected source`

`tiered.cu`が別patchまたはlocal editで変更されています。

```bash
cd "$HOME/Summer.cpp"
cp ggml/src/ggml-cuda/tiered.cu /tmp/tiered.cu.backup
git restore ggml/src/ggml-cuda/tiered.cu
bash scripts/install-summer.sh
```

### `operation not supported`

古いbinaryを実行しているか、DRAM fallback適用前のsourceをbuildしている可能性があります。

```bash
cd "$HOME/Summer.cpp"
rm -rf build
bash scripts/install-summer.sh
```

### `CUDA error: an illegal memory access was encountered`

TuringでSSD selective streamingを使った場合に確認されています。`--dram-mib 1`のような設定は避け、VRAM + DRAMへ戻します。

```bash
llama-tiered \
  -m "$HOME/models/model.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  -n 32 \
  "test"
```

### `llama-tiered: command not found`

```bash
export PATH="$HOME/.local/bin:$PATH"
hash -r
command -v llama-tiered
```

永続化:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
source "$HOME/.bashrc"
```

### modelが見つからない

```bash
find "$HOME/models" -type f -iname '*.gguf'
```

## SSD streamingの状態

SSD tierはstacked MoE expert tensorを`MUL_MAT_ID`実行時にCUDA VMMへ一時mapする設計です。

現在の制約:

- Turing/GTX 16ではselective mapping時にMMVQ kernelのvectorized readがunmapped pageへ到達する可能性がある
- correctness-firstの同期処理で、転送とcomputeのoverlapは未実装
- persistent expert cacheは未実装
- GPU architectureごとのphysical validationが必要
- 本READMEの既定構成ではSSD streamingを使用しない

SSD pathを試す場合は、必ず短い生成、`CUDA_LAUNCH_BLOCKING=1`、`compute-sanitizer`などで検証してください。本番用途には推奨しません。

## Library API

```cpp
#include "llama-tiered.h"

llama_model_params model_params = llama_model_default_params();

llama_tiered_memory_params memory = llama_tiered_memory_default_params();
memory.vram_budget_bytes = 3800ull * 1024 * 1024;
memory.dram_budget_bytes = 6500ull * 1024 * 1024;

llama_tiered_model * owner = llama_tiered_model_load_from_file(
        "model.gguf",
        model_params,
        memory);

if (!owner) {
    fprintf(stderr, "tiered load failed: %s\n", llama_tiered_last_error());
    return 1;
}

llama_model * model = llama_tiered_model_get_model(owner);

// ownerが生存している間にllama_contextを作成して使用する。

llama_tiered_model_free(owner);
```

`llama_tiered_model_get_model()`が返すpointerはborrowedです。直接`llama_model_free()`へ渡さないでください。

## Project structure

```text
Summer.cpp/
├── examples/tiered-memory/                 tiered memoryの詳細設計
├── ggml/src/ggml-cuda/tiered.cu           CUDA tiered backend
├── scripts/
│   ├── apply-tiered-dram-pinned-fallback.py
│   ├── apply-tiered-dram-matmul-staging.py
│   ├── apply-tiered-no-prompt-echo.py
│   └── install-summer.sh
├── src/                                    llama library
└── build/bin/llama-tiered                  build後の実行file
```

## Uninstall

```bash
rm -f "$HOME/.local/bin/llama-tiered"
rm -f "$HOME/.local/bin/Summer.CPP"
rm -f "$HOME/.local/bin/summer"
```

sourceとmodelも不要な場合のみ削除してください。

```bash
rm -rf "$HOME/Summer.cpp"
# rm -rf "$HOME/models"  # GGUFも削除する場合だけ実行
```

## Upstream

Summer.cppは[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)を基盤にしています。llama.cpp、ggml、CUDA backend、量子化実装、model loaderを開発しているupstream contributorに感謝します。

詳細なupstream documentation:

- [llama.cpp build guide](docs/build.md)
- [CUDA backend documentation](docs/backend)
- [Model documentation](docs/models.md)
- [Contributing](CONTRIBUTING.md)

## ライセンス

このリポジトリはupstream llama.cppと同じMIT Licenseです。詳細は[LICENSE](LICENSE)を参照してください。
