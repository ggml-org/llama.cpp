<div align="center">

# Summer.cpp

### VRAM보다 큰 GGUF 모델을 NVIDIA GPU, DRAM, SSD에 나누어 실행하는 llama.cpp fork

**계층형 메모리 실행 · 분할 GGUF 지원 · CUDA 로컬 추론**

[日本語](README.md) · [English](README.en.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · **한국어** · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md)

[![Platform](https://img.shields.io/badge/platform-Linux-111827?logo=linux&logoColor=white)](#요구-사항)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?logo=nvidia&logoColor=white)](#요구-사항)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-00599C?logo=cplusplus&logoColor=white)](#수동-빌드)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#라이선스)

</div>

> [!IMPORTANT]
> 현재 가장 안정적인 구성은 **VRAM + DRAM**입니다. GTX 1660 SUPER에서 `--vram-mib 3800 --dram-mib 6500`, SSD 배치 0 MiB 구성을 검증했습니다. Turing GPU의 선택적 SSD 스트리밍은 아직 실험적입니다.

## 개요

Summer.cpp는 llama.cpp에 계층형 메모리 백엔드와 전용 실행 파일 `llama-tiered`를 추가한 fork입니다. 큰 GGUF tensor를 용도와 메모리 예산에 따라 다음 계층에 배치합니다.

| 계층 | 위치 | 용도 |
|---|---|---|
| VRAM | CUDA device memory | 자주 사용하는 dense weight, embedding, hot tensor |
| DRAM | CUDA mapped host memory | VRAM에 들어가지 않는 weight를 zero-copy 또는 mapped pinned copy로 참조 |
| SSD | file-backed GGUF mapping | 선택된 MoE expert를 필요할 때 stage하고 hot expert를 적응형 cache에 유지 |

```text
GGUF file
   │ mmap
   ▼
Placement planner
   ├── VRAM: resident CUDA allocations
   ├── DRAM: mapped host memory / pinned copy fallback
   └── SSD : selected MoE slabs -> adaptive VRAM cache -> reusable scratch
```

## 검증된 구성

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 SUPER |
| Compute capability | 7.5 |
| 모델 | Qwen3.6-35B-A3B-UD-IQ1_M.gguf |
| GGUF 크기 | 약 9.35 GiB |
| VRAM 예산 | 3800 MiB |
| DRAM 예산 | 6500 MiB |
| SSD 배치 | 0 MiB |
| 로드 시간 | 약 5.65초 |
| Prompt 처리 | 약 31.7 tokens/s |
| Token 생성 | 약 27.7 tokens/s |

위 수치는 짧은 prompt의 한 측정 사례입니다. 모델, context, sampler, CPU, PCIe, driver, 백그라운드 부하에 따라 달라집니다.

## 요구 사항

- Linux, Ubuntu 22.04 또는 24.04 권장
- NVIDIA GPU와 드라이버
- CUDA Toolkit 및 `nvcc`
- CMake와 C++17 컴파일러
- Python 3.10 이상
- GGUF 모델
- 모델을 저장할 충분한 SSD 공간
- DRAM tier 사용 시 충분한 시스템 RAM

GTX 1660 SUPER와 약 9.35 GiB 모델에는 OS와 다른 프로세스를 포함해 최소 16 GiB 시스템 RAM을 권장합니다. DRAM fallback은 file mmap과 별도로 mapped pinned copy를 할당할 수 있습니다.

```bash
nvidia-smi
nvcc --version
cmake --version
python3 --version
```

## 빠른 설치

### 1. 의존성 설치

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  python3
```

CUDA Toolkit이 없다면 현재 NVIDIA 드라이버와 GPU에 맞는 버전을 설치하십시오.

### 2. Clone

```bash
git clone https://github.com/vnlpscale/Summer.cpp.git "$HOME/Summer.cpp"
cd "$HOME/Summer.cpp"
```

### 3. 빌드 및 설치

```bash
bash scripts/install-summer.sh
```

설치 스크립트는 Turing/GTX 16용 DRAM fallback patch를 적용하고, CUDA `llama-tiered`를 Release로 빌드해 `~/.local/bin`에 설치하며, 이전 SummerCLI 명령을 제거하고 `~/models`를 생성합니다.

GTX 1660 SUPER에서 명시적으로 지정하려면:

```bash
CUDA_ARCH=75 FORCE_MMQ=ON bash scripts/install-summer.sh
```

강제 MMQ가 필요 없는 Tensor Core GPU에서는:

```bash
FORCE_MMQ=OFF bash scripts/install-summer.sh
```

### 4. PATH 설정

```bash
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
source "$HOME/.bashrc"
hash -r
command -v llama-tiered
```

### 5. 모델 배치

```bash
mkdir -p "$HOME/models"
cp /path/to/model.gguf "$HOME/models/"
```

split GGUF를 사용할 때는 모든 part를 같은 디렉터리에 두십시오.

### 6. 실행

```bash
llama-tiered \
  -m "$HOME/models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  -n 128 \
  "자기소개를 해 주세요."
```

CLI는 llama.cpp 블록 로고와 Summer.cpp 배너를 표준 오류로 출력합니다. 따라서 표준 출력의 생성 텍스트는 pipe 처리에 사용할 수 있습니다.

## 수동 빌드

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

## 적응형 Expert Cache

SSD에 MoE weight를 배치하는 대형 모델은 `--cache-mib`를 사용할 수 있습니다. cache 용량은 `--vram-mib`에 포함되며 resident weight 예산에서 자동 차감됩니다.

```bash
llama-tiered \
  -m "$HOME/models/Laguna-S-2.1-UD-IQ1_S.gguf" \
  --vram-mib 3800 \
  --dram-mib 6500 \
  --cache-mib 1024 \
  -n 128 \
  "안녕하세요"
```

cache는 single-row decode에서 router 이력으로 hot expert를 선택하고, multi-row prompt batch에서는 오염을 피하기 위해 자동으로 bypass됩니다. 종료 log에는 hit rate, H2D/D2D 전송량, admission, eviction 수가 표시됩니다.

## 메모리 예산 조정

GTX 1660 SUPER의 시작값:

```text
--vram-mib 3800 --dram-mib 6500
```

- GPU allocation error가 발생하면 `--vram-mib`를 낮춥니다.
- tensor가 SSD에 배치되면 `--dram-mib`를 높입니다.
- 시스템 RAM이 부족하면 더 작거나 더 강하게 양자화된 GGUF를 사용합니다.
- GPU가 데스크톱 표시에도 사용되면 VRAM 여유를 더 남깁니다.

안정성을 우선할 때는 시작 log에 `SSD 0.00 MiB`가 표시되는 구성을 사용하십시오.

## 문제 해결

- `build/bin/llama-tiered`가 없음: `bash scripts/install-summer.sh`를 다시 실행합니다.
- Turing/GTX 16에서 `invalid argument`: DRAM pinned fallback을 적용하고 다시 빌드합니다.
- `tensor_state layout did not match expected source`: `ggml/src/ggml-cuda/tiered.cu`를 복원한 뒤 설치 스크립트를 다시 실행합니다.
- `operation not supported`: `build`를 삭제하고 최신 patch로 재빌드합니다.
- CUDA illegal memory access: 업데이트 후 재빌드하고 짧은 prompt를 `compute-sanitizer --tool memcheck`로 확인합니다.
- `summer: command not found`: 지원되는 명령은 `llama-tiered`이며 `~/.local/bin`이 `PATH`에 있어야 합니다.

## SSD 스트리밍 상태

SSD tier는 stacked MoE expert tensor를 GGUF mmap에 유지하고 `MUL_MAT_ID` 실행 시 선택된 expert slab만 재사용 가능한 VRAM scratch로 전송합니다. 현재는 source page가 page-locked RAM에 상주할 수 있고, scratch가 가장 큰 stacked expert tensor 크기로 할당되며, 전송과 계산이 overlap되지 않고, adaptive cache가 single-row decode만 지원하며, 같은 모델을 공유하는 여러 context의 graph 실행이 직렬화됩니다. GPU architecture별 실기 검증도 필요합니다.

GTX 1660 SUPER와 Laguna-S-2.1 IQ1_S에서는 1, 16, 128 token 생성과 `compute-sanitizer`를 검증했습니다. 다른 GPU와 모델에서는 짧은 생성부터 확인하십시오.

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
// owner가 살아 있는 동안 llama_context를 생성하고 사용합니다.
llama_tiered_model_free(owner);
```

`llama_tiered_model_get_model()`의 반환값은 borrowed pointer입니다. `llama_model_free()`에 직접 전달하지 마십시오.

## 업스트림 및 라이선스

Summer.cpp는 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)를 기반으로 하며 동일한 MIT License를 사용합니다. 자세한 내용은 [LICENSE](LICENSE)를 참조하십시오.
