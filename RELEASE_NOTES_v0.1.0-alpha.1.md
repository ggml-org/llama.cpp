# Summer.cpp v0.1.0-alpha.1

Summer.cppの初回プレリリースです。

Summer.cppは、VRAMを超えるGGUFモデルをNVIDIA GPU、system DRAM、SSDへ階層配置するtiered-memory backendと、対話用の`summer` CLIを追加したllama.cpp forkです。

## 主な機能

- VRAM、DRAM、SSDの3-tier tensor placement
- CUDA mapped host memoryとmapped pinned copy fallback
- MoE expert weight向けCUDA VMM selective streaming
- `llama-tiered`専用実行ファイル
- streaming表示対応の`summer` CLI
- GGUFモデルの検索と切り替え
- `<think>...</think>`および内部prompt echoの除去
- 空きVRAM検出、VRAM安全余裕、bounded OOM retry
- モデルサイズに応じたDRAM budget調整とsystem RAM事前検査
- Turing GPU向けDRAM matmul temporary VRAM staging
- 明示的なローカルPython runner

## 実機確認済み構成

- GPU: NVIDIA GeForce GTX 1660 SUPER
- Compute capability: 7.5
- Model: Qwen3.6-35B-A3B-UD-IQ1_M.gguf
- GGUF size: 約9.35 GiB
- VRAM budget: 3800 MiB
- DRAM budget: 6500 MiB
- SSD placement: 0 MiB
- Load time: 約5.65秒
- Prompt processing: 約31.7 tokens/s
- Token generation: 約27.7 tokens/s

性能値は短いpromptでの参考値です。model、context、CPU、PCIe、driver、background workloadにより変動します。

## インストール

```bash
git clone https://github.com/vnlpscale/Summer.cpp.git "$HOME/Summer.cpp"
cd "$HOME/Summer.cpp"
CUDA_ARCH=75 FORCE_MMQ=ON bash scripts/install-summer.sh
hash -r
summer
```

`CUDA_ARCH`は使用するGPUに合わせて変更してください。省略時はinstallerが`nvidia-smi`から検出します。

## 必要環境

- Linux
- NVIDIA GPU、NVIDIA Driver、CUDA Toolkit
- CMake、C++17 compiler
- Python 3.10以上
- GGUF model
- DRAM tierを使用できる十分なsystem RAM

## 既知の制約

- 現時点で最も安定している構成はVRAM + DRAMです。
- Turing世代でのSSD selective streamingは実験的です。
- `summer` CLIは各turnで`llama-tiered` processを起動するため、毎回model loadが発生します。
- GPUを使用する別processがある場合、必要な空きVRAMを確保できず起動を拒否します。
- GGUF model fileはReleaseに含まれません。
- prebuilt CUDA binaryは含まれません。対象GPU向けにlocal buildしてください。
- Python runnerはsandboxではなく、実行userと同じ権限で動作します。

## ライセンス

MIT License。詳細はrepositoryの`LICENSE`を参照してください。
