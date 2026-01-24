# iFairy ARM 3W LUT (V2) — 状态 / 性能记录

Status: Draft (2026-01-24)

本文件用于替代旧的 `IFAIRY_ARM_3W_LUT_STATUS.md` 的后续增量记录（旧文件不再修改）。

相关方案文档：
- `IFAIRY_ARM_3W_LUT_V2_REFACTOR_PLAN.md`
- `IFAIRY_ARM_3W_LUT_V2_LUT_C_INTEGRATION_PLAN.md`

---

## 基线（Baseline）

### merged64（现有 fastest 路径）
- Machine: Mac16,12 (Apple M4), macOS 26.2 (25C56)
- Build: `cmake -B build-rel -DCMAKE_BUILD_TYPE=Release` + `cmake --build build-rel` (OpenMP not found)
- Command:
  - `./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 1 --no-warmup`
- Result (build `a3329995`):
  - `pp128`: `162.82 tok/s`
  - `tg256`: `87.78 tok/s`

### merged64（GGML_IFAIRY_ARM_LUT=ON, CPU-only 配置）
- Build:
  - `cmake -B build-rel-lut -DCMAKE_BUILD_TYPE=Release -DGGML_IFAIRY_ARM_LUT=ON`
  - `cmake --build build-rel-lut`
- Command:
  - `GGML_IFAIRY_LUT=1 ./build-rel-lut/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 1 --no-warmup`
- Result (build `a3329995`):
  - `pp128`: `32.43 tok/s`
  - `tg256`: `25.56 tok/s`

### microbench（GGML_IFAIRY_ARM_LUT=ON）
- `./build-rel-lut/bin/ifairy-actq-microbench`: `ns/iter=623.10`
- `./build-rel-lut/bin/ifairy-vecdot-microbench`: `ns/vecdot=45.42`
- `./build-rel-lut/bin/ifairy-microbench` (merged64 N==1, m=256 k=4096): `ns/iter=170151.2`

---

## 变更记录（Changelog）

按日期追加（YYYY-MM-DD）：

### 2026-01-24
- 初始化 V2 文档占位（尚未进行代码重构与 lut_c 接入）。

### 2026-01-24 (working tree)
- `test-ifairy`:
  - `./build-rel/bin/test-ifairy`: PASS (LUT tests skipped, GGML_IFAIRY_ARM_LUT disabled)
  - `./build-rel-lut/bin/test-ifairy`: PASS
- `llama-bench` (model: `models/Fairy-plus-minus-i-700M/ifairy.gguf`, threads=4, pp128+tg256):
  - `./build-rel/bin/llama-bench ...`: `pp128=169.09 tok/s`, `tg256=91.25 tok/s`
  - `GGML_IFAIRY_LUT=1 ./build-rel-lut/bin/llama-bench ...`: `pp128=35.36 tok/s`, `tg256=31.10 tok/s`
- microbench (`GGML_IFAIRY_ARM_LUT=ON` build):
  - `./build-rel-lut/bin/ifairy-actq-microbench`: `ns/iter=634.91`
  - `./build-rel-lut/bin/ifairy-vecdot-microbench`: `ns/vecdot=58.49`
  - `./build-rel-lut/bin/ifairy-microbench` (merged64 N==1, m=256 k=4096): `ns/iter=179699.4`

---

## A/B 结果（Raw Logs）

建议每条记录包含：
- 完整命令行（含 env）
- 原始输出（tok/s）
- 备注（M/K/N、线程数、模型）
