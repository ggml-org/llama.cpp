# iFairy ARM 3W LUT (V2) — 状态 / 性能记录

Status: Draft (2026-01-25)

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
  - `./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 3`
- Result (build `f0fac4ca`):
  - `pp128`: `151.50 ± 3.87 tok/s`
  - `tg256`: `80.30 ± 2.92 tok/s`

### lut16（GGML_IFAIRY_ARM_LUT=ON, GGML_IFAIRY_LUT=1）
- Build:
  - `cmake -B build-rel-lut -DCMAKE_BUILD_TYPE=Release -DGGML_IFAIRY_ARM_LUT=ON`
  - `cmake --build build-rel-lut`
- Command:
  - `GGML_IFAIRY_LUT=1 ./build-rel-lut/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 3`
- Result (build `2fb48487`):
  - `pp128`: `108.95 ± 1.10 tok/s`
  - `tg256`: `65.89 ± 2.05 tok/s`

### lut_c（GGML_IFAIRY_LUT_IMPL=lut_c）
- Command:
  - `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut_c ./build-rel-lut/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 3`
- Result (build `2fb48487`):
  - `pp128`: `101.21 ± 1.19 tok/s`
  - `tg256`: `68.98 ± 2.18 tok/s`

### microbench（GGML_IFAIRY_ARM_LUT=ON）
- `./build-rel-lut/bin/ifairy-actq-microbench`: `ns/iter=623.10`
- `./build-rel-lut/bin/ifairy-vecdot-microbench`: `ns/vecdot=45.42`
- `./build-rel-lut/bin/ifairy-microbench` (merged64 N==1, m=256 k=4096): `ns/iter=170151.2`

### lut16（当前 V2 核心路径）
- microbench（GGML_IFAIRY_ARM_LUT=ON）:
  - `./build-rel-lut/bin/ifairy-microbench` (lut16 N==1, m=256 k=4096): `ns/iter=42004.8`

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

### 2026-01-25 (working tree)
- Correctness:
  - `./build-rel/bin/test-ifairy`: PASS (LUT backend tests skipped, GGML_IFAIRY_ARM_LUT disabled)
  - `./build-rel-lut/bin/test-ifairy`: PASS
- `llama-bench` (model: `models/Fairy-plus-minus-i-700M/ifairy.gguf`, threads=4, pp128+tg256, repetitions=3):
  - `./build-rel/bin/llama-bench ...`: `pp128=151.50 ± 3.87 tok/s`, `tg256=80.30 ± 2.92 tok/s` (raw: `tmp/bench/bench_build-rel.txt`)
  - `GGML_IFAIRY_LUT=1 ./build-rel-lut/bin/llama-bench ...`: `pp128=94.69 ± 0.65 tok/s`, `tg256=61.36 ± 1.81 tok/s` (raw: `tmp/bench/bench_build-rel-lut.txt`)
- microbench（GGML_IFAIRY_ARM_LUT=ON）:
  - `./build-rel-lut/bin/ifairy-actq-microbench`: `ns/iter=605.49`
  - `./build-rel-lut/bin/ifairy-vecdot-microbench`: `ns/vecdot=44.38`
  - `./build-rel-lut/bin/ifairy-microbench` (lut16 N==1, m=256 k=4096): `ns/iter=42004.8`
- xctrace CPU Counters（模板：`test.tracetemplate`；统计：仅计 `Running` 且 core 不迁移的相邻采样增量）：
  - microbench (`tmp/xctrace/ifairy_lut_microbench_cpu_counters.trace`, ~10s):
    - `ARM_STALL=18676194623`, `CORE_ACTIVE_CYCLE=39673256029`, `ARM_L1D_CACHE_LMISS_RD=128322948`, `ARM_L1D_CACHE_RD=26161839493`, `L1D_TLB_MISS=221092`
    - `stall_ratio=0.470750`, `l1d_miss_rate=0.004905`, `tlb_miss_per_active=0.00000557`
  - llama-bench (`tmp/xctrace/ifairy_lut_llama_bench_cpu_counters.trace`, ~4s):
    - `ARM_STALL=4436605053`, `CORE_ACTIVE_CYCLE=12980115906`, `ARM_L1D_CACHE_LMISS_RD=29846840`, `ARM_L1D_CACHE_RD=8020881750`, `L1D_TLB_MISS=30783799`
    - `stall_ratio=0.341800`, `l1d_miss_rate=0.003721`, `tlb_miss_per_active=0.00237161`

### 2026-01-25 (build `2fb48487`)
- Correctness:
  - `./build-rel-lut/bin/test-ifairy`: PASS
- `llama-bench` (model: `models/Fairy-plus-minus-i-700M/ifairy.gguf`, threads=4, pp128+tg256, repetitions=3):
  - `GGML_IFAIRY_LUT=1 ./build-rel-lut/bin/llama-bench ...`: `pp128=108.95 ± 1.10 tok/s`, `tg256=65.89 ± 2.05 tok/s` (raw: `tmp/bench/bench_build-rel-lut_lut16_2fb48487.txt`)
  - `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut_c ./build-rel-lut/bin/llama-bench ...`: `pp128=101.21 ± 1.19 tok/s`, `tg256=68.98 ± 2.18 tok/s` (raw: `tmp/bench/bench_build-rel-lut_lut_c_2fb48487.txt`)

---

## A/B 结果（Raw Logs）

建议每条记录包含：
- 完整命令行（含 env）
- 原始输出（tok/s）
- 备注（M/K/N、线程数、模型）
