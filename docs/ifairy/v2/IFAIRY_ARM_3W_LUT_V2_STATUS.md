# iFairy ARM 3W LUT (V2) — 状态 / 性能记录

Status: Draft (2026-03-13)

本文件用于替代旧的 `../legacy/IFAIRY_ARM_3W_LUT_STATUS.md` 的后续增量记录（旧文件不再修改）。

相关方案文档：
- `IFAIRY_ARM_3W_LUT_V2_REFACTOR_PLAN.md`
- `IFAIRY_ARM_3W_LUT_V2_LUT_C_INTEGRATION_PLAN.md`

---

## 基线（Baseline）

### vec_dot（现有 fastest 路径）
- Machine: Mac16,12 (Apple M4), macOS 26.2 (25C56)
- Build: `cmake -B build-rel -DCMAKE_BUILD_TYPE=Release` + `cmake --build build-rel` (OpenMP not found)
- Command:
  - `./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 3`
- Result (build `f0fac4ca`):
  - `pp128`: `151.50 ± 3.87 tok/s`
  - `tg256`: `80.30 ± 2.92 tok/s`

### lut16（GGML_IFAIRY_LUT_CPU=ON, GGML_IFAIRY_LUT=1）
- Build:
  - `cmake -B build-rel-lut -DCMAKE_BUILD_TYPE=Release -DGGML_IFAIRY_LUT_CPU=ON`
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

### microbench（GGML_IFAIRY_LUT_CPU=ON）
- `./build-rel-lut/bin/ifairy-actq-microbench`: `ns/iter=623.10`
- `./build-rel-lut/bin/ifairy-vecdot-microbench`: `ns/vecdot=45.42`
- `./build-rel-lut/bin/ifairy-microbench` (merged64 N==1, m=256 k=4096): `ns/iter=170151.2`

### lut16（当前 V2 核心路径）
- microbench（GGML_IFAIRY_LUT_CPU=ON）:
  - `./build-rel-lut/bin/ifairy-microbench` (lut16 N==1, m=256 k=4096): `ns/iter=42004.8`

---

## 变更记录（Changelog）

按日期追加（YYYY-MM-DD）：

### 2026-03-13 (build `8d73f59a`)
- 变更摘要：
  - 在 `src/llama-quant.cpp` 中补齐 `ifairy` bare `output` 张量的识别逻辑，使 `llama-quantize --output-tensor-type q6_k` 可以直接作用于 `ifairy.gguf` 的输出层，而不要求张量名必须是 `output.weight`。
  - 将 `output -> q6_k` 保留为可用优化项；当前不建议将其与 LUT 一起无条件默认开启。
- 输出层量化命令（仅量化 `output`，其余 `ifairy` 权重保持原型）：
  - `./build-rel/bin/llama-quantize --allow-requantize --output-tensor-type q6_k models/Fairy-plus-minus-i-700M/ifairy.gguf tmp/bench-ifairy-output-q6k/ifairy-output-q6k.gguf IFairy $(sysctl -n hw.ncpu)`
- 模型格式校验：
  - 原模型：`output=F16`, `token_embd=F32`, blocks=`F16_I2`
  - q6_k 模型：`output=Q6_K`, `token_embd=F32`, blocks=`F16_I2`
  - 模型大小：`549 MiB -> 439 MiB`（约 `-20.1%`）
- 精度评估（Machine: Mac16,12 / Apple M4；`GGML_IFAIRY_LUT=1`; `threads=4`; `wikitext-2` 子集 16 chunks / 32768 tokens）：
  - 语料准备：
    - `mkdir -p tmp/wikitext-2 && cd tmp/wikitext-2 && curl -L -o wikitext-2-raw-v1.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip && unzip -o wikitext-2-raw-v1.zip`
  - Baseline:
    - `GGML_IFAIRY_LUT=1 ./build-rel/bin/llama-perplexity -m models/Fairy-plus-minus-i-700M/ifairy.gguf -f tmp/wikitext-2/wikitext-2-raw/wiki.test.raw -c 2048 --chunks 16 -t 4 -ngl 0 --device none --no-warmup`
    - `PPL = 32.7540 +/- 1.05057`
  - q6_k:
    - `GGML_IFAIRY_LUT=1 ./build-rel/bin/llama-perplexity -m tmp/bench-ifairy-output-q6k/ifairy-output-q6k.gguf -f tmp/wikitext-2/wikitext-2-raw/wiki.test.raw -c 2048 --chunks 16 -t 4 -ngl 0 --device none --no-warmup`
    - `PPL = 32.7137 +/- 1.04931`
  - Delta:
    - `-0.0403`（约 `-0.12%`，视为噪声级）
  - 生成 smoke test：
    - 英文事实型 prompt 下 baseline 与 q6_k 均可正确生成 `Paris`
    - 中文事实型 prompt 下 baseline 与 q6_k 均回显 prompt，表现为模型自身能力限制，未观察到 q6_k 独有的退化
  - Raw logs:
    - `tmp/bench-ifairy-output-q6k/ppl16-baseline.txt`
    - `tmp/bench-ifairy-output-q6k/ppl16-output-q6k.txt`
    - `tmp/bench-ifairy-output-q6k/gen-debug-baseline.txt`
    - `tmp/bench-ifairy-output-q6k/gen-debug-q6k.txt`
    - `tmp/bench-ifairy-output-q6k/gen-fact-baseline.txt`
    - `tmp/bench-ifairy-output-q6k/gen-fact-q6k.txt`
- 性能评估（decode 优先；`build-rel/bin/llama-bench`; `-p 1 -n 512 -r 3`; 每次单项 benchmark 后 `sleep 60s` 冷却；`t=10` 测试被手动中断，不纳入结论）：
  - Command template:
    - `GGML_IFAIRY_LUT={0,1} ./build-rel/bin/llama-bench -m {model} -t {1,2,4,6,8} -ngl 0 --device none -p 1 -n 512 -r 3 -o jsonl`
  - `tg512`（decode）结果：

    | threads | LUT=0 baseline | LUT=0 q6_k | q6_k delta | LUT=1 baseline | LUT=1 q6_k | q6_k delta |
    |---|---:|---:|---:|---:|---:|---:|
    | 1 | 32.92 | 32.95 | +0.1% | 35.51 | 36.27 | +2.1% |
    | 2 | 54.58 | 58.53 | +7.2% | 57.42 | 60.48 | +5.3% |
    | 4 | 75.57 | 83.96 | +11.1% | 72.37 | 81.65 | +12.8% |
    | 6 | 75.48 | 83.60 | +10.8% | 56.29 | 60.61 | +7.7% |
    | 8 | 72.81 | 80.24 | +10.2% | 59.84 | 63.47 | +6.1% |

  - LUT on/off 对比（`tg512`）：

    | threads | baseline: LUT1 vs LUT0 | q6_k: LUT1 vs LUT0 |
    |---|---:|---:|
    | 1 | +7.9% | +10.1% |
    | 2 | +5.2% | +3.3% |
    | 4 | -4.2% | -2.8% |
    | 6 | -25.4% | -27.5% |
    | 8 | -17.8% | -20.9% |

  - 结论：
    - `output -> q6_k` 在已完成线程档位中是稳定正收益，decode 大致为 `+5%` 到 `+13%`
    - 当前 Apple M4 + 本模型上，`GGML_IFAIRY_LUT=1` 仅在低线程（`t=1/2`）有优势
    - 从 `t=4` 开始，no-LUT 路径反而更快；`t=6/8` 下 LUT 明显掉队
    - 当前建议：
      - 低线程（`1-2`）：`q6_k + LUT=1`
      - 中高线程（`4-8`）：`q6_k + LUT=0`
      - `q6_k` 保留为可用优化，但不应与 LUT 绑定为统一默认配置
  - Raw logs:
    - `tmp/bench-ifairy-output-q6k/matrix/*.jsonl`

### 2026-03-12 (build `ac58bc67`; baseline `b12bfdb6`)
- 变更摘要：
  - 将 x86 上的 2-weight merged LUT layout 迁移到 ARM LUT16 路径。
  - 对 `N == 1` 的 decode 小批量场景改为每线程 fused preprocess + qgemm，避免共享 LUT/barrier 开销。
- Correctness:
  - `./build-rel/bin/test-ifairy --ifairy-lut-only`: PASS
  - `./build-rel-lut/bin/test-ifairy --ifairy-lut-only`: PASS
- `llama-bench`（Machine: Mac16,12 / Apple M4；model=`models/Fairy-plus-minus-i-700M/ifairy.gguf`; threads=4; repetitions=3）：
  - Command (current):
    - `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 ./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 3`
  - Result (current `ac58bc67`):
    - `pp128`: `143.99 ± 7.42 tok/s`
    - `tg256`: `69.32 ± 1.03 tok/s`
  - Command (baseline `b12bfdb6`):
    - `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 ./build-rel/bin/llama-bench -m /Users/liweitao/Downloads/Codefield/cpp/llama.cpp/models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 3`
  - Result (baseline `b12bfdb6`):
    - `pp128`: `102.99 ± 1.57 tok/s`
    - `tg256`: `63.83 ± 1.13 tok/s`
  - Delta vs baseline:
    - `pp128`: `+39.8%`
    - `tg256`: `+8.6%`
- backend 诊断（输出 hash 一致）：
  - `./build-rel/bin/test-ifairy --ifairy-lut-backend-bench 4096 1 1536 4 10 100`
    - current `ac58bc67`: `lut16_ms_per_iter=0.128064`, `output_hash=0x3553f5fa20e46383`
    - baseline `b12bfdb6`: `lut16_ms_per_iter=0.141745`, `output_hash=0x3553f5fa20e46383`
    - delta: `+9.7%`
  - `./build-rel/bin/test-ifairy --ifairy-lut-backend-bench 4096 64 1536 4 5 20`
    - current `ac58bc67`: `lut16_ms_per_iter=4.076835`, `output_hash=0x54cc415c8e1c8383`
    - baseline `b12bfdb6`: `lut16_ms_per_iter=5.736187`, `output_hash=0x54cc415c8e1c8383`
    - delta: `+28.9%`

### 2026-01-28 (build `35a9928b`)
- Correctness:
  - `./build-rel/bin/test-ifairy`: PASS (LUT backend tests skipped, GGML_IFAIRY_LUT_CPU disabled)
  - `./build-rel-lut/bin/test-ifairy`: PASS
- `llama-cli` (lut_c sanity, no-cnv):
  - `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut_c ./build-rel-lut/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv`
- `llama-bench` (model: `models/Fairy-plus-minus-i-700M/ifairy.gguf`, threads=4, pp128+tg256, repetitions=3):
  - `GGML_IFAIRY_LUT=1 ./build-rel-lut/bin/llama-bench ...`: `pp128=112.76 ± 0.83 tok/s`, `tg256=67.07 ± 0.31 tok/s` (raw: `tmp/bench/bench_build-rel-lut_lut16_35a9928b.txt`)
  - `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut_c ./build-rel-lut/bin/llama-bench ...`: `pp128=110.10 ± 2.45 tok/s`, `tg256=65.13 ± 3.31 tok/s` (raw: `tmp/bench/bench_build-rel-lut_lut_c_35a9928b.txt`)

### 2026-01-24
- 初始化 V2 文档占位（尚未进行代码重构与 lut_c 接入）。

### 2026-01-24 (working tree)
- `test-ifairy`:
  - `./build-rel/bin/test-ifairy`: PASS (LUT tests skipped, GGML_IFAIRY_LUT_CPU disabled)
  - `./build-rel-lut/bin/test-ifairy`: PASS
- `llama-bench` (model: `models/Fairy-plus-minus-i-700M/ifairy.gguf`, threads=4, pp128+tg256):
  - `./build-rel/bin/llama-bench ...`: `pp128=169.09 tok/s`, `tg256=91.25 tok/s`
  - `GGML_IFAIRY_LUT=1 ./build-rel-lut/bin/llama-bench ...`: `pp128=35.36 tok/s`, `tg256=31.10 tok/s`
- microbench (`GGML_IFAIRY_LUT_CPU=ON` build):
  - `./build-rel-lut/bin/ifairy-actq-microbench`: `ns/iter=634.91`
  - `./build-rel-lut/bin/ifairy-vecdot-microbench`: `ns/vecdot=58.49`
  - `./build-rel-lut/bin/ifairy-microbench` (merged64 N==1, m=256 k=4096): `ns/iter=179699.4`

### 2026-01-25 (working tree)
- Correctness:
  - `./build-rel/bin/test-ifairy`: PASS (LUT backend tests skipped, GGML_IFAIRY_LUT_CPU disabled)
  - `./build-rel-lut/bin/test-ifairy`: PASS
- `llama-bench` (model: `models/Fairy-plus-minus-i-700M/ifairy.gguf`, threads=4, pp128+tg256, repetitions=3):
  - `./build-rel/bin/llama-bench ...`: `pp128=151.50 ± 3.87 tok/s`, `tg256=80.30 ± 2.92 tok/s` (raw: `tmp/bench/bench_build-rel.txt`)
  - `GGML_IFAIRY_LUT=1 ./build-rel-lut/bin/llama-bench ...`: `pp128=94.69 ± 0.65 tok/s`, `tg256=61.36 ± 1.81 tok/s` (raw: `tmp/bench/bench_build-rel-lut.txt`)
- microbench（GGML_IFAIRY_LUT_CPU=ON）:
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

### 2026-01-25 (lut16 qgemm decode simplification; working tree)
- Correctness:
  - `./build-rel/bin/test-ifairy`: PASS
  - `./build-rel-lut/bin/test-ifairy`: PASS
- `llama-bench` (model: `models/Fairy-plus-minus-i-700M/ifairy.gguf`, threads=4, pp128+tg256, repetitions=3):
  - `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 ./build-rel-lut/bin/llama-bench ...`: `pp128=114.50 ± 0.87 tok/s`, `tg256=70.14 ± 0.17 tok/s` (raw: `tmp/bench/bench_build-rel-lut_lut16_opt1.txt`)

### 2026-01-25 (xctrace: llama-bench, lut16 vs lut_c; build `2fb48487`)
- Commands: 见 `IFAIRY_ARM_3W_LUT_V2_XCTRACE_CPU_COUNTERS.md`
- Traces:
  - CPU Counters: `tmp/xctrace/lut16_llama_bench_cpu_counters.trace`, `tmp/xctrace/lut_c_llama_bench_cpu_counters.trace`
  - Time Profiler: `tmp/xctrace/lut16_llama_bench_time_profiler.trace`, `tmp/xctrace/lut_c_llama_bench_time_profiler.trace`
- CPU Counters summary（schema=`counters-profile`；filter：`process==llama-bench` 且 `state==Running`）：
  - lut16: `samples=90424`, `ARM_STALL=59300330463`, `CORE_ACTIVE_CYCLE=166258363154`, `ARM_L1D_CACHE_LMISS_RD=468689982`, `ARM_L1D_CACHE_RD=120092438462`, `L1D_TLB_MISS=172670353`
    - `stall_ratio=0.356676`, `l1d_miss_rate=0.003903`, `tlb_miss_per_active=0.001039`
  - lut_c: `samples=91377`, `ARM_STALL=76019849911`, `CORE_ACTIVE_CYCLE=198922166699`, `ARM_L1D_CACHE_LMISS_RD=658355729`, `ARM_L1D_CACHE_RD=143056368346`, `L1D_TLB_MISS=240202220`
    - `stall_ratio=0.382159`, `l1d_miss_rate=0.004602`, `tlb_miss_per_active=0.001208`
- Time Profiler leaf（Top 10）：
  - lut16: `tmp/xctrace/lut16_llama_bench_time_profile.leaf.txt`
    - `ggml_ifairy_lut_qgemm_lut16` (63.27%, 44485ms)
    - `ggml_vec_dot_f16` (14.27%, 10031ms)
    - `ggml_graph_compute_thread` (10.25%, 7207ms)
  - lut_c: `tmp/xctrace/lut_c_llama_bench_time_profile.leaf.txt`
    - `ggml_ifairy_lut_qgemm_lut16` (59.88%, 45996ms)
    - `ggml_graph_compute_thread` (15.37%, 11805ms)
    - `ggml_vec_dot_f16` (12.56%, 9645ms)
- 下一步主攻热点（按 leaf 占比）：`ggml_ifairy_lut_qgemm_lut16`（接入 `lut_c` kernel 后应成为主要对比对象）

---

## A/B 结果（Raw Logs）

建议每条记录包含：
- 完整命令行（含 env）
- 原始输出（tok/s）
- 备注（M/K/N、线程数、模型）
