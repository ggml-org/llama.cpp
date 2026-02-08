# iFairy ARM 3W LUT (V2) — xctrace CPU Counters / Time Profiler 测试方案

Status: Draft (2026-01-25)

目标：
- 用一套可复现的 xctrace 流程对 LUT 的端到端与 microbench 做性能归因
- 统一记录：Time Profiler（热点符号）+ CPU Counters（stall/cache/tlb 指标）
- 支持对比：`GGML_IFAIRY_LUT_IMPL=lut16` vs `GGML_IFAIRY_LUT_IMPL=lut_c`（或后续新 backend）

注意：
- 本方案使用本地模板 `test.tracetemplate`（不入库），其指标选择参考 `IFAIRY_VEC_DOT_DECODE_BOUNDS_XCTRACE.md`。
- 模板中已选择 counters：
  - `ARM_STALL`
  - `CORE_ACTIVE_CYCLE`
  - `ARM_L1D_CACHE_LMISS_RD`
  - `ARM_L1D_CACHE_RD`
  - `L1D_TLB_MISS`

---

## 1) 环境 / 前置

- Xcode / Instruments 可用（`xcrun xctrace`）
- 已编译：
  - baseline: `build-rel`
  - LUT build: `build-rel-lut`（`-DGGML_IFAIRY_LUT_CPU=ON`）

推荐命令（Release CPU）：
- `cmake -B build-rel -DCMAKE_BUILD_TYPE=Release`
- `cmake --build build-rel`
- `cmake -B build-rel-lut -DCMAKE_BUILD_TYPE=Release -DGGML_IFAIRY_LUT_CPU=ON`
- `cmake --build build-rel-lut`

---

## 2) 统一 workload（端到端）

统一用 `--repetitions 3`：
- `./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 3`
- `GGML_IFAIRY_LUT=1 ./build-rel-lut/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 3`

对比 backend：
- `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 ...`
- `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut_c ...`

---

## 3) 采样：CPU Counters（自定义 template）

### 3.1 record

样例（lut16）：
- `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 xcrun xctrace record --template test.tracetemplate --output tmp/xctrace/lut16_llama_bench_cpu_counters.trace --time-limit 30s --no-prompt --launch -- ./build-rel-lut/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 3`

样例（lut_c）：
- `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut_c xcrun xctrace record --template test.tracetemplate --output tmp/xctrace/lut_c_llama_bench_cpu_counters.trace --time-limit 30s --no-prompt --launch -- ./build-rel-lut/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 3`

说明：
- `--time-limit 30s` 是上限；llama-bench 退出后 Instruments 仍可能等待到 time limit（TOC 会显示 `Time limit reached`）。
- counters 的 event 顺序请以 `--toc` 输出为准：
  - `xcrun xctrace export --toc --input tmp/xctrace/lut16_llama_bench_cpu_counters.trace`

### 3.2 export（建议用 counters-profile）

导出 `counters-profile`（每条采样窗口的 counters；便于直接求和）：
- `xcrun xctrace export --input tmp/xctrace/lut16_llama_bench_cpu_counters.trace --xpath '/trace-toc/run[@number=\"1\"]/data/table[@schema=\"counters-profile\"]' > tmp/xctrace/lut16_llama_bench_counters_profile.xml`
- `xcrun xctrace export --input tmp/xctrace/lut_c_llama_bench_cpu_counters.trace --xpath '/trace-toc/run[@number=\"1\"]/data/table[@schema=\"counters-profile\"]' > tmp/xctrace/lut_c_llama_bench_counters_profile.xml`

### 3.3 summary（脚本）

用脚本对 `process==llama-bench` + `state==Running` 的 samples 求和：
- `python3 scripts/ifairy_xctrace_counters_profile_summary.py --process-name llama-bench --events ARM_STALL CORE_ACTIVE_CYCLE ARM_L1D_CACHE_LMISS_RD ARM_L1D_CACHE_RD L1D_TLB_MISS < tmp/xctrace/lut16_llama_bench_counters_profile.xml`

建议派生指标（可在 V2_STATUS 中记录）：
- `stall_ratio = ARM_STALL / CORE_ACTIVE_CYCLE`
- `l1d_miss_rate = ARM_L1D_CACHE_LMISS_RD / ARM_L1D_CACHE_RD`
- `tlb_miss_per_active = L1D_TLB_MISS / CORE_ACTIVE_CYCLE`

---

## 4) 采样：Time Profiler（热点符号）

### 4.1 record

- `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 xcrun xctrace record --template \"Time Profiler\" --output tmp/xctrace/lut16_llama_bench_time_profiler.trace --time-limit 30s --no-prompt --launch -- ./build-rel-lut/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 3`
- `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut_c xcrun xctrace record --template \"Time Profiler\" --output tmp/xctrace/lut_c_llama_bench_time_profiler.trace --time-limit 30s --no-prompt --launch -- ./build-rel-lut/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 3`

### 4.2 export + leaf summary

导出 `time-profile`（包含符号名；优先于 `time-sample`）：
- `xcrun xctrace export --input tmp/xctrace/lut16_llama_bench_time_profiler.trace --xpath '/trace-toc/run[@number=\"1\"]/data/table[@schema=\"time-profile\"]' > tmp/xctrace/lut16_llama_bench_time_profile.xml`
- `xcrun xctrace export --input tmp/xctrace/lut_c_llama_bench_time_profiler.trace --xpath '/trace-toc/run[@number=\"1\"]/data/table[@schema=\"time-profile\"]' > tmp/xctrace/lut_c_llama_bench_time_profile.xml`

leaf（Top-N）：
- `python3 scripts/ifairy_xctrace_leaf.py --top 60 < tmp/xctrace/lut16_llama_bench_time_profile.xml > tmp/xctrace/lut16_llama_bench_time_profile.leaf.txt`
- `python3 scripts/ifairy_xctrace_leaf.py --top 60 < tmp/xctrace/lut_c_llama_bench_time_profile.xml > tmp/xctrace/lut_c_llama_bench_time_profile.leaf.txt`

建议在 V2_STATUS 中记录：
- Top 10 leaf 符号（% + ms）
- 备注“下一步主攻热点”（通常是最大的 leaf）

