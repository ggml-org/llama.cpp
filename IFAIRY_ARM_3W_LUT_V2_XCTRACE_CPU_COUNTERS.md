# iFairy ARM 3W LUT (V2) — xctrace CPU Counters 方案（`test.tracetemplate`）

Status: Draft (2026-01-24)

本文件记录一套可复用的 **xctrace CPU Counters** 测试方案，用于在 V2 重构与 `lut_c/` 接入阶段对性能瓶颈做定量分析。

参考（仅作方法说明，不修改）：
- `IFAIRY_VEC_DOT_DECODE_BOUNDS_XCTRACE.md`

---

## 1) 模板与指标

本 repo 根目录包含自定义 counters 模板：
- `test.tracetemplate`

模板中选择的指标：
- `ARM_STALL`
- `CORE_ACTIVE_CYCLE`
- `ARM_L1D_CACHE_LMISS_RD`
- `ARM_L1D_CACHE_RD`
- `L1D_TLB_MISS`

建议在分析时至少关注这些派生比值（用于快速判断瓶颈方向）：
- `stall_ratio = ARM_STALL / CORE_ACTIVE_CYCLE`
- `l1d_miss_rate = ARM_L1D_CACHE_LMISS_RD / ARM_L1D_CACHE_RD`
- `tlb_miss_per_active = L1D_TLB_MISS / CORE_ACTIVE_CYCLE`

> 说明：不同 macOS / 不同模板下，export 表里 counter 的顺序与字段名可能不同；以实际导出为准。

---

## 2) 采集命令（推荐）

### 2.1 目录准备

```bash
mkdir -p tmp/xctrace
```

### 2.2 microbench（更稳定，适合做内核迭代）

以 decode-style（N==1）为例：

```bash
xcrun xctrace record --template test.tracetemplate \
  --output tmp/xctrace/ifairy_lut_microbench_cpu_counters.trace \
  --time-limit 10s --no-prompt \
  --launch -- ./build-rel-lut/bin/ifairy-microbench \
    --m 256 --k 4096 --iters 1000000000 --warmup 0 --seed 1 \
  > /dev/null
```

### 2.3 llama-bench（端到端）

```bash
GGML_IFAIRY_LUT=1 xcrun xctrace record --template test.tracetemplate \
  --output tmp/xctrace/ifairy_lut_llama_bench_cpu_counters.trace \
  --time-limit 30s --no-prompt \
  --launch -- ./build-rel-lut/bin/llama-bench \
    -m models/Fairy-plus-minus-i-700M/ifairy.gguf \
    --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 1 --no-warmup \
  > /dev/null
```

---

## 3) 导出与统计（最小工作流）

导出 counters 表：

```bash
xcrun xctrace export \
  --input tmp/xctrace/ifairy_lut_microbench_cpu_counters.trace \
  --xpath '/trace-toc/run[@number=\"1\"]/data/table[@schema=\"kdebug-counters-with-time-sample\"]' \
  > tmp/xctrace/ifairy_lut_microbench_cpu_counters.xml
```

然后按 `IFAIRY_VEC_DOT_DECODE_BOUNDS_XCTRACE.md` 的方法：
- 从导出的 `pmc-events` 序列中取相邻采样点增量
- 只在 `Running` 且 core 不迁移时计入（避免污染）
- 计算 `stall_ratio / l1d_miss_rate / tlb_miss_per_active`

建议把每次采样的：
- 完整命令（含 env）
- 关键比值与原始计数
追加记录到：`IFAIRY_ARM_3W_LUT_V2_STATUS.md`

