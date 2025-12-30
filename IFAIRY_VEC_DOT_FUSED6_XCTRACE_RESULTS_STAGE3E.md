# iFairy vec_dot（非 LUT）Stage 3E：`sdot` accumulator banking 结果记录

本记录对应 `IFAIRY_VEC_DOT_FUSED6_XCTRACE_PLAN.md#L490` 之后的 Stage 4+ 优先级 #1（打断 `sdot` 依赖链）。

## 0) 对比范围

- **Baseline**：`490a4df6`（未做 `sdot` banking）
- **Opt**：工作区改动（`ggml/src/ggml-cpu/arch/arm/quants.c::ggml_vec_dot_ifairy_q16_K` 的 `sdot` bank 分流）

语义不变：严格 `w * conj(x)`；并假设激活始终为 tensor-scale（`x[i].d_real/d_imag` 对所有 block 相同），在 vec_dot 侧跨 block 先累加 int32，再做一次缩放合成。

## 1) 改动摘要（Opt）

- 将 `acc_*0/acc_*1` 从“low/high nibble 分流”改为 **按激活寄存器分流**：
  - 所有使用 `v3/v5` 的 `sdot` 写入 `acc_*0`
  - 所有使用 `v4/v6` 的 `sdot` 写入 `acc_*1`
- block 末尾仍然 `acc_*0 += acc_*1` 后再做水平求和，数值等价。

## 2) 正确性验证

Baseline：
- `tmp/ifairy-stage3e-baseline/test-ifairy.txt`
- `tmp/ifairy-stage3e-baseline/test-ifairy_strict.txt`
- microbench verify（max_abs_diff=0）：
  - `tmp/ifairy-stage3e-baseline/vecdot_verify_k1536.txt`
  - `tmp/ifairy-stage3e-baseline/vecdot_verify_k4096.txt`

Opt：
- `tmp/ifairy-stage3e-opt/test-ifairy.txt`
- `tmp/ifairy-stage3e-opt/test-ifairy_strict.txt`
- microbench verify（max_abs_diff=0）：
  - `tmp/ifairy-stage3e-opt/vecdot_verify_k1536.txt`
  - `tmp/ifairy-stage3e-opt/vecdot_verify_k4096.txt`

## 3) microbench（ns/iter）

命令（示例）：

```bash
./build-rel/bin/ifairy-vecdot-microbench --k 1536 --iters 50000000 --warmup 2000 --seed 1 --x-scale tensor --no-verify
./build-rel/bin/ifairy-vecdot-microbench --k 4096 --iters 20000000 --warmup 2000 --seed 1 --x-scale tensor --no-verify
```

结果：

- Baseline：
  - k=1536：`tmp/ifairy-stage3e-baseline/vecdot_microbench_k1536.txt`（`ns/iter=46.40`）
  - k=4096：`tmp/ifairy-stage3e-baseline/vecdot_microbench_k4096.txt`（`ns/iter=122.23`）
- Opt：
  - k=1536：`tmp/ifairy-stage3e-opt/vecdot_microbench_k1536.txt`（`ns/iter=46.75`）
  - k=4096：`tmp/ifairy-stage3e-opt/vecdot_microbench_k4096.txt`（`ns/iter=121.80`）

备注：k=1536/4096 的变化都在 ~1% 内，强 DVFS 噪声敏感，需结合 counters/端到端再判断。

## 4) xctrace：CPU Counters（cycles + bottleneck ratios）

命令（示例）：

```bash
xcrun xctrace record --template 'CPU Counters' --time-limit 20s \
  --output tmp/xctrace/vecdot.trace \
  --launch -- ./build-rel/bin/ifairy-vecdot-microbench --k 4096 --iters 50000000 --warmup 2000 --seed 1 --x-scale tensor --no-verify

xcrun xctrace export --input tmp/xctrace/vecdot.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="MetricAggregationForThread"]' \
  | python3 scripts/ifairy_xctrace_cpu_counters_summary.py
```

Baseline：
- k=1536：`tmp/ifairy-stage3e-baseline/vecdot_cpu_counters_k1536.metric.summary.txt`
  - cycles: `9573734103`
- k=4096：`tmp/ifairy-stage3e-baseline/vecdot_cpu_counters_k4096.metric.summary.txt`
  - cycles: `24562880266`
- core type 分布（ns）：`tmp/ifairy-stage3e-baseline/core_type_sums.txt`

Opt：
- k=1536：`tmp/ifairy-stage3e-opt/vecdot_cpu_counters_k1536.metric.summary.txt`
  - cycles: `9524211864`
- k=4096：`tmp/ifairy-stage3e-opt/vecdot_cpu_counters_k4096.metric.summary.txt`
  - cycles: `24584089612`
- core type 分布（ns）：`tmp/ifairy-stage3e-opt/core_type_sums.txt`

初步结论：
- cycles 口径下，banking 对 k=1536 有 ~0.5% 下降、对 k=4096 基本持平（略高），属于“接近噪声级”的收益/波动。
- 下一步更值得把精力放到 `tbl`/寄存器压力与更深的软件流水（见计划文档 Stage 4+ #2/#3）。

## 5) xctrace：CPU Profiler（leaf cycles）

Baseline：
- `tmp/ifairy-stage3e-baseline/vecdot_cpu_profiler_leaf_k4096.txt`
  - `ggml_vec_dot_ifairy_q16_K`: `95.70%`

Opt：
- `tmp/ifairy-stage3e-opt/vecdot_cpu_profiler_leaf_k4096.txt`
  - `ggml_vec_dot_ifairy_q16_K`: `96.59%`

## 6) 端到端：llama-bench（repetitions=3，默认 warmup）

命令：

```bash
GGML_IFAIRY_LUT=0 GGML_IFAIRY_VEC_DOT_ACT_TENSOR=1 \
  ./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf \
  --threads 4 --n-prompt 64 --n-gen 128 -ngl 0 --device none --repetitions 3
```

Baseline：`tmp/ifairy-stage3e-baseline/llama-bench.txt`
- pp64: `161.15 ± 0.15`
- tg128: `90.93 ± 2.28`

Opt：`tmp/ifairy-stage3e-opt/llama-bench.txt`
- pp64: `160.00 ± 0.93`
- tg128: `92.10 ± 0.59`

备注：端到端变化同样处于小幅波动区间（~1% 量级），需要更多轮次/更严格控频口径辅助判断。

