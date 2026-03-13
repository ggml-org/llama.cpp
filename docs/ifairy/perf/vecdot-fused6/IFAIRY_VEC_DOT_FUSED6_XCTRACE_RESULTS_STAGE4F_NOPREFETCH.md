# iFairy vec_dot（非 LUT）Stage 4F：移除外层 `__builtin_prefetch` 实验记录

本记录对应 `IFAIRY_VEC_DOT_FUSED6_XCTRACE_PLAN.md#L490` 的 Stage 4+ 优先级 #3（“删除现有 `__builtin_prefetch`”对照实验）。

## 0) 对比范围

- Baseline：`490a4df6`（无本实验改动）
- 参考 Opt（上一轮）：`tmp/ifairy-stage3e-opt/`（包含 `sdot` accumulator banking）
- 本轮（Stage 4F）：在上一轮基础上 **移除** `ggml_vec_dot_ifairy_q16_K()` 外层 block loop 的两条 `__builtin_prefetch(w + i + 1)` / `__builtin_prefetch(x + i + 1)`。

## 1) 正确性

- `tmp/ifairy-stage4f-noprefetch/test-ifairy.txt`
- `tmp/ifairy-stage4f-noprefetch/test-ifairy_strict.txt`
- microbench verify（max_abs_diff=0）：
  - `tmp/ifairy-stage4f-noprefetch/vecdot_verify_k1536.txt`
  - `tmp/ifairy-stage4f-noprefetch/vecdot_verify_k4096.txt`

## 2) microbench（ns/iter）

```bash
./build-rel/bin/ifairy-vecdot-microbench --k 1536 --iters 50000000 --warmup 2000 --seed 1 --x-scale tensor --no-verify
./build-rel/bin/ifairy-vecdot-microbench --k 4096 --iters 50000000 --warmup 2000 --seed 1 --x-scale tensor --no-verify
```

- Stage 4F（no-prefetch）：
  - k=1536：`tmp/ifairy-stage4f-noprefetch/vecdot_microbench_k1536.txt`（`ns/iter=47.28`）
  - k=4096：`tmp/ifairy-stage4f-noprefetch/vecdot_microbench_k4096.txt`（`ns/iter=122.29`）
- 参考 Opt（banking）：
  - k=1536：`tmp/ifairy-stage3e-opt/vecdot_microbench_k1536.txt`（`ns/iter=46.75`）
  - k=4096：`tmp/ifairy-stage3e-opt/vecdot_microbench_k4096.txt`（`ns/iter=121.80`）

## 3) xctrace：CPU Counters（cycles）

Stage 4F（no-prefetch）：
- k=1536：`tmp/ifairy-stage4f-noprefetch/vecdot_cpu_counters_k1536.metric.summary.txt`
  - cycles: `9753676342`
- k=4096：`tmp/ifairy-stage4f-noprefetch/vecdot_cpu_counters_k4096.metric.summary.txt`
  - cycles: `24565195247`
- core type 分布（ns）：`tmp/ifairy-stage4f-noprefetch/core_type_sums.txt`

参考 Opt（banking）：
- k=1536：`tmp/ifairy-stage3e-opt/vecdot_cpu_counters_k1536.metric.summary.txt`
  - cycles: `9524211864`
- k=4096：`tmp/ifairy-stage3e-opt/vecdot_cpu_counters_k4096.metric.summary.txt`
  - cycles: `24584089612`

## 4) xctrace：CPU Profiler（leaf cycles）

- Stage 4F：`tmp/ifairy-stage4f-noprefetch/vecdot_cpu_profiler_leaf_k4096.txt`

## 5) 端到端：llama-bench（repetitions=3）

```bash
GGML_IFAIRY_LUT=0 GGML_IFAIRY_VEC_DOT_ACT_TENSOR=1 \
  ./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf \
  --threads 4 --n-prompt 64 --n-gen 128 -ngl 0 --device none --repetitions 3
```

- Stage 4F run1：`tmp/ifairy-stage4f-noprefetch/llama-bench.txt`
- Stage 4F run2：`tmp/ifairy-stage4f-noprefetch/llama-bench_run2.txt`

## 6) 初步结论

- microbench（k=1536/4096）与 cycles 口径均未显示明确收益，且 k=1536 有明显回退倾向。
- 本实验更像是“无收益/有风险”的尝试：后续建议回滚此改动，或将 prefetch 做成更精确的预取粒度/距离实验（先用 counters 证明，再决定保留与否）。

