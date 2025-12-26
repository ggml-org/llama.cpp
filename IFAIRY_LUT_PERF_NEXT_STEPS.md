# iFairy LUT 性能优化分析与下一步工作

> 基于 `IFAIRY_ARM_3W_LUT_STATUS.md` 的 0.1 tok/s 记录（截至 2025-12-26）与 `IFAIRY_ARM_3W_LUT_API_PLAN.md` 的 6. 性能提升规划，结合当前代码实现分析。

## 1. 当前性能现状（基于最新 bench 记录）

### 1.1 最新 tok/s 记录（95c5bf1f+dirty，Apple M4，4 threads）

| Layout/KERNEL | pp128 (tok/s) | tg256 (tok/s) | 备注 |
|--------------|---------------|---------------|------|
| `auto`（当前默认偏向 `merged64`） | 29.47 | 24.36 | `/tmp/ifairy_bench_20251226T174919Z_auto_default.jsonl` |

### 1.2 关键观察

1. **auto 默认策略已修正**：在 `GGML_IFAIRY_LUT_LAYOUT=auto` 且 `KERNEL=auto` 时，默认优先走 `merged64`（覆盖 prefill+decode）。
2. **Prefill 瓶颈已解除**：在当前模型/机器上，prefill 不再停留在 ~4 tok/s（旧瓶颈来自 `auto` 仍走 `legacy`）。
3. **BK tiling 对该模型可能是负收益**：`K/QK_K` blocks 较少时（本模型 `K=1536` => 6 blocks），`BK_BLOCKS=1` 会引入多次 `preprocess+barrier`，需要以 A/B 结果为准，不应默认开启。

## 2. 优先级排序（建议）

基于当前数据与代码分析：

| 优先级 | 任务 | 预期收益 | 风险 |
|--------|------|----------|------|
| **P0** | ✅ auto 默认偏向 merged64（prefill+decode） | 已显著提升 pp/tg | 低（策略变更） |
| **P1** | 复采样 profile（merged64），确认新热点 | 定向优化基础 | 无（调研） |
| **P2** | merged64 qgemm 热路径微优化 | +5~15% | 中等（需 A/B） |
| **P3** | prefill 场景：评估 BK tiling 的收益窗口 | 仅在大 K/多 blocks 可能收益 | 中（需验证 correctness + A/B） |
| **P4** | 减少框架开销（barrier/调度） | +5~10% | 中等 |

## 3. 具体优化点与实现方案

### 3.1 【P0】提升 auto 策略：默认 merged64（prefill+decode）

**现状问题**：
- 代码位置：`ggml/src/ggml-cpu/ggml-cpu.c:1546-1577`
- 当前逻辑（已修正）：`auto` + `KERNEL=auto` 默认优先 `merged64`（不再要求用户显式设置 env）

**已落地改动要点**：

```c
// ggml/src/ggml-cpu/ggml-cpu.c 约 1546 行
if (cfg->lut_layout_auto) {
    layout = GGML_IFAIRY_LUT_LAYOUT_LEGACY;

    // NEW: auto/merged64 -> 默认优先 merged64（prefill+decode）
    if (cfg->lut_kernel == GGML_IFAIRY_LUT_KERNEL_AUTO ||
        cfg->lut_kernel == GGML_IFAIRY_LUT_KERNEL_MERGED64) {
        layout = GGML_IFAIRY_LUT_LAYOUT_MERGED64;
    } else if (N == 1) {
        // decode-only overrides
        if (cfg->lut_kernel == GGML_IFAIRY_LUT_KERNEL_TBL) {
            layout = GGML_IFAIRY_LUT_LAYOUT_TBL64;
        } else if (cfg->lut_kernel == GGML_IFAIRY_LUT_KERNEL_SDOT) {
            layout = GGML_IFAIRY_LUT_LAYOUT_COMPACT;
        }
    }
}
```

**验收**：
- `GGML_IFAIRY_LUT=1`（无需显式设置 `KERNEL`/`LAYOUT`）下，pp/tg tok/s 与 merged64 基线对齐（见 1.1）。
- 回退：`GGML_IFAIRY_LUT_LAYOUT=legacy|compact` 可一键回退；`KERNEL=tbl|sdot` 仍作为 decode-only A/B 选项。

**风险**：低。仅配置策略变更，不改热路径逻辑。

---

### 3.2 【P1】复采样 profile（merged64），确认新热点

**目的**：以当前默认策略（`auto → merged64`）复采样，确认默认路径的 top1/top2 热点，避免“优化的不是默认路径”。

**建议命令**（`xctrace` Time Profiler）：

```bash
# decode-only（N==1）
xcrun xctrace record --template 'Time Profiler' --output /tmp/xctrace_ifairy_decode.trace \
  --env GGML_IFAIRY_LUT=1 \
  --env GGML_IFAIRY_LUT_BK_BLOCKS=0 \
  --env GGML_IFAIRY_LUT_BM=0 \
  --env GGML_IFAIRY_LUT_FULLACC=0 \
  --launch -- ./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf \
    -p 0 -n 256 -b 2048 -ub 512 -t 4 -ngl 0 -dev none -r 1 --no-warmup -o jsonl

# prefill-only（N>1）
xcrun xctrace record --template 'Time Profiler' --output /tmp/xctrace_ifairy_prefill.trace \
  --env GGML_IFAIRY_LUT=1 \
  --env GGML_IFAIRY_LUT_BK_BLOCKS=0 \
  --env GGML_IFAIRY_LUT_BM=0 \
  --env GGML_IFAIRY_LUT_FULLACC=0 \
  --launch -- ./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf \
    -p 128 -n 0 -b 2048 -ub 512 -t 4 -ngl 0 -dev none -r 1 --no-warmup -o jsonl
```

**关注指标**：
- `ggml_ifairy_lut_qgemm_ex_merged64` 自耗时占比（top1 是否仍占绝对大头）
- `ggml_ifairy_3w_encode` / `ggml_ifairy_lut_preprocess_ex_merged64` 是否抬头成为 top2/top3
- `ggml_graph_compute_thread`（leaf）是否异常偏高（提示调度/同步/碎 kernel 问题）

**采样结果（2025-12-26，本机，4 threads，leaf CPU time share）**：
- decode-only（`-p 0 -n 256`，`avg_ts≈25.12 tok/s`）：`ggml_ifairy_lut_qgemm_ex_merged64` ~73%，`ggml_graph_compute_thread` ~13%，`ggml_vec_dot_f16` ~8%，`ggml_ifairy_3w_encode` ~2%，`ggml_ifairy_lut_preprocess_ex_merged64` ~1%。
- prefill-only（`-p 128 -n 0`，`avg_ts≈30.21 tok/s`）：`ggml_ifairy_lut_qgemm_ex_merged64` ~82%，`ggml_ifairy_3w_encode` ~6%，`ggml_graph_compute_thread` ~5%，`ggml_graph_compute_secondary_thread` ~3%，`ggml_ifairy_lut_preprocess_ex_merged64` ~2%。

**下一步**：现阶段主矛盾仍然是 `qgemm_ex_merged64`，优先推进 3.3（P2）的 qgemm 热路径微优化；`preprocess_ex` 暂不是瓶颈，不要把收益“搬家”到 preprocess。

---

### 3.3 【P2】merged64 qgemm 热路径微优化

**代码位置**：`ggml/src/ggml-ifairy-lut-qgemm.cpp:2037-2095`（NEON 路径）

#### 3.3.1 Prefetch 距离调优

**现状**：
- 默认 `GGML_IFAIRY_LUT_PREFETCH_DIST=2`（2 groups = 512B ahead）
- Apple M4 L1 cache line = 64B，prefetch 队列较深

**建议**：A/B 测试 `PREFETCH_DIST=4` 或 `PREFETCH_DIST=8`

```bash
# A/B 短测
GGML_IFAIRY_LUT_PREFETCH_DIST=2 ./build-rel/bin/llama-bench ...  # baseline
GGML_IFAIRY_LUT_PREFETCH_DIST=4 ./build-rel/bin/llama-bench ...  # candidate
GGML_IFAIRY_LUT_PREFETCH_DIST=8 ./build-rel/bin/llama-bench ...  # candidate
```

#### 3.3.2 增加 indexes 数组的 prefetch

**现状**：只对 group table 做 prefetch，未对 `idx_blk[]` 做 prefetch

**建议改动**（`ggml-ifairy-lut-qgemm.cpp` 约 2055 行）：

```c
// 在 prefetch group table 的同时，也 prefetch indexes
#if defined(__aarch64__) && defined(__ARM_NEON)
if (do_prefetch && (gi + pf_dist + 4) < groups_per_block) {
    __builtin_prefetch(grp + (gi + pf_dist) * grp_stride, 0, 3);
    __builtin_prefetch(idx_blk + gi + pf_dist + 4, 0, 3);  // NEW: prefetch indexes
}
#endif
```

#### 3.3.3 Unroll 8 groups（实验性）

**现状**：merged64 NEON 路径每次处理 4 groups

**建议**：在寄存器压力允许的情况下尝试 8-group unroll

```c
// 实验：8-group unroll（需要验证寄存器压力）
for (; gi + 8 <= groups_per_block; gi += 8) {
    // 4 个 isum + 4 个 isum2 = 8 个累加器
    // ...
}
```

**风险**：中等。需要 A/B 验证，可能因寄存器溢出反而变慢。

---

### 3.4 【P3】Prefill 场景：启用 merged64 的 BK tiling

**现状问题**：
- 代码位置：`ggml/src/ggml-cpu/ggml-cpu.c`（tile_blocks 决策处）
- 当前逻辑（已更新）：`tbl64` 仍禁用 tiling；`merged64` 仅在 decode（`N==1`）禁用 tiling，prefill（`N>1`）允许启用 `BK_BLOCKS` 做 A/B。
- 注意：对 blocks 较少的模型（例如 `K=1536 => 6 blocks`），`BK_BLOCKS=1` 会把一次 preprocess 拆成多次 `preprocess+barrier`，很可能是负收益；不应默认开启，必须以 A/B 记录为准。

```c
// 当前代码（要点）
if (layout == GGML_IFAIRY_LUT_LAYOUT_TBL64 || (layout == GGML_IFAIRY_LUT_LAYOUT_MERGED64 && N == 1)) {
    tile_blocks = 0;
}
```

**落地进度**：

- ✅ 已允许 merged64 在 `N>1` 场景启用 tiling（仍保留 decode `N==1` 禁用）
- ✅ 单测补齐：新增 merged64 tiling regression（`tests/test-ifairy.cpp`）

**验收**：
- 以 `llama-bench` 的 prefill 主导口径为准（例如更大 `--n-prompt`、更小 `--n-gen`），并记录到 `STATUS.md:0.1`；
- decode（`tg256`）不回退，strict 必须通过。

**风险**：中。tiling 更容易引入同步/构表开销与形状相关的性能回退，默认策略应保持保守（BK 默认 0）。

---

### 3.5 【P4】减少框架开销（barrier/调度）

**现状**（基于历史 profile）：
- `ggml_graph_compute_thread` 约占 30%
- decode 场景线程同步开销明显

**已完成项**（见 API_PLAN.md 6.1）：
- ✅ indexes 预热移到 `ggml_graph_compute()` 前
- ✅ env 解析/分发只做一次（缓存到 `threadpool->ifairy_lut_cfg`）
- ✅ decode 线程数策略开关（`DECODE_NTH`/`DECODE_THRESHOLD`）

**待推进项**：

#### 3.5.1 减少 tiling 路径的 barrier 次数

**代码位置**：`ggml/src/ggml-cpu/ggml-cpu.c:1837-1976`

**现状**：每个 K-tile 都有一次 `preprocess + barrier + qgemm + barrier`

**建议**：评估 double-buffering 在非 FULLACC 模式下的可行性

#### 3.5.2 decode 场景线程数自动 clamp

**代码位置**：`ggml/src/ggml-cpu/ggml-cpu.c:700-704`（`DECODE_NTH`/`DECODE_THRESHOLD`）

**现状**：这两个 env 开关默认禁用（`0`）

**建议**：做 A/B 测试，确定最佳默认值

```bash
# A/B 测试线程数 clamp
GGML_IFAIRY_LUT_DECODE_NTH=2 ./build-rel/bin/llama-bench ... --threads 4
GGML_IFAIRY_LUT_DECODE_NTH=1 ./build-rel/bin/llama-bench ... --threads 4
```

---

## 4. 实施路线图（建议）

```
Week 1:
├─ [P0] auto 策略优化：默认 merged64（prefill+decode）
│   ├─ 代码改动（ggml-cpu.c 约 5 行）
│   ├─ A/B 验证
│   └─ 更新 STATUS.md tok/s 记录
│
├─ [P1] 复采样 profile（merged64）
│   ├─ 收集 3 次采样数据
│   └─ 确认新热点分布

Week 2:
├─ [P2] merged64 qgemm 微优化
│   ├─ prefetch 距离 A/B
│   ├─ indexes prefetch 实验
│   └─ 更新 STATUS.md

Week 3-4:
├─ [P3] prefill tiling（仅在大 K/多 blocks 时评估）
│   ├─ correctness 验证（strict）
│   └─ prefill A/B（确认不回退后再考虑默认策略）

├─ [P4] 框架开销优化（如 profile 显示 graph_compute_thread 占比仍高）
│   ├─ DECODE_NTH 最佳值确定
│   └─ barrier 减少评估
```

## 5. 快速验证命令

### 5.1 当前 baseline 记录

```bash
# decode baseline (auto)
GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 \
./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf \
  --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 1 --no-warmup

# decode with merged64 (目标：验证 P0 后此命令可省略 KERNEL 设置)
GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_KERNEL=merged64 \
./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf \
  --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 1 --no-warmup
```

### 5.2 A/B 短测模板

```bash
# 短测（减少热漂移）
GGML_IFAIRY_LUT=1 [ENV_VAR_A] ./build-rel/bin/llama-bench \
  -m models/Fairy-plus-minus-i-700M/ifairy.gguf \
  --threads 4 --n-prompt 8 --n-gen 8 -ngl 0 --device none --repetitions 1 --no-warmup

# ABABAB 交替跑 6 次，取 mean/min/max
```

### 5.3 Correctness 验证

```bash
# 必须通过
./build-rel/bin/test-ifairy

# strict 对照
GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_VALIDATE_STRICT=1 ./build-rel/bin/test-ifairy
```

## 6. 风险与回退

| 优化项 | 回退方式 |
|--------|----------|
| P0 auto 策略 | `GGML_IFAIRY_LUT_LAYOUT=legacy` 强制回退 |
| P2 prefetch 调优 | `GGML_IFAIRY_LUT_PREFETCH_DIST=2`（默认值） |
| P3 tiling 启用 | `GGML_IFAIRY_LUT_BK_BLOCKS=0`（禁用 tiling） |
| P4 线程 clamp | `GGML_IFAIRY_LUT_DECODE_NTH=0`（禁用 clamp） |

## 7. 相关代码文件索引

| 文件 | 内容 |
|------|------|
| `ggml/src/ggml-cpu/ggml-cpu.c:1546-1600` | auto 策略 / 路由逻辑 |
| `ggml/src/ggml-ifairy-lut-qgemm.cpp:2001-2205` | merged64 qgemm 实现 |
| `ggml/src/ggml-ifairy-lut-preprocess.cpp:476-681` | merged64 preprocess 实现 |
| `ggml/src/ggml-ifairy-lut.h` | API 与常量定义 |

---

> 本文档生成于 2025-12-26，基于 commit `2617cc59`（当前 HEAD: `fea00e76`）的分析。后续更新请同步维护 `IFAIRY_ARM_3W_LUT_STATUS.md` 的 tok/s 记录。
