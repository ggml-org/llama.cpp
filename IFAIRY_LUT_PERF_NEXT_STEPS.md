# iFairy LUT 性能优化分析与下一步工作

> 基于 `IFAIRY_ARM_3W_LUT_STATUS.md` 的 0.1 tok/s 记录（截至 2025-12-26）与 `IFAIRY_ARM_3W_LUT_API_PLAN.md` 的 6. 性能提升规划，结合当前代码实现分析。

## 1. 当前性能现状（基于最新 bench 记录）

### 1.1 最新 tok/s 记录（2617cc59，Apple M4，4 threads）

| Kernel | pp128 (tok/s) | tg256 (tok/s) | 备注 |
|--------|---------------|---------------|------|
| **merged64** | 4.39 | **24.75** | 当前 decode 最优 |
| auto | 4.33 | 17.75 | 默认策略 |
| tbl | 4.38 | 7.04 | 明显落后 |

### 1.2 关键观察

1. **Decode 场景（tg256）**：`merged64` 相对 `auto` 提升 **+39%**（24.75 vs 17.75），收益显著
2. **Prefill 场景（pp128）**：所有 kernel 都在 ~4 tok/s 附近，明显是瓶颈
3. **Auto 策略问题**：当前 `auto` 在 `N==1` 时未自动选择 `merged64`，需要显式设置 `KERNEL=merged64`

## 2. 优先级排序（建议）

基于当前数据与代码分析：

| 优先级 | 任务 | 预期收益 | 风险 |
|--------|------|----------|------|
| **P0** | 让 auto 策略在 decode 场景默认选 merged64 | +39% decode tok/s | 低（配置变更） |
| **P1** | 复采样 profile（merged64），确认新热点 | 定向优化基础 | 无（调研） |
| **P2** | merged64 qgemm 热路径微优化 | +5~15% | 中等（需 A/B） |
| **P3** | prefill 场景：启用 merged64 的 BK tiling | 提升 prefill tok/s | 高（需验证 correctness） |
| **P4** | 减少框架开销（barrier/调度） | +5~10% | 中等 |

## 3. 具体优化点与实现方案

### 3.1 【P0】提升 auto 策略：decode 场景默认 merged64

**现状问题**：
- 代码位置：`ggml/src/ggml-cpu/ggml-cpu.c:1546-1577`
- 当前逻辑：`auto` + `N==1` 时，只有显式设置 `KERNEL=merged64` 才会走 merged64
- 实际表现：merged64 比 auto 快 39%，但用户必须手动设置 env

**建议改动**：

```c
// ggml/src/ggml-cpu/ggml-cpu.c 约 1546 行
if (cfg->lut_layout_auto) {
    layout = GGML_IFAIRY_LUT_LAYOUT_LEGACY;  // prefill 默认
    if (N == 1) {
        // P0: decode 场景默认优先 merged64
        if (cfg->lut_kernel == GGML_IFAIRY_LUT_KERNEL_AUTO) {
            layout = GGML_IFAIRY_LUT_LAYOUT_MERGED64;  // NEW: auto 时优先 merged64
        } else if (cfg->lut_kernel == GGML_IFAIRY_LUT_KERNEL_TBL) {
            layout = GGML_IFAIRY_LUT_LAYOUT_TBL64;
        } else if (cfg->lut_kernel == GGML_IFAIRY_LUT_KERNEL_MERGED64) {
            layout = GGML_IFAIRY_LUT_LAYOUT_MERGED64;
        } else if (cfg->lut_kernel == GGML_IFAIRY_LUT_KERNEL_SDOT) {
            layout = GGML_IFAIRY_LUT_LAYOUT_COMPACT;
        }
    }
}
```

**验收**：
- `GGML_IFAIRY_LUT=1`（无需显式设置 KERNEL）下，decode tok/s 达到 ~24 tok/s
- 必须保留 `GGML_IFAIRY_LUT_KERNEL=legacy/compact/...` 回退机制

**风险**：低。仅配置策略变更，不改热路径逻辑。

---

### 3.2 【P1】复采样 profile（merged64），确认新热点

**目的**：merged64 加速后，`qgemm_ex` 占比可能已下降，需确认下一瓶颈是否转移到 `preprocess_ex` 或 `graph_compute_thread`。

**建议命令**（Xcode Instruments / sample）：

```bash
GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_KERNEL=merged64 \
./build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf \
  --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 128 -no-cnv
```

**关注指标**：
- `ggml_ifairy_lut_qgemm_ex_merged64` 自耗时占比（目标 < 50%）
- `ggml_ifairy_lut_preprocess_ex_merged64` 是否上升为新热点
- `ggml_graph_compute_thread` 是否仍占 ~30%

**下一步**：根据 profile 结果决定继续压 qgemm 还是转向 preprocess/框架开销。

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
- 代码位置：`ggml/src/ggml-cpu/ggml-cpu.c:1582-1586`
- 当前逻辑：`tbl64/merged64` 强制 `tile_blocks = 0`，即禁用 BK tiling
- 影响：prefill（大 K）无法受益于 tiling 的 cache 友好性

```c
// 当前代码
if (layout == GGML_IFAIRY_LUT_LAYOUT_TBL64 || layout == GGML_IFAIRY_LUT_LAYOUT_MERGED64) {
    tile_blocks = 0;  // force no tiling for now
}
```

**建议分阶段落地**：

1. **Phase 1**：为 merged64 增加 tiled preprocess 支持（`ggml-ifairy-lut-preprocess.cpp`）
2. **Phase 2**：在 `ggml-cpu.c` 中有条件地启用 tiling（仅当 `N > 1` 或 `K > threshold`）
3. **Phase 3**：A/B 验证 prefill 场景 tok/s

**验收**：
- `pp128` tok/s 提升 ≥ 20%
- `tg256` tok/s 不回退
- strict 验证通过

**风险**：高。需要新增 tiled merged64 的 correctness 验证。

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
├─ [P0] auto 策略优化：decode 默认 merged64
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
├─ [P3] prefill tiling（如 profile 显示 preprocess 不是瓶颈）
│   ├─ merged64 tiled preprocess 实现
│   ├─ correctness 验证（strict）
│   └─ prefill A/B

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
