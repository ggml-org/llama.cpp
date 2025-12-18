# iFairy LUT 性能分析与优化规划评估

**日期**: 2025-12-18
**基准 commit**: `79c915e5`（已恢复到 `0ec52a5a` tok/s 档位）
**平台**: Apple M4, 4 threads

---

## 1. 当前性能热点分布

根据最新 Xcode Profile 采样数据：

| 函数 | 占比 | 说明 |
|------|------|------|
| `ggml_ifairy_lut_qgemm_ex` | **53.5%** | LUT 查表 + 累加核心循环 |
| `ggml_graph_compute_thread` | **30.5%** | 线程调度/同步/图执行框架 |
| `ggml_compute_forward_mul_mat` | 7.5% | mul_mat 路由与工作区管理 |
| 其他 | <3% | preprocess、量化等 |

**关键发现**：
- 主热点仍集中在 `qgemm_ex`（53.5%），但相比之前记录的 63% 已有所下降
- 框架开销 `ggml_graph_compute_thread` 占比 30.5%，明显偏高（说明同步/调度成本不可忽略）
- `ggml_compute_forward_mul_mat` 的 7.5% 主要是路由判断和工作区切分逻辑

---

## 2. 当前实现瓶颈分析

### 2.1 `ggml_ifairy_lut_qgemm_ex`（53.5%）

#### 2.1.1 Compact 布局热路径（N==1 decode 场景）

分析代码 `ggml-ifairy-lut.cpp:1416-1680`：

```c
// N==1 fast-path 的核心循环
for (int64_t blk = 0; blk < blocks; ++blk) {
    int32x4_t isum0 = vdupq_n_s32(0);
    int32x4_t isum1 = vdupq_n_s32(0);

    // 4-way unroll 的 group 循环
    for (; gi + 3 < groups_per_block; gi += 4) {
        // 每个 group 需要：
        // 1. 读取 4 个 pattern (4 loads)
        // 2. 解码 c0/c1/c2 (12 次位运算)
        // 3. 计算 4 组地址偏移 (12 次指针计算)
        // 4. 12 次 vld1_dup_s32 (position 查表)
        // 5. 12 次 vmovl_s8 (int8 → int16 扩展)
        // 6. 12 次 vaddq_s16 (position 求和)
        // 7. 4 次 vaddw_s16 (累加到 int32)
    }

    // per-block scale 乘法
    const float32x4_t scv = vcombine_f32(srsi, srsi);
    accv = vmlaq_f32(accv, sumsf, scv);
}
```

**瓶颈点**：

| 问题 | 影响 | 优先级 |
|------|------|--------|
| 每 group 需 3 次 position 查表 + 3 次 widen + 3 次 add | 指令数偏多，依赖链长 | P0 |
| `vld1_dup_s32` 虽只 load 4B，但需要先计算地址 `t0 + c0` | 地址计算开销 | P1 |
| 4-way unroll 使用了大量寄存器（12 个 `int32x2_t`） | 可能导致 register spill | P2 |
| `c02 = pat >> 4` 后面某些路径仍有冗余 `& 3` | 微小，但可消除 | P3 |

#### 2.1.2 Legacy 布局热路径

Legacy 布局的查表更直接（直接从 64-pattern 表读 4 个 int16），但：
- LUT 带宽大（512B/group vs compact 的 48B/group）
- 对 cache 压力更大

当前 A/B 测试结果：`legacy mean=19.45 tok/s` vs `compact mean=16.80 tok/s`（compact 仍未稳定胜出）

### 2.2 `ggml_graph_compute_thread`（30.5%）

**主要来源**：

1. **barrier 同步**（`ggml_barrier`）
   - `transform_tensor` 后需要 barrier 等待索引生成
   - `preprocess_ex` 后需要 barrier 等待 LUT 构建
   - BK tiling 时每个 K-tile 都有 `preprocess + barrier`

2. **线程调度开销**
   - decode 场景 N=1，但 4 线程都要参与
   - 线程分配策略在小 N 时可能不够高效

3. **工作区管理**
   - 每次 mul_mat 都要计算 `need = quant_bytes + shared_bytes + tmp_all`
   - `GGML_ASSERT(need == ggml_ifairy_lut_get_wsize(...))` 保证一致性但有计算开销

### 2.3 `ggml_compute_forward_mul_mat`（7.5%）

主要是：
- LUT 路由判断（`ggml_ifairy_lut_can_mul_mat`）
- env 变量解析（`getenv` + `strcmp`）
- 工作区大小计算

---

## 3. 与 API_PLAN.md 规划的映射评估

### 3.1 回归恢复状态（§6.0.1）

| 步骤 | 状态 | 说明 |
|------|------|------|
| R0: 恢复 `preprocess_ex` 构表热路径 | ✅ 完成 | `79c915e5` 已回退为 `memset + direct stores` |
| R1: 回退/对照 `qgemm_ex` unroll 与 prefetch | ⏸️ 建议冻结 | 当前已达标，不急于改动 |
| R2: 暂时禁用 N==1 fast-path | ⏸️ 建议冻结 | 当前已有 env 开关控制 |
| R3: 简化激活量化并行切分 | ⏸️ 建议冻结 | 已有按 K-block range 分片 |

**结论**：R0 已完成，tok/s 已恢复到 `legacy ~18-19` / `compact ~16-19` 范围。建议按 API_PLAN 的指导"冻结 R1/R2/R3"，稳定当前状态。

### 3.2 §6.1 继续压 `qgemm_ex` 热点（适用性评估）

| 任务 | 当前代码状态 | 收益预期 | 风险 | 建议 |
|------|--------------|----------|------|------|
| unroll + 多累加器 | 已有 4-way unroll + `isum0/isum1` 交错 | 低（已实现） | - | 维持现状 |
| 减少地址计算 | 当前用 `t0 + c0` 索引 | 中 | 低 | 可探索 |
| prefetch 策略 | 已有 `GGML_IFAIRY_LUT_PREFETCH` 开关 | 中 | 低 | 需 A/B 验证 |
| N==1 快路 | 已实现，有 `N1_FASTPATH` 开关 | 已收益 | - | 维持现状 |
| 减少 call/拷贝开销 | 非 tiling 时仍有 per-row qgemm | 中 | 中 | 可考虑 |

**分析**：
- 4-way unroll 已实现，2-way 实测更慢（见 STATUS.md 失败案例）
- `c2 = pat >> 4` 已优化（不再 `& 3`）
- prefetch 收益不明显（A/B 重叠大）

### 3.3 §6.2 降低框架开销（高优先级）

当前 `ggml_graph_compute_thread` 占 30.5%，说明框架开销占比偏高。

| 问题 | 现状 | 建议 |
|------|------|------|
| barrier 次数 | 每次 mul_mat 至少 2 次 | 探索减少 barrier |
| N==1 线程利用率 | 4 线程但只有 1 列 | 考虑单线程 fast-path |
| env 解析开销 | 每次调用都 getenv | 已优化为 cached |
| FULLACC 自动策略 | 小 N + acc_bytes ≤ 8MB 时自动启用 | 合理 |

**关键优化方向**：
1. **decode 场景（N=1）考虑减少线程数**：当 N < nth 时，多线程的同步开销可能超过并行收益
2. **探索 barrier 合并**：能否把 `transform + preprocess` 的 barrier 合并

### 3.4 §6.3 让 BK/BM tiling "不再变慢"

当前 tiling 路径有 `FULLACC` 模式减少 preprocess 次数，但：
- tiling 的收益不稳定（STATUS.md 记录波动大）
- decode 场景 N=1 时 tiling 几乎无意义

**建议**：保持现有 `BK_BLOCKS=0` 默认策略（非 tiling），tiling 作为可选参数。

---

## 4. 工程地基评估（§7）

### 4.1 已完成项

| 项目 | 状态 | commit |
|------|------|--------|
| size/overflow 断言 | ✅ | `2a39f249` |
| prefetch 可控 | ✅ | `627dea55`/`46dcb0cb` |
| env 解析收敛 | ✅ | `62a4ad8f` |
| 错误可观测性 | ✅ | `0f1af549` |
| 路由边界明确 | ✅ | `34d8df05`/`10c98502` |
| compact group bytes 常量化 | ✅ | `GGML_IFAIRY_LUT_COMPACT_GROUP_BYTES` |
| wsize 一致性断言 | ✅ | ggml-cpu.c:1399 |

### 4.2 待推进项（按优先级）

| 优先级 | 项目 | 现状 | 建议 |
|--------|------|------|------|
| P1 | 减少全局状态 | 仍有 `g_ifairy_lut_*` 全局容器 | 可接受，暂不改 |
| P1 | 线程安全 | `g_ifairy_lut_mutex` 保护 | 已足够 |
| P2 | 代码拆分 | `ggml-ifairy-lut.cpp` 约 2300 行 | 可考虑拆分 |
| P2 | 测试补齐 | `test-ifairy` 覆盖有限 | 补充 edge case |

---

## 5. 优化建议（按收益/风险排序）

### 5.1 短期（低风险，可立即尝试）

1. **探索 decode 场景单线程 fast-path**
   - 条件：`N == 1 && M * K * 2 < threshold`
   - 收益：消除 barrier + 线程调度开销
   - 风险：低（加 env 开关即可）

2. **减少 `ggml_compute_forward_mul_mat` 中的 env 解析**
   - 当前每次调用都有多次 `getenv`
   - 可考虑在 `ggml_graph_compute` 层面缓存一次

3. **A/B 验证 prefetch 策略**
   - 在更稳定的测试条件下（冷却后）重跑 prefetch A/B
   - 确定是否真正有收益

### 5.2 中期（中等风险，需要验证）

1. **compact 布局的进一步优化**
   - 探索把 3 次 position 查表合并为更少的操作
   - 可能方向：预计算 `c0 + c1*4 + c2*16` 作为单一索引

2. **减少 barrier 次数**
   - 探索 `transform_tensor` 是否可以在图构建阶段完成
   - 探索 `preprocess` 是否可以与 `qgemm` 流水化

3. **优化地址计算**
   - `t0[c0]` 的 `c0` 乘以 4 可以预计算
   - 或者用 `vqtbl` 代替分散 load（需要重新评估，之前实测更慢）

### 5.3 长期（高风险，需要重大改动）

1. **形状专用内核**
   - 为最热的 `(M, K, N)` 组合提供模板化内核
   - 风险：代码膨胀，维护成本高

2. **LUT 布局重设计**
   - 探索 BitNet 风格的 nibble 索引 + `vqtbl` 查表
   - 需要重新设计 preprocess 和 qgemm

---

## 6. 性能目标与验收标准

### 6.1 当前基线（2025-12-18）

| 配置 | tok/s (mean) | 来源 |
|------|--------------|------|
| legacy, 4t, 256tok | ~18-19 | STATUS.md |
| compact, 4t, 256tok | ~16-19 | STATUS.md |

### 6.2 目标

| 阶段 | 目标 | 指标 |
|------|------|------|
| 短期（稳定） | 保持当前档位，减少波动 | legacy ≥ 18, compact ≥ 17 (稳定) |
| 中期（提升） | 在 decode 场景提升 10% | legacy ≥ 20, compact ≥ 20 |
| 长期（上限） | 逼近纯带宽限制 | 需要 profile 确定理论上限 |

### 6.3 验收方法

1. **正确性**：`test-ifairy` + `GGML_IFAIRY_LUT_VALIDATE_STRICT=1`
2. **性能**：长测 3-run（`-n 256`），冷却间隔 ≥ 60s，记录 min/max/mean
3. **稳定性**：min/max 差距 ≤ 15%

---

## 7. 总结

1. **当前状态良好**：R0 回归恢复已完成，tok/s 已超过 `0ec52a5a` 档位
2. **主瓶颈明确**：`qgemm_ex` 53.5% + `ggml_graph_compute_thread` 30.5% 占大头
3. **框架开销值得关注**：30.5% 的同步/调度开销说明 decode 场景可能不适合 4 线程
4. **建议策略**：
   - 冻结 R1/R2/R3，稳定当前状态
   - 优先探索 decode 单线程 fast-path
   - compact 布局继续 A/B 优化，目标是稳定超过 legacy
5. **避免踩坑**：
   - 不要轻易改 4-way unroll
   - 不要在没有 A/B 的情况下改 prefetch 策略
   - 每次改动后必须跑 test-ifairy + tok/s 记录
