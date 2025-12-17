# iFairy 2-bit 复数模型 · ARM NEON 3‑Weight LUT 设计（现实现 + 路线图）

> 本文描述 ggml 中 iFairy 3‑weight LUT 路径的数学语义、索引/表结构、内核计算与关键工程约定。
>
> - API/路由与线程模型见 `IFAIRY_ARM_3W_LUT_API_PLAN.md`
> - 性能记录与复现命令见 `IFAIRY_ARM_3W_LUT_STATUS.md`

---

## 1. 目标与约束

### 1.1 目标

- 为 `GGML_TYPE_IFAIRY`（2‑bit 复数权重）提供 CPU-only 的 3‑weight LUT 加速路径（ARM64 NEON 优先、标量回退）。
- 与 ggml baseline（`ggml_vec_dot_ifairy_q16_K_generic`）在语义上严格一致：计算 `w * conj(x)`。
- 保持 GGUF/权重文件格式不变；加速路径可通过环境变量完全关闭并回退到旧实现。

### 1.2 约束（当前实现）

- 编译期开关：`GGML_IFAIRY_ARM_LUT`（开启时 CMake 强制关闭各类加速后端以保证 CPU-only）。
- 形状约束：`K % QK_K == 0`（当前 `QK_K=256`），否则 LUT 路由不生效。
- 激活输入支持两种形式：
  - `GGML_TYPE_F32`：复数 bf16-pair 容器（每个 complex 占 4B：`[bf16 real][bf16 imag]`），推理中常见；
  - `GGML_TYPE_IFAIRY_Q16`：`block_ifairy_q16`（int8 实/虚平面 + fp16 缩放），供直接走 LUT 内核使用。
- 输出张量为 `GGML_TYPE_F32`（同样按 bf16-pair 写回；内部通过 `pack_bf16` 控制写回方式）。

---

## 2. 数学语义（必须与 baseline 一致）

### 2.1 2‑bit 复数权重集合

每个 2‑bit code `c ∈ {0,1,2,3}` 表示一个复数单位权重 `w(c)`：

- `0 → −1`
- `1 → +1`
- `2 → −i`
- `3 → +i`

对应系数可写成 `(wr, wi) ∈ {(-1,0),(1,0),(0,-1),(0,1)}`。

### 2.2 baseline 点积语义

给定激活 `x = xr + i·xi`，baseline 点积使用的是 `conj(x) = xr − i·xi`：

```text
w * conj(x) = (wr + i·wi) * (xr − i·xi)
           = (wr·xr + wi·xi) + i·(wi·xr − wr·xi)
```

因此每个输出元素都可以通过 4 个“实数通道累加”表达：

```text
ac = Σ (wr * xr)
ad = Σ (wr * xi)
bc = Σ (wi * xr)
bd = Σ (wi * xi)

out_r = coeff_w_real * ac + coeff_w_imag * bd
out_i = coeff_w_imag * bc - coeff_w_real * ad
```

其中 `coeff_w_real/imag` 为权重侧的量化缩放（当前实现读取 `block_ifairy::d_real/d_imag`）。

这一拆分是当前 LUT 路径选择 **“4 通道累加（ac/ad/bc/bd）”** 的根本原因：它能直接对齐 `w * conj(x)`，避免“看起来像复数乘但语义其实不等价”的隐蔽错误。

---

## 3. 3‑weight 索引与分组（当前实现）

### 3.1 直接 6-bit pattern（每组三权重 1 byte）

对每组三个 2‑bit 权重 `(c0,c1,c2)`，用 6-bit pattern 编码：

```text
pat = c0 | (c1 << 2) | (c2 << 4)   // pat ∈ [0,63]
```

该编码由 `ggml/src/ggml-quants.c::ggml_ifairy_pack_triplet_direct()` 定义并被运行时内核直接消费。

### 3.2 按 `QK_K=256` block 内分组（不改变模型）

当前实现不做跨 block 的 K3 padding，而是在每个 256-block 内构造固定的 `86` 组：

- `intra=0..84`：覆盖 block 内 `0..254` 的 85 个 triplet（步长 3）。
- `intra=85`：尾组 `(255, pad, pad)`（缺失位置按 code=0 处理）。

整行的索引长度为：

```text
groups_per_row = (K / 256) * 86
```

索引缓冲布局为行主序：`rows × groups_per_row` bytes。

---

## 4. LUT 表结构（两种 layout）

### 4.1 `legacy`：`64 patterns × 4 channels × int16`（大表，路径更直接）

- 每个 `(col, group)` 构建一张表：`t[ch][pat]`，其中 `ch ∈ {ac,ad,bc,bd}`，`pat ∈ [0,63]`。
- 表项为 int16（预处理阶段已经把激活量化并应用了 per-block 的激活缩放因子，表项表示“整数域贡献”）。
- 内核循环中只做：`pat = indexes[g]`，对 4 个通道做一次查表并累加到浮点/整数累加器。

该布局的优点是“解释最直接、对照最容易”；缺点是工作区较大（每 group 每 col 为 `512B`）。

### 4.2 `compact`：`3 positions × 4 codes × 4 channels × int8`（小表，带宽更友好）

观察到每个 position 只依赖该位置的 code（0..3）与该位置的激活 `(xr,xi)`，且对 4 通道 `(ac,ad,bc,bd)` 的贡献是稀疏/可加的：

- real 类 code（`±1`）只影响 `(ac,ad)`；
- imag 类 code（`±i`）只影响 `(bc,bd)`；
- 三个 position 的贡献可直接相加得到 group 的总贡献。

因此可以把 per-group LUT 由 “64 pattern 大表” 改写成 “3 个 position 小表”：

```text
grp[pos][code][ch]  // pos ∈ {0,1,2}, code ∈ [0,3], ch ∈ [0,3]
```

总大小：`3 * 4 * 4 = 48 bytes / group / col`。

NEON 内核里对每个 group 只需要：

1) 从 `indexes[g]` 拆出 `c0,c1,c2`（按 2-bit 取出）；
2) 分别读取 `pos0/pos1/pos2` 的 16B 表，按 `c0/c1/c2` 索引得到 4×int8，再 widen/add；
3) 继续累加到通道累加器。

### 4.3 缩放数组（per-block）

激活量化为 `block_ifairy_q16` 时，每个 256-block 有 `d_real/d_imag`。当前实现把它展开到 `lut_scales` 中：

- `lut_scales[(col, block)] = (scale_real, scale_imag)`（2 floats）
- LUT 表项表达的是“在该缩放体系下的整数贡献”，在写回时再乘权重缩放合成最终输出。

---

## 5. 调度与线程模型（摘要）

集成点在 `ggml/src/ggml-cpu/ggml-cpu.c::ggml_compute_forward_mul_mat`（`#if defined(GGML_IFAIRY_ARM_LUT)`）。

- 索引生成：thread 0 负责 `ggml_ifairy_lut_transform_tensor()`，随后 barrier，同一 op 的所有线程共享索引。
- 非 tiling：
  - `preprocess_ex()` 并行构表（跨列/跨 group 切分），随后 barrier；
  - `qgemm_ex()` 按行分片并行写回。
- BK tiling（按 256-block tile）：
  - 每个 K-tile 重复一次 `preprocess_ex()` + barrier；
  - 可选 `FULLACC` 模式用共享 accumulator，减少按 BM 行块重复构表/同步。
- 严格对照：`GGML_IFAIRY_LUT_VALIDATE_STRICT=1` 进入 strict 模式，用于验证正确性；当前 strict 会禁用 tiling（避免假设不一致）。

---

## 6. 运行时开关（当前实现）

- `GGML_IFAIRY_LUT=0/1`：禁用/启用 LUT（默认启用）。
- `GGML_IFAIRY_LUT_LAYOUT=legacy|compact`：选择 LUT 布局（默认 `legacy`）。
- `GGML_IFAIRY_LUT_BK_BLOCKS=<int>`：K 维按 256-block 做 tiling（`0` 禁用；strict 下强制禁用）。
- `GGML_IFAIRY_LUT_BM=<int>`：BM 行块大小（仅 tiling 生效）。
- `GGML_IFAIRY_LUT_FULLACC=0/1`：tiling 下共享 accumulator（未设置时可能按 `(N,acc_bytes)` 自动启用）。
- `GGML_IFAIRY_LUT_VALIDATE_STRICT=0/1`：严格对照（验证用）。
- `GGML_IFAIRY_LUT_DEBUG=0/1`：路由诊断（默认关闭）。

---

## 7. 当前主任务（与代码审查结论对齐）

P0（必须尽快做）：

- 内存/生命周期：减少 `new/delete` + 全局容器；补齐 size/overflow/bounds 检查，避免 silent failure。
- 线程安全：明确并发模型；缩小锁粒度，避免持锁做重活；补充并发/压力测试。

P1（近期做）：

- 可维护性重构：拆分 `ggml/src/ggml-ifairy-lut.cpp`，减少 legacy/compact 重复代码。
- 错误处理一致性：统一 `return false`/`GGML_ASSERT`/日志策略。
- 路由健壮性：补齐更明确的 CPU feature 判定（NEON/dotprod）与可控回退。

P2（持续）：

- 测试补齐：对齐/极小维度/大维度、分配失败、misaligned buffer、并发 transform 等 edge case。
- 性能回归：把 decode（`N≈1`）与 prefill 形状的 tok/s 作为可复现基线（见 `IFAIRY_ARM_3W_LUT_STATUS.md`）。

---

## 附录 A：可选优化——`16 patterns + factor` 的 canonical 分解（未用于当前运行时）

> 该分解对应 BitNet TL1 风格的 “small LUT + 变换因子” 设计，用于探索 `vqtbl` 查表形态；当前 iFairy 运行时实现采用的是“直接 64 pattern（legacy）/ 3pos×4code（compact）”，并不依赖本附录。

对任意三元组 `(w0,w1,w2)`（每个 `wi ∈ {±1, ±i}`），总可以写成：

```text
(w0, w1, w2) = factor · (1, u1, u2)
```

一种自然选择是：

- `factor = w0`
- `u1 = w1 / w0`
- `u2 = w2 / w0`

于是：

- `factor ∈ {1, i, -1, -i}`（4 种）
- `(u1,u2)` 只有 `4^2 = 16` 种 canonical pattern

因此可以预先枚举得到：

- `canonical_idx[64]`：`pat -> idx'`（`idx' ∈ [0,15]`）
- `factor_exp[64]`：`pat -> e`（`i^e = factor`，`e ∈ {0,1,2,3}`）

离线枚举脚本：

- `scripts/ifairy_3w_lut_enum.py`
- 该脚本应与运行时一致地使用 `pat = c0 | (c1 << 2) | (c2 << 4)` 的 bit 顺序。

