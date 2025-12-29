# iFairy：vec_dot（非 LUT）按 Tensor 维度量化激活 + 6×QK_K(1536) 融合计算 + xctrace 采样方案

目标读者：准备在 Apple Silicon（AArch64 + NEON + DOTPROD）上优化 `iFairy` 的 **vec_dot 路径**（`GGML_IFAIRY_LUT=0` 或未命中 LUT 路由）的人。

本文是“可落地的工程设计方案”，重点覆盖：

1) **激活（activation）按 tensor 维度量化**（让 `block_ifairy_q16::d_real/d_imag` 在同一列的所有 256-block 上相同），以便把缩放系数外提并做跨 block 融合。
2) **在 `ggml_vec_dot_ifairy_q16_K()` 中把 `nb==6` 的 6 个 block 做“计算融合”**（减少每-block 标量工作与水平归约次数；尽量把累加留在 int32/NEON 侧，末尾一次缩放）。
3) **用 xctrace 做 A/B 采样**（Time Profiler + 可选 CPU Counters），给出可复现的命令、导出与分析口径。

---

## 0. 背景与当前实现快照（用于对照）

### 0.1 数据布局契约（QK_K=256）

源：`ggml/src/ggml-common.h`

- `block_ifairy`（权重，2-bit 相位码）：
  - `uint8_t qs[QK_K/4]`（256 个 2-bit code）
  - `ggml_half d_real, d_imag`
- `block_ifairy_q16`（激活，int8 量化的复数）：
  - `uint8_t x_real[QK_K], x_imag[QK_K]`（实际使用处按 `int8_t *` 读）
  - `ggml_half d_real, d_imag`

重要：**结构体 layout 固定，不能改 QK_K**（否则牵连面极大）。

### 0.2 当前 vec_dot（ARM DOTPROD）内核

主要函数：`ggml/src/ggml-cpu/arch/arm/quants.c::ggml_vec_dot_ifairy_q16_K()`

当前逻辑（简化）：

- 外层：`for i in [0..nb)`，每个 `i` 对应一个 `QK_K=256` block
- 内层：`for j in {0,128}`（每次 128 个元素），用 inline asm：
  - 解码权重 2-bit code → `wr/wi ∈ {-1,0,1}`（通过 `tbl` 查表）
  - `sdot` 执行 8-bit dot accumulate（`xr/xi` 与 `wr/wi`）得到 4 个 int32 累加（`ac/ad/bc/bd`）
- 每个 block 结束：
  - `vaddvq_s32` 做 4 次水平归约 → `sum_ac/sum_ad/sum_bc/sum_bd`
  - 读取 `x[i].d_real/d_imag`（每 block 的 scale）
  - `acc += x_scale[i] * sum_*`（float）
- 函数末尾：`coeff_w_* = w[0].d_*` 外提后做最终合成并写出 bf16 pair

**现状问题（针对 nb==6, n==1536）**：

- 即便 `x` 的 scale 逻辑上可以是 tensor 级别，当前实现依然每 block：
  - load/convert `x[i].d_*`
  - 4 次 `(float)sum_*` 转换
  - 4 次 mul + 4 次 add
  - 4 次 `vaddvq_s32` 水平归约

在 DOTPROD 内核已经很“紧”的情况下，这些“碎片开销”可能开始可见。

---

## 1) 需求明确化（Spec）

### 1.1 需求

- 当走 **vec_dot 路径（非 LUT）**时，激活量化采用 **tensor 维度的共享 scale**：
  - 对同一列（K 维长度为 `n`）的所有 block：`x[i].d_real == x[0].d_real` 且 `x[i].d_imag == x[0].d_imag`
- 在 `n==1536`（即 `nb == 6`）的常见形状上，`ggml_vec_dot_ifairy_q16_K()` 做 **“6 block 融合计算”**：
  - 尽可能把跨 block 的累加留在 int32/NEON 侧，减少水平归约次数与标量 FP 计算次数
  - 目标是减少每次 vec_dot 的额外开销（通常为个位数 % 级别的收益）

### 1.2 不变式（必须保持）

- 语义：必须严格是 `w * conj(x)`（不是 `w * x`）。
  - `(w_r + i w_i) * (x_r - i x_i) = (w_r x_r + w_i x_i) + i (w_i x_r - w_r x_i)`
- 数据布局：仍按 `QK_K=256` 的 block 分割存储。
- LUT 路径不受影响：`GGML_IFAIRY_LUT=1` 命中 LUT 时，不应被本方案强制改变行为。

### 1.3 验收标准（Acceptance Criteria）

- 构建：`build-rel` Release 成功。
- 正确性：
  - `./build-rel/bin/test-ifairy` 通过。
  - 对 vec_dot 路径新增/补充：在 `GGML_IFAIRY_LUT=0` 下的等价性测试（建议新增一个小测试或在现有测试中加分支）。
- 性能（以可复现方式记录）：
  - microbench：同输入/同迭代次数下，融合版 `ns/iter` 降低，并且 xctrace leaf 指向 `ggml_vec_dot_ifairy_q16_K`（或其子符号）为主要热点。
  - 端到端：`llama-bench` 在 `GGML_IFAIRY_LUT=0` 下的 `eval tok/s` 不回退（或给出原因与局限）。

---

## 2) 设计总览（Plan）

整体改动建议拆为两块（可以分 PR/分阶段落地）：

1) **激活按 tensor 维度量化（vec_dot-only）**
   - 新增（或条件启用）一条量化路径：同一列的 `block_ifairy_q16` 共享 `d_real/d_imag`
   - 让 vec_dot 能可靠触发“scale 外提 + 融合” fastpath
2) **`ggml_vec_dot_ifairy_q16_K()` 针对 `nb==6` 的融合 fastpath**
   - 在满足 scale 不变式时：跨 6 个 block 合并累加与归约

这两块相互独立：即便先只做第 2 块，也可以通过“运行时检查 scale 是否相同”来决定是否启用融合；第 1 块负责让该条件更频繁成立。

---

## 3) 激活按 Tensor 维度量化（vec_dot-only）

### 3.1 为什么不能直接改 `quantize_row_ifairy_q16()` 的语义

当前 `quantize_row_ifairy_q16()`（ARM 版本在 `ggml/src/ggml-cpu/arch/arm/quants.c`，ref 在 `ggml/src/ggml-quants.c`）是 **per-block scale**：

- 逐 block 扫描 `QK_K` 个复数，得到该 block 的 `max_real/max_imag`
- 计算 `iscale=127/max` 并量化

而 LUT 路径（以及部分并行切分）存在对激活的 **K 分段量化**（例如 `k_part`）调用：

- 如果把 `quantize_row_ifairy_q16()` 改成“必须看到全 K 才能算 global max”，那么对 `k_part` 的调用会得到错误 scale（会破坏 LUT 预处理/或者未来其它并行策略）。

因此建议：

- **保留现有 `quantize_row_ifairy_q16()` 的 per-block 语义**（兼容所有调用方）
- 新增一个 **显式的 tensor-scale 量化 API**，只在 vec_dot（非 LUT）路径中调用

### 3.2 建议新增 API（最小侵入）

新增两个函数（名字仅建议）：

- `quantize_row_ifairy_q16_tensor_ref(const float * x, block_ifairy_q16 * y, int64_t k)`
  - 放在 `ggml/src/ggml-quants.c`（与 `quantize_row_ifairy_q16_ref` 同域）
- `quantize_row_ifairy_q16_tensor(const float * x, void * vy, int64_t k)`
  - 放在 `ggml/src/ggml-cpu/arch/arm/quants.c`（用 NEON/asm 做加速；不支持时 fallback 到 `_ref`）

接口特征：

- 输入 `x`：复数以 bf16-pair 容器存储（与现状一致）
- 输出 `y`：`k/QK_K` 个 `block_ifairy_q16`
- 语义：对整个 `k` 扫描得到 `max_real/max_imag`，然后用这两个 scale 量化所有 block
- 仍然把 `d_real/d_imag` 写入每个 block（数值相同），以保持结构体不变

### 3.3 量化算法（两遍）

以 ref 版为例（伪代码）：

```c
assert(k % QK_K == 0);
nb = k / QK_K;

// pass1: global max over entire k
max_r = eps; max_i = eps;
for j in [0..k):
  (xr, xi) = bf16pair_to_fp32(x[j])
  max_r = max(max_r, abs(xr))
  max_i = max(max_i, abs(xi))

iscale_r = (max_r > 0) ? 127/max_r : 0
iscale_i = (max_i > 0) ? 127/max_i : 0
d_r = (iscale_r > 0) ? 1/iscale_r : 0
d_i = (iscale_i > 0) ? 1/iscale_i : 0

// pass2: quantize each block with shared scale
for ib in [0..nb):
  y[ib].d_real = fp16(d_r)
  y[ib].d_imag = fp16(d_i)
  for lane in [0..QK_K):
    (xr, xi) = bf16pair_to_fp32(x[ib*QK_K + lane])
    y[ib].x_real[lane] = clamp_i8(round(xr*iscale_r))
    y[ib].x_imag[lane] = clamp_i8(round(xi*iscale_i))
```

### 3.4 并行/切分策略（避免破坏 decode-like 分段）

vec_dot（非 LUT）路径下，通常激活量化是“按列整段 K”完成的；如果未来也需要 `k_part` 分段并行，tensor-scale 需要 **两阶段并行**：

1) 第 1 阶段：每线程扫描自己负责的 K 子区间，产出 `local_max_r/local_max_i`
2) barrier + reduce：得到全局 `max_r/max_i`
3) 第 2 阶段：每线程量化自己负责的 K 子区间（共享 iscale）

工程上建议把“reduce buffer”（每线程写一个 `float[2]`）放在线程池 scratch 或临时栈上，避免 malloc。

### 3.5 路由开关（只影响 vec_dot，避免影响 LUT）

建议加一个独立 env（不和 LUT 的 env 混用），例如：

- `GGML_IFAIRY_VEC_DOT_ACT_TENSOR=0/1`（默认 1 或默认 0 都可以；取决于你对质量的容忍度）

启用条件（建议）：

- `GGML_IFAIRY_LUT=0`（或“未命中 LUT 路由”）
- 且 `src0->type == GGML_TYPE_IFAIRY` 且 `src1` 需要从 `F32` 转为 `IFAIRY_Q16`

落点：

- `ggml/src/ggml-cpu/ggml-cpu.c::ggml_compute_forward_mul_mat()` 的非 LUT 分支里，在准备 `act_q` 时选择
  - tensor-scale quant：`quantize_row_ifairy_q16_tensor(...)`
  - 否则：现有 `quantize_row_ifairy_q16(...)`

---

## 4) `ggml_vec_dot_ifairy_q16_K()`：6-block 融合 fastpath 设计

### 4.1 关键观察：跨 block 累加可以安全留在 int32

在 iFairy 的 2-bit 相位编码里：

- `wr/wi ∈ {-1,0,1}`
- 激活为 `int8 ∈ [-127,127]`

因此单个 block 内的 `sum_*` 最大量级约为 `256*127 = 32512`，跨 6 个 block 约 `195072`，远小于 `2^31`。

结论：

- **可以用 int32（甚至 NEON int32 向量）跨 block 累加，不需要 int64**（更省指令和寄存器）

### 4.2 fastpath 触发条件

在 `ggml_vec_dot_ifairy_q16_K()`（ARM DOTPROD 路径）里增加 fastpath：

- `nb == 6`（即 `n == 1536`）
- 激活 scale 为 tensor 级：
  - `x[i].d_real == x[0].d_real` 且 `x[i].d_imag == x[0].d_imag`（建议 `#ifndef NDEBUG` 下做 assert；Release 下可做一次 `if` 快速检查）
- 权重 scale 为 tensor 级（当前代码已假设）：建议同样在 Debug 下 assert：
  - `w[i].d_real == w[0].d_real` 且 `w[i].d_imag == w[0].d_imag`

若条件不满足，走现有 per-block 路径（保持行为不变）。

### 4.3 fastpath 的核心变化：把“每 block 的水平归约 + 标量缩放”变成“跨 6 block 合并后一次处理”

#### 4.3.1 当前 per-block 后处理（对照）

每个 block：

- `sum_* = vaddvq_s32(acc_*)` ×4
- `x_scale = fp16_to_fp32(x[i].d_*)` ×2
- `acc += x_scale * sum_*` ×4

#### 4.3.2 融合版后处理（建议实现）

在外层 `i` 循环外声明全局 int32 向量累加器：

- `int32x4_t acc_ac_total = vzero;`（以及 `ad/bc/bd`）

对每个 block：

- 仍然用现有 inline asm 得到 `acc_*0/acc_*1`
- block 末尾不做 `vaddvq`，只做：
  - `acc_ac_total += (acc_ac0 + acc_ac1)`
  - `acc_ad_total += (acc_ad0 + acc_ad1)`
  - ...

循环结束后：

- `sum_ac_total = vaddvq_s32(acc_ac_total)`（仅 4 次水平归约）
- 读取一次 `x_real = fp16_to_fp32(x[0].d_real)` 与 `x_imag = fp16_to_fp32(x[0].d_imag)`
- 计算：
  - `acc_ac_xr = x_real * (float)sum_ac_total`
  - `acc_bd_xi = x_imag * (float)sum_bd_total`
  - `acc_bc_xr = x_real * (float)sum_bc_total`
  - `acc_ad_xi = x_imag * (float)sum_ad_total`
- 末尾仍按现有公式合成并输出 bf16 pair：
  - `real = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi`
  - `imag = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi`

#### 4.3.3 伪代码

```c
if (nb == 6 && x_scales_uniform && w_scales_uniform) {
  int32x4_t acT=v0, adT=v0, bcT=v0, bdT=v0;
  for i=0..5:
    run_asm_to_fill(acc_ac0,acc_ac1, ...);
    acT += acc_ac0 + acc_ac1;
    adT += acc_ad0 + acc_ad1;
    bcT += acc_bc0 + acc_bc1;
    bdT += acc_bd0 + acc_bd1;

  int32 sum_ac = vaddvq_s32(acT); ...;
  float x_real = fp16(x[0].d_real); float x_imag = fp16(x[0].d_imag);
  float acc_ac_xr = x_real * (float) sum_ac;
  ...
  out = combine_with_w_coeff_and_store_bf16pair();
  return;
}

// fallback: existing per-block path
```

### 4.4 可能的数值差异与处理策略

融合后会改变浮点运算顺序（把 6 次 `acc += scale*(float)sum` 变成 1 次 `scale*(float)(sum_total)`）。

这会带来极小的 rounding 差异（通常更“接近真实值”，但不保证 bit-identical）。

建议：

- 默认允许这种差异（以测试通过为准）
- 如果确实需要最大程度复现旧结果，可退一步：
  - 仍在每个 block 末尾生成 `sum_*`，但只把 `x_real/x_imag` 外提（减少 load/convert），并继续每 block 做 `acc += x_scale * sum`（这样浮点路径与旧版几乎一致）

### 4.5 额外微优化（可选，按 xctrace 结果决定）

1) **prefetch 策略微调**
   - `nb==6` 可把 `__builtin_prefetch(w + i + 1)` 改成固定预取（减少分支/地址计算），但收益通常很小
2) **减少 `vaddq` 次数**
   - `acT += acc_ac0; acT += acc_ac1;` vs `acT += (acc_ac0 + acc_ac1)`：选择让编译器更容易生成最少指令的写法
3) **fastpath 粒度**
   - 先只做 `nb==6`（最贴合 1536）
   - 若验证收益稳定，再考虑推广到 `nb % 6 == 0` 或更一般的 `x_scales_uniform`（避免过早扩大改动面）

---

## 5) xctrace 采样方案（A/B 可复现）

本节给出两层采样：

- **层 A：Time Profiler（现有脚本可直接解析 leaf）**：定位热点与“是否真的把时间花在 vec_dot 上”
- **层 B：CPU Counters（可选）**：拿到 cycles / instructions / IPC / cache miss 等硬指标（需要自定义模板）

### 5.1 建议新增 microbench：`tools/ifairy-vecdot-microbench`

目的：把测量从端到端图执行中剥离，减少噪声。

建议实现要点：

- 参数：`--k`（默认 1536）、`--iters`、`--warmup`、`--seed`
- 数据：
  - `w`：`block_ifairy[nb]`（`d_real/d_imag` 固定为 1.0）
  - `x`：`block_ifairy_q16[nb]`（填随机 int8；`d_real/d_imag` 可配置为“per-block”或“tensor-scale（全部相同）”）
- 校验：
  - 用 `ggml_vec_dot_ifairy_q16_K_generic()` 做参考结果
  - 比较 bf16 输出对应的 fp32（容差建议 `1e-3` 或按实际确定）
- A/B 开关：
  - 方案 1（推荐）：在 `ggml_vec_dot_ifairy_q16_K()` 内部用 env 控制是否启用 fastpath（例如 `GGML_IFAIRY_VEC_DOT_FUSED6=0/1`），microbench 不需要链接两份实现
  - 方案 2：保留旧实现为 `..._baseline()`，新实现为 `..._fused6()`，microbench 用参数选择调用

### 5.2 xctrace：Time Profiler（推荐主口径）

参考仓库已有口径与脚本：`scripts/ifairy_xctrace_leaf.py`（解析 `time-profile` schema）。

#### 5.2.1 录制命令（A/B 各一份 trace）

```bash
# A: baseline（禁用融合；或让条件不成立）
xcrun xctrace record --template 'Time Profiler' --output tmp/xctrace/ifairy-vecdot_base.trace \
  --time-limit 10s --window 6s --no-prompt \
  --env GGML_IFAIRY_LUT=0 \
  --env GGML_IFAIRY_VEC_DOT_FUSED6=0 \
  --launch -- ./build-rel/bin/ifairy-vecdot-microbench --k 1536 --iters 2000000 --warmup 20000 --seed 1

# B: fused6（启用融合；并确保 x 使用 tensor-scale）
xcrun xctrace record --template 'Time Profiler' --output tmp/xctrace/ifairy-vecdot_fused6.trace \
  --time-limit 10s --window 6s --no-prompt \
  --env GGML_IFAIRY_LUT=0 \
  --env GGML_IFAIRY_VEC_DOT_FUSED6=1 \
  --launch -- ./build-rel/bin/ifairy-vecdot-microbench --k 1536 --iters 2000000 --warmup 20000 --seed 1
```

说明：

- `--window` 取稳态区间（避开 warm-up）
- `xctrace record` 到 time-limit 时可能返回非零（常见 `54`）但 trace 仍有效（以文件为准）
- 若写缓存目录报错，可参考仓库已有 workaround：
  - `CFFIXED_USER_HOME=/tmp/xctracehome HOME=/tmp/xctracehome xcrun xctrace ...`

#### 5.2.2 导出与 leaf 摘要

```bash
xcrun xctrace export --input tmp/xctrace/ifairy-vecdot_base.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
  | python3 scripts/ifairy_xctrace_leaf.py --top 20 > tmp/xctrace/ifairy-vecdot_base.leaf.txt

xcrun xctrace export --input tmp/xctrace/ifairy-vecdot_fused6.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
  | python3 scripts/ifairy_xctrace_leaf.py --top 20 > tmp/xctrace/ifairy-vecdot_fused6.leaf.txt
```

解读要点：

- leaf top1 应该非常接近 `ggml_vec_dot_ifairy_q16_K`（microbench 的理想状态下接近 100%）
- 若 leaf 分散，说明 microbench 仍被其它开销主导（例如随机生成、校验、打印），需要收敛到“只跑内核”

### 5.3 xctrace：CPU Counters（可选增强口径）

目标：获取更“硬”的指标（cycles / instructions / IPC / cache misses）。

实践建议：

1) 先在 Instruments GUI 中创建一个自定义 “CPU Counters” 模板（勾选你关心的 counters），保存为自定义模板名（例如 `IFAIRY-VecDot-Counters`）
2) 用 xctrace CLI 复用该模板录制 A/B

CLI 示例：

```bash
xcrun xctrace list templates | rg -n \"IFAIRY-VecDot-Counters|Counters\"

xcrun xctrace record --template 'IFAIRY-VecDot-Counters' --output tmp/xctrace/ifairy-vecdot_base_counters.trace \
  --time-limit 10s --no-prompt \
  --env GGML_IFAIRY_VEC_DOT_FUSED6=0 \
  --launch -- ./build-rel/bin/ifairy-vecdot-microbench --k 1536 --iters 2000000 --warmup 20000 --seed 1

xcrun xctrace record --template 'IFAIRY-VecDot-Counters' --output tmp/xctrace/ifairy-vecdot_fused6_counters.trace \
  --time-limit 10s --no-prompt \
  --env GGML_IFAIRY_VEC_DOT_FUSED6=1 \
  --launch -- ./build-rel/bin/ifairy-vecdot-microbench --k 1536 --iters 2000000 --warmup 20000 --seed 1
```

导出方式（第一次需要看 toc 找到 schema）：

```bash
xcrun xctrace export --input tmp/xctrace/ifairy-vecdot_base_counters.trace --toc > tmp/xctrace/toc_base.xml
xcrun xctrace export --input tmp/xctrace/ifairy-vecdot_fused6_counters.trace --toc > tmp/xctrace/toc_fused6.xml

# 然后在 toc 里搜索 schema 名（可能包含 "counters"），再用对应 xpath 导出 table。
```

建议最小指标集合：

- `Cycles`（或等价 fixed counter）
- `Instructions Retired`
- `IPC`（后处理：`inst/cycles`）
- 至少一个缓存指标（L1D miss 或 L2 miss；以 Instruments 实际可选项为准）

注意：

- 部分 macOS 版本在新硬件上可能限制 counters（如果遇到 “Operation not permitted”，优先升级系统或改用 Time Profiler 口径先推进工程）

---

## 6) A/B 实验矩阵（建议最少跑这些组合）

### 6.1 microbench 维度

- `k=1536`（重点）
- `k=256,512,4096`（看 fastpath 是否只在 1536 有效）

### 6.2 模式

1) `x` per-block scale（不应触发 fused6 fastpath）
2) `x` tensor-scale（应触发 fused6 fastpath）

### 6.3 线程与噪声控制

- microbench 建议单线程（它本身就是单核内核测量）
- 端到端用 `llama-bench` 时：
  - 固定 `-t`，固定 `--repetitions 1`/`--no-warmup`，并多跑几次取中位数

---

## 7) 风险与对策

### 7.1 模型质量风险（tensor-scale 量化误差可能更大）

按 tensor 维度量化会让 scale 受整段 K 的最大值支配，可能放大局部误差。

对策：

- 默认用 env gating（只在确认质量可接受时启用）
- 在 `test-ifairy` 或额外脚本中加入“输出差异统计”，确保误差在可接受范围（至少不出现明显崩坏）

### 7.2 数值不一致风险（融合改变浮点运算顺序）

对策：

- 先在 microbench 做 `generic vs optimized` 的误差统计（max abs / max rel）
- 若误差不可接受，改用“只外提 x_scale，不合并 sum”版本（保持更接近原实现的浮点路径）

### 7.3 维护风险（fastpath 变多、难读）

对策：

- fastpath 严格用条件封装（`if (nb == 6 && scales_uniform)`），不污染通用路径
- Debug 下用 assert 固化“不变式”

---

## 8) 建议落地顺序（Stage-Gated）

1) **只加 microbench + xctrace 口径**（不改核心内核）：把测量链路跑通
2) **加 tensor-scale 量化（vec_dot-only）**：确保 `x` scale 一致性可控
3) **加 fused6 fastpath**：先实现最小版本（跨 block 合并 int32 + 末尾一次归约/缩放）
4) **验证**：
   - `test-ifairy` + `GGML_IFAIRY_LUT=0` 的 vec_dot 相关用例
   - microbench A/B + xctrace leaf
   - `llama-bench`（LUT=0）端到端 tok/s

---

## 9) 关键代码位置索引（便于实现时快速定位）

- vec_dot 内核（ARM）：`ggml/src/ggml-cpu/arch/arm/quants.c::ggml_vec_dot_ifairy_q16_K`
- vec_dot 参考实现：`ggml/src/ggml-cpu/quants.c::ggml_vec_dot_ifairy_q16_K_generic`
- 激活量化（ARM）：`ggml/src/ggml-cpu/arch/arm/quants.c::quantize_row_ifairy_q16`
- 激活量化（ref）：`ggml/src/ggml-quants.c::quantize_row_ifairy_q16_ref`
- mul_mat 路由（LUT vs 非 LUT）：`ggml/src/ggml-cpu/ggml-cpu.c::ggml_compute_forward_mul_mat`
- xctrace leaf 解析脚本：`scripts/ifairy_xctrace_leaf.py`

