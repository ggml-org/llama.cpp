# iFairy 2-bit 复数模型 · ARM NEON 3‑Weight LUT 设计（TL1）

> 本文从「零代码」视角设计一套面向 ARMv8.2+NEON/DOTPROD 的 iFairy 2-bit 复数三权重（3‑weight）LUT 推理方案，充分吸收以往 2‑weight / 3‑weight 尝试中的经验教训，并借鉴 BitNet 风格的 LUT 矩阵乘思路。文档仅描述架构与算法，不依赖具体源码文件。

---

## 1. 目标与约束

### 1.1 目标

- 为 2‑bit 复数权重模型设计一套 **三权重 LUT（3‑weight LUT）** 推理路径，用于矩阵向量乘 / 小批量矩阵乘。
- 充分利用 ARM NEON 查表与向量加法（`vqtbl1q_s8`、`vaddq_*` 等），在 CPU 上获得高吞吐。
- 与标量实现在数值上等价（逻辑正确），并在性能上尽量接近或超越基于两权重分组（2‑weight）的 LUT 方案。
- 保持架构可扩展：3‑weight LUT 作为独立的「可选加速路径」，可以轻松回退到 2‑weight LUT 或逐元素点积。

### 1.2 约束

- **权重编码**：每个权重使用 2 bit 表示复数集合 `{−1, +1, −i, +i}`，其中
  - `00 → −1`（实部）
  - `01 → +1`（实部）
  - `10 → −i`（虚部）
  - `11 → +i`（虚部）
- **激活格式**：
  - 模型侧激活为复数：实部、虚部分别量化为 int8，并带有各自的缩放因子。
  - 需要 per‑tensor 或 per‑行的激活缩放，保证 LUT 内的 int8 不溢出。
  - 当前实现复用 `block_ifairy_q16` 作为激活容器（int8 实/虚分平面 + fp16 缩放），不新增激活类型。原因：该量化块已在 ggml 注册、量化/缩放链路完整，符合「预量化 + 查表」性能模型；直接使用 float 激活会在预处理引入更多浮点访存，抵消 LUT 吞吐优势。(但是激活的tensor type 为 GGML_TYPE_F32, 也就是用 F32类型做存储，前后16位分别是实部和虚部的f16)
- **硬件假设**：
  - ARMv8.2‑A + NEON。推荐平台额外支持 DOTPROD 指令集（`__ARM_FEATURE_DOTPROD`），但 **本 3‑weight LUT 路径仅依赖 NEON 基本向量与查表指令**（`vqtbl1q_s8`、`vaddq_*` 等），不强制使用 `vdotq_s32`。
  - 假设内存对齐到 16 / 64 字节，有利于加载和预取。
- **软件约束**：
  - 3‑weight LUT 引擎必须是可选的：通过编译期宏或运行时环境变量开启/关闭。
  - 不改变已有权重文件格式和前向计算接口，只在内部引入 LUT 加速路径。

---

## 2. 数学基础与三权重组合

### 2.1 2‑bit 复数权重语义

每个 2‑bit 权重编码 `c ∈ {0,1,2,3}` 表示一个复数单位权重 `w(c)`：

- `c = 0 (00)`：`w = −1`
- `c = 1 (01)`：`w = +1`
- `c = 2 (10)`：`w = −i`
- `c = 3 (11)`：`w = +i`

对给定激活 `x = x_r + i x_i`，其对输出的复数贡献为

```text
contrib(c, x) = w(c) · x
```

展开可得

```text
contrib_real  = Re(w(c) · x)
contrib_imag  = Im(w(c) · x)
```

权重行与激活列的点积就是对所有权重‑激活对的贡献求和。

### 2.2 三权重组合（3‑weight）

3‑weight LUT 的基本单元是连续的三个 2‑bit 权重：

```text
codes = (c0, c1, c2),  ci ∈ {0,1,2,3}
```

对应的 3 个权重分别作用在激活向量的 3 个位置 `(x0, x1, x2)` 上。标量形式下的复数贡献为

```text
S_real = Re(w(c0)·x0 + w(c1)·x1 + w(c2)·x2)
S_imag = Im(w(c0)·x0 + w(c1)·x1 + w(c2)·x2)
```

直接在主循环中逐个展开会产生大量标量运算与分支，难以发挥 NEON 性能。LUT 的目标是在 **预处理阶段** 将这些模式尽量折叠，使主循环只需要：

1. 通过一个紧凑的索引（6bit）查表得到「基准复数对」；
2. 应用一个简单的「变换因子」在向量层面进行 lane 交换与取负；
3. 使用 NEON 向量加法将查表结果累加到整数累加器中。

---

## 3. 总体架构与模块划分

整个 3‑weight LUT 推理链路拆分为三个阶段：

1. **权重转换阶段（Weight Transform）**
   - 从原始 2‑bit 复数权重格式提取出三权重组的 6bit 索引。
   - 为每个权重张量构建对应的统计信息（缩放、平铺参数等）。
   - 为三权重 LUT 的运行时准备索引缓冲区（index buffer）。

2. **激活预处理阶段（Activation Preprocess）**
   - 对输入激活进行 per‑tensor/per‑行量化，得到 LUT 缩放因子。
   - 将 q16 或 float 激活转为 int8，并按 NEON 友好的布局进行 pack。
   - 构建 3‑weight LUT 表（两个小表 + 变换头规则），为主内核提供查表入口。

3. **矩阵乘内核阶段（Matmul Kernel）**
   - 在 NEON 内核中按行遍历权重、按列加载激活与 LUT。
   - 使用 6bit 索引拆解为「变换头 + 表选择 + 行索引」组合，查表得到复数对。
   - 应用变换因子（±1、±i），并用 NEON 向量加法完成累加。
   - 将累加结果与缩放因子合并，输出为 float 或 bf16。

各阶段之间仅通过「权重索引缓冲 + 激活 pack + LUT 表 + 缩放数组」交换数据，不依赖任何特定文件结构。

---

## 4. 6bit 索引设计与变换头

### 4.1 索引结构

对每组三个 2‑bit 权重 `(c0, c1, c2)`，先按固定约定打包成一个 **原始 6bit 索引**：

```text
// 约定：高位在前，c0 对应最左侧权重
idx_raw = (c0 << 4) | (c1 << 2) | c2   // ∈ [0,63]
```

该约定在整个路径中保持不变，`canonical_idx[64]` / `factor[64]` 常量表、权重转换阶段的索引缓冲构造以及参考脚本的枚举逻辑都必须遵守这一拼接方式。

为了便于说明，可以将 `idx_raw` 拆分为三部分：

```text
idx_raw = [b5 b4 | b3 | b2 b1 b0]
          高2位    表位    低3位
```

- `b5 b4`（高 2 位）：**变换头（transform head）**，决定变换因子与「码字翻转模式」。
- `b3`：**表选择位**，`0` 选用「表 0」，`1` 选用「表 1」。
- `b2 b1 b0`：**行索引**，对应表内的 8 行之一。

### 4.2 变换头语义与 canonical 映射

出发点是：权重集合 `{−1, +1, −i, +i}` 在乘法下构成一个 4 元循环群。对任意三元组

```text
(w0, w1, w2),  wi ∈ {−1, +1, −i, +i}
```

总可以写成

```text
(w0, w1, w2) = f · (1, u1, u2)
```

其中 `f ∈ {−1, +1, −i, +i}` 为「公共因子」，`u1, u2 ∈ {−1, +1, −i, +i}`。也就是说：

- 规范化后令首位恒为 `1`，只需枚举 `(u1, u2)` 即可；
- 因此 **规范模式（canonical pattern）只有 4² = 16 种**，可以用 4bit 表示；
- 原始 64 种三权重组合 = 16 个规范模式 × 4 个复数因子。

变换头 `b5 b4` 的角色，就是把「原始三元组」映射为「规范模式 + 复数因子」的一种编码方式。可以选取如下语义（仅作为一种可能的构造方案）：

- `00`：保持码字不变，对应因子 `f =  1`
- `01`：整体乘以 `−1`，对应因子 `f = −1`
- `10`：整体乘以 ` i`，对应因子 `f =  i`
- `11`：整体乘以 `−i`，对应因子 `f = −i`

实现上，并不推荐在内核中动态解析 `head`，而是 **离线预先展开为两张常量表**：

```text
canonical_idx[64] : 原始 6bit 索引 → 4bit 规范索引（表号 + 行号）
factor[64]        : 原始 6bit 索引 → {1, −1, i, −i}
```

这里的「规范索引」正是前述 16 个 `(1, u1, u2)` 模式之一；可以直接用低 4bit 表示为

```text
idx_canonical ∈ [0, 15]  // 4bit : [表号 | 行号]
```

**重要约定**：

- 4.2 中的「head 翻转规则」仅用于定义 `canonical_idx[64]` / `factor[64]` 这两张常量表；
- **NEON 内核不再显式解析 head bit**，只需要：
  - 在权重转换阶段用原始 6bit 索引 `idx_raw` 查 `canonical_idx[idx_raw]` / `factor[idx_raw]`；
  - 把得到的 `idx' = idx_canonical` 与 `factor` 写入索引缓冲；
  - 运行时直接用 `idx'` 查 LUT，用 `factor` 做复数变换（参见 8.3）。

### 4.3 表选择与行索引

逻辑上，可以把 16 个规范模式视作「2 张表 × 每表 8 行」：

- 索引第 3 位 `b3` 作为表选择位：
  - `b3 = 0` 时选择 **表 0**；
  - `b3 = 1` 时选择 **表 1**。
- 索引低 3 位 `b2 b1 b0` 作为 LUT 表内的行索引（0–7）。

每个 `(表号, 行号)` 对应一个规范模式 `(1, u1, u2)` 在给定激活 `(x0, x1, x2)` 上的**基准复数贡献**：

```text
[r, i] = Σj cj · xj   ,  cj ∈ {1, −1, i, −i}, 且 c0 = 1
```

实现层面，为了让 nibble 直接作为 `vqtbl1q_s8` 的索引，本设计采用 **实/虚分平面 + 16 项扁平表** 的布局：

- 对每个三权重组构建两条长度为 16 的数组：
  - `lut_real[16]`：按 `idx' ∈ [0,15]` 存储实部；
  - `lut_imag[16]`：按 `idx' ∈ [0,15]` 存储虚部；
- 编码约定：

```text
idx' = (表号 << 3) | 行号
```

这样，NEON 内核可以直接使用 nibble `idx'` 做查表：

- `v_real = vqtbl1q_s8(LUT_REAL, idx_vec)`
- `v_imag = vqtbl1q_s8(LUT_IMAG, idx_vec)`

无需在内核中对索引做额外的移位 / 加减运算，避免了「交错存储」带来的性能隐患。类似的 LUT 构造与扁平布局，可参考 BitNet TL1 的 `lut_ctor` 与 `tbl_impl_*` 实现（`BitNet/preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl1.h:96` 以及 `BitNet/preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl1.h:186`）。

为便于后续 LUT 构造与内核实现查阅，当前实现中 `idx' ∈ [0,15]` 与规范模式 `(u1, u2)` 的对应关系（来源于 `ifairy_lut_baseline[16]`）整理如下：

| idx' | u1 | u2 |
|:----:|:--:|:--:|
|  0   |  1 |  1 |
|  1   |  1 |  i |
|  2   |  1 | −1 |
|  3   |  1 | −i |
|  4   |  i |  1 |
|  5   |  i |  i |
|  6   |  i | −1 |
|  7   |  i | −i |
|  8   | −1 |  1 |
|  9   | −1 |  i |
| 10   | −1 | −1 |
| 11   | −1 | −i |
| 12   | −i |  1 |
| 13   | −i |  i |
| 14   | −i | −1 |
| 15   | −i | −i |

---

## 5. 复数变换因子与向量操作

### 5.1 数学定义

对 LUT 表返回的基准复数向量 `v_base = [r, i]`，变换因子作用如下：

- `factor = 1`：`v = [ r,  i]`
- `factor = −1`：`v = [−r, −i]`
- `factor = i`：
  - 复数：`i · (r + i i) = −i·r + i·r = [−i, r]`
  - 向量：交换两 lane，再对「偶数 lane（实部）」取负
- `factor = −i`：
  - 复数：`−i · (r + i i) = i · (−r − i i) = [i, −r]`
  - 向量：交换两 lane，再对「奇数 lane（虚部）」取负

这里的「偶数/奇数 lane」以 0‑based 索引计：

- lane 0：实部
- lane 1：虚部

### 5.2 标量实现规则

标量版可以这样写：

```text
switch (factor) {
  case 1:
    out_r =  r;  out_i =  i;  break;
  case -1:
    out_r = -r;  out_i = -i;  break;
  case i:
    out_r = -i;  out_i =  r;  break;
  case -i:
    out_r =  i;  out_i = -r;  break;
}
```

也可以进一步用「交换 + 有条件取负」来统一实现，方便映射到 NEON。

### 5.3 NEON 向量实现

在 NEON 中，我们通常以 `int8x16_t` 或 `int16x8_t` 的形式批量处理多个复数对。可以用以下原则实现变换：

1. **交换实部与虚部**：
   - 对每个复数对 `[r, i]`，使用 `vrev64q_s8` 或 `tbl` 实现 lane 交换。
2. **按 lane 取负**：
   - 使用预构造的掩码向量，对需要取负的 lane 执行 `veor` + 加一，或直接使用 `vnegq_s8` 和 `vbsl` 融合。

一个通用的向量化思路：

- 准备四个掩码：
  - `mask_identity`：全 0（不取负，不交换）。
  - `mask_neg_all`：两 lane 均取负。
  - `mask_neg_real`：仅偶数 lane（实部）取负。
  - `mask_neg_imag`：仅奇数 lane（虚部）取负。
- 对于 `factor = 1 / −1 / i / −i`：
  - 根据 factor 选择是否执行 lane 交换；
  - 根据 factor 选择使用哪一个「取负掩码」。

具体实现可结合 `vbsl`（按掩码选择）和 `vnegq_s8` 或 `veor` 组合完成。

---

## 6. 数据与内存布局设计

### 6.1 权重侧布局

权重数据在存储层面保持原有的 2‑bit 压缩格式，不做改变。3‑weight LUT 为运行时增加一块独立的索引缓冲：

- **原始权重缓冲**：
  - 按行存放，以 2 bit 压缩形式表示复数权重。
  - 每行长度为 `k`（列数），每个块包含固定数量的权重。
- **三权重索引缓冲**：
  - 每 3 个权重占 1 个字节（仅用低 6bit）。
  - 按「行优先、组内顺序」存放，即：
    - 第 0 行：`idx[0], idx[1], ..., idx[k/3 - 1]`
    - 第 1 行：...
  - 尾部 `k % 3 != 0` 的情况：
    - 加载 / 转换权重时，强制将每行 `K` 向上 pad 到 3 的倍数：`K3 = ((K + 2) / 3) * 3`（与 6.3 保持一致）。
    - 多出来的 1–2 个权重 code 可以统一填为某个固定值（例如 `c = 0`），但在激活预处理和 LUT 构造时，对应位置的激活直接置 0，这样这些 pad 权重不会对结果产生任何贡献。

索引缓冲应按 16 或 64 字节对齐，以便 NEON 批量加载和预取。

### 6.2 激活与 LUT 布局

激活预处理阶段产出以下几块数据：

1. **量化激活 pack**
   - 实部和虚部各一份 int8 数组。
   - 按实现选择 NEON 友好的布局打包，只要 LUT 构造阶段能够方便访问每个位置的实部和虚部值即可；3‑weight 设计下不依赖 `vdotq_s32`，更多关注顺序访问与 cache 友好性。

2. **LUT 表（实/虚分平面）**
   - 逻辑上，每个三权重组有两张「8 行 × 复数对」的小表（表 0 / 表 1，各 8 行，每行 `[r, i]`）。
   - 为了让 nibble 直接作为 `vqtbl1q_s8` 索引，实际存储时对这两张表做「扁平 + 分平面」重排：
     - `lut_real[16]`：按 `idx' ∈ [0,15]` 存储 16 个规范模式的实部；
     - `lut_imag[16]`：按 `idx' ∈ [0,15]` 存储 16 个规范模式的虚部；
     - 其中 `idx' = (表号 << 3) | 行号`，与 4.3 的定义一致。
   - 每列激活向量包含 `G = K3 / 3` 个三权重组，因此 **一列对应 G 组 `lut_real` / `lut_imag`**：
     - 可以按组顺序连续存放，也可以按 BK 分块组织；
     - 所有权重行在该次 matmul 中共享这些 LUT（LUT 仅依赖激活）。

3. **缩放数据**
   - 激活缩放：例如实部、虚部各一个浮点缩放，或每行独立缩放。
   - 权重缩放：从权重转换阶段带入，设计上允许每个权重块有各自的实/虚缩放；**当前 iFairy 量化实现采用 per‑tensor 权重缩放，同一张量内不同 `block_ifairy` 共享同一个 `s_w`，因此也可以视作 per‑tensor 量化路径。**

整体工作区布局示例（以「按列预处理激活」为例）：

```text
[ 量化激活 pack ][ LUT 表 (G 组 × (lut_real + lut_imag)) ][ LUT 缩放因子 ]
```

对多列激活，重复上述块并按 64 字节对齐。

### 6.3 激活分组与 NEON lane 对齐

为了兼顾「每 3 个权重一组」与 NEON 16‑lane 的向量宽度，本设计采用按组（group）对齐的方式：

- 逻辑维度：
  - 原始隐藏维长度为 `K`。
  - 将 `K` 向上补齐到 3 的倍数：`K3 = ((K + 2) / 3) * 3`，多出的权重和激活记为 0。
  - 三权重组数为 `G = K3 / 3`。
- 向量维度：
  - 内核以「每 16 组三权重」为基本单元处理：一次从索引缓冲加载 16 个索引字节，打包为一个 `uint8x16_t codes`。
  - `codes & 0x0f` 提取 nibble（0–15）作为 LUT 查表索引；`codes >> 4` 提取变换头。
- K 方向循环：
  - 以 `g` 为三权重组下标，循环 `for (g = 0; g < G; g += 16)`。
  - 不再直接按「标量权重数量」迭代，而是按「组数」迭代，解决了「3」与「16」的对齐问题。

这种分组方式与 BitNet TL1 中「按 nibble / pair 为基本单位」的思想一致，只是这里的基本单位换成了 3‑weight 组。

### 6.4 工作区大小估算公式（wsize）

工作区用于存放激活 pack、三权重 LUT 表以及 LUT 缩放因子。以单列激活为例，定义：

- `K3 = ((K + 2) / 3) * 3`：向上补齐到 3 的倍数的隐藏维长度。
- 三权重组数 `G = K3 / 3`。

- 激活 pack（实 / 虚各一份 int8）：

```text
act_bytes_per_col      = 2 * K3 * sizeof(int8_t)
```

- 三权重 LUT 表：每个三权重组对应一对 `(lut_real[16], lut_imag[16])`，共 32 个 int8：

```text
lut_bytes_per_group    = 2 * 16 * sizeof(int8_t)          // real + imag
lut_bytes_per_col      = G * lut_bytes_per_group
```

- LUT 缩放因子（实 / 虚各一 float）：

```text
scale_bytes_per_col    = 2 * sizeof(float)
```

对于有 `N` 列激活的 matmul，忽略按 BK 分块的额外表时，工作区大小估算为：

```text
wsize = ALIGN64( N * (act_bytes_per_col + lut_bytes_per_col + scale_bytes_per_col) )
```

其中 `ALIGN64(x) = ((x + 63) / 64) * 64`，用于满足分配器 64B 对齐的要求。若未来按 K 方向引入固定 BK 分块，并为每个 BK tile 构建独立 LUT，则可将 `lut_bytes_per_col` 改写为 `G_tile * lut_bytes_per_group`，其中 `G_tile = BK / 3`，与 BitNet TL1 中的 per‑BK LUT 组织方式类似（参见 `BitNet/preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl1.h:184`）。

需要注意的是：上式的 `wsize` 仅覆盖「激活工作区」相关缓冲（量化激活 + LUT 表 + LUT 缩放因子），**不包含权重侧索引缓冲与块级缩放数组**。索引缓冲大小完全由索引格式决定，见 6.5 小节。

从 cache 的角度看，当隐藏维 `K` 很大时，直接为整列构建 LUT 会迅速超过典型 ARM L1 D‑Cache 的容量（64KB 左右）：

- LUT 大小近似为 `lut_bytes_per_col ≈ (K3 / 3) × 32` 字节。
- 例如 `K = 8192` 时，`K3 ≈ 8192`，对应 LUT 约 `8192 / 3 × 32 ≈ 87KB`，明显大于 64KB。

因此，建议在大 `K` 场景下启用 K 方向的 BK 分块，使每个 BK tile 的 LUT 常驻 L1 的一部分：

- 令 `lut_bytes_per_tile = (BK / 3) × 32`，推荐控制在 L1 D‑Cache 的 50% 以内（约 32KB）。
- 对 64KB L1 而言，一个实用选择是 `BK ≈ 3072`（`3072 / 3 × 32 ≈ 32KB`）。

8.2 中的「按 K 方向 block」循环应与该 BK 选择保持一致：对每个 BK tile 分别构建 LUT 并完成对应行段的累加，避免在处理不同行时对整列 LUT 反复产生 L1 miss。

### 6.5 索引缓冲格式与大小

三权重索引缓冲属于权重转换阶段的持久数据，而非每次 matmul 的临时工作区。本设计采用紧凑的单字节编码：

```text
index_byte:
  bits[0..3] : idx'         // 4bit 规范索引 ∈ [0,15]
  bit [4]    : 保留位       // 目前填 0，预留扩展
  bit [5]    : neg_real     // 0/1 : 是否对实部取负（0x20）
  bit [6]    : neg_imag     // 0/1 : 是否对虚部取负（0x40）
  bit [7]    : do_swap      // 0/1 : 是否交换实/虚（0x80）
```

其中 `(do_swap, neg_real, neg_imag)` 对应某一复数因子 `f ∈ {1, −1, i, −i}` 的向量化操作编码（参见 5.3 与 8.3）：

- `f =  1` ：`do_swap = 0, neg_real = 0, neg_imag = 0`
- `f = −1` ：`do_swap = 0, neg_real = 1, neg_imag = 1`
- `f =  i` ：`do_swap = 1, neg_real = 1, neg_imag = 0`
- `f = −i` ：`do_swap = 1, neg_real = 0, neg_imag = 1`

这样，NEON 内核可以直接从字节中提取：

- `idx_vec      = codes & 0x0f`
- `swap_mask    = (codes & 0x80)`
- `neg_real_mask= (codes & 0x20)`
- `neg_imag_mask= (codes & 0x40)`

再通过少量 `vbsl` / `veor` / `vrev` 实现复数变换，无需额外 `vqtbl1`。

索引缓冲空间占用：

```text
index_bytes_per_row = G * sizeof(uint8_t)      // 每组三权重 1 字节
index_bytes_total   = M * index_bytes_per_row  // M 行权重
```

该部分内存归属于权重 tensor 的「extra」结构，与 BitNet TL1 中的 `tensor->extra` 类似（参见 `BitNet/preset_kernels/bitnet_b1_58-large/bitnet-lut-kernels-tl1.h:326`），不计入 `wsize`。

### 6.6 内存开销小结

为便于做内存预算，这里汇总三权重路径在激活侧工作区与权重侧额外信息的大致开销（忽略对齐与实现细节）：

- 每列激活工作区（临时）：
  - 激活 pack：`act_bytes_per_col = 2 * K3 * sizeof(int8_t)`。
  - LUT 表：`lut_bytes_per_col = (K3 / 3) * 32` 字节（每组三权重 32 个 int8）。
  - LUT 缩放：`scale_bytes_per_col = 2 * sizeof(float)`（实 / 虚各一）。
- 每行权重额外信息（持久）：
  - 索引缓冲：`index_bytes_per_row = (K3 / 3) * sizeof(uint8_t)`。
  - 权重缩放：若每个 `block_ifairy` 保存一对 `(d_real, d_imag)` 浮点缩放，行内共有 `B_row` 个 block，则约为 `wscale_bytes_per_row ≈ B_row * 2 * sizeof(float)`，其中 `B_row` 取决于具体量化 block 大小（例如按 `K` 或通道方向分块）。
- 数量级：
  - 激活侧工作区：`O(N * K)`，随列数 `N` 线性增长。
  - 权重侧索引与缩放：`O(M * K)`，一次构建后在整个推理过程中复用。

实际实现时，可在此基础上加上对齐与额外缓冲的常数项，作为整体内存 budget 的参考。

---

## 7. 激活预处理与 LUT 构造

### 7.1 Per‑Tensor 激活量化

激活预处理的第一步是确定量化尺度：

1. 对实部激活，统计绝对值最大值 `max_r`；
2. 对虚部激活，统计绝对值最大值 `max_i`；
3. 计算缩放因子：

```text
inv_r = max_r > 0 ? 127 / max_r : 0
inv_i = max_i > 0 ? 127 / max_i : 0
```

NEON 优化方式：

- 使用 `vabsq_f32` + `vmaxq_f32` 在向量层面进行 max‑reduction；
- 使用 `vmaxvq_f32` 把向量最大值收缩为标量。

### 7.2 激活量化与打包

在已知 `inv_r, inv_i` 的前提下，将激活量化为 int8：

1. 将原始激活（可能为 int8+缩放或 float）转换到浮点值。
2. 乘以对应的 `inv_r` 或 `inv_i`。
3. 使用「四舍六入五成双」或向最近整数取整（`vcvtnq_s32_f32`），再截断到 int16。
4. 使用饱和裁剪到 `[-127, 127]`，并最终转为 int8。

NEON 实现要点：

- 使用 `vmulq_f32` 进行标度缩放；
- 使用 `vcvtnq_s32_f32` 做向最近整数取整；
- 使用 `vmovn_s32` 将 int32 收缩为 int16；
- 使用 `vmaxq_s16` / `vminq_s16` 实现饱和裁剪；
- 最后用 `vmovn_s16` 或标量转换得到 int8。

为了适配三权重 LUT 内核，激活可以按「每 16 个值一组」，事先整理成 NEON 友好的 layout，使 LUT 构造阶段能高效读出 `x_r[j], x_i[j]`。3‑weight BitNet 风格的实现中，LUT 与累加器内部均推荐采用「实/虚分平面」布局，激活 pack 的具体布局由实现自行选择，只需保证良好的顺序访存和对齐。

### 7.3 三权重 LUT 构造（按组 + 16 个规范模式）

三权重 LUT 的构造遵循两个原则：

- **原则 1：离线只需 16 个规范模式**
  - 任意 3×2‑bit 组合 `(c0, c1, c2)` 在 4.2 中已经说明，可以写成

    ```text
    (w(c0), w(c1), w(c2)) = f · (1, u1, u2)
    ```

    其中 `f ∈ {1, −1, i, −i}`，`u1, u2 ∈ {1, −1, i, −i}`。
  - 因此离线阶段只需枚举 16 个 `(1, u1, u2)` 规范模式，构造 `canonical_idx[64]` / `factor[64]` 常量表（参见 10.1），**不需要在运行时对每个组枚举 64 种组合**。

- **原则 2：运行时按组构建 16 项小 LUT**
  - 对给定列激活，隐藏维被划分为 `G` 个三权重组（6.3）。
  - 对于每个组 `g`，只需要为 16 个规范模式构建一对 `lut_real_g[16]` / `lut_imag_g[16]`：

    ```text
    Φ_g(k) = Σj c_kj · x_gj ,  k ∈ [0,15]
    ```

    其中 `c_kj` 是第 `k` 个规范模式在位置 `j ∈ {0,1,2}` 上的权重（`c_k0 = 1`），`x_gj` 是该组对应的激活。

基于上述约定，一个具体的运行时 LUT 构造流程如下：

1. 针对给定的激活 pack，按照 6.3 将激活划分为 `G` 个三权重组 `(x_g0, x_g1, x_g2)`。
2. 对于每个组 `g`：
   - 在 int16 或 int32 累加器中，依次遍历 16 个规范模式 `k ∈ [0,15]`，根据模式定义累加

     ```text
     S_real = Re(Σj c_kj · x_gj) ,  S_imag = Im(Σj c_kj · x_gj)
     ```

   - 将得到的 `(S_real, S_imag)` 按 `idx' = k` 写入 `lut_real_g[idx']` / `lut_imag_g[idx']`，写入前进行饱和裁剪。

**溢出与截断处理**：

- 对每个 `(g, k)` 的中间和，使用 int32 或 int16 累加，避免 3 项相加时的溢出。
- 在写入 LUT 表时统一执行饱和裁剪：

```text
clamp(v) = max(-127, min(127, v))
```

- NEON 中可以使用 `vqaddq_s16` 和 `vqmovn_s16` 等指令实现饱和加与饱和收缩。

如此，**每个三权重组只需 16 个规范模式表项**，原始 64 种三权重组合通过索引缓冲中的 `idx'` + `factor`（8.3）在内核运行时还原。

**实现提示：充分利用 {1,−1,±i} 结构与 SIMD**

- 由于 `c_kj ∈ {1, −1, i, −i}`，构表阶段不需要真正的复数乘法：
  - 实数乘法可以分解为「复制 + 加减」；
  - 虚数乘法可以分解为「实虚交换 + 加减」。
- 对固定的 `(x_g0, x_g1, x_g2)`，16 个规范模式可以按「模板」展开：
  - 例如先预计算 `(+x_g0, −x_g0, +i·x_g0, −i·x_g0)`，`(+x_g1, …)`，`(+x_g2, …)` 四组；
  - 再用加减组合这些预计算结果构出 16 个 `(S_real, S_imag)`。
- 可以进一步使用 NEON 对 LUT 构造做 SIMD 化：
  - 将多个组的 `x_gj` 打包为向量；
  - 利用 `vaddq_s16` / `vsubq_s16` / `vrev64q_s16` / `vbsl` 在 int16 域内批量生成多个 `k` 的结果；
  - 最终用 `vqmovn_s16` 将结果收缩为 int8 并用 `vst1q_s8` 一次性写出 16 字节，避免逐 byte 写入导致的 store‑to‑load forwarding penalty。

在典型 Transformer 配置下，隐藏维 `K` 一般不超过 8192。激活量化后落在 `[-127, 127]`，单个三权重组的理论最大绝对和约为 `3 * 127 = 381`；整行的分组数为 `G = K3 / 3`，则行内累加的绝对上界近似为 `381 * G ≈ 127 * K3`。对 `K = 4096` 与 `K = 8192` 有：

- `127 * 4096 ≈ 5.2e5`
- `127 * 8192 ≈ 1.0e6`

均远小于 `2^31 ≈ 2.1e9`，因此使用 int32 作为 LUT 构造与内核累加器的位宽是安全的。相对地，int16 累加器上界仅为 32767，在大 `K` 场景下存在明显溢出风险，不建议为了「省寄存器」而在累加路径上使用 int16。

实践中建议参考 BitNet TL1 中 `lut_ctor` 的思路（`BitNet/preset_kernels/bitnet_b1_58-large/bitnet-lut-kernels-tl1.h:48`），尽量将 LUT 构造阶段写成少量线性模板加法，而不是双层嵌套的「模式 × 位置」复数乘循环。

### 7.4 LUT 构造粒度与复用

结合 6.2 / 6.3 与 7.3，本设计对 LUT 粒度与复用作如下明确约定：

- **按组构造，按列复用**：
  - LUT 仅依赖激活，不依赖权重内容。
  - 对于一次 `M×K · K×N` 的 matmul：
    - 每一列激活向量先量化并打包一次（7.1 / 7.2）；
    - 对该列的 `G = K3 / 3` 个三权重组，分别构建一对 `lut_real_g[16]` / `lut_imag_g[16]`；
    - 这 `G` 组 LUT 在该列上 **被所有权重行共享**，不会随行号重复构建。

- **不在不同激活之间复用**：
  - 不在不同列之间复用 LUT（不同列激活不同）；
  - 不在不同层之间复用；
  - 不在不同 token 之间复用（除非未来引入更高级的缓存策略，当前设计不讨论）。

- **与 BitNet TL1 的对应关系**：
  - BitNet TL1 中，`ggml_preprocessor` 为右矩阵按 BK 分块构建 per‑tile LUT（`BitNet/preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl1.h:96`），`tbl_impl_*` 在内核中按行复用这些 LUT（`BitNet/preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl1.h:186`）。
  - 本 3‑weight 设计沿用同样的复用思想：**激活侧一次构表，多行权重只读共享**，差异仅在于：
    - BitNet TL1 的 LUT 面向实数 2‑bit / 3‑weight 组合；
    - 本方案的 LUT 面向复数 2‑bit 三权重组，并显式拆成实/虚两个平面。

- **复杂度与 amortize 分析**：
  - LUT 构造复杂度约为 `O(N × G)`，其中 `G = K3 / 3`；
  - 内核复杂度约为 `O(M × N × G)`；
  - 当 `M` 足够大（典型 Transformer 层）时，LUT 构造的单次成本会被大量输出行复用摊薄，整体仍然受内核吞吐主导。

这样，7.3 的「按组构造 16 项规范模式小表」与 6.4 的 `lut_bytes_per_col = G * lut_bytes_per_group` 语义保持一致，不再存在「per‑group LUT」与「每列只有一套小 LUT」之间的歧义。

### 7.5 缩放与输出复数的小结公式

综合激活量化、LUT 内部整数运算与权重缩放，单个输出复数 `y = y_r + i y_i` 的 float 版本可抽象为：

```text
// 激活侧：
x_q   ≈ x * inv_x                  // inv_x ∈ {inv_r, inv_i}，见 7.1

// LUT / 索引侧（整数域）：
acc_r = Σ_blocks Σ_groups S_r(block, group)    // int32
acc_i = Σ_blocks Σ_groups S_i(block, group)    // int32

// 每个 block 具有自己的权重缩放 s_w_block（来自 block_ifairy 中的 d_real / d_imag）
// 激活侧有 per‑tensor（或 per‑列）缩放 s_act_r, s_act_i = 1 / inv_r, 1 / inv_i

// 最终输出（以逐 block accumulate‑and‑dump 的视角）：
y_r ≈ Σ_blocks ( acc_r_block * s_w_block * s_act_r )
y_i ≈ Σ_blocks ( acc_i_block * s_w_block * s_act_i )
```

在具体实现中，可以将 `s_act_r/s_act_i` 视为「LUT 缩放因子」存入工作区（6.4），block‑wise 的 `s_w_block` 则随索引缓冲或额外数组一同保存，并在 8.2 所述的「处理完一个 K‑block 后」立即应用到该 block 的整数累加结果上，避免不同缩放因子在同一个 int32 累加器中混合。

当前 iFairy 训练 / 量化 pipeline 中，权重张量本身采用的是 per‑tensor 量化：同一张量内所有 `block_ifairy` 共享同一个权重缩放 `s_w`。这意味着从数学上可以把整行视作统一权重缩放的 per‑tensor 路径；本设计在公式中仍保留按 block 的 `s_w_block` 写法，是为了兼容未来可能的 per‑block 量化与更灵活的 BK 分块实现，不会改变现有 per‑tensor 量化权重的数值语义。

---

## 8. NEON 矩阵乘内核设计

### 8.1 设计目标

- 在保持数值正确性的前提下，最大限度利用 NEON：
  - 使用 `vqtbl1q_s8` 快速查表；
  - 使用 `vaddq_s16` / `vaddq_s32` 等向量加法累加 LUT 查表结果；
  - 避免复杂分支和跨 lane 的散乱访问。
- 内核应支持常见的矩阵形状（如 1536×4096、1536×1536、4096×1536），对这些形状进行展开和特化。
- 兼顾可读性与可维护性：将内核封装为「按行切片」的函数，方便多线程调度。

### 8.2 内核总体结构

以「单列激活 + 多行权重」的 matvec 为例，NEON 内核的循环结构建议如下：

1. 外层循环遍历权重行 `row`。
2. 对每一行，按「K 方向 block」循环（例如每次处理 `BK` 个标量权重，对应 `G_block = BK / 3` 个三权重组）：
   - 初始化一组 int32 累加向量（例如 4 组）用于本 block 内的整数累加，分别对应实部与虚部。
   - 按三权重组下标 `g` 遍历该 block 覆盖的组区间，步长为 16 组：每次加载 16 个索引字节，组成 `uint8x16_t codes`（参见 6.3 的分组方式）。
3. 对每个「block 内的 group‑tile」：
   - 预取下一块权重、索引与激活数据。
   - 从索引缓冲加载 16 个编码字节 `codes`，其中每个字节同时包含 `idx'` 与变换 opcode（参见 6.5）。
   - 使用 NEON 操作得到：
     - `idx_vec = codes & 0x0f` 作为 nibble，直接传给 `vqtbl1q_s8`；
     - `swap_mask / neg_real_mask / neg_imag_mask` 分别来自 `codes` 的高位，用于控制复数变换。
   - 使用 `vqtbl1q_s8` 分别在 `lut_real_g` / `lut_imag_g` 上查表，取出对应的 16 组三权重合并贡献（实/虚分平面）。
   - 根据 opcode 对查出的实/虚向量做统一的 swap / 取负变换（参见 8.3），不做逐 lane 的复杂 bit 操作。
   - 用 `vaddq_s16` / `vaddq_s32` 将查表结果累加到本 block 的整数累加器中。
4. 在处理完该 block 覆盖的所有组后：
   - 将整数累加器扩展为 float32（例如 `vmovl_s16` → `vcvtq_f32_s32`）；
   - 乘以该 block 对应的权重缩放因子 `s_w_block` 与 LUT 缩放因子 `s_act`；
   - 累加到该行对应的 float32 结果向量中；
   - 清零整数累加器，继续处理下一个 block。
5. 在处理完该行的所有 blocks 后，float32 结果向量即为该行最终输出（可选择在更外层再做 bias / 激活等处理）。

在实际 NEON 实现中，强烈建议在上述结构的基础上，在最内层循环对输出行方向做寄存器层面的行展开。例如一次同时处理 4 行权重（4 个 output channel）：

- 对当前 group‑tile 只加载一次该 tile 的 `LUT_REAL_g` / `LUT_IMAG_g` 到寄存器，在 4 行之间复用；
- 分别为 Row 0..3 加载索引字节（4 份 `codes_row`），查表并施加 `(do_swap, neg_real, neg_imag)` 变换后，累加到各自的 `acc_r[row]` / `acc_i[row]`；
- 寄存器预算示意：
  - LUT：2 个向量寄存器；
  - 累加器：4 行 × 2（实 / 虚）≈ 8 个向量寄存器；
  - 额外临时 / 索引寄存器：约 4–6 个；
  - 总计 < 20 个 NEON 寄存器，明显低于 ARMv8 NEON 提供的 32 个寄存器。

这样，LUT 加载的成本被 4 行共享，单行输出对应的「Load LUT 次数 / 3 个权重」显著下降，算术强度更高，内核更接近 compute‑bound。

### 8.3 索引解码与查表

索引解码是 3‑weight 内核的关键步骤。推荐的做法是将「头部→规范索引+变换因子」的复杂逻辑 **完全前移到权重转换阶段**，运行时仅保留简单、无分支的解码：

1. 权重转换阶段（离线或模型加载阶段）：
   - 对每个三权重组，计算原始 6bit 索引 `idx_raw`；
   - 使用 4.2 中的两张常量表：

     ```text
     idx'   = canonical_idx[idx_raw]   // 4bit : [表号 | 行号]
     factor = factor[idx_raw]          // {1, −1, i, −i}
     ```

   - 将 `idx'` 与 `factor` 预编码为 6.5 所述的单字节索引格式：`idx'` 占低 4bit，高 3bit 编码为 `(do_swap, neg_real, neg_imag)`。
   - 运行时代码**不再显式读取或解析 head bit**。

2. 内核运行时：
   - 从索引缓冲中读取 16 个编码字节 `codes`，每个字节包含：

     ```text
     idx'         = codes & 0x0f
     do_swap      = (codes & 0x80) != 0
     neg_real     = (codes & 0x20) != 0
     neg_imag     = (codes & 0x40) != 0
     ```

   - 将 `idx'` 打包成 `uint8x16_t idx_vec`，分别调用 `vqtbl1q_s8(LUT_REAL_g, idx_vec)` / `vqtbl1q_s8(LUT_IMAG_g, idx_vec)` 查表（参见 4.3 的扁平布局）；
   - 根据 `(do_swap, neg_real, neg_imag)` 对查出的复数向量施加变换（5.1–5.3），典型的分支无关 NEON 序列为：

     - 若需要交换实/虚：对 `(v_real, v_imag)` 使用 `vzip` / `vrev64` 或 `tbl` 交换；
     - 若需要取负：通过预先构造好的全 1 掩码向量配合 `veor` + 加一，或直接使用 `vneg` + `vbsl`，分别对实部 / 虚部进行条件取负；
     - 由于 `(do_swap,neg_real,neg_imag)` 只占 3bit，可以预先约定四种常见模式（对应 `1,−1,i,−i`），在实现中将「交换 + 取负」合并为 2–3 条 SIMD 指令，而无需额外的 `vqtbl1`。

如此，即使 4.2 中的「head 语义」较复杂，也只体现在一次性的常量表生成上，不会拉高内核解码开销；索引缓冲中保存的是已经展开为「规范索引 + 变换 opcode」后的结果，NEON 内核仅做按位掩码与向量加减，避免在最内层循环中出现额外的查表或分支。

### 8.4 NEON 指令选择与流水线

在 ARMv8.2 上，可以依赖以下指令：

- `vld1q_s8` / `vld1q_u8`：加载 int8 / uint8 向量；
- `vqtbl1q_s8`：从 16 个表项中按索引查 16 个结果；
- `vshrq_n_u8` / `vandq_u8`：对索引字节做移位和掩码，提取 nibble 与高位 opcode；
- `vaddq_s16` / `vaddq_s32` / `vaddvq_s32`：累加与水平归约；
- `vrev64q_s8` / `tbl`：对 lane 做交换；
- `vbsl` / `veor` / `vnegq_s8`：按 factor 进行符号翻转。

流水线建议：

- 每个循环内同时维护两套累加器（`acc_*0`、`acc_*1`），交替更新，以增加指令级并行度。
- 在处理当前块时，预取后续块的权重 / 激活 / 索引。
- 控制块大小，使 LUT 表与激活 pack 可以长期驻留在 L1 cache 中。

### 8.5 多线程切分策略

针对常见的 matvec / 小批 matmul，推荐的多线程切分策略为：

- matvec（`M×K · K×1`）：
  - 按行分片（row‑blocking）：将输出行区间切分成若干块，每个线程处理一部分行；
  - 激活 pack 与 LUT 仅依赖单列激活，可由所有线程只读共享。
- 小批 matmul（`M×K · K×N`，N 较小）：
  - 若 N 较小（如 1–4），仍优先按行分片，N 方向在内核中展开；
  - 每列激活各自构建一份 pack + LUT，所有线程只读共享；
  - 工作区大小按列数 N 计算，与线程数无关。

这种策略兼顾 cache 友好与实现简单性：权重和索引缓冲按行分段访问，激活侧数据与 LUT 在所有线程间共享，不需要为每个线程复制大块工作区。

---

## 9. 经验教训与反模式

在以往的 2‑weight / 3‑weight LUT 尝试中，出现过若干性能和复杂度问题，本方案在设计时明确规避：

1. **过度复杂的轴/符号拆分**
   - 将权重拆成「axis（实/虚）」和「sign（正/负）」两个 bit‑stream，再在 NEON 内核中进行大量 bit 操作和掩码展开，会带来大量非线性开销和寄存器压力。
   - 本设计采用「变换头 + 小表 + 复数变换因子」方式，在权重转换阶段完成复杂映射，内核只保留简单变换。

2. **庞大的 per‑块 LUT 表**
   - 为每个 3‑weight 组构建完整 64 项、每项 4 通道的 LUT 会极大增加工作区，破坏 cache 局部性。
   - 本设计仅使用两个 8 行的小表，依赖变换头与 canonical 映射重用表项，避免 per‑块膨胀。

3. **构建阶段重复读写大缓冲**
   - 先将权重解码到多个临时 buffer，再在内核重新读取，会导致访存次数成倍增加。
   - 本设计要求在权重转换阶段直接生成索引和缩放数据，不写临时复数权重缓冲。

4. **内核中频繁解压索引与重排激活**
   - 在内核运行中大量执行「bit 拼接 + 解压 + 多次 `vqtbl1`」会击穿吞吐。
   - 本设计要求激活在预处理阶段按内核友好的形式一次性打包完成，内核不再做复杂重排。

5. **工作区尺寸估算不准确**
   - 错误的 wsize 估算可能导致缓冲区重叠、溢出或对齐问题。
   - 本设计在架构层面明确：工作区大小 = 激活 pack + LUT 表 + LUT 缩放，各项都有明确公式，并按 64 字节对齐。

通过这些约束，本方案尽量避免历史坑点，使 3‑weight LUT 路径的复杂度控制在可维护范围内。

此外，本设计在测试与评估阶段推荐关注以下几个维度：

- 访存与 cache：
  - 每次 matvec/matmul 的总读写字节数（权重、索引、激活 pack、LUT、输出）；
  - L1/L2 miss rate 与带宽使用情况。
- 算术强度与指令构成：
  - 每输出一个复数结果所需的 `vqtbl1q` 次数、`vaddq_*` 次数；
  - 与 2‑weight / 旧 3‑weight 路径相比，是否减少了 per‑lane bit 操作与冗余查表。
- 寄存器压力与编译器行为：
  - 是否出现大量 NEON 寄存器 spill/load；
  - 是否能在 O3 下自动获得合理的循环展开与指令调度。

这些指标有助于判断 3‑weight LUT 的收益主要来自「减少访存」还是「提高算术强度」，并指导后续微调。

---

## 10. 实现步骤与里程碑

为了从零实现完整的 3‑weight LUT 路径，建议按以下步骤推进：

### 10.1 步骤一：规格验证与离线枚举

- 使用脚本（C/ Python 均可）穷举所有 3×2‑bit 组合：
  - 对每个 `(c0, c1, c2)` 和一组测试激活 `(x0, x1, x2)`，计算标量真值。
  - 设计 6bit 索引构造规则，验证通过小表 + 变换头可以覆盖所有情况。
- 输出：
  - `canonical_idx[64]` 与 `factor[64]` 表；
  - LUT 表中对应各行的基准复数值草案。

#### 10.1.1 当前实现记录（离线枚举脚本）

- 脚本位置：`scripts/ifairy_3w_lut_enum.py`。从仓库根目录运行：

  ```bash
  python scripts/ifairy_3w_lut_enum.py
  ```

- 权重语义与 6bit 索引：
  - 权重编码沿用 1.2 的约定：`0 → −1`，`1 → +1`，`2 → −i`，`3 → +i`。
  - 原始 6bit 索引实现为：

    ```text
    idx_raw = (c0 << 4) | (c1 << 2) | c2   // ∈ [0,63]
    ```

- 规范分解与 `canonical_idx[64]`：
  - 对每个三元组 `(c0, c1, c2)`，对应权重 `(w0, w1, w2)` 满足：

    ```text
    (w0, w1, w2) = factor · (1, u1, u2)
    ```

  - 当前实现选择 `factor = w0`，并在「i 的指数」空间中工作：

    ```text
    weight(c) = i^e,  e ∈ {0,1,2,3} → {1, i, −1, −i}
    e0 = exp(w0), e1 = exp(w1), e2 = exp(w2)
    factor_exp = e0
    u1_exp     = (e1 − e0) mod 4
    u2_exp     = (e2 − e0) mod 4
    ```

  - 16 个规范模式由 `(u1_exp, u2_exp)` 唯一确定，`canonical_idx` 采用纯编码约定：

    ```text
    idx' = (u1_exp << 2) | u2_exp    // idx' ∈ [0,15]
    canonical_idx[idx_raw] = idx'
    ```

- `factor[64]` 的编码：
  - 脚本输出的 `factor_exp[64]` 为指数数组：

    ```text
    factor[idx_raw] = e ∈ {0,1,2,3}，满足  i^e = factor
    ```

  - 映射关系在脚本中显式给出：`0 → 1`，`1 → i`，`2 → −1`，`3 → −i`。后续在 6.5/10.2 中再将该指数编码映射到 `(do_swap, neg_real, neg_imag)` 三个比特。

- 测试激活与 LUT 基准值：
  - 为了验证规范分解的正确性，脚本内部固定一组测试激活：

    ```text
    x0 =  1 + 2i
    x1 =  3 − 4i
    x2 = −5 + 6i
    ```

  - 对每个三元组 `(c0, c1, c2)`，脚本同时计算：

    ```text
    S_direct = w0·x0 + w1·x1 + w2·x2

    S_base   = 1·x0 + u1·x1 + u2·x2
    S_via    = factor · S_base
    ```

    并断言 `S_direct == S_via`（浮点容差内）。对同一 `idx'`，所有出现的 `S_base` 也必须一致。

  - 脚本最终生成：
    - `ifairy_canonical_idx[64]` 与 `ifairy_factor_exp[64]` 的 C 风格数组（仅离线使用）；
    - `ifairy_lut_baseline[16]`，按 `idx' ∈ [0,15]` 给出每个规范模式在上述测试激活下的基准复数贡献 `S_base`。

- 目前导出的 `canonical_idx` / `factor_exp` 示例（便于查表实现时对照；实际 C 代码可按需复制到实现文件中）：

  ```c
  // canonical_idx[64] : raw 6-bit index -> 4-bit canonical index (idx' ∈ [0,15])
  // factor_exp[64]    : raw 6-bit index -> exponent e, where i^e = factor
  //                    (e ∈ {0,1,2,3} corresponds to {1, i, -1, -i})

  static const uint8_t ifairy_canonical_idx[64] = {
       0,  2,  1,  3,  8, 10,  9, 11,  // [ 0.. 7]
       4,  6,  5,  7, 12, 14, 13, 15,  // [ 8..15]
      10,  8, 11,  9,  2,  0,  3,  1,  // [16..23]
      14, 12, 15, 13,  6,  4,  7,  5,  // [24..31]
      15, 13, 12, 14,  7,  5,  4,  6,  // [32..39]
       3,  1,  0,  2, 11,  9,  8, 10,  // [40..47]
       5,  7,  6,  4, 13, 15, 14, 12,  // [48..55]
       9, 11, 10,  8,  1,  3,  2,  0,  // [56..63]
  };

  static const uint8_t ifairy_factor_exp[64] = {
       2,  2,  2,  2,  2,  2,  2,  2,  // [ 0.. 7] -> (-1, -1, -1, -1, -1, -1, -1, -1)
       2,  2,  2,  2,  2,  2,  2,  2,  // [ 8..15] -> (-1, -1, -1, -1, -1, -1, -1, -1)
       0,  0,  0,  0,  0,  0,  0,  0,  // [16..23] -> ( 1,  1,  1,  1,  1,  1,  1,  1)
       0,  0,  0,  0,  0,  0,  0,  0,  // [24..31] -> ( 1,  1,  1,  1,  1,  1,  1,  1)
       3,  3,  3,  3,  3,  3,  3,  3,  // [32..39] -> (-i, -i, -i, -i, -i, -i, -i, -i)
       3,  3,  3,  3,  3,  3,  3,  3,  // [40..47] -> (-i, -i, -i, -i, -i, -i, -i, -i)
       1,  1,  1,  1,  1,  1,  1,  1,  // [48..55] -> ( i,  i,  i,  i,  i,  i,  i,  i)
       1,  1,  1,  1,  1,  1,  1,  1,  // [56..63] -> ( i,  i,  i,  i,  i,  i,  i,  i)
  };
  ```

> 注：以上指数编码与规范索引布局是当前实现的具体选择，用于后续 10.2/10.3 的索引缓冲与 LUT 结构设计。如果后续实验发现更合适的编码（例如为了更方便映射到 NEON 掩码），可以在保持文档与脚本同步的前提下调整。

### 10.2 步骤二：权重转换与索引缓冲

- 为每个权重张量实现一个权重转换模块：
  - 遍历每行、每 3 个权重，对照 `canonical_idx` / `factor` 表构造 6bit 索引与变换因子。
  - 将二者打包到索引缓冲（例如一个字节存 idx'，另一个字节存 factor，或压缩到 1 字节）。使用 6.5 中定义的单字节格式存储 idx' 与复数变换 opcode。
  - 同时生成权重缩放数组。
- 确保索引缓冲对齐、与原始权重一一对应，并记录必要的 tile 参数（例如行块大小、列块大小）。

> **实现进展（已落地代码）**  
> - 常量表：`ggml/src/ggml-quants.c` 中新增 `ifairy_canonical_idx[64]` 与 `ifairy_factor_exp[64]`，对应 4.2 的规范映射。  
> - opcode 映射：指数 → `(swap, neg_real, neg_imag)` 编码，规则与 6.5 保持一致：`1→0x00`，`i→0xA0`，`-1→0x60`，`-i→0xC0`。  
> - 索引生成 API：`ggml_ifairy_3w_encode()`（声明于 `ggml/src/ggml-quants.h`）直接从压缩 2‑bit 权重生成三权重索引缓冲，k 按 3 向上取整并在尾部用 code=0 做 padding；`ggml_ifairy_3w_get_index_info()` / `ggml_ifairy_3w_index_buffer_size(_aligned64)` 提供布局与容量计算。  
> - 单元校验：`tests/test-ifairy.cpp` 增加 `test_ifairy_lut_index`，验证前 3 组编码 `[0x69, 0xA0, 0x0D]` 及 padding 行为（K=256 → groups=86，末尾填 0x60），作为 10.2 的回归守护。  
> - 下一步：在权重加载/转换流程中调用上述 API 写入权重侧索引缓冲，并在 NEON 内核消费该缓冲。

### 10.3 步骤三：激活预处理与 LUT 构造

- 实现激活量化与 pack：
  - 先统计 max 值、求缩放因子；
  - 再将激活量化为 int8，并按内核需要的 layout 打包。
- 实现三权重 LUT 构造：
  - 对每个激活 pack、每个三权重组 `g`，构建一对长度为 16 的扁平表 `lut_real_g[16]` / `lut_imag_g[16]`，对应 16 个规范模式；
  - 使用 int16/int32 累加后饱和裁剪为 int8，并尽量使用 NEON（`vaddq_s16` / `vsubq_s16` / `vrev64q_s16` / `vqmovn_s16` / `vst1q_s8`）实现批量构表，避免标量复数乘和逐 byte 写入。

> 当前进展（标量 reference）：`ggml/src/ggml-ifairy-lut.cpp` 已实现标量版 LUT 构造与 qgemm，并接入 mul_mat 路由。构表使用 ifairy_q16 激活（或 F32 bf16-pair，经临时量化），按三权重组和 16 个规范模式累加后饱和到 int8；激活缩放取第一块 fp16 缩放。qgemm 按索引 opcode（swap/neg）应用 LUT 值并累加为 int32，再乘激活缩放与权重缩放（首块 d_real/d_imag）输出 float/bf16。

### 10.4 步骤四：标量参考内核

- 实现一个纯标量版本的 3‑weight LUT matvec：
  - 直接读取索引与 LUT 表，按 factor 应用复数变换；
  - 使用简单的标量运算完成累加与缩放；
  - 真值来源于「原始 2‑bit 权重 + 浮点激活」的逐元素复数乘累加（不经过 LUT），标量 LUT 结果应在量化误差范围内逼近该真值；
  - NEON 内核则应与标量 LUT 结果在整数域一致（仅允许浮点收尾时 1 LSB 级别差异）。

> 当前进展：`ggml/src/ggml-ifairy-lut.cpp` 提供标量参考预处理与 qgemm：\n> - 预处理：读取 ifairy_q16 激活（或 F32 bf16-pair 临时量化），按 K3 分组与 16 个规范模式构建 `lut_real/lut_imag`，int8 饱和；激活缩放填充为第一块的 fp16 缩放。 \n> - qgemm：按索引 opcode（swap/neg）应用 LUT，int32 累加后乘激活缩放与权重缩放（取第一块权重的 d_real/d_imag）输出 float/bf16（mul_mat 路由使用 bf16 packed）。 \n> - 已接入 mul_mat 路由，标量为默认回退；缩放仍为 per-tensor，后续与权重侧 block scale 对齐。***

### 10.5 步骤五：NEON 内核与优化

- 在 NEON + DOTPROD 平台上实现 3‑weight LUT 内核：
  - 使用 `vqtbl1q_s8` 查表；
  - 使用 `vaddq_s16` / `vaddq_s32` 以及双路累加流水线；
  - 使用 `__builtin_prefetch` 或类似机制预取后续块。
- 逐步优化：
  - 微调块大小（例如每次处理 32 或 64 个 3‑weight 组）；
  - 展开循环以减少分支；
  - 调整 LUT 表和激活 pack 的布局，使其更利于 cache 与寄存器复用。

> 当前：mul_mat 路由已接通标量 LUT，`transform_tensor` 生成索引 shadow（挂 `extra`），`get_wsize` 上报 per-thread LUT+scale/可选激活量化缓冲，mul_mat 按行拆分并用 workbuf 预处理后 qgemm，输出按 bf16-packed（每个 float 容器存实/虚 bf16）。已修复 ifairy RMSNorm 以 bf16-pair 方式解包累加/重打包；二元算子仅对 ifairy_add/mul 放宽 NaN 检查；`ggml_abort` 增加 last-node 诊断打印；mul_mat 每线程 workbuf slice 现额外包含 `N*sizeof(float)` 的行缓冲，qgemm 直接写入后依列主序偏移（`col*nb1 + row*nb0`）回填 dst，彻底消除了 `_platform_memmove` 崩溃；构建阶段，`GGML_IFAIRY_ARM_LUT=ON` 会强制将 Metal/CUDA/HIP/MUSA/Vulkan/OpenCL/SYCL/WebGPU/zDNN 等非 CPU 后端设置为 OFF 并提示 warning，确保 LUT 调试仅在 CPU 路径上运行。`GGML_IFAIRY_LUT=1` 已可跑通 llama-cli。待改进：索引释放仍依赖全局 `ggml_ifairy_lut_free`，未随 tensor 回收；缩放仍为 per-tensor；NEON 核心未就绪；GGUF ifairy 类型仍触发 loader “unknown type” 警告；需继续验证其他 ifairy 专用算子的 bf16 打包一致性。***

### 10.6 步骤六：测试与基准

- 单元测试：
  - 3‑weight LUT vs 逐元素点积；
  - 标量 LUT vs NEON LUT；
  - 多种矩阵形状与随机种子覆盖。
  - 当前：`tests/test-ifairy.cpp::test_ifairy_lut_scalar_matmul` 覆盖标量 encode + preprocess + qgemm，针对 K=QK_K、N=2 的场景，将 LUT 输出与按量化值解码的直接复数点积逐元素比对，阈值 1e‑3；`./build/bin/test-ifairy` 通过；`GGML_IFAIRY_LUT=1 ./build/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv` 可正常生成输出（仍有 ifairy 类型 loader 警告）。
- 性能测试：
  - 记录典型模型与形状下的 token/s 或 ms/token；
  - 对比：非 LUT 标准vecdot实现；
- 根据结果调整是否默认启用 3‑weight LUT。

### 10.7 Debug 与提速路线（从“能跑”到“正确且快”）

> 现状：在 Debug build 下，Time Profiler 显示热点集中于 `ggml_ifairy_lut_qgemm`（~85%）与 `ggml_ifairy_factor_to_opcode`（~7%）；同时 `GGML_IFAIRY_LUT=1` 可能出现“输出乱码/不可读”现象。此处给出一条可由 Codex 全流程执行的定位与提速路线。

#### A) 先把性能测试从 Debug 拉回 Release

- Debug 会放大函数调用与分支成本，导致 profiler 结论失真。必须先建 Release 目录用于性能结论：
  ```bash
  cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release -DGGML_IFAIRY_ARM_LUT=ON
  cmake --build build-rel --target llama-cli test-ifairy -j 8
  ```

#### B) 正确性优先：为 LUT 路径增加“抽样校验”与可复现对照

1) **对照复现（LUT=0 vs LUT=1）**
```bash
GGML_IFAIRY_LUT=0 build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv
GGML_IFAIRY_LUT=1 build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv
```
- 若 LUT=1 输出乱码，优先判定为 LUT 数学/缩放与 ifairy 参考实现不一致，而不是 tokenizer 问题。

2) **建议实现：`GGML_IFAIRY_LUT_VALIDATE=1`**
- 在 LUT mul_mat 分支中对“前 N 个 ifairy mul_mat”做抽样校验（比如 op 计数 < 8）。
- 对抽样的少量行（例如 row=0、row=M/2）用现有 ifairy vecdot（`ggml_vec_dot_ifairy_q16_K`）计算参考输出，与 LUT 输出比较 max diff；超阈值直接 abort 并打印 src/dst 形状与 nb。
- 目的：把“乱码”快速定位到某一层某一 matmul，而不是凭感觉猜。

3) **缩放一致性检查（关键）**
- ifairy 的激活量化容器 `block_ifairy_q16` 是 per-block `d_real/d_imag`，而不是 per-tensor；LUT 路径若只取“第一块缩放”会造成系统性数值错误，最终表现为输出乱码。
- 修复优先级：先让 LUT 数学对齐 `ggml_vec_dot_ifairy_q16_K` 的缩放位置与公式，再做 NEON 提速。

#### C) 解释热点：为什么 `ggml_ifairy_factor_to_opcode` 会占 7%

- 该函数应当只在“索引/权重转换”阶段使用。如果 profiler 显示占比明显，通常意味着：
  - `transform_tensor` 在推理过程中被重复触发（graph 重建/weight view/extra 丢失）；
  - 或者索引生成没有被缓存/复用。
- 建议（工程修复）：把 transform 前移到模型加载阶段，或构建全局缓存（key=`tensor->data` 指针）保证同一权重只转换一次；后续改为 shadow tensor 或可追踪 buffer 管理生命周期。

#### D) 提速路线（按收益排序）

1) **Release + 关闭日志**：默认不开 `GGML_IFAIRY_LUT_DEBUG`，避免 can_mul_mat 的高频日志污染性能。
2) **标量止血**（NEON 之前）：
   - 为 `n==1` 的 matvec 写专用 qgemm（去掉 N/stride 分支与多余 memcpy）。
   - `code` byte 解码改为 `decode_table[256]`，减少 bit/branch。
   - 索引生成缓存：消灭推理期重复 transform。
3) **NEON 关键提速（最终目标）**：
   - 按 BitNet TL1 思路实现 BM=16：对同一 group，LUT 常驻寄存器，用 vqtbl 对 **16 行并行查表**，把当前 85% hotpath 的标量循环改为 SIMD。
   - 优先实现 `n==1`，再扩展 `n==2/4`。
4) **缩放与分组布局优化**：
   - 尽量让 3‑weight 分组不跨 `QK_K=256` block（按 block 内分组并 padding），从结构上匹配 per-block 缩放并减少 LUT 复杂度。

#### E) CLI Profiling（Codex 可直接跑）

```bash
mkdir -p tmp/traces
GGML_IFAIRY_LUT=1 xcrun xctrace record \
  --template "Time Profiler" \
  --time-limit 15s \
  --output tmp/traces/ifairy_lut_timeprofile.trace \
  --launch -- build-rel/bin/llama-cli \
    -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv
```

> 以上步骤完成后，再进入 10.5 的 NEON 内核实现；否则在数学/缩放未对齐前，提速只会把“乱码”跑得更快。***

#### F) 输出“乱码”专项修正计划

1) **严格校验模式（短期确保正确）**  
   - 新增 env `GGML_IFAIRY_LUT_VALIDATE_STRICT=1`：在 LUT 分支内，如发现三权重组跨 `QK_K` block，则退化为逐权重计算（每权重单独应用激活缩放和权重量化缩放，复数乘累加），仍在 LUT 路径，不回退 vec_dot；仅用于验证输出可读性，默认关闭。
   - 可与抽样校验共用：只对前 N 个 ifairy mul_mat、少量行做严格模式，以减少性能冲击。

2) **分组与缩放重排（正式修复）**  
   - 方案 A（推荐）：索引编码按 block 切分，每个 `QK_K=256` 内 `(QK_K/3)` 组，尾部 padding；预处理/LUT/qgemm 也按 block 切片。这样每组权重与激活缩放都属于同一 block，无需平均近似。  
   - 方案 B（过渡）：扩展 scale buffer 为“每组 3 个权重的缩放”（6 float），即便组跨 block 也能精确应用缩放；在 A 落地后回收为 A。

3) **落地顺序**  
   - 先实现严格校验模式，确认文本恢复可读，并记录具体层/组是否跨 block。  
   - 若严格模式可读，则优先推进方案 A；如改索引风险高，先落方案 B 保证正确输出，再迭代到 A。  
   - 保持 Release 构建，默认关闭 debug 输出，避免性能噪声。***

### G) 当前进展（方案 A 落地后的状态补充）
- 索引/预处理/qgemm 已统一使用方案 A：每个 `QK_K=256` block 丢弃末尾 1 个权重（255=85*3），`groups_per_block=85`。`ggml_ifairy_3w_encode`、wsize、预处理以及标量 qgemm 均按新分组计算。
- 严格校验模式：`GGML_IFAIRY_LUT_VALIDATE_STRICT=1` 时 qgemm 直接按量化权重 + 激活重构累加用于校验；`tests/test-ifairy` 在 strict/非 strict 下均 0 diff 通过。
- 类型注册：`llama-model-loader` 已加入 `LLAMA_FTYPE_MOSTLY_IFAIRY` 映射，加载时不再报警 “unknown type ifairy”。
- 问题仍存：`GGML_IFAIRY_LUT=1`（即便 strict=1）运行 `llama-cli` 输出仍为乱码，需继续排查（可能与方案 A 丢弃 block 尾元素或缩放/布局差异有关）；首要目标是让 LUT 路径输出可读文本，再推进 NEON 与性能优化。

---

## 11. 未来扩展方向

在 3‑weight LUT 稳定后，可以考虑以下扩展：

- **更高阶组合**：例如 4‑weight 或多组组合的混合 LUT，需要重新设计索引与表结构。
- **自适应量化策略**：针对不同层、不同通道，使用不同的 LUT 粒度和缩放策略。
- **自动生成内核**：通过脚本自动生成特定矩阵形状的 NEON 内核，以减少手写 SIMD 代码量。
- **跨平台迁移**：将本方案的「索引 + 小表 + 变换因子」思想迁移到 x86 AVX2/AVX‑512 等平台。

本设计文档仅给出 iFairy 2‑bit 复数模型在 ARM NEON 上实现三权重 LUT 的结构化方案与关键算法约定，为后续具体实现与优化提供统一的参考框架。实现过程中若有新的经验与坑点，应及时回填至本文档，以保持设计与现实的同步。
