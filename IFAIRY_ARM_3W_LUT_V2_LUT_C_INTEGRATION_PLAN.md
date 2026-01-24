# iFairy ARM 3W LUT (V2) — 接入 `lut_c/` 算法方案

Status: Draft (2026-01-24)

本文件描述如何把 `lut_c/` 中的 LUT 算法以“backend”的形式集成到 ggml iFairy LUT 路径中，并以 V2 重构方案为前置（见 `IFAIRY_ARM_3W_LUT_V2_REFACTOR_PLAN.md`）。

目标：在不破坏现有 merged64 fastest baseline 的前提下，提供一个可 A/B 的 `lut_c` backend，并力求在目标 workload 上获得**一致或更优**的 tok/s。

---

## 1. `lut_c/` 关键要素提取（可迁移点）

### 1.1 数据分组与约束
- `QK_K = 256`
- per block 分为 `groups_per_block = (QK_K + 2) / 3 = 86` 个 3-weight group（包含 tail group：`(255, pad, pad)`，不跨 block）

### 1.2 16-pattern LUT（从 64 pattern 压缩到 16）
`lut_c` 的核心技巧是把 64 种 triplet pattern 映射为 16 个 canonical index + 2 个 flag 位：

- 先把原始 6-bit pattern（`pat=c0|(c1<<2)|(c2<<4)`）映射到 **0..15**：
  - `idx16 = three_vals2index_uint8[pat]`（`lut_c/lut/mul_mat_with_lut.c` 有现成表）
- 再把 `pat` 的高 2bit 搬到输出 byte 的 bit6/7：
  - `flags = ((pat << 2) & 0xC0)`
- 最终 per-group/per-row 的 packed byte：
  - `code = idx16 | flags`

在 kernel 中：
- `index = code & 0x3f`（实际应落在 0..15，满足 `vqtbl1q_*` 的 16-entry 查表）
- `flag0 = (code & 0x40)`, `flag1 = (code & 0x80)` 用于在 4 个 base 表上做符号/排列选择

### 1.3 Q8 激活量化的 scale 常数：`127/3 ≈ 42.6`
`lut_c` 在生成 LUT 时对每个 block 分别计算：
- `scale_real = max(|x_real|) / 42.6`
- `scale_imag = max(|x_imag|) / 42.6`

原因：LUT 项本质是最多 **3 个**（量化后的）激活分量的线性组合；把每个分量限制在 `[-42,42]`，可避免 `3*val` 溢出 int8（`3*42=126`）。

这点需要在 ggml 集成中保持一致，否则会出现溢出/饱和导致误差或性能波动。

### 1.4 M 维度 16 行 tile + NEON TBL
`lut_c` 的 qgemm 形式（N==1）是：
- 每次处理 16 行输出（`row += 16`）
- 每个 group：加载 16 个权重 code（1 byte/row-lane），对 `int8x16x4` LUT 做 `vqtbl1q_s8` 查表
- 累加到 `int16`（避免 int8 溢出），最后做 scale 反量化和写回

这一结构对 decode（N==1）非常友好：重用 LUT、对 M 维 vectorize、减少标量 loop。

---

## 2. 与现有 ggml merged64 的关键差异

| 维度 | merged64（现有 baseline） | lut_c（拟接入） |
|---|---|---|
| weight 侧 transform | per-row `indexes`（1B/group） | per-16rows pack（每 group 16B，额外存 d_real/d_imag[16]） |
| LUT layout | 64 pattern × 4ch × int8（256B/group） | 16-entry vec table × 4ch（`int8x16x4`，64B/group） |
| kernel 向量化方向 | 更偏 decode-first（按 row 标量/小向量累加） | M 方向 16 行 SIMD + TBL |
| scale | act scales 来自 `block_ifairy_q16` | 明确使用 `max/42.6`（每 block real/imag） |

因此，在 ggml 中接入 lut_c 不能仅复用当前 `indexes`，必须为 lut_c backend 提供：
- 独立的 weight transform（生成 per-16rows packed weights）
- 独立的 preprocess（按 42.6 规则生成 `int8x16x4` LUT）
- 独立的 qgemm（16 行 tile）

---

## 3. V2 backend 形式的集成设计

### 3.1 新增 backend：`lut_c16x3`（建议命名）
在 V2 backend 注册表中增加：
- `merged64`（现有 baseline）
- `lut_c16x3`（新）

运行时选择：
- 默认：`GGML_IFAIRY_LUT_IMPL=auto` → `merged64`（保证不回退）
- A/B：`GGML_IFAIRY_LUT_IMPL=lut_c` 强制走新 backend

### 3.2 权重 transform（lut_c backend）
目标：把 `GGML_TYPE_IFAIRY` 的 `block_ifairy` 权重转成类似 `lut_c` 的预排布：
- 主体：`qs[groups_per_block][16]`（每 group 对应 16 行 lane 的 packed code）
- 额外：`d_real[16]`, `d_imag[16]`（每行的权重量化 scale）

建议实现方式：
- 增加 `ggml_ifairy_3w_encode_lutc16()`：输入 `block_ifairy`，输出 per-row 的 `code`（1B/group，含 idx16+flags）。该编码可直接复用 `ggml_ifairy_3w_encode()` 的分组遍历方式，只是把 `pat` 变换为 `code`：
  - `pat = c0|(c1<<2)|(c2<<4)`
  - `code = map64to16[pat] | ((pat<<2)&0xC0)`
- 然后将 per-row 的 `code` 做一次转置/pack：把连续 16 行的同一 group 打包成 16-byte 向量，放到 `qs[group][lane]`。

缓存策略：
- 复用当前 `tensor->extra` 缓存机制（参考 `ggml-ifairy-lut-transform.cpp` 的 index cache），把 lut_c backend 的 packed buffer 缓存为 weights usage 的 `ggml_backend_buffer_t`。

### 3.3 preprocess（lut_c backend）
输入 act 支持两种：
- `GGML_TYPE_F32`（bf16-pair complex container）→ 量化为 Q8（按 42.6 规则）
- `GGML_TYPE_IFAIRY_Q16` → 先转成等价的 Q8 表达（或直接生成 LUT）

输出：
- `lut`：按列、按 block、按 group 存储 `int8x16x4`（逻辑上 4×16 bytes）
- `scales`：每 block 存 `scale_real/scale_imag`（float）

建议复用与对齐点：
- 复用现有 act 读取/stride 约定（`act_stride`）
- 复用 “tail group 不跨 block” 规则（保证与 `ggml_ifairy_3w_*` 的分组一致）

### 3.4 qgemm（lut_c backend）
主路径聚焦 decode（`n==1`）：
- 外层：`for row in [0..m) step 16`
- 中层：遍历 blocks / groups
- 内层：加载 `qs[group][16]` → `vqtbl1q` 查表 → 累加到 4 路 `int16` accumulator（对应 ac/ad/bc/bd 或等价组合）
- 尾部：应用 `scale_act` 与 `scale_w` 做反量化，写回 dst（支持 `pack_bf16` 与 `add`）

边界处理：
- `m` 非 16 对齐：保留标量/小向量 tail（类似 `lut_c` 里的 `__M__ALIGNED_16/__M__ALIGNED_8` 分支策略）
- `k` 保持 `K%256==0`（与现状一致）

---

## 4. 正确性策略（接入阶段必须）

建议在 `test-ifairy` 增加“backend A/B”模式（只用于测试/开发）：
- 固定小尺寸与大尺寸用例（覆盖 tail、对齐、不同 thread）
- 对比：
  - `merged64` vs reference
  - `lut_c16x3` vs reference
  - （可选）`lut_c16x3` vs `merged64`（便于排查误差来源）

并要求继续满足全局语义：`w * conj(x)`。

---

## 5. 性能策略（接入阶段必须）

### 5.1 保底策略：默认不切换
`GGML_IFAIRY_LUT_IMPL=auto` 默认仍选 `merged64`，确保不回退。

### 5.2 收集数据：以 V2_STATUS 统一记录
每次对 lut_c backend 做性能 claim，必须在 `IFAIRY_ARM_3W_LUT_V2_STATUS.md` 记录：
- 机器型号、OS、编译参数
- 完整 bench 命令行 + env
- 原始输出（tok/s）
- 如需定位瓶颈：补充 xctrace CPU Counters 采样（模板：`test.tracetemplate`，方案：`IFAIRY_ARM_3W_LUT_V2_XCTRACE_CPU_COUNTERS.md`）

---

## 6. 集成步骤建议（按风险分级）

1) **只集成 weight transform + preprocess**（不接入 qgemm）
- 先把 packed weights 与 `int8x16x4` LUT 在内存布局上跑通，并通过单元测试验证 LUT 构造正确

2) **接入 N==1 decode kernel**（优先价值）
- 以 `llama-bench` decode workload 做 A/B

3) **扩展到 N>1 或保留 fallback**
- 如果 lut_c 的设计主要针对 N==1，可明确在 backend `can_mul_mat()` 中限制为 `n==1`
- N>1 继续走 merged64

---

## 7. 需要从 lut_c 抽取/迁移的最小代码资产

建议只迁移“算法必要部分”，避免把 `lut_c` 当成新库：
- `three_vals2index_uint8[64]` 映射表（或等价生成逻辑）
- LUT 生成公式（ac/bd/ad/bc 的 16-entry 构造）
- 16 行 tile kernel（TBL 查表 + int16 累加 + scale 写回）

其余（`lut_c/main.c`、测试数据脚本）不进入 ggml 生产路径，可仅作为开发对照。
