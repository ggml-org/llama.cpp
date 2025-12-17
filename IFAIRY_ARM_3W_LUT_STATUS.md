# iFairy ARM 3‑Weight LUT · 现状与后续工作（NEON 标量混合版）

本文记录当前 `GGML_IFAIRY_ARM_LUT`（CPU-only）下 iFairy 3-weight LUT 的代码现状（含 NEON 加速实现）、可复现的 tok/s 记录、以及下一步工作列表。接口/路由约定见 `IFAIRY_ARM_3W_LUT_API_PLAN.md`，算法与数据结构见 `IFAIRY_ARM_3W_LUT_DESIGN.md`。

## 0. 快速使用（建议默认）

- 前提：构建时启用了 `GGML_IFAIRY_ARM_LUT`（`ggml/CMakeLists.txt` 会在 configure 阶段强制关闭 Metal/CUDA/HIP/MUSA/Vulkan/OpenCL/SYCL/WebGPU/zDNN 等加速后端，以保证 CPU-only）。
- 推荐二进制：`./build-rel/bin/llama-cli`（避免误用旧的 `./build/bin/llama-cli` 导致输出异常，见 3.1）
- 推荐扫参脚本：`bash scripts/ifairy_lut_sweep.sh`（固定 seed/prompt，输出按 tok/s 排序）
- LUT 表布局默认走 `legacy`（更稳；`compact` 在部分设备/形状上更快），如需测试紧凑表：`GGML_IFAIRY_LUT_LAYOUT=compact`（见 1.1 / 0.1 记录）
- `BK/BM/FULLACC` 调参在不同形状/版本上波动较大：以 sweep 输出为准，不建议凭经验固定写死
- 每次修改 LUT 相关代码后，先做一次 `llama-cli` sanity check（固定 seed/prompt）确认不输出 gibberish（见 3.1）

**常用环境变量（LUT 路径）**

- `GGML_IFAIRY_LUT=0/1`：禁用/启用 LUT（默认启用）
- `GGML_IFAIRY_LUT_LAYOUT=legacy|compact`：LUT 表布局选择（默认 `legacy`）
- `GGML_IFAIRY_LUT_BK_BLOCKS=<int>`：K 维按 `QK_K=256` 的 block 做 tiling（0=禁用）
- `GGML_IFAIRY_LUT_BM=<int>`：M 维行块大小（仅 tiling 时生效）
- `GGML_IFAIRY_LUT_FULLACC=0/1`：tiled 下启用共享大累加器，减少重复 `preprocess + barrier`
- `GGML_IFAIRY_LUT_VALIDATE_STRICT=0/1`：严格对照 reference（用于验证，不用于性能跑分）
- `GGML_IFAIRY_LUT_DEBUG=0/1`：路由/形状诊断（默认关闭；跑分时不要开）
- `GGML_IFAIRY_LUT_PREFETCH=0/1`：控制 LUT 路径内的 prefetch（默认启用；设为 `0` 用于 profile/sweep 对照；覆盖 legacy/compact 的 `qgemm_ex/accum4_ex`）

## 0.0 当前共识（按优先级）

> 基于 `CODE_REVIEW_lwt_3_LUT.md` 的结论：性能收益已证明，但接下来优先把可维护性与健壮性补齐，避免后续优化建立在不稳的地基上。

- P0：内存/生命周期（减少 `new/delete` + 全局容器；补齐 size/overflow/bounds 检查，避免 silent failure）
- P0：线程安全（明确并发模型，缩小锁粒度，补并发/压力测试）
- P1：可维护性重构（拆分 `ggml/src/ggml-ifairy-lut.cpp`，减少 legacy/compact 重复代码）
- P1：错误处理一致性（统一 `return false`/`GGML_ASSERT`/日志策略）
- P2：测试与回归（维度边界、分配失败、misaligned buffer、并发 transform；以及 decode/prefill 形状的性能基线）

## 0.1 tok/s 记录（更新本文档时必填）

> 约定：每次修改/更新本文件（`IFAIRY_ARM_3W_LUT_STATUS.md`）都在这里追加一条 tok/s 记录，避免“写了很多优化但没有可复现数字”。

**基准命令（固定 prompt/seed/thread）**

> 注：为避免 `n_ctx` 默认值变动带来的不可比波动，后续记录统一显式固定 `-c 2048`（与模型训练 ctx 对齐）。

`GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 ./build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -c 2048 --seed 1 -p "I believe life is" -n 256 -no-cnv`

| time (UTC) | git | machine | threads | tokens | env | eval tok/s |
|---|---|---|---:|---:|---|---:|
| 2025-12-16T18:28:00Z | `9b782e0f` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1` | 1.85 |
| 2025-12-16T18:28:00Z | `9b782e0f` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=2 GGML_IFAIRY_LUT_BM=64 GGML_IFAIRY_LUT_FULLACC=1` | 2.58 |
| 2025-12-17T04:49:00Z | `9b782e0f` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_LAYOUT=legacy` | 4.12 |
| 2025-12-17T04:49:00Z | `9b782e0f` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_LAYOUT=compact` | 3.94 |
| 2025-12-17T06:53:24Z | `257c494b` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 7.05 |
| 2025-12-17T06:53:24Z | `257c494b` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 6.88 |
| 2025-12-17T06:59:40Z | `257c494b` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 8.01 |
| 2025-12-17T06:59:40Z | `257c494b` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 8.08 |
| 2025-12-17T08:06:50Z | `6ff807dc` | Apple M4 | 4 | 128 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 9.02 |
| 2025-12-17T08:27:00Z | `38c185d5` | Apple M4 | 4 | 128 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 10.31 |
| 2025-12-17T08:40:52Z | `e8e6c47b` | Apple M4 | 4 | 128 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 18.96 |
| 2025-12-17T08:40:52Z | `e8e6c47b` | Apple M4 | 4 | 128 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 17.56 |
| 2025-12-17T09:05:12Z | `20f90418` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 19.28 |
| 2025-12-17T09:05:12Z | `20f90418` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 21.59 |
| 2025-12-17T13:17:09Z | `0ec52a5a` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 15.39 |
| 2025-12-17T13:17:09Z | `0ec52a5a` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 16.99 |
| 2025-12-17T14:20:20Z | `0aeaa6c9` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 2.71 |
| 2025-12-17T14:20:20Z | `0aeaa6c9` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 4.75 |
| 2025-12-17T14:57:00Z | `0aeaa6c9+dirty` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 4.13 |
| 2025-12-17T14:58:30Z | `0aeaa6c9+dirty` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 3.17 |
| 2025-12-17T17:50:33Z | `a785693e+dirty` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 8.21 |
| 2025-12-17T17:50:33Z | `a785693e+dirty` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 7.10 |
| 2025-12-17T18:53:21Z | `a3296bec` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 5.71 |
| 2025-12-17T18:53:21Z | `a3296bec` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 5.17 |
| 2025-12-17T19:06:29Z | `627dea55` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 8.19 |
| 2025-12-17T19:06:29Z | `627dea55` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 7.16 |
| 2025-12-17T19:06:29Z | `627dea55` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact GGML_IFAIRY_LUT_PREFETCH=0` | 7.17 |
| 2025-12-17T19:23:32Z | `46dcb0cb` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 8.15 |
| 2025-12-17T19:23:32Z | `46dcb0cb` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 7.17 |
| 2025-12-17T19:23:32Z | `46dcb0cb` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact GGML_IFAIRY_LUT_PREFETCH=0` | 7.19 |
| 2025-12-17T19:32:20Z | `2a39f249` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 8.15 |
| 2025-12-17T19:32:20Z | `2a39f249` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 7.15 |

## 0.2 Xcode Profile（以 decode 场景为准）

> 目的：明确“该优化应该打在哪儿”，避免继续做低收益的微调。

**配置（你提供的条件）**

- 环境：`GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact`
- 命令：`./build-rel/bin/llama-cli -m /Users/liweitao/Downloads/Codefield/cpp/llama.cpp/models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 128 -no-cnv`

**热点占比（Xcode 采样结果）**

- `ggml_ifairy_lut_qgemm_ex`：63%
- `ggml_graph_compute_thread`：24%
- `ggml_compute_forward_mul_mat`：6%
- 其他：< 2.5%

**更新（你最新的采样结果）**

- `ggml_ifairy_lut_qgemm_ex`：52%
- `ggml_graph_compute_thread`：30%

**更新（你最新的采样结果 v2）**

- `ggml_ifairy_lut_qgemm_ex`：69%
- `ggml_graph_compute_thread`：12%

**解读**

- 主要瓶颈已非常明确：继续提升 tok/s，优先级应集中在 `ggml_ifairy_lut_qgemm_ex`（降低每次 matmul 的单位成本）
- `ggml_graph_compute_thread` 的占比说明“线程调度/同步/图执行框架开销”也不可忽略；需要减少 barrier/减少 kernel 次数/减少不必要的工作区搬运

## 1. 当前现状（可工作的 LUT 路径：NEON 优先，标量回退）

- 路由位置：`ggml/src/ggml-cpu/ggml-cpu.c` 的 `ggml_compute_forward_mul_mat()` 内，当
  - `src0->type == GGML_TYPE_IFAIRY`
  - `src1->type == GGML_TYPE_F32` 或 `GGML_TYPE_IFAIRY_Q16`
  - `dst->type == GGML_TYPE_F32`
  - `K % QK_K == 0`
  - 且未被 `GGML_IFAIRY_LUT=0` 禁用
  时，走 LUT 路径（否则回退到原有 mul_mat）。
- 索引：`ggml/src/ggml-quants.c` 生成 **3-weight 直接 6-bit pattern**（1 byte 存 0..63）：
  - `pat = c0 | (c1<<2) | (c2<<4)`
  - 分组按 `QK_K=256` block 内部进行：`85` 个 triplet + `1` 个尾组（`{255, pad, pad}`），不跨 block、不丢维度。
- LUT 构表：`ggml/src/ggml-ifairy-lut.cpp::ggml_ifairy_lut_preprocess()`
  - 支持两种表布局（通过 `GGML_IFAIRY_LUT_LAYOUT=legacy|compact` 选择，默认 `legacy`）：
    - `legacy`：每组构造完整的 `4 × 64`（`int16`）pattern 表（`512B/group`）
    - `compact`：每组构造紧凑表：`3 positions × 4 codes × 4 channels = 48B/group`（`int8`）
  - `compact` 的每个 position 是一个 16B 表：`tbl_pos[code*4 + 0..3] = {ac,ad,bc,bd}`（`int8`），其中 `code∈{0,1,2,3}` 对应 `(-1,0)/(1,0)/(0,-1)/(0,1)`
  - scale：每个 **block** 2 个 `float`（`d_real/d_imag`，被该 block 的全部 `86` 个 group 共享）
- GEMM：`ggml/src/ggml-ifairy-lut.cpp::ggml_ifairy_lut_qgemm()`
  - `legacy`：每 group 直接读取 `{ac,ad,bc,bd}`（`int16`）并 widen+accum 到 `int32`
  - `compact`：对每个 group 做 3 次 position 查表并相加得到 `{ac,ad,bc,bd}`，再 widen+accum；NEON 下使用 3 次 32-bit load（4B）+ `vaddw_s16` 走整数累加
  - 输出默认以 **bf16-pair packed in F32** 的方式写回（与现有 ifairy vec_dot 约定一致）。
  - 在 `__aarch64__ + __ARM_NEON` 下使用 NEON（否则走标量）。

### 1.1 选择 `legacy` 还是 `compact`？

- `legacy/compact` 在同一机器上多次运行会有一定波动：以 0.1 的 tok/s 记录为准；当前默认策略仍为 `legacy`（更稳）
- `compact` 的主要价值是显著降低 per-group LUT 带宽/工作集（`512B -> 48B`），后续要想稳定胜出，需要继续压低 per-group 的额外指令开销

### 1.2 为什么 LUT 是“四通道”而不是直接存实部/虚部？

当前 correctness-first 采用 `sum_ac/sum_ad/sum_bc/sum_bd` 四通道（本质是把复数乘法拆成 4 个可独立累加的基底和），原因：

- **严格复现 baseline 语义**：`ggml_vec_dot_ifairy_q16_K_generic` 在 `w * conj(x)` 下天然需要 `Σ(xr*wr) / Σ(xi*wr) / Σ(xr*wi) / Σ(xi*wi)` 四项，最后再组合成 `(out_r,out_i)`。
- **scale/系数无法在 LUT 阶段完全合并**：激活块有 `d_real/d_imag` 两套 scale，权重行还有 `d_real/d_imag` 两个系数；其中权重系数是 **per-row** 的，LUT 预处理是 **per-column** 的，不能把权重系数 bake 进 LUT，否则会退化成“每行一份 LUT”，内存/构表成本不可接受。
- **累加阶段不可消除**：点积跨 `K` 的求和必须在 qgemm 里做，因为每个 group 查哪个 `pat` 是由权重索引决定的；能做的优化是把乘法移出 inner-loop（当前四通道设计已经把权重系数的浮点乘法移到每个输出一次）。

## 2. 本次清理/整理做了什么（NEON 版本落地前的清理）

### 2.1 移除 debug 导致的“非必要改动”

- 删除/收敛了运行时大量 `fprintf`/对照打印，避免多线程下的噪声与非确定性输出。
- 清理了 `todo_*`/注释掉的断言等临时代码残留。

### 2.2 严格校验语义改为“验证 LUT 输出”

- `GGML_IFAIRY_LUT_VALIDATE_STRICT=1` 现在用于 **对 LUT 输出做 reference 对照并断言**（而不是直接走 reference 旁路输出）。
- 位置：`ggml/src/ggml-ifairy-lut.cpp::ggml_ifairy_lut_qgemm()`。

### 2.3 工作区布局整理：LUT/scale 线程共享

- `ggml_ifairy_lut_get_wsize()` 与 `ggml_compute_forward_mul_mat()` 对齐为：
  - `quant_bytes`（仅当 `src1=F32` 时需要）
  - `shared_bytes = pad(lut_bytes + scale_bytes)`（**一次构表，所有线程共享**）
  - `tmp_bytes * nth`（每线程一行的临时输出缓冲）
- 位置：
  - `ggml/src/ggml-ifairy-lut.cpp::ggml_ifairy_lut_get_wsize()`
  - `ggml/src/ggml-cpu/ggml-cpu.c` 的 LUT 分支

### 2.4 量化相关的清理与健壮性修复

- `ggml/src/ggml-cpu/quants.c::quantize_row_ifairy()` 去掉了对整行 `malloc/free` 的临时缓冲，改为两遍扫描直接量化，保持与 `quantize_row_ifairy_ref()` 语义一致。
- `ggml/src/ggml-quants.c::quantize_row_ifairy_q16_ref()` 补齐了 `int8` 饱和到 `[-127, 127]` 的 clamp，避免极端情况下的溢出风险。

## 3. 推荐验证方式（本地）

1) 重新编译（Release）  
`cmake --build build-rel --config Release -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)`

2) 单测  
`./build-rel/bin/test-ifairy`

3) 运行 LUT 并开启严格对照（慢，但用于确认一致性）  
`GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_VALIDATE_STRICT=1 ./build-rel/bin/test-ifairy`

4) CLI 快速 sanity（tok/s 与输出可读性）  
`GGML_IFAIRY_LUT=1 ./build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv`

5) 可选：BK/BM tile（用于探索 cache/带宽优化）  
`GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=2 GGML_IFAIRY_LUT_BM=64 ./build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv`  
备注：`GGML_IFAIRY_LUT_VALIDATE_STRICT=1` 时会自动禁用 tiling（strict 目前假设 full-K 单次计算）。

6) 回归（decode/布局/tiling 一致性）  
`./build-rel/bin/test-ifairy` 内置 `Test 5: iFairy LUT backend tiling regression`（会在测试内部设置 `GGML_IFAIRY_LUT=1`：先对比 `BK/BM` tiling 与非 tiling 的输出 bitwise 一致性；再对比 `N==1`（decode-like）下 `GGML_IFAIRY_LUT_LAYOUT=legacy` vs `compact` 的输出 bitwise 一致性；若 `GGML_IFAIRY_ARM_LUT` 未启用则自动跳过）。

7) 性能测试复现脚本（tok/s，对比 LUT=0/1 与不同开关）  
```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT}/build-rel/bin/llama-cli"
MODEL="${ROOT}/models/Fairy-plus-minus-i-700M/ifairy.gguf"

COMMON=( -m "${MODEL}" --gpu-layers 0 -t 4 -b 1 --seed 1 -p "I believe life is" -n 512 -no-cnv )

run_case() {
  local name="$1"
  shift
  echo
  echo "==== ${name} ===="
  "$@" "${COMMON[@]}" 2>&1 | tee "/tmp/ifairy_lut_${name}.log" | grep -E "tok/s|eval time|prompt eval time|sampling time" || true
}

run_case "lut0" env GGML_IFAIRY_LUT=0 "${BIN}"
run_case "lut1" env GGML_IFAIRY_LUT=1 "${BIN}"
run_case "lut1_fullacc" env GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_FULLACC=1 "${BIN}"
run_case "lut1_bk2_fullacc" env GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=2 GGML_IFAIRY_LUT_FULLACC=1 "${BIN}"
```
备注：如需验证输出一致性，先跑 `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_VALIDATE_STRICT=1 ./build-rel/bin/test-ifairy`；性能脚本不建议开 strict（会强制一些路径禁用/变慢）。

> 备注（脚本输出 `tok/s=nan` 的常见原因）：`llama-cli` 的性能日志字段在不同版本里可能是 `X tokens per second`（当前默认）而不是 `X tok/s`。  
> 若你本地脚本仍在按 `tok/s` 解析，会导致抓不到数值而写入 `nan`；已在 `scripts/ifairy_lut_sweep.sh` 里同时兼容两种格式，并只从 `eval time` 行提取（排除 `prompt eval time` / `sampling time`）。

### 3.1 常见问题：`llama-cli` 输出 gibberish（例如 `I believe life isDocuments CeUNTares cred`）

这类输出在 iFairy 上通常说明“算子输出已不可信”，常见原因：

1) **跑了旧的二进制**（二进制与源码/模型不匹配）：模型仍能加载，但类型表/算子实现落后，从而生成乱码  
2) **LUT 路径回归**（改了 LUT/调度代码但未做 sanity check）：算子返回了错误结果，采样就会变成乱码

**快速判断（看启动日志）**

- 若出现 `llama_model_loader: unknown type ifairy` 或 `print_info: file type   = unknown, may not work`，基本可以确定你在跑旧的 `llama-cli`。
- 正常情况下应看到 `print_info: file type   = IFairy`，且不应出现 `unknown type ifairy`。

**修复方式**

1) 重新编译你实际在用的那套 build 目录（不要只改源码不 rebuild）：

- `build-rel`（推荐）：
  - `cmake --build build-rel --config Release -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)`
- `build`（如果你习惯用 `./build/bin/llama-cli`）：
  - 建议重新配置为 Release：`cmake -B build -DCMAKE_BUILD_TYPE=Release`
  - 然后编译：`cmake --build build -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)`

2) 用固定 seed 做一次 sanity check（输出应可读）：

`./build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 --seed 1 -p "I believe life is" -n 16 -no-cnv`

如果仍然是乱码：

- 先确认不是旧二进制（例如 `which llama-cli` / `ls -la build*/bin/llama-cli`）
- 再用 `GGML_IFAIRY_LUT=0` 复测：如果关闭 LUT 后输出恢复正常，则优先排查 LUT 路径（预处理/查表/写回/并行分工等）

## 4. 后续工作（按优先级）

> 目标：优先提升 Apple Silicon（ARM64 + NEON）的 tok/s，且不破坏 `w * conj(x)` 语义与现有输出一致性。

（优先级依据：见 0.2 的 Xcode Profile，`ggml_ifairy_lut_qgemm_ex` 占比 63%）

1) **把 63% 的热点继续压下去：优化 `ggml_ifairy_lut_qgemm_ex`（compact 优先）**  
   - 目标：减少每 group 的 load/widen/add 指令数与依赖链，减少 L1 miss（尤其是 decode：`N≈1`、每 token 都要跑一遍）
   - 任务拆解（不需要形状专用内核；目标是把 inner-loop 变“更像纯带宽”）：
     - **unroll + 多累加器**：对 group 循环做 2/4-way unroll，采用 `isum0/isum1` 交错累加，减少 load-use 依赖链
     - **减少地址计算**：把每个 position 的 16B 表当作 `4×int32`，用 `t0[c0]` 方式索引（减少 `*4`/LEA）
     - **prefetch 策略**：prefetch `grp + k_ifairy_lut_group_bytes` 与 `idx_g + k`；对比“prefetch 太早/太晚/无效”的差异（Xcode 可直接看到 L1 miss）
     - **N==1 快路（仍属于 LUT，不是形状模板）**：为 decode 常见 `N==1` 在 `qgemm_ex` 内加一个 runtime 分支，消掉 col 循环与部分指针运算
     - **减少 call/拷贝开销（仍属于 LUT）**：非 tiling 情况下避免“每 row 调一次 qgemm + memcpy”，改为每线程处理连续 row-block 并直接写回 `dst`
   - 交付/验收：
     - `./build-rel/bin/test-ifairy` 全通过
     - 在 0.2 的 decode 配置下，`ggml_ifairy_lut_qgemm_ex` 占比下降（目标 < 55%），同时 eval tok/s 上升（目标 +10%）

### 4.1 近期进展（与本次改动相关）

- 修复 strict 验证误报：legacy `qgemm_ex` 的 strict reference 之前用错了 `ifairy` 权重 bit-pack 解码（导致 `GGML_IFAIRY_LUT_VALIDATE_STRICT=1` 直接断言失败）；已修正为与 `ggml-quants.c`/单测一致的 `chunk/lane/part` 解码。
- compact decode 优化尝试：为 `GGML_IFAIRY_LUT_LAYOUT=compact` 增加 `N==1` fast-path，并把 group 循环改为 4-way unroll（交错 `isum0/isum1`）。当前 sanity 输出正常、单测与 strict 全通过；但 tok/s 仍未回升到预期（见 0.1 最新两条记录）。
- 工程健壮性：把 `compact` 的 group bytes 统一为头文件常量 `GGML_IFAIRY_LUT_COMPACT_GROUP_BYTES`，并在 `ggml-cpu.c` 加入 `GGML_ASSERT(need == ggml_ifairy_lut_get_wsize(...))`，避免工作区切分公式漂移导致的 silent memory corruption。
- 小幅 hot-path 清理：`ggml_ifairy_lut_qgemm_ex` 的非 strict 路径不再计算 `act_blocks` 指针（strict 才需要 reference 对照）。
- decode 场景并行化：当 `src1=F32` 且 `N < nth`（常见 `N==1`）时，把激活量化从“按列分片”改为“按 K-block range 分片”，减少线程空转（`a3296bec`）。
- 可控 prefetch：新增 `GGML_IFAIRY_LUT_PREFETCH=0/1`（默认启用）用于 profile/sweep 对照，避免“盲目 prefetch”只能靠改代码来试（`627dea55`）。
- prefetch 工程修复：确保 `GGML_IFAIRY_LUT_PREFETCH=0` 能覆盖 legacy/compact 的 `qgemm_ex/accum4_ex` 全部 prefetch 点位，避免 fast-path 里出现“关不掉的 prefetch”（`46dcb0cb`）。
- overflow 断言：为 `ggml_ifairy_lut_get_wsize` 与 `ggml-cpu.c` 的 LUT 工作区切分补齐 size_t overflow 断言，避免 size wrap 导致的 silent 越界（`2a39f249`）。

2) **降低 `ggml_graph_compute_thread` 的框架开销（24%）**  
   - 目标：减少同步与小 kernel 调度开销，让更多时间落在“有效算术”上
   - 可做：
     - 继续减少 LUT 路径里的 barrier 次数（尤其 tiled/BK 版本），能用 `FULLACC` 解决的重复构表/重复同步尽量消掉
     - 检查是否存在 “很小但很频繁” 的算子（例如某些额外拷贝/转换）可以在 LUT 路径内合并或延后
     - 对 decode 场景（`N≈1`）评估线程数与切分策略：避免线程空转/争用（Xcode 能看到大量线程在等待就说明要改切分）

3) **减少重复 preprocess 与同步开销（先让 BK/BM “不再变慢”）**  
   - 当前已落地：NEON 构表（`pat` 维度向量化）+ NEON 累加（标量回退）。  
   - 已有实验性 BK/BM tiling，但在部分 workload 上会因 `preprocess + barrier` 频繁而变慢。  
   - 已实现一条“full accumulator” 的 tiled 路径（默认对小 `N` 自动启用，可用 `GGML_IFAIRY_LUT_FULLACC=0/1` 控制）：为整个 `M×N` 维护共享 `{ac,ad,bc,bd}` 累加器，使每个 K-tile 的 `preprocess` 只做一次（不再按 BM 行块重复），显著减少 barrier 次数。
   - 已实现（保持 LUT 路线）：
     - `preprocess` 多线程协作构表：`N>=nth` 按 col 切分；`N<nth` 按 group stride 切分以避免 false sharing（减少“线程 0 构表，其余线程等待”）
     - activation 量化（`src1=F32 -> ifairy_q16`）按 col 并行，减少 thread 0 独占量化带来的空转
   - 下一步优先项：
     - 在 tiled/BK 路径引入 **双缓冲 LUT + pipeline**（一部分线程构下一 tile，另一部分线程消费上一 tile），减少每 tile 的同步次数并尽量重叠构表与累加

4) **降低 LUT 工作区与带宽（提高上限）**  
   - 已完成一项低风险带宽优化：把 activation `d_real/d_imag` 的 `scales` 从“每 group 一份”改为“每 block 一份”（1 个 block 内 86 个 group 共享），显著减少 `scales` 读写与工作区占用。
   - 已完成一项结构性带宽优化：把原本的 `4×64 int16` per-group pattern LUT 压缩为 **`3×4×4 int8`（48B/group）**（`GGML_IFAIRY_LUT_LAYOUT=compact`），并在 NEON 下用 32-bit load（4B）做 position 查表 + 整数累加（`vqtbl` 版本曾尝试但在 M4 上不占优，暂不作为默认实现）。
   - 已完成一项低风险算术/带宽优化：在 `qgemm/accum4` 内先按 block 累加 `int32` 的 `{ac,ad,bc,bd}`，再做一次 `float` 转换与缩放（减少 per-group 的 `vcvt + mul`）。
   - 下一步（更激进）：在不破坏 `w * conj(x)` 语义的前提下，进一步减少 per-group 的指令数（例如把 3 次 position 查表的开销通过 unroll/流水化隐藏，或探索更“共享 LUT”的 canonical 方案）。

5) **索引生命周期/缓存策略升级（工程化与复用）**  
   - 已完成：`ggml_ifairy_lut_transform_tensor()` 把索引缓冲改为使用 `ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ...)` 分配，并缓存到全局 map（key: `{data ptr, nbytes, k, rows}`），同一份权重 data（含 view/复用场景）只生成一次索引。
   - 已完成：`ifairy_lut_extra` 增加 `index_buffer` 字段，区分 backend-buffer 分配与 legacy `ggml_aligned_malloc()`；`ggml_ifairy_lut_free()` 统一释放缓存 buffer + extras，避免 double-free。
   - 后续：如果要进一步“按 ctx/图生命周期”做精确释放，需要引入 refcount/弱引用或把 indexes 变成真正的 tensor（目前先以工程可用、复用明确为主）。

6) **再做 BM/BK 调参（把调参放在结构优化之后）**  
   - 在 1/2 完成前，单纯调 `GGML_IFAIRY_LUT_BK_BLOCKS / GGML_IFAIRY_LUT_BM` 往往波动大且不可复现；结构性开销下降后再调参更稳定。
   - 推荐方法：用脚本扫参，固定 seed/prompt/token，直接输出按 tok/s 排序的结果：  
     `bash scripts/ifairy_lut_sweep.sh`  
     可通过环境变量覆盖：`THREADS=4 TOKENS=512 BK_LIST="0 1 2 4" BM_LIST="32 64 128" FULLACC_LIST="0 1" bash scripts/ifairy_lut_sweep.sh`
   - 备注：扫参会多次启动 `llama-cli`（每次都会加载模型），所以默认只扫少量组合；确认方向后再扩大 `BK_LIST/BM_LIST` 范围。

（贯穿）**测试与性能记录**  
   - 已补充 `tests/test-ifairy.cpp` 的 **CPU backend tiling 回归**：固定小形状（`K=512` 强制多 tile），对比 tiling 与非 tiling 输出 **bitwise 一致**。  
   - 继续补充 “LUT vs reference” 单测形状覆盖，并固定 `llama-cli` 命令/seed 记录 LUT=0 vs LUT=1 tok/s；必要时用 `llama-bench`/`llama-perplexity` 做对照。

## 5. 进一步性能提升路线图（参考 `BitNet/docs/lut-arm.md`）

`BitNet/docs/lut-arm.md` 的 ARM LUT 实现有几个很“工程化”的性能关键点：**int8 QLUT + `vqtbl` 查表**、**int32 累加**、**更少的 scale/元数据**、以及（在它的约束下）**对固定形状做专用内核**。（iFairy 的 `compact` 布局当前用 32-bit load 查表；`vqtbl` 版本曾尝试但在 M4 上更慢。）iFairy 这条 3-weight LUT 路径虽然语义/布局不同，但可以借鉴同样的方向来继续提速。

### 5.1 BitNet ARM LUT 的关键做法（可借鉴的点）

- **激活一次性量化 + 构表**：先对激活做 `max(abs(x))` 归一化得到 `lut_scales[0]`，再把激活量化并重排为 `QLUT`（查表友好、按 nibble 直接索引）。
- **`vqtbl` 代替“随机查表”**：把 LUT 排成 16-entry（或多表拼接）的 byte 表，靠 `vqtbl1q_s8` 做“向量化查表”，避免 NEON 不擅长的通用 gather-load。
- **累加与反量化分离**：K 维循环里只做整数查表与 `int32` 累加，最后统一用 `Scales/LUT_Scales` 做一次反量化到 float。
- **分块与工作区布局清晰**：`QLUT`、`lut_scales`、（可选）fp16 缓冲按固定顺序布置，线程 0 构表后 barrier，再并行 qgemm。
- **形状专用内核（取舍明确）**：BitNet 的 ARM 路径只覆盖少量固定 `(m,k)`，用编译期常量把 unroll/布局都做死，以换取吞吐。

### 5.2 映射到 iFairy：优先级最高的提升方向

1) **已完成：把“64-pattern × 4ch × int16” LUT 压缩到 `compact int8` 的形态（`48B/group`）**  
   - 通过“3 个 position 的可加性分解”（每个 code 只对 `{ac,ad}` 或 `{bc,bd}` 贡献 ±x），把 per-group LUT 从 `512B` 降到 `48B`。
   - `compact` 的 NEON 热路径当前用 32-bit load（4B）做 position 查表 + `int32` 累加；`vqtbl` 版本曾尝试但在 M4 上不占优。

2) **进一步降低 `preprocess` 的开销与同步频率（让 tiling 更稳定地赢）**  
   - 继续沿着当前 `FULLACC` 的方向，把 “一份 LUT / 多次消费” 做到更彻底：减少 barrier 次数、减少重复构表、把构表做成对 `K-tile` 的 pipeline（例如线程 0 预取/构表下一 tile，其余线程消费上一 tile）。
   - 参考 BitNet 的做法，尽量把 scale/元数据压到最少（例如：在误差可接受的前提下，把 per-block scale 进一步合并到 per-tile/per-col）。

3) **把热路径“常见形状”做成更激进的专用内核（可选，但上限高）**  
   - BitNet 选择“只做 matvec + 固定 (m,k)”来换取最强内核；iFairy 目前是通用 `mul_mat`（N 可变、K 可变），上限会被通用性拖累。  
   - 实际落地方式：先用 profile/日志统计 iFairy 推理中最热的 `(M,K,N)` 组合（例如 N 常为 1/2，K 常为 256 的倍数），然后为 1~2 组最热形状提供专用 fast-path（模板化 K-tile、固定 unroll、固定布局）。

4) **继续做 cache/预取与布局细化（低风险“小刀”）**  
   - 参考 BitNet 对工作区布局/对齐的强调：保证 LUT、indexes、scales 都是 64B 对齐；对 `indexes` 做更有针对性的预取；对 LUT 做 “按访问顺序” 的线性布局，降低 cache miss。  
   - 对当前 `accum4` 的 unroll/预取策略继续微调（配合 `scripts/ifairy_lut_sweep.sh` 扫参，记录稳定的 tok/s）。

### 5.3 建议的推进顺序（避免做无用功）

1) 先把 **`compact int8` LUT 压缩方案**做出一个 correctness 版本（严格对照 + 单测覆盖），明确误差与内存/速度收益。  
2) 再把它接入现有 `BK/BM/FULLACC` 框架，跑 sweep 找稳定配置，并用 `llama-cli` 固定 prompt/seed 记录 tok/s。  
3) 最后再决定是否要走 BitNet 那种“形状专用内核”的路线（收益高，但维护成本也高）。
