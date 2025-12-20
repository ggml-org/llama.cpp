# iFairy 3‑Weight LUT · API 与工程约定（以当前实现为准）

> 目标：在 iFairy 2‑bit 复数权重体系上落地 3‑weight LUT 路径，接口与工程骨架对齐《IFAIRY_ARM_3W_LUT_DESIGN.md》，并吸收 BitNet TL1 的分层思路，同时把生命周期、并行、回退与验证机制描述清楚。
>
> 本文只关注“接口/数据约定/路由与线程模型”；性能复现与调参记录见《IFAIRY_ARM_3W_LUT_STATUS.md》。

## 1. 范围与前提

- 编译期开关：`GGML_IFAIRY_ARM_LUT`（见 `ggml/CMakeLists.txt`）。开启时 CMake 会强制关闭 Metal/CUDA/HIP/MUSA/Vulkan/OpenCL/SYCL/WebGPU/zDNN 等加速后端，保证 CPU-only。
- 运行时开关：`GGML_IFAIRY_LUT=0` 禁用 LUT（默认启用：未设置或非 `0`）。
- 权重：`src0->type == GGML_TYPE_IFAIRY`（2‑bit 复数权重，压缩存储）。
- 激活：`src1->type == GGML_TYPE_F32`（复数 bf16-pair 容器）或 `src1->type == GGML_TYPE_IFAIRY_Q16`（`block_ifairy_q16`）。
- 输出：`dst->type == GGML_TYPE_F32`（同样以 bf16-pair 写回；见 `pack_bf16` 参数）。
- 形状约束：要求逻辑 `K % QK_K == 0`（当前 `QK_K=256`），否则 `can_mul_mat` 回退。
- 数学语义：与 `ggml_vec_dot_ifairy_q16_K_generic` 一致，计算 `w * conj(x)`（不是 `w * x`）。

## 2. 核心数据结构与内存约定

### 2.1 3W 索引（weights → indexes）

- 一组三权重用 1 byte 记录 6-bit pattern：
  - `pat = c0 | (c1 << 2) | (c2 << 4)`，其中 `ci ∈ {0,1,2,3}` 为原始 2-bit ifairy code。
  - 高 2 bit 预留（当前为 0）。
- 分组策略（不改变模型语义）：对每个 `QK_K=256` block：
  - `intra=0..84`：覆盖 `0..254` 的 85 个 triplet；
  - `intra=85`：尾组 `(255, pad, pad)`（缺失位置按 code=0 处理）。
  - `groups_per_block = (QK_K + 2) / 3 = 86`。
- 索引缓冲布局：按行连续，`rows × groups_per_row`，其中 `groups_per_row = (K / QK_K) * 86`。
- 生成入口：`ggml_ifairy_3w_encode()`（`ggml/src/ggml-quants.c`）。

### 2.2 LUT 工作区（activations → lut + scales）

当前实现支持两种 LUT 布局（由 `GGML_IFAIRY_LUT_LAYOUT` 选择；`auto` 走默认策略）：

- `legacy`：每组 `4 ch × 64 pat × int16`（`512 B / group / col`）。
- `compact`：每组 `3 pos × 4 codes × 4 ch × int8`（`48 B / group / col`），NEON 内核用 32-bit load + widen + add 的方式累加。

缩放数组（`lut_scales`）与 LUT 分离：

- 激活缩放是 **per-block**：每个 `(col, block)` 存 `2 floats`（real/imag），与 `block_ifairy_q16::d_real/d_imag` 对齐。
- 权重缩放来自 `block_ifairy::d_real/d_imag`（当前实现按行读取，通常取 `w_row[0]`；若未来演进为 per-block 权重缩放，需要在这里同步更新契约）。

### 2.3 `tensor->extra`（索引生命周期）

当前实现通过 `struct ifairy_lut_extra`（`ggml/src/ggml-ifairy-lut.h`）把索引挂到权重张量：

```c
struct ifairy_lut_extra {
    uint8_t * indexes;
    size_t    size;
    struct ggml_tensor * index_tensor;
    ggml_backend_buffer_t index_buffer;
};
```

- `ggml_ifairy_lut_transform_tensor()` 负责创建/复用索引 buffer，并设置 `tensor->extra`。
- `ggml_ifairy_lut_free()` 负责释放内部缓存与 `extra`（现状：仍有全局状态；后续会改为更可维护的生命周期绑定，见 7）。

## 3. API（以头文件为准）

头文件：`ggml/src/ggml-ifairy-lut.h`
实现：`ggml/src/ggml-ifairy-lut.cpp` + `ggml/src/ggml-ifairy-lut-{transform,preprocess,qgemm}.cpp`

- 初始化/释放：`ggml_ifairy_lut_init()`, `ggml_ifairy_lut_free()`
- 路由与工作区：`ggml_ifairy_lut_can_mul_mat()`, `ggml_ifairy_lut_get_wsize()`
- 索引生成：`ggml_ifairy_lut_transform_tensor()`
- 预处理：`ggml_ifairy_lut_preprocess()`, `ggml_ifairy_lut_preprocess_ex()`
- GEMM/累加：`ggml_ifairy_lut_qgemm()`, `ggml_ifairy_lut_qgemm_ex()`, `ggml_ifairy_lut_accum4_ex()`
- 标量回退：`ggml_ifairy_lut_mul_mat_scalar()`

约定：

- `pack_bf16=true` 时，`dst` 以 bf16-pair 写回（但承载 tensor 的 `type` 仍为 `GGML_TYPE_F32`）。
- `strict=true` 时用于对照/验证（可能禁用 tiling 等优化；不要用于跑分）。

## 4. ggml 路由与线程模型（以现实现为准）

集成点：`ggml/src/ggml-cpu/ggml-cpu.c::ggml_compute_forward_mul_mat`（在 `#if defined(GGML_IFAIRY_ARM_LUT)` 下）。

执行流程要点（简化）：

1) thread 0 负责确保 `src0->extra/indexes` 已生成（`transform_tensor`），随后 `ggml_barrier`。

2) 读取运行时开关：`strict/BK/BM/FULLACC/layout`，并计算工作区切分：
   - `act_q`（可选）：`src1==F32` 时的临时量化缓冲；
   - `lut + scales`：shared 区域（按 tile 大小）；
   - `tmp`：每线程 scratch（accumulator 等）。

3) 非 tiling：
   - 所有线程并行执行 `preprocess_ex()` 填充 `lut+scales`，随后 barrier；
   - 每线程处理自己负责的 row range，调用 `qgemm_ex()` 写回。

4) BK tiling：
   - 每个 K-tile 重复一次 `preprocess_ex()` + barrier；
   - `FULLACC` 模式下可用共享 accumulator，减少按 BM 行块重复构表/同步。

## 5. 运行时开关（当前实现）

- `GGML_IFAIRY_LUT=0/1`：禁用/启用 LUT（默认启用）。
- `GGML_IFAIRY_LUT_LAYOUT=legacy|compact|auto`：选择 LUT 布局（默认 `legacy`；`auto` 走默认策略）。
- `GGML_IFAIRY_LUT_BK_BLOCKS=<int>`：K 维按 256-block 做 tiling（`0` 禁用；strict 下强制禁用）。
- `GGML_IFAIRY_LUT_BM=<int>`：BM 行块大小（仅 tiling 生效）。
- `GGML_IFAIRY_LUT_FULLACC=0/1`：tiling 下共享 accumulator（未设置时可能按 `(N,acc_bytes)` 自动启用）。
- `GGML_IFAIRY_LUT_VALIDATE_STRICT=0/1`：严格对照（验证用）。
- `GGML_IFAIRY_LUT_DEBUG=0/1`：打印少量路由诊断（默认关闭）。
- `GGML_IFAIRY_LUT_PREFETCH=0/1`：控制 LUT 热路径中的 prefetch（默认启用；设为 `0` 方便 profile/sweep 对照；覆盖 legacy/compact 的 `qgemm_ex/accum4_ex`）。
- `GGML_IFAIRY_LUT_N1_FASTPATH=0/1`：控制 `compact` 的 `N==1` decode 快路（默认启用；设为 `0` 强制走通用路径做 A/B）。
- `GGML_IFAIRY_LUT_COMPACT_N1_UNROLL=2|4`：控制 `compact` 的 `N==1` 快路 group-loop 的 unroll（默认 `4`；设为 `2` 用于 A/B）。

## 6. 性能提升规划（主线，必须把 tok/s 拉上去）

> 2025-12-18 更新：基于《IFAIRY_LUT_PERF_ANALYSIS_20251218.md》的复盘，decode 场景热点大致为 `ggml_ifairy_lut_qgemm_ex ≈ 53~55%` + `ggml_graph_compute_thread ≈ 30%`。因此主线需要同时推进：压 `qgemm_ex` 单位成本 + 处理同步/调度开销（否则上限会被框架吞掉）。

### 6.0 复现与验收口径（统一）

- **构建**：使用 Release，并确保你跑的就是你编译的二进制（优先 `build-rel`）。
- **推荐基准命令**：以 `IFAIRY_ARM_3W_LUT_STATUS.md` 的 `0.1 tok/s 记录`表头命令为准（固定 `--seed/-t/-b/-c/-p/-n` 与 LUT env 组合）。
- **短测 vs 长测**：
  - A/B 调优：优先用短测 `-n 64` 做 `ABABAB` 交替跑，减少热漂移偏置；
  - 最终记录：用长测 `-n 256` 对每个 layout 连续跑 3 次，记录 `min/max/mean` 后再下结论（长测之间要给足冷却；若出现明显 outlier/单调下降，先冷却后重测，否则结论无效）。
- **双 build A/B（强烈建议）**：保留一个“上一个稳定基线”的 build 目录（例如 `build-rel-a`），用“旧 bin vs 新 bin”做 `ABABAB`，避免跨时段热漂移导致误判。
- **正确性门槛**：
  - `./build-rel/bin/test-ifairy` 必须通过；
  - `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_VALIDATE_STRICT=1 ./build-rel/bin/test-ifairy` 必须通过（验证用，不跑分）。
- **性能门槛**：
  - 每次性能相关改动，都在 `IFAIRY_ARM_3W_LUT_STATUS.md` 的 `0.1 tok/s 记录`追加一条可复现记录（固定 seed/prompt/thread/token/env）。
  - A/B 的原始日志（建议 TSV）落到 `/tmp/` 并在 `STATUS.md` 引用路径（避免只剩结论没证据）。
  - 主观体验（输出可读/不卡）不作为性能结论，必须以 `eval tok/s` 为准。
- **文档更新**：功能落地后同步更新 `IFAIRY_ARM_3W_LUT_STATUS.md`（tok/s 记录）、`IFAIRY_ARM_3W_LUT_API_PLAN.md`、`IFAIRY_ARM_3W_LUT_DESIGN.md` 与相关 `AGENTS.md`，确保无过时信息。

### 6.0.1 回归恢复（仅在 tok/s 明显回落时启用）

> 背景：在 `0ec52a5a` 之后出现过大幅性能回归（详见 `IFAIRY_LUT_PERF_REGRESSION_ANALYSIS.md`）。当前（2025-12-18）已恢复到并超过该档位，因此恢复步骤保留为“回退手册”，但默认冻结（避免在达标后继续扩大热路径改动面）。

恢复顺序（每一步都必须：Release rebuild + `test-ifairy` + strict + 追加 tok/s 记录）：

1) **优先恢复 `ggml_ifairy_lut_preprocess_ex` 的构表热路径**  
   - 目标：把 `compact` 的构表回到“最少临时变量 + 最少指令 + 顺序写入”的版本（倾向 `0ec52a5a` 风格的 direct store），避免 `pack`/`vcreate`/`vcombine` 等额外开销主导。

2) **回退/对照 `qgemm_ex` 的 unroll 与 prefetch 策略**  
   - 原则：先做 A/B（2-way vs 4-way，prefetch unconditional vs conditional），以 tok/s 与 profile 结果为准。

3) **暂时禁用 `N==1` fast-path（直到明确在目标机器上稳定赢）**  
   - decode 的快路要么“显著更快”，要么就先关掉，避免“引入分支 + 代码体积 + 寄存器压力”但收益不确定。

4) **回退/简化 `src1=F32` 的激活量化并行切分（以 decode 为准）**  
   - 原则：decode (`N≈1`) 更怕额外的调度/分片/算术开销；优先选简单路径，先把 tok/s 拉回去，再讨论更细的并行化。

验收：

- Apple Silicon / 4 threads / 固定命令下：`legacy` ≥ `15 tok/s` 且 `compact` ≥ `17 tok/s`（以 `STATUS.md` 记录为准）
- `GGML_IFAIRY_LUT_VALIDATE_STRICT=1` 下对照全通过（允许变慢，但必须正确）

当前结论（2025-12-18）：

- ✅ R0 已完成（`79c915e5`）：`preprocess_ex(compact)` 回退为 `memset + direct stores` 后，`legacy/compact` tok/s 已恢复并超过 `0ec52a5a` 档位（见 `IFAIRY_ARM_3W_LUT_STATUS.md` 最新记录）。
- 建议先“冻结 R1/R2/R3”，把当前状态稳定住：避免在已经达标时继续改动热路径扩面导致新的不可控回归；后续若 tok/s 再次回落或确有上限诉求，再按 R1→R2→R3 做 A/B。
- 复盘与后续方向以《IFAIRY_LUT_PERF_ANALYSIS_20251218.md》为准：继续压 `qgemm_ex`，并正面处理 `ggml_graph_compute_thread`（同步/调度）占比偏高的问题。

### 6.1（恢复后）继续压 `ggml_ifairy_lut_qgemm_ex` 热点（`compact` 优先）

目标：把 decode 常见的 `N≈1` 进一步提速，并让 `compact` 在 Apple Silicon 上稳定胜出。

> 性能分析（见 `IFAIRY_LUT_PERF_ANALYSIS_20251218.md`）：`qgemm_ex` 约 53.5% + `ggml_graph_compute_thread` 约 30.5%，因此“算子优化 + 框架开销”要并行推进。

任务清单（按收益预期排序）：

- **减少 3 次 position 查表的结构性开销（P0，优先探索）**：当前 `compact` 每 group 必做 3 次 position 查表 + widen + add，指令密度高且依赖链长；若能把 “3 次” 减到 “2 次”，通常比微调 unroll/prefetch 更可能拉开稳定差距。
  - 方向 A：`(c0,c1)` 预合并表（16 组合）+ `pos2`（4 组合），将每 group 查表从 3 次降到 2 次（注意 preprocess 成本与 cache footprint 平衡）。
    - 进展：已尝试 `GGML_IFAIRY_LUT_LAYOUT=compact2`（pos0+pos1 合并为 16-way int16 表），在当前实现与机器上出现明显 tok/s 回退（见 `IFAIRY_ARM_3W_LUT_STATUS.md` 失败案例），暂不继续扩面；后续只在 preprocess 能做到“更便宜/不增加写带宽”时再考虑重启。
  - 方向 B：小型 64-pattern 表（int8/小 int16），把查表变回“一次读 4ch”的形态，同时尽量保持比 legacy 更小的工作集。
- **unroll + 多累加器**：对 group 循环做 2/4-way unroll，采用 `isum0/isum1` 交错累加，减少 load-use 依赖链。
  - 经验：在 Apple M4 上尝试把 `compact` 的 `N==1` fast-path 从 4-way 收敛到 2-way 会明显变慢（见 `IFAIRY_ARM_3W_LUT_STATUS.md` 的失败案例记录）；不要在没有 A/B 的情况下改 unroll。
  - 建议：保留 perf-safe A/B 开关 `GGML_IFAIRY_LUT_COMPACT_N1_UNROLL=2|4`（默认 `4`），避免为了试 unroll 反复改代码并引入回归点。
- **减少地址计算**：把每个 position 的 16B 表当作 `4×int32`，用 `t0[c0]` 方式索引（减少 `*4`/LEA）。
  - 已试验（本机，`compact`，`N==1`，`-n 64` 双向交替 A/B）：将 `grp + k * pos_bytes` 改为“基址 + `pos_stride`”访问，未观察到稳定收益（两轮对照方向相反），已回退；该类微改动暂降级为 P2（除非 profile 明确显示地址计算占比异常）。
  - 小点：当 `pat` 已经做过 `& 0x3f` 掩码时，`c2` 可直接用 `pat >> 4`（不需要再 `& 3`），减少一条冗余指令并缩短依赖链。
- **prefetch 策略**：对 `grp + k_ifairy_lut_group_bytes` 与 `idx_g + ...` 做可控预取（以 Xcode Profile 的 L1 miss 变化为准，避免“盲目 prefetch”）。
- **N==1 快路**：在 `qgemm_ex` 内增加运行时分支，消掉 col 循环与部分指针运算（仍属 LUT 通用内核，不做形状模板爆炸）。
  - 建议保留一个“perf-safe”开关：`GGML_IFAIRY_LUT_N1_FASTPATH=0` 可强制走通用路径，用于回归/调优 A/B（避免“更复杂但更慢”的快路悄悄常驻）。
- **减少 call/拷贝开销**：非 tiling 情况下尽量避免“每 row 调一次 qgemm + memcpy”，让每线程处理连续 row-block 并直接写回 `dst`。

验收（至少满足其一，并记录到 `STATUS.md`）：

- decode 场景下 `eval tok/s` 相对当前基线提升 ≥ 10%；或
- profile 中 `ggml_ifairy_lut_qgemm_ex` 占比下降到 < 55%。

### 6.2 优先级 2：降低 `ggml_graph_compute_thread` 的框架开销（同步/调度）

目标：减少 barrier 与小 kernel 调度开销，让更多时间落在“有效算术”。

分析结论（2025-12-18）：`ggml_graph_compute_thread` 占比约 30.5%，decode 场景更容易被同步/调度吞没。

建议动作：

- **优先：减少“缓存命中时仍然同步”的开销**：例如 `transform_tensor` 若命中缓存（indexes 已存在且不会再变），应避免每次 mul_mat 都做一次全线程 barrier（只在首次生成/真正做了 transform 时 barrier）。
- **decode 线程策略复评（用数据说话）**：`N==1` 时 threads 未必越多越快；允许用 env 加开关做实验，避免把策略写死。
  - 建议优先做一个低风险试验：为 `N==1` 增加“单线程/少线程”可控开关（例如 `GGML_IFAIRY_LUT_DECODE_NTH=1/2/4`），严格 A/B 验证是否能降低 `ggml_graph_compute_thread` 占比并抬升 tok/s。
- 继续减少 LUT 路径里的 barrier 次数（尤其 tiled/BK 版本），能用 `FULLACC` 解决的重复构表/重复同步尽量消掉。
- 检查是否存在“很小但很频繁”的额外拷贝/转换可在 LUT 路径合并或延后。
- 对 decode 场景（`N≈1`）重新评估线程数与切分策略，避免线程空转/争用（以 profile 里线程等待为准）。
- ✅ 已做（`a3296bec`）：当 `src1=F32` 且 `N < nth`（常见 `N==1`）时，激活量化改为按 `K/QK_K` block 做 range 分片，避免“只有 thread0 量化，其它线程 barrier 等待”。

### 6.3 优先级 3：让 BK/BM tiling “不再变慢”，并可稳定获益

目标：在更大 K/prefill 场景下，BK/BM 不引入额外同步瓶颈；`FULLACC` 能稳定 amortize `preprocess + barrier`。

建议动作：

- `FULLACC` 自动策略只在明确收益场景启用（小 `N` + `acc_bytes` 可控），并允许用 env 强制开/关复现差异。
- 将 “构表下一 tile / 消费上一 tile” 做成 pipeline（先以可验证的 CPU-only 版本落地，再谈进一步重排）。
- 把 `preprocess_ex` 的切分策略固定为“对 N 或 group 做分片”，避免 false sharing 与 “线程 0 构表、其余等待”。
- 调参只用 sweep 驱动（避免“凭直觉写死 BK/BM”）：优先用 `scripts/ifairy_lut_sweep.sh` 固定 seed/prompt 跑完再决定默认策略。

### 6.4 降低 LUT 工作区与带宽（提高上限）

目标：在不破坏 `w * conj(x)` 语义的前提下，继续降低 per-group 指令数与工作区读写。

- 固定 `compact` 的带宽优势：保持 `48B/group`，避免“表变大但算力没下降”的回退。
- 优先把 `qgemm_ex` 内部累加保持在 `int32`，只在 block/row 级做一次 `float` 缩放与写回。
- 谨慎探索更激进的 LUT 合并方案（例如减少 position 查表次数），必须有 profile 证据 + A/B 记录支撑。

### 6.5 索引生命周期/缓存策略升级（工程化）

目标：减少重复索引构建与全局状态副作用，确保复用清晰、释放可控。

- 维持 “相同权重 data 只生成一次 index” 的缓存策略，并确保 `ggml_ifairy_lut_free()` 能完全回收。
- 如需更精细的生命周期管理，优先考虑按 ctx/graph 绑定或 refcount，避免引入难以追踪的全局泄漏。

### 6.6 再做 BM/BK 调参（放在结构优化之后）

- 在 6.1/6.2/6.3 有明确进展前，不建议依赖 `BK/BM` 调参提升吞吐。
- 统一用 `scripts/ifairy_lut_sweep.sh` 做 sweep，固定 seed/prompt/token 并保留日志。

### 6.7 可选：形状专用 fast-path（谨慎引入）

> BitNet TL1 通过“少量固定形状专用内核”换取吞吐上限。iFairy 若要走这条路，必须先用 profile/日志确认最热 `(M,K,N)`，再只为 1~2 个形状提供 fast-path，避免维护成本失控。

> 当前约定：本阶段先不考虑 6.7（除非回归恢复完成且 profile 明确显示收益空间）。

可选落地方向：

- decode 形状 `N==1` 的进一步专用化（例如减少分支、固定 stride、固定 accumulator 布局）。
- 对最热形状做有限度的模板化（避免在代码库里散落大量形状分支）。

## 7. 工程地基（并行推进，避免性能“回归/难复现”）

> 性能冲刺不等于忽略地基；这些问题一旦踩中，会直接把 tok/s 或可复现性拉垮。

### 7.0 P0：可复现性与回归门槛（必须落地为“流程”）

- **双 build A/B**：保留一个“上一个稳定基线”的 build 目录（例如 `build-rel-a`），每次改动用 “旧 bin vs 新 bin” 做 `ABABAB` 交替跑，避免跨时段热漂移导致误判。
- **原始日志留档**：A/B 的 raw 日志或 TSV 存 `/tmp/`，并在 `IFAIRY_ARM_3W_LUT_STATUS.md` 引用路径（防止只剩结论没证据）。
- **env cache 规则**：只缓存“不会被测试用例在进程内动态修改”的 env；若某 env 在 `test-ifairy` 用 `scoped_env_var` 修改，则该 env 不应做进程级 cache（否则测试与复现会失真）。
  - 当前已缓存：`GGML_IFAIRY_LUT_PREFETCH`、`GGML_IFAIRY_LUT_N1_FASTPATH`、`GGML_IFAIRY_LUT_COMPACT_N1_UNROLL`（进程内不会动态变化）。
- **profile 使用规范**：profile 用于“定位主矛盾”，不要用单次采样占比当 KPI；需要记录至少 2~3 次采样的波动范围。

### 7.1 P1：健壮性与一致性（不影响热路径的硬化优先）

- ✅ size/overflow：为 `ggml_ifairy_lut_get_wsize` 与 `ggml-cpu.c` 的 LUT 工作区切分加入 overflow 断言，避免 size_t wrap 后的越界访问（`2a39f249`）。
- ✅ prefetch 可控：`GGML_IFAIRY_LUT_PREFETCH=0/1` 可覆盖 legacy/compact 的 `qgemm_ex/accum4_ex`，方便 profile/sweep 对照。
- ✅ env 解析收敛：将 LUT 相关 env 解析 helper 集中复用，减少重复与语义漂移。
- ✅ 路由/配置健壮性：无效 layout/BK/BM 在 debug 下 warn/clamp，减少 silent fallback。
- ✅ 工作区一致性：`compact` group bytes 常量化，并在 `ggml-cpu.c` 断言 `need == get_wsize(...)`，避免切分公式漂移导致的 silent memory corruption。

### 7.2 P2：可维护性（在性能稳定后再推进）

- ✅ 代码拆分：`ggml/src/ggml-ifairy-lut.cpp` 已按 preprocess/qgemm/transform/common 拆分（`ggml-ifairy-lut-{preprocess,qgemm,transform}.cpp` + `ggml-ifairy-lut-impl.h`），减少 legacy/compact 重复代码。
- ✅ 测试补齐（`tests/test-ifairy.cpp`）：
  - 对齐与误对齐：`test_ifairy_lut_index_alignment()` 覆盖 64B 对齐尺寸与 misaligned index buffer 编码。
  - 小维度：`test_ifairy_lut_scalar_small_dims()`（`M=N=1,K=QK_K`）。
  - 大维度：`test_ifairy_lut_backend_large_dims()`（tiling vs non-tiling）。
  - 分配失败/缓冲不足：`test_ifairy_lut_index_encode_failure()`（短 buffer 返回 false）。
  - 并发 transform：`test_ifairy_lut_transform_cache()`。
  - 形状对齐/路由：`test_ifairy_lut_transform_invalid_shape()`（`K % QK_K != 0`）。
  - 关键 env 语义：`test_ifairy_lut_env_semantics()` + `test_ifairy_lut_layout_auto_policy()`。
