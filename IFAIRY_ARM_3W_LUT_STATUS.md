# iFairy ARM 3‑Weight LUT · 现状与后续工作（NEON 标量混合版）

本文记录当前 `GGML_IFAIRY_ARM_LUT`（CPU-only）下 iFairy 3-weight LUT 的代码现状（含 NEON 加速实现）、最近一次清理/整理的结果，以及下一步工作列表。

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
  - 每组三权重对应该列激活的 `4 × 64` 表（`sum_ac/sum_ad/sum_bc/sum_bd`，`int16`），并采用 `pat` 维度上的 **4 通道交织布局**（便于 NEON `vst4/vld1`）。
  - scale：每个 **block** 2 个 `float`（`d_real/d_imag`，被该 block 的全部 `86` 个 group 共享），用于减少重复读写与带宽。
- GEMM：`ggml/src/ggml-ifairy-lut.cpp::ggml_ifairy_lut_qgemm()`
  - 使用索引查表 + scale 累加，最终按 `ggml_vec_dot_ifairy_q16_K_generic` 语义合成输出（`w * conj(x)`）。
  - 当前实现会先在 **每个 block 内以 int32 累加 86 个 group 的 `{ac,ad,bc,bd}`**，再按该 block 的 `d_real/d_imag` 一次性缩放并累加到 float（减少 per-group 的 float convert/mul）。
  - 输出默认以 **bf16-pair packed in F32** 的方式写回（与现有 ifairy vec_dot 约定一致）。
  - 在 `__aarch64__ + __ARM_NEON` 下使用 NEON 累加（否则走标量）。

### 1.1 为什么 LUT 是“四通道”而不是直接存实部/虚部？

当前 correctness-first 采用 `sum_ac/sum_ad/sum_bc/sum_bd` 四通道（本质是把复数乘法拆成 4 个可独立累加的基底和），原因：

- **严格复现 baseline 语义**：`ggml_vec_dot_ifairy_q16_K_generic` 在 `w * conj(x)` 下天然需要 `Σ(xr*wr) / Σ(xi*wr) / Σ(xr*wi) / Σ(xi*wi)` 四项，最后再组合成 `(out_r,out_i)`。
- **scale/系数无法在 LUT 阶段完全合并**：激活块有 `d_real/d_imag` 两套 scale，权重行还有 `d_real/d_imag` 两个系数；其中权重系数是 **per-row** 的，LUT 预处理是 **per-column** 的，不能把权重系数 bake 进 LUT，否则会退化成“每行一份 LUT”，内存/构表成本不可接受。
- **累加阶段不可消除**：点积跨 `K` 的求和必须在 qgemm 里做，因为每个 group 查哪个 `pat` 是由权重索引决定的；能做的优化是把乘法移出 inner-loop（当前四通道设计已经把权重系数的浮点乘法移到每个输出一次）。

## 2. 本次清理/整理做了什么（NEON 版本落地前的清理）

### 2.1 移除 debug 导致的“非必要改动”

- 删除/收敛了运行时大量 `fprintf`/对照打印（包括 `GGML_IFAIRY_LUT_COMPARE`、`GGML_IFAIRY_LUT_VALIDATE_VECDOT` 相关路径），避免多线程下的噪声与非确定性输出。
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
`cmake --build build-rel --config Release -j $(nproc)`

2) 单测  
`./build-rel/bin/test-ifairy`

3) 运行 LUT 并开启严格对照（慢，但用于确认一致性）  
`GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_VALIDATE_STRICT=1 ./build-rel/bin/test-ifairy`

4) CLI 快速 sanity（tok/s 与输出可读性）  
`GGML_IFAIRY_LUT=1 ./build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv`

5) 可选：BK/BM tile（用于探索 cache/带宽优化）  
`GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=2 GGML_IFAIRY_LUT_BM=64 ./build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv`  
备注：`GGML_IFAIRY_LUT_VALIDATE_STRICT=1` 时会自动禁用 tiling（strict 目前假设 full-K 单次计算）。

6) 回归（tiling vs 非 tiling 一致性）  
`./build-rel/bin/test-ifairy` 内置 `Test 5: iFairy LUT backend tiling regression`（会在测试内部设置 `GGML_IFAIRY_LUT=1`、并对比 `BK/BM` tiling 与非 tiling 的输出 bitwise 一致性；若 `GGML_IFAIRY_ARM_LUT` 未启用则自动跳过）。

## 4. 后续工作（按优先级）

> 目标：优先提升 Apple Silicon（ARM64 + NEON）的 tok/s，且不破坏 `w * conj(x)` 语义与现有输出一致性。

1) **减少重复 preprocess 与同步开销（先让 BK/BM “不再变慢”）**  
   - 当前已落地：NEON 构表（`pat` 维度向量化）+ NEON 累加（标量回退）。  
   - 已有实验性 BK/BM tiling，但在部分 workload 上会因 `preprocess + barrier` 频繁而变慢。  
   - 已实现一条“full accumulator” 的 tiled 路径（默认对小 `N` 自动启用，可用 `GGML_IFAIRY_LUT_FULLACC=0/1` 控制）：为整个 `M×N` 维护共享 `{ac,ad,bc,bd}` 累加器，使每个 K-tile 的 `preprocess` 只做一次（不再按 BM 行块重复），显著减少 barrier 次数。
   - 下一步优先项：在此基础上继续降低 barrier 成本、做 unroll/预取，并评估 DOTPROD 版本。

2) **降低 LUT 工作区与带宽（提高上限）**  
   - 已完成一项低风险带宽优化：把 activation `d_real/d_imag` 的 `scales` 从“每 group 一份”改为“每 block 一份”（1 个 block 内 86 个 group 共享），显著减少 `scales` 读写与工作区占用。
   - 已完成一项低风险算术/带宽优化：在 `qgemm/accum4` 内先按 block 累加 `int32` 的 `{ac,ad,bc,bd}`，再做一次 `float` 转换与缩放（减少 per-group 的 `vcvt + mul`）。
   - 从 `4×64 int16` correctness-first 结构，演进到更紧凑的布局（例如 “16 canonical + factor” 等方向），降低 `lut/scales/indexes` 的带宽压力与 cache miss。
   - 备注：若要推进 “16 canonical + factor”，需要配合 **int8 LUT + `vqtbl` 查表/整数累加** 的整体重构；单纯在当前 float 累加框架里增加 canonical 变换通常会带来额外 shuffle/分支而变慢。

3) **索引生命周期/缓存策略升级（工程化与复用）**  
   - 现状：`transform_tensor()` 生成的索引缓冲挂在 `tensor->extra`，在 CPU backend free 时统一释放。  
   - 下一步：把索引缓存与生命周期绑定做得更清晰（例如 index_tensor/后端 buffer 管理、复用策略、跨图复用边界），并确保 teardown 路径一致。

4) **再做 BM/BK 调参（把调参放在结构优化之后）**  
   - 在 1/2 完成前，单纯调 `GGML_IFAIRY_LUT_BK_BLOCKS / GGML_IFAIRY_LUT_BM` 往往波动大且不可复现；结构性开销下降后再调参更稳定。

（贯穿）**测试与性能记录**  
   - 已补充 `tests/test-ifairy.cpp` 的 **CPU backend tiling 回归**：固定小形状（`K=512` 强制多 tile），对比 tiling 与非 tiling 输出 **bitwise 一致**。  
   - 继续补充 “LUT vs reference” 单测形状覆盖，并固定 `llama-cli` 命令/seed 记录 LUT=0 vs LUT=1 tok/s；必要时用 `llama-bench`/`llama-perplexity` 做对照。
