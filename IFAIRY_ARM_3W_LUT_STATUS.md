# iFairy ARM 3‑Weight LUT · 现状与后续工作（代码清理后）

本文记录当前 `GGML_IFAIRY_ARM_LUT`（CPU-only）下 **非 NEON** 标量 LUT 路径的代码现状、最近一次清理/整理的结果，以及下一步工作列表。

## 1. 当前现状（可工作的标量 LUT 路径）

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
  - 每组三权重对应该列激活的 `4 × 64` 表（`sum_ac/sum_ad/sum_bc/sum_bd`，`int16`）。
  - scale：每组 2 个 `float`（real/imag）。
- GEMM：`ggml/src/ggml-ifairy-lut.cpp::ggml_ifairy_lut_qgemm()`
  - 使用索引查表 + scale 累加，最终按 `ggml_vec_dot_ifairy_q16_K_generic` 语义合成输出（`w * conj(x)`）。
  - 输出默认以 **bf16-pair packed in F32** 的方式写回（与现有 ifairy vec_dot 约定一致）。

## 2. 本次清理/整理做了什么（非 NEON 标量路径）

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

## 4. 后续工作（按优先级）

1) **NEON 预处理 + NEON qgemm**（保持 `w * conj(x)` 语义不变）  
   - 先实现 BK tile 的 LUT 构表与查表累加，再迭代 DOTPROD 版本。
2) **索引生命周期/缓存策略**  
   - 当前 `transform_tensor()` 生成的索引缓冲挂在 `tensor->extra`，在 CPU backend free 时统一释放；后续可考虑更细粒度的生命周期绑定与复用策略。
3) **降低 LUT 工作区与带宽**  
   - 从 `4×64 int16` 的 correctness-first 结构，演进到 “16 canonical + factor” 或更紧凑布局，控制 cache 压力。
4) **补充测试与回归策略**  
   - 添加“LUT vs reference” 的针对性单测（覆盖多种 M/N/K 形状），并把 `GGML_IFAIRY_LUT_VALIDATE_STRICT` 纳入 CI/本地脚本流程（可仅跑小规模）。
5) **性能记录与调参**  
   - 固定 `llama-cli` 命令与 seed，记录 LUT=0 vs LUT=1 的 tok/s；引入 `llama-bench`/`llama-perplexity` 做质量与性能对照。

