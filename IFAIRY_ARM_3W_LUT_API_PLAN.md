# iFairy 3‑Weight LUT API 设计与开发规划（修订版）

> 目标：在 iFairy 2‑bit 复数权重体系上落地 3‑weight LUT 路径，接口与工程骨架对齐《IFAIRY_ARM_3W_LUT_DESIGN.md》，并吸收 BitNet TL1（`BitNet/docs/lut-arm.md`、`bitnet-lut-kernels-tl1.h`）的分层与接口风格，同时回应并行、生命周期、类型与回退等风险。

## 1. 范围与假设

- 平台：ARMv8.2+NEON（DOTPROD 可选），仅 CPU 后端；NEON 不可用时有标量 LUT fallback。
- 权重：`GGML_TYPE_IFAIRY`（2‑bit 复数，压缩存储）。
- 激活：复用现有 `GGML_TYPE_IFAIRY_Q16`（block_ifairy_q16，int8 实/虚分平面 + fp16 缩放）；不新增新类型。
- 索引：3×2‑bit → 1 字节格式已在 `ggml-quants.[ch]` 完成。
- 内核 v1 聚焦 matvec / 小 N（例如 N ≤ 4）；大 N 通过外层循环封装。
- 构建约束：启用 `GGML_IFAIRY_ARM_LUT` 时，CMake 在 configure 阶段会自动把 Metal/CUDA/HIP/MUSA/Vulkan/OpenCL/SYCL/WebGPU/zDNN 等后端强制为 `OFF`，并提示 warning，确保整个链路保持 CPU-only，无需在命令行手动追加多组 `-DGGML_*` 关闭 GPU。

> 继续使用 ifairy_q16 的理由  
> - 生态复用：已在 ggml 注册，量化/缩放/ROPE 等链路完整，无需新增类型与算子。  
> - 性能匹配：LUT 构表/查表期望使用 int8 激活 + 缩放，避免关键路径浮点运算，符合 BitNet TL1 的“预量化+查表”模型。  
> - 带宽/溢出控制：int8 + 缩放便于控制 LUT 范围；直接用 float 激活会在预处理阶段引入更多访存与浮点乘，抵消 LUT 吞吐优势。  
> - 类型检查简单：can_mul_mat/路由只需认 ifairy_q16，避免再造一套激活容器。  

## 2. 数据结构与内存组织

### 2.1 索引缓冲
- 编码：低 4bit=idx'，bit5=neg_real，bit6=neg_imag，bit7=swap；padding code=0。
- 构造（方案 A 已落地）：按 block 切分，每个 `QK_K=256` 丢弃末尾一个权重（255 个有效值可被 3 整除），`groups_per_block = (QK_K-1)/3 = 85`，`k_padded = blocks * (QK_K-1)`，行优先，每组三权重 1 字节。
- 对齐：64B；默认保留原始 2‑bit 权重。可选 flag 允许仅保留索引、释放原始权重（减内存）。

### 2.2 工作区（激活侧）
- 组成：激活 pack（直接读取 ifairy_q16，不新增 pack）、per‑group LUT（`lut_real[16]` + `lut_imag[16]`，共 32B/组）、LUT 缩放。
- 尺寸（方案 A）：`groups = blocks * 85`，每列需要 `groups*32` LUT bytes + `groups*2*sizeof(float)` scales，F32 激活时还需 `blocks*sizeof(block_ifairy_q16)` 的量化缓冲；多线程按线程数乘以 64B 对齐 slice。
- BK tile 时，改用 tile 粒度计算；保证每线程有独立 slice，避免同步。

### 2.3 权重附加信息
- 结构（拟）：  
  ```
  struct ifairy_lut_extra {
      int K, K3;
      int BK, BM;
      int groups_per_row;
      const uint8_t * indexes;          // 指向索引数据，只读
      const struct ggml_tensor * index_tensor; // shadow tensor 由 ggml 管理生命周期
      ggml_half d_real, d_imag;         // per-tensor 权重量化缩放
      // 预留：const ggml_half * block_scales; // 若未来有 per-block 缩放
  };
  ```
- 生命周期：索引存入 shadow tensor（常规 ggml tensor），`extra` 仅挂指针；无需手工 free。`ggml_ifairy_lut_free` 不回收 per-tensor 数据。

## 3. API 设计（函数/文件布局）

### 3.1 公共头（`ggml-ifairy-lut.h` 或扩展 ggml-bitnet.h 风格）
- `void   ggml_ifairy_lut_init(void);`            // 平台探测、内核指针注册
- `void   ggml_ifairy_lut_free(void);`            // 清理全局状态（不回收 shadow tensor）
- `bool   ggml_ifairy_lut_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);`
- `size_t ggml_ifairy_lut_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst, int n_threads);`
- `bool   ggml_ifairy_lut_transform_tensor(struct ggml_tensor * tensor, struct ggml_tensor ** index_tensor_out);` // 生成索引/extra
- `void   ggml_ifairy_lut_preprocess(int m, int k, int n, const void * act /* ifairy_q16 */, size_t act_stride, void * lut_scales, void * lut_buf);`
- `void   ggml_ifairy_lut_qgemm(int m, int k, int n, const void * qweights, const uint8_t * indexes, const void * lut, const void * lut_scales, float * dst, size_t dst_stride);`

> 说明：v1 定位 matvec/小 N，N 通过参数显式传入；大 N 上层循环调用 preprocess+qgemm。

### 3.2 内部实现文件
- `ggml/src/ggml-ifairy-lut.cpp`：C++，对外 C 接口；wsize/调度/路由。
- `ggml/src/ggml-cpu/arch/arm/ifairy-lut.cpp`：NEON/标量核 + 预处理（intrinsics），接口 `extern "C"`.
- 复用 `ggml-ifairy_3w` 编码表，无重复定义。

### 3.3 ggml 集成点
- `ggml.c::ggml_compute_forward_mul_mat` 新增 ifairy LUT 分支。
- CMake：`GGML_IFAIRY_ARM_LUT` 控制编译；仅 ARM 生成。

## 4. 行为约定与回退

- K/K3：所有路径统一使用逻辑 K，内部通过 `K3=((K+2)/3)*3`；`extra` 存 K 与 K3，避免漂移。
- 组粒度：每 16 组索引批处理；LUT 在 tile 内一次加载，常驻寄存器（避免循环内重复 load）。
- 缩放：v1 仅 per‑tensor `d_real/d_imag` + per‑tensor LUT scale；若未来有 block scales，内核在 BK 汇总后乘以 block scale（不修改 LUT）。
- 线程模型：完全并行，按 M 维分片；每线程独立 preprocess（量化/LUT）+ qgemm，无全局 barrier；wsize = n_threads * per_thread_slice。
- 回退策略：
  - `GGML_IFAIRY_LUT=0` 或 `can_mul_mat==false` → 旧 ifairy 路径。
  - `GGML_IFAIRY_LUT=1` & NEON 可用 → LUT NEON。
  - `GGML_IFAIRY_LUT=1` & NEON 不可用 → LUT 标量。
- 内存策略：默认保留原始权重；可选 `GGML_IFAIRY_LUT_DROP_ORIG=1` 释放原权重（后续实现）。

## 5. 开发步骤（按优先级）

1) **transform**：调用 `ggml_ifairy_3w_encode` 生成索引；创建 shadow index tensor；填充 `ifairy_lut_extra`（K/K3/BK/BM/ptr）。
2) **wsize**：包含 per-thread slice（激活 int8 读 + LUT + scales），按 64B 对齐。
3) **preprocess**：标量/NEON，按 BK tile 构造 `lut_real/lut_imag`；输入 ifairy_q16（实/虚分平面 int8 + fp16 缩放）；支持 N≤4 循环。
4) **qgemm**：标量/NEON 3‑weight 核心；16 组索引批处理，opcode 变换后 int32 累加，末尾乘权重/激活缩放。
5) **ggml 路由**：mul_mat 分支路由至 LUT；按 M 分片并行，每线程独立 preprocess+qgemm。
6) **测试**：对齐/精度/多线程/NEON vs 标量；环境变量开关；K3 padding；m,k=1536/4096 组合。
7) **性能记录**：tok/s 对比旧 ifairy；确认 LUT 常驻寄存器（tile 内不重复 load）。

## 6. 命名与文件组织

- `ggml-ifairy-lut.h/cpp`：API + 路由。
- `ggml-cpu/arch/arm/ifairy-lut.cpp`：内核/预处理（C++ intrinsics）。
- `ggml-quants.[ch]`：已含索引编码表，直接复用。
- CMake：`GGML_IFAIRY_ARM_LUT` 选项；仅 ARM 构建。

## 7. 风险与决策点

- BK/BM：初版按设计文档建议（如 BK≈3072），存于 extra；后续基准微调。
- 批量：v1 小 N；大 N 通过外层循环；如需真 mul_mat 接口，后续扩充 API 的 N/stride。
- 类型：`src0->type=GGML_TYPE_IFAIRY`，`src1->type` 支持 `GGML_TYPE_IFAIRY_Q16` 或 bf16-pair 容器的 `GGML_TYPE_F32`，`dst->type=F32`，K 必须按 `QK_K` 对齐。
- 生命周期：索引目前挂 `tensor->extra` 并记录在全局列表，由 `ggml_ifairy_lut_free` 统一释放；尚未绑定张量释放生命周期，后续需改为 shadow tensor 或回收回调。
- C/C++：统一使用 .cpp + extern "C" 接口，避免模板落入 .c。
- 内存放大：索引 + 原权重双存；提供可选 flag 释放原权重以降内存。

## 8. can_mul_mat 判定伪码

```
if (getenv("GGML_IFAIRY_LUT")== "0") return false;
if (src0->type != GGML_TYPE_IFAIRY) return false;
if (src1->type not in {GGML_TYPE_IFAIRY_Q16, GGML_TYPE_F32}) return false; // F32 为 bf16-pair 容器
if (dst->type != GGML_TYPE_F32) return false;
if (src0->ne[0] % QK_K != 0 || src0->ne[0] != src1->ne[0]) return false;
// 后续：可加平台/NEON 检查
return true; // transform_tensor 会按需生成索引；有 NEON 走 NEON，否者走标量 LUT
```

## 9. 数据约定示意（便于实现）

- act：ifairy_q16，实/虚分平面各 K3 个 int8，`d_real/d_imag` 为 fp16；`act_stride` 传入列/批的字节跨度。
- lut_buf：按列/线程分配；每组 32B（real16+imag16），共 `groups_per_row` 组，64B 对齐，可按 BK 分段。
- lut_scales：默认 fp32，每列/每 tensor 1 元素；若改为 per-BK，扩展为数组。
- qweights：原始 ifairy 压缩权重（不重排）；可选在 drop 原权重模式下仅保留索引。
- indexes：`extra->indexes` 指向的 1B/组 buffer。
- dst：float32 输出，`dst_stride` 控制列步长；后续由 ggml 处理类型转换。

## 10. 回退与开关

- 环境变量：`GGML_IFAIRY_LUT=0/1`（默认 1 开启）；`GGML_IFAIRY_LUT_DROP_ORIG=1` 允许 transform 后释放原权重（待实现）。
- 路由：env 关或 can_mul_mat 失败 → 旧 ifairy 路径；通过且 NEON 可用 → NEON LUT；通过但无 NEON → 标量 LUT。

## 11. 测试与精度准则

- 覆盖：K 非 3 倍数（padding），m,k=1536/4096 组合，多线程>1，NEON/标量双路径，小 N 与 matvec。
- 比对：LUT vs 直接复数 dot（同缩放），允许最大 diff 约 1e‑3~1e‑2（视缩放确定），记录 RMSE/Max error。
- 开关：环境变量 0/1，drop 原权重 flag。
- 性能：tok/s 对比旧 ifairy，确认 LUT 加载不成为热点（tile 内寄存器驻留）。

## 12. 计划拆解(可执行 TODO)
- [x] 接口骨架文件与 CMake 线路（init/free/can/wsize/transform/preprocess/qgemm）。— 头文件与 CMake 选项 `GGML_IFAIRY_ARM_LUT` 已生效；`transform_tensor` 现生成索引并挂 `tensor->extra`；`wsize` 上报每线程 LUT/scale/可选激活量化缓冲；预处理/qgemm 标量实现可复用测试。
- [x] 标量预处理 + 标量 qgemm（用于正确性基准）。
  - 进度：`ggml/src/ggml-ifairy-lut.cpp` 提供标量 reference（构表 + qgemm）；严格校验模式可直接按三权重量化值重构累加（不依赖 LUT 缩放近似）。预处理使用 ifairy_q16 激活构建 16 模式 LUT；方案 A 采用“每 256 丢弃 1 个权重”避免组跨 block。`tests/test-ifairy.cpp::test_ifairy_lut_scalar_matmul` 更新为与方案 A 对齐（跳过第 256 个权重），现已 0 diff 通过。
- [x] ggml 路由集成（mul_mat 分支 + workbuf 管理）。
  - 进度：`ggml_ifairy_lut_get_wsize` 计算每线程 LUT/scales/可选激活量化缓冲并通过 `ggml_cpu_extra_work_size` 上报；`mul_mat` 检测 ifairy 类型时调用 `transform_tensor` 懒生成索引，按线程切分行，复用 per-thread workbuf 预处理激活→LUT，使用 LUT qgemm 写入 bf16-packed 输出；缺陷：索引释放依赖 `ggml_ifairy_lut_free`，暂未在张量释放时自动回收；GGUF 类型注册仍未解决 loader “unknown type ifairy” 警告。
  - 路由策略：env `GGML_IFAIRY_LUT=0` 关闭；形状需 K%QK_K==0，dst F32；支持 `src1` 为 ifairy_q16 或 bf16-pair F32；暂用标量 LUT，NEON 待后续；多线程拆 row。
  - 缩放对齐：仍使用 per-tensor 缩放（首块 d_real/d_imag）；未实现 per-block scales，后续需对齐权重侧 block scale 方案。
  - 内存路径：激活量化缓冲仅由线程 0 填充一次，其余线程共享；每个线程的 workbuf slice 现在包含 LUT + scale + `N*sizeof(float)` 的临时行缓冲，`ggml_ifairy_lut_qgemm` 直接写入该缓冲后以列主序偏移（`col*nb1 + row*nb0`）回填 dst，彻底避免 heap 分配与 `_platform_memmove` 越界；`GGML_IFAIRY_LUT_DEBUG=1` 还会打印当前 op 的形状，便于后续排查。
- [ ] NEON 预处理/LUT 构造（按 BK tile）。
- [ ] NEON qgemm（16 组解码流水 + 行展开）。
- [ ] 单测补充（LUT vs 真值；路径开关）。
- [ ] 性能/正确性记录，更新设计文档。

## 13. 现状与已知问题（LUT=1 路径）
- 现状：`./build-rel/bin/test-ifairy` 全部通过（含 LUT 标量路径，方案 A 丢弃第 256 元素）；`GGML_IFAIRY_LUT_VALIDATE_STRICT=1` 校验模式可复现 0 diff。`./build-rel/bin/llama-cli ...`（LUT=1，strict=1）已消除 “unknown type ifairy” 警告（增加了 ftype 映射），但输出仍为乱码，需继续定位（可能与方案 A 丢弃数据/缩放链路有关）。
- 已修：ifairy RMSNorm 按 bf16-pair 解包重打包；二元算子 NaN 检查对 ifairy_add/mul 放宽；mul_mat 支持 F32 激活临时量化 + workbuf LUT；`ggml_abort` 增加 last-node 诊断输出；`ggml_ifairy_lut_can_mul_mat` 接受 F32/ifairy_q16 激活并检查 K 对齐；方案 A 索引与 LUT 链路贯通，严格模式单测对齐。
- 构建约束：`GGML_IFAIRY_ARM_LUT=ON` 时，CMake 自动强制关闭 Metal/CUDA/HIP/MUSA/Vulkan/OpenCL/SYCL/WebGPU/zDNN 等非 CPU 后端（对手动开启的选项给出 warning），保证 LUT 调试与运行仅走 CPU 路径；无需用户在命令行追加一串 `-DGGML_*` 禁用 GPU。
- 未决/调试计划：
  1) 清理索引释放：仍依赖 `ggml_ifairy_lut_free`，未绑 tensor 生命周期，需补回收或改为可追踪 buffer。
  2) 缩放策略：仍为 per-tensor，确认是否需 per-block；与 LUT 构造/累加对齐。
  3) GGUF 类型注册：已在 `llama-model-loader` 注册 `LLAMA_FTYPE_MOSTLY_IFAIRY`，告警消除。
  4) 继续验证其他 ifairy 专用算子（rope/split/merge 等）在 bf16 打包上的一致性；确认非 ifairy 权重的 matmul 合理回退，不该回退的层需检查类型/形状。
  5) 解决 LUT 路径生成乱码的问题（目前 strict 校验通过，CLI 输出仍异常），收敛到可读输出后再切回非 strict 模式并恢复性能调优。

### 13.1 当前困难（debug 结论汇总）
- **“strict 仍乱码”**：`GGML_IFAIRY_LUT=1` + `GGML_IFAIRY_LUT_VALIDATE_STRICT=1` 时仍输出乱码，说明问题不只是 LUT 表近似/溢出，而是 **LUT 路由本身的数学/形状语义与 baseline 不一致**（例如丢弃元素改变模型等效计算）。
- **方案 A 丢弃每 block 末元素的语义风险**：方案 A 为了让 `255=85*3`，当前实现“每 256 个权重丢弃最后一个元素”。这会改变真实模型的线性层（相当于强行把每 block 的最后一维权重置零），理论上足以破坏输出可读性；单测之所以能 0 diff，是因为单测的 reference 也同步跳过了该元素。
- **比较基准与 vec_dot 的链接限制**：希望直接用 `ggml_vec_dot_ifairy_q16_K` 对照，但它在 `ggml-cpu` 目标里；而 LUT 实现在 `ggml-base`，不能直接链接调用。当前在 `GGML_IFAIRY_LUT_COMPARE` 下内联实现了与 vec_dot 等价的逐权重解码基准（per-block scales），用于定位差异。
- **对照结果（仍有偏差）**：在 `GGML_IFAIRY_LUT_COMPARE=4` 的最小输出模式下，首行首列出现稳定偏差（例如 `diff≈(+0.0396,-0.1080)`），该偏差会随层数放大，最终表现为乱码。

### 13.2 可能的解决方案（按优先级）
1) **停止丢弃真实权重（回滚 A 的“drop”语义）**：方案 A 的核心是“分组不跨 block”，但不应通过丢弃真实维度来达成。更合理做法：
   - A'：保留 K=256 全量权重，只对分组产生 `85` 个三元组 + 1 个“尾部单/双元素”特殊组（或尾部作为 padding 参与）；qgemm 处理尾组时只累加有效元素（1 或 2 个），并保持与 vec_dot 完全等价。
   - 或者：仍构造 86 组（最后一组 1 个元素 + 2 个 padding），索引编码支持“不足 3 个元素”并在预处理里把缺失激活置 0。
2) **方案 B（每组三独立缩放）作为过渡**：保留原 `K3=((K+2)/3)*3` 分组（允许跨 block），把 scale buffer 扩展为每组三个权重/激活的缩放（6 floats）以保证数学严格一致；确认输出可读后再回到 A'/NEON 优化。
3) **把对照从“局部 compare”升级为可控的图级对照**：在 ggml mul_mat 路由里针对前几次 ifairy matmul：
   - 同时计算 baseline（旧 vec_dot 路径）与 LUT strict 输出，比较 max diff；一旦超阈值就打印层名/shape/nb 并 abort，快速定位第一个出错层。
4) **溢出假设的验证方式**：LUT 表构造时 int8 饱和、qgemm 累加为 float/double，本身不应产生 NaN；若怀疑饱和误差过大，可在 strict/compare 中统计 LUT 表值分布与饱和率（|val|==127 的比例），再决定是否扩大 LUT（int16）或改成 int32 累加后再缩放。

## 14. Debug / 提速 / 清理方案（可由 Codex 全流程执行）

> 目标：解决“LUT=1 输出乱码 + 很慢”的现状，并把热点从标量 `ggml_ifairy_lut_qgemm` 迁移到 NEON/更合理的数据布局；同时清理当前临时 debug 输出/日志，保留可开关的诊断工具。

### 14.1 复现与对照（必须先固定基线）

1) **固定构建类型与命令行参数**
- Release/RelWithDebInfo 用于性能；Debug 仅用于定位（Debug 会显著放大热点比例）。
- 统一 `-t/-b/-n/-p`，确保可复现。

2) **对照输出：LUT=0 vs LUT=1**
```bash
GGML_IFAIRY_LUT=0 ./build/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv
GGML_IFAIRY_LUT=1 ./build/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv
```
- 若 LUT=1 输出“乱码/不可读”，优先判定为 **数值/缩放/布局不一致**（而不是 tokenizer 问题），因为 LUT=0 同模型可读。

3) **记录关键信息（便于后续对齐）**
- 记录同一 prompt 下首 16 token 的输出文本（LUT=0 与 LUT=1）。
- 记录 `llama_perf_context_print` 的 eval tok/s，作为提速前基线。

### 14.2 CLI 可执行 Profiling（不依赖 GUI）

> 你已在 Xcode Time Profiler 看到 85% `ggml_ifairy_lut_qgemm`、7% `ggml_ifairy_factor_to_opcode`。下面给出 CLI 可复现实验流程（Codex 可直接跑）。

1) **Release 构建一个新目录（避免 Debug 干扰）**
```bash
cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release -DGGML_IFAIRY_ARM_LUT=ON
cmake --build build-rel --target llama-cli test-ifairy -j 8
```

2) **Time Profiler（xctrace）**
```bash
mkdir -p tmp/traces
GGML_IFAIRY_LUT=1 xcrun xctrace record \
  --template "Time Profiler" \
  --time-limit 15s \
  --output tmp/traces/ifairy_lut_timeprofile.trace \
  --launch -- build-rel/bin/llama-cli \
    -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv
```
- 产物 `tmp/traces/ifairy_lut_timeprofile.trace` 可在本机用 Xcode 打开，也可用 `xctrace export` 导出文本报告（如需我可以补一套 export 命令）。

3) **无符号的热点确认（quick sanity）**
```bash
time GGML_IFAIRY_LUT=1 build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv
```

### 14.3 正确性 Debug（把“乱码”定位到具体层/具体 matmul）

> 原则：在不把性能拖到不可接受的前提下，用“抽样验证”把错误收敛到一个 op（某层某个 mul_mat），再判断是索引/预处理/LUT/qgemm/缩放/输出布局哪个环节。

1) **增加“抽样校验”开关（建议实现）**
- 新增 env：`GGML_IFAIRY_LUT_VALIDATE=1`（默认 0）
- 可选参数：`GGML_IFAIRY_LUT_VALIDATE_OPS=8`（只校验前 N 个 ifairy mul_mat）
- 可选参数：`GGML_IFAIRY_LUT_VALIDATE_ROWS=2`（每个 op 校验若干行，例如 row=0 与 row=M/2）
- 校验方式：在 LUT 分支内，对抽样行调用现有 `ggml_vec_dot_ifairy_q16_K`（参考 `ggml/src/ggml-cpu/quants.c` 的 ifairy vecdot）计算“参考输出”，与 LUT 输出逐元素 diff；超过阈值直接 `ggml_abort` 并打印：
  - 当前 op 的 `src0/src1/dst` 形状与 `nb[]`
  - `max_abs_diff_real/imag`
  - 触发的 layer/op 序号（可用一个全局计数器）

2) **先验证布局一致性（dst 写回）**
- 当前 ifairy mul_mat 的 `dst` 是列主序（`col*nb1 + row*nb0`），任何写回都必须遵循该布局。
- 通过 `GGML_IFAIRY_LUT_DEBUG=1` 打印形状/stride 只做短期保留；定位完成后应删去或默认关闭（避免影响性能）。

3) **分层排查路径（建议顺序）**
- A. **索引是否稳定**：`ggml_ifairy_factor_to_opcode` 占比 7% 往往意味着 **索引生成/转换被重复触发**（例如 graph 重建/视图 tensor 导致 extra 丢失）。建议加计数器统计 `ggml_ifairy_lut_transform_tensor()` 调用次数与涉及的 `tensor->data` 指针，确认是否每 token / 每层重复生成。
- B. **缩放对齐**：现有 LUT v1 若仍采用 per‑tensor 激活缩放，会与真实 `block_ifairy_q16` 的 per‑block `d_real/d_imag` 不一致，极易导致输出乱码。应优先把 LUT 路径在数值上对齐 `ggml_vec_dot_ifairy_q16_K` 的公式与缩放位置（见 14.4）。
- C. **qgemm 数学一致性**：在抽样行上，把 LUT 输出与 vecdot 输出逐项打印（仅在 validate 模式），判断差异是系统性偏移（缩放错）还是随机（索引/查表错）。

### 14.4 缩放/数学对齐（从“能跑”到“输出可读”的关键）

> 现状提示：LUT=1 “能跑但乱码”通常不是性能问题，而是 **数学/缩放策略不一致**。必须把 LUT 的累计与 `ggml_vec_dot_ifairy_q16_K` 的参考实现对齐，至少达到“文本可读 + 与 LUT=0 输出相近”的程度，再谈 NEON 提速。

建议分两阶段推进（都可由 Codex 落地）：

**阶段 1：先保证正确（允许慢一点）**
- 预处理阶段把 `block_ifairy_q16` 反量化成 `float ax_real[K] / ax_imag[K]`（应用每个 block 的 `d_real/d_imag`），并据此构建 **float LUT**（每组 16 pattern × 复数 2 float）。
- qgemm 只做 float 累加（每行遍历 groups，按索引选 pattern 累加 real/imag），最后写入 dst 的 bf16-pair 容器。
- 目标：抽样校验通过；`llama-cli` 输出变得可读，且与 LUT=0 的文本相近（允许少量差异）。

**阶段 2：在正确基础上提速（最终目标）**
- 调整索引与分组使 LUT 组尽量不跨 `QK_K=256` block（例如按 block 内分组与 padding），让激活缩放可以在 block 级别一致应用，避免 float LUT 的内存与计算开销。
- 引入 NEON 内核：按 BitNet TL1 的思路，固定 group 的 LUT 常驻寄存器，**对 16 行（BM=16）并行做 vqtbl 查表**，把标量 `qgemm` 的 85% hotpath 转为 SIMD。

### 14.5 提速清单（按收益从高到低）

1) **构建切换到 Release**：先用 `build-rel` 验证性能；Debug 仅用于定位。
2) **避免重复 transform（降低 `ggml_ifairy_factor_to_opcode` 占比）**
- 把 `ggml_ifairy_lut_transform_tensor()` 从 compute 期懒调用，前移到模型加载/初始化阶段（或至少做全局缓存：key=`src0->data`，value=索引 buffer），保证同一权重不重复生成。
- 若 `tensor->extra` 在 view/copy 中丢失，改用 “shadow tensor + 追踪 buffer” 挂到模型 context 或全局缓存表，避免跟随 tensor 复制语义丢失。
3) **标量 qgemm 结构优化（NEON 之前的止血）**
- 为 `n==1`（典型 matvec）写专用内核，移除 `for col` 分支与 stride 计算。
- 使用 `decode_table[256]` 把 `code -> {idx,swap,neg_r,neg_i}` 预解码，降低 bit/branch 开销。
4) **NEON qgemm（核心提速点）**
- BM=16：一次处理 16 行，同一 group 的 LUT table（16B real + 16B imag）加载一次，用 vqtbl 查表得到 16 行的贡献向量并累加。
- 优先做 `n==1`，再扩展小 N（2/4）。
5) **预处理提速**
- 预处理同样可按 BM/BK 分块，并用 NEON 做 16 pattern 的批量构表（参考 BitNet TL1 的 LUT 构造部分）。

### 14.6 清理无用信息（确保最终不污染性能与日志）

- `GGML_IFAIRY_LUT_DEBUG=1`：仅用于短期定位；实现上必须保证默认完全无输出、无额外 work。
- `GGML_IFAIRY_LUT_VALIDATE=1`：只在开发/CI 使用；默认关闭。
- 清理 can_mul_mat 的频繁 `GGML_LOG_WARN`（只在 debug 模式打印一次或按采样打印），否则会严重干扰 profiler 与性能。

### 14.7 LUT 输出“乱码”专项行动（新增）

> 现状：Release + LUT=1 运行不再崩溃，但生成文本不可读。必须先保证数学正确再谈提速。

1) **严格校验模式（需实现）**  
   - 环境变量 `GGML_IFAIRY_LUT_VALIDATE_STRICT=1`：仅在 debug/QA 使用。  
   - 做法：在 `ggml_ifairy_lut_qgemm` 中，若检测到“三权重跨 block”则退化为逐权重计算：对三个权重各自读取激活缩放+权重量化缩放，按复数乘累加（仍在 LUT 分支，不回退 vec_dot），确保数值正确。  
   - 校验输出：若与参考（可选抽样行）差异超阈值，打印 op/shape/组索引并中止。

2) **分组重排（正式方案）**  
   - 目标：消除跨 block 的三权重组，让每组只落在同一 `QK_K=256` block。  
   - 方案 A：索引编码阶段按 block 切分，每 block 内 `(QK_K/3)` 组，尾部 padding；预处理/LUT/qgemm 也按 block 切片，scale buffer 仍为每组 2 float。  
   - 方案 B（过渡）：扩展 scale buffer 为每组 3×复缩放（共 6 float），即便组跨 block 也能精确应用各自缩放；待 A 完成后再收敛。

3) **实施顺序**  
   - 先实现严格校验模式，验证输出恢复可读；记录具体层/组是否有越界组。  
   - 若严格模式可读，则优先落地方案 A（按 block 切分重新编码索引+LUT）。若改索引风险大，可先落方案 B，保持正确输出后再优化内存。  
   - 保持 Release 构建，关闭默认 debug 日志，避免性能噪声。
---
---

> 本文档为 API/规划，不含实现代码；落地时需确保与《IFAIRY_ARM_3W_LUT_DESIGN.md》一致，并优先采用“每线程独立预处理+计算”以避免 barrier 瓶颈。***
