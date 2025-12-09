# iFairy 3‑Weight LUT API 设计与开发规划（修订版）

> 目标：在 iFairy 2‑bit 复数权重体系上落地 3‑weight LUT 路径，接口与工程骨架对齐《IFAIRY_ARM_3W_LUT_DESIGN.md》，并吸收 BitNet TL1（`BitNet/docs/lut-arm.md`、`bitnet-lut-kernels-tl1.h`）的分层与接口风格，同时回应并行、生命周期、类型与回退等风险。

## 1. 范围与假设

- 平台：ARMv8.2+NEON（DOTPROD 可选），仅 CPU 后端；NEON 不可用时有标量 LUT fallback。
- 权重：`GGML_TYPE_IFAIRY`（2‑bit 复数，压缩存储）。
- 激活：复用现有 `GGML_TYPE_IFAIRY_Q16`（block_ifairy_q16，int8 实/虚分平面 + fp16 缩放）；不新增新类型。
- 索引：3×2‑bit → 1 字节格式已在 `ggml-quants.[ch]` 完成。
- 内核 v1 聚焦 matvec / 小 N（例如 N ≤ 4）；大 N 通过外层循环封装。

> 继续使用 ifairy_q16 的理由  
> - 生态复用：已在 ggml 注册，量化/缩放/ROPE 等链路完整，无需新增类型与算子。  
> - 性能匹配：LUT 构表/查表期望使用 int8 激活 + 缩放，避免关键路径浮点运算，符合 BitNet TL1 的“预量化+查表”模型。  
> - 带宽/溢出控制：int8 + 缩放便于控制 LUT 范围；直接用 float 激活会在预处理阶段引入更多访存与浮点乘，抵消 LUT 吞吐优势。  
> - 类型检查简单：can_mul_mat/路由只需认 ifairy_q16，避免再造一套激活容器。  

## 2. 数据结构与内存组织

### 2.1 索引缓冲
- 编码：低 4bit=idx'，bit5=neg_real，bit6=neg_imag，bit7=swap；padding code=0。
- 构造：`ggml_ifairy_3w_encode()`，`K3 = ((K + 2)/3)*3`，行优先，每组三权重 1 字节。
- 对齐：64B；默认保留原始 2‑bit 权重。可选 flag 允许仅保留索引、释放原始权重（减内存）。

### 2.2 工作区（激活侧）
- 组成：激活 pack（直接读取 ifairy_q16，不新增 pack）、per‑group LUT（`lut_real[16]` + `lut_imag[16]`，共 32B/组）、LUT 缩放。
- 尺寸：`wsize_per_col = 2*K3 + (K3/3)*32 + 2*sizeof(float)`；多线程按线程数乘以对齐后的 slice。
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
- 类型：严格要求 `src0->type=GGML_TYPE_IFAIRY`，`src1->type=GGML_TYPE_IFAIRY_Q16`，`dst->type=F32`。
- 生命周期：索引用 shadow tensor；`extra` 只读；避免裸指针泄露。
- C/C++：统一使用 .cpp + extern "C" 接口，避免模板落入 .c。
- 内存放大：索引 + 原权重双存；提供可选 flag 释放原权重以降内存。

## 8. can_mul_mat 判定伪码

```
if (getenv("GGML_IFAIRY_LUT")== "0") return false;
if (src0->type != GGML_TYPE_IFAIRY || src1->type != GGML_TYPE_IFAIRY_Q16) return false;
if (dst->type != GGML_TYPE_F32) return false;
if (src0->backend != CPU || src1->backend != CPU) return false;
if (src1->ne[1] > N_limit /* e.g. 4 */) return false;
extra = (ifairy_lut_extra *) src0->extra;
if (!extra || !extra->indexes || extra->K != src0->ne[0]) return false;
if (!shape_supported(extra->K, src0->ne[1]) && extra->BK==0) return false;
return true; // NEON available -> NEON path; else scalar LUT
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
- [x] 接口骨架文件与 CMake 线路（init/free/can/wsize/transform/preprocess/qgemm）。— 头文件与 CMake 选项 `GGML_IFAIRY_ARM_LUT` 已生效；`can_mul_mat`/预处理/qgemm 提供标量实现，`wsize/transform` 仍占位，后续补齐索引 shadow tensor 与 workbuf 估算。
- [x] 标量预处理 + 标量 qgemm（用于正确性基准）。
  - 进度：`ggml/src/ggml-ifairy-lut.cpp` 提供标量 reference（构表 + qgemm）。预处理使用 ifairy_q16 激活构建 16 模式 LUT，激活缩放取第一块 fp16 缩放；qgemm 按索引 opcode 应用 LUT，int32 累加后乘激活缩放与权重缩放（取第一块 d_real/d_imag）输出 float。`tests/test-ifairy.cpp::test_ifairy_lut_scalar_matmul` 新增单测，对 K=QK_K 场景下的 encode + preprocess + qgemm 与直接复数点积（按量化值解码）进行逐元素比对。
- [x] ggml 路由集成（mul_mat 分支 + workbuf 管理）。
  - 进度：`ggml_ifairy_lut_get_wsize` 计算每线程 LUT/scales 工作区并通过 `ggml_cpu_extra_work_size` 上报；`mul_mat` 检测 ifairy 类型时调用 `transform_tensor` 生成索引 shadow（挂在 `tensor->extra`），按线程切分行，复用 per-thread workbuf 预处理激活→LUT，使用 LUT qgemm 写入 bf16-packed 输出；缺陷：索引释放依赖 `ggml_ifairy_lut_free`，暂未在张量释放时自动回收；GGUF 类型注册仍未解决 loader “unknown type ifairy” 警告。
  - 路由策略：env `GGML_IFAIRY_LUT=0` 关闭；形状需 K%QK_K==0，dst F32；暂用标量 LUT，NEON 待后续；多线程拆 row。
  - 缩放对齐：仍使用 per-tensor 缩放（首块 d_real/d_imag）；未实现 per-block scales。
- [ ] NEON 预处理/LUT 构造（按 BK tile）。
- [ ] NEON qgemm（16 组解码流水 + 行展开）。
- [ ] 单测补充（LUT vs 真值；路径开关）。
- [ ] 性能/正确性记录，更新设计文档。

---

> 本文档为 API/规划，不含实现代码；落地时需确保与《IFAIRY_ARM_3W_LUT_DESIGN.md》一致，并优先采用“每线程独立预处理+计算”以避免 barrier 瓶颈。***
