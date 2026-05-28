# IFAIRY64 OpenCL 算子实现指南

Status: Draft (2026-05-28)

本文面向刚接触 `ggml` 后端和 OpenCL 后端的新开发者，目标是把后续 `IFAIRY64` OpenCL 完整路径需要实现的算子、代码位置、语义、测试方法和验收标准讲清楚。

当前已经完成的前置工作：

- OpenCL 后端已有 `GGML_OPENCL_IFAIRY64` opt-in gate。
- OpenCL 后端已有 `GGML_TYPE_IFAIRY64` 严格白名单，但 kernel-ready gate 仍为 false。
- `GGML_TYPE_IFAIRY64` 权重在 OpenCL buffer 中已经被 staged 成 SoA raw-weight layout：
  - `q`: row-major packed 2-bit codes，16 bytes per 64-value block
  - `d`: row-major fp16 scale pairs，即 `(d_real, d_imag)`，4 bytes per 64-value block
- `set_tensor` / `get_tensor` 已经能在 raw `block_ifairy64` 和 OpenCL `q`/`d` 之间转换。

## 总体原则

### 语义不变量

所有 `IFAIRY64` matmul 都必须匹配 CPU 语义：

```text
w = wr + i*wi
x = xr + i*xi
w * conj(x) = (wr*xr + wi*xi) + i*(wi*xr - wr*xi)
```

不要把它误写成 `w * x`。这是最重要的 correctness gate。

### Fallback 规则

任何没有完整实现和测试的 shape / op 都必须继续返回 `supports_op=false`，交给 scheduler fallback 到 CPU。

具体入口在：

- `ggml/src/ggml-opencl/ggml-opencl.cpp`
- `ggml_opencl_is_ifairy_op()`
- `ggml_opencl_can_ifairy64_mul_mat()`
- `ggml_opencl_ifairy64_kernel_ready()`
- `ggml_opencl_supports_ifairy_op()`

实现某个算子时，不要一开始就把宽泛 shape 全部打开。先从最小白名单开始，测试通过后逐步扩大。

### 推荐实现顺序

1. `GGML_OP_MUL_MAT`: `IFAIRY64 x F32 -> F32`，decode 单 token / 小 batch。
2. `GGML_OP_IFAIRY_SPLIT`
3. `GGML_OP_IFAIRY_MERGE`
4. `GGML_OP_IFAIRY_ADD`
5. `GGML_OP_IFAIRY_MUL`
6. `GGML_UNARY_OP_IFAIRY_RELU2`
7. `GGML_OP_IFAIRY_RMSNORM`
8. `GGML_OP_IFAIRY_ROPE`
9. 扩展 `MUL_MAT` 的 batch/prefill shape。
10. 可选：OpenCL activation quantization / LUT packed path。

这个顺序先打通主干 matmul，再补 Fairy2i graph 中围绕 complex tensor 的轻量算子，最后处理更复杂的 norm/rope 和性能优化。

## 单个算子的完整开发流程

不要把“写一个 OpenCL 算子”理解成只写 `.cl` kernel。对 `ggml` 后端来说，一个算子真正完成，需要同时完成语义确认、能力声明、kernel 编译、C++ 调度、设备内存访问、CPU 对照测试和 fallback 测试。建议每个算子都按下面流程推进。

### 阶段 0：给算子写一张实现卡

开始动代码前，先把这个算子的边界写成一张小表。这个表不一定要提交到文档里，但开发时必须有。

示例：

```text
op: GGML_OP_IFAIRY_MUL
第一版支持: F32 src0, F32 src1, F32 dst, contiguous, no view
第一版不支持: F16, non-contiguous, arbitrary broadcast
CPU reference: ggml_compute_forward_ifairy_mul()
OpenCL kernel: kernel_ifairy_mul_f32
C++ wrapper: ggml_cl_ifairy_mul()
supports gate: ggml_opencl_can_ifairy_mul()
正确性要求: CPU vs OpenCL abs_diff <= 1e-6
fallback cases: type mismatch, non-contiguous, unsupported broadcast, env disabled
```

这一步的目的不是写得漂亮，而是防止后面 `supports_op` 打开过宽。只要一个条件没有写进实现卡，就默认不支持。

### 阶段 1：追清楚 CPU 语义

先不要写 OpenCL。先找到这个 op 的三类代码：

- graph 构造入口：谁创建这个 op，shape 是怎么来的。
- CPU reference：CPU 后端怎么计算。
- 现有测试：有没有已经覆盖相关语义。

常用定位命令：

```bash
rg -n "GGML_OP_IFAIRY_MUL|ggml_ifairy_mul|ifairy_mul" ggml/src tests src
rg -n "GGML_UNARY_OP_IFAIRY_RELU2|ggml_ifairy_relu2|ifairy_relu2" ggml/src tests src
rg -n "GGML_OP_IFAIRY_ROPE|ggml_ifairy_rope|ifairy_rope" ggml/src tests src
```

读 CPU reference 时至少记录 5 件事：

- `src0/src1/dst` 的 type 约束。
- `ne[]` 表示的逻辑 shape，以及 `nb[]` 是否参与寻址。
- 是否允许 broadcast、repeat、view、non-contiguous。
- `op_params` 里有哪些参数，单位是什么。
- 浮点运算顺序是否影响误差容差。

对 `MUL_MAT` 还必须额外确认 complex 语义：

```text
real = wr*xr + wi*xi
imag = wi*xr - wr*xi
```

如果这里还没有把 CPU index mapping 讲清楚，不要进入 kernel 阶段。

### 阶段 2：先写 capability gate，不写 kernel

每个算子先新增一个很窄的 `can_*` helper。位置：

- `ggml/src/ggml-opencl/ggml-opencl.cpp`

建议命名：

```cpp
static bool ggml_opencl_can_ifairy_split(const struct ggml_tensor * op);
static bool ggml_opencl_can_ifairy_merge(const struct ggml_tensor * op);
static bool ggml_opencl_can_ifairy_add(const struct ggml_tensor * op);
static bool ggml_opencl_can_ifairy_mul(const struct ggml_tensor * op);
static bool ggml_opencl_can_ifairy_relu2(const struct ggml_tensor * op);
static bool ggml_opencl_can_ifairy_rmsnorm(const struct ggml_tensor * op);
static bool ggml_opencl_can_ifairy_rope(const struct ggml_tensor * op);
```

第一版 helper 的风格应该偏保守：

```cpp
static bool ggml_opencl_can_ifairy_mul(const struct ggml_tensor * op) {
    if (op->op != GGML_OP_IFAIRY_MUL) {
        return false;
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
        return false;
    }

    if (src0->view_src != nullptr || src1->view_src != nullptr || op->view_src != nullptr) {
        return false;
    }

    return ggml_nelements(src0) == ggml_nelements(op) &&
           ggml_nelements(src1) == ggml_nelements(op);
}
```

这段不是要求照抄，而是说明原则：每个条件都显式检查，没检查的能力就不声明支持。

然后在 `ggml_opencl_supports_ifairy_op()` 里接入这个 helper，但 kernel 未完成前仍让最终返回 false。例如：

```cpp
if (ggml_opencl_can_ifairy_mul(op)) {
    return ggml_opencl_ifairy64_kernel_ready(op);
}
```

`ggml_opencl_ifairy64_kernel_ready()` 可以先按 op 返回 false。这样第一步提交只改变“代码知道这个 shape 将来可以支持”，不会让 scheduler 把 graph 交给还不存在的 kernel。

### 阶段 3：先写 fallback 测试

在 kernel 还没实现时，先加负向测试，证明 OpenCL 不会错误接管 unsupported op。位置：

- `tests/test-ifairy.cpp`

建议覆盖：

- `GGML_OPENCL_IFAIRY64` 没设置时，所有 iFairy OpenCL op 都不支持。
- type 不匹配时不支持。
- non-contiguous 或 view 不支持。
- 当前算子未 ready 时不支持。

测试应该直接检查 device capability：

```text
ggml_backend_dev_supports_op(opencl_dev, op) == false
```

如果测试环境没有 OpenCL device，这类测试要 skip，不要 fail。这样 CI 或开发机器没有 OpenCL 时仍能跑普通 iFairy 测试。

这个阶段完成后，应该能证明：即使代码里新增了 helper，真实推理仍会 fallback 到 CPU。

### 阶段 4：写最小 kernel，只追求可验证

新增或扩展：

- `ggml/src/ggml-opencl/kernels/ifairy64.cl`

第一版 kernel 要优先选择简单、可对照的写法：

- 搬运类 op：一个 work-item 处理一个 scalar。
- elementwise op：一个 work-item 处理一个 scalar 或一个 complex pair。
- reduction op：一个 workgroup 处理一行。
- matmul decode：一个 workgroup 处理一个输出元素或一个输出 row。

每个 kernel 都必须有越界保护：

```c
const int i = get_global_id(0);
if (i >= ne) {
    return;
}
```

第一版不要用复杂 tile、subgroup、image object 或 vectorized layout。先用 raw buffer，保证 CPU/OpenCL 对照可以快速定位错误。性能优化放到正确性通过之后。

写 kernel 时参数顺序建议固定为：

1. `src0` buffer 和 offset。
2. `src1` buffer 和 offset，如果有。
3. `dst` buffer 和 offset。
4. `ne[]`。
5. `nb[]`，如果支持 stride。
6. `op_params` 或 scalar 参数。

新手常见错误是 C++ wrapper 和 `.cl` 参数顺序不一致。建议 C++ 侧用递增的 `cl_uint k = 0` 设置参数，避免手动编号跳号。

### 阶段 5：把 kernel 加进 OpenCL 编译路径

需要改：

- `ggml/src/ggml-opencl/kernels/ifairy64.cl`
- `ggml/src/ggml-opencl/CMakeLists.txt`
- `ggml/src/ggml-opencl/ggml-opencl.cpp`

接线顺序：

1. 在 CMake 的 `GGML_OPENCL_KERNELS` 加 `ifairy64`。
2. 在 `ggml_backend_opencl_context` 加 `program_ifairy64` 和对应 `kernel_*` 字段。
3. 在 `load_cl_kernels()` 中读取 `ifairy64.cl`。
4. 用 `build_program_from_source()` 编译。
5. 用 `clCreateKernel()` 创建每个 kernel。

如果启用了 embedded kernels，`ifairy64.cl` 会生成对应 `.cl.h`。因此新增 kernel 文件后，至少要跑一次 OpenCL 构建，确认普通文件模式和 embed 模式都没有遗漏。第一轮可以先验证普通文件模式：

```bash
cmake -B build-opencl -DCMAKE_BUILD_TYPE=Release \
  -DGGML_IFAIRY_LUT_CPU=OFF \
  -DGGML_OPENCL=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
  -DGGML_OPENCL_EMBED_KERNELS=OFF

cmake --build build-opencl --target ggml-opencl -j 2
```

这个阶段的验收只是“kernel 能编译、backend 能初始化”。还不是 correctness 验收。

### 阶段 6：写 C++ wrapper

位置：

- `ggml/src/ggml-opencl/ggml-opencl.cpp`

wrapper 的职责是把 `ggml_tensor` 翻译成 OpenCL kernel 参数。它不应该重新实现复杂数学。

普通 F32 tensor 使用：

```cpp
ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
cl_mem data = extra->data_device;
cl_ulong offset = extra->offset + tensor->view_offs;
```

`GGML_TYPE_IFAIRY64` weight 使用前置工作里新增的 extra：

```cpp
ggml_tensor_extra_cl_ifairy64 * extra_w =
    (ggml_tensor_extra_cl_ifairy64 *) src0->extra;
cl_mem q = extra_w->q;
cl_mem d = extra_w->d;
```

第一版如果不支持 view，就在 `can_*` 返回 false，并在 wrapper 里用 `GGML_ASSERT(tensor->view_src == nullptr)` 保护。不要在 kernel 里假装支持 stride。

wrapper 写完后，检查这些点：

- `src0/src1/dst` 都有 `extra`。
- `dst->extra` 是普通 OpenCL buffer，而不是 iFairy weight extra。
- 所有 `ne[]` 从 `int64_t` 转 `int` 前都在合理范围内。
- `global_work_size` 覆盖所有输出元素。
- kernel 内有越界保护，所以 global size 可以向上取整。
- 所有 `clSetKernelArg()` 都用 `CL_CHECK` 包住。

### 阶段 7：新增 iFairy compute 分发

当前 `ggml_cl_compute_forward()` 对 iFairy op 会直接返回 false。第一个 kernel 写好后，建议改成独立 helper：

```cpp
static bool ggml_cl_compute_forward_ifairy(ggml_backend_t backend, ggml_tensor * tensor);
```

分发逻辑建议类似：

```cpp
static bool ggml_cl_compute_forward_ifairy(ggml_backend_t backend, ggml_tensor * tensor) {
    switch (tensor->op) {
        case GGML_OP_IFAIRY_MUL:
            if (!ggml_opencl_can_ifairy_mul(tensor)) {
                return false;
            }
            ggml_cl_ifairy_mul(backend, tensor->src[0], tensor->src[1], tensor);
            return true;
        case GGML_OP_UNARY:
            if (ggml_get_unary_op(tensor) != GGML_UNARY_OP_IFAIRY_RELU2) {
                return false;
            }
            if (!ggml_opencl_can_ifairy_relu2(tensor)) {
                return false;
            }
            ggml_cl_ifairy_relu2(backend, tensor->src[0], tensor);
            return true;
        default:
            return false;
    }
}
```

然后把入口改成：

```cpp
if (ggml_opencl_is_ifairy_op(tensor)) {
    return ggml_cl_compute_forward_ifairy(backend, tensor);
}
```

这个 helper 要和 `supports_op` 使用同一套 `can_*` 条件。否则 scheduler 认为支持的 op，compute 阶段可能又返回 false，真实运行会变得难以排查。

### 阶段 8：打开 kernel-ready gate

当 wrapper、kernel、正向测试都准备好以后，再让 ready gate 对这个 op 返回 true。

建议写成明确 switch，而不是一个全局布尔：

```cpp
static bool ggml_opencl_ifairy64_kernel_ready(const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_IFAIRY_MUL:
            return ggml_opencl_can_ifairy_mul(op);
        default:
            return false;
    }
}
```

如果是 unary subtype，要继续检查 subtype：

```cpp
case GGML_OP_UNARY:
    return ggml_get_unary_op(op) == GGML_UNARY_OP_IFAIRY_RELU2 &&
           ggml_opencl_can_ifairy_relu2(op);
```

这一步是最容易破坏 fallback 的地方。每打开一个 op，都要同步补对应的 unsupported shape 测试。

### 阶段 9：写 CPU vs OpenCL 正向测试

每个算子都要有同一输入下的 CPU/OpenCL 对照。推荐测试结构：

1. 构造 deterministic 输入。
2. 用 CPU backend 跑一遍 graph，保存输出。
3. 用 OpenCL backend 跑同一个逻辑 graph，保存输出。
4. 比较输出。
5. 再构造 unsupported shape，确认 OpenCL 不支持或 scheduler fallback。

输入数据不要只用随机数。每个算子至少要有一个“能暴露符号/索引错误”的人工 case：

- split/merge：递增整数，例如 `0, 1, 2, ...`，最容易发现搬运错位。
- add/mul/relu2：包含负数、0、小正数、大正数。
- rmsnorm：包含全 0 附近、极小值、普通值。
- rope：position 0 和 position 1 必测。
- matmul：构造 `wr/wi/xr/xi` 不对称的 case，用来抓 `conj` 符号错误。

容差按算子类型定：

- 纯搬运：bitwise exact。
- 单元素 F32 add/mul/relu：通常 bitwise exact。
- rope/rmsnorm：`abs_diff <= 1e-5` 或 `rel_diff <= 1e-5`。
- raw decode matmul：第一版可用 `abs_diff <= 1e-3`，稳定后再收紧。

### 阶段 10：失败时按层定位

如果 CPU/OpenCL 不一致，不要直接改 kernel。按下面顺序缩小问题：

1. `supports_op` 是否真的把这个 op 放到 OpenCL。
2. `ggml_cl_compute_forward_ifairy()` 是否进入了正确 case。
3. wrapper 传入的 `ne[]/nb[]/offset` 是否和 CPU 逻辑一致。
4. kernel 的 global id 到 logical index 映射是否正确。
5. 对 `MUL_MAT`，先把 K 降到 64，只处理一个 block。
6. 对 `MUL_MAT`，构造只有 real 或只有 imag 的输入，分别验证四个分量：
   - `wr*xr`
   - `wi*xi`
   - `wi*xr`
   - `wr*xi`
7. 如果只有大 shape 错，小 shape 对，优先检查 stride、batch index、global size 向上取整和越界 guard。

建议临时在 kernel 中写 debug-friendly 输出，例如让 dst 写入 source index 或 block id。确认 index mapping 后再恢复数学计算。

### 阶段 11：做性能优化前的冻结点

第一个 correctness 版本合入前，不要做复杂优化。先冻结一个简单版本，后续性能 commit 才容易回归定位。

可以进入性能优化的条件：

- 小 shape correctness 通过。
- model-like shape correctness 通过。
- fallback 测试通过。
- CPU-only iFairy LUT 测试仍通过。
- `git diff --check` 通过。

然后再考虑：

- vectorized load/store。
- 多输出元素 per workgroup。
- local memory 缓存 activation。
- subgroup reduction。
- `MUL_MAT` decode/prefill 分 kernel。
- LUT packed layout。

### 阶段 12：推荐提交粒度

一个算子不要一次性提交一个巨大 commit。建议拆成：

1. `opencl: gate ifairy <op>`：只加 `can_*`、fallback 测试、文档状态。
2. `opencl: add ifairy <op> kernel skeleton`：加 `.cl`、CMake、program/kernel 创建，仍不打开 ready gate。
3. `opencl: wire ifairy <op>`：加 wrapper、compute 分发、ready gate、CPU vs OpenCL 测试。
4. `opencl: expand ifairy <op> shapes`：每扩大一种 shape，配对应测试。

这样任何一个阶段出问题，都可以明确知道是 capability、kernel 编译、调度还是数学语义的问题。

## OpenCL 后端接线方法

每新增一个 OpenCL kernel，通常要经过下面这些接线点。前一节讲“开发顺序”，本节讲“代码具体接在哪里”。

### 1. 新增 `.cl` 文件

位置：

- `ggml/src/ggml-opencl/kernels/`

建议新增：

- `ifairy64.cl`

也可以后续拆分为：

- `ifairy64_mul_mat.cl`
- `ifairy64_ops.cl`

第一版建议放在一个 `ifairy64.cl`，减少 CMake 和 program 管理成本。等 kernel 数量稳定后再拆。

### 2. 加入 CMake kernel 列表

位置：

- `ggml/src/ggml-opencl/CMakeLists.txt`

在 `GGML_OPENCL_KERNELS` 里加入：

```cmake
ifairy64
```

这样 `GGML_OPENCL_EMBED_KERNELS=ON` 时会被 embed，关闭 embed 时会复制到 runtime 输出目录。

### 3. 在 OpenCL context 中保存 program/kernel

位置：

- `ggml/src/ggml-opencl/ggml-opencl.cpp`
- `struct ggml_backend_opencl_context`

新增字段，例如：

```cpp
cl_program program_ifairy64;
cl_kernel  kernel_ifairy64_mul_mat_f32;
cl_kernel  kernel_ifairy_split_f32;
```

命名要和已有 OpenCL backend 风格一致：`program_*` 保存 program，`kernel_*` 保存具体 kernel。

### 4. 编译 program 并创建 kernel

位置：

- `ggml/src/ggml-opencl/ggml-opencl.cpp`
- 搜索 `read_file("mul_mv_f32_f32.cl")` 或 `read_file("rms_norm.cl")`

按已有模式加入：

```cpp
const std::string kernel_src = read_file("ifairy64.cl");
backend_ctx->program_ifairy64 =
    build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
CL_CHECK((backend_ctx->kernel_ifairy64_mul_mat_f32 =
    clCreateKernel(backend_ctx->program_ifairy64, "kernel_ifairy64_mul_mat_f32", &err), err));
```

如果 kernel 只支持某些 GPU family，不要在创建时跳过。更好的做法是在 `supports_op` 中按条件返回 false。

### 5. C++ wrapper 调 kernel

位置：

- `ggml/src/ggml-opencl/ggml-opencl.cpp`

已有常见 wrapper 模式：

- `ggml_cl_mul_mat()`
- `ggml_cl_rms_norm()`
- `ggml_cl_rope()`
- `ggml_cl_add()`
- `ggml_cl_mul()`

新增 iFairy wrapper 建议放在普通 op wrapper 附近，例如：

```cpp
static void ggml_cl_ifairy64_mul_mat(...);
static void ggml_cl_ifairy_split(...);
```

wrapper 职责：

- 检查 `src` / `dst` / `extra` 不为空。
- 从 tensor 读取 `ne` / `nb` / `view_offs`。
- 从 `ggml_tensor_extra_cl_ifairy64` 拿 `q` / `d`。
- `clSetKernelArg`。
- 计算 global/local size。
- 调 `backend_ctx->enqueue_ndrange_kernel(...)`。

### 6. 在 compute switch 接线

位置：

- `ggml_cl_compute_forward()`

当前 iFairy op 会提前：

```cpp
if (ggml_opencl_is_ifairy_op(tensor)) {
    return false;
}
```

当某个 kernel 完成后，要改成：

```cpp
if (ggml_opencl_is_ifairy_op(tensor)) {
    return ggml_cl_compute_forward_ifairy(backend, tensor);
}
```

新增 helper：

```cpp
static bool ggml_cl_compute_forward_ifairy(ggml_backend_t backend, ggml_tensor * tensor);
```

不要把 iFairy case 混进普通 `switch` 的很多分支里。单独 helper 更容易保证 fallback 边界。

### 7. 打开 capability gate

位置：

- `ggml_opencl_ifairy64_kernel_ready()`
- `ggml_opencl_supports_ifairy_op()`

每完成一个算子，只打开对应 op 和 shape。例如第一步只打开：

```cpp
return ggml_opencl_can_ifairy64_mul_mat(op);
```

不要直接对所有 `ggml_opencl_is_ifairy_op()` 返回 true。

## 算子 1：`GGML_OP_MUL_MAT` / `IFAIRY64 x F32 -> F32`

### 功能

这是最核心的算子。模型中的 Fairy2i linear 权重是 `GGML_TYPE_IFAIRY64`，输入 activation 第一版按 `F32` 读取，输出 `F32`。

输入输出关系：

- `src0`: weight，shape `[K, M, 1, 1]`，type `GGML_TYPE_IFAIRY64`
- `src1`: activation，shape `[K, N, batch2, batch3]`，type `GGML_TYPE_F32`
- `dst`: output，shape `[M, N, batch2, batch3]`，type `GGML_TYPE_F32`

第一版建议只支持：

- `src0->ne[2] == 1`
- `src0->ne[3] == 1`
- `src0` contiguous
- `src1` contiguous
- `K % 64 == 0`

### 代码位置

OpenCL kernel：

- `ggml/src/ggml-opencl/kernels/ifairy64.cl`

C++ wrapper：

- `ggml/src/ggml-opencl/ggml-opencl.cpp`
- 新增 `ggml_cl_ifairy64_mul_mat_f32()`

调度白名单：

- `ggml_opencl_can_ifairy64_mul_mat()`
- `ggml_opencl_ifairy64_kernel_ready()`
- `ggml_cl_compute_forward_ifairy()`

CPU reference：

- `ggml/src/ggml-cpu/quants.c`
- `ggml_vec_dot_ifairy64_q16_K()`
- `ggml_vec_dot_ifairy64_q16_K_generic()`
- `dequantize_row_ifairy64()`

### 实现建议

第一版不要追求最优性能。目标是写出一个容易验证的 raw decode kernel。

推荐 kernel 粒度：

- 一个 work-item 或一个 small workgroup 负责一个 `(m, n, batch)` 输出。
- 循环 K 维，每 64 个元素读取一个 `block_ifairy64`。
- 从 `q` 读取 packed 2-bit code。
- 从 `d` 读取 `d_real` / `d_imag`。
- activation 先从 `src1` 读取 F32 pair。

注意：iFairy complex tensor 通常把 real/imag 以特殊方式展开。写 kernel 前必须从 CPU reference 和 graph 里确认 activation 的 memory convention。不要靠猜。推荐先写一个 CPU-side test helper 生成极小 K=64 的人工输入，再对比 OpenCL 输出。

### 正确性测试

建议在 `tests/test-ifairy.cpp` 里新增 OpenCL-gated 测试：

- 测试名建议：`test_ifairy64_opencl_mul_mat_f32_decode()`
- 只有检测到 OpenCL backend 且 `GGML_OPENCL_IFAIRY64=1` 时运行。
- 小 shape：
  - `K=64, M=1, N=1`
  - `K=128, M=3, N=1`
  - `K=256, M=17, N=2`
- 随机权重用现有 `quantize_row_ifairy64_ref()` 或测试内构造 `block_ifairy64`。
- CPU reference 用 CPU backend 跑同一个 graph，或直接调用 reference helper 计算。
- OpenCL 输出和 CPU 输出比较：
  - 第一版 raw decode 建议 `abs_diff <= 1e-3`。
  - 如果 kernel 内做了与 CPU 完全相同的 quantization，可以收紧。

同时加 fallback 测试：

- `src1 != F32` 时 OpenCL `supports_op=false`
- 非 contiguous src1 时 `supports_op=false`
- `MUL_MAT_ID` 继续 `supports_op=false`

### 验证命令

```bash
cmake --build build-opencl --target ggml-opencl test-ifairy -j 2
GGML_OPENCL_IFAIRY64=1 ./build-opencl/bin/test-ifairy --ifairy-opencl-only
./build-rel/bin/test-ifairy --ifairy-lut-only
```

如果还没有 `--ifairy-opencl-only`，先给 `tests/test-ifairy.cpp` 加一个只跑 OpenCL 专项测试的 flag。

## 算子 2：`GGML_OP_IFAIRY_SPLIT`

### 功能

`ggml_ifairy_split()` 把 compact complex layout 展开成 real/imag 友好的 layout。它通常用于 attention、FFN 里需要在 real-valued op 和 iFairy complex op 之间转换。

构图入口：

- `ggml/src/ggml.c`
- `ggml_ifairy_split_impl()`

CPU reference：

- `ggml/src/ggml-cpu/ops.cpp`
- `ggml_compute_forward_ifairy_split_impl()`

### 代码位置

OpenCL kernel：

- `ggml/src/ggml-opencl/kernels/ifairy64.cl`
- kernel 名建议：`kernel_ifairy_split_f32`

C++ wrapper：

- `ggml_cl_ifairy_split()`

### 实现建议

先只支持 `F32 -> F32`，并要求 contiguous。按 CPU reference 的 index mapping 搬运元素，不要重新定义 layout。

建议写法：

- 每个 work-item 处理一个 scalar。
- 根据 CPU reference 的公式计算 source offset 和 destination offset。
- 对任意 rank 使用 `ne` / `nb` 传参，第一版可先限制 contiguous，让 offset 公式更简单。

### 测试

在 `tests/test-ifairy.cpp` 新增：

- 输入人工填充递增数字，便于发现 index 错误。
- shape 覆盖：
  - 1 token
  - 多 token
  - odd-ish channel count，但必须满足 op 本身约束
- CPU graph vs OpenCL graph bitwise 比较。

验收：

- 对纯搬运 op，要求 bitwise exact。

## 算子 3：`GGML_OP_IFAIRY_MERGE`

### 功能

`ggml_ifairy_merge()` 是 split 的反向布局转换。它把 split 后的 real/imag layout 合并回 compact complex layout。

构图入口：

- `ggml/src/ggml.c`
- `ggml_ifairy_merge_impl()`

CPU reference：

- `ggml/src/ggml-cpu/ops.cpp`
- `ggml_compute_forward_ifairy_merge_impl()`

### 实现建议

这应该和 split 成对实现。建议同一个 `.cl` 文件里放：

- `kernel_ifairy_split_f32`
- `kernel_ifairy_merge_f32`

测试时一定要加 roundtrip：

```text
x -> split -> merge == x
x -> merge -> split == x  // 如果 shape 合法
```

验收：

- bitwise exact。

## 算子 4：`GGML_OP_IFAIRY_ADD`

### 功能

complex layout 下的逐元素加法，用于 residual 等路径。

构图入口：

- `ggml/src/ggml.c`
- `ggml_ifairy_add_impl()`

CPU reference：

- `ggml/src/ggml-cpu/binary-ops.cpp`
- `ggml_compute_forward_ifairy_add()`

### 实现建议

第一版只支持：

- `src0->type == F32`
- `src1->type == F32`
- `dst->type == F32`
- contiguous
- `ggml_can_repeat(src1, src0)` 里最常见的 broadcast 模式

不要直接支持所有 broadcast。先覆盖 Fairy2i graph 中实际出现的 residual broadcast。

### 测试

测试内容：

- same-shape add
- residual 常见 broadcast
- unsupported broadcast fallback

验收：

- bitwise exact 或 `abs_diff == 0`，因为 F32 add 顺序和 CPU 单元素一致。

## 算子 5：`GGML_OP_IFAIRY_MUL`

### 功能

complex layout 下的逐元素乘法，FFN gate 路径会用到。

构图入口：

- `ggml/src/ggml.c`
- `ggml_ifairy_mul_impl()`

CPU reference：

- `ggml/src/ggml-cpu/binary-ops.cpp`
- `ggml_compute_forward_ifairy_mul()`

### 实现建议

先确认 CPU reference 是逐元素 real multiply，还是按 complex 语义组合 real/imag。不要假设它等同普通 `GGML_OP_MUL`。

实现上仍按 CPU reference 的 index 公式写。

第一版限制：

- F32 only
- contiguous
- 常见 broadcast only

### 测试

测试内容：

- same-shape multiply
- gate/up shape multiply
- 包含正负数和 0

验收：

- 如果只是逐元素 F32 乘，允许 bitwise exact。
- 如果涉及 complex cross term，使用 `abs_diff <= 1e-6`。

## 算子 6：`GGML_UNARY_OP_IFAIRY_RELU2`

### 功能

iFairy 专用 unary activation。Fairy2i FFN gate 中会用：

- 构图调用：`ggml_ifairy_relu2()`
- ggml op：`GGML_OP_UNARY`
- unary subtype：`GGML_UNARY_OP_IFAIRY_RELU2`

CPU reference：

- `ggml/src/ggml-cpu/unary-ops.cpp`
- `ggml_compute_forward_ifairy_relu2()`

### 代码位置

OpenCL 后端普通 unary 在：

- `ggml_cl_compute_forward()`
- `GGML_OP_UNARY` switch
- `ggml_cl_relu()` / `ggml_cl_gelu()` 等 wrapper
- `ggml/src/ggml-opencl/kernels/relu.cl`

建议不要把 iFairy relu2 塞进普通 `relu.cl`。放进 `ifairy64.cl`，并由 `ggml_cl_compute_forward_ifairy()` 分发。

### 测试

输入覆盖：

- 负数
- 0
- 小正数
- 大正数
- real/imag 交错位置

验收：

- bitwise exact，或按 CPU reference 允许极小误差。

## 算子 7：`GGML_OP_IFAIRY_RMSNORM`

### 功能

iFairy layout 下的 RMSNorm。它和普通 `GGML_OP_RMS_NORM` 的区别是输入 layout 和归约维度可能不同，不能直接复用普通 OpenCL rms_norm kernel。

构图入口：

- `ggml/src/ggml.c`
- `ggml_ifairy_rms_norm_impl()`

CPU reference：

- `ggml/src/ggml-cpu/ops.cpp`
- `ggml_compute_forward_ifairy_rmsnorm()`

### 实现建议

先读 CPU reference，确认：

- 每一行的归约长度。
- real/imag 是否一起参与 RMS。
- epsilon 从 `op_params` 如何读取。

OpenCL 实现可以参考普通 RMSNorm：

- `ggml/src/ggml-opencl/kernels/rms_norm.cl`
- `ggml_cl_rms_norm()`

第一版：

- 一个 workgroup 处理一行。
- local memory 做 sum of squares reduction。
- 写回 F32。
- 限制 `ne[0]` 到已有 OpenCL 可处理 workgroup size 范围。

### 测试

测试内容：

- 小维度手算 case。
- 多行。
- 不同 eps。
- 接近 0 的输入，避免除零/NaN。

验收：

- `abs_diff <= 1e-5` 或 `rel_diff <= 1e-5`。
- 不允许 NaN。

## 算子 8：`GGML_OP_IFAIRY_ROPE`

### 功能

iFairy layout 下的 rotary embedding。attention 的 Q/K 会经过它。

构图入口：

- `ggml/src/ggml.c`
- `ggml_ifairy_rope_impl()`

CPU reference：

- `ggml/src/ggml-cpu/ops.cpp`
- `ggml_compute_forward_ifairy_rope()`

普通 OpenCL rope 可参考：

- `ggml/src/ggml-opencl/kernels/rope.cl`
- `ggml_cl_rope()`

### 实现建议

这是最容易写错的轻量算子，建议放在后面实现。

先支持 Fairy2i 当前实际用到的最小模式：

- `mode == 0`
- 不支持 mrope
- 不支持 vision rope
- `src0 == F32`
- `positions == I32`
- contiguous

从 `op_params` 读取：

- `n_dims`
- `mode`
- `n_ctx_orig`
- `freq_base`
- `freq_scale`
- `ext_factor`
- `attn_factor`
- `beta_fast`
- `beta_slow`

即使第一版只支持一部分参数，也要在 `supports_op` 里严格检查，不支持就 fallback。

### 测试

测试内容：

- position 0：输出应等于输入。
- position 1/2：和 CPU reference 对齐。
- 多 token positions。
- `n_dims` 小值和实际模型常见值。

验收：

- `abs_diff <= 1e-5`。
- 对 position 0 的 case 尽量 bitwise exact。

## 算子 9：扩展 `MUL_MAT` batch / prefill

### 功能

第一版 matmul 通常会先覆盖 decode，即 `N` 很小。prefill 时 `N` 更大，访问模式和并行策略不同。

### 实现建议

分两条 kernel：

- decode/gemv kernel：`N <= 4` 或 `N == 1`
- prefill/gemm kernel：`N > 4`

decode 关注每个输出 row 的 K 维归约；prefill 关注 tile 化 `M x N` 输出。

不要过早优化。先让 prefill path 正确运行，再做：

- vectorized code decode
- K tile
- M tile
- local memory caching activation
- subgroup reduction

### 测试

shape 覆盖：

- decode: `N=1`
- small batch: `N=2,4`
- prefill: `N=8,16,32`
- odd M: `M=17`
- actual model-like M/K，例如从 Fairy2i 32B 抽取几组层大小

验收：

- 与 CPU reference 误差稳定。
- 不同 N 下结果一致。

## 可选路径：OpenCL activation quantization / LUT

### 何时做

不要在 raw decode matmul 正确前做 LUT。LUT path 会同时引入：

- activation quantization
- per-column LUT preprocess
- packed weight tile
- qgemm kernel

如果 raw decode 已经正确但性能不足，再开始 LUT。

### 代码参考

CPU LUT 参考：

- `ggml/src/ggml-ifairy-lut.h`
- `ggml/src/ggml-ifairy-lut.cpp`
- `ggml/src/ggml-ifairy-lut-transform.cpp`
- `ggml/src/ggml-ifairy-lut-preprocess.cpp`
- `ggml/src/ggml-ifairy-lut-qgemm.cpp`

OpenCL 可能新增：

- `ggml_tensor_extra_cl_ifairy64_lut`
- `kernel_ifairy64_preprocess_lut`
- `kernel_ifairy64_qgemm_lut`

### 测试

LUT path 必须和 raw decode path 对比：

- CPU vecdot reference
- CPU LUT reference
- OpenCL raw decode
- OpenCL LUT

性能测试前，先做 bitwise/容差矩阵。

## 测试设计总览

### 单元测试位置

首选：

- `tests/test-ifairy.cpp`

可以新增 flags：

- `--ifairy-opencl-only`
- `--ifairy-opencl-mulmat-only`
- `--ifairy-opencl-ops-only`

这样没有 OpenCL 环境时，普通测试不会被迫依赖设备。

### Backend comparison 测试

如果要接入通用 backend test：

- `tests/test-backend-ops.cpp`

但 iFairy ops 比较特殊，建议先在 `test-ifairy.cpp` 中写专项测试。等 shape 支持稳定后，再考虑放入通用 backend ops。

### 测试写法模板

每个算子测试都应有两条路径：

1. CPU reference path
   - CPU backend 执行同一个 graph。
   - 或直接调用 CPU reference helper。
2. OpenCL path
   - OpenCL backend 执行同一个 graph。
   - 设置 `GGML_OPENCL_IFAIRY64=1`。

比较：

- 搬运类算子：bitwise exact。
- elementwise F32：通常 bitwise exact。
- reduction / norm / rope / matmul：用 abs/rel tolerance。

### 正向测试的落地细节

只比较输出不够。因为如果 OpenCL 没有接管，scheduler fallback 到 CPU，输出也会正确。每个正向测试至少要证明两件事：

- positive case 下 `ggml_backend_dev_supports_op(opencl_dev, op) == true`。
- 同一逻辑输入下，OpenCL 输出和 CPU reference 输出一致。

推荐把测试拆成 3 层 helper：

```text
make_case()
  只生成原始输入数组、shape、op 参数，不依赖任何 backend。

build_graph(ctx, case)
  在给定 ggml_context 里创建 tensor 和 op。
  CPU 和 OpenCL 路径各建一次，不共享 tensor 指针。

run_and_compare(case)
  先跑 CPU reference，保存 expected。
  再跑 OpenCL，保存 actual。
  最后按该算子的 tolerance 比较。
```

这样写的好处是：输入生成、graph 构造、backend 执行三件事互相独立。出现错误时能快速判断是数据问题、shape 问题还是 OpenCL kernel 问题。

输入填充建议：

- 用固定 seed，不用当前时间。
- 小 shape 用人工 pattern，例如 `value = i * 0.25f - 3.0f`。
- 对 complex 语义，手写几组 `wr/wi/xr/xi`，不要只依赖随机数。
- 对 broadcast，单独构造能看出 broadcast 维度的 pattern，例如每行/每列加不同偏移。

graph 构造建议：

- CPU path 和 OpenCL path 各自创建 `ggml_context`，避免 tensor `extra`、backend buffer 或 view 状态互相污染。
- 输入 raw vector 保存在 C++ `std::vector` 中，然后分别 copy 到 CPU tensor 和 OpenCL tensor。
- 输出也分别回读到 `std::vector<float>`，不要直接比较 backend 内部指针。
- 每个测试结束释放 backend buffer/context，避免多个 case 之间复用旧 `extra`。

正向测试的基本流程可以写成伪代码：

```text
case = make_ifairy_mul_case(shape, seed)

cpu_graph = build_graph(cpu_ctx, case)
cpu_out = run_graph_on_cpu(cpu_graph)

opencl_graph = build_graph(opencl_ctx, case)
assert(ggml_backend_dev_supports_op(opencl_dev, opencl_graph.dst))
opencl_out = run_graph_on_opencl(opencl_graph)

compare(cpu_out, opencl_out, tolerance)
```

如果使用 scheduler 测试整图，除了比较输出，还要先对目标 node 调 `ggml_backend_dev_supports_op()`。否则测试无法区分“OpenCL 算子正确”和“OpenCL 没接、CPU fallback 正确”。

### OpenCL 测试环境处理

OpenCL 专项测试要能在没有 OpenCL device 的机器上跳过。建议规则：

- 找不到 OpenCL backend/device：打印 skip，并返回 pass。
- `GGML_OPENCL_IFAIRY64` 未设置：只跑 env-disabled fallback 测试，不跑 positive correctness。
- positive correctness 必须显式设置 `GGML_OPENCL_IFAIRY64=1`。

不要让普通 `./build-rel/bin/test-ifairy --ifairy-lut-only` 依赖 OpenCL。OpenCL 专项测试应由单独 flag 触发，例如：

```bash
GGML_OPENCL_IFAIRY64=1 ./build-opencl/bin/test-ifairy --ifairy-opencl-only
```

### 调试输出和回归保护

调试时可以临时在 wrapper 里打印 kernel 名、shape 和 global/local size，但提交前要删除。更推荐的长期方法是：

- positive test 断言 `supports_op == true`。
- fallback test 断言 `supports_op == false`。
- 数值 mismatch 时打印前几个不同元素的 index、CPU 值、OpenCL 值、abs diff。
- 对搬运类 op，mismatch 时额外打印 logical index 到 byte offset 的映射。

如果启用了 OpenCL profiling，可以用 `cl_profiling.csv` 确认目标 kernel 被 enqueue。但 profiling 只能作为调试辅助手段，不能替代 correctness 测试。

### Fallback 测试

每打开一个 op，都要加 unsupported case：

- type 不匹配。
- shape 不匹配。
- stride/view 不支持。
- env gate 未打开。

期望：

- `ggml_backend_dev_supports_op(opencl_dev, op) == false`
- scheduler 能把 graph 放到 CPU 完成。

### 最小验证命令

```bash
cmake -B build-opencl -DCMAKE_BUILD_TYPE=Release \
  -DGGML_IFAIRY_LUT_CPU=OFF \
  -DGGML_OPENCL=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
  -DGGML_OPENCL_EMBED_KERNELS=OFF

cmake --build build-opencl --target ggml-opencl test-ifairy -j 2

GGML_OPENCL_IFAIRY64=1 ./build-opencl/bin/test-ifairy --ifairy-opencl-only
./build-rel/bin/test-ifairy --ifairy-lut-only
```

仓库要求的格式/静态检查如果工具可用，也要跑：

```bash
git diff --check
git clang-format --style=file --diff <merge-base> -- '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hh' '*.hpp'
clang-tidy -p build-opencl ggml/src/ggml-opencl/ggml-opencl.cpp
```

## 端到端验证

单元测试通过后，再做模型级 smoke。

建议命令模板：

```bash
GGML_OPENCL_IFAIRY64=1 \
./build-opencl/bin/llama-cli \
  -m /path/to/fairy2i_32b.gguf \
  -ngl 999 \
  -t 4 \
  -c 512 \
  -b 32 \
  -ub 32 \
  -fa off \
  --no-warmup \
  -no-cnv \
  --seed 1234 \
  -n 16 \
  -p "I believe life is"
```

验收：

- 模型能加载。
- 没有 OpenCL assert。
- 没有 NaN。
- 固定 seed 下输出稳定。
- 和 CPU-only path 的 logits 或 token 序列在预期误差内。

性能验证用 `llama-bench`，但只有在 correctness 过关后才记录性能结论。

## 常见错误

### 把 `w * conj(x)` 写成 `w * x`

症状：

- 小测试可能看起来误差不大。
- 模型输出质量明显坏。

解决：

- 用专门构造的 complex case 测试虚部符号。

### `supports_op` 打开太宽

症状：

- 某些 graph split 被 OpenCL 接走，但 compute switch 没实现。
- 运行时 assert `op not supported`。

解决：

- 每个 kernel 完成前保持 kernel-ready gate false。
- 每个 shape 单独白名单。

### 忽略 view/stride

症状：

- test-backend 小 shape 过了，真实模型错。

解决：

- 第一版直接要求 contiguous 和 no view。
- 后续支持 stride 时，要把 `nb[]` 全部传入 kernel。

### `get_tensor` 不能还原 raw bytes

症状：

- backend compare 或 graph copy 到 CPU 失败。

解决：

- 对 `GGML_TYPE_IFAIRY64` 保持 `q`/`d -> raw block_ifairy64` roundtrip 测试。

## 完成定义

一个 OpenCL iFairy 算子只有同时满足下面条件，才算完成：

- `supports_op` 只打开已支持 shape。
- OpenCL kernel 实现对应语义。
- `ggml_cl_compute_forward_ifairy()` 能正确分发。
- `tests/test-ifairy.cpp` 有 CPU vs OpenCL correctness 测试。
- unsupported shape 有 fallback 测试。
- `cmake --build build-opencl --target ggml-opencl test-ifairy -j 2` 通过。
- `./build-rel/bin/test-ifairy --ifairy-lut-only` 仍通过。
- 如果修改了 env gate 或路由，更新 `docs/ifairy/v2/IFAIRY64_STATUS.md`。
