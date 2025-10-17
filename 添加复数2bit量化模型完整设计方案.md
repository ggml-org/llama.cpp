# llama.cpp中添加复数2-bit量化新模型完整设计方案

> **重要说明**: 本文档已根据 iFairy 的实际实现进行重构。与原始理论方案相比，iFairy 采用了更优化的**合并存储**方案，其中每个 2-bit 值直接表示一个离散复数 {-1, +1, -i, +i}，而非分别量化实部和虚部。这种设计显著提高了压缩率（30x vs 7x），并特别适合频域/相位编码的神经网络。

## 目录
1. [需求概述](#需求概述)
2. [核心设计思路](#核心设计思路)
3. [详细实现方案](#详细实现方案)
4. [完整修改清单](#完整修改清单)
5. [测试与验证](#测试与验证)
6. [性能优化建议](#性能优化建议)

---

## 1. 需求概述

### 1.1 需求分析

需要在llama.cpp中添加一个新的模型架构,该模型具有以下特点:

1. **全新的2-bit量化方法** (与现有IQ2_XXS/TQ2_0等不同)
2. **复数权重支持** (Real + Imaginary双重权重)
3. **新架构类型** (非Transformer标准架构)

### 1.2 技术挑战

- **量化格式设计**: 2-bit量化需要高效的存储与反量化
- **复数运算支持**: GGML底层不支持复数,需要扩展
- **架构兼容性**: 保持与现有llama.cpp生态的兼容

### 1.3 命名约定 (iFairy 实际实现)

基于实际实现，使用以下命名:
- **架构名**: `IFAIRY`
- **量化类型**: `GGML_TYPE_IFAIRY`
- **文件前缀**: `ifairy`
- **块结构**: `block_ifairy`
- **函数命名**: `quantize_row_ifairy_ref`, `dequantize_row_ifairy`, `ggml_vec_dot_ifairy_q8_0`

---

## 2. 核心设计思路

### 2.1 复数权重表示 (iFairy 实际实现)

#### 实际采用方案: 合并存储 + 离散复数值

**iFairy 的独特设计**:

```
每个复数权重 = 离散复数值 × 缩放因子

内存布局（每个 2-bit 值直接表示一个复数）:
[q0, q1, q2, q3, ...] 其中 qi ∈ {-1, +1, -i, +i}

量化映射:
  00 → -1 * d_real
  01 → +1 * d_real
  10 → -i * d_imag
  11 → +i * d_imag
```

**与原方案对比**:

| 特性 | iFairy 方案 | 原方案 A（分离存储） | 原方案 B（交错存储） |
|------|-------------|---------------------|---------------------|
| 存储方式 | 合并（每个2-bit表示一个复数） | 分离（Real和Imag各自量化） | 交错（R,I,R,I...） |
| 量化值 | 4个离散复数 {-1,+1,-i,+i} | 4个级别 × 2部分 | 4个级别 × 2部分 |
| 缩放因子 | 全局共享（d_real, d_imag） | 每块独立 | 每块独立 |
| 内存效率 | 最优（30x压缩率） | 较好（7x压缩率） | 较好（7x压缩率） |
| SIMD优化 | 需要特殊处理 | 简单 | 中等 |

**iFairy 优点**:
- **极致压缩**: 每个复数仅用2-bit表示，压缩率达到30x
- **硬件友好**: 量化值恰好对应单位复数的4个象限，适合快速FFT/DFT运算
- **模型兼容**: 特别适合频域/相位编码的神经网络（如iFairy模型）

**实现关键点**:
- 量化时判断复数主要分量（实部或虚部）
- 反量化时根据编码恢复对应的复数值
- 点积需要处理复数乘法规则: `(a+bi)×(c+di) = (ac-bd) + (ad+bc)i`

### 2.2 2-bit量化方案 (iFairy 实际实现)

使用**分组量化 + 合并存储**策略:

```c
// 每个块处理 QK_K (256) 个 2-bit 值
// 由于是复数，每个复数用一个 2-bit 值表示（离散化到 4 个复数值）
#define QK_K 256  // 量化块大小（来自 K-quants）

typedef struct {
    uint8_t qs[QK_K/4];  // 64 字节，存储 256 个 2-bit 值
                         // 每个字节存储 4 个 2-bit 值
                         // 对应 256 个复数的量化值
    ggml_half d_real;    // FP16 缩放因子（实部）
    ggml_half d_imag;    // FP16 缩放因子（虚部）
} block_ifairy;

// 总大小: 64 + 2 + 2 = 68 字节
// 存储 256 个复数 = 512 个 fp32 值 (2048 字节)
// 压缩率: 2048/68 = 30.1x
```

**量化策略**:
- 2-bit表示4个离散复数值: {-1, +1, -i, +i}
- 编码映射:
  - `00` → -1 (实部负)
  - `01` → +1 (实部正)
  - `10` → -i (虚部负)
  - `11` → +i (虚部正)
- 缩放因子全局共享（对所有块）
- 合并存储：实部和虚部在同一个量化数组中

### 2.3 复数矩阵乘法

复数矩阵乘法公式: 使用共轭乘法
```
(A_r + iA_i) × (B_r + iB_i) = (A_r×B_r + A_i×B_i) + i(A_r×B_i - A_i×B_r)
```

需要4次实数矩阵乘法:
1. `Real_result = Real_A × Real_B + Imag_A × Imag_B`
2. `Imag_result = Real_A × Imag_B - Imag_A × Real_B`

---

## 3. 详细实现方案

### 3.1 GGML层修改

#### 3.1.1 添加新量化类型

**文件**: `ggml/include/ggml.h:377-419`

```c
enum ggml_type {
    // ... 现有类型 ...
    GGML_TYPE_MXFP4   = 39,
    GGML_TYPE_IFAIRY  = 40,  // 新增: iFairy 复数2-bit量化
    GGML_TYPE_COUNT   = 41,  // 更新计数
};
```

**注意**:
- 选择下一个可用的枚举值(当前MXFP4=39)
- 更新`GGML_TYPE_COUNT`
- 命名使用 `IFAIRY` 而非 `CQ2_0`

#### 3.1.2 定义量化块结构

**文件**: `ggml/src/ggml-common.h` (新增部分)

```c
// 注意：QK_K 已在 ggml-common.h 中定义，通常为 256

// iFairy 量化块结构
typedef struct {
    uint8_t qs[QK_K/4];  // 64 字节: 存储 256 个 2-bit 值
                         // 每个 2-bit 值表示一个离散复数: {-1, +1, -i, +i}
    ggml_half d_real;    // FP16 缩放因子（实部）
    ggml_half d_imag;    // FP16 缩放因子（虚部）
} block_ifairy;

static_assert(sizeof(block_ifairy) == QK_K/4 + 2*sizeof(ggml_half),
              "block_ifairy size incorrect");
```

**关键点**:
- 使用 `QK_K` 宏（与 K-quants 系列保持一致）
- 每个块存储 256 个离散复数值
- 总大小：68 字节（64 + 2 + 2）

#### 3.1.3 注册类型特征

**文件**: `ggml/src/ggml.c` (类型特征表)

```c
// 在 ggml_type_traits_cpu 数组中添加
[GGML_TYPE_IFAIRY] = {
    .type_name       = "ifairy",
    .blck_size       = QK_K,
    .type_size       = sizeof(block_ifairy),
    .is_quantized    = true,
    .to_float        = (ggml_to_float_t) dequantize_row_ifairy,
    .from_float      = quantize_ifairy,
    .from_float_ref  = (ggml_from_float_t) quantize_row_ifairy_ref,
    .vec_dot         = ggml_vec_dot_ifairy_q8_0,  // 使用Q8_0作为对照
    .vec_dot_type    = GGML_TYPE_Q8_0,
    .nrows           = 1,
},
```

**注意**:
- `blck_size` 设置为 `QK_K` (256)
- `vec_dot` 函数需要专门实现复数点积
- `to_float` 和 `from_float_ref` 的签名与标准量化不同（需要处理复数）

### 3.2 量化实现

#### 3.2.1 量化函数

**文件**: `ggml/src/ggml-cpu/quants.h` 或 `ggml/src/ggml-quants.h`

```c
// 函数声明（注意：与标准量化不同，需要处理实部和虚部）
void quantize_row_ifairy_ref(const float * GGML_RESTRICT x_real,
                              const float * GGML_RESTRICT x_imag,
                              block_ifairy * GGML_RESTRICT y,
                              int64_t k);

void dequantize_row_ifairy(const block_ifairy * GGML_RESTRICT x,
                            float * GGML_RESTRICT y_real,
                            float * GGML_RESTRICT y_imag,
                            int64_t k);

size_t quantize_ifairy(const float * GGML_RESTRICT src_real,
                       const float * GGML_RESTRICT src_imag,
                       void * GGML_RESTRICT dst,
                       int64_t nrow,
                       int64_t n_per_row,
                       const float * quant_weights);

void ggml_vec_dot_ifairy_q8_0(int n,
                               float * GGML_RESTRICT s,
                               size_t bs,
                               const void * GGML_RESTRICT vx,
                               size_t bx,
                               const void * GGML_RESTRICT vy,
                               size_t by,
                               int nrc);
```

**关键说明**:
- 量化和反量化函数接受**两个输入/输出数组**（实部和虚部）
- 这与标准量化类型不同，需要在类型特征表中特殊处理

**实际实现已在 3.2.2 节详细说明，这里不再重复**

#### 3.2.2 复数点积实现

**重要说明**: 基于 iFairy 的实际实现，我们的命名和数据布局与原方案不同：
- 函数命名使用 `ifairy` 而非 `cq2`
- 权重采用**合并存储**（interleaved storage）：实部和虚部交错存储在同一个量化块中
- 每个块处理 QK_K (256) 个复数，其中实部和虚部交错排列
- 量化格式：2-bit 编码为 4 个离散值 {-1, 1, -i, i}，使用二进制编码 00, 01, 10, 11

**数据布局说明**:

```
block_ifairy 结构（QK_K = 256）:
{
    qs[QK_K/4]:    // 64 字节，存储 256 个 2-bit 值
                   // 每个字节存储 4 个 2-bit 值
                   // 对应 128 个复数（实部虚部交错）
    d_real:        // FP16 缩放因子（实部）
    d_imag:        // FP16 缩放因子（虚部）
}

量化值映射:
  00 (0) -> -1   (实部负)
  01 (1) -> +1   (实部正)
  10 (2) -> -i   (虚部负)
  11 (3) -> +i   (虚部正)
```

**文件**: `ggml/src/ggml-quants.c`

```c
// iFairy 量化实现（参考实现）
// 权重按 quantize_row_ifairy_ref 进行量化，激活也采用实部虚部交错存储
void quantize_row_ifairy_ref(
    const float * GGML_RESTRICT x_real,  // 实部输入
    const float * GGML_RESTRICT x_imag,  // 虚部输入
    block_ifairy * GGML_RESTRICT y,      // 量化输出
    int64_t k) {                         // 元素总数（实部或虚部）

    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    // 计算全局缩放因子（对所有块共享）
    float d_real = 0;
    float d_imag = 0;
    for (int64_t i = 0; i < k; i++) {
        d_real = MAX(d_real, fabsf(x_real[i]));
        d_imag = MAX(d_imag, fabsf(x_imag[i]));
    }

    // 量化每个块
    for (int64_t i = 0; i < nb; i++) {
        y[i].d_real = GGML_FP32_TO_FP16(d_real);
        y[i].d_imag = GGML_FP32_TO_FP16(d_imag);

        // 每个块处理 QK_K 个复数（实际上是合并存储）
        // qs 数组大小为 QK_K/4 字节，存储 QK_K 个 2-bit 值
        // 按 32 个元素为一组进行处理（便于 SIMD 优化）
        for (size_t j = 0; j < sizeof(y->qs); j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                // 每个字节存储 4 个 2-bit 值
                for (size_t n = 0; n < 4; ++n) {
                    int xi = 0;
                    // 根据实部和虚部的值判断量化值
                    if (x_real[m + n*32] == 0) {
                        // 纯虚数: -i 或 i
                        if (x_imag[m + n*32] > 0) {
                            xi = 3; // i (11)
                        } else {
                            xi = 2; // -i (10)
                        }
                    } else {
                        // 实数: -1 或 1
                        if (x_real[m + n*32] > 0) {
                            xi = 1; // 1 (01)
                        } else {
                            xi = 0; // -1 (00)
                        }
                    }
                    q += xi << (2*n);  // 将 2-bit 值打包到字节中
                }
                y[i].qs[j + m] = q;
            }
            x_real += 4*32;  // 移动到下一组 128 个元素
            x_imag += 4*32;
        }
    }
}

// iFairy 反量化实现
void dequantize_row_ifairy(
    const block_ifairy * GGML_RESTRICT x,  // 量化输入
    float * GGML_RESTRICT y_real,          // 实部输出
    float * GGML_RESTRICT y_imag,          // 虚部输出
    int64_t k) {                           // 元素总数

    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; ++i) {
        const float d_real = GGML_FP16_TO_FP32(x[i].d_real);
        const float d_imag = GGML_FP16_TO_FP32(x[i].d_imag);

        // 解包 2-bit 量化值
        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {      // 每字节 4 个 2-bit 值
                for (size_t m = 0; m < 32; ++m) {  // 每组 32 个字节
                    int8_t q = (x[i].qs[j + m] >> (l*2)) & 3;

                    // 解码量化值:
                    //   q=0 (00) -> (-1, 0)  实部=-1
                    //   q=1 (01) -> (+1, 0)  实部=+1
                    //   q=2 (10) -> (0, -1)  虚部=-1
                    //   q=3 (11) -> (0, +1)  虚部=+1
                    *y_real++ = (float)((q == 1) - (q == 0)) * d_real;
                    *y_imag++ = (float)((q == 3) - (q == 2)) * d_imag;
                }
            }
        }
    }
}

// iFairy × Q8_0 复数点积实现（标量版本）
// 注意：激活也采用实部虚部交错存储
void ggml_vec_dot_ifairy_q8_0(
    int n,                              // 向量长度（复数个数）
    float * GGML_RESTRICT s,            // 输出：点积结果（real, imag）
    size_t bs,                          // 步长
    const void * GGML_RESTRICT vx,      // 输入 x（ifairy 量化）
    size_t bx,
    const void * GGML_RESTRICT vy,      // 输入 y（Q8_0 量化，实部虚部分离）
    size_t by,
    int nrc) {                          // 行数

    const int qk = QK_K;
    const int nb = n / qk;
    assert(n % qk == 0);
    assert(nrc == 1);  // 暂不支持多行

    const block_ifairy * GGML_RESTRICT x = (const block_ifairy *)vx;
    const block_q8_0 * GGML_RESTRICT y = (const block_q8_0 *)vy;

    // 复数点积: sum((a_r + i*a_i) * (b_r + i*b_i))
    //         = sum(a_r*b_r - a_i*b_i) + i*sum(a_r*b_i + a_i*b_r)

    float sum_real = 0.0f;
    float sum_imag = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d_real = GGML_FP16_TO_FP32(x[i].d_real);
        const float d_imag = GGML_FP16_TO_FP32(x[i].d_imag);

        // 假设 y 也是交错存储或分离存储
        // 这里需要根据实际激活的存储格式调整
        const float d_y_real = GGML_FP16_TO_FP32(y[2*i].d);     // 实部块
        const float d_y_imag = GGML_FP16_TO_FP32(y[2*i+1].d);   // 虚部块

        int32_t acc_rr = 0;  // real × real
        int32_t acc_ii = 0;  // imag × imag
        int32_t acc_ri = 0;  // real × imag
        int32_t acc_ir = 0;  // imag × real

        // 遍历块内元素（每块 QK_K/4 字节，每字节 4 个 2-bit 值）
        for (size_t j = 0; j < QK_K/4; j++) {
            uint8_t qx = x[i].qs[j];

            for (int k = 0; k < 4; k++) {  // 每字节 4 个 2-bit 值
                int8_t q_val = (qx >> (2*k)) & 3;

                // 解码 x 的实部和虚部
                int8_t x_r = (q_val == 1) - (q_val == 0);  // -1 或 +1 或 0
                int8_t x_i = (q_val == 3) - (q_val == 2);  // -1 或 +1 或 0

                // 从 y 读取对应位置的值
                int elem_idx = j * 4 + k;
                int8_t y_r = y[2*i].qs[elem_idx];      // 实部
                int8_t y_i = y[2*i+1].qs[elem_idx];    // 虚部

                // 累加四个乘积项
                acc_rr += x_r * y_r;
                acc_ii += x_i * y_i;
                acc_ri += x_r * y_i;
                acc_ir += x_i * y_r;
            }
        }

        // 应用缩放因子并累加到结果
        sum_real += (acc_rr - acc_ii) * d_real * d_y_real;
        sum_imag += (acc_ri + acc_ir) * d_imag * d_y_imag;
    }

    s[0] = sum_real;
    s[bs] = sum_imag;  // 假设输出也采用某种交错格式
}
```

**关键差异说明**:

1. **命名**: 使用 `ifairy` 而非 `cq2`
2. **数据布局**: 合并存储（interleaved），实部虚部在同一个 `qs` 数组中
3. **量化方案**: 4 个离散值 {-1, 1, -i, i}，而非原方案的 {-1.5, -0.5, 0.5, 1.5}
4. **缩放因子**: 全局共享（对所有块），而非每块独立
5. **块大小**: QK_K = 256（与 K-quants 一致），而非原方案的 64

#### 3.2.3 SIMD优化(ARM NEON示例)

**文件**: `ggml/src/ggml-cpu/arch/arm/quants.c`

```c
#if defined(__ARM_NEON)

void ggml_vec_dot_cq2_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
                               const void * GGML_RESTRICT vx, size_t bx,
                               const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const block_cq2_0 * GGML_RESTRICT x = vx;
    const block_q8_0  * GGML_RESTRICT y = vy;
    const int nb = n / QK8_0;

    float32x4_t sumv_real = vdupq_n_f32(0.0f);
    float32x4_t sumv_imag = vdupq_n_f32(0.0f);

    // 解码查找表
    const int8_t dequant_lut[4] = {-12, -4, 4, 12};  // 对应 [-1.5, -0.5, 0.5, 1.5] * 8

    for (int i = 0; i < nb; i++) {
        const float d_real = GGML_FP16_TO_FP32(x[i].d_real) * GGML_FP16_TO_FP32(y[i].d);
        const float d_imag = GGML_FP16_TO_FP32(x[i].d_imag) * GGML_FP16_TO_FP32(y[i].d);

        // 使用NEON向量化处理
        // (实际实现需要详细的bit解包和向量化点积)

        // ... NEON intrinsics ...
    }

    s[0] = vaddvq_f32(sumv_real);
    s[bs] = vaddvq_f32(sumv_imag);
}

#endif
```

### 3.3 llama.cpp模型层修改

#### 3.3.1 添加新架构类型

**文件**: `src/llama-arch.h:11`

```cpp
enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_FALCON,
    // ... 现有架构 ...
    LLM_ARCH_QWEN2VL,
    LLM_ARCH_COMPLEXFORMER,  // 新增: 复数架构
    LLM_ARCH_UNKNOWN,
};
```

**文件**: `src/llama-arch.cpp` (架构名称映射)

```cpp
static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_LLAMA,           "llama"      },
    // ... 现有映射 ...
    { LLM_ARCH_COMPLEXFORMER,   "complexformer" },
};
```

#### 3.3.2 定义架构键值

**文件**: `src/llama-arch.h` (LLM_KV枚举)

```cpp
enum llm_kv {
    LLM_KV_GENERAL_ARCHITECTURE,
    // ... 现有键 ...

    // ComplexFormer特定键
    LLM_KV_COMPLEXFORMER_ATTENTION_HEAD_COUNT,
    LLM_KV_COMPLEXFORMER_EMBEDDING_LENGTH,
    LLM_KV_COMPLEXFORMER_BLOCK_COUNT,
    LLM_KV_COMPLEXFORMER_FEED_FORWARD_LENGTH,
    LLM_KV_COMPLEXFORMER_COMPLEX_DIM,  // 复数维度
};
```

**文件**: `src/llama-arch.cpp` (键名称映射)

```cpp
static const std::map<llm_kv, const char *> LLM_KV_NAMES = {
    // ... 现有映射 ...

    { LLM_KV_COMPLEXFORMER_ATTENTION_HEAD_COUNT,  "%s.attention.head_count" },
    { LLM_KV_COMPLEXFORMER_EMBEDDING_LENGTH,      "%s.embedding_length"     },
    { LLM_KV_COMPLEXFORMER_BLOCK_COUNT,           "%s.block_count"          },
    { LLM_KV_COMPLEXFORMER_FEED_FORWARD_LENGTH,   "%s.feed_forward_length"  },
    { LLM_KV_COMPLEXFORMER_COMPLEX_DIM,           "%s.complex_dimension"    },
};
```

#### 3.3.3 定义张量名称

**文件**: `src/llama-arch.h` (LLM_TENSOR枚举)

```cpp
enum llm_tensor {
    LLM_TENSOR_TOKEN_EMBD,
    LLM_TENSOR_OUTPUT_NORM,
    // ... 现有张量 ...

    // ComplexFormer特定张量
    LLM_TENSOR_COMPLEX_REAL_PROJ,   // 复数Real投影
    LLM_TENSOR_COMPLEX_IMAG_PROJ,   // 复数Imag投影
    LLM_TENSOR_COMPLEX_ATTN_Q_REAL,
    LLM_TENSOR_COMPLEX_ATTN_Q_IMAG,
    LLM_TENSOR_COMPLEX_ATTN_K_REAL,
    LLM_TENSOR_COMPLEX_ATTN_K_IMAG,
    LLM_TENSOR_COMPLEX_ATTN_V_REAL,
    LLM_TENSOR_COMPLEX_ATTN_V_IMAG,
    LLM_TENSOR_COMPLEX_FFN_GATE_REAL,
    LLM_TENSOR_COMPLEX_FFN_GATE_IMAG,
    LLM_TENSOR_COMPLEX_FFN_DOWN_REAL,
    LLM_TENSOR_COMPLEX_FFN_DOWN_IMAG,
    LLM_TENSOR_COMPLEX_FFN_UP_REAL,
    LLM_TENSOR_COMPLEX_FFN_UP_IMAG,
};
```

**文件**: `src/llama-arch.cpp` (张量名称映射)

```cpp
static const std::map<llm_arch, std::map<llm_tensor, const char *>> LLM_TENSOR_NAMES = {
    // ... 现有映射 ...

    {
        LLM_ARCH_COMPLEXFORMER,
        {
            { LLM_TENSOR_TOKEN_EMBD,              "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,             "output_norm" },
            { LLM_TENSOR_OUTPUT,                  "output" },
            { LLM_TENSOR_COMPLEX_REAL_PROJ,       "blk.%d.complex.real_proj" },
            { LLM_TENSOR_COMPLEX_IMAG_PROJ,       "blk.%d.complex.imag_proj" },
            { LLM_TENSOR_COMPLEX_ATTN_Q_REAL,     "blk.%d.attn_q.real" },
            { LLM_TENSOR_COMPLEX_ATTN_Q_IMAG,     "blk.%d.attn_q.imag" },
            { LLM_TENSOR_COMPLEX_ATTN_K_REAL,     "blk.%d.attn_k.real" },
            { LLM_TENSOR_COMPLEX_ATTN_K_IMAG,     "blk.%d.attn_k.imag" },
            { LLM_TENSOR_COMPLEX_ATTN_V_REAL,     "blk.%d.attn_v.real" },
            { LLM_TENSOR_COMPLEX_ATTN_V_IMAG,     "blk.%d.attn_v.imag" },
            { LLM_TENSOR_COMPLEX_FFN_GATE_REAL,   "blk.%d.ffn_gate.real" },
            { LLM_TENSOR_COMPLEX_FFN_GATE_IMAG,   "blk.%d.ffn_gate.imag" },
            { LLM_TENSOR_COMPLEX_FFN_DOWN_REAL,   "blk.%d.ffn_down.real" },
            { LLM_TENSOR_COMPLEX_FFN_DOWN_IMAG,   "blk.%d.ffn_down.imag" },
            { LLM_TENSOR_COMPLEX_FFN_UP_REAL,     "blk.%d.ffn_up.real" },
            { LLM_TENSOR_COMPLEX_FFN_UP_IMAG,     "blk.%d.ffn_up.imag" },
        },
    },
};
```

#### 3.3.4 模型加载逻辑

**文件**: `src/llama-model.cpp` (llm_load_tensors函数)

```cpp
static bool llm_load_tensors(
        llama_model_loader & ml,
        llama_model & model,
        int n_gpu_layers,
        // ... 其他参数
        ) {

    const auto & hparams = model.hparams;
    const llm_arch arch = model.arch;

    // ... 现有代码 ...

    switch (arch) {
        // ... 现有架构case ...

        case LLM_ARCH_COMPLEXFORMER:
            {
                model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

                // 输出层
                {
                    model.output_norm = ml.create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                    model.output      = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
                }

                const int i_gpu_start = n_layer - n_gpu_layers;

                for (int i = 0; i < n_layer; ++i) {
                    const ggml_backend_type backend = int(i) < i_gpu_start ? GGML_BACKEND_CPU : llama_backend_offload;
                    auto ctx_layer = ctx_for_layer(i);
                    auto ctx_split = ctx_for_layer_split(i);

                    // 注意力层 (Real + Imag)
                    layer.attn_q_real = ml.create_tensor(ctx_split, tn(LLM_TENSOR_COMPLEX_ATTN_Q_REAL, "weight", i), {n_embd, n_embd_head_k * n_head});
                    layer.attn_q_imag = ml.create_tensor(ctx_split, tn(LLM_TENSOR_COMPLEX_ATTN_Q_IMAG, "weight", i), {n_embd, n_embd_head_k * n_head});

                    layer.attn_k_real = ml.create_tensor(ctx_split, tn(LLM_TENSOR_COMPLEX_ATTN_K_REAL, "weight", i), {n_embd, n_embd_head_k * n_head_kv});
                    layer.attn_k_imag = ml.create_tensor(ctx_split, tn(LLM_TENSOR_COMPLEX_ATTN_K_IMAG, "weight", i), {n_embd, n_embd_head_k * n_head_kv});

                    layer.attn_v_real = ml.create_tensor(ctx_split, tn(LLM_TENSOR_COMPLEX_ATTN_V_REAL, "weight", i), {n_embd, n_embd_head_v * n_head_kv});
                    layer.attn_v_imag = ml.create_tensor(ctx_split, tn(LLM_TENSOR_COMPLEX_ATTN_V_IMAG, "weight", i), {n_embd, n_embd_head_v * n_head_kv});

                    // FFN层 (Real + Imag)
                    layer.ffn_gate_real = ml.create_tensor(ctx_split, tn(LLM_TENSOR_COMPLEX_FFN_GATE_REAL, "weight", i), {n_embd, n_ff});
                    layer.ffn_gate_imag = ml.create_tensor(ctx_split, tn(LLM_TENSOR_COMPLEX_FFN_GATE_IMAG, "weight", i), {n_embd, n_ff});

                    layer.ffn_down_real = ml.create_tensor(ctx_split, tn(LLM_TENSOR_COMPLEX_FFN_DOWN_REAL, "weight", i), {n_ff, n_embd});
                    layer.ffn_down_imag = ml.create_tensor(ctx_split, tn(LLM_TENSOR_COMPLEX_FFN_DOWN_IMAG, "weight", i), {n_ff, n_embd});

                    layer.ffn_up_real = ml.create_tensor(ctx_split, tn(LLM_TENSOR_COMPLEX_FFN_UP_REAL, "weight", i), {n_embd, n_ff});
                    layer.ffn_up_imag = ml.create_tensor(ctx_split, tn(LLM_TENSOR_COMPLEX_FFN_UP_IMAG, "weight", i), {n_embd, n_ff});
                }
            } break;

        default:
            throw std::runtime_error("unknown architecture");
    }

    return true;
}
```

#### 3.3.5 构建计算图

**文件**: `src/llama-graph.cpp` (新增build_complexformer函数)

```cpp
static struct ggml_cgraph * llama_build_graph_complexformer(
         llama_context & lctx,
   const llama_batch & batch,
              bool     worst_case) {

    const auto & model = lctx.model;
    const auto & hparams = model.hparams;

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_layer     = hparams.n_layer;
    const int64_t n_head      = hparams.n_head;
    const int64_t n_head_kv   = hparams.n_head_kv;
    const int64_t n_embd_head = hparams.n_embd_head_k;

    auto & buf_compute = lctx.buf_compute;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.data,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph  * gf   = ggml_new_graph(ctx0);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    // 输入embedding
    inpL = ggml_get_rows(ctx0, model.tok_embd, batch.token);

    for (int il = 0; il < n_layer; ++il) {
        // 注意力层
        {
            // 复数注意力: Q = Q_real + i*Q_imag
            struct ggml_tensor * Qcur_real = ggml_mul_mat(ctx0, model.layers[il].attn_q_real, inpL);
            struct ggml_tensor * Qcur_imag = ggml_mul_mat(ctx0, model.layers[il].attn_q_imag, inpL);

            struct ggml_tensor * Kcur_real = ggml_mul_mat(ctx0, model.layers[il].attn_k_real, inpL);
            struct ggml_tensor * Kcur_imag = ggml_mul_mat(ctx0, model.layers[il].attn_k_imag, inpL);

            struct ggml_tensor * Vcur_real = ggml_mul_mat(ctx0, model.layers[il].attn_v_real, inpL);
            struct ggml_tensor * Vcur_imag = ggml_mul_mat(ctx0, model.layers[il].attn_v_imag, inpL);

            // 复数点积: Q·K* = (Q_r + iQ_i)·(K_r - iK_i)
            //                = (Q_r·K_r + Q_i·K_i) + i(Q_i·K_r - Q_r·K_i)
            struct ggml_tensor * KQ_real = ggml_add(ctx0,
                ggml_mul_mat(ctx0, Kcur_real, Qcur_real),
                ggml_mul_mat(ctx0, Kcur_imag, Qcur_imag)
            );

            struct ggml_tensor * KQ_imag = ggml_sub(ctx0,
                ggml_mul_mat(ctx0, Kcur_real, Qcur_imag),
                ggml_mul_mat(ctx0, Kcur_imag, Qcur_real)
            );

            // 计算复数模: |z|^2 = real^2 + imag^2
            struct ggml_tensor * KQ_abs = ggml_sqrt(ctx0,
                ggml_add(ctx0,
                    ggml_sqr(ctx0, KQ_real),
                    ggml_sqr(ctx0, KQ_imag)
                )
            );

            // Softmax (在复数模上)
            struct ggml_tensor * KQ_soft = ggml_soft_max(ctx0, KQ_abs);

            // 复数乘法: softmax(KQ) * V
            cur = ggml_mul_mat(ctx0, Vcur_real, KQ_soft);  // 简化版

            // ... 投影层 ...
        }

        // FFN层
        {
            struct ggml_tensor * ffn_gate_real = ggml_mul_mat(ctx0, model.layers[il].ffn_gate_real, cur);
            struct ggml_tensor * ffn_gate_imag = ggml_mul_mat(ctx0, model.layers[il].ffn_gate_imag, cur);

            struct ggml_tensor * ffn_up_real = ggml_mul_mat(ctx0, model.layers[il].ffn_up_real, cur);
            struct ggml_tensor * ffn_up_imag = ggml_mul_mat(ctx0, model.layers[il].ffn_up_imag, cur);

            // 复数乘法: gate * up
            struct ggml_tensor * ffn_mul_real = ggml_sub(ctx0,
                ggml_mul(ctx0, ffn_gate_real, ffn_up_real),
                ggml_mul(ctx0, ffn_gate_imag, ffn_up_imag)
            );

            // ... Down投影 ...
            cur = ggml_mul_mat(ctx0, model.layers[il].ffn_down_real, ffn_mul_real);
        }

        inpL = ggml_add(ctx0, inpL, cur);  // 残差连接
    }

    // 输出层
    cur = ggml_norm(ctx0, inpL);
    cur = ggml_mul_mat(ctx0, model.output, cur);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
```

### 3.4 GGUF文件格式支持

#### 3.4.1 添加元数据键

**文件**: `ggml/include/gguf.h`

```c
// 在GGUF key常量中添加
#define GGUF_KEY_COMPLEXFORMER_ATTENTION_HEAD_COUNT  "complexformer.attention.head_count"
#define GGUF_KEY_COMPLEXFORMER_EMBEDDING_LENGTH      "complexformer.embedding_length"
#define GGUF_KEY_COMPLEXFORMER_BLOCK_COUNT           "complexformer.block_count"
#define GGUF_KEY_COMPLEXFORMER_FEED_FORWARD_LENGTH   "complexformer.feed_forward_length"
#define GGUF_KEY_COMPLEXFORMER_COMPLEX_DIM           "complexformer.complex_dimension"
```

#### 3.4.2 转换脚本

**文件**: `convert_hf_to_gguf.py` (新增ComplexFormer转换器)

```python
@Model.register("ComplexFormerForCausalLM")
class ComplexFormerModel(Model):
    model_arch = gguf.MODEL_ARCH.COMPLEXFORMER

    def set_vocab(self):
        # 词汇表处理
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        hparams = self.hparams

        # 基础参数
        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(hparams["num_hidden_layers"])
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(hparams["num_key_value_heads"])

        # ComplexFormer特定参数
        self.gguf_writer.add_uint32("complexformer.complex_dimension", hparams.get("complex_dim", hparams["hidden_size"]))

        # Layer Norm
        self.gguf_writer.add_layer_norm_rms_eps(hparams["rms_norm_eps"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # 处理复数权重
        # 假设HuggingFace模型将Real和Imag分开存储

        if "attn.q_proj" in name:
            # 拆分为Real和Imag
            half_dim = data_torch.shape[0] // 2
            real_part = data_torch[:half_dim, :]
            imag_part = data_torch[half_dim:, :]

            yield (name.replace("q_proj", "attn_q.real"), real_part)
            yield (name.replace("q_proj", "attn_q.imag"), imag_part)

        elif "attn.k_proj" in name:
            half_dim = data_torch.shape[0] // 2
            yield (name.replace("k_proj", "attn_k.real"), data_torch[:half_dim, :])
            yield (name.replace("k_proj", "attn_k.imag"), data_torch[half_dim:, :])

        # ... 其他层处理 ...

        else:
            # 默认处理
            yield (name, data_torch)
```

### 3.5 量化工具支持

**文件**: `src/llama-quant.cpp`

```cpp
static void llama_model_quantize_internal(
        const std::string & fname_inp,
        const std::string & fname_out,
        const llama_model_quantize_params * params) {

    // ... 现有代码 ...

    // 为CQ2_0量化添加特殊处理
    const bool is_complex_quant = (params->ftype == LLAMA_FTYPE_MOSTLY_CQ2_0);

    for (const auto & tensor : model_loader.tensors_map) {
        const std::string & name = tensor.first;

        // 跳过非权重张量
        if (should_skip_tensor(name)) continue;

        // 复数张量特殊处理
        if (is_complex_quant && (name.find(".real") != std::string::npos ||
                                  name.find(".imag") != std::string::npos)) {
            // Real和Imag分别量化
            quantize_tensor_complex(tensor, GGML_TYPE_CQ2_0);
        } else {
            // 标准量化
            quantize_tensor_standard(tensor, target_type);
        }
    }
}
```

### 3.6 后端支持(可选)

#### 3.6.1 CUDA支持

**文件**: `ggml/src/ggml-cuda/cq2_quant.cu` (新建)

```cuda
#include "common.cuh"

// CUDA kernel for CQ2_0 dequantization
__global__ void dequantize_block_cq2_0(const block_cq2_0 * x, float * dst, int k) {
    const int i = blockIdx.x;
    const int tid = threadIdx.x;

    const float d_real = __half2float(x[i].d_real);
    const float d_imag = __half2float(x[i].d_imag);

    // 解码Real部分
    if (tid < QK_CQ2_0/2) {
        const int byte_idx = tid / 4;
        const int bit_offset = (tid % 4) * 2;
        const uint8_t qi = (x[i].qs_real[byte_idx] >> bit_offset) & 0x03;

        const float dequant_lut[4] = {-1.5f, -0.5f, 0.5f, 1.5f};
        dst[i*QK_CQ2_0 + tid] = dequant_lut[qi] * d_real;
    }

    // 解码Imag部分
    if (tid < QK_CQ2_0/2) {
        const int byte_idx = tid / 4;
        const int bit_offset = (tid % 4) * 2;
        const uint8_t qi = (x[i].qs_imag[byte_idx] >> bit_offset) & 0x03;

        const float dequant_lut[4] = {-1.5f, -0.5f, 0.5f, 1.5f};
        dst[k + i*QK_CQ2_0 + tid] = dequant_lut[qi] * d_imag;
    }
}

// 复数矩阵乘法kernel (简化版)
__global__ void ggml_mul_mat_cq2_0_cuda(
    const block_cq2_0 * x, const float * y, float * dst,
    int m, int n, int k) {

    // ... CUDA实现 ...
    // 使用Tensor Cores加速(如果可用)
}
```

**文件**: `ggml/src/ggml-cuda/ggml-cuda.cu` (注册kernel)

```cuda
// 在type_traits中添加
case GGML_TYPE_CQ2_0:
    dequantize_row_cuda = dequantize_block_cq2_0;
    mul_mat_kernel = ggml_mul_mat_cq2_0_cuda;
    break;
```

---

## 4. 完整修改清单

### 4.1 GGML底层修改

| 文件路径 | 修改内容 | 难度 |
|---------|---------|------|
| `ggml/include/ggml.h` | 添加`GGML_TYPE_CQ2_0`枚举 | ⭐ |
| `ggml/src/ggml-common.h` | 定义`block_cq2_0`结构体,添加`QK_CQ2_0`常量 | ⭐⭐ |
| `ggml/src/ggml.c` | 注册类型特征表 | ⭐⭐ |
| `ggml/src/ggml-cpu/quants.h` | 声明量化/反量化/点积函数 | ⭐ |
| `ggml/src/ggml-cpu/quants.c` | 实现量化/反量化/点积(标量版本) | ⭐⭐⭐⭐ |
| `ggml/src/ggml-cpu/arch/arm/quants.c` | ARM NEON优化 | ⭐⭐⭐⭐⭐ |
| `ggml/src/ggml-cpu/arch/x86/quants.c` | x86 AVX2/AVX512优化 | ⭐⭐⭐⭐⭐ |
| `ggml/src/ggml-cpu/traits.cpp` | 更新类型特征 | ⭐⭐ |

### 4.2 llama.cpp模型层修改

| 文件路径 | 修改内容 | 难度 |
|---------|---------|------|
| `src/llama-arch.h` | 添加`LLM_ARCH_COMPLEXFORMER`枚举 | ⭐ |
| `src/llama-arch.h` | 添加ComplexFormer专用KV键枚举 | ⭐⭐ |
| `src/llama-arch.h` | 添加ComplexFormer专用张量枚举 | ⭐⭐ |
| `src/llama-arch.cpp` | 添加架构/KV/张量名称映射 | ⭐⭐ |
| `src/llama-model.h` | 扩展`llama_layer`结构体(添加Real/Imag张量) | ⭐⭐ |
| `src/llama-model.cpp` | 实现`llm_load_tensors`的ComplexFormer分支 | ⭐⭐⭐⭐ |
| `src/llama-graph.cpp` | 实现`llama_build_graph_complexformer` | ⭐⭐⭐⭐⭐ |
| `src/llama-hparams.cpp` | 添加ComplexFormer超参数验证 | ⭐⭐ |
| `src/llama-quant.cpp` | 添加CQ2_0量化逻辑 | ⭐⭐⭐ |

### 4.3 转换工具修改

| 文件路径 | 修改内容 | 难度 |
|---------|---------|------|
| `convert_hf_to_gguf.py` | 添加`ComplexFormerModel`类 | ⭐⭐⭐⭐ |
| `gguf-py/gguf/constants.py` | 添加`MODEL_ARCH.COMPLEXFORMER` | ⭐ |
| `gguf-py/gguf/gguf_writer.py` | 添加ComplexFormer元数据写入方法 | ⭐⭐ |

### 4.4 后端支持(可选)

| 文件路径 | 修改内容 | 难度 |
|---------|---------|------|
| `ggml/src/ggml-cuda/cq2_quant.cu` | CUDA kernel实现 | ⭐⭐⭐⭐⭐ |
| `ggml/src/ggml-metal/ggml-metal.metal` | Metal shader实现 | ⭐⭐⭐⭐⭐ |
| `ggml/src/ggml-vulkan/vulkan-shaders/` | Vulkan shader实现 | ⭐⭐⭐⭐⭐ |

### 4.5 测试文件(新建)

| 文件路径 | 内容 | 难度 |
|---------|------|------|
| `tests/test-backend-ops-cq2.cpp` | CQ2_0量化正确性测试 | ⭐⭐⭐ |
| `tests/test-quantize-fns-cq2.cpp` | 量化函数单元测试 | ⭐⭐⭐ |
| `examples/complexformer/` | ComplexFormer推理示例 | ⭐⭐⭐⭐ |

---

## 5. 测试与验证

### 5.1 单元测试

#### 5.1.1 量化精度测试

**文件**: `tests/test-quantize-fns-cq2.cpp` (新建)

```cpp
#include "ggml.h"
#include <cmath>
#include <vector>

void test_cq2_0_quantization() {
    const int n = 256;  // 128个复数
    std::vector<float> src(n * 2);  // Real + Imag
    std::vector<float> dst(n * 2);

    // 生成随机复数
    for (int i = 0; i < n; i++) {
        src[i] = (rand() / (float)RAND_MAX) * 4.0f - 2.0f;  // Real部分
        src[n + i] = (rand() / (float)RAND_MAX) * 4.0f - 2.0f;  // Imag部分
    }

    // 量化
    std::vector<block_cq2_0> quant(n / QK_CQ2_0);
    quantize_row_cq2_0(src.data(), quant.data(), n);

    // 反量化
    dequantize_row_cq2_0(quant.data(), dst.data(), n);

    // 计算误差
    float max_error = 0.0f;
    for (int i = 0; i < n * 2; i++) {
        float error = fabsf(src[i] - dst[i]);
        max_error = fmaxf(max_error, error);
    }

    printf("CQ2_0 max quantization error: %.6f\n", max_error);
    assert(max_error < 0.5f);  // 2-bit误差阈值
}

void test_cq2_0_dot_product() {
    const int n = 256;
    std::vector<float> a(n * 2), b(n * 2);

    // 生成测试数据
    for (int i = 0; i < n * 2; i++) {
        a[i] = (i % 10) / 10.0f;
        b[i] = ((i + 5) % 10) / 10.0f;
    }

    // 量化
    std::vector<block_cq2_0> qa(n / QK_CQ2_0);
    std::vector<block_q8_0> qb(n / QK8_0);
    quantize_row_cq2_0(a.data(), qa.data(), n);
    quantize_row_q8_0(b.data(), qb.data(), n);

    // 复数点积
    float result[2] = {0.0f, 0.0f};
    ggml_vec_dot_cq2_0_q8_0(n, result, 1, qa.data(), 0, qb.data(), 0, 1);

    // 计算参考结果
    float ref_real = 0.0f, ref_imag = 0.0f;
    for (int i = 0; i < n/2; i++) {
        ref_real += a[i]*b[i] - a[n+i]*b[n+i];
        ref_imag += a[i]*b[n+i] + a[n+i]*b[i];
    }

    printf("CQ2_0 dot product: (%.4f, %.4f) vs ref (%.4f, %.4f)\n",
           result[0], result[1], ref_real, ref_imag);

    assert(fabsf(result[0] - ref_real) / fabsf(ref_real) < 0.1f);
    assert(fabsf(result[1] - ref_imag) / fabsf(ref_imag) < 0.1f);
}

int main() {
    test_cq2_0_quantization();
    test_cq2_0_dot_product();
    printf("All tests passed!\n");
    return 0;
}
```

#### 5.1.2 编译测试

```bash
# 添加到CMakeLists.txt
add_executable(test-quantize-fns-cq2 tests/test-quantize-fns-cq2.cpp)
target_link_libraries(test-quantize-fns-cq2 PRIVATE ggml)

# 运行测试
./build/bin/test-quantize-fns-cq2
```

### 5.2 集成测试

#### 5.2.1 模型转换测试

```bash
# 1. 准备HuggingFace ComplexFormer模型
# 假设已有: ~/models/complexformer-1b

# 2. 转换为GGUF
python convert_hf_to_gguf.py ~/models/complexformer-1b \
    --outfile ~/models/complexformer-1b-f32.gguf \
    --outtype f32

# 3. 量化为CQ2_0
./build/bin/llama-quantize \
    ~/models/complexformer-1b-f32.gguf \
    ~/models/complexformer-1b-cq2_0.gguf \
    CQ2_0

# 4. 验证模型
./build/bin/llama-cli \
    -m ~/models/complexformer-1b-cq2_0.gguf \
    -p "Hello world" \
    -n 128
```

#### 5.2.2 性能基准测试

```bash
# 吞吐量测试
./build/bin/llama-bench \
    -m ~/models/complexformer-1b-cq2_0.gguf \
    -p 512 -n 128 -t 8

# 与其他量化对比
for quant in f32 q4_0 q8_0 cq2_0; do
    echo "Testing $quant..."
    ./build/bin/llama-bench \
        -m ~/models/complexformer-1b-${quant}.gguf \
        -p 512 -n 128 -t 8 | tee bench_${quant}.log
done
```

### 5.3 正确性验证

#### 5.3.1 与原始模型对比

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import numpy as np

# 加载HuggingFace模型
hf_model = AutoModelForCausalLM.from_pretrained("~/models/complexformer-1b")
tokenizer = AutoTokenizer.from_pretrained("~/models/complexformer-1b")

# 测试输入
prompt = "The quick brown fox"
inputs = tokenizer(prompt, return_tensors="pt")

# HuggingFace输出
with torch.no_grad():
    hf_outputs = hf_model.generate(**inputs, max_length=50)
hf_text = tokenizer.decode(hf_outputs[0])

print(f"HuggingFace: {hf_text}")

# llama.cpp输出
llama_output = subprocess.check_output([
    "./build/bin/llama-cli",
    "-m", "~/models/complexformer-1b-cq2_0.gguf",
    "-p", prompt,
    "-n", "50"
]).decode()

print(f"llama.cpp: {llama_output}")

# 比较(允许量化误差)
# ...
```

---

## 6. 性能优化建议

### 6.1 SIMD优化优先级

| 平台 | 指令集 | 优先级 | 预期加速 |
|------|--------|--------|---------|
| ARM | NEON | ⭐⭐⭐⭐⭐ | 4-8x |
| ARM | MATMUL_INT8 | ⭐⭐⭐⭐ | 8-16x |
| x86 | AVX2 | ⭐⭐⭐⭐⭐ | 6-12x |
| x86 | AVX512 | ⭐⭐⭐⭐ | 12-24x |
| RISC-V | RVV | ⭐⭐⭐ | 4-8x |

### 6.2 内存优化

#### 6.2.1 预分配缓冲区

```cpp
// 在llama_context中添加复数专用缓冲区
struct llama_context {
    // ... 现有字段 ...

    // ComplexFormer专用
    struct ggml_tensor * complex_real_buf;
    struct ggml_tensor * complex_imag_buf;
    struct ggml_tensor * complex_temp_buf;
};
```

#### 6.2.2 缓存友好的数据布局

```
推荐布局:
[Block0_Real][Block0_Imag][Block1_Real][Block1_Imag]...

避免:
[All_Real_Blocks][All_Imag_Blocks]
```

### 6.3 计算优化

#### 6.3.1 融合操作

```cpp
// 融合: Real×Real - Imag×Imag
__global__ void fused_complex_mul_real_part(
    const float* a_real, const float* a_imag,
    const float* b_real, const float* b_imag,
    float* out, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a_real[i]*b_real[i] - a_imag[i]*b_imag[i];
    }
}
```

#### 6.3.2 查找表优化

```cpp
// 预计算2-bit解码表
static const int8_t dequant_lut_cq2_0[4] = {-12, -4, 4, 12};  // * 1/8 = [-1.5, -0.5, 0.5, 1.5]

// SIMD查找
__m128i lookup_2bit_avx(__m128i indices) {
    const __m128i lut = _mm_set_epi8(12,12,12,4,4,4,-4,-4,-4,-12,-12,-12,0,0,0,0);
    return _mm_shuffle_epi8(lut, indices);
}
```

### 6.4 多线程优化

```cpp
// 并行处理Real和Imag
#pragma omp parallel sections
{
    #pragma omp section
    {
        // 处理Real部分
        ggml_compute_forward_mul_mat(ctx, dst_real, src_real, weight_real);
    }

    #pragma omp section
    {
        // 处理Imag部分
        ggml_compute_forward_mul_mat(ctx, dst_imag, src_imag, weight_imag);
    }
}
```

---

## 7. 总结

### 7.1 实现步骤摘要

1. **第一阶段: GGML基础** (1-2周)
   - 定义`GGML_TYPE_CQ2_0`枚举
   - 实现`block_cq2_0`结构体
   - 编写标量版量化/反量化函数
   - 编写基础点积函数

2. **第二阶段: llama.cpp集成** (1-2周)
   - 添加`LLM_ARCH_COMPLEXFORMER`架构
   - 定义KV键和张量名称
   - 实现模型加载逻辑
   - 编写计算图构建函数

3. **第三阶段: 工具链** (1周)
   - 实现HuggingFace转换器
   - 添加量化工具支持
   - 编写测试用例

4. **第四阶段: 优化** (2-4周)
   - ARM NEON优化
   - x86 AVX2/AVX512优化
   - CUDA/Metal后端(可选)

5. **第五阶段: 验证** (1周)
   - 正确性测试
   - 性能基准测试
   - 与原始模型对比

### 7.2 关键挑战

| 挑战 | 难度 | 解决方案 |
|------|------|---------|
| 复数运算实现 | ⭐⭐⭐⭐ | 拆分为Real/Imag独立处理 |
| 2-bit量化精度 | ⭐⭐⭐⭐ | 分组量化+自适应缩放 |
| SIMD优化 | ⭐⭐⭐⭐⭐ | 参考现有IQ2实现,使用查找表 |
| 内存布局 | ⭐⭐⭐ | 分离存储,缓存对齐 |
| 测试覆盖 | ⭐⭐⭐⭐ | 单元测试+集成测试+对比测试 |

### 7.3 预期效果

- **模型大小**: 相比FP32减少~7x (2-bit量化)
- **速度**: 相比FP32提升2-4x (量化加速)
- **精度损失**: 困惑度增加5-10% (可接受范围)
- **内存占用**: 相比FP32减少~7x

### 7.4 扩展方向

1. **更激进的量化**: 1.5-bit, 1-bit
2. **混合精度**: 关键层FP16,其余CQ2_0
3. **稀疏化**: 结合剪枝技术
4. **硬件加速**: 专用复数运算单元

---

## 附录

### A. 参考代码片段

完整代码过长,建议参考:
- `ggml/src/ggml-cpu/quants.c:ggml_vec_dot_iq2_xxs_q8_K` (2-bit量化参考)
- `src/llama-model.cpp:LLM_ARCH_LLAMA` (架构加载参考)
- `src/llama-graph.cpp:llama_build_graph_llama` (计算图参考)

### B. 调试工具

```bash
# 打印GGUF元数据
./build/bin/llama-inspect ~/models/complexformer-1b-cq2_0.gguf

# 导出计算图
LLAMA_DEBUG=1 ./build/bin/llama-cli -m model.gguf -p "test" --verbose

# 性能分析
perf record ./build/bin/llama-bench -m model.gguf
perf report
```

### C. 社区资源

- [GGML文档](https://github.com/ggerganov/ggml)
- [llama.cpp Wiki](https://github.com/ggerganov/llama.cpp/wiki)
- [量化技术论文](https://arxiv.org/abs/2206.01861)
- [复数神经网络综述](https://arxiv.org/abs/2101.12249)

---

**文档版本**: 1.0
**最后更新**: 2025-10-06
**作者**: Claude Code Design
