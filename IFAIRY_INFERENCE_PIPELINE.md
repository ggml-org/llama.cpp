# iFairy 模型推理流程文档

本文档详细描述了 iFairy 模型从转换、量化到推理计算的完整流程。

## 目录

1. [模型架构概述](#模型架构概述)
2. [模型转换流程](#模型转换流程)
3. [量化实现](#量化实现)
4. [计算图加载](#计算图加载)
5. [矩阵乘法与推理计算](#矩阵乘法与推理计算)
6. [关键代码位置](#关键代码位置)

---

## 1. 模型架构概述

### 1.1 架构定义

iFairy 模型是一个基于复数神经网络的语言模型架构,在 llama.cpp 中的架构枚举为 `LLM_ARCH_IFAIRY`。

**文件位置**: `src/llama-arch.h:73`

```cpp
enum llm_arch {
    ...
    LLM_ARCH_BITNET,
    LLM_ARCH_IFAIRY,  // line 73
    LLM_ARCH_T5,
    ...
};
```

**架构名称映射**: `src/llama-arch.cpp:69`

```cpp
{ LLM_ARCH_IFAIRY, "ifairy" },
```

### 1.2 模型参数

从模型加载代码 (`src/llama-model.cpp:1607-1615`) 可以看出 iFairy 模型的关键参数:

- **RMS Norm epsilon**: `hparams.f_norm_rms_eps`
- **层数**: 26 层 (对应 1.3B 参数量的模型)
- **隐藏维度**: 1536
- **中间层维度**: 4096
- **注意力头数**: 16
- **KV 头数**: 16
- **最大位置编码**: 2048
- **词汇表大小**: 由配置指定
- **RoPE theta**: 10000.0

### 1.3 张量结构

iFairy 模型的特点是使用**复数表示**,每个权重张量都分为实部(real)和虚部(imag)两部分。

**张量命名映射** (`src/llama-arch.cpp:1506-1526`):

```cpp
{
    LLM_ARCH_IFAIRY,
    {
        { LLM_TENSOR_TOKEN_EMBD_IMAG,    "token_embd_imag" },
        { LLM_TENSOR_TOKEN_EMBD_REAL,    "token_embd_real" },
        { LLM_TENSOR_OUTPUT_NORM_IMAG,   "output_norm_imag" },
        { LLM_TENSOR_OUTPUT_NORM_REAL,   "output_norm_real" },
        { LLM_TENSOR_ATTN_Q_IMAG,        "blk.%d.attn_q_imag" },
        { LLM_TENSOR_ATTN_Q_REAL,        "blk.%d.attn_q_real" },
        { LLM_TENSOR_ATTN_K_IMAG,        "blk.%d.attn_k_imag" },
        { LLM_TENSOR_ATTN_K_REAL,        "blk.%d.attn_k_real" },
        { LLM_TENSOR_ATTN_V_IMAG,        "blk.%d.attn_v_imag" },
        { LLM_TENSOR_ATTN_V_REAL,        "blk.%d.attn_v_real" },
        { LLM_TENSOR_ATTN_OUT_IMAG,      "blk.%d.attn_output_imag" },
        { LLM_TENSOR_ATTN_OUT_REAL,      "blk.%d.attn_output_real" },
        { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
        { LLM_TENSOR_ATTN_SUB_NORM,      "blk.%d.attn_sub_norm" },
        { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
        { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
        { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
        { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
        { LLM_TENSOR_FFN_SUB_NORM,       "blk.%d.ffn_sub_norm" },
    },
},
```

---

## 2. 模型转换流程

### 2.1 转换脚本

**文件位置**: `gguf-py/convert_ifairy.py`

转换脚本负责将 Hugging Face 格式的模型转换为 GGUF 格式。

### 2.2 转换步骤

#### 步骤 1: 加载模型配置

```python
# convert_ifairy.py:82-91
config_path = os.path.join(model_dir, 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
```

从 `config.json` 中读取模型的超参数配置。

#### 步骤 2: 创建 GGUF Writer 并设置元数据

```python
# convert_ifairy.py:94-112
writer = gguf.GGUFWriter(output_file, model_name)
writer.add_name(model_name)
writer.add_context_length(config["max_position_embeddings"])  # 2048
writer.add_embedding_length(config["hidden_size"])             # 1536
writer.add_block_count(config["num_hidden_layers"])            # 24
writer.add_feed_forward_length(config["intermediate_size"])    # 4096
writer.add_head_count(config["num_attention_heads"])           # 16
writer.add_head_count_kv(config["num_key_value_heads"])        # 16
writer.add_layer_norm_eps(config["rms_norm_eps"])              # 1e-05
writer.add_rope_freq_base(config["rope_theta"])                # 10000.0
writer.add_file_type(gguf.LlamaFileType.MOSTLY_IFAIRY)
writer.add_vocab_size(config["vocab_size"])
writer.add_tokenizer_model("llama")
```

#### 步骤 3: 加载和量化权重

```python
# convert_ifairy.py:118-169
# 支持从 model.safetensors.index.json 中加载分片模型
model_files = list(Path(model_dir).glob("*.safetensors"))

for model_file_path in model_files:
    with safe_open(model_file_path, framework="pt") as f:
        for key in f.keys():
            tensor_data = f.get_tensor(key).to(torch.float16)

            # 量化处理 (见下一节详细说明)
            tensor_data = quant(key, tensor_data, f, weight_map)
            numpy_array = tensor_data.cpu().numpy().astype(np.float16)

            # 张量名称映射
            mapper = gguf.get_tensor_name_map(model_arch, config["num_hidden_layers"])
            mapped_name = mapper.get_name(key)

            # 添加到 GGUF 文件,使用自定义的 F16_I2 类型
            writer.add_tensor(mapped_name, numpy_array, raw_dtype=gguf.GGMLQuantizationType.F16_I2)
```

#### 步骤 4: 写入 GGUF 文件

```python
# convert_ifairy.py:175-179
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
```

### 2.3 转换时量化

iFairy 模型在转换阶段就进行了量化处理,这是其独特之处。

**量化函数** (`convert_ifairy.py:20-68`):

```python
def forward(w_real: torch.Tensor, w_imag: torch.Tensor):
    """
    基于相位的量化方法:
    - 将复数权重 w = w_real + j*w_imag 转换为相位表示
    - 根据相位将权重量化为 {-1, 1, -i, i} 四个值之一
    """
    phase = torch.angle(w_real + 1j * w_imag)

    # 相位区间划分:
    # [-π/4, π/4):     实部为正, qw_real = 1
    # [π/4, 3π/4):     虚部为正, qw_imag = 1
    # [3π/4, -3π/4):   实部为负, qw_real = -1
    # [-3π/4, -π/4):   虚部为负, qw_imag = -1

    real_pos = (phase >= -torch.pi / 4) & (phase < torch.pi / 4)
    real_neg = (phase >= 3 * torch.pi / 4) | (phase < -3 * torch.pi / 4)
    imag_pos = (phase >= torch.pi / 4) & (phase < 3 * torch.pi / 4)
    imag_neg = (phase >= -3 * torch.pi / 4) & (phase < -torch.pi / 4)

    # 计算缩放因子
    real_scale = 1.0 / torch.clamp(w_real[real_pos|real_neg].abs().mean(), min=1e-5)
    imag_scale = 1.0 / torch.clamp(w_imag[imag_pos|imag_neg].abs().mean(), min=1e-5)

    # 量化
    qw_real = torch.zeros_like(w_real)
    qw_imag = torch.zeros_like(w_imag)
    qw_real[real_pos] = 1.0
    qw_imag[imag_pos] = 1.0
    qw_real[real_neg] = -1.0
    qw_imag[imag_neg] = -1.0

    # 应用缩放因子
    qw_real = qw_real / real_scale
    qw_imag = qw_imag / imag_scale

    return qw_real, qw_imag

def quant(key, tensor, f, weight_map):
    """
    处理实部和虚部的配对量化
    """
    if 'real' in key:
        imag_key = key.replace('real', 'imag')
        # 加载对应的虚部张量
        imag_tensor = get_tensor_from_file(imag_key, f, weight_map)
        q_real, q_imag = forward(tensor, imag_tensor)
        return q_real
    elif 'imag' in key:
        real_key = key.replace('imag', 'real')
        # 加载对应的实部张量
        real_tensor = get_tensor_from_file(real_key, f, weight_map)
        q_real, q_imag = forward(real_tensor, tensor)
        return q_imag
    else:
        return tensor  # 非复数张量不量化
```

---

## 3. 量化实现

### 3.1 量化数据类型

**GGML 类型定义** (`ggml/include/ggml.h:418`):

```cpp
enum ggml_type {
    ...
    GGML_TYPE_MXFP4   = 39,
    GGML_TYPE_IFAIRY  = 40,  // 自定义的复数量化类型
    GGML_TYPE_COUNT   = 41,
};
```

### 3.2 量化块结构

**块定义** (`ggml/src/ggml-common.h:260-264`):

```c
typedef struct {
    uint8_t qs[QK_K/4];      // 2 bits per element, QK_K=256
    ggml_half d_real;         // 实部缩放因子 (FP16)
    ggml_half d_imag;         // 虚部缩放因子 (FP16)
} block_ifairy;

static_assert(sizeof(block_ifairy) == sizeof(ggml_half) * 2 + QK_K / 4,
              "wrong ifairy block size/padding");
```

**块大小计算**:
- `QK_K = 256` (每个块包含 256 个元素)
- `qs` 数组: `256 / 4 = 64` 字节 (每个元素 2 bits)
- 缩放因子: `2 * sizeof(ggml_half) = 4` 字节
- **总大小**: 68 字节/块

**存储格式**:
- 每 2 bits 编码一个复数值: `00`(-1), `01`(1), `10`(-i), `11`(i)
- 实际值 = 量化值 * 缩放因子

### 3.3 量化函数

**量化实现** (`ggml/src/ggml-quants.c:2274-2319`):

```c
void quantize_row_ifairy_ref(const float * GGML_RESTRICT x_real,
                              const float * GGML_RESTRICT x_imag,
                              block_ifairy * GGML_RESTRICT y,
                              int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    // 1. 计算全局缩放因子 (整个张量的最大绝对值)
    float d_real = 0;
    float d_imag = 0;
    for (int64_t i = 0; i < k; i++) {
        d_real = MAX(d_real, (int)fabsf(x_real[i]));
        d_imag = MAX(d_imag, (int)fabsf(x_imag[i]));
    }

    const float id_real = d_real ? 1.0f/d_real : 0.0f;
    const float id_imag = d_imag ? 1.0f/d_imag : 0.0f;

    // 2. 按块量化
    for (int64_t i = 0; i < nb; i++) {
        y[i].d_real = GGML_FP32_TO_FP16(d_real);
        y[i].d_imag = GGML_FP32_TO_FP16(d_imag);

        // 每个块处理 256 个元素
        for (size_t j = 0; j < sizeof(y->qs); j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;

                // 每次处理 4 个元素,打包到一个字节
                for (size_t n = 0; n < 4; ++n) {
                    // 编码规则:
                    // 00 (0) -> -1 (实部为负)
                    // 01 (1) ->  1 (实部为正)
                    // 10 (2) -> -i (虚部为负)
                    // 11 (3) ->  i (虚部为正)

                    int xi = 0;
                    if (x_real[m + n*32] == 0) {
                        xi = (x_imag[m + n*32] > 0) ? 3 : 2;  // i 或 -i
                    } else {
                        xi = (x_real[m + n*32] > 0) ? 1 : 0;  // 1 或 -1
                    }
                    q += xi << (2*n);  // 2 bits per element
                }
                y[i].qs[j + m] = q;
            }
            x_real += 4*32;
            x_imag += 4*32;
        }
    }
}
```

### 3.4 反量化函数

**反量化实现** (`ggml/src/ggml-quants.c:2329-2348`):

```c
void dequantize_row_ifairy(const block_ifairy * GGML_RESTRICT x,
                            float * GGML_RESTRICT y_real,
                            float * GGML_RESTRICT y_imag,
                            int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; ++i) {
        // 加载缩放因子
        const float d_real = GGML_FP16_TO_FP32(x[i].d_real);
        const float d_imag = GGML_FP16_TO_FP32(x[i].d_imag);

        // 解码每个块
        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    // 提取 2 bits
                    int8_t q = (x[i].qs[j + m] >> (l*2)) & 3;

                    // 解码并应用缩放因子
                    // q=0: real=-1, imag=0
                    // q=1: real=1, imag=0
                    // q=2: real=0, imag=-1
                    // q=3: real=0, imag=1
                    *y_real++ = (float) ((q == 1) - (q == 0)) * d_real;
                    *y_imag++ = (float) ((q == 3) - (q == 2)) * d_imag;
                }
            }
        }
    }
}
```

### 3.5 量化特点总结

1. **2-bit 量化**: 每个复数元素只用 2 bits 表示,压缩比极高
2. **四值量化**: 只能表示 `{-1, 1, -i, i}` 四个值
3. **独立缩放**: 实部和虚部使用独立的缩放因子
4. **块级结构**: 每 256 个元素为一个块,共享缩放因子

---

## 4. 计算图加载

### 4.1 模型加载

模型加载时会读取 GGUF 文件中的张量,并根据张量类型自动选择对应的处理方法。

**类型信息注册** (`ggml/src/ggml.c:876`):

```c
[39] = { // GGML_TYPE_IFAIRY
    .type_name                = "ifairy",
    .blck_size                = QK_K,           // 256
    .type_size                = sizeof(block_ifairy),  // 68 bytes
    .is_quantized             = true,
    .to_float                 = dequantize_row_ifairy,
    .from_float_ref           = quantize_row_ifairy_ref,
    // ... 其他字段
},
```

### 4.2 自定义 RoPE 算子

iFairy 模型使用了自定义的 RoPE (Rotary Position Embedding) 实现。

**算子定义** (`ggml/include/ggml.h:559`):

```cpp
enum ggml_op {
    ...
    GGML_OP_GLU,
    GGML_OP_IFAIRY_ROPE,  // 自定义 RoPE 算子
    GGML_OP_COUNT,
};
```

**API 接口** (`ggml/include/ggml.h:1103-1108`):

```cpp
GGML_API struct ggml_tensor * ggml_ifairy_rope(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        int                   n_dims,
        int                   mode);
```

**RoPE 实现** (`ggml/src/ggml-cpu/ops.cpp:5901-6093`):

```cpp
static void ggml_compute_forward_rope_ifairy(
        const ggml_compute_params * params,
        ggml_tensor * dst,
        const bool forward) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];  // position IDs
    const ggml_tensor * src2 = dst->src[2];  // freq factors (可选)

    // 提取参数
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    // ... 提取其他参数

    // RoPE 计算逻辑 (mode==0 的情况)
    // 处理方式与标准 RoPE 类似,但针对复数表示进行了优化
    // 特别处理: 在 n_dims 之后的通道直接从 FP16 复制数据

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            // 获取位置索引
            const int64_t p = ((int32_t *) src1->data)[i2];

            // 预计算 cos 和 sin 值
            for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                const float theta = p * powf(freq_base, -i0 / n_dims);
                const float cos_theta = cosf(theta);
                const float sin_theta = sinf(theta);
                // 缓存到 cache 数组
            }

            // 应用旋转
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                    const int64_t ic = i0/2;
                    const float x0 = src[0];
                    const float x1 = src[n_dims];

                    dst_data[0]      = x0*cos_theta - x1*sin_theta;
                    dst_data[n_dims] = x0*sin_theta + x1*cos_theta;
                }

                // 对于 n_dims 之后的维度,从 FP16 源数据复制
                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                    dst_data[0] = GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)src)[1]);
                    dst_data[1] = GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)src)[0]);
                }
            }
        }
    }
}
```

### 4.3 计算图构建

在推理时,llama.cpp 会构建计算图,其中包含:

1. **Token Embedding**: 查表得到输入 token 的嵌入向量 (实部+虚部)
2. **Transformer 层** (重复 26 次):
   - **Attention Norm**: RMS Norm
   - **Self-Attention**:
     - Q/K/V 投影 (使用复数权重)
     - iFairy RoPE 位置编码
     - Attention 计算
     - Output 投影
   - **Attention Sub Norm**: 残差连接后的 Norm
   - **FFN Norm**: RMS Norm
   - **Feed-Forward**:
     - Gate 投影
     - Up 投影
     - Down 投影
   - **FFN Sub Norm**: 残差连接后的 Norm
3. **Output Norm**: 最终的 RMS Norm
4. **LM Head**: 输出 logits

---

## 5. 矩阵乘法与推理计算

### 5.1 矩阵乘法算子

iFairy 模型使用标准的 `ggml_mul_mat` 算子进行矩阵乘法,但会根据张量类型自动选择对应的向量点积实现。

**向量点积类型映射** (`ggml/src/ggml-cpu/ggml-cpu.c:379-382`):

```c
[GGML_TYPE_IFAIRY] = {
    .from_float        = quantize_row_ifairy_ref,
    .to_float          = dequantize_row_ifairy,
    .vec_dot_type      = GGML_TYPE_IFAIRY,
    .vec_dot           = (ggml_vec_dot_t) ggml_vec_dot_ifairy_q8_0,
    // ...
},
```

### 5.2 向量点积实现

由于 iFairy 使用 2-bit 量化,向量点积需要:

1. **解码量化数据**: 将 2-bit 数据解码为 `{-1, 1, -i, i}`
2. **复数乘法**: 计算 `(a_real + j*a_imag) * (b_real + j*b_imag)`
3. **累加**: 累加所有元素的乘积

**复数乘法公式**:
```
(a_r + j*a_i) * (b_r + j*b_i) = (a_r*b_r - a_i*b_i) + j*(a_r*b_i + a_i*b_r)
```

### 5.3 推理流程

```
输入 Token IDs
    ↓
Token Embedding (查表)
    ↓
[对于每一层 Layer 0~25]
│   ↓
│   Attention Norm (RMS Norm)
│   ↓
│   Self-Attention
│   │ - Q = X @ W_q_real + j * (X @ W_q_imag)
│   │ - K = X @ W_k_real + j * (X @ W_k_imag)
│   │ - V = X @ W_v_real + j * (X @ W_v_imag)
│   │ - Q, K = iFairy_RoPE(Q, K)
│   │ - Attn = softmax(Q @ K^T / sqrt(d_k)) @ V
│   │ - Out = Attn @ W_o_real + j * (Attn @ W_o_imag)
│   ↓
│   Attention Sub Norm + Residual
│   ↓
│   FFN Norm (RMS Norm)
│   ↓
│   Feed-Forward
│   │ - Gate = X @ W_gate
│   │ - Up = X @ W_up
│   │ - Hidden = SiLU(Gate) * Up
│   │ - Out = Hidden @ W_down
│   ↓
│   FFN Sub Norm + Residual
│   ↓
[下一层]
    ↓
Output Norm (RMS Norm)
    ↓
LM Head (取实部)
    ↓
输出 Logits
```

### 5.4 性能优化

llama.cpp 针对 iFairy 模型的性能优化包括:

1. **SIMD 优化**: 使用 AVX2/NEON 指令集加速向量点积
2. **多线程**: 利用多核 CPU 并行计算
3. **缓存优化**: 优化内存访问模式,提高缓存命中率
4. **量化加速**: 2-bit 量化显著减少内存带宽需求

---

## 6. 关键代码位置

### 6.1 模型架构定义

- **架构枚举**: `src/llama-arch.h:73`
- **架构名称**: `src/llama-arch.cpp:69`
- **张量映射**: `src/llama-arch.cpp:1506-1526`
- **模型加载**: `src/llama-model.cpp:1607-1615`

### 6.2 转换脚本

- **转换脚本**: `gguf-py/convert_ifairy.py`
- **张量映射**: `gguf-py/gguf/tensor_mapping.py`
- **常量定义**: `gguf-py/gguf/constants.py`

### 6.3 量化实现

- **GGML 类型**: `ggml/include/ggml.h:418`
- **块结构**: `ggml/src/ggml-common.h:260-264`
- **量化函数**: `ggml/src/ggml-quants.c:2274-2327`
- **反量化**: `ggml/src/ggml-quants.c:2329-2348`
- **类型特性**: `ggml/src/ggml-quants.h:36,64`

### 6.4 计算图构建

- **RoPE 算子**: `ggml/include/ggml.h:559,1103-1108`
- **RoPE 实现**: `ggml/src/ggml-cpu/ops.cpp:5901-6093`
- **算子调度**: `ggml/src/ggml-cpu/ggml-cpu.c:1864,2259,2771`

### 6.5 推理执行

- **类型注册**: `ggml/src/ggml.c:876`
- **算子名称**: `ggml/src/ggml.c:1026`
- **向量点积**: `ggml/src/ggml-cpu/ggml-cpu.c:379-382`
- **创建函数**: `ggml/src/ggml.c:4006`

---

## 7. 总结

iFairy 模型的推理流程可以概括为:

1. **转换阶段**: 使用 `convert_ifairy.py` 将 Hugging Face 模型转换为 GGUF 格式,同时进行基于相位的量化
2. **量化存储**: 使用自定义的 `GGML_TYPE_IFAIRY` 类型,以 2-bit 量化存储复数权重
3. **模型加载**: llama.cpp 加载 GGUF 文件,识别 iFairy 架构并注册相应的算子
4. **计算图构建**: 构建包含 iFairy RoPE 和复数矩阵乘法的计算图
5. **推理执行**: 通过高效的向量点积和 SIMD 优化执行推理

这种设计使得 iFairy 模型能够在保持较高精度的同时,实现极致的模型压缩(2-bit)和高效推理。

---

**文档版本**: 1.0
**最后更新**: 2025-10-11
**作者**: Claude Code Analysis
