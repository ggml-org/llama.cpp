# KQV Trace Monitor

这个工具专门用于追踪和分析llama.cpp中名为"kqv_out"的张量及其源张量层次结构。

## 功能特性

- **KQV_OUT张量检测**: 自动检测并监控所有名称包含"kqv_out"的张量
- **源张量追踪**: 递归追踪kqv_out张量的源张量层次结构
- **层级过滤**: 可以指定只监控特定层的kqv_out张量
- **统计信息**: 提供张量的详细统计信息（均值、标准差、最小值、最大值）
- **可配置追踪**: 可以选择是否启用源张量追踪功能

## 编译

```bash
# 在llama.cpp根目录下编译
cmake --build build-arm64 --config Release --target llama-kqv-trace-monitor -j12
```

## 使用方法

### 基本用法

```bash
# 监控所有层的kqv_out张量
./build-arm64/bin/llama-kqv-trace-monitor -m model.gguf -p "Hello world" -ngl 0 -t 12

# 只监控第0层的kqv_out张量
./build-arm64/bin/llama-kqv-trace-monitor -m model.gguf -p "Hello world" -ngl 0 -t 12 --layer 0

# 禁用源张量追踪（只显示kqv_out本身的信息）
./build-arm64/bin/llama-kqv-trace-monitor -m model.gguf -p "Hello world" -ngl 0 -t 12 --no-trace-sources
```

### 参数说明

- `--layer <n>`: 只监控指定层（从0开始）的kqv_out张量。省略此参数则监控所有层
- `--no-trace-sources`: 禁用源张量追踪，只显示kqv_out张量本身的信息
- 其他参数与标准llama.cpp工具相同

### 输出示例

```
=== KQV_OUT TENSOR DETECTED ===
ggml_debug_kqv_trace:                 kqv_out_l0 = (f32)        ADD(wo_0{4096, 4096, 1, 1}, kqv_out{4096, 4, 1, 1}) = {4096, 4, 1, 1}

[KQV-TRACE] Layer 0 - kqv_out_l0: shape=[4096,4,1,1] type=f32 elements=16384
[KQV-TRACE]   stats: mean=0.001234, std=0.567890, min=-2.345678, max=3.456789

[KQV-TRACE] Source tensor hierarchy:
[SRC-0] kqv_out_l0: op=ADD, shape=[4096,4,1,1], type=f32
  [SRC-0] wo_0: op=NONE, shape=[4096,4096,1,1], type=f16
  [SRC-1] kqv_out: op=FLASH_ATTN_EXT, shape=[4096,4,1,1], type=f32
    [SRC-0] q_cur: op=MUL_MAT, shape=[128,32,4,1], type=f32
    [SRC-1] k_cur: op=VIEW, shape=[128,32,256,1], type=f16
    [SRC-2] v_cur: op=VIEW, shape=[128,32,256,1], type=f16
===============================
```

## 输出说明

### 张量信息
- **张量名称**: 显示检测到的kqv_out张量名称
- **操作类型**: 显示该张量是通过什么操作生成的
- **形状**: 显示张量的维度信息
- **数据类型**: 显示张量的数据类型（f32, f16等）

### 统计信息
- **mean**: 张量所有元素的平均值
- **std**: 标准差
- **min**: 最小值
- **max**: 最大值
- **elements**: 总元素数量

### 源张量层次结构
- 递归显示kqv_out张量的所有源张量
- 使用缩进表示层次关系
- 最多追踪3层深度以避免过深的递归
- 显示每个源张量的操作类型、形状和数据类型

## 应用场景

1. **调试注意力机制**: 了解kqv_out张量是如何从Q、K、V张量计算得出的
2. **性能分析**: 分析注意力计算的中间结果
3. **模型验证**: 验证注意力机制的实现是否正确
4. **优化分析**: 了解注意力计算的数据流和依赖关系

## 注意事项

- 工具会自动检测GPU/CPU内存，并在需要时复制数据进行分析
- 源张量追踪有深度限制（最多3层）以避免输出过于冗长
- 只处理F32和F16类型的张量数据
- 建议在小批量数据上测试，避免输出过多信息 