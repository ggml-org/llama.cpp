# CANN Backend Multi-Stream Parallel Implementation

## 思考过程记录

### 1. 分析Vulkan后端的多流并行实现

通过分析PR #15489 和 #15850，我了解到Vulkan后端的多流并行实现包含以下两个关键部分：

#### 1.1 PR #15489: 重写同步机制，允许节点之间的重叠执行

**核心思想**：
- 追踪需要同步的节点列表
- 只有当新节点依赖于未完成的节点时才进行同步
- 这允许一些重叠执行，从而提高性能

**关键实现**：
- 使用内存范围（地址）来判断依赖关系，而不是直接查看图结构
- 每个预分配的临时缓冲区（如dequantization或split_k）都有一个bool标记，指示它们是否被使用过并需要同步
- 性能提升：在RTX 5090上，部分模型性能提升约5-8%

#### 1.2 PR #15850: 图排序优化，允许更多的并行执行

**核心思想**：
- 添加backend proc（`graph_optimize`）允许后端修改计算图
- Vulkan实现会分析哪些节点相互依赖，并贪婪地重排序它们
- 将不相互依赖的节点分组在一起

**关键实现**：
- `ggml_vk_graph_optimize`函数实现图优化
- 保留特定的fusion pattern不被重排序（如RMS_NORM + MUL）
- 使用两遍扫描：第一遍抓取"real"节点，第二遍抓取view节点
- 最多检查接下来的20个节点是否可以提前执行

### 2. CANN后端当前状态分析

**现有基础设施**：
- 已有stream管理（`cann_ctx->stream()`）
- 支持ACL Graph模式
- 已有同步机制（`aclrtSynchronizeStream`）
- 后端接口中 `graph_optimize` 目前为NULL

**需要添加的功能**：
1. 实现 `ggml_backend_cann_graph_optimize` 函数
2. 可能需要添加多流支持
3. 添加环境变量控制开关

### 3. 设计方案

#### 3.1 实现 `graph_optimize` 函数

参考Vulkan的实现，我们需要：

```cpp
static void ggml_backend_cann_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * graph);
```

**核心逻辑**：
1. 判断是否禁用优化（环境变量控制）
2. 定义辅助函数判断节点是否为"空"（VIEW, RESHAPE等）
3. 定义辅助函数判断节点依赖关系
4. 重排序算法：
   - 遍历所有未使用的节点
   - 找到可以与当前节点并行执行的节点（不相互依赖）
   - 保留fusion pattern
   - 更新节点顺序

#### 3.2 环境变量

- `GGML_CANN_DISABLE_GRAPH_OPTIMIZE`: 禁用图优化

### 4. 实现计划

1. 在 `ggml-cann.cpp` 中实现 `ggml_backend_cann_graph_optimize`
2. 在 `ggml_backend_cann_interface` 中注册该函数
3. 编译验证
4. 使用Qwen 0.5B模型验证功能正确性

### 5. 预期收益

根据Vulkan后端的测试结果，图优化可以带来：
- 小模型（1B参数）：约5-8%的性能提升
- 中等模型（8B参数）：约3-4%的性能提升
- MoE模型：约6-7%的性能提升

这些收益来自于减少同步次数，允许更多操作并行执行。

## 实现代码

### 修改文件

`ggml/src/ggml-cann/ggml-cann.cpp`

### 主要更改

1. **添加头文件**：
   - `<algorithm>` - 用于 `std::find`
   - `<set>` - 用于 `std::set`
   - `<vector>` - 用于 `std::vector`

2. **实现 `ggml_backend_cann_graph_optimize` 函数**：
   - 位于 `ggml_backend_cann_event_wait` 函数之后
   - 约250行代码
   - 参考Vulkan后端的实现

3. **注册到backend interface**：
   - 修改 `ggml_backend_cann_interface` 结构体
   - 将 `graph_optimize` 从 `NULL` 改为 `ggml_backend_cann_graph_optimize`

### 关键算法

```cpp
// 核心优化算法伪代码
while (还有未处理的节点) {
    current_set = [下一个未处理的节点]
    
    // 保留fusion pattern
    if (match_pattern(ADD + RMS_NORM)) {
        keep_pattern_together()
        continue
    }
    
    // 第一遍：抓取可并行执行的"real"节点
    for (接下来的20个节点) {
        if (节点不依赖于未处理的节点) {
            if (支持fusion pattern) {
                add_to_current_set()
            }
        }
    }
    
    // 第二遍：抓取view节点
    for (接下来的20个节点) {
        if (is_empty(节点) && 依赖已满足) {
            add_to_current_set()
        }
    }
    
    // 更新节点顺序
    new_order.append(current_set)
}
```

### 支持的Fusion Pattern

- RMS_NORM + MUL
- MUL_MAT + ADD
- MUL_MAT_ID + ADD
- ADD + ADD
- ADD + RMS_NORM（CANN特有）
- ROPE + VIEW + SET_ROWS

## 测试结果

### 测试环境
- 模型：Qwen 2.5 0.5B Instruct FP16
- 设备：4x Ascend 910B4
- 测试命令：`llama-cli -m qwen2.5:0.5b-instruct-fp16 -n 50 -ngl 99`

### 测试输出
```
> Hello, how are you?

Hello! I'm Qwen, an AI developed by Alibaba Cloud. I'm here to answer any questions you may have and help with anything else you need help with. How can I assist you today?

[ Prompt: 1346.4 t/s | Generation: 142.8 t/s ]
```

### 结论
- ✅ 编译通过
- ✅ 模型加载成功
- ✅ 推理输出正确
- ✅ 正常退出

## 使用方法

### 启用图优化（默认）
```bash
./llama-cli -m model.gguf -ngl 99
```

### 禁用图优化
```bash
GGML_CANN_DISABLE_GRAPH_OPTIMIZE=1 ./llama-cli -m model.gguf -ngl 99
```

## 后续优化建议

1. **添加性能测试**：使用llama-bench进行before/after性能对比
2. **多流支持**：进一步实现真正的多流并行，利用CANN的多stream能力
3. **更多fusion pattern**：根据CANN的特性添加更多融合优化
4. **环境变量调优**：添加更细粒度的控制参数
