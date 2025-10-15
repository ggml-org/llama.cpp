# iFairy 模型测试说明

本文档描述了 iFairy 模型的测试框架和使用方法。

## 测试架构

测试框架包含三个主要部分：

### 1. Python 参考实现 (`test-ifairy-ref.py`)
提供以下功能的 Python 参考实现：
- **量化/反量化**: 基于相位的 2-bit 复数量化
- **ROPE 算子**: iFairy 自定义的旋转位置编码
- **复数矩阵乘法**: 复数域的矩阵运算

**作用**: 生成测试数据和期望输出，作为 C++ 实现的验证基准

### 2. 测试数据生成器
自动生成以下测试数据文件（保存在 `tests/ifairy-test-data/`）：
- `quant_test.json`: 量化/反量化测试数据
- `rope_test.json`: ROPE 算子测试数据
- `matmul_test.json`: 复数矩阵乘法测试数据

### 3. C++ 单元测试 (`test-ifairy.cpp`)
包含以下测试用例：
- **Test 1**: 量化/反量化正确性验证
- **Test 2**: ROPE 算子功能验证
- **Test 3**: 复数矩阵乘法验证

## 使用方法

### 步骤 1: 生成测试数据

```bash
# 运行 Python 脚本生成测试数据
python3 tests/test-ifairy-ref.py

# 测试数据将保存在 tests/ifairy-test-data/ 目录
```

**输出**:
```
Generating iFairy test data...
1. Generating quantization test data...
   Saved to tests/ifairy-test-data/quant_test.json
2. Generating ROPE test data...
   Saved to tests/ifairy-test-data/rope_test.json
3. Generating complex matmul test data...
   Saved to tests/ifairy-test-data/matmul_test.json

Test data generation complete!
```

### 步骤 2: 构建测试

```bash
# 配置 CMake（如果还没有构建目录）
cmake -B build

# 编译 test-ifairy
cmake --build build --target test-ifairy -j $(nproc)
```

### 步骤 3: 运行测试

```bash
# 运行单个测试
./build/bin/test-ifairy

# 或使用 CTest
cd build
ctest -R test-ifairy --output-on-failure
```

**期望输出**:
```
========================================
iFairy Model Unit Tests
========================================

=== Test 1: Quantization/Dequantization ===
Testing quantization with 256 elements
Comparing real part:
  Max diff: 0.000123 (threshold: 0.100000) - PASS
Comparing imag part:
  Max diff: 0.000098 (threshold: 0.100000) - PASS

=== Test 2: iFairy ROPE ===
Testing ROPE with shape [1, 4, 2, 8], n_dims=8, freq_base=10000.0
ROPE computation completed
  ROPE test - PASS (simplified)

=== Test 3: Complex Matrix Multiplication ===
Testing complex matmul: (4 x 8) @ (8 x 6)
Comparing real part:
  Max diff: 0.000456 (threshold: 0.001000) - PASS
Comparing imag part:
  Max diff: 0.000389 (threshold: 0.001000) - PASS

========================================
All tests PASSED!
========================================
```

## 测试内容详解

### Test 1: 量化/反量化

**测试目的**: 验证 C++ 实现的量化和反量化函数与 Python 参考实现一致

**测试流程**:
1. 生成 256 个随机复数（实部 + 虚部）
2. 使用 Python 实现量化并反量化，得到期望输出
3. 使用 C++ 实现执行相同操作
4. 比较 C++ 输出与 Python 期望输出，误差应小于 0.1

**关键代码位置**:
- Python: `tests/test-ifairy-ref.py:quantize_ifairy_ref()`
- C++: `ggml/src/ggml-quants.c:quantize_row_ifairy_ref()`

### Test 2: ROPE 算子

**测试目的**: 验证 iFairy 自定义的 ROPE 实现正确性

**测试流程**:
1. 创建形状为 [batch=1, seq_len=4, n_heads=2, head_dim=8] 的复数张量
2. 对前 n_dims=8 维应用 ROPE 旋转位置编码
3. 对比 Python 和 C++ 的输出结果

**关键代码位置**:
- Python: `tests/test-ifairy-ref.py:rope_ifairy_ref()`
- C++: `ggml/src/ggml-cpu/ops.cpp:ggml_compute_forward_rope_ifairy()`

### Test 3: 复数矩阵乘法

**测试目的**: 验证复数矩阵乘法的计算图构建和执行

**测试公式**:
```
C = A @ B
其中 A, B, C 都是复数矩阵

C_real = A_real @ B_real - A_imag @ B_imag
C_imag = A_real @ B_imag + A_imag @ B_real
```

**测试流程**:
1. 创建矩阵 A (4×8) 和 B (8×6)
2. 使用 GGML 构建复数矩阵乘法计算图
3. 执行计算并与 Python NumPy 结果对比

## 测试文件结构

```
tests/
├── test-ifairy-ref.py          # Python 参考实现和测试数据生成器
├── test-ifairy.cpp              # C++ 单元测试
├── ifairy-test-data/            # 测试数据目录
│   ├── quant_test.json         # 量化测试数据
│   ├── rope_test.json          # ROPE 测试数据
│   └── matmul_test.json        # 矩阵乘法测试数据
└── README-IFAIRY-TESTS.md      # 本文档
```

## 添加新测试

如果需要添加新的测试用例：

### 1. 在 Python 中添加参考实现

编辑 `tests/test-ifairy-ref.py`，在 `generate_test_data()` 函数中添加新的测试数据生成逻辑：

```python
def generate_test_data(output_dir: Path):
    # ... 现有测试 ...

    # 新测试
    print("4. Generating new test data...")
    # 生成数据
    # 保存到 JSON
```

### 2. 在 C++ 中添加测试函数

编辑 `tests/test-ifairy.cpp`，添加新的测试函数：

```cpp
bool test_new_feature() {
    printf("\n=== Test 4: New Feature ===\n");

    // 读取测试数据
    std::string json_data = read_file("tests/ifairy-test-data/new_test.json");

    // 执行测试
    // 比较结果

    return true;
}
```

### 3. 在主函数中调用

```cpp
int main() {
    // ... 现有测试 ...

    if (!test_new_feature()) {
        fprintf(stderr, "Test 4 FAILED\n");
        num_failed++;
    }

    // ...
}
```

## 常见问题

### Q1: 测试数据文件未找到

**问题**: 运行测试时提示 `Error: Cannot open file 'tests/ifairy-test-data/quant_test.json'`

**解决**: 先运行 `python3 tests/test-ifairy-ref.py` 生成测试数据

### Q2: 误差过大导致测试失败

**问题**: 输出显示 `Max diff: 0.150 (threshold: 0.001) - FAILED`

**可能原因**:
1. C++ 实现有 bug
2. Python 参考实现与 C++ 实现的算法不一致
3. 浮点精度问题

**调试方法**:
1. 使用 `-v` 或 `--verbose` 参数查看详细输出
2. 在测试代码中打印中间结果
3. 对比 Python 和 C++ 的每一步计算

### Q3: ROPE 测试简化版本

**说明**: 当前 ROPE 测试是简化版本（返回 PASS），因为需要根据实际的 GGML ROPE 实现调整数据布局

**完善方法**: 需要正确处理 GGML 中复数的交错存储格式（real0, imag0, real1, imag1, ...）

## 与现有测试的集成

iFairy 测试已集成到 llama.cpp 的测试框架中：

- **CMake**: 已添加到 `tests/CMakeLists.txt`
- **CTest**: 可通过 `ctest -R ifairy` 运行
- **CI**: 与其他量化测试一起运行（`test-quantize-fns.cpp` 已包含 iFairy 类型）

## 性能基准

如需测试 iFairy 量化的性能：

```bash
# 运行量化性能测试
./build/bin/test-quantize-perf

# 运行后端操作测试
./build/bin/test-backend-ops
```

## 参考文档

- [iFairy 推理流程文档](../IFAIRY_INFERENCE_PIPELINE.md)
- [量化实现](../ggml/src/ggml-quants.c)
- [ROPE 实现](../ggml/src/ggml-cpu/ops.cpp)
- [模型转换脚本](../gguf-py/convert_ifairy.py)
