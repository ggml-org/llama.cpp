import torch
import numpy as np
from convert_ifairy import *

def test_complex_tensor_operations():
    """
    测试复数张量合并与拆分操作的数据完整性
    """
    print("=" * 60)
    print("复数张量合并与拆分操作完整性测试")
    print("=" * 60)
    
    # 测试1：基本功能测试
    print("\n1. 基本功能测试")
    print("-" * 30)
    
    # 创建简单的测试数据
    real_part = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16, device='cuda')
    imag_part = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float16, device='cuda')
    
    print(f"原始实部: {real_part.cpu().numpy()}")
    print(f"原始虚部: {imag_part.cpu().numpy()}")
    
    # 合并操作
    merged_tensor = combine_complex_tensors(imag_part, real_part)
    print(f"合并后张量形状: {merged_tensor.shape}")
    print(f"合并后数据类型: {merged_tensor.dtype}")
    
    # 拆分操作
    recovered_imag, recovered_real = split_complex_tensors(merged_tensor)
    
    print(f"恢复实部: {recovered_real.cpu().numpy()}")
    print(f"恢复虚部: {recovered_imag.cpu().numpy()}")
    
    # 验证数据一致性
    real_match = torch.allclose(real_part, recovered_real, atol=1e-4)
    imag_match = torch.allclose(imag_part, recovered_imag, atol=1e-4)
    
    print(f"实部数据一致: {real_match}")
    print(f"虚部数据一致: {imag_match}")
    
    # 测试2：随机数据大规模测试
    print("\n2. 随机数据大规模测试")
    print("-" * 30)
    
    # 生成随机测试数据
    torch.manual_seed(42)  # 固定随机种子确保可重复性
    shape = (100, 50)  # 较大的张量形状
    
    original_real = torch.randn(shape, dtype=torch.float16, device='cuda')
    original_imag = torch.randn(shape, dtype=torch.float16, device='cuda')
    
    # 合并和拆分
    merged = combine_complex_tensors(original_imag, original_real)
    recovered_imag, recovered_real = split_complex_tensors(merged)
    
    # 计算数值误差
    real_error = torch.max(torch.abs(original_real - recovered_real))
    imag_error = torch.max(torch.abs(original_imag - recovered_imag))
    
    print(f"最大实部误差: {real_error.item():.6f}")
    print(f"最大虚部误差: {imag_error.item():.6f}")
    print(f"实部数据完全一致: {torch.allclose(original_real, recovered_real, atol=1e-6)}")
    print(f"虚部数据完全一致: {torch.allclose(original_imag, recovered_imag, atol=1e-6)}")
    
    # 测试3：边缘情况测试
    print("\n3. 边缘情况测试")
    print("-" * 30)
    
    # 测试零值
    zero_real = torch.zeros((5, 5), dtype=torch.float16, device='cuda')
    zero_imag = torch.zeros((5, 5), dtype=torch.float16, device='cuda')
    
    zero_merged = combine_complex_tensors(zero_imag, zero_real)
    zero_recovered_imag, zero_recovered_real = split_complex_tensors(zero_merged)
    
    zero_test = torch.allclose(zero_real, zero_recovered_real) and torch.allclose(zero_imag, zero_recovered_imag)
    print(f"零值测试通过: {zero_test}")
    
    # 测试极值
    max_val = torch.finfo(torch.float16).max
    extreme_real = torch.tensor([max_val, -max_val], dtype=torch.float16, device='cuda')
    extreme_imag = torch.tensor([-max_val, max_val], dtype=torch.float16, device='cuda')
    
    extreme_merged = combine_complex_tensors(extreme_imag, extreme_real)
    extreme_recovered_imag, extreme_recovered_real = split_complex_tensors(extreme_merged)
    
    extreme_test = torch.allclose(extreme_real, extreme_recovered_real) and \
                  torch.allclose(extreme_imag, extreme_recovered_imag)
    print(f"极值测试通过: {extreme_test}")
    
    # 测试4：内存布局验证
    print("\n4. 内存布局验证")
    print("-" * 30)
    
    # 检查合并后张量的内存使用
    original_size = original_real.element_size() * original_real.numel() + \
                   original_imag.element_size() * original_imag.numel()
    merged_size = merged.element_size() * merged.numel()
    
    print(f"原始数据内存占用: {original_size} 字节")
    print(f"合并后内存占用: {merged_size} 字节")
    print(f"内存占用比例: {merged_size/original_size:.2f}")
    
    # 测试5：性能测试
    print("\n5. 性能测试")
    print("-" * 30)
    
    import time
    
    # 大规模性能测试
    large_shape = (1000, 1000)
    large_real = torch.randn(large_shape, dtype=torch.float16, device='cuda')
    large_imag = torch.randn(large_shape, dtype=torch.float16, device='cuda')
    
    # 预热GPU
    for _ in range(10):
        _ = combine_complex_tensors(large_imag, large_real)
    
    # 正式测试
    start_time = time.time()
    for _ in range(100):
        merged_large = combine_complex_tensors(large_imag, large_real)
        recovered_large_imag, recovered_large_real = split_complex_tensors(merged_large)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100 * 1000  # 转换为毫秒
    print(f"平均处理时间: {avg_time:.2f} 毫秒")
    
    # 最终验证
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    all_tests_passed = (
        real_match and imag_match and 
        zero_test and extreme_test and
        torch.allclose(original_real, recovered_real) and
        torch.allclose(original_imag, recovered_imag) and
        torch.allclose(large_real, recovered_large_real) and
        torch.allclose(large_imag, recovered_large_imag)
    )
    
    if all_tests_passed:
        print("✅ 所有测试通过！复数张量合并与拆分操作完全可逆。")
        print("✅ 数据在转换过程中没有损失或破坏。")
    else:
        print("❌ 部分测试失败，请检查实现逻辑。")
    
    return all_tests_passed

# 运行测试
if __name__ == "__main__":
    test_complex_tensor_operations()