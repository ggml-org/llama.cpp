import torch
import numpy as np
from convert_ifairy import *

def test_merged_tensor_conversion():
    """
    测试merged_tensor的CPU转换和numpy转换的正确性
    """
    print("=" * 60)
    print("Merged Tensor转换正确性测试")
    print("=" * 60)
    
    # 创建测试数据
    print("\n1. 创建测试数据")
    print("-" * 30)
    
    # 生成随机的复数张量数据
    torch.manual_seed(42)  # 固定随机种子确保可重复性
    shape = (3, 4)  # 测试张量形状
    
    # 创建float16的实部和虚部
    original_real = torch.randn(shape, dtype=torch.float16, device='cuda')
    original_imag = torch.randn(shape, dtype=torch.float16, device='cuda')
    
    print(f"原始实部形状: {original_real.shape}")
    print(f"原始虚部形状: {original_imag.shape}")
    print(f"原始实部数据类型: {original_real.dtype}")
    print(f"原始虚部数据类型: {original_imag.dtype}")
    
    # 合并为complex32张量
    merged_tensor = combine_complex_tensors(original_imag, original_real)
    print(f"合并后张量形状: {merged_tensor.shape}")
    print(f"合并后数据类型: {merged_tensor.dtype}")
    
    # 测试转换过程
    print("\n2. 转换过程测试")
    print("-" * 30)
    
    # 第一步：转移到CPU
    cpu_tensor = merged_tensor.cpu()
    print(f"CPU张量设备: {cpu_tensor.device}")
    print(f"CPU张量形状: {cpu_tensor.shape}")
    print(f"CPU张量数据类型: {cpu_tensor.dtype}")
    
    # 第二步：转换为numpy数组
    numpy_array = cpu_tensor.numpy()
    print(f"NumPy数组形状: {numpy_array.shape}")
    print(f"NumPy数组数据类型: {numpy_array.dtype}")
    
    # 第三步：转换为float32类型（虽然可能已经是float32）
    numpy_float32 = numpy_array.astype(np.float32)
    print(f"Float32数组形状: {numpy_float32.shape}")
    print(f"Float32数组数据类型: {numpy_float32.dtype}")
    
    # 反向转换过程
    print("\n3. 反向转换过程")
    print("-" * 30)
    
    # 第一步：从numpy转回torch tensor
    recovered_tensor = torch.from_numpy(numpy_float32).to('cuda')
    print(f"恢复张量设备: {recovered_tensor.device}")
    print(f"恢复张量形状: {recovered_tensor.shape}")
    print(f"恢复张量数据类型: {recovered_tensor.dtype}")
    
    # 第二步：拆分回实部和虚部
    recovered_imag, recovered_real = split_complex_tensors(recovered_tensor)
    print(f"恢复实部形状: {recovered_real.shape}")
    print(f"恢复虚部形状: {recovered_imag.shape}")
    
    # 数据完整性验证
    print("\n4. 数据完整性验证")
    print("-" * 30)
    
    # 验证原始数据与恢复数据的一致性
    real_match = torch.allclose(original_real, recovered_real, atol=1e-6)
    imag_match = torch.allclose(original_imag, recovered_imag, atol=1e-6)
    
    print(f"实部数据完全一致: {real_match}")
    print(f"虚部数据完全一致: {imag_match}")
    
    # 计算数值误差
    real_max_error = torch.max(torch.abs(original_real - recovered_real))
    imag_max_error = torch.max(torch.abs(original_imag - recovered_imag))
    real_mean_error = torch.mean(torch.abs(original_real - recovered_real))
    imag_mean_error = torch.mean(torch.abs(original_imag - recovered_imag))
    
    print(f"实部最大误差: {real_max_error.item():.10f}")
    print(f"虚部最大误差: {imag_max_error.item():.10f}")
    print(f"实部平均误差: {real_mean_error.item():.10f}")
    print(f"虚部平均误差: {imag_mean_error.item():.10f}")
    
    # 详细数据对比（前几个元素）
    print("\n5. 详细数据对比（前6个元素）")
    print("-" * 30)
    
    print("原始实部:", original_real.flatten()[:6].cpu().numpy())
    print("恢复实部:", recovered_real.flatten()[:6].cpu().numpy())
    print("原始虚部:", original_imag.flatten()[:6].cpu().numpy())
    print("恢复虚部:", recovered_imag.flatten()[:6].cpu().numpy())
    
    # 内存布局验证
    print("\n6. 内存布局验证")
    print("-" * 30)
    
    # 检查内存地址是否相同（应该不同，因为经历了复制）
    print(f"原始merged_tensor内存地址: {id(merged_tensor)}")
    print(f"恢复tensor内存地址: {id(recovered_tensor)}")
    print(f"内存地址相同: {id(merged_tensor) == id(recovered_tensor)}")
    
    # 检查数据内容是否相同
    content_match = torch.allclose(merged_tensor, recovered_tensor, atol=1e-6)
    print(f"数据内容完全一致: {content_match}")
    
    # 性能测试
    print("\n7. 性能测试")
    print("-" * 30)
    
    import time
    
    # 测试转换性能
    start_time = time.time()
    for _ in range(1000):
        temp_numpy = merged_tensor.cpu().numpy().astype(np.float32)
        temp_tensor = torch.from_numpy(temp_numpy).to('cuda')
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 1000 * 1000  # 转换为毫秒
    print(f"平均转换时间: {avg_time:.4f} 毫秒")
    
    # 最终验证结果
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    conversion_successful = real_match and imag_match and content_match
    
    if conversion_successful:
        print("✅ 转换测试通过！")
        print("✅ merged_tensor.cpu().numpy().astype(np.float32) 转换完全可逆")
        print("✅ 数据在转换过程中保持完整无损")
    else:
        print("❌ 转换测试失败！")
        print("❌ 数据在转换过程中出现了误差")
    
    return conversion_successful

def detailed_bitwise_test():
    """
    详细的位级精度测试
    """
    print("\n" + "=" * 60)
    print("位级精度测试")
    print("=" * 60)
    
    # 创建特定的测试值
    test_values = [
        (1.0, 0.5),    # 正常值
        (0.0, 0.0),    # 零值
        (-1.0, -0.5),  # 负值
        (torch.finfo(torch.float16).max, torch.finfo(torch.float16).min),  # 极值
    ]
    
    all_passed = True
    
    for i, (real_val, imag_val) in enumerate(test_values):
        print(f"\n测试案例 {i+1}: real={real_val}, imag={imag_val}")
        print("-" * 40)
        
        # 创建张量
        real_tensor = torch.tensor([real_val], dtype=torch.float16, device='cuda')
        imag_tensor = torch.tensor([imag_val], dtype=torch.float16, device='cuda')
        
        # 完整转换流程
        merged = combine_complex_tensors(imag_tensor, real_tensor)
        numpy_converted = merged.cpu().numpy().astype(np.float32)
        recovered_tensor = torch.from_numpy(numpy_converted).to('cuda')
        recovered_imag, recovered_real = split_complex_tensors(recovered_tensor)
        
        # 位级比较
        original_bits = merged.cpu().view(torch.int32).numpy()
        recovered_bits = recovered_tensor.cpu().view(torch.int32).numpy()
        
        bits_match = np.array_equal(original_bits, recovered_bits)
        values_match = torch.allclose(real_tensor, recovered_real) and \
                      torch.allclose(imag_tensor, recovered_imag)
        
        print(f"位模式匹配: {bits_match}")
        print(f"数值匹配: {values_match}")
        print(f"原始位: {format(original_bits[0], '032b')}")
        print(f"恢复位: {format(recovered_bits[0], '032b')}")
        
        if not (bits_match and values_match):
            all_passed = False
    
    print(f"\n位级测试总体结果: {'通过' if all_passed else '失败'}")
    return all_passed

# 运行测试
if __name__ == "__main__":
    # 运行主测试
    main_test_passed = test_merged_tensor_conversion()
    
    # 运行位级测试
    bit_test_passed = detailed_bitwise_test()
    
    # 最终总结
    print("\n" + "=" * 60)
    print("最终测试总结")
    print("=" * 60)
    
    if main_test_passed and bit_test_passed:
        print("🎉 所有测试完全通过！")
        print("✅ merged_tensor.cpu().numpy().astype(np.float32) 转换验证成功")
        print("✅ 数据转换过程完全可逆且无损")
    else:
        print("⚠️  部分测试未通过，请检查实现细节")