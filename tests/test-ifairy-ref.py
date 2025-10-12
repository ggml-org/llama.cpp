#!/usr/bin/env python3
"""
iFairy 模型参考实现 - 用于生成测试数据和验证 C++ 实现

本脚本提供以下功能：
1. ROPE (Rotary Position Embedding) 的参考实现
2. 量化/反量化的参考实现
3. 复数矩阵乘法的参考实现
"""

import numpy as np
import json
from pathlib import Path


# ============================================================================
# 1. 量化函数参考实现
# ============================================================================

def quantize_ifairy_ref(w_real: np.ndarray, w_imag: np.ndarray):
    """
    iFairy 量化函数的参考实现

    将复数权重量化为 {-1, 1, -i, i} 四个值之一
    基于相位的量化方法

    Args:
        w_real: 实部数组
        w_imag: 虚部数组

    Returns:
        (qw_real, qw_imag, scale_real, scale_imag): 量化后的实部、虚部和缩放因子
    """
    # 计算相位
    phase = np.angle(w_real + 1j * w_imag)

    # 根据相位区间确定量化值
    # [-π/4, π/4):     实部为正, qw_real = 1
    # [π/4, 3π/4):     虚部为正, qw_imag = 1
    # [3π/4, -3π/4):   实部为负, qw_real = -1
    # [-3π/4, -π/4):   虚部为负, qw_imag = -1

    real_pos = (phase >= -np.pi / 4) & (phase < np.pi / 4)
    real_neg = (phase >= 3 * np.pi / 4) | (phase < -3 * np.pi / 4)
    imag_pos = (phase >= np.pi / 4) & (phase < 3 * np.pi / 4)
    imag_neg = (phase >= -3 * np.pi / 4) & (phase < -np.pi / 4)

    # 计算缩放因子（使用相应区域的平均绝对值）
    real_scale = 1.0 / np.clip(np.abs(w_real[real_pos | real_neg]).mean(), 1e-5, None)
    imag_scale = 1.0 / np.clip(np.abs(w_imag[imag_pos | imag_neg]).mean(), 1e-5, None)

    # 量化
    qw_real = np.zeros_like(w_real)
    qw_imag = np.zeros_like(w_imag)

    qw_real[real_pos] = 1.0
    qw_real[real_neg] = -1.0
    qw_imag[imag_pos] = 1.0
    qw_imag[imag_neg] = -1.0

    # 应用缩放因子
    qw_real = qw_real / real_scale
    qw_imag = qw_imag / imag_scale

    return qw_real, qw_imag, real_scale, imag_scale


def dequantize_ifairy_ref(q_bits: np.ndarray, scale_real: float, scale_imag: float):
    """
    iFairy 反量化函数的参考实现

    Args:
        q_bits: 量化后的 2-bit 数据 (0=-1, 1=1, 2=-i, 3=i)
        scale_real: 实部缩放因子
        scale_imag: 虚部缩放因子

    Returns:
        (dq_real, dq_imag): 反量化后的实部和虚部
    """
    # 解码 2-bit 数据
    # 0 -> real=-1, imag=0
    # 1 -> real=1, imag=0
    # 2 -> real=0, imag=-1
    # 3 -> real=0, imag=1

    dq_real = np.where(q_bits == 1, 1.0, np.where(q_bits == 0, -1.0, 0.0)) * scale_real
    dq_imag = np.where(q_bits == 3, 1.0, np.where(q_bits == 2, -1.0, 0.0)) * scale_imag

    return dq_real, dq_imag


def quantize_to_2bit(w_real: np.ndarray, w_imag: np.ndarray):
    """
    将复数权重量化为 2-bit 表示

    Returns:
        (q_bits, scale_real, scale_imag): 2-bit 量化数据和缩放因子
    """
    qw_real, qw_imag, scale_real, scale_imag = quantize_ifairy_ref(w_real, w_imag)

    # 转换为 2-bit 编码
    q_bits = np.zeros(w_real.shape, dtype=np.uint8)

    # 根据量化后的值确定 2-bit 编码
    q_bits[qw_real > 0] = 1   # real = 1
    q_bits[qw_real < 0] = 0   # real = -1
    q_bits[qw_imag > 0] = 3   # imag = i
    q_bits[qw_imag < 0] = 2   # imag = -i

    return q_bits, scale_real, scale_imag


# ============================================================================
# 2. ROPE 函数参考实现
# ============================================================================

def rope_ifairy_ref(x_real: np.ndarray, x_imag: np.ndarray, positions: np.ndarray,
                     n_dims: int, freq_base: float = 10000.0):
    """
    iFairy ROPE 位置编码的参考实现

    Args:
        x_real: 输入张量的实部 [batch, seq_len, n_heads, head_dim]
        x_imag: 输入张量的虚部 [batch, seq_len, n_heads, head_dim]
        positions: 位置索引 [seq_len]
        n_dims: 应用 ROPE 的维度数（通常是 head_dim 的一半）
        freq_base: 频率基数

    Returns:
        (rotated_real, rotated_imag): 应用 ROPE 后的实部和虚部
    """
    batch, seq_len, n_heads, head_dim = x_real.shape

    # 初始化输出
    out_real = np.copy(x_real)
    out_imag = np.copy(x_imag)

    # 对每个位置应用旋转
    for b in range(batch):
        for s in range(seq_len):
            pos = positions[s]

            # 对每个头应用旋转
            for h in range(n_heads):
                # 应用 ROPE 到前 n_dims 个维度（成对处理）
                for i in range(0, n_dims, 2):
                    # 计算旋转角度
                    theta = pos * (freq_base ** (-i / n_dims))
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)

                    # 提取当前对
                    x0_real = x_real[b, s, h, i]
                    x1_real = x_real[b, s, h, i + 1]
                    x0_imag = x_imag[b, s, h, i]
                    x1_imag = x_imag[b, s, h, i + 1]

                    # 应用旋转矩阵（复数旋转）
                    # [cos -sin] [x0]
                    # [sin  cos] [x1]
                    out_real[b, s, h, i] = x0_real * cos_theta - x1_real * sin_theta
                    out_real[b, s, h, i + 1] = x0_real * sin_theta + x1_real * cos_theta

                    out_imag[b, s, h, i] = x0_imag * cos_theta - x1_imag * sin_theta
                    out_imag[b, s, h, i + 1] = x0_imag * sin_theta + x1_imag * cos_theta

    return out_real, out_imag


# ============================================================================
# 3. 复数矩阵乘法参考实现
# ============================================================================

def complex_matmul_ref(a_real: np.ndarray, a_imag: np.ndarray,
                       b_real: np.ndarray, b_imag: np.ndarray):
    """
    复数矩阵乘法的参考实现

    (a_real + j*a_imag) @ (b_real + j*b_imag)
    = (a_real @ b_real - a_imag @ b_imag) + j*(a_real @ b_imag + a_imag @ b_real)

    Args:
        a_real, a_imag: 矩阵 A 的实部和虚部
        b_real, b_imag: 矩阵 B 的实部和虚部

    Returns:
        (c_real, c_imag): 结果矩阵 C 的实部和虚部
    """
    c_real = np.matmul(a_real, b_real) - np.matmul(a_imag, b_imag)
    c_imag = np.matmul(a_real, b_imag) + np.matmul(a_imag, b_real)

    return c_real, c_imag


# ============================================================================
# 4. 测试数据生成
# ============================================================================

def generate_test_data(output_dir: Path):
    """
    生成测试数据并保存为 JSON 文件
    """
    output_dir.mkdir(exist_ok=True)

    print("Generating iFairy test data...")

    # ========== 测试 1: 量化/反量化 ==========
    print("1. Generating quantization test data...")
    np.random.seed(42)

    # 生成随机复数权重 (256 个元素，对应一个量化块)
    w_real = np.random.randn(256).astype(np.float32) * 0.5
    w_imag = np.random.randn(256).astype(np.float32) * 0.5

    # 量化
    qw_real, qw_imag, scale_real, scale_imag = quantize_ifairy_ref(w_real, w_imag)
    q_bits, scale_r, scale_i = quantize_to_2bit(w_real, w_imag)

    # 反量化
    dq_real, dq_imag = dequantize_ifairy_ref(q_bits, scale_r, scale_i)

    quant_data = {
        "input_real": w_real.tolist(),
        "input_imag": w_imag.tolist(),
        "quantized_real": qw_real.tolist(),
        "quantized_imag": qw_imag.tolist(),
        "scale_real": float(scale_real),
        "scale_imag": float(scale_imag),
        "q_bits": q_bits.tolist(),
        "dequantized_real": dq_real.tolist(),
        "dequantized_imag": dq_imag.tolist()
    }

    with open(output_dir / "quant_test.json", "w") as f:
        json.dump(quant_data, f, indent=2)

    print(f"   Saved to {output_dir / 'quant_test.json'}")

    # ========== 测试 2: ROPE ==========
    print("2. Generating ROPE test data...")

    batch, seq_len, n_heads, head_dim = 1, 4, 2, 8
    n_dims = head_dim  # 应用 ROPE 到所有维度

    x_real = np.random.randn(batch, seq_len, n_heads, head_dim).astype(np.float32) * 0.1
    x_imag = np.random.randn(batch, seq_len, n_heads, head_dim).astype(np.float32) * 0.1
    positions = np.array([0, 1, 2, 3], dtype=np.int32)

    # 应用 ROPE
    rotated_real, rotated_imag = rope_ifairy_ref(x_real, x_imag, positions, n_dims)

    rope_data = {
        "input_real": x_real.flatten().tolist(),
        "input_imag": x_imag.flatten().tolist(),
        "positions": positions.tolist(),
        "batch": batch,
        "seq_len": seq_len,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "n_dims": n_dims,
        "freq_base": 10000.0,
        "output_real": rotated_real.flatten().tolist(),
        "output_imag": rotated_imag.flatten().tolist()
    }

    with open(output_dir / "rope_test.json", "w") as f:
        json.dump(rope_data, f, indent=2)

    print(f"   Saved to {output_dir / 'rope_test.json'}")

    # ========== 测试 3: 复数矩阵乘法 ==========
    print("3. Generating complex matmul test data...")

    M, K, N = 4, 8, 6
    a_real = np.random.randn(M, K).astype(np.float32) * 0.1
    a_imag = np.random.randn(M, K).astype(np.float32) * 0.1
    b_real = np.random.randn(K, N).astype(np.float32) * 0.1
    b_imag = np.random.randn(K, N).astype(np.float32) * 0.1

    # 复数矩阵乘法
    c_real, c_imag = complex_matmul_ref(a_real, a_imag, b_real, b_imag)

    matmul_data = {
        "a_real": a_real.flatten().tolist(),
        "a_imag": a_imag.flatten().tolist(),
        "b_real": b_real.flatten().tolist(),
        "b_imag": b_imag.flatten().tolist(),
        "c_real": c_real.flatten().tolist(),
        "c_imag": c_imag.flatten().tolist(),
        "M": M, "K": K, "N": N
    }

    with open(output_dir / "matmul_test.json", "w") as f:
        json.dump(matmul_data, f, indent=2)

    print(f"   Saved to {output_dir / 'matmul_test.json'}")

    print("\nTest data generation complete!")


# ============================================================================
# 5. 主函数
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path(__file__).parent / "ifairy-test-data"

    generate_test_data(output_dir)
