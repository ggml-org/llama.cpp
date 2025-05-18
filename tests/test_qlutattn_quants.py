import numpy as np
import torch

def quantize_tensor_numpy(w_np, n_bit=8, zero_point=True, q_group_size=-1):
    org_w_shape = w_np.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        if w_np.ndim == 1:
             reshaped_w = w_np.reshape(-1, q_group_size)
        else:
            num_elements_except_last = np.prod(org_w_shape[:-1])
            reshaped_w = w_np.reshape(num_elements_except_last, org_w_shape[-1])
            reshaped_w = reshaped_w.reshape(-1, q_group_size)
    elif q_group_size == -1:
        reshaped_w = w_np.reshape(1, -1) if w_np.ndim == 1 else w_np.reshape(-1, org_w_shape[-1])
    else:
        reshaped_w = w_np.reshape(1, -1)

    assert reshaped_w.ndim == 2

    if zero_point:
        max_val = np.amax(reshaped_w, axis=1, keepdims=True)
        min_val = np.amin(reshaped_w, axis=1, keepdims=True)
        max_int = 2 ** n_bit - 1
        scales = np.maximum(max_val - min_val, 1e-5) / max_int
        zeros_int = np.clip(-np.round(min_val / scales), 0, max_int)
    else:
        max_val = np.maximum(np.amax(np.abs(reshaped_w), axis=1, keepdims=True), 1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -2 ** (n_bit - 1)
        scales = max_val / max_int
        zeros_int = 0

    assert np.isnan(scales).sum() == 0
    assert np.isnan(reshaped_w).sum() == 0

    quantized_w = np.clip(np.round(reshaped_w / scales + zeros_int), 
                         -2**(n_bit - 1) if not zero_point else 0, 
                         2**n_bit - 1 if not zero_point else 2**n_bit - 1)
    final_quantized_w = quantized_w.reshape(org_w_shape)
    final_scales = scales.reshape(reshaped_w.shape[0], -1)

    if zero_point:
        final_quantized_w = final_quantized_w.astype(np.uint8)
        final_zeros = (zeros_int.astype(np.float32) - (2 ** (n_bit - 1))) * scales
        final_zeros = final_zeros.reshape(reshaped_w.shape[0], -1)
    else:
        final_quantized_w = (final_quantized_w - min_int).astype(np.uint8)
        final_zeros = None

    return final_quantized_w, final_scales, final_zeros

def dequantize_tensor_numpy(w_quant_np, scales_np, zeros_np_transformed, n_bit=8, zero_point=True, q_group_size=-1, original_shape=None):
    original_shape = w_quant_np.shape if original_shape is None else original_shape
    w_dequant = w_quant_np.astype(np.float32)

    if q_group_size > 0:
        assert original_shape[-1] % q_group_size == 0
        if w_quant_np.ndim == 1:
            reshaped_w_dequant = w_dequant.reshape(-1, q_group_size)
        else:
            num_elements = np.prod(original_shape[:-1])
            reshaped_w_dequant = w_dequant.reshape(num_elements, original_shape[-1]).reshape(-1, q_group_size)
    elif q_group_size == -1:
        reshaped_w_dequant = w_dequant.reshape(1, -1) if w_quant_np.ndim == 1 else w_dequant.reshape(-1, original_shape[-1])
    else:
        reshaped_w_dequant = w_dequant.reshape(1, -1)

    if zero_point:
        K = 2**(n_bit - 1)
        w_dequant_val = reshaped_w_dequant * scales_np - zeros_np_transformed - (scales_np * K)
    else:
        min_int = -2 ** (n_bit - 1)
        w_dequant_val = (reshaped_w_dequant + min_int) * scales_np
        
    return w_dequant_val.reshape(original_shape)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def test_quantization():
    nbits = 2
    q_group_size = 128

    print(f"\n--- Test Case 1: 1D array, zero_point=True, no grouping, nbits={nbits}, q_group_size={q_group_size} ---")
    w_orig_1d = np.random.uniform(0, 1, size=q_group_size).astype(np.float32)
    w_q, s, z = quantize_tensor_numpy(w_orig_1d, n_bit=nbits, zero_point=True, q_group_size=q_group_size)
    w_deq = dequantize_tensor_numpy(w_q, s, z, n_bit=nbits, zero_point=True, q_group_size=q_group_size, original_shape=w_orig_1d.shape)
    print(f"MSE: {mean_squared_error(w_orig_1d, w_deq):.6f}")

if __name__ == '__main__':
    test_quantization()

