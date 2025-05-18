# 单算子调用方式
import torch
import torch_npu
import math

def load_float_array_to_tensor(file_path, shape, dtype):
    with open(file_path, 'r') as file:
        # 读取文件内容并按空格分割
        data = file.read().strip().split()
        # 将字符串转换为浮点数
        float_array = [float(num) for num in data]
        # 转换为 PyTorch 张量
        tensor = torch.tensor(float_array, dtype=dtype).reshape(shape).npu()
    return tensor

batch = 1
nhead_q = 4
nhead_kv = nhead_q
seq_q = 1
dims  = 64
seq_kv = 512
layout="BNSD"

scale_value = 1 / pow(dims, 0.5)

q_tensor = load_float_array_to_tensor("/data/home/2101111451/pr/llama.cpp/output_acl_short_0_q.txt", 
                                      (batch, nhead_q, seq_q, dims), torch.float16)
k_tensor = load_float_array_to_tensor("/data/home/2101111451/pr/llama.cpp/output_acl_short_3_k.txt", 
                                      (batch, nhead_kv, seq_kv, dims), torch.float16)

v_tensor = load_float_array_to_tensor("/data/home/2101111451/pr/llama.cpp/output_acl_short_4_v.txt", 
                                      (batch, nhead_kv, seq_kv, dims), torch.float16)

pse_tensor = load_float_array_to_tensor("/data/home/2101111451/pr/llama.cpp/output_acl_short_1_mask.txt", 
                                      (1, 1, -1, seq_kv), torch.float16)

print(q_tensor.shape, k_tensor.shape, v_tensor.shape, pse_tensor.shape)

# 调用IFA算子
out = torch_npu.npu_incre_flash_attention(q_tensor, k_tensor, v_tensor, pse_shift=pse_tensor,
                                          num_heads=nhead_q, num_key_value_heads=nhead_kv, 
                                          input_layout=layout, scale_value=scale_value)

