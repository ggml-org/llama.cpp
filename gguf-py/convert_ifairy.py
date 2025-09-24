#!/usr/bin/env python3
"""
自定义 ComplexNetLM 架构 Hugging Face 模型转换为 GGUF 格式脚本
此脚本使用修改后的 gguf-py 库，支持自定义的 I2 数据类型映射。
"""
import sys
import os
import json
import argparse
from pathlib import Path

# 关键步骤：确保使用你修改过的 gguf-py 库
# 假设你的 gguf-py 目录路径，请修改为你的实际路径
#CUSTOM_GGUF_PY_PATH = "/path/to/your/gguf-py"
#sys.path.insert(0, CUSTOM_GGUF_PY_PATH)

import gguf
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModel
import numpy as np

# 从hf上直接扒下来的
def forward(w_real: torch.Tensor, w_imag: torch.Tensor):
    w_imag = w_imag.to('cuda') # 本来没这两行，我这里不写不支持angle操作
    w_real = w_real.to('cuda')

    phase = torch.angle(w_real + 1j * w_imag)
    real_pos = (phase >= -torch.pi / 4) & (phase < torch.pi / 4)
    real_neg = (phase >= 3 * torch.pi / 4) | (phase < -3 * torch.pi / 4)
    imag_pos = (phase >= torch.pi / 4) & (phase < 3 * torch.pi / 4)
    imag_neg = (phase >= -3 * torch.pi / 4) & (phase < -torch.pi / 4)
    real_scale = 1.0 / torch.clamp(w_real[real_pos|real_neg].abs().mean(), min=1e-5)
    imag_scale = 1.0 / torch.clamp(w_imag[imag_pos|imag_neg].abs().mean(), min=1e-5)
    
    qw_real = torch.zeros_like(w_real)
    qw_imag = torch.zeros_like(w_imag)

    qw_real[real_pos] = 1.0
    qw_imag[imag_pos] = 1.0
    qw_real[real_neg] = -1.0
    qw_imag[imag_neg] = -1.0

    qw_real = qw_real / real_scale
    qw_imag = qw_imag / imag_scale

    return qw_real, qw_imag

# 接收key和对应的tensor，返回量化后的tensor
def quant(key, tensor, f):
    if 'real' in key:
        imag_key = key.replace('real', 'imag')
        imag_tensor = f.get_tensor(imag_key).to(torch.float16)
        q_real, q_imag = forward(tensor, imag_tensor)
        return q_real
    elif 'imag' in key:
        real_key = key.replace('imag', 'real')
        real_tensor = f.get_tensor(real_key).to(torch.float16)
        q_real, q_imag = forward(real_tensor, tensor)
        return q_imag
    else:
        return tensor

def main():
    parser = argparse.ArgumentParser(description='Convert Hugging Face ComplexNetLM to GGUF with custom I2 type')
    parser.add_argument('model_dir', type=str, help='Path to the source Hugging Face model directory')
    parser.add_argument('output_file', type=str, help='Path to the output GGUF file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    model_dir = args.model_dir
    output_file = args.output_file
    verbose = args.verbose

    # 1. 加载模型配置
    config_path = os.path.join(model_dir, 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        if verbose:
            print(f"成功加载配置文件: {config_path}")
    except Exception as e:
        print(f"错误：无法从 {model_dir} 加载 config.json: {e}")
        sys.exit(1)

    # 2. 创建 GGUFWriter 并设置基本模型信息
    model_name = config.get("_name_or_path", "complexnet_model")
    writer = gguf.GGUFWriter(output_file, model_name)

    # 3. 根据 config.json 添加关键元数据
    # 模型架构和基本信息
    writer.add_name(model_name)
    writer.add_context_length(config["max_position_embeddings"]) # 2048
    writer.add_embedding_length(config["hidden_size"]) # 1536
    writer.add_block_count(config["num_hidden_layers"]) # 24
    writer.add_feed_forward_length(config["intermediate_size"]) # 4096
    writer.add_head_count(config["num_attention_heads"]) # 16
    writer.add_head_count_kv(config["num_key_value_heads"]) # 16
    writer.add_layer_norm_eps(config["rms_norm_eps"]) # 1e-05
    writer.add_rope_freq_base(config["rope_theta"]) # 10000.0
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_IFAIRY)

    # 词汇表和分词器信息 (需要根据你的分词器实际情况调整)
    writer.add_vocab_size(config["vocab_size"]) # 32000
    # 假设使用类似 LLaMA 的分词器，请根据你的 tokenizer_config.json 确认
    writer.add_tokenizer_model("llama") 

    # 添加自定义复数模型和量化方法的标识
    #writer.add_custom_metadata("complex_format", "split_real_imag")
    #writer.add_custom_metadata("quantization.method", "your_custom_i2_quantization") # 请替换为你的量化方法名
    #writer.add_custom_metadata("model.architecture", config["model_type"]) # "complexnet"

    # 4. 处理并添加张量
    # 确定模型权重文件
    model_files = list(Path(model_dir).glob("*.safetensors"))
    if not model_files:
        print(f"在 {model_dir} 中未找到 .safetensors 文件。")
        sys.exit(1)

    # 使用第一个找到的 safetensors 文件
    model_file_path = str(model_files[0])
    if verbose:
        print(f"使用模型文件: {model_file_path}")

    # 确定你自定义的 I2 数据类型在 gguf 中的枚举值
    # 假设你在 constants.py 中已定义，例如：GGML_TYPE_I2 = 100
    # 请根据你的实际定义修改！
    CUSTOM_GGML_TYPE_I2 = 100

    try:
        with safe_open(model_file_path, framework="pt") as f:
            tensor_keys = f.keys()
            if verbose:
                print(f"找到 {len(tensor_keys)} 个张量")

            for key in tensor_keys:
                tensor_data = f.get_tensor(key).to(torch.float16)
                tensor_data = quant(key, tensor_data, f)
                numpy_array = tensor_data.cpu().numpy().astype(np.float16)

                # 使用你修改后的 tensor_mapping.py 中的映射
                # 这里假设你的映射能正确工作，将 HF 名称映射到 GGUF 名称
                # 1. 首先，确定你的模型架构（根据你的 config.json）
                model_arch = gguf.MODEL_ARCH.IFAIRY # 假设你在 constants.py 中定义了

                # 2. 创建 TensorNameMap 实例
                # 参数通常是：模型架构、块数量（从 config 中获取）
                mapper = gguf.get_tensor_name_map(model_arch, config["num_hidden_layers"]) # 例如 24 层

                # 3. 使用 mapper 的 get_name 或类似方法来获取映射后的名称
                # 你需要查看 tensor_mapping.py 中 TensorNameMap 类的具体方法名
                # 假设方法名为 get_gguf_name，它接收原始名称并返回映射后的名称
                try:
                    if "lm_head" in key:
                        key = "lm_head"
                    mapped_name = mapper.get_name(key) # 这是关键调用！
                    if mapped_name is None:
                        # 如果没有找到映射，可以跳过或按原名称处理（不推荐）
                        print(f"Warning: No mapping found for tensor '{key}'. Skipping or using original name.")
                        continue # 或者 mapped_name = key
                except Exception as e:
                    print(f"Error mapping tensor name '{key}': {e}")
                    continue

                # 添加张量，指定自定义的 I2 类型
                writer.add_tensor(mapped_name, numpy_array, raw_dtype=gguf.GGMLQuantizationType.F16_I2)

                if verbose:
                    print(f"添加张量: {mapped_name} (形状: {numpy_array.shape}, 原始类型: F16, 目标类型: I2)")

    except Exception as e:
        print(f"处理张量时出错: {e}")
        sys.exit(1)

    # 5. 写入文件
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        print(f"转换成功！GGUF 文件已保存至: {output_file}")
    except Exception as e:
        print(f"写入 GGUF 文件时出错: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
