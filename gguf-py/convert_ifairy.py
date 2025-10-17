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

def combine_complex_tensors(imag_part, real_part):
    """
    将虚部和实部合并为一个float32张量
    前16位存储虚部，后16位存储实部
    
    参数:
        imag_part: float16虚部张量
        real_part: float16实部张量
    返回:
        merged_tensor: float32合并张量
    """
    # 将float16转换为16位整数表示
    imag_int = imag_part.view(torch.int16)
    real_int = real_part.view(torch.int16)
    
    # 将16位整数扩展为32位整数（高位补零）
    imag_32 = imag_int.to(torch.int32)
    real_32 = real_int.to(torch.int32)
    
    # 将虚部移到高16位，实部在低16位
    merged_int = (imag_32 << 16) | (real_32 & 0xFFFF)
    
    # 将合并后的整数重新解释为float32
    merged_tensor = merged_int.view(torch.float32)
    
    return merged_tensor

def split_complex_tensors(merged_tensor):
    """
    将合并的float32张量拆分为虚部和实部
    
    参数:
        merged_tensor: float32合并张量
    返回:
        (imag_part, real_part): 元组，包含float16虚部和实部
    """
    # 将float32转换为32位整数表示
    merged_int = merged_tensor.view(torch.int32)
    
    # 提取高16位（虚部）
    imag_int = (merged_int >> 16).to(torch.int16)
    # 提取低16位（实部）
    real_int = (merged_int & 0xFFFF).to(torch.int16)
    
    # 将整数重新解释为float16
    imag_part = imag_int.view(torch.float16)
    real_part = real_int.view(torch.float16)
    
    return imag_part, real_part

# 接收key和对应的tensor，返回量化后的tensor
def quant_and_merge(key, tensor, f, weight_map):
    if 'real' in key:
        imag_key = key.replace('real', 'imag')
        f_name = weight_map.get(imag_key)
        if f_name is not None:
            with safe_open(f_name, framework="pt") as f1:
                imag_tensor = f1.get_tensor(imag_key).to(torch.float16)
        else:
            imag_tensor = f.get_tensor(imag_key).to(torch.float16)
        q_real, q_imag = forward(tensor, imag_tensor)
        return combine_complex_tensors(q_imag, q_real)
    elif 'imag' in key:
        real_key = key.replace('imag', 'real')
        f_name = weight_map.get(real_key)
        if f_name is not None:
            with safe_open(f_name, framework="pt") as f1:
                real_tensor = f1.get_tensor(real_key).to(torch.float16)
        else:
            real_tensor = f.get_tensor(real_key).to(torch.float16)
        q_real, q_imag = forward(real_tensor, tensor)   
        return combine_complex_tensors(q_imag, q_real)
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
    writer = gguf.GGUFWriter(output_file, arch=gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.IFAIRY])

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

    # 词汇表和分词器信息
    writer.add_vocab_size(config["vocab_size"])
    writer.add_tokenizer_model("llama") 

    index_name = "model.safetensors.index.json"
    index_file = Path(model_dir) / index_name

    # 4. 处理并添加张量
    model_files = list(Path(model_dir).glob("*.safetensors"))
    if not model_files:
        print(f"在 {model_dir} 中未找到 .safetensors 文件。")
        sys.exit(1)

    weight_map = {}
    if index_file.is_file():    
        if verbose:
            print(f"gguf: loading model weight map from '{index_name}'")
        with open(index_file, "r", encoding="utf-8") as f:
            index: dict[str, Any] = json.load(f)
            weight_map = index.get("weight_map")
            if weight_map is None or not isinstance(weight_map, dict):
                raise ValueError(f"Can't load 'weight_map' from {index_name!r}")

    try:
        for model_file_path in model_files:
            model_file_path = str(model_file_path)
            if verbose:
                print(f"gguf: loading model weights from '{model_file_path}'")
            with safe_open(model_file_path, framework="pt") as f:
                tensor_keys = f.keys()
                if verbose:
                    print(f"找到 {len(tensor_keys)} 个张量")

                for key in tensor_keys:
                    if 'lm_head' in key or 'embeddings' in key or 'norm' in key:
                        tensor_data = f.get_tensor(key).to(torch.float16)
                        numpy_array = tensor_data.cpu().numpy().astype(np.float16)
                        model_arch = gguf.MODEL_ARCH.IFAIRY
                        mapper = gguf.get_tensor_name_map(model_arch, config["num_hidden_layers"])
                        try:
                            mapped_name = mapper.get_name(key)
                            if mapped_name is None:
                                # 直接报错
                                raise Exception(f"No mapping found for tensor '{key}'")
                        except Exception as e:
                            print(f"Error mapping tensor name '{key}': {e}")
                            exit(1)
                        writer.add_tensor(mapped_name, numpy_array, raw_dtype=gguf.GGMLQuantizationType.F16)
                        if verbose:
                            print(f"添加张量: {mapped_name} (形状: {numpy_array.shape}")
                        continue
                    if '_imag' in key:
                        continue
                    tensor_data = f.get_tensor(key).to(torch.float16)
                    tensor_data = quant_and_merge(key, tensor_data, f, weight_map)
                    numpy_array = tensor_data.cpu().numpy().astype(np.float32)

                    model_arch = gguf.MODEL_ARCH.IFAIRY

                    mapper = gguf.get_tensor_name_map(model_arch, config["num_hidden_layers"])

                    try:
                        if '_real' in key or '_imag' in key:
                            key = key.replace('_real', '').replace('_imag', '')
                        mapped_name = mapper.get_name(key)
                        if mapped_name is None:
                            # 直接报错
                            raise Exception(f"No mapping found for tensor '{key}'")
                    except Exception as e:
                        print(f"Error mapping tensor name '{key}': {e}")
                        continue

                    # 添加张量，指定自定义的类型
                    writer.add_tensor(mapped_name, numpy_array, raw_dtype=gguf.GGMLQuantizationType.F16_I2)

                    if verbose:
                        print(f"添加张量: {mapped_name} (形状: {numpy_array.shape}")

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
