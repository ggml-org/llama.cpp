import argparse
from typing import Dict

import torch
import numpy as np
from gguf import *
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    AutoProcessor,
    Qwen2VLConfig
)


VISION = "clip.vision"


def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)


def to_gguf_name(name: str) -> str:
    og = name
    name = name.replace("text_model", "t").replace("vision_model", "v")
    name = name.replace("blocks", "blk").replace("embeddings.", "")
    name = name.replace("attn.", "attn_")
    name = name.replace("mlp.fc1", "ffn_down").replace("mlp.fc2", "ffn_up").replace("proj.", "out.")
    # name = name.replace("layrnorm", "ln").replace("layer_norm", "ln").replace("layernorm", "ln")
    name = name.replace("norm1", "ln1").replace("norm2", "ln2")
    name = name.replace("merger.mlp", 'mm')
    print(f"[to_gguf_name] {og} --> {name}")
    return name


def find_vision_tensors(qwen2vl, dtype) -> Dict[str, np.ndarray]:
    vision_model = qwen2vl.visual
    tensor_map = {}
    for name, ten in vision_model.state_dict().items():
        ten = ten.numpy()
        if 'qkv' in name:
            if ten.ndim == 2: # weight
                c3, _ = ten.shape
            else:             # bias
                c3 = ten.shape[0]
            assert c3 % 3 == 0
            c = c3 // 3
            wq = ten[:c]
            wk = ten[c: c * 2]
            wv = ten[c * 2:]
            tensor_map[to_gguf_name(f"vision_model.{name}").replace("qkv", "q")] = wq
            tensor_map[to_gguf_name(f"vision_model.{name}").replace("qkv", "k")] = wk
            tensor_map[to_gguf_name(f"vision_model.{name}").replace("qkv", "v")] = wv
        elif 'merger' in name:
            if name.endswith("ln_q.weight"):
                tensor_map['v.post_ln.weight'] = ten
            elif name.endswith("ln_q.bias"):
                tensor_map['v.post_ln.bias'] = ten
            else:
                # "merger.mlp.%d.weight/bias" --> "mm.%d.weight/bias"
                tensor_map[to_gguf_name(name)] = ten
        elif 'patch_embed.proj.weight' in name:
            # NOTE: split Conv3D into Conv2Ds
            c1, c2, kt, kh, kw = ten.shape
            assert kt == 2, "Current implmentation only support temporal_patch_size of 2"
            tensor_map["v.patch_embd.weight"] = ten[:, :, 0, ...]
            tensor_map["v.patch_embd.weight.1"] = ten[:, :, 1, ...]
        else:
            tensor_map[to_gguf_name(f"vision_model.{name}")] = ten

    for new_name, ten in tensor_map.items():
        if ten.ndim <= 1 or new_name.endswith("_norm.weight"):
            tensor_map[new_name] = ten.astype(np.float32)
        else:
            tensor_map[new_name] = ten.astype(dtype)
    tensor_map["v.position_embd.weight"] = np.zeros([10, 10], dtype=np.float32)  # dummy tensor, just here as a placeholder
    return tensor_map


def main(args):
    if args.data_type == 'fp32':
        dtype = torch.float32
        np_dtype = np.float32
        ftype = 0
    elif args.data_type == 'fp16':
        dtype = torch.float32
        np_dtype = np.float16
        ftype = 1
    else:
        raise ValueError()

    local_model = False
    model_path = ""
    model_name = args.model_name
    print("model_name: ", model_name)
    qwen2vl = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map="cpu"
    )
    cfg: Qwen2VLConfig = qwen2vl.config  # type: ignore[reportAssignmentType]
    vcfg = cfg.vision_config

    if os.path.isdir(model_name):
        local_model = True
        if model_name.endswith(os.sep):
            model_name = model_name[:-1]
        model_path = model_name
        model_name = os.path.basename(model_name)
    fname_out = f"{model_name.replace('/', '-').lower()}-vision.gguf"

    fout = GGUFWriter(path=fname_out, arch="clip")
    fout.add_description("image encoder for Qwen2VL")

    fout.add_file_type(ftype)
    fout.add_bool("clip.has_text_encoder", False)
    fout.add_bool("clip.has_vision_encoder", True)
    fout.add_bool("clip.has_qwen2vl_merger", True)
    fout.add_string("clip.projector_type", "qwen2vl_merger")

    print(cfg.vision_config)
    if 'silu' in cfg.vision_config.hidden_act.lower():
        fout.add_bool("clip.use_silu", True)
        fout.add_bool("clip.use_gelu", False)
    elif 'gelu' in cfg.vision_config.hidden_act.lower():
        fout.add_bool("clip.use_silu", False)
        fout.add_bool("clip.use_gelu", 'quick' not in cfg.vision_config.hidden_act.lower())
    else:
        raise ValueError()

    tensor_map = find_vision_tensors(qwen2vl, np_dtype)
    for name, data in tensor_map.items():
        fout.add_tensor(name, data)

    fout.add_uint32("clip.vision.patch_size", vcfg.patch_size)
    fout.add_uint32("clip.vision.image_size", 14 * 40)  # some reasonable size that is divable by (14*2)
    fout.add_uint32(k(KEY_EMBEDDING_LENGTH, VISION), vcfg.embed_dim)
    fout.add_uint32("clip.vision.projection_dim", vcfg.hidden_size)
    fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, VISION), vcfg.num_heads)
    fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, VISION), 1e-6)
    fout.add_uint32(k(KEY_BLOCK_COUNT, VISION), vcfg.depth)
    fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, VISION), 0)  # not sure what this does, put 0 here as a placeholder
    fout.add_name(model_name)
    """
    HACK: Since vision rope related parameter aren't stored in the `Qwen2VLConfig,
            it will be hardcoded in the `clip_image_build_graph` from `clip.cpp`.
    """

    if local_model:
        processor: Qwen2VLProcessor = AutoProcessor.from_pretrained(model_path)
    else:
        processor: Qwen2VLProcessor = AutoProcessor.from_pretrained(model_name)
    fout.add_array("clip.vision.image_mean", processor.image_processor.image_mean) # type: ignore[reportAttributeAccessIssue]
    fout.add_array("clip.vision.image_std", processor.image_processor.image_std) # type: ignore[reportAttributeAccessIssue]

    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()
    print("save model as: ", fname_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", nargs='?', default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--data_type", nargs='?', choices=['fp32', 'fp16'], default="fp32")
    args = parser.parse_args()
    main(args)
