#!/usr/bin/env python3
"""Export the pinned MLX Parakeet encoder, without its ASR decoder."""
import argparse
from pathlib import Path
import sys
import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gguf-py'))
import gguf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('model', type=Path)
parser.add_argument('fixture', type=Path)
parser.add_argument('output', type=Path)
a = parser.parse_args()
if a.output.exists(): parser.error('output exists')
mx.set_default_device(mx.cpu)
w = gguf.GGUFWriter(a.output, 'clip')
w.add_bool('clip.has_audio_encoder', True)
w.add_bool('clip.has_vision_encoder', False)
w.add_string('clip.projector_type', 'parakeet')
w.add_bool('clip.audio.parakeet_mlx_frontend', True)
for k, v in {'embedding_length':512, 'feed_forward_length':2048, 'block_count':17, 'projection_dim':512, 'attention.head_count':8, 'num_mel_bins':80, 'subsampling_factor':8, 'conv_kernel_size':9}.items(): w.add_uint32('clip.audio.' + k, v)
w.add_float32('clip.audio.attention.layer_norm_epsilon', 1e-5)
mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MMPROJ, 17)
for name, value in mx.load(str(a.model / 'model.safetensors')).items():
    if not name.startswith('encoder.'): continue
    # from_pretrained casts all encoder weights to BF16 before inference.
    value = np.array(value.astype(mx.bfloat16).astype(mx.float32))
    target = 'sound_encoder.' + name
    target = target.replace('pre_encode.conv.', 'subsampling.layers.').replace('pre_encode.out.', 'subsampling.linear.')
    for before, after in [('linear_q','q_proj'), ('linear_k','k_proj'), ('linear_v','v_proj'), ('linear_out','o_proj'), ('linear_pos','relative_k_proj'), ('pos_bias_u','bias_u'), ('pos_bias_v','bias_v')]:
        target = target.replace('self_attn.' + before, 'self_attn.' + after)
    target = target.replace('conv.batch_norm.', 'conv.norm.')
    if 'subsampling.layers' in target:
        if target.endswith('weight'): value = value.transpose(0,3,1,2)
        else: value = value.reshape(1,-1,1,1)
    elif 'depthwise_conv.weight' in target:
        value = value[:,:,0]
    elif 'pointwise_conv' in target and target.endswith('weight'):
        value = value[:,0,:]
    mapped = mapping.get_name(target, try_suffixes=('.weight','.bias'))
    if mapped is None: raise ValueError('unmapped tensor: ' + name)
    w.add_tensor(mapped, np.ascontiguousarray(value))
    print(name, mapped, value.shape, flush=True)
w.add_tensor('a.mel_filters', np.fromfile(a.fixture/'filters.f32', dtype=np.float32))
w.add_tensor('a.window', np.fromfile(a.fixture/'window.f32', dtype=np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
