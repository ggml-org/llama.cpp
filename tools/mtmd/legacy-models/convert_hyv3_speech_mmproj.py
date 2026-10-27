#!/usr/bin/env python3
"""Export the HYV3Speech audio tower as a mmproj-only GGUF.

Reads the 403 `audio_tower.*` tensors from a merged HYV3Speech checkpoint and
writes a projector GGUF loadable by `llama-mtmd-cli --mmproj`.

    python3 convert_hyv3_speech_mmproj.py <hf_dir> <out.gguf>

Everything is stored as F32: the source weights are BF16 and downcasting them
to F16 would lose mantissa bits for no benefit (the tower is 330M params).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gguf-py"))

import gguf  # noqa: E402

ARCH = "clip"
PROJ_TYPE = "hyv3_speech"
K = gguf.Keys

MAX_POS_EMB = 1500


def f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().contiguous().float().numpy()


def lin(t: torch.Tensor) -> np.ndarray:
    """torch nn.Linear weight (out, in).

    GGUF stores dims reversed, so ggml reads ne = (in, out) from the torch
    layout as-is: do NOT transpose here.
    """
    return f32(t)


def conv2d_w(t: torch.Tensor) -> np.ndarray:
    """torch Conv2d weight (oc, ic, kh, kw) -> ggml ne (kw, kh, ic, oc)."""
    return f32(t)


def sinusoid_pos_embd(length: int, channels: int) -> np.ndarray:
    """Fixed sin/cos table, matching the reference SinusoidsPositionEmbedding.

    Built in fp32, sin half followed by cos half (not interleaved).
    Kept as (length, channels): ggml reads ne = (channels, length).
    """
    increment = math.log(10_000.0) / (channels // 2 - 1)
    inv_timescales = torch.exp(-increment * torch.arange(channels // 2).float())
    scaled = torch.arange(length).float()[:, None] * inv_timescales[None, :]
    return f32(torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1))


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    hf_dir = Path(sys.argv[1])
    out_path = sys.argv[2]

    cfg = json.loads((hf_dir / "config.json").read_text())
    a_cfg = cfg["audio_config"]
    n_layer = a_cfg["encoder_layers"]
    d_model = a_cfg["d_model"]
    n_head = a_cfg["encoder_attention_heads"]
    n_ff = a_cfg["encoder_ffn_dim"]
    n_mel = a_cfg["num_mel_bins"]
    out_dim = a_cfg["output_dim"]
    fsk = a_cfg["fsk_size"]
    llm_dim = a_cfg["llm_dim"]

    sd = {}
    for shard in sorted(hf_dir.glob("*.safetensors")):
        for k, v in load_file(str(shard)).items():
            if k.startswith("audio_tower."):
                sd[k] = v
    if len(sd) != 403:
        print(f"-- warning: expected 403 audio_tower tensors, found {len(sd)}")

    def get(name: str) -> torch.Tensor:
        if name not in sd:
            raise SystemExit(f"missing tensor: {name}")
        return sd[name]

    w = gguf.GGUFWriter(path=out_path, arch=ARCH, endianess=gguf.GGUFEndian.LITTLE)
    w.add_type(gguf.GGUFType.MMPROJ)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    w.add_name("hyv3_speech_audio_tower")
    w.add_description("HYV3Speech audio encoder + FSK adapter")

    w.add_bool(K.Clip.HAS_VISION_ENCODER, False)
    w.add_bool(K.Clip.HAS_AUDIO_ENCODER, True)
    w.add_string(K.Clip.PROJECTOR_TYPE, PROJ_TYPE)

    w.add_uint32(K.ClipAudio.NUM_MEL_BINS, n_mel)
    w.add_uint32(K.ClipAudio.EMBEDDING_LENGTH, d_model)
    w.add_uint32(K.ClipAudio.FEED_FORWARD_LENGTH, n_ff)
    w.add_uint32(K.ClipAudio.BLOCK_COUNT, n_layer)
    w.add_uint32(K.ClipAudio.PROJECTION_DIM, llm_dim)
    w.add_uint32(K.ClipAudio.Attention.HEAD_COUNT, n_head)
    w.add_float32(K.ClipAudio.Attention.LAYERNORM_EPS, 1e-5)
    w.add_uint32(K.ClipAudio.Projector.STACK_FACTOR, fsk)
    w.add_uint32(K.ClipAudio.Projector.DOWNSAMPLE_RATE, fsk)

    # conv stack
    for i in (1, 2, 3):
        w.add_tensor(f"a.conv2d.{i}.weight", conv2d_w(get(f"audio_tower.speech_encoder.conv2d{i}.weight")))
        w.add_tensor(f"a.conv2d.{i}.bias", f32(get(f"audio_tower.speech_encoder.conv2d{i}.bias")))
    w.add_tensor("a.conv_out.weight", lin(get("audio_tower.speech_encoder.conv_out.weight")))

    # transformer blocks
    P = "audio_tower.speech_encoder.layers."
    for i in range(n_layer):
        b = f"a.blk.{i}"
        for src, dst in (("q_proj", "attn_q"), ("k_proj", "attn_k"),
                         ("v_proj", "attn_v"), ("out_proj", "attn_out")):
            w.add_tensor(f"{b}.{dst}.weight", lin(get(f"{P}{i}.self_attn.{src}.weight")))
            w.add_tensor(f"{b}.{dst}.bias", f32(get(f"{P}{i}.self_attn.{src}.bias")))
        for src, dst in (("self_attn_layer_norm", "ln1"), ("final_layer_norm", "ln2")):
            w.add_tensor(f"{b}.{dst}.weight", f32(get(f"{P}{i}.{src}.weight")))
            w.add_tensor(f"{b}.{dst}.bias", f32(get(f"{P}{i}.{src}.bias")))
        w.add_tensor(f"{b}.ffn_up.weight", lin(get(f"{P}{i}.fc1.weight")))
        w.add_tensor(f"{b}.ffn_up.bias", f32(get(f"{P}{i}.fc1.bias")))
        w.add_tensor(f"{b}.ffn_down.weight", lin(get(f"{P}{i}.fc2.weight")))
        w.add_tensor(f"{b}.ffn_down.bias", f32(get(f"{P}{i}.fc2.bias")))

    # encoder head
    w.add_tensor("a.post_ln.weight", f32(get("audio_tower.speech_encoder.ln_post.weight")))
    w.add_tensor("a.post_ln.bias", f32(get("audio_tower.speech_encoder.ln_post.bias")))
    w.add_tensor("a.proj1.weight", lin(get("audio_tower.speech_encoder.proj1.weight")))
    w.add_tensor("a.proj1.bias", f32(get("audio_tower.speech_encoder.proj1.bias")))
    w.add_tensor("a.proj2.weight", lin(get("audio_tower.speech_encoder.proj2.weight")))
    w.add_tensor("a.proj2.bias", f32(get("audio_tower.speech_encoder.proj2.bias")))

    # FSK adapter: Linear -> SiLU -> Linear -> LayerNorm
    A = "audio_tower.speech_adapter.layers."
    w.add_tensor("mm.a.mlp.0.weight", lin(get(f"{A}0.weight")))
    w.add_tensor("mm.a.mlp.0.bias", f32(get(f"{A}0.bias")))
    w.add_tensor("mm.a.mlp.3.weight", lin(get(f"{A}3.weight")))
    w.add_tensor("mm.a.mlp.3.bias", f32(get(f"{A}3.bias")))
    w.add_tensor("mm.a.output_norm.weight", f32(get(f"{A}4.weight")))
    w.add_tensor("mm.a.output_norm.bias", f32(get(f"{A}4.bias")))

    # sinusoidal table is not in the checkpoint, generate it
    w.add_tensor("a.sinusoid_pos_embd", sinusoid_pos_embd(MAX_POS_EMB, d_model))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"-- wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
