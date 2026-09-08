# Copyright (c) 2026 codec.cpp contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from .base import ModelBase, MmprojModel, gguf


@ModelBase.register("Qwen3ForCausalLM")
@ModelBase.example("ekwek/Soprano-1.1-80M")
class SopranoModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = False

    def get_audio_config(self):
        return {"num_hidden_layers": 8}

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.SOPRANO)
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(768)
        self.gguf_writer.add_gen_audio_feed_forward_length(2304)
        self.gguf_writer.add_gen_audio_block_count(8)
        self.gguf_writer.add_gen_audio_head_count(1)
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(1e-6)

    def get_tensors(self):
        state = torch.load(self.dir_model / "decoder.pth", map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        yield from state.items()

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if "dwconv.weight" in new_name:
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch, name, bid):
        if name == "head.istft.window":
            expected = torch.hann_window(2048)
            if not torch.equal(data_torch.float(), expected):
                raise ValueError("Soprano requires the periodic 2048-sample Hann window")
            return
        if name == "decoder.embed.weight":
            data_torch = data_torch.squeeze(-1)
        yield self.map_tensor_name(name), data_torch
