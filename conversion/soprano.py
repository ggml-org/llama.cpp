from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .base import ModelBase, MmprojModel, gguf, logger
from .qwen import Qwen3Model


def _is_soprano_checkpoint(dir_model: Path) -> bool:
    if not (dir_model / "decoder.pth").is_file() or not (dir_model / "config.json").is_file():
        return False
    with open(dir_model / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("model_type") != "qwen3" or config.get("bos_token_id") != 3 or config.get("eos_token_id") != 3:
        return False
    if not (dir_model / "tokenizer.json").is_file():
        return False
    with open(dir_model / "tokenizer.json", encoding="utf-8") as f:
        added = {t["id"]: t["content"] for t in json.load(f).get("added_tokens", [])}
    return [added.get(i) for i in range(4)] == ["[UNK]", "[TEXT]", "[START]", "[STOP]"]


@ModelBase.register_hparams_loader(_is_soprano_checkpoint)
def _load_soprano_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected Soprano checkpoint")
    hparams = ModelBase.load_hparams(dir_model, False, guess=False)
    # the text model stays Qwen3, the mmproj holds the decoder
    hparams["text_config"] = {"architectures": ["SopranoTextModel"]}
    hparams["architectures"] = ["SopranoModel"]
    return hparams


@ModelBase.register("SopranoTextModel")
class SopranoTextModel(Qwen3Model):
    model_arch = gguf.MODEL_ARCH.QWEN3

    def get_vocab_base_pre(self, tokenizer) -> str:
        return "soprano"

    def set_vocab(self):
        super().set_vocab()
        self.gguf_writer.add_unk_token_id(0)


@ModelBase.register("SopranoModel")
@ModelBase.example("ekwek/Soprano-80M", "ekwek/Soprano-1.1-80M")
class SopranoModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = False

    def _load_decoder(self) -> dict[str, torch.Tensor]:
        state = torch.load(self.dir_model / "decoder.pth", map_location="cpu", mmap=True, weights_only=True)
        return state.get("state_dict", state)

    def get_audio_config(self):
        state = self._load_decoder()
        n_embd, n_inp, _ = state["decoder.embed.weight"].shape
        n_ff = state["decoder.convnext.0.pwconv1.weight"].shape[0]
        n_layer = sum(1 for name in state if name.endswith(".pwconv1.weight"))
        if n_inp != self.n_embd_text:
            raise ValueError(f"Soprano decoder input dim {n_inp} does not match the backbone dim {self.n_embd_text}")
        return {"num_hidden_layers": n_layer, "hidden_size": n_embd, "intermediate_size": n_ff}

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.SOPRANO)
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_gen_audio_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_gen_audio_block_count(self.hparams["num_hidden_layers"])
        self.gguf_writer.add_gen_audio_head_count(1)
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(1e-6)

    def get_tensors(self):
        yield from self._load_decoder().items()

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if "dwconv.weight" in new_name or (name == "decoder.embed.weight" and n_dims == 3):
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch, name, bid):
        if name == "head.istft.window":
            # the ISTFT in mtmd builds its own window
            if not torch.equal(data_torch.float(), torch.hann_window(data_torch.shape[0])):
                raise ValueError("Soprano requires a periodic Hann window")
            return
        if name == "decoder.embed.weight" and data_torch.shape[-1] == 1:
            data_torch = data_torch.squeeze(-1)

        T = gguf.MODEL_TENSOR
        if name.startswith("decoder.convnext."):
            _, _, blk, key, *suffix = name.split(".")
            block_map = {
                "dwconv":  T.A_GEN_WAV_UP_DWCONV,
                "norm":    T.A_GEN_WAV_UP_NORM,
                "pwconv1": T.A_GEN_WAV_UP_PW1,
                "pwconv2": T.A_GEN_WAV_UP_PW2,
                "gamma":   T.A_GEN_WAV_UP_GAMMA,
            }
            yield self.format_tensor_name(block_map[key], int(blk), suffix="".join("." + s for s in suffix)), data_torch
            return

        tensor_map = {
            "decoder.embed":            T.A_GEN_WAV_INPUT,
            "decoder.norm":             T.A_GEN_WAV_NORM,
            "decoder.final_layer_norm": T.A_GEN_WAV_OUTPUT_NORM,
            "head.out":                 T.A_GEN_WAV_OUTPUT,
        }
        key, suffix = name.rsplit(".", 1)
        yield self.format_tensor_name(tensor_map[key], suffix="." + suffix), data_torch
