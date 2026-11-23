from __future__ import annotations

import json
import math
import re

import torch

from .base import LazyTorchTensor, MmprojModel, ModelBase, gguf
from .llama import LlamaModel


@ModelBase.register("ApertusForCausalLM")
@ModelBase.example("swiss-ai/Apertus-8B-Instruct-2509")
class ApertusModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.APERTUS
    undo_permute = False

    _alpha_n = {}
    _alpha_p = {}
    _beta = {}
    _eps = {}

    def modify_tensors(self, data_torch, name, bid):
        # Handle xIELU activation parameters
        n_layers = self.hparams["num_hidden_layers"]
        if name.endswith(".act_fn.alpha_n"):
            self._alpha_n[bid] = data_torch.to("cpu").float().item()
            if (len(self._alpha_n) == n_layers):
                self.gguf_writer.add_xielu_alpha_n([self._alpha_n[k] for k in sorted(self._alpha_n)])
            return
        if name.endswith(".act_fn.alpha_p"):
            self._alpha_p[bid] = data_torch.to("cpu").float().item()
            if (len(self._alpha_p) == n_layers):
                self.gguf_writer.add_xielu_alpha_p([self._alpha_p[k] for k in sorted(self._alpha_p)])
            return
        if name.endswith(".act_fn.beta"):
            self._beta[bid] = data_torch.to("cpu").float().item()
            if (len(self._beta) == n_layers):
                self.gguf_writer.add_xielu_beta([self._beta[k] for k in sorted(self._beta)])
            return
        if name.endswith(".act_fn.eps"):
            self._eps[bid] = data_torch.to("cpu").float().item()
            if (len(self._eps) == n_layers):
                self.gguf_writer.add_xielu_eps([self._eps[k] for k in sorted(self._eps)])
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register(
    "Apertus1p5ForConditionalGeneration",
    "Apertus1p5TextForCausalLM",
)
@ModelBase.example("swiss-ai/Apertus-v1.5-8B", "swiss-ai/Apertus-v1.5-70B")
class Apertus1p5Model(ApertusModel):
    model_arch = gguf.MODEL_ARCH.APERTUS

    def set_vocab(self):
        super().set_vocab()

        with open(self.dir_model / "tokenizer.json", encoding="utf-8") as f:
            tokenizer_json = json.load(f)
        added_tokens = {token["content"]: token["id"] for token in tokenizer_json["added_tokens"]}

        self.gguf_writer.add_eot_token_id(added_tokens["<|assistant_end|>"])
        self.gguf_writer.add_eom_token_id(added_tokens["<|tools_suffix|>"])

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        n_vocab = self.hparams["vocab_size"]
        n_vocab_out = self.hparams.get("output_vocab_size")
        if n_vocab_out is None:
            n_vocab_out = n_vocab
        if type(n_vocab_out) is not int or not 0 < n_vocab_out <= n_vocab:
            raise ValueError(f"Invalid output vocabulary size {n_vocab_out} for input vocabulary size {n_vocab}")
        self.gguf_writer.add_vocab_size_out(n_vocab_out)


@ModelBase.register("Apertus1p5ForConditionalGeneration")
@ModelBase.example("swiss-ai/Apertus-v1.5-8B", "swiss-ai/Apertus-v1.5-70B")
class Apertus1p5MmprojModel(MmprojModel):
    has_vision_encoder = True
    has_audio_encoder = True

    def get_vision_config(self):
        config_name = "vision_config" if "vision_config" in self.global_config else "vision_tokenizer_config"
        config = dict(self.global_config[config_name])
        config["image_size"] = config["resolution"]
        config["patch_size"] = 2 ** (len(config["channel_multiplier"]) - 1)
        config["hidden_size"] = config["base_channels"]
        config["intermediate_size"] = 0
        config["num_hidden_layers"] = len(config["channel_multiplier"])
        config["num_attention_heads"] = 1
        return config

    def get_audio_config(self):
        config_name = "audio_config" if self.global_config.get("audio_config") else "audio_tokenizer_config"
        config = dict(self.global_config[config_name])
        config["intermediate_size"] = 0
        config["num_hidden_layers"] = len(config["upsampling_ratios"])
        config["num_attention_heads"] = 1
        return config

    def set_gguf_parameters(self):
        assert self.hparams_vision is not None
        assert self.hparams_audio is not None
        audio = self.hparams_audio
        vision = self.hparams_vision
        image_size = self.preprocessor_config.get("size", self.preprocessor_config)
        if not isinstance(image_size, dict) or "min_pixels" not in image_size or "max_pixels" not in image_size:
            raise ValueError("Apertus image processor must specify both min_pixels and max_pixels")
        spatial_factor = 2 ** (len(vision["channel_multiplier"]) - 1)
        attention_levels = [
            level for level in range(len(vision["channel_multiplier"]))
            if vision["resolution"] // (2 ** level) in vision["attn_resolutions"]
        ]
        offset = self.global_config["image_token_offset"]
        if offset < 0 or vision["codebook_size"] <= 0 or offset + vision["codebook_size"] > self.global_config["text_config"]["vocab_size"]:
            raise ValueError("Apertus image codebook exceeds the input vocabulary")

        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_vision_encoder(True)
        self.gguf_writer.add_clip_has_audio_encoder(True)
        self.gguf_writer.add_clip_vision_projector_type(gguf.VisionProjectorType.APERTUS_VISION)
        self.gguf_writer.add_vision_projection_dim(self.n_embd_text)
        self.gguf_writer.add_vision_image_size(vision["resolution"])
        self.gguf_writer.add_vision_patch_size(spatial_factor)
        self.gguf_writer.add_vision_embedding_length(vision["base_channels"])
        self.gguf_writer.add_vision_feed_forward_length(0)
        self.gguf_writer.add_vision_block_count(len(vision["channel_multiplier"]))
        self.gguf_writer.add_vision_head_count(1)
        self.gguf_writer.add_vision_attention_layernorm_eps(1e-6)
        self.gguf_writer.add_vision_image_mean(self.preprocessor_config["image_mean"])
        self.gguf_writer.add_vision_image_std(self.preprocessor_config["image_std"])
        self.gguf_writer.add_vision_min_pixels(image_size["min_pixels"])
        self.gguf_writer.add_vision_max_pixels(image_size["max_pixels"])
        self.gguf_writer.add_vision_codebook_size(vision["codebook_size"])
        self.gguf_writer.add_vision_codebook_dim(vision["embed_dim"])
        self.gguf_writer.add_vision_token_offset(offset)
        self.gguf_writer.add_vision_spatial_factor(spatial_factor)
        self.gguf_writer.add_vision_channel_multipliers(vision["channel_multiplier"])
        self.gguf_writer.add_vision_residual_block_count(vision["num_res_blocks"])
        self.gguf_writer.add_vision_attention_levels(attention_levels)

        audio_offset = self.global_config["audio_token_offset"]
        if audio_offset < 0 or audio["codebook_size"] <= 0 or audio_offset + audio["codebook_size"] > self.global_config["text_config"]["vocab_size"]:
            raise ValueError("Apertus audio codebook exceeds the input vocabulary")
        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.APERTUS_AUDIO)
        self.gguf_writer.add_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_audio_embedding_length(audio["hidden_size"])
        self.gguf_writer.add_audio_feed_forward_length(0)
        self.gguf_writer.add_audio_block_count(len(audio["upsampling_ratios"]))
        self.gguf_writer.add_audio_head_count(1)
        self.gguf_writer.add_audio_attention_layernorm_eps(1e-6)
        self.gguf_writer.add_audio_num_mel_bins(1)
        self.gguf_writer.add_audio_codebook_size(audio["codebook_size"])
        self.gguf_writer.add_audio_codebook_dim(audio["codebook_dim"])
        self.gguf_writer.add_audio_token_offset(audio_offset)
        self.gguf_writer.add_audio_sample_rate(audio["sampling_rate"])
        self.gguf_writer.add_audio_hop_length(math.prod(audio["upsampling_ratios"]))
        self.gguf_writer.add_audio_downsample_ratios(list(reversed(audio["upsampling_ratios"])))
        self.gguf_writer.add_audio_residual_block_count(audio["num_residual_layers"])
        self.gguf_writer.add_audio_dilation_growth_rate(audio["dilation_growth_rate"])
        self.gguf_writer.add_audio_lstm_block_count(audio["num_lstm_layers"])

    @classmethod
    def filter_tensors(cls, item):
        name, _ = item
        if name.startswith(("model.vision_tokenizer.", "model.audio_tokenizer.encoder.")) or name == "model.audio_tokenizer.quantizer.codebook.embed":
            return item
        return None

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        return gguf.GGMLQuantizationType.F32

    def modify_tensors(self, data_torch, name, bid):
        if name.startswith("model.vision_tokenizer."):
            new_name = "v." + name.removeprefix("model.vision_tokenizer.")
            new_name = re.sub(r"^v\.encoder\.stages\.(\d+)\.layers\.(\d+)\.resnet\.", r"v.encoder.down.\1.block.\2.", new_name)
            new_name = re.sub(r"^v\.encoder\.stages\.(\d+)\.layers\.(\d+)\.attention\.", r"v.encoder.down.\1.attn.\2.", new_name)
            new_name = re.sub(r"^v\.encoder\.stages\.(\d+)\.downsample\.", r"v.encoder.down.\1.downsample.", new_name)
            yield new_name.replace("quantize.embedding.", "codebook."), data_torch
        elif name.startswith("model.audio_tokenizer.encoder."):
            weight_norm_g = ".parametrizations.weight.original0"
            weight_norm_v = ".parametrizations.weight.original1"
            if name.endswith(weight_norm_v):
                return
            if name.endswith(weight_norm_g):
                v_name = name.removesuffix(weight_norm_g) + weight_norm_v
                g = LazyTorchTensor.to_eager(data_torch).float()
                v = LazyTorchTensor.to_eager(self.model_tensors[v_name]()).float()
                data_torch = torch._weight_norm(v, g, dim=0)
                name = name.removesuffix(weight_norm_g) + ".weight"
            yield "a." + name.removeprefix("model.audio_tokenizer."), data_torch
        elif name == "model.audio_tokenizer.quantizer.codebook.embed":
            yield "a.codebook.weight", data_torch
