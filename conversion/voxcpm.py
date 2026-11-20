from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, MmprojModel, TextModel, LazyTorchTensor, gguf, logger
from .drama3 import fold_snake
from .minicpm import MiniCPMModel

# sample rate of the AudioVAE decoder output, picks the row of the sample-rate conditioning
VAE_OUT_SAMPLE_RATE = 48000


def _load_config(dir_model: Path) -> dict[str, Any] | None:
    path = dir_model / "config.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    return config if config.get("architecture") == "voxcpm2" else None


def _is_voxcpm2_checkpoint(dir_model: Path) -> bool:
    return _load_config(dir_model) is not None


@ModelBase.register_hparams_loader(_is_voxcpm2_checkpoint)
def _load_voxcpm2_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected VoxCPM2 checkpoint")
    config = _load_config(dir_model)
    assert config is not None
    if config["lm_config"].get("use_mup"):
        raise NotImplementedError("VoxCPM2: use_mup = true is not supported")
    if config["dit_config"].get("mean_mode"):
        raise NotImplementedError("VoxCPM2: dit mean_mode = true is not supported")
    if not config.get("residual_lm_no_rope"):
        raise NotImplementedError("VoxCPM2: residual LM with RoPE is not supported")
    for key in ("hidden_dim", "ffn_dim", "num_heads", "num_layers", "kv_channels"):
        if config["encoder_config"][key] != config["dit_config"][key]:
            raise NotImplementedError(f"VoxCPM2: LocEnc and LocDiT differ in {key}")
    return {**config, "architectures": ["VoxCPM2Model"], "text_config": config["lm_config"], "audio_config": config["dit_config"]}


@ModelBase.register("VoxCPM2Model")
@ModelBase.example("openbmb/VoxCPM2")
class VoxCPM2TextModel(MiniCPMModel):
    model_arch = gguf.MODEL_ARCH.MINICPM

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        # the text-semantic LM is a plain MiniCPM, the rest goes to the mmproj
        if not name.startswith("base_lm."):
            return None
        return super().filter_tensors(("model." + name.removeprefix("base_lm."), gen))

    def set_gguf_parameters(self):
        TextModel.set_gguf_parameters(self)
        # use_mup = false: no embedding, residual or logit scaling
        self.gguf_writer.add_embedding_scale(1.0)
        self.gguf_writer.add_residual_scale(1.0)
        self.gguf_writer.add_logit_scale(1.0)

    def set_vocab(self):
        # the checkpoint has no tokenizer.model and its tokenizer class is remote code, so read tokenizer.json
        # SPM-style BPE: a merged token scores minus its merge rank, so the SPM tokenizer merges in the BPE order
        with open(self.dir_model / "tokenizer.json", encoding="utf-8") as f:
            tok = json.load(f)
        with open(self.dir_model / "tokenizer_config.json", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        model = tok["model"]
        if model["type"] != "BPE" or not model.get("byte_fallback"):
            raise NotImplementedError("VoxCPM2: unexpected tokenizer type")

        vocab: dict[str, int] = model["vocab"]
        vocab_size = self.hparams["vocab_size"]
        tokens = [f"[PAD{i}]" for i in range(vocab_size)]
        scores = [-1e9] * vocab_size
        toktypes = [gguf.TokenType.UNUSED] * vocab_size
        for text, i in vocab.items():
            tokens[i] = text
            toktypes[i] = gguf.TokenType.BYTE if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", text) else gguf.TokenType.NORMAL
        for rank, merge in enumerate(model["merges"]):
            a, b = merge.split(" ") if isinstance(merge, str) else merge
            i = vocab[a + b]
            scores[i] = max(scores[i], -float(rank))

        # the special tokens (e.g. <|audio_start|>) are only listed in tokenizer_config.json
        specials = {int(i): t["content"] for i, t in tok_cfg.get("added_tokens_decoder", {}).items() if t.get("special")}
        specials.update({t["id"]: t["content"] for t in tok["added_tokens"] if t.get("special")})
        for i, text in specials.items():
            tokens[i] = text
            toktypes[i] = gguf.TokenType.CONTROL

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)


@ModelBase.register("VoxCPM2Model")
@ModelBase.example("openbmb/VoxCPM2")
class VoxCPM2MmprojModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = True  # AudioVAE encoder, for voice cloning

    _LAYER_MAP = {
        "input_layernorm.weight":          "ln1",
        "self_attn.q_proj.weight":         "attn_q",
        "self_attn.k_proj.weight":         "attn_k",
        "self_attn.v_proj.weight":         "attn_v",
        "self_attn.o_proj.weight":         "attn_out",
        "post_attention_layernorm.weight": "ln2",
        "mlp.gate_proj.weight":            "ffn_gate",
        "mlp.up_proj.weight":              "ffn_up",
        "mlp.down_proj.weight":            "ffn_down",
    }

    # libllama names, mtmd loads the residual LM as a text model
    _RES_LAYER_MAP = {
        "input_layernorm.weight":          "attn_norm",
        "self_attn.q_proj.weight":         "attn_q",
        "self_attn.k_proj.weight":         "attn_k",
        "self_attn.v_proj.weight":         "attn_v",
        "self_attn.o_proj.weight":         "attn_output",
        "post_attention_layernorm.weight": "ffn_norm",
        "mlp.gate_proj.weight":            "ffn_gate",
        "mlp.up_proj.weight":              "ffn_up",
        "mlp.down_proj.weight":            "ffn_down",
    }

    _PROJ_MAP = {
        "fsq_layer.in_proj":  "a.gen.fsq.in",
        "fsq_layer.out_proj": "a.gen.fsq.out",
        "enc_to_lm_proj":     "a.gen.enc_to_lm",
        "lm_to_dit_proj":     "a.gen.lm_to_dit",
        "res_to_dit_proj":    "a.gen.res_to_dit",
        "fusion_concat_proj": "a.gen.fusion",
        "stop_proj":          "a.gen.stop_proj",
        "stop_head":          "a.gen.stop_head",
        "feat_encoder.in_proj":             "a.gen.enc.in_proj",
        "feat_decoder.estimator.in_proj":   "a.gen.dit.in_proj",
        "feat_decoder.estimator.cond_proj": "a.gen.dit.cond_proj",
        "feat_decoder.estimator.out_proj":  "a.gen.dit.out_proj",
        "feat_decoder.estimator.time_mlp.linear_1": "a.gen.dit.time_mlp.0",
        "feat_decoder.estimator.time_mlp.linear_2": "a.gen.dit.time_mlp.1",
    }

    def set_gguf_parameters(self):
        lm  = self.global_config["lm_config"]
        dit = self.global_config["dit_config"]
        vae = self.global_config["audio_vae_config"]
        self.gguf_writer.add_file_type(self.ftype)

        # LocEnc + LocDiT + residual LM + AudioVAE decoder, LocEnc has the hparams of LocDiT
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.VOXCPM2_GEN)
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(dit["hidden_dim"])
        self.gguf_writer.add_gen_audio_feed_forward_length(dit["ffn_dim"])
        self.gguf_writer.add_gen_audio_block_count(dit["num_layers"])
        self.gguf_writer.add_gen_audio_head_count(dit["num_heads"])
        self.gguf_writer.add_gen_audio_head_count_kv(lm["num_key_value_heads"])
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(lm["rms_norm_eps"])

        # AudioVAE encoder: raw waveform -> latents, LocEnc in the gen model then makes the backbone rows
        self.gguf_writer.add_clip_has_audio_encoder(True)
        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.VOXCPM2_SPKENC)
        self.gguf_writer.add_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_audio_num_mel_bins(1)
        self.gguf_writer.add_audio_block_count(0)
        self.gguf_writer.add_audio_embedding_length(vae["encoder_dim"])
        self.gguf_writer.add_audio_feed_forward_length(vae["encoder_dim"])
        self.gguf_writer.add_audio_head_count(1)
        self.gguf_writer.add_audio_attention_layernorm_eps(lm["rms_norm_eps"])
        # the ODE solver, the RoPE factors and the VAE layout are hard-coded in tools/mtmd

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # the AudioVAE convs and the 1D tensors stay F32
        if new_name.startswith(("a.vae.", "a.gen.vae.")) or n_dims < 2:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if item[0].startswith("base_lm."):
            return None
        return item

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # already named by generate_extra_tensors()
        if name.startswith("a."):
            yield (name, data_torch)
            return

        # dt is always 0 (mean_mode = false), delta_time_mlp gives a constant that is folded into the time_mlp bias
        if name.startswith("feat_decoder.estimator.delta_time_mlp."):
            return
        if name == "feat_decoder.estimator.time_mlp.linear_2.bias":
            data_torch = LazyTorchTensor.to_eager(data_torch).float() + self._dtime_const()

        if name == "feat_encoder.special_token":
            yield ("a.gen.enc.special_token", data_torch.reshape(-1))
            return

        for src, dst in (("residual_lm.", "a.gen.res"), ("feat_encoder.encoder.", "a.gen.enc"),
                         ("feat_decoder.estimator.decoder.", "a.gen.code")):
            if not name.startswith(src):
                continue
            key = name.removeprefix(src)
            if key == "norm.weight":
                yield (f"{dst}.output_norm.weight" if dst == "a.gen.res" else f"{dst}.post_ln.weight", data_torch)
                return
            assert bid is not None, name
            key = key.removeprefix(f"layers.{bid}.")
            layer_map = self._RES_LAYER_MAP if dst == "a.gen.res" else self._LAYER_MAP
            yield (f"{dst}.blk.{bid}.{layer_map[key]}.weight", data_torch)
            return

        base, _, suffix = name.rpartition(".")
        yield (f"{self._PROJ_MAP[base]}.{suffix}", data_torch)

    # delta_time_mlp(SinusoidalPosEmb(0)), see VoxCPMLocDiTV2.forward()
    def _dtime_const(self) -> Tensor:
        d = {k: LazyTorchTensor.to_eager(self.model_tensors[f"feat_decoder.estimator.delta_time_mlp.{k}"]()).float()
             for k in ("linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias")}
        dim = d["linear_1.weight"].shape[1]
        emb = torch.cat([torch.zeros(dim // 2), torch.ones(dim // 2)])  # sin(0), cos(0)
        h = torch.nn.functional.silu(d["linear_1.weight"] @ emb + d["linear_1.bias"])
        return d["linear_2.weight"] @ h + d["linear_2.bias"]

    @staticmethod
    def _wn(sd: dict[str, Tensor], prefix: str) -> Tensor:
        # torch weight norm, over every dim but 0
        g, v = sd[prefix + ".weight_g"].float(), sd[prefix + ".weight_v"].float()
        return g * v / v.flatten(1).norm(dim=1).view(-1, *[1] * (v.dim() - 1))

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # LongRoPE factors of LocEnc and LocDiT, picked as in MiniCPMLongRoPE._set_cos_sin_cache()
        lm = self.global_config["lm_config"]
        rope = lm["rope_scaling"]
        if rope.get("type") != "longrope":
            raise NotImplementedError(f"VoxCPM2: unexpected rope_scaling {rope.get('type')}")
        if lm["max_position_embeddings"] > rope["original_max_position_embeddings"]:
            raise NotImplementedError("VoxCPM2: the long RoPE factors also scale cos/sin, not supported")
        yield ("a.gen.rope_factors", torch.tensor(rope["short_factor"], dtype=torch.float32))

        path = self.dir_model / "audiovae.pth"
        logger.info(f"gguf: loading AudioVAE from '{path}'")
        sd = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
        sd = sd.get("state_dict", sd)
        vae = self.global_config["audio_vae_config"]

        def conv(dst: str, src: str) -> Iterable[tuple[str, Tensor]]:
            yield (f"{dst}.weight", self._wn(sd, src))
            yield (f"{dst}.bias", sd[src + ".bias"].float())

        def snake(dst: str, src: str) -> Iterable[tuple[str, Tensor]]:
            alpha, beta = fold_snake(sd[src + ".alpha"])
            yield (f"{dst}.alpha", alpha)
            yield (f"{dst}.beta", beta)

        def res_units(dst: str, src: str, first: int) -> Iterable[tuple[str, Tensor]]:
            # dilation 1, 3, 9: Snake, depthwise conv k = 7, Snake, conv k = 1
            for r in range(3):
                s = f"{src}.{first + r}.block"
                d = f"{dst}.res.{r}"
                yield from snake(f"{d}.snake1", f"{s}.0")
                yield from conv(f"{d}.conv1", f"{s}.1")
                yield from snake(f"{d}.snake2", f"{s}.2")
                yield from conv(f"{d}.conv2", f"{s}.3")

        # encoder: conv, then per stride (3 residual units, Snake, strided conv), then fc_mu
        p = "a.vae.enc"
        yield from conv(f"{p}.conv_in", "encoder.block.0")
        for b in range(len(vae["encoder_rates"])):
            s = f"encoder.block.{b + 1}.block"
            yield from res_units(f"{p}.blk.{b}", s, 0)
            yield from snake(f"{p}.blk.{b}.snake", f"{s}.3")
            yield from conv(f"{p}.blk.{b}.down", f"{s}.4")
        yield from conv(f"{p}.fc_mu", "encoder.fc_mu")

        # decoder: depthwise conv, conv k = 1, then per stride (sample-rate scale and bias, Snake, transposed conv, 3 residual units), then Snake, conv
        bounds = vae["sr_bin_boundaries"]
        sr_idx = sum(1 for b in bounds if b < VAE_OUT_SAMPLE_RATE)  # torch.bucketize
        if vae["out_sample_rate"] != VAE_OUT_SAMPLE_RATE:
            raise NotImplementedError(f"VoxCPM2: unexpected out_sample_rate {vae['out_sample_rate']}")
        p = "a.gen.vae"
        yield from conv(f"{p}.conv_in_dw", "decoder.model.0")
        yield from conv(f"{p}.conv_in", "decoder.model.1")
        n_up = len(vae["decoder_rates"])
        for b in range(n_up):
            c = f"decoder.sr_cond_model.{b + 2}"
            yield (f"{p}.blk.{b}.sr_scale", sd[c + ".scale_embed.weight"][sr_idx].float())
            yield (f"{p}.blk.{b}.sr_bias",  sd[c + ".bias_embed.weight"][sr_idx].float())
            s = f"decoder.model.{b + 2}.block"
            yield from snake(f"{p}.blk.{b}.snake", f"{s}.0")
            yield from conv(f"{p}.blk.{b}.up", f"{s}.1")
            yield from res_units(f"{p}.blk.{b}", s, 2)
        yield from snake(f"{p}.post_snake", f"decoder.model.{n_up + 2}")
        yield from conv(f"{p}.conv_out", f"decoder.model.{n_up + 3}")

        assert math.prod(vae["encoder_rates"]) * VAE_OUT_SAMPLE_RATE == math.prod(vae["decoder_rates"]) * vae["sample_rate"]
