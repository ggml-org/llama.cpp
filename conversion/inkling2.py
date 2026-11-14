from __future__ import annotations

import json
import re
from pathlib import Path
from functools import partial
from typing import Callable, Iterable, TYPE_CHECKING, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, MmprojModel, TextModel, gguf, logger


def _e4m3_table() -> np.ndarray:
    codes = np.arange(256, dtype=np.uint8)
    exponent = ((codes & 0x7F) >> 3).astype(np.int32)
    mantissa = (codes & 7).astype(np.float32)
    values = np.where(exponent == 0, np.ldexp(mantissa, -9), np.ldexp(1 + mantissa / 8, exponent - 7))
    values = np.where(codes & 0x80, -values, values)
    values[(codes & 0x7F) == 0x7F] = np.nan
    return values.astype(np.float32)


_E4M3 = _e4m3_table()
_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
_EXPERT_CHUNK_ROWS = 128


def _dequant_expert_chunk(weight: Callable[[], Tensor], scale: Callable[[], Tensor], amax: Callable[[], Tensor] | None, denominator: Callable[[], Tensor] | None, name: str, expert: int, rows: slice) -> np.ndarray:
    # Fresh lazy tensors release their mappings after each chunk instead of caching the layer.
    packed = LazyTorchTensor.to_eager(weight().view(torch.uint8)[expert, rows]).numpy()
    scales = LazyTorchTensor.to_eager(scale().view(torch.uint8)[expert, rows]).numpy()
    if amax is not None:
        block_scale = _E4M3[scales]
        if np.isnan(block_scale).any():
            raise ValueError(f"Inkling2: {name}_scale expert {expert}: E4M3FN NaN")
        values = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
        values[:, 0::2] = _E2M1[packed & 15]
        values[:, 1::2] = _E2M1[packed >> 4]
        values.reshape(values.shape[0], -1, 16)[:] *= block_scale[:, :, None]
        assert denominator is not None
        a = LazyTorchTensor.to_eager(amax()[expert]).item()
        d = LazyTorchTensor.to_eager(denominator()).item()
        if not np.isfinite(a) or a < 0:
            raise ValueError(f"Inkling2: {name}_amax expert {expert}: expected finite nonnegative value")
        if not np.isfinite(d) or d <= 0:
            raise ValueError(f"Inkling2: {name}_outer_scale_denominator expert {expert}: expected finite positive value")
        with np.errstate(over="ignore", invalid="ignore"):
            values *= np.float32(a) / np.float32(d)
    else:
        if np.any(scales == 255):
            raise ValueError(f"Inkling2: {name}_scale expert {expert}: UE8M0 byte 255 is NaN")
        values = _E4M3[packed]
        if np.isnan(values).any():
            raise ValueError(f"Inkling2: {name} expert {expert}: E4M3FN NaN")
        block_scale = np.ldexp(np.ones(scales.shape, dtype=np.float32), scales.astype(np.int32) - 127)
        with np.errstate(over="ignore", invalid="ignore"):
            values.reshape(values.shape[0], -1, 32)[:] *= block_scale[:, :, None]
    if not np.isfinite(values).all():
        raise ValueError(f"Inkling2: {name} expert {expert}: non-finite dequantized weight")
    return values


def _is_inkling2(dir_model: Path) -> bool:
    if not (dir_model / "config.json").is_file():
        return False
    with open(dir_model / "config.json", encoding="utf-8") as f:
        return json.load(f).get("model_type") in ("inkling2", "inkling2_text")


@ModelBase.register_hparams_loader(_is_inkling2)
def _load_inkling2_hparams(dir_model: Path):
    # AutoConfig injects MXFP8 defaults and can replace the original layer schedule.
    with open(dir_model / "config.json", encoding="utf-8") as f:
        return json.load(f)


def _set_inkling2_vocab(model: TextModel):
    tokens, types, pre = model.get_vocab_base()
    model.gguf_writer.add_tokenizer_model("gpt2")
    model.gguf_writer.add_tokenizer_pre(pre)
    model.gguf_writer.add_token_list(tokens)
    model.gguf_writer.add_token_types(types)
    special = gguf.SpecialVocab(model.dir_model, load_merges=True)
    eos = model.hparams.get("eos_token_id", 200006)
    special.special_token_ids["eos"] = eos
    special.special_token_ids["eot"] = eos
    special.add_special_token["bos"] = False
    special.add_to_gguf(model.gguf_writer)


@ModelBase.register("Inkling2ForConditionalGeneration", "Inkling2ForCausalLM")
@ModelBase.example("thinkingmachines/Inkling-2-Small")
class Inkling2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.INKLING2

    def __init__(self, dir_model: Path, *args, **kwargs):
        self._expert_quant: dict[str, dict] = {}
        self._packed_experts: dict[str, gguf.LazyChunkedTensor] = {}
        quant_path = dir_model / "hf_quant_config.json"
        if quant_path.is_file():
            with open(quant_path, encoding="utf-8") as f:
                groups = json.load(f).get("model", {}).get("parameter_groups", [])
            for group in groups:
                for name in group.get("tensors", []):
                    item = self.filter_tensors((name, lambda: torch.empty(0)))
                    if item is not None and item[0].endswith((".w13_weight", ".w2_weight")):
                        name = item[0]
                        if name in self._expert_quant:
                            raise ValueError(f"Inkling2: duplicate hf_quant_config.json scheme for {name}")
                        self._expert_quant[name] = group
        super().__init__(dir_model, *args, **kwargs)
        logger.warning("Inkling2: converting text only; use --mmproj for vision and --dspark for mtp.safetensors")
        hp = self.hparams
        for name, default in (("q_bias", False), ("o_bias", False), ("n_shared_experts", 0), ("use_value_norm_no_gamma", True), ("attn_res_normalize_values", True), ("scoring_func", "sigmoid")):
            if hp.get(name, default) != default:
                raise ValueError(f"Inkling2: unsupported {name}={hp[name]!r}")
        types = hp.get("layer_types")
        if types is None:
            local = hp.get("local_layer_ids", [i for i in range(self.block_count) if (i + 1) % 6])
            types = ["hybrid_sliding" if i in local else "hybrid" for i in range(self.block_count)]
        if len(types) != self.block_count or set(types) - {"hybrid", "hybrid_sliding"}:
            raise ValueError("Inkling2: invalid layer_types")
        self.is_swa = [t == "hybrid_sliding" for t in types]

    def dequant_model(self):
        for name in list(self.model_tensors):
            if not name.endswith((".w13_weight", ".w2_weight")):
                continue
            weight = self.model_tensors[name]
            tensor = weight()
            companions = [name + suffix for suffix in ("_scale", "_amax", "_outer_scale_denominator")]
            if tensor.dtype.is_floating_point and tensor.element_size() > 1:
                if any(n in self.model_tensors for n in companions):
                    raise ValueError(f"Inkling2: floating-point {name} has packed quantization companions")
                continue
            config = self._expert_quant.get(name, {})
            scheme = config.get("weight_transform")
            if scheme not in ("nvfp4_nearest", "mxfp8_nearest"):
                raise ValueError(f"Inkling2: {name}: missing or unsupported hf_quant_config.json scheme {scheme!r}")
            nvfp4 = scheme == "nvfp4_nearest"
            if nvfp4 and config.get("weight_scale_format") != "amax_denominator":
                raise ValueError(f"Inkling2: {name}: expected NVFP4 amax_denominator scale format")
            required = companions if nvfp4 else companions[:1]
            for companion in required:
                if companion not in self.model_tensors:
                    raise ValueError(f"Inkling2: {name}: missing companion {companion}")
            if not nvfp4 and any(n in self.model_tensors for n in companions[1:]):
                raise ValueError(f"Inkling2: {name}: MXFP8 must not have amax or outer_scale_denominator")
            scale = self.model_tensors[companions[0]]
            scale_tensor = scale()
            weight_dtype, scale_dtype = (torch.uint8, torch.float8_e4m3fn) if nvfp4 else (torch.float8_e4m3fn, torch.uint8)
            if tensor.dtype != weight_dtype or scale_tensor.dtype != scale_dtype:
                raise ValueError(f"Inkling2: {name}: expected {weight_dtype} weights and {scale_dtype} scales, got {tensor.dtype} and {scale_tensor.dtype}")
            experts = self.hparams.get("n_routed_experts", 384)
            hidden = self.hparams["hidden_size"]
            inter = self.hparams.get("moe_intermediate_size", self.hparams.get("intermediate_size", 2304))
            w13 = name.endswith(".w13_weight")
            rows, cols = (2 * inter, hidden) if w13 else (hidden, inter)
            shape = (experts, rows, cols)
            if min(shape) <= 0 or cols % (16 if nvfp4 else 32) or tuple(tensor.shape) != (experts, rows, cols // (2 if nvfp4 else 1)):
                raise ValueError(f"Inkling2: {name}: invalid packed shape {tuple(tensor.shape)} for logical shape {shape}")
            ss = tuple(scale_tensor.shape)
            if len(ss) != 3 or ss[0] != experts or min(ss) <= 0 or rows % ss[1] or cols % ss[2]:
                raise ValueError(f"Inkling2: {name}_scale: shape {ss} does not divide logical shape {shape}")
            block = (rows // ss[1], cols // ss[2])
            if block != (1, 16 if nvfp4 else 32):
                raise ValueError(f"Inkling2: {name}_scale: unsupported stored block shape {block}")
            if list(block) != config.get("weight_block_shape"):
                logger.warning("Inkling2: %s: hf_quant_config.json weight_block_shape %s conflicts with stored weight/scale shapes %s/%s; using %s", name, config.get("weight_block_shape"), tuple(tensor.shape), ss, list(block))
            amax = denominator = None
            if nvfp4:
                amax, denominator = (self.model_tensors[n] for n in companions[1:])
                for companion, gen, expected in ((companions[1], amax, (experts,)), (companions[2], denominator, (1,))):
                    t = gen()
                    if t.dtype != torch.float32 or tuple(t.shape) != expected:
                        raise ValueError(f"Inkling2: {companion}: expected F32 {expected}, got {t.dtype} {tuple(t.shape)}")
            chunks: list[Callable[[], np.ndarray]] = []
            for expert in range(experts):
                for half in range(2 if w13 else 1):
                    for start in range(0, rows // 2 if w13 else rows, _EXPERT_CHUNK_ROWS):
                        stop = min(start + _EXPERT_CHUNK_ROWS, rows // 2 if w13 else rows)
                        # The same packed row slice selects the weight and its scale groups.
                        row_slice = slice(2 * start + half, 2 * stop + half, 2) if w13 else slice(start, stop)
                        chunks.append(partial(_dequant_expert_chunk, weight, scale, amax, denominator, name, expert, row_slice))
            self._packed_experts[name] = gguf.LazyChunkedTensor(chunks, shape, np.float32)
            self.model_tensors[name] = lambda shape=shape: torch.empty(shape, dtype=torch.float32, device="meta")
            for companion in required:
                del self.model_tensors[companion]
        for name, gen in self.model_tensors.items():
            t = gen()
            if not t.dtype.is_floating_point or t.element_size() == 1 or name.endswith(("_weight_scale", "_weight_amax", "_weight_outer_scale_denominator")):
                raise ValueError(f"Inkling2: unexpected packed tensor or orphan companion {name}")

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if any(p in name for p in ("visual.", "vision_tower.", "mtp.", "markov_head", "nextn_predict")):
            return None
        name = name.replace("model.language_model.", "model.").replace("model.llm.", "model.")
        name = name.replace("model.unembed.weight", "lm_head.weight")
        name = name.replace("model.embed.weight", "model.embed_tokens.weight")
        name = name.replace("model.embed_norm.", "model.embed_tokens.embed_norm.")
        name = re.sub(r"attn_res\.attn_queries\.(\d+)", lambda m: f"layers.{int(m[1]) + 1}.attention_input.query", name)
        name = re.sub(r"attn_res\.mlp_queries\.(\d+)", r"layers.\1.mlp_input.query", name)
        name = name.replace("attn_res.output_query", "final_input.query")
        for old, new in (("wq_du", "q_proj"), ("wk_dv", "k_proj"), ("wv_dv", "v_proj"), ("wr_du", "r_proj"), ("wo_ud", "o_proj"), ("gate_attn", "gate_proj"), ("attn_sink_proj", "sink_proj")):
            name = name.replace(f".attn.{old}", f".self_attn.{new}")
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace("attn_sconv.", "post_attention_convolution.").replace("mlp_sconv.", "post_mlp_convolution.")
        name = re.sub(r"(convolution|sconv)\.weight$", r"\1.conv1d.weight", name)
        name = re.sub(r"(convolution|sconv)\.static_w$", r"\1.static_weight", name)
        name = re.sub(r"(convolution|sconv)\.weight_proj$", r"\1.weight_proj.weight", name)
        name = name.replace(".mlp_norm.", ".post_attention_layernorm.").replace(".attn_norm.", ".input_layernorm.")
        name = re.sub(r"\.(q|k|v)_input_norm\.", r".\1_input_layernorm.", name)
        name = name.replace("mlp.gate.bias", "mlp.gate.e_score_correction_bias")
        return name, gen

    def get_vocab_base(self):
        tokens, types, pre = super().get_vocab_base()
        n = self.hparams.get("unpadded_vocab_size", 200058)
        return tokens[:n], types[:n], pre

    def set_vocab(self):
        _set_inkling2_vocab(self)

    def set_gguf_parameters(self):
        hp = self.hparams
        hp["intermediate_size"] = hp.get("moe_intermediate_size", hp.get("intermediate_size", 2304))
        hp["num_attention_heads"] = [hp.get("swa_num_attention_heads", hp["num_attention_heads"]) if s else hp["num_attention_heads"] for s in self.is_swa]
        hp["num_key_value_heads"] = [hp.get("swa_num_key_value_heads", hp["num_key_value_heads"]) if s else hp["num_key_value_heads"] for s in self.is_swa]
        hp.setdefault("n_routed_experts", 384)
        hp.setdefault("num_experts_per_tok", 8)
        super().set_gguf_parameters()
        w = self.gguf_writer
        w.add_vocab_size(hp["vocab_size"])
        w.add_key_length_swa(hp.get("swa_head_dim", hp["head_dim"]))
        w.add_value_length_swa(hp.get("swa_head_dim", hp["head_dim"]))
        w.add_rope_dimension_count(0)
        w.add_sliding_window(hp.get("sliding_window_size", 1024))
        w.add_sliding_window_pattern(self.is_swa)
        w.add_shortconv_l_cache(hp.get("conv_kernel_size", hp.get("sconv_kernel_size", 4)))
        w.add_expert_weights_scale(hp.get("route_scale", 8.0))
        w.add_activation_situ_beta(hp.get("mlp_gate_softcap", 4.0))
        w.add_activation_situ_linear_beta(hp.get("mlp_up_softcap", 25.0))
        w.add_attn_res_block_size(hp.get("n_layers_per_attn_res_block", 6))
        w.add_final_logit_softcapping(hp.get("final_logit_softcapping", 30.0))
        for key, field, default in (
            ("ATTENTION_RELATIVE_LENGTH", "d_rel", 16),
            ("ATTENTION_RELATIVE_EXTENT", "rel_extent", 1024),
            ("ATTENTION_QK_CLIP", "qk_l2_clip_threshold", 22.627416997969522),
            ("ATTENTION_RELATIVE_CLIP", "rel_l2_clip_threshold", 8.0),
            ("ATTENTION_LOG_SCALE_FLOOR", "log_scaling_n_floor", 128000),
            ("ATTENTION_LOG_SCALE_ALPHA", "log_scaling_alpha", 0.1),
            ("VOCAB_SIZE_UNPADDED", "unpadded_vocab_size", 200058),
        ):
            value = hp.get(field, default)
            add = w.add_float32 if isinstance(default, float) else w.add_uint32
            add(getattr(gguf.Keys.LLM, key).format(arch="inkling2"), value)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if new_name.endswith(("_conv.weight", "_clip.weight", "_res_query.weight", "attn_rel_bias.weight")):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name in self._packed_experts:
            target = name.replace(".w13_weight", ".gate_up_proj").replace(".w2_weight", ".down_proj")
            yield self.map_tensor_name(target, try_suffixes=()) + ".weight", cast("Tensor", self._packed_experts[name])
            return
        if name.endswith(".w13_weight"):
            shape = data_torch.shape
            data_torch = data_torch.reshape(shape[0], shape[1] // 2, 2, shape[2]).transpose(1, 2).reshape(shape)
            name = name.replace(".w13_weight", ".gate_up_proj")
        name = name.replace(".w2_weight", ".down_proj")
        if name.endswith(".static_weight"):
            data_torch = data_torch.flip(0).T.contiguous()
            name = name.replace(".static_weight", ".conv1d.weight")
        elif name.endswith(".conv1d.weight"):
            data_torch = data_torch.squeeze(1) if data_torch.ndim == 3 else data_torch
        elif name.endswith(".rel_logits_proj.proj"):
            data_torch = data_torch.T.contiguous()
        if name.endswith((".query", ".scale", ".proj", ".gate_up_proj", ".down_proj", ".e_score_correction_bias")):
            new_name = self.map_tensor_name(name, try_suffixes=())
            suffix = ".bias" if name.endswith(".e_score_correction_bias") else ".weight"
            yield new_name + suffix, data_torch
        else:
            yield from super().modify_tensors(data_torch, name, bid)


class Inkling2DSparkModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DFLASH

    def __init__(self, dir_model: Path, *args, **kwargs):
        config = ModelBase.load_hparams(dir_model, False)
        text = config.get("text_config", config)
        mtp = config.get("mtp_config")
        if not mtp:
            raise ValueError("Inkling2 DSpark requires mtp_config")
        targets = mtp["target_layer_ids"]
        if (len(targets) < 2 or targets != sorted(set(targets)) or targets[0] < 0 or
                targets[-1] != text["num_hidden_layers"] or mtp["hidden_size"] != text["hidden_size"] or
                mtp["markov_head_type"] != "rnn_hidden" or mtp.get("q_bias") or mtp.get("o_bias")):
            raise ValueError("Unsupported Inkling2 DSpark configuration")
        if (mtp["block_size"] != 8 or not 0 < mtp["markov_rank"] <= mtp["hidden_size"] or
                mtp["num_attention_heads"] < 1 or mtp["num_key_value_heads"] < 1 or
                mtp["num_attention_heads"] % mtp["num_key_value_heads"] or mtp["head_dim"] < 2 or mtp["head_dim"] % 2 or
                mtp["sliding_window_size"] < 1 or
                any(i < 0 or i >= mtp["num_hidden_layers"] for i in mtp["local_layer_ids"])):
            raise ValueError("Invalid Inkling2 DSpark dimensions")
        hp = {**mtp, "architectures": ["Inkling2ForCausalLM"],
              "vocab_size": text["vocab_size"], "unpadded_vocab_size": text["unpadded_vocab_size"],
              "eos_token_id": text.get("eos_token_id", 200006),
              "final_logit_softcapping": text["final_logit_softcapping"]}
        super().__init__(dir_model, *args, hparams=hp, **kwargs)

    def index_tensors(self, remote_hf_model_id=None):
        if remote_hf_model_id is not None:
            raise ValueError("Inkling2 DSpark requires a local mtp.safetensors")
        if not (self.dir_model / "mtp.safetensors").is_file():
            raise ValueError("Inkling2 DSpark requires mtp.safetensors")
        return self.index_local_tensors("mtp")

    def get_vocab_base(self):
        tokens, types, pre = super().get_vocab_base()
        n = self.hparams["unpadded_vocab_size"]
        return tokens[:n], types[:n], pre

    def set_vocab(self):
        _set_inkling2_vocab(self)
        self.gguf_writer.add_mask_token_id(0)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp, w = self.hparams, self.gguf_writer
        w.add_vocab_size(hp["vocab_size"])
        w.add_uint32(gguf.Keys.LLM.VOCAB_SIZE_UNPADDED.format(arch="dflash"), hp["unpadded_vocab_size"])
        w.add_block_size(hp["block_size"])
        w.add_target_layers(hp["target_layer_ids"])
        w.add_uint32(gguf.Keys.LLM.DSPARK_MARKOV_RANK.format(arch="dflash"), hp["markov_rank"])
        w.add_sample_from_anchor(False)
        w.add_has_confidence_head(False)
        w.add_bool("dflash.attention.causal", False)
        w.add_sliding_window(hp["sliding_window_size"])
        w.add_sliding_window_pattern([i in hp["local_layer_ids"] for i in range(self.block_count)])
        w.add_rope_dimension_count(hp["head_dim"])
        w.add_final_logit_softcapping(hp["final_logit_softcapping"])

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if name == "fuse_weight":
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name in ("mask_embed", "fuse_weight"):
            yield self.map_tensor_name(name, try_suffixes=()) + ".weight", data_torch
        else:
            yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Inkling2ForConditionalGeneration")
class Inkling2VisionModel(MmprojModel):
    def get_vision_config(self):
        config = self.global_config.get("vision_config")
        if not config:
            return None
        config = dict(config)
        config["patch_size"] = config.get("native_patch_size", config.get("patch_size", 20))
        config["spatial_merge_size"] = config.get("adapter_merge_size", config.get("spatial_merge_size", 2))
        config["intermediate_size"] = config.get("intermediate_size") or int(config["hidden_size"] * config.get("mlp_ratio", 4.0))
        config["image_size"] = config["posemb_grid_size"] * config["patch_size"]
        if config.get("num_channels", 3) != 3 or config.get("hidden_act", "gelu") != "gelu":
            raise ValueError("Inkling2 vision requires RGB patches and exact GELU")
        return config

    def set_gguf_parameters(self):
        self.preprocessor_config.setdefault("image_mean", [0.5, 0.5, 0.5])
        self.preprocessor_config.setdefault("image_std", [0.5, 0.5, 0.5])
        super().set_gguf_parameters()
        config = self.hparams
        rope = config.get("rope_parameters") or {"rope_type": "axial", "rope_theta": 10000.0}
        if rope["rope_type"] != "axial":
            raise ValueError("Inkling2 vision requires axial RoPE")
        w = self.gguf_writer
        w.add_clip_vision_projector_type(gguf.VisionProjectorType.INKLING2)
        w.add_vision_attention_layernorm_eps(config["layer_norm_eps"])
        w.add_vision_spatial_merge_size(config["spatial_merge_size"])
        w.add_uint32(gguf.Keys.ClipVision.POSITION_GRID_SIZE, config["posemb_grid_size"])
        w.add_uint32(gguf.Keys.ClipVision.Projector.HIDDEN_SIZE, config["projection_hidden_dim"])
        w.add_bool(gguf.Keys.ClipVision.Projector.USE_BIAS, config.get("projector_bias", False))
        w.add_float32(gguf.Keys.ClipVision.ROPE_FREQ_BASE, float(rope["rope_theta"]))

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if not name.startswith(("model.visual.", "model.vision_tower.")):
            return None
        name = name.replace("model.visual.backbone.embeddings.", "model.vision_tower.embeddings.")
        name = name.replace("model.visual.backbone.ln_pre.", "model.vision_tower.pre_layernorm.")
        name = name.replace("model.visual.backbone.layers.", "model.vision_tower.layers.")
        name = name.replace("model.visual.proj_in.", "model.vision_tower.merger.linear_1.")
        name = name.replace("model.visual.proj_out.", "model.vision_tower.merger.linear_2.")
        for old, new in (("ln_1", "layernorm_before"), ("ln_2", "layernorm_after"), ("mlp.c_fc", "mlp.fc1"), ("mlp.c_proj", "mlp.fc2"), ("ls_1.gamma", "lambda_1"), ("ls_2.gamma", "lambda_2")):
            name = name.replace("." + old, "." + new)
        if name.endswith(("position_embedding", "lambda_1", "lambda_2")):
            name += ".weight"
        return name, gen

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        for index in (1, 2):
            if name.startswith(f"model.vision_tower.merger.linear_{index}."):
                suffix = name.rsplit(".", 1)[1]
                yield self.format_tensor_name(gguf.MODEL_TENSOR.V_MMPROJ, bid=index, suffix="." + suffix), data_torch
                return
        yield from super().modify_tensors(data_torch, name, bid)
