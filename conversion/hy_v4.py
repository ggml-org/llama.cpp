from __future__ import annotations

import re
from typing import Callable, Iterable

import torch

from .base import MmprojModel, ModelBase, gguf, logger
from .deepseek import DeepseekV2Model


def split_gate_up(weight: torch.Tensor, moe_intermediate_size: int):
    """Split a fused stacked gate_up expert tensor into (gate, up).

    weight: [n_expert, 2*moe_intermediate_size, hidden] (gate first, up second).
    Returns (gate, up) each [n_expert, moe_intermediate_size, hidden].
    """
    assert weight.shape[1] == 2 * moe_intermediate_size, f"{weight.shape[1]} != 2*{moe_intermediate_size}"
    gate = weight[:, :moe_intermediate_size, :].contiguous()
    up = weight[:, moe_intermediate_size:, :].contiguous()
    return gate, up


def rope_rows_to_neox(weight: torch.Tensor, head_dim: int, rope_dim: int) -> torch.Tensor:
    """Reorder the rope rows of each head from interleaved pairs to NeoX halves.

    The rope rows are the last rope_dim rows of each head_dim block. The M-RoPE kernels only support NeoX ordering, so this lets the graph use them for the same rotation.
    """
    shape = weight.shape
    w = weight.reshape(-1, head_dim, *shape[1:])
    nope, rope = w[:, :head_dim - rope_dim], w[:, head_dim - rope_dim:]
    rope = torch.cat((rope[:, 0::2], rope[:, 1::2]), dim=1)
    return torch.cat((nope, rope), dim=1).reshape(shape)


@ModelBase.register("HYV4ForCausalLM")
@ModelBase.example("tencent/Hy4-preview")
class HYV4Model(DeepseekV2Model):
    """HY_V4: DeepSeek-V3 style MLA + MoE with iHC, a gated MLA output and a learnable sink.

    Reuses DeepseekV2Model for the vocab and the MLA metadata, but overrides the tensor mapping
    because HY_V4 ships pre-stacked / fused experts plus extra iHC, gate and sink tensors. The
    rope rows are mapped straight through (no permute) - the graph rotates consecutive pairs.

    DSA is supported: indexer weights are exported for the layers marked "full" in indexer_types.
    "shared" layers reuse the top-k of the last preceding full layer at inference time, so they
    carry no indexer weights.

    MTP (num_nextn_predict_layers) is dropped, so the GGUF cannot be used for speculative
    decoding. The reference only runs the MTP layers while training or while speculating, so they
    cannot change single-token logits.
    """

    model_arch = gguf.MODEL_ARCH.HY_V4

    merge_expert = False

    # tensors a "full" indexer layer must carry
    INDEXER_SUFFIXES = frozenset({
        "self_attn.indexer.wq_b.weight",
        "self_attn.indexer.wk.weight",
        "self_attn.indexer.k_norm.weight",
        "self_attn.indexer.k_norm.bias",
        "self_attn.indexer.weights_proj.weight",
    })

    @classmethod
    def filter_tensors(cls, item):
        # drop MTP here, not in modify_tensors, so the weights are never read
        if item[0].startswith("model.mtp_layers."):
            return None
        return super().filter_tensors(item)

    def _check_indexer_hparams(self):
        for key in ("index_n_heads", "index_head_dim", "index_topk"):
            if key not in self.hparams:
                raise ValueError(f"HY_V4 has DSA layers but no {key}")

    def indexer_is_full(self) -> list[bool] | None:
        """Per-layer indexer ownership, or None when the checkpoint has no DSA.

        indexer_types entries are "full" (owns an indexer) or "shared" (reuses the preceding
        full layer's top-k). Missing indexer_types with sparse layers means every sparse layer
        owns one.
        """
        hparams = self.hparams
        n_layer = hparams["num_hidden_layers"]
        indexer_types = hparams.get("indexer_types")

        # the reference drives DSA off indexer_types alone; layer_types is only a fallback for
        # checkpoints predating it (it was renamed to deepseek_sparse_attention upstream)
        if indexer_types is None:
            layer_types = hparams.get("layer_types") or []
            sparse = {"sparse_attention", "deepseek_sparse_attention"}
            if not any(t in sparse for t in layer_types):
                return None
            if len(layer_types) < n_layer:
                raise ValueError(f"HY_V4 layer_types has {len(layer_types)} entries, need {n_layer}")
            self._check_indexer_hparams()
            return [t in sparse for t in layer_types[:n_layer]]

        self._check_indexer_hparams()

        if len(indexer_types) < n_layer:
            raise ValueError(f"HY_V4 indexer_types has {len(indexer_types)} entries, need {n_layer}")
        unknown = {t for t in indexer_types[:n_layer]} - {"full", "shared"}
        if unknown:
            raise ValueError(f"HY_V4 unknown indexer_types values: {sorted(unknown)}")
        is_full = [t == "full" for t in indexer_types[:n_layer]]
        if is_full and not is_full[0]:
            raise ValueError("HY_V4 layer 0 must be indexer_types 'full' (nothing precedes it to share)")
        return is_full

    def set_gguf_parameters(self):
        hparams = self.hparams

        # rope sections mean "rope rows are NeoX ordered" to the graph, only the VL model writes them
        self.rope_parameters = {k: v for k, v in self.rope_parameters.items() if k != "mrope_section"}

        # HY4 has n_group == topk_group == 1 (no group routing). Drop the keys so the base does
        # not emit expert_group_count/used; llama.cpp then takes the ungrouped MoE path.
        if hparams.get("n_group") == 1 and hparams.get("topk_group") == 1:
            hparams.pop("n_group", None)
            hparams.pop("topk_group", None)

        # HY_V4 config expresses dense/sparse layers via mlp_layer_types, but DeepseekV2Model
        # needs first_k_dense_replace. Derive it as the contiguous leading "dense" block
        # (the real config.json also carries first_k_dense_replace; prefer it when present,
        # but assert the two agree so a mismatch fails loudly).
        mlp_types = hparams.get("mlp_layer_types")
        explicit = hparams.get("first_k_dense_replace")
        derived = None
        if mlp_types is not None:
            lead = 0
            for t in mlp_types:
                if t == "dense":
                    lead += 1
                else:
                    break
            if any(t == "dense" for t in mlp_types[lead:]):
                raise NotImplementedError("HY_V4 converter expects a contiguous leading dense block")
            derived = lead
        if explicit is not None and derived is not None and explicit != derived:
            raise ValueError(
                f"HY_V4 first_k_dense_replace ({explicit}) disagrees with mlp_layer_types "
                f"leading-dense count ({derived})"
            )
        if explicit is None:
            if derived is None:
                raise ValueError("HY_V4 needs first_k_dense_replace or mlp_layer_types to place dense layers")
            hparams["first_k_dense_replace"] = derived

        # reuse DeepseekV2 MLA + MoE metadata (forces num_key_value_heads=1, writes q/kv lora,
        # key/value lengths, expert counts, weights scale/norm, rope dims, etc.)
        super().set_gguf_parameters()

        # HY4 uses DeepSeek-V3 sigmoid routing with e_score_correction_bias. The config has no
        # scoring_func key, so the base does not write a gating func; set it explicitly.
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        # routed-expert SwiGLU logits clamp (only routed experts; shared/dense are not clamped,
        # so swiglu_clamp_shexp is intentionally not written). 0.0 disables the clamp.
        swiglu_limit = float(hparams.get("swiglu_limit", 0.0) or 0.0)
        if swiglu_limit > 0.0:
            self.gguf_writer.add_swiglu_clamp_exp([swiglu_limit] * self.block_count)

        # iHC (independent Hyper-Connections)
        self.gguf_writer.add_hyper_connection_count(hparams["hc_mult"])
        self.gguf_writer.add_hyper_connection_epsilon(hparams["hc_eps"])
        self.gguf_writer.add_hyper_connection_magnitude(hparams["hc_magnitude"])

        # is_full is written explicitly; the graph must not infer it from tensor presence
        is_full = self.indexer_is_full()
        if is_full is not None:
            self.gguf_writer.add_indexer_head_count(hparams["index_n_heads"])
            self.gguf_writer.add_indexer_key_length(hparams["index_head_dim"])
            self.gguf_writer.add_indexer_top_k(hparams["index_topk"])
            # the loader reads this key over n_layer(), which excludes the draft block
            self.gguf_writer.add_indexer_types(is_full[:hparams["num_hidden_layers"]])
            logger.info(
                "HY_V4 DSA: %d/%d layers own an indexer (top_k=%d, n_heads=%d, head_dim=%d)",
                sum(is_full), len(is_full), hparams["index_topk"],
                hparams["index_n_heads"], hparams["index_head_dim"],
            )

        if hparams.get("num_nextn_predict_layers", 0):
            logger.warning(
                "HY_V4: dropping %d MTP (nextn) layer(s) - the reference runs them only under "
                "training / speculative decoding. This GGUF cannot be used for speculative decoding.",
                hparams["num_nextn_predict_layers"],
            )

    def prepare_tensors(self):
        # Hy4-preview for some reason has num_key_value_heads equal to 8, so override it here
        # without this conversion/deepseek.py fails on assert
        self.hparams["num_key_value_heads"] = self.hparams["num_attention_heads"]

        # validate before the base materializes tensors, so a mismatch fails early
        is_full = self.indexer_is_full()
        if is_full is not None:
            present: dict[int, set[str]] = {}
            for name in self.model_tensors:
                m = re.match(r"model\.layers\.(\d+)\.(self_attn\.indexer\..+)$", name)
                if m:
                    present.setdefault(int(m.group(1)), set()).add(m.group(2))
            for il, expect_full in enumerate(is_full):
                seen = present.get(il, set())
                if expect_full and seen != self.INDEXER_SUFFIXES:
                    raise ValueError(
                        f"HY_V4 layer {il} is indexer_types 'full' but is missing indexer tensors: "
                        f"{sorted(self.INDEXER_SUFFIXES - seen)}"
                    )
                if not expect_full and seen:
                    raise ValueError(
                        f"HY_V4 layer {il} is indexer_types 'shared' but carries indexer tensors: "
                        f"{sorted(seen)}"
                    )

        super().prepare_tensors()

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # iHC mixing matrices are 2D .weight tensors that the reference keeps in fp32
        # (_keep_in_fp32_modules_strict). 1D tensors (hc_base/scale, attn_sinks,
        # e_score_correction_bias) and the router (FFN_GATE_INP) are already forced F32 by the
        # base rules. Force the HC *_fn matrices here.
        if new_name.endswith(("hc_attn_fn.weight", "hc_ffn_fn.weight", "output_hc_fn.weight")):
            return gguf.GGMLQuantizationType.F32
        # indexer k_norm is fp32 in the reference; the base rules already cover
        # *_norm.weight and INDEXER_PROJ, but not this bias
        if self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.INDEXER_K_NORM, bid, suffix=".bias"):
            return gguf.GGMLQuantizationType.F32
        # enable_lm_head_fp32: mirror the reference fp32 LM-head matmul by keeping output F32.
        if new_name == "output.weight" and self.hparams.get("enable_lm_head_fp32", False):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        hparams = self.hparams
        moe_inter = hparams["moe_intermediate_size"]

        tn = self.format_tensor_name

        # fused stacked experts: split gate_up into gate/up
        if name.endswith("mlp.experts.gate_up_proj"):
            gate, up = split_gate_up(data_torch, moe_inter)
            yield from super().modify_tensors(gate, tn(gguf.MODEL_TENSOR.FFN_GATE_EXP, bid), bid)
            yield from super().modify_tensors(up,   tn(gguf.MODEL_TENSOR.FFN_UP_EXP,   bid), bid)
            return

        # add .weight suffixes
        if name.endswith("mlp.experts.down_proj") or name.endswith(".self_attn.learnable_sink_param"):
            name += ".weight"

        if re.search(r"\.hc_head\.hc_head_(?:fn|base|scale)$", name):
            name += ".weight"

        if re.search(r"\.hc_(?:attn|mlp)_layer\.hc_pre\.hc_(?:fn|base|scale)$", name):
            name += ".weight"

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("HYV4VLForConditionalGeneration")
class HYV4VLTextModel(HYV4Model):
    """Text tower of the HY_V4 VL model.

    The LLM uses interleaved M-RoPE over GPT-J style pairs. The rope rows are reordered to NeoX halves here, so the graph can run the IMROPE kernel. For text-only positions this is the same rotation as the NORM rope of HYV4Model.
    """

    model_arch = gguf.MODEL_ARCH.HY_V4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rope = self.rope_parameters
        if not rope.get("mrope_interleaved", False):
            raise NotImplementedError("HY_V4 VL expects mrope_interleaved = true")
        section = list(rope["mrope_section"])
        if len(section) == 4 and section[3] == 0:
            section = section[:3]
        self.mrope_section = section
        if len(self.mrope_section) != 3 or 2 * sum(self.mrope_section) != self.hparams["qk_rope_head_dim"]:
            raise ValueError(f"HY_V4 VL mrope_section {self.mrope_section} must sum to qk_rope_head_dim / 2")

    @classmethod
    def filter_tensors(cls, item):
        if item[0].startswith("vit."):
            return None
        return super().filter_tensors(item)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_dimension_sections(self.mrope_section + [0])

    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        hparams = self.hparams
        rope_dim = hparams["qk_rope_head_dim"]
        qk_head_dim = hparams["qk_nope_head_dim"] + rope_dim
        index_head_dim = hparams.get("index_head_dim", 0)

        if name.endswith(".self_attn.q_b_proj.weight"):
            data_torch = rope_rows_to_neox(data_torch, qk_head_dim, rope_dim)
        elif name.endswith(".self_attn.kv_a_proj_with_mqa.weight"):
            data_torch = rope_rows_to_neox(data_torch, hparams["kv_lora_rank"] + rope_dim, rope_dim)
        elif re.search(r"\.self_attn\.indexer\.(?:wq_b\.weight|wk\.weight|k_norm\.weight|k_norm\.bias)$", name):
            # k_norm sits between wk and the rope, so its affine params follow the same order
            data_torch = rope_rows_to_neox(data_torch, index_head_dim, rope_dim)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("HYV4VLForConditionalGeneration")
class HYV4VLVisionModel(MmprojModel):
    """Vision tower of the HY_V4 VL model: a SigLIP-shaped encoder with a rmsnorm_mlp_adapter projector.

    Positions come from a learned 128x128 grid, resampled to the patch grid. The projector merges 2x2 patches with a strided conv2d, runs a 1x1 conv2d and appends a newline embedding to every row.
    """

    projector_type = gguf.VisionProjectorType.HY_V4_VL

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        # there is no image_size, the position-embedding grid is max_image_size / patch_size
        self.hparams_vision["image_size"] = self.hparams_vision.get("max_image_size", 2048)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None
        vcfg = self.hparams_vision
        projector_type = vcfg.get("projector_type")
        if projector_type != "rmsnorm_mlp_adapter":
            raise ValueError(f"unsupported projector_type {projector_type!r}")
        # the graph implements the plain variant only
        unsupported = {
            "use_2d_rope": bool(vcfg.get("use_2d_rope", False)),
            "cat_extra_token": bool(vcfg.get("cat_extra_token", False)),
            "use_image_newline_token": not vcfg.get("use_image_newline_token", True),
            "perceive_post_norm": bool(vcfg.get("perceive_post_norm", False)),
            "hidden_act": vcfg.get("hidden_act", "gelu") != "gelu",
            "interpolate_mode": vcfg.get("interpolate_mode", "bilinear") != "bilinear",
        }
        for key, bad in unsupported.items():
            if bad:
                raise NotImplementedError(f"HY_V4 VL vision config {key}={vcfg.get(key)!r} is not supported")
        self.gguf_writer.add_clip_projector_type(self.projector_type)
        self.gguf_writer.add_vision_use_gelu(True)
        # rms_norm_eps is also the LayerNorm eps of the encoder blocks
        self.gguf_writer.add_vision_attention_layernorm_eps(vcfg.get("rms_norm_eps", 1e-5))
        self.gguf_writer.add_vision_spatial_merge_size(vcfg.get("spatial_merge_size", 2))
        self.gguf_writer.add_vision_min_pixels(int(self.preprocessor_config["min_pixels"]))
        self.gguf_writer.add_vision_max_pixels(int(self.preprocessor_config["max_pixels"]))

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], torch.Tensor]]) -> tuple[str, Callable[[], torch.Tensor]] | None:
        if not item[0].startswith("vit."):
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        # row 0 of the position embedding is an unused CLS slot, the reference resamples rows 1:
        if "position_embedding" in name:
            data_torch = data_torch[1:]
        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # force conv weights to F32 or F16 to avoid BF16 IM2COL issues on Metal
        if ("mm.0." in new_name or "mm.2." in new_name) and new_name.endswith(".weight"):
            return gguf.GGMLQuantizationType.F16 if self.ftype == gguf.LlamaFileType.MOSTLY_F16 else gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)
