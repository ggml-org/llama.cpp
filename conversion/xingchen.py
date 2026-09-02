from __future__ import annotations

import re
from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .deepseek import DeepseekV2Model


@ModelBase.register("XingChen4ForCausalLM")
class XingChen4Model(DeepseekV2Model):
    """XingChen4: DeepSeek-V2/V3 backbone (MLA + MoE) + mHC residual mixing."""

    model_arch = gguf.MODEL_ARCH.XINGCHEN4

# map (prefix, kind) -> MODEL_TENSOR enum
    _hc_tensor_map = {
        ("hc_attn", "fn"):    gguf.MODEL_TENSOR.HC_ATTN_FN,
        ("hc_attn", "base"):  gguf.MODEL_TENSOR.HC_ATTN_BASE,
        ("hc_attn", "scale"): gguf.MODEL_TENSOR.HC_ATTN_SCALE,
        ("hc_ffn",  "fn"):    gguf.MODEL_TENSOR.HC_FFN_FN,
        ("hc_ffn",  "base"):  gguf.MODEL_TENSOR.HC_FFN_BASE,
        ("hc_ffn",  "scale"): gguf.MODEL_TENSOR.HC_FFN_SCALE,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # buffer for alpha tensors: {bid: {"attn": {}, "ffn": {}}}
        self._tc4_alphas: dict[int, dict[str, dict[str, Tensor]]] = {}

    def set_gguf_parameters(self):
        # XingChen4 config uses rope_scaling.type = "rope", which is equivalent
        # to DeepSeek's "yarn" (vLLM maps it to "deepseek_yarn", TRT-LLM maps
        # it to "yarn").  The base converter only recognises "yarn", so patch
        # the rope_type before delegating to V2's set_gguf_parameters.
        rope_type = self.rope_parameters.get("rope_type") or self.rope_parameters.get("type")
        if rope_type == "rope":
            self.rope_parameters["rope_type"] = "yarn"
            if "type" in self.rope_parameters:
                self.rope_parameters["type"] = "yarn"
            # base.py's YARN branch accesses this with a direct key lookup
            self.rope_parameters.setdefault("original_max_position_embeddings",
                self.hparams.get("original_max_position_embeddings", 4096))

        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_hyper_connection_count(
            hparams.get("hc_mult",  1))
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(
            hparams.get("hc_sinkhorn_iters",  20))
        self.gguf_writer.add_hyper_connection_epsilon(
            hparams.get("hc_eps",  1e-6))

    def set_vocab(self):
        # XingChen4 uses a SentencePiece tokenizer (tokenizer.model + XingChen4Tokenizer).
        # V2's set_vocab tries GPT2/BPE first, which fails for SPM. Use the SPM path directly.
        self._set_vocab_sentencepiece()
        # Override the pre-tokenizer type: _set_vocab_sentencepiece writes "default",
        # but XingChen4's SPM-style BPE needs a dedicated pre-type for correct pre-tokenization
        # (no word-level pre-split, no byte encoding, just newline splitting — same as Gemma4).
        self.gguf_writer.add_tokenizer_pre("xingchen4")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # handle mHC tensors: new format is model.layers.{N}.{attn_hc|ffn_hc}.{hc_fn|hc_base|hc_scale}
        #                      old format is model.layers.{N}.{attn_hc|ffn_hc}.{mapping_weight|bias|alpha_pre|alpha_post|alpha_res}
        match = re.match(r"model\.layers\.(\d+)\.(attn_hc|ffn_hc)\.(.+)$", name)
        if match:
            layer_idx = int(match.group(1))
            hc_type = match.group(2)  # "attn_hc" or "ffn_hc"
            param = match.group(3)    # "hc_fn", "hc_base", "hc_scale" (new) or "mapping_weight", "bias", "alpha_*" (old)

            if bid is None:
                bid = layer_idx

            prefix = "hc_attn" if hc_type == "attn_hc" else "hc_ffn"

            # --- New format: hc_fn / hc_base / hc_scale (direct 1:1 mapping) ---
            if param == "hc_fn":
                tensor_enum = self._hc_tensor_map[(prefix, "fn")]
                gguf_name = self.format_tensor_name(tensor_enum, bid)
                yield (gguf_name, data_torch)
                return

            if param == "hc_base":
                tensor_enum = self._hc_tensor_map[(prefix, "base")]
                gguf_name = self.format_tensor_name(tensor_enum, bid)
                yield (gguf_name, data_torch)
                return

            if param == "hc_scale":
                tensor_enum = self._hc_tensor_map[(prefix, "scale")]
                gguf_name = self.format_tensor_name(tensor_enum, bid)
                yield (gguf_name, data_torch)
                return

            # --- Legacy format: mapping_weight / bias / alpha_pre+alpha_post+alpha_res (concatenated) ---
            if param == "mapping_weight":
                tensor_enum = self._hc_tensor_map[(prefix, "fn")]
                gguf_name = self.format_tensor_name(tensor_enum, bid)
                yield (gguf_name, data_torch)
                return

            if param == "bias":
                tensor_enum = self._hc_tensor_map[(prefix, "base")]
                gguf_name = self.format_tensor_name(tensor_enum, bid)
                yield (gguf_name, data_torch)
                return

            if param in ("alpha_pre", "alpha_post", "alpha_res"):
                # buffer and concatenate when all three are collected
                if bid not in self._tc4_alphas:
                    self._tc4_alphas[bid] = {}
                if hc_type not in self._tc4_alphas[bid]:
                    self._tc4_alphas[bid][hc_type] = {}
                self._tc4_alphas[bid][hc_type][param] = data_torch

                alphas = self._tc4_alphas[bid][hc_type]
                if len(alphas) == 3:
                    scale = torch.cat([
                        alphas["alpha_pre"],
                        alphas["alpha_post"],
                        alphas["alpha_res"],
                    ])
                    tensor_enum = self._hc_tensor_map[(prefix, "scale")]
                    gguf_name = self.format_tensor_name(tensor_enum, bid)
                    del self._tc4_alphas[bid][hc_type]
                    yield (gguf_name, scale)
                return
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        for bid, hc_dict in self._tc4_alphas.items():
            for hc_type, alphas in hc_dict.items():
                if alphas:
                    raise ValueError(
                        f"Unprocessed mHC alpha tensors for layer {bid}, "
                        f"{hc_type}: {list(alphas.keys())}")
