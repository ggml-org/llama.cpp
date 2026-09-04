from __future__ import annotations

import re
from typing import Iterable

import torch
from torch import Tensor

import gguf

from .base import ModelBase
from .glm import GlmMoeDsaModel


@ModelBase.register("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM")
@ModelBase.example("zai-org/GLM-5.3-Flash")
class Glm5NextModel(GlmMoeDsaModel):
    """GLM-5.3-Flash.

    Trunk that alternates KDA linear attention (34 layers) with MLA + DSA sparse
    attention (11 layers), wrapped in hyper-connection streams. The pieces are
    already in tree: the KDA tensors follow kimi-linear, the hyper-connection and
    k-pool compressor tensors follow deepseek4, and the MLA/MoE/NextN half is
    inherited from GLM-5.2 (GlmMoeDsaModel).
    """

    model_arch = gguf.MODEL_ARCH.GLM5NEXT

    # Tensors that carry no per-layer index and are named differently from the
    # generic mapping, resolved by suffix (same approach as DeepseekV4Model).
    _direct_map = {
        "hc_attn_fn":   (gguf.MODEL_TENSOR.HC_ATTN_FN,    ""),
        "hc_attn_base": (gguf.MODEL_TENSOR.HC_ATTN_BASE,  ""),
        "hc_attn_scale": (gguf.MODEL_TENSOR.HC_ATTN_SCALE, ""),
        "hc_ffn_fn":    (gguf.MODEL_TENSOR.HC_FFN_FN,     ""),
        "hc_ffn_base":  (gguf.MODEL_TENSOR.HC_FFN_BASE,   ""),
        "hc_ffn_scale": (gguf.MODEL_TENSOR.HC_FFN_SCALE,  ""),
        "self_attn.indexer.index_kpool_compress_ape":
            (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_APE,   ""),
        "self_attn.indexer.index_kpool_compress_gate":
            (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_WGATE, ""),
    }

    def index_tensors(self, remote_hf_model_id: str | None = None):
        # TextModel lifts text_config to the root, but only after this runs -
        # and the parent already needs num_hidden_layers from it here.
        # Skip None values: AutoConfig.to_dict() materialises keys that the JSON
        # omits, so text_config carries architectures=None and would clobber the
        # valid top-level value.
        if "text_config" in self.hparams:
            self.hparams = {
                **self.hparams,
                **{k: v for k, v in self.hparams["text_config"].items() if v is not None},
            }
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item):
        name = item[0]
        # text-only for now: drop the vision tower
        if name.startswith("model.visual.") or name.startswith("visual."):
            return None
        return super().filter_tensors(item)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        # hyper-connections (mHC): identical formulation to DeepSeek-V4, so the
        # existing sinkhorn graph applies unchanged.
        self.gguf_writer.add_hyper_connection_count(hparams["hc_mult"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(hparams["hc_sinkhorn_iters"])
        self.gguf_writer.add_hyper_connection_epsilon(hparams["hc_eps"])

        # KDA linear attention
        linear = hparams["linear_attn_config"]
        self.gguf_writer.add_ssm_conv_kernel(linear["short_conv_kernel_size"])
        self.gguf_writer.add_ssm_inner_size(linear["num_heads"] * linear["head_dim"])
        self.gguf_writer.add_ssm_state_size(linear["head_dim"])
        self.gguf_writer.add_ssm_group_count(linear["num_heads"])

        # k-pool compression inside the DSA indexer
        self.gguf_writer.add_indexer_block_size(hparams["index_kpool"])

        # clamped SwiGLU
        if (limit := hparams.get("swiglu_limit")) is not None:
            self.gguf_writer.add_swiglu_clamp_exp([limit] * self.block_count)
            self.gguf_writer.add_swiglu_clamp_shexp([limit] * self.block_count)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # the checkpoint wraps the trunk for the multimodal head
        name = re.sub(r"^model\.language_model\.", "model.", name)

        # KDA decay conventions, same as conversion/kimi_linear.py: the graph
        # expects ssm_a to already hold -exp(A_log), and the time-step bias to
        # be named like a bias so it is not loaded as a MUL_MAT weight.
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch.float())
        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"

        for suffix, (tensor, ext) in self._direct_map.items():
            if name.endswith(suffix) and bid is not None:
                return [(self.format_tensor_name(tensor, bid) + ext, data_torch)]

        return super().modify_tensors(data_torch, name, bid)
