from __future__ import annotations

import re
from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .deepseek import DeepseekV2Model


@ModelBase.register("Xing4_0ForCausalLM", "XingChen4ForCausalLM")
class Xing4_0Model(DeepseekV2Model):
    """Xing4_0 (formerly XingChen4): DeepSeek-V2/V3 backbone (MLA + MoE) + mHC residual mixing."""

    model_arch = gguf.MODEL_ARCH.XING4_0

    # NextN/MTP: the checkpoint appends the MTP block past num_hidden_layers
    # (model.layers.40 -> blk.40), mirroring DeepSeek-V3.2.  The C++ loader
    # (src/models/xing4_0.cpp) expects the MTP block to carry the standard
    # MLA + MoE tensors plus nextn.eh_proj/enorm/hnorm (and optionally
    # embed_tokens/shared_head.*), but NO mHC tensors -- those are trunk-only.
    skip_mtp = False
    supports_mtp_export = True
    _n_main_layers: int | None = None

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
        self._xing4_0_alphas: dict[int, dict[str, dict[str, Tensor]]] = {}

        self.block_count = self.hparams["num_hidden_layers"]
        if not self.no_mtp:
            self.block_count += self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def index_tensors(self, remote_hf_model_id: str | None = None):
        # needed by filter_tensors() before any tensor is filtered
        type(self)._n_main_layers = self.hparams["num_hidden_layers"]
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem

        # the NextN/MTP block lives past num_hidden_layers (model.layers.40 -> blk.40)
        assert cls._n_main_layers is not None
        is_mtp = (m := re.match(r"model\.layers\.(\d+)\.", name)) is not None and int(m.group(1)) >= cls._n_main_layers

        if is_mtp and re.search(r"\.(attn_hc|ffn_hc)\.", name):
            return None

        # --no-mtp: drop the appended NextN block entirely.
        if is_mtp and cls.no_mtp:
            return None
        # --mtp: keep ONLY NextN-block tensors plus the shared embeddings/
        # norm/lm_head (so the resulting GGUF carries just the draft head).
        if cls.mtp_only and not is_mtp and name not in (
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        ):
            return None

        return name, gen

    def set_gguf_parameters(self):
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

        # NextN/MTP prediction layers
        if not self.no_mtp and (num_nextn_predict_layers := hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(num_nextn_predict_layers)

        self.gguf_writer.add_hyper_connection_count(
            hparams.get("hc_mult",  1))
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(
            hparams.get("hc_sinkhorn_iters",  20))
        self.gguf_writer.add_hyper_connection_epsilon(
            hparams.get("hc_eps",  1e-6))

    def set_vocab(self):
        # Xing4_0 uses a SentencePiece tokenizer (tokenizer.model + Xing4_0Tokenizer).
        # V2's set_vocab tries GPT2/BPE first, which fails for SPM. Use the SPM path directly.
        # Note: no add_tokenizer_pre override — the C++ SPM load path ignores
        # tokenizer.ggml.pre entirely (pre-tokenizers only apply to BPE vocabs).
        self._set_vocab_sentencepiece()

    def prepare_metadata(self, vocab_only: bool):
        from_dir = self.fname_out.is_dir()
        super().prepare_metadata(vocab_only=vocab_only)

        if not self.mtp_only or not from_dir:
            return

        output_type: str = self.ftype.name.partition("_")[2]
        fname_default: str = gguf.naming_convention(
            self.metadata.name, self.metadata.basename, self.metadata.finetune,
            self.metadata.version, size_label=None, output_type=output_type, model_type=None)
        self.fname_out = self.fname_out.parent / f"mtp-{fname_default}.gguf"

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
                if bid not in self._xing4_0_alphas:
                    self._xing4_0_alphas[bid] = {}
                if hc_type not in self._xing4_0_alphas[bid]:
                    self._xing4_0_alphas[bid][hc_type] = {}
                self._xing4_0_alphas[bid][hc_type][param] = data_torch

                alphas = self._xing4_0_alphas[bid][hc_type]
                if len(alphas) == 3:
                    scale = torch.cat([
                        alphas["alpha_pre"],
                        alphas["alpha_post"],
                        alphas["alpha_res"],
                    ])
                    tensor_enum = self._hc_tensor_map[(prefix, "scale")]
                    gguf_name = self.format_tensor_name(tensor_enum, bid)
                    del self._xing4_0_alphas[bid][hc_type]
                    yield (gguf_name, scale)
                return
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        for bid, hc_dict in self._xing4_0_alphas.items():
            for hc_type, alphas in hc_dict.items():
                if alphas:
                    raise ValueError(
                        f"Unprocessed mHC alpha tensors for layer {bid}, "
                        f"{hc_type}: {list(alphas.keys())}")
