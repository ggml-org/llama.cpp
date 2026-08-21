from __future__ import annotations

import math
import re

from typing import TYPE_CHECKING, Callable, Iterable

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf

from .deepseek import DeepseekV2Model


@ModelBase.register("Dots3NoteForCausalLM", "Dots3NoteForConditionalGeneration", "Dots3NoteTextForCausalLM")
class Dots3NoteModel(DeepseekV2Model):
    model_arch = gguf.MODEL_ARCH.DOTS3NOTE
    skip_mtp = False
    supports_mtp_export = True

    # trunk layer count, stashed before indexing for filter_tensors (mirrors DeepseekV32Model)
    _n_main_layers: int | None = None

    def index_tensors(self, remote_hf_model_id: str | None = None):
        type(self)._n_main_layers = self.hparams["num_hidden_layers"]
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        hparams = self.hparams

        # config file doesn't specify MTP block, detect it from model weight
        self.n_nextn = 1 if "model.mtp.embed_tokens.weight" in self.model_tensors else 0
        if self.n_nextn:
            self.block_count += self.n_nextn
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        self.layer_types = hparams["layer_types"]
        if len(self.layer_types) < hparams["num_hidden_layers"]:
            raise ValueError("layer_types is shorter than num_hidden_layers")

        if hparams.get("use_dsa", True) is not True:
            raise ValueError("dots3-note conversion requires use_dsa=true")
        if hparams.get("normalization", "RMSNorm") != "RMSNorm" or hparams.get("final_norm", "RMSNorm") != "RMSNorm":
            raise ValueError("dots3-note conversion only supports RMSNorm")
        if hparams.get("k_rope_only_layernorm", True) is not True:
            raise ValueError("dots3-note conversion requires k_rope_only_layernorm=true")
        if hparams.get("topk_method", "noaux_tc") != "noaux_tc" or hparams.get("scoring_func") != "sigmoid":
            raise ValueError("dots3-note conversion only supports noaux_tc/sigmoid expert gating")
        if hparams.get("n_group", 1) != 1 or hparams.get("topk_group", 1) != 1:
            raise ValueError("dots3-note conversion does not support grouped expert routing")
        if hparams.get("use_dynamic_rsf", False) or hparams.get("moe_gating_fp32", False):
            raise ValueError("dots3-note conversion does not support use_dynamic_rsf/moe_gating_fp32")
        for key in ("attention_gate_type", "swa_attention_gate_type"):
            if hparams.get(key, "headwise") != "headwise":
                raise ValueError(f"dots3-note conversion only supports headwise attention gate, got {key}={hparams.get(key)!r}")
        if hparams["swa_qk_nope_head_dim"] + hparams["swa_qk_rope_head_dim"] != hparams.get("swa_head_dim", 256):
            raise ValueError("swa_head_dim must equal swa_qk_nope_head_dim + swa_qk_rope_head_dim")
        if hparams["swa_qk_rope_head_dim"] != hparams["qk_rope_head_dim"]:
            # both layer kinds share a single rope_dimension_count
            raise ValueError("swa_qk_rope_head_dim must match qk_rope_head_dim")

        self.apply_lora_rescale = hparams.get("apply_mla_qkv_lora_rescale", False)

    def _is_swa_layer(self, bid: int) -> bool:
        if bid >= self.hparams["num_hidden_layers"]:
            # note: the NextN/MTP block uses the sliding-attention MLA
            return True
        return self.layer_types[bid] == "sliding_attention"

    def set_vocab(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|endofassistant|>"])  # ty: ignore[unresolved-attribute]
        special_vocab.add_to_gguf(self.gguf_writer)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem
        if name.startswith(("vision_encoder.", "audio_encoder.")):
            return None

        assert cls._n_main_layers is not None
        is_mtp = name.startswith("model.mtp.") or \
            ((m := re.match(r"model\.layers\.(\d+)\.", name)) is not None and int(m.group(1)) >= cls._n_main_layers)

        # --no-mtp: drop the NextN/MTP block; --mtp: keep only that block plus the shared embeddings/norm/lm_head
        if is_mtp and cls.no_mtp:
            return None
        if cls.mtp_only and not is_mtp and name not in (
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        ):
            return None

        return name, gen

    def set_gguf_parameters(self):
        hparams = self.hparams

        # head_count is a per-layer array because the two layer kinds have different head counts
        n_layer = hparams["num_hidden_layers"]
        hparams["num_attention_heads"] = [
            hparams["swa_num_attention_heads"] if self._is_swa_layer(il) else hparams["num_attention_heads"]
            for il in range(self.block_count)
        ]

        # prevent the base class from emitting key/value_length from the unused head_dim
        hparams.pop("head_dim", None)

        super().set_gguf_parameters()

        # MLA geometry of the sliding-window layers (rope.freq_base_swa is emitted by the base class)
        swa_kv_lora_rank = hparams["swa_kv_lora_rank"]
        self.gguf_writer.add_sliding_window(hparams["sliding_window_size"])
        self.gguf_writer.add_sliding_window_pattern([self._is_swa_layer(il) for il in range(n_layer)])
        self.gguf_writer.add_kv_lora_rank_swa(swa_kv_lora_rank)
        self.gguf_writer.add_key_length_swa(swa_kv_lora_rank + hparams["swa_qk_rope_head_dim"])
        self.gguf_writer.add_value_length_swa(swa_kv_lora_rank)
        self.gguf_writer.add_key_length_mla_swa(hparams["swa_qk_nope_head_dim"] + hparams["swa_qk_rope_head_dim"])
        self.gguf_writer.add_value_length_mla_swa(hparams["swa_v_head_dim"])
        if hparams["swa_q_lora_rank"] != hparams["q_lora_rank"]:
            raise ValueError("dots3-note conversion assumes a shared q_lora_rank for both layer kinds")

        if self.n_nextn:
            self.gguf_writer.add_nextn_predict_layers(self.n_nextn)

        # DSA indexer (full-attention layers only)
        self.gguf_writer.add_indexer_head_count(hparams["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(hparams["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(hparams["index_topk"])
        self.gguf_writer.add_indexer_types([not self._is_swa_layer(il) for il in range(n_layer)])

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
        # move the MTP token embedding into the NextN block so the standard nextn mapping picks it up
        if name == "model.mtp.embed_tokens.weight":
            name = f"model.layers.{self.hparams['num_hidden_layers']}.embed_tokens.weight"
            bid = self.hparams["num_hidden_layers"]

        # fold the activation rescale sqrt(n_embd/lora_rank) into the preceding RMSNorm weight
        # this also covers the indexer wq_b, which reads the same rescaled q_lora activation
        if self.apply_lora_rescale and bid is not None:
            if name.endswith("q_a_layernorm.weight"):
                data_torch = data_torch * math.sqrt(self.hparams["hidden_size"] / self.hparams["q_lora_rank"])
            elif name.endswith("kv_a_layernorm.weight"):
                rank = self.hparams["swa_kv_lora_rank"] if self._is_swa_layer(bid) else self.hparams["kv_lora_rank"]
                data_torch = data_torch * math.sqrt(self.hparams["hidden_size"] / rank)

        # MLA absorption: split kv_b_proj into k_b (transposed) and v_b, per-layer-kind geometry
        if name.endswith("kv_b_proj.weight"):
            assert bid is not None
            if self._is_swa_layer(bid):
                n_head = self.hparams["swa_num_attention_heads"]
                qk_nope_head_dim = self.hparams["swa_qk_nope_head_dim"]
                v_head_dim = self.hparams["swa_v_head_dim"]
            else:
                n_head = self.hparams["num_attention_heads"]
                qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
                v_head_dim = self.hparams["v_head_dim"]
            if isinstance(n_head, list):  # set_gguf_parameters turns this into a per-layer array
                n_head = n_head[bid]

            assert data_torch.shape[0] == n_head * (qk_nope_head_dim + v_head_dim)

            kv_b = data_torch.view(n_head, qk_nope_head_dim + v_head_dim, data_torch.shape[-1])
            k_b, v_b = kv_b.split([qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)

            yield from ModelBase.modify_tensors(self, k_b, name.replace("kv_b_proj", "k_b_proj"), bid)
            yield from ModelBase.modify_tensors(self, v_b, name.replace("kv_b_proj", "v_b_proj"), bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)
