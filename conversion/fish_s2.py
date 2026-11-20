from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .drama3 import Drama3MmprojModel, _neox_rows
from .qwen import Qwen3Model

# Fish Audio S2 Pro (dual-AR)
# text model: Qwen3 backbone (slow AR), samples the semantic code of each frame
# mmproj: fast AR code predictor (other codebooks) and Fish DAC decoder (codec.pth, same as drama3)

# residual codebook size of codec.pth, the fast AR head is cut to it
_N_RESIDUAL_CODES = 1024


def _load_hparams(dir_model: Path) -> dict[str, Any]:
    hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
    cfg = hparams["text_config"]
    hparams["text_config"] = {
        "hidden_size":             cfg["dim"],
        "num_hidden_layers":       cfg["n_layer"],
        "num_attention_heads":     cfg["n_head"],
        "num_key_value_heads":     cfg["n_local_heads"],
        "head_dim":                cfg["head_dim"],
        "intermediate_size":       cfg["intermediate_size"],
        "rms_norm_eps":            cfg["norm_eps"],
        "rope_theta":              cfg["rope_base"],
        "max_position_embeddings": cfg["max_seq_len"],
        "vocab_size":              cfg["vocab_size"],
        "tie_word_embeddings":     cfg["tie_word_embeddings"],
    }
    return hparams


@ModelBase.register("fish_qwen3_omni")
@ModelBase.example("fishaudio/s2-pro")
class FishS2TextModel(Qwen3Model):
    model_arch = gguf.MODEL_ARCH.QWEN3

    def __init__(self, dir_model: Path, *args, **kwargs):
        kwargs.pop("hparams", None)
        super().__init__(dir_model, *args, hparams=_load_hparams(dir_model), **kwargs)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # the rest is mapped by tensor_mapping.py
        name = name.removeprefix("text_model.")
        T = gguf.MODEL_TENSOR
        head_dim = self.hparams["head_dim"]
        if name.endswith(".attention.wqkv.weight"):
            n_q  = self.hparams["num_attention_heads"] * head_dim
            n_kv = self.hparams["num_key_value_heads"] * head_dim
            q, k, v = data_torch.split([n_q, n_kv, n_kv], dim=0)
            yield (self.format_tensor_name(T.ATTN_Q, bid), _neox_rows(q, head_dim))
            yield (self.format_tensor_name(T.ATTN_K, bid), _neox_rows(k, head_dim))
            yield (self.format_tensor_name(T.ATTN_V, bid), v)
            return
        if name.endswith(".attention.q_norm.weight"):
            yield (self.format_tensor_name(T.ATTN_Q_NORM, bid), _neox_rows(data_torch, head_dim))
            return
        if name.endswith(".attention.k_norm.weight"):
            yield (self.format_tensor_name(T.ATTN_K_NORM, bid), _neox_rows(data_torch, head_dim))
            return
        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("fish_qwen3_omni")
@ModelBase.example("fishaudio/s2-pro")
class FishS2MmprojModel(Drama3MmprojModel):
    gen_projector_type = gguf.VisionProjectorType.FISH_S2_GEN

    _FAST_AR_MAP = {
        "attention_norm.weight":     gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_NORM,
        "attention.wo.weight":       gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_OUT,
        "ffn_norm.weight":           gguf.MODEL_TENSOR.A_GEN_CODE_FFN_NORM,
        "feed_forward.w1.weight":    gguf.MODEL_TENSOR.A_GEN_CODE_FFN_GATE,
        "feed_forward.w3.weight":    gguf.MODEL_TENSOR.A_GEN_CODE_FFN_UP,
        "feed_forward.w2.weight":    gguf.MODEL_TENSOR.A_GEN_CODE_FFN_DOWN,
    }

    def __init__(self, dir_model: Path, *args, **kwargs):
        kwargs.pop("hparams", None)
        super().__init__(dir_model, *args, hparams=_load_hparams(dir_model), **kwargs)

    def get_audio_config(self) -> dict[str, Any] | None:
        cfg = dict(self.global_config["audio_decoder_config"])
        cfg["num_hidden_layers"] = cfg["n_layer"]
        return cfg

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name = item[0]
        if name.startswith("audio_decoder.") or name == "text_model.model.embeddings.weight":
            return item
        return None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        cfg = self.hparams

        # already named by generate_extra_tensors()
        if name.startswith("a."):
            yield (name, data_torch)
            return

        if name == "text_model.model.embeddings.weight":
            # semantic band of the backbone embeddings
            begin = self.global_config["semantic_start_token_id"]
            yield (self.format_tensor_name(T.A_GEN_CODE_OUT_EMBD), data_torch[begin:begin + cfg["vocab_size"]])
            return

        if name == "audio_decoder.codebook_embeddings.weight":
            # backbone input of a speech row, one table per codebook
            yield (self.format_tensor_name(T.A_GEN_CODE_EMBD), data_torch.reshape(cfg["num_codebooks"], -1, data_torch.shape[-1]))
            return

        if name == "audio_decoder.embeddings.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_TOK_EMBD), data_torch)
            return

        if name == "audio_decoder.output.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_HEAD), data_torch[:_N_RESIDUAL_CODES])
            return

        if name == "audio_decoder.norm.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_OUTPUT_NORM), data_torch)
            return

        assert bid is not None and name.startswith(f"audio_decoder.layers.{bid}.")
        key = name.removeprefix(f"audio_decoder.layers.{bid}.")
        if key == "attention.wqkv.weight":
            head_dim = cfg["head_dim"]
            n_q  = cfg["n_head"] * head_dim
            n_kv = cfg["n_local_heads"] * head_dim
            q, k, v = data_torch.split([n_q, n_kv, n_kv], dim=0)
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_Q, bid), _neox_rows(q, head_dim))
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_K, bid), _neox_rows(k, head_dim))
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_V, bid), v)
            return
        yield (self.format_tensor_name(self._FAST_AR_MAP[key], bid), data_torch)
