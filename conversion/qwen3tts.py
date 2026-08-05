from __future__ import annotations

import json
import re

from typing import Any, Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("Qwen3TTSForConditionalGeneration")
class Qwen3TTSTalkerModel(TextModel):
    # Qwen3-TTS talker: a 28-layer speech-code LM that autoregressively
    # emits codebook-0 codec tokens, plus a 5-layer code predictor
    # ("subtalker") that predicts codebooks 1..14 conditioned on the
    # backbone hidden state.
    #
    # GGUF layout:
    #  - backbone layers -> blk.0..27
    #  - code predictor layers -> blk.28..32 (offset by block_count,
    #    deliberately NOT carried as nextn layers: the predictor is not
    #    an MTP drafter for the backbone)
    #  - codec ids live outside the text vocab; the codec control ids
    #    (pad/bos/eos/think/...) are GGUF metadata, not vocab tokens
    #  - per-codebook predictor tensors carry a .{cid} suffix
    #    (cp_codec_embd.3.weight, cp_head.3.weight, ...)
    #  - the speaker encoder (voice cloning) is dropped: presets only
    model_arch = gguf.MODEL_ARCH.QWEN3TTS_TALKER

    _cp_layer_re = re.compile(r"^talker\.code_predictor\.model\.layers\.(\d+)\.(.+)$")
    _cp_codec_embd_re = re.compile(r"^talker\.code_predictor\.model\.codec_embedding\.(\d+)\.weight$")
    _cp_head_re = re.compile(r"^talker\.code_predictor\.lm_head\.(\d+)\.weight$")

    def __init__(self, *args: Any, **kwargs: Any):
        dir_model = args[0] if args else kwargs.get("dir_model")
        if kwargs.get("hparams") is None and dir_model is not None:
            with open(dir_model / "config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            talker = dict(config["talker_config"])
            talker["architectures"] = config["architectures"]
            # hparams["vocab_size"] is consumed by get_vocab_base() for the
            # TEXT vocab; the talker config uses it for the codec vocab, so
            # rename before the base class sees it
            talker["codec_vocab_size"] = talker.pop("vocab_size")
            talker["vocab_size"] = talker["text_vocab_size"]
            kwargs["hparams"] = talker
        super().__init__(*args, **kwargs)
        self.cp_config = self.hparams.get("code_predictor_config", {})
        if self.cp_config:
            # rebuild the tensor map so the cp layers at blk.{block_count + i}
            # resolve (the map only expands block patterns for range(n_blocks))
            n_total = self.block_count + self.cp_config["num_hidden_layers"]
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, n_total)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        # voice cloning path; out of scope (presets only)
        if name.startswith("speaker_encoder."):
            return None
        # the talker subtree IS the model here; return it directly so the
        # base class multimodal filter (which drops "talker." names) never
        # sees it
        if name.startswith("talker."):
            return name, gen
        return super().filter_tensors(item)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        w = self.gguf_writer
        w.add_tts_codec_vocab_size(self.find_hparam(["codec_vocab_size"]))
        w.add_tts_num_code_groups(self.find_hparam(["num_code_groups"]))
        w.add_tts_position_id_per_seconds(self.find_hparam(["position_id_per_seconds"]))
        w.add_tts_codec_pad_id(self.find_hparam(["codec_pad_id"]))
        w.add_tts_codec_bos_id(self.find_hparam(["codec_bos_id"]))
        w.add_tts_codec_eos_id(self.find_hparam(["codec_eos_token_id"]))
        w.add_tts_codec_think_id(self.find_hparam(["codec_think_id"]))
        w.add_tts_codec_nothink_id(self.find_hparam(["codec_nothink_id"]))
        w.add_tts_codec_think_bos_id(self.find_hparam(["codec_think_bos_id"]))
        w.add_tts_codec_think_eos_id(self.find_hparam(["codec_think_eos_id"]))
        if (langs := self.hparams.get("codec_language_id")) is not None:
            names = sorted(langs.keys())
            w.add_tts_codec_language_names(names)
            w.add_tts_codec_language_ids([langs[n] for n in names])
        # code predictor dims (rope theta / norm eps match the backbone)
        if self.cp_config:
            w.add_tts_predictor_layers(self.cp_config["num_hidden_layers"])
            w.add_tts_cp_hidden_size(self.cp_config["hidden_size"])
            w.add_tts_cp_feed_forward_length(self.cp_config["intermediate_size"])
            w.add_tts_cp_head_count(self.cp_config["num_attention_heads"])
            w.add_tts_cp_head_count_kv(self.cp_config["num_key_value_heads"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if (m := self._cp_layer_re.match(name)) is not None:
            # predictor layer i lives at blk.{block_count + i}
            cp_bid = self.block_count + int(m.group(1))
            yield from super().modify_tensors(data_torch, f"model.layers.{cp_bid}.{m.group(2)}", cp_bid)
            return

        if (m := self._cp_codec_embd_re.match(name)) is not None:
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.TTS_CP_CODEC_EMBD, suffix=f".{m.group(1)}.weight"), data_torch)
            return

        if (m := self._cp_head_re.match(name)) is not None:
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.TTS_CP_HEAD, suffix=f".{m.group(1)}.weight"), data_torch)
            return

        direct = {
            "talker.model.text_embedding.weight": (gguf.MODEL_TENSOR.TOKEN_EMBD, ".weight"),
            "talker.model.codec_embedding.weight": (gguf.MODEL_TENSOR.TTS_CODEC_EMBD, ".weight"),
            "talker.model.norm.weight": (gguf.MODEL_TENSOR.OUTPUT_NORM, ".weight"),
            "talker.codec_head.weight": (gguf.MODEL_TENSOR.TTS_CODEC_HEAD, ".weight"),
            "talker.text_projection.linear_fc1.weight": (gguf.MODEL_TENSOR.TTS_TEXT_PROJ_1, ".weight"),
            "talker.text_projection.linear_fc1.bias": (gguf.MODEL_TENSOR.TTS_TEXT_PROJ_1, ".bias"),
            "talker.text_projection.linear_fc2.weight": (gguf.MODEL_TENSOR.TTS_TEXT_PROJ_2, ".weight"),
            "talker.text_projection.linear_fc2.bias": (gguf.MODEL_TENSOR.TTS_TEXT_PROJ_2, ".bias"),
            "talker.code_predictor.small_to_mtp_projection.weight": (gguf.MODEL_TENSOR.TTS_CP_PROJ, ".weight"),
            "talker.code_predictor.small_to_mtp_projection.bias": (gguf.MODEL_TENSOR.TTS_CP_PROJ, ".bias"),
            "talker.code_predictor.model.norm.weight": (gguf.MODEL_TENSOR.TTS_CP_NORM, ".weight"),
        }
        if name in direct:
            key, suffix = direct[name]
            yield (self.format_tensor_name(key, suffix=suffix), data_torch)
            return

        # backbone layers: strip the talker prefix and use the standard map
        if name.startswith("talker.model."):
            yield from super().modify_tensors(data_torch, "model." + name[len("talker.model."):], bid)
            return

        raise ValueError(f"unhandled tensor: {name!r}")
