from __future__ import annotations

import math

from typing import Any, Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("T5Gemma2ForConditionalGeneration", "T5Gemma2Model")
class T5Gemma2Model(TextModel):
    """Convert the text-only portion of a T5Gemma2 encoder-decoder model."""

    model_arch = gguf.MODEL_ARCH.T5GEMMA2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not args:
            raise TypeError("T5Gemma2Model requires a model directory")

        supplied_hparams = kwargs.pop("hparams", None)
        root_hparams = (
            ModelBase.load_hparams(args[0], False)
            if supplied_hparams is None
            else supplied_hparams
        )
        try:
            encoder_hparams = root_hparams["encoder"]["text_config"]
            decoder_hparams = root_hparams["decoder"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "T5Gemma2 config must contain encoder.text_config and decoder"
            ) from error

        # TextModel expects a single flat text config. Decoder values are used
        # for the common dimensions while the complete root remains available
        # for encoder-specific validation and metadata.
        flat_hparams = {**root_hparams, **decoder_hparams}
        flat_hparams["architectures"] = root_hparams["architectures"]
        kwargs["hparams"] = flat_hparams
        self.root_hparams = root_hparams
        self.encoder_hparams = encoder_hparams
        self.decoder_hparams = decoder_hparams
        super().__init__(*args, **kwargs)
        self._validate_hparams()

    def _validate_hparams(self) -> None:
        matching_keys = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rms_norm_eps",
            "max_position_embeddings",
            "sliding_window",
            "layer_types",
            "rope_parameters",
            "query_pre_attn_scalar",
            "hidden_activation",
            "vocab_size",
        )
        mismatches = [
            key
            for key in matching_keys
            if self.encoder_hparams.get(key) != self.decoder_hparams.get(key)
        ]
        if mismatches:
            raise ValueError(
                "This T5Gemma2 converter currently requires matching encoder and "
                f"decoder configs; mismatched keys: {mismatches}"
            )

        if self.root_hparams.get("tie_word_embeddings") is not True:
            raise ValueError("T5Gemma2 text conversion requires tied embeddings")
        if self.decoder_hparams.get("bos_token_id") is None:
            raise ValueError("decoder.bos_token_id is required")

    @classmethod
    def filter_tensors(
        cls, item: tuple[str, Callable[[], Tensor]]
    ) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        if name.startswith("encoder.text_model."):
            name = "model.encoder." + name.removeprefix("encoder.text_model.")
        elif name.startswith("encoder."):
            name = "model." + name
        elif name.startswith("decoder."):
            name = "model." + name
        item = name, item[1]
        if name == "model.encoder.embed_tokens.eoi_embedding":
            logger.info("Skipping text-only unsupported EOI embedding")
            return None
        return super().filter_tensors(item)

    def set_vocab(self) -> None:
        # T5Gemma2 declares GemmaTokenizer but stores a byte-fallback BPE in
        # tokenizer.json. This is the same export path used by text Gemma3.
        self._set_vocab_gpt2()

    def set_gguf_parameters(self) -> None:
        super().set_gguf_parameters()
        hparams = self.decoder_hparams
        layer_types = hparams["layer_types"]
        rope_parameters = hparams["rope_parameters"]
        full_rope = rope_parameters["full_attention"]
        sliding_rope = rope_parameters["sliding_attention"]

        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_decoder_block_count(hparams["num_hidden_layers"])
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(
            [layer_type == "sliding_attention" for layer_type in layer_types]
        )
        self.gguf_writer.add_decoder_start_token_id(hparams["bos_token_id"])
        scales = self.root_hparams.get("bongard_embedding_scales")
        if scales is not None:
            if scales["encoder"] != scales["decoder"]:
                raise ValueError("Text export requires equal encoder/decoder embedding scales")
            embedding_scale = scales["encoder"]
        else:
            embedding_scale = math.sqrt(hparams["hidden_size"])
        self.gguf_writer.add_embedding_scale(embedding_scale)
        self.gguf_writer.add_attention_scale(
            1.0 / math.sqrt(hparams["query_pre_attn_scalar"])
        )
        self.gguf_writer.add_hidden_act(hparams["hidden_activation"])
        self.gguf_writer.add_rope_dimension_count(hparams["head_dim"])
        self.gguf_writer.add_rope_dimension_count_swa(hparams["head_dim"])

        # TextModel writes these too, but keep explicit validation here so a
        # malformed nested T5Gemma2 config cannot silently use wrong defaults.
        if full_rope != {
            "factor": 8.0,
            "rope_theta": 1_000_000,
            "rope_type": "linear",
        }:
            raise ValueError(f"Unexpected full-attention RoPE config: {full_rope}")
        if sliding_rope != {
            "rope_theta": 10_000,
            "rope_type": "default",
        }:
            raise ValueError(f"Unexpected sliding-attention RoPE config: {sliding_rope}")

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        if name == "model.encoder.embed_tokens.weight":
            name = "shared.weight"

        # HF T5Gemma2 RMSNorm computes x * (1 + weight). Norm tensors are kept
        # in F32 by the generic GGUF converter, so this shift remains lossless.
        if name.endswith("norm.weight"):
            data_torch = data_torch + 1.0

        yield from super().modify_tensors(data_torch, name, bid)
