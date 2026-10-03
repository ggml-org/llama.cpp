from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("GearForCausalLM", "GearModel")
class GearModel(TextModel):
    model_arch = gguf.MODEL_ARCH.GEAR

    def set_gguf_parameters(self):
        layer_types = self.hparams["layer_types"]
        if len(layer_types) != self.block_count:
            raise ValueError(f"Expected {self.block_count} Gear layer types, got {len(layer_types)}")

        self.hparams["num_key_value_heads"] = [
            self.hparams["num_key_value_heads"] if layer_type != "conv_mixer" else 0
            for layer_type in layer_types
        ]

        super().set_gguf_parameters()

        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])

        head_dim = self.hparams.get("head_dim")
        if head_dim is not None:
            self.gguf_writer.add_rope_dimension_count(head_dim)
            self.gguf_writer.add_rope_dimension_count_swa(head_dim)
            self.gguf_writer.add_key_length_swa(head_dim)
            self.gguf_writer.add_value_length_swa(head_dim)
            logger.info(f"gguf: rope dimension count = {head_dim}")
            logger.info(f"gguf: rope dimension count swa = {head_dim}")

        if (sliding_window := self.hparams.get("sliding_window")) is not None:
            self.gguf_writer.add_sliding_window(sliding_window)
            self.gguf_writer.add_sliding_window_pattern([layer_type == "sliding_attention" for layer_type in layer_types])
            logger.info(f"gguf: sliding window = {sliding_window}")
            logger.info(f"gguf: sliding window pattern length = {len(layer_types)}")

        if (conv_l_cache := self.hparams.get("conv_L_cache")) is not None:
            self.gguf_writer.add_shortconv_l_cache(conv_l_cache)
            logger.info(f"gguf: shortconv l cache = {conv_l_cache}")

        if (hidden_act := self.hparams.get("hidden_activation")) is not None:
            self.gguf_writer.add_hidden_act(hidden_act)
            logger.info(f"gguf: hidden activation = {hidden_act}")

    def set_vocab(self):
        self._set_vocab_gpt2()

    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(self.dir_model / "tokenizer.json"))
        vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)
        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()
        added_tokens_decoder = tokenizer.added_tokens_decoder

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
                continue

            token = reverse_vocab[i]
            if token in added_vocab:
                if not added_tokens_decoder[i].normalized:
                    previous_token = token
                    token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))
                    if previous_token != token:
                        logger.info(f"{repr(previous_token)} is encoded and decoded back to {repr(token)} using PreTrainedTokenizerFast")

                if added_tokens_decoder[i].special or self.does_token_look_special(token):
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    token = token.replace(b"\xe2\x96\x81".decode("utf-8"), " ")
                    toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                toktypes.append(gguf.TokenType.NORMAL)
            tokens.append(token)

        return tokens, toktypes, tokpre

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(("local_mixer.key_conv.weight", "local_mixer.value_conv.weight")):
            data_torch = data_torch.squeeze(1).float()

        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.GEAR_MIX_KEY_CONV, bid):
            return gguf.GGMLQuantizationType.F32
        if self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.GEAR_MIX_VALUE_CONV, bid):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)
