from __future__ import annotations
import json
import os
from .base import (
    ModelBase, TextModel, SentencePieceTokenTypes,
    gguf, logger
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("T5WithLMHeadModel")
@ModelBase.register("T5ForConditionalGeneration")
@ModelBase.register("MT5ForConditionalGeneration")
@ModelBase.register("UMT5ForConditionalGeneration")
class T5Model(TextModel):
    model_arch = gguf.MODEL_ARCH.T5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_token_embeddings_found = False

    def set_vocab(self):
        # to avoid TypeError: Descriptors cannot be created directly
        # exception when importing sentencepiece_model_pb2
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        from sentencepiece import SentencePieceProcessor
        from sentencepiece import sentencepiece_model_pb2 as model
        tokenizer_path = self.dir_model / 'tokenizer.model'
        # many older models use spiece.model tokenizer model filename
        if not tokenizer_path.is_file():
            tokenizer_path = self.dir_model / 'spiece.model'
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"File not found: {tokenizer_path}")
        sentencepiece_model = model.ModelProto()  # pyright: ignore[reportAttributeAccessIssue]
        sentencepiece_model.ParseFromString(open(tokenizer_path, "rb").read())
        # some models like Pile-T5 family use BPE tokenizer instead of Unigram
        if sentencepiece_model.trainer_spec.model_type == 2:  # BPE
            # assure the tokenizer model file name is correct
            assert tokenizer_path.name == 'tokenizer.model'
            return self._set_vocab_sentencepiece()
        else:
            assert sentencepiece_model.trainer_spec.model_type == 1  # UNIGRAM
        add_prefix = sentencepiece_model.normalizer_spec.add_dummy_prefix
        remove_whitespaces = sentencepiece_model.normalizer_spec.remove_extra_whitespaces
        precompiled_charsmap = sentencepiece_model.normalizer_spec.precompiled_charsmap
        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))
        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())
        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size
        for token_id in range(tokenizer.vocab_size()):
            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)
            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE
            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype
        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)
                for key in added_tokens_json:
                    token_id = added_tokens_json[key]
                    if token_id >= vocab_size:
                        logger.warning(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                        continue
                    tokens[token_id] = key.encode("utf-8")
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED
        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            logger.debug(f"Padding vocab with {pad_count} token(s) - [PAD1] through [PAD{pad_count}]")
            for i in range(1, pad_count + 1):
                tokens.append(bytes(f"[PAD{i}]", encoding="utf-8"))
                scores.append(-1000.0)
                toktypes.append(SentencePieceTokenTypes.UNUSED)
        self.gguf_writer.add_tokenizer_model("t5")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_add_space_prefix(add_prefix)
        self.gguf_writer.add_remove_extra_whitespaces(remove_whitespaces)
        if precompiled_charsmap:
            self.gguf_writer.add_precompiled_charsmap(precompiled_charsmap)
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        if (n_ctx := self.find_hparam(["n_positions"], optional=True)) is None:
            logger.warning("Couldn't find context length in config.json, assuming default value of 512")
            n_ctx = 512
        self.gguf_writer.add_context_length(n_ctx)
        self.gguf_writer.add_embedding_length(self.hparams["d_model"])
        self.gguf_writer.add_feed_forward_length(self.hparams["d_ff"])
        self.gguf_writer.add_block_count(self.hparams["num_layers"])
        if (dec_n_layer := self.hparams.get("num_decoder_layers")) is not None:
            self.gguf_writer.add_decoder_block_count(dec_n_layer)
        self.gguf_writer.add_head_count(self.hparams["num_heads"])
        self.gguf_writer.add_key_length(self.hparams["d_kv"])
        self.gguf_writer.add_value_length(self.hparams["d_kv"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_relative_attn_buckets_count(self.hparams["relative_attention_num_buckets"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_decoder_start_token_id(self.hparams["decoder_start_token_id"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        # T5 based models contain shared token embeddings tensors saved randomly as either "encoder.embed_tokens.weight",
        # "decoder.embed_tokens.weight" or "shared.weight" tensor. In some models there are even multiple of them stored
        # in the safetensors files. We use the first tensor from these three as the token embeddings for both encoder
        # and decoder and ignore the remaining ones.
        if name in ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight", "shared.weight"]:
            if not self.shared_token_embeddings_found:
                name = "shared.weight"
                self.shared_token_embeddings_found = True
            else:
                logger.debug(f"Skipping shared tensor {name!r} in safetensors so that convert can end normally.")
                return []
        return [(self.map_tensor_name(name), data_torch)]


@ModelBase.register("T5EncoderModel")
class T5EncoderModel(TextModel):
    model_arch = gguf.MODEL_ARCH.T5ENCODER

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_token_embeddings_found = False

    def set_vocab(self):
        # to avoid TypeError: Descriptors cannot be created directly
        # exception when importing sentencepiece_model_pb2
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        from sentencepiece import SentencePieceProcessor
        from sentencepiece import sentencepiece_model_pb2 as model
        tokenizer_path = self.dir_model / 'tokenizer.model'
        # many older models use spiece.model tokenizer model filename
        if not tokenizer_path.is_file():
            tokenizer_path = self.dir_model / 'spiece.model'
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"File not found: {tokenizer_path}")
        sentencepiece_model = model.ModelProto()  # pyright: ignore[reportAttributeAccessIssue]
        sentencepiece_model.ParseFromString(open(tokenizer_path, "rb").read())
        # some models like Pile-T5 family use BPE tokenizer instead of Unigram
        if sentencepiece_model.trainer_spec.model_type == 2:  # BPE
            # assure the tokenizer model file name is correct
            assert tokenizer_path.name == 'tokenizer.model'
            return self._set_vocab_sentencepiece()
        else:
            assert sentencepiece_model.trainer_spec.model_type == 1  # UNIGRAM
        add_prefix = sentencepiece_model.normalizer_spec.add_dummy_prefix
        remove_whitespaces = sentencepiece_model.normalizer_spec.remove_extra_whitespaces
        precompiled_charsmap = sentencepiece_model.normalizer_spec.precompiled_charsmap
        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))
        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())
        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size
        for token_id in range(tokenizer.vocab_size()):
            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)
            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE
            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype
        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)
                for key in added_tokens_json:
                    token_id = added_tokens_json[key]
                    if token_id >= vocab_size:
                        logger.warning(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                        continue
                    tokens[token_id] = key.encode("utf-8")
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED
        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            logger.debug(f"Padding vocab with {pad_count} token(s) - [PAD1] through [PAD{pad_count}]")
            for i in range(1, pad_count + 1):
                tokens.append(bytes(f"[PAD{i}]", encoding="utf-8"))
                scores.append(-1000.0)
                toktypes.append(SentencePieceTokenTypes.UNUSED)
        self.gguf_writer.add_tokenizer_model("t5")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_add_space_prefix(add_prefix)
        self.gguf_writer.add_remove_extra_whitespaces(remove_whitespaces)
        if precompiled_charsmap:
            self.gguf_writer.add_precompiled_charsmap(precompiled_charsmap)
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        if (n_ctx := self.find_hparam(["n_positions"], optional=True)) is None:
            logger.warning("Couldn't find context length in config.json, assuming default value of 512")
            n_ctx = 512
        self.gguf_writer.add_context_length(n_ctx)
        self.gguf_writer.add_embedding_length(self.hparams["d_model"])
        self.gguf_writer.add_feed_forward_length(self.hparams["d_ff"])
        self.gguf_writer.add_block_count(self.hparams["num_layers"])
        self.gguf_writer.add_head_count(self.hparams["num_heads"])
        self.gguf_writer.add_key_length(self.hparams["d_kv"])
        self.gguf_writer.add_value_length(self.hparams["d_kv"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_relative_attn_buckets_count(self.hparams["relative_attention_num_buckets"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        # T5 based models contain shared token embeddings tensors saved randomly as either "encoder.embed_tokens.weight",
        # "decoder.embed_tokens.weight" or "shared.weight" tensor. In some models there are even multiple of them stored
        # in the safetensors files. We use the first tensor from these three as the token embeddings for both encoder
        # and decoder and ignore the remaining ones.
        if name in ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight", "shared.weight"]:
            if not self.shared_token_embeddings_found:
                name = "shared.weight"
                self.shared_token_embeddings_found = True
            else:
                logger.debug(f"Skipping shared tensor {name!r} in safetensors so that convert can end normally.")
                return []
        return [(self.map_tensor_name(name), data_torch)]
