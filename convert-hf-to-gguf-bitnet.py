#!/usr/bin/env python3

from __future__ import annotations

import logging
import argparse
import contextlib
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Iterator, Sequence, TypeVar, cast
import configparser

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

# from convert import LlamaHfVocab, permute

logger = logging.getLogger("hf-to-gguf")


###### MODEL DEFINITIONS ######

class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


AnyModel = TypeVar("AnyModel", bound="type[Model]")


class Model(ABC):
    _model_classes: dict[str, type[Model]] = {}

    def __init__(self, dir_model: Path, ftype: int, fname_out: Path, is_big_endian: bool, use_temp_file: bool):
        self.dir_model = dir_model
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.use_temp_file = use_temp_file
        self.is_safetensors = self._is_model_safetensors()
        self.num_parts = Model.count_model_parts(self.dir_model, ".safetensors" if self.is_safetensors else ".bin")
        self.part_names = self._get_part_names()
        self.hparams = Model.load_hparams(self.dir_model)
        self.gguf_writer = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[self.model_arch], endianess=self.endianess, use_temp_file=self.use_temp_file)
        self.block_count = self.find_hparam(["n_layers", "num_hidden_layers", "n_layer"])
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    @property
    @abstractmethod
    def model_arch(self) -> gguf.MODEL_ARCH:
        pass

    def find_hparam(self, keys: Sequence[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in self.hparams), None)
        if key is not None:
            return self.hparams[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def set_vocab(self):
        self._set_vocab_gpt2()

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        for part_name in self.part_names:
            logger.info(f"gguf: loading model part '{part_name}'")
            ctx: ContextManager[Any]
            if self.is_safetensors:
                from safetensors import safe_open
                ctx = cast(ContextManager[Any], safe_open(self.dir_model / part_name, framework="pt", device="cpu"))
            else:
                ctx = contextlib.nullcontext(torch.load(str(self.dir_model / part_name), map_location="cpu", mmap=True, weights_only=True))

            with ctx as model_part:
                for name in model_part.keys():
                    data = model_part.get_tensor(name) if self.is_safetensors else model_part[name]
                    yield name, data

    def match_model_tensor_name(self, name: str, key: gguf.MODEL_TENSOR, bid: int | None, suffix: str = ".weight") -> bool:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            return False
        key_name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in key_name:
            if bid is None:
                return False
            key_name = key_name.format(bid=bid)
        else:
            if bid is not None:
                return False
        return name == (key_name + suffix)

    def map_tensor_name(self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")) -> str:
        new_name = self.tensor_map.get_name(key=name, try_suffixes=try_suffixes)
        if new_name is None:
            raise ValueError(f"Can not map tensor {name!r}")
        return new_name

    def set_gguf_parameters(self):
        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_block_count(self.block_count)

        if (n_ctx := self.find_hparam(["max_position_embeddings", "n_ctx"], optional=True)) is not None:
            self.gguf_writer.add_context_length(n_ctx)
            logger.info(f"gguf: context length = {n_ctx}")

        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        self.gguf_writer.add_embedding_length(n_embd)
        logger.info(f"gguf: embedding length = {n_embd}")

        if (n_ff := self.find_hparam(["intermediate_size", "n_inner"], optional=True)) is not None:
            self.gguf_writer.add_feed_forward_length(n_ff)
            logger.info(f"gguf: feed forward length = {n_ff}")

        n_head = self.find_hparam(["num_attention_heads", "n_head"])
        self.gguf_writer.add_head_count(n_head)
        logger.info(f"gguf: head count = {n_head}")

        if (n_head_kv := self.hparams.get("num_key_value_heads")) is not None:
            self.gguf_writer.add_head_count_kv(n_head_kv)
            logger.info(f"gguf: key-value head count = {n_head_kv}")

        if (rope_theta := self.hparams.get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base(rope_theta)
            logger.info(f"gguf: rope theta = {rope_theta}")
        if (f_rms_eps := self.hparams.get("rms_norm_eps")) is not None:
            self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
            logger.info(f"gguf: rms norm epsilon = {f_rms_eps}")
        if (f_norm_eps := self.find_hparam(["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_eps(f_norm_eps)
            logger.info(f"gguf: layer norm epsilon = {f_norm_eps}")
        if (n_experts := self.hparams.get("num_local_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
            logger.info(f"gguf: expert count = {n_experts}")
        if (n_experts_used := self.hparams.get("num_experts_per_tok")) is not None:
            self.gguf_writer.add_expert_used_count(n_experts_used)
            logger.info(f"gguf: experts used count = {n_experts_used}")

        self.gguf_writer.add_file_type(self.ftype)
        logger.info(f"gguf: file type = {self.ftype}")

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".attention.rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and (n_dims == 1 or new_name.endswith("_norm.weight")):
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            logger.info(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_writer.add_tensor(new_name, data)

    def write(self):
        self.write_tensors()
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file()
        self.gguf_writer.close()

    def write_vocab(self):
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.close()

    @staticmethod
    def count_model_parts(dir_model: Path, prefix: str) -> int:
        num_parts = 0
        for filename in os.listdir(dir_model):
            if filename.endswith(prefix):
                num_parts += 1

        return num_parts

    @staticmethod
    def load_hparams(dir_model):
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def register(cls, *names: str) -> Callable[[AnyModel], AnyModel]:
        assert names

        def func(modelcls: type[Model]):
            for name in names:
                cls._model_classes[name] = modelcls
            return modelcls
        return func

    @classmethod
    def from_model_architecture(cls, arch):
        try:
            return cls._model_classes[arch]
        except KeyError:
            raise NotImplementedError(f'Architecture {arch!r} not supported!') from None

    def _is_model_safetensors(self) -> bool:
        return Model.count_model_parts(self.dir_model, ".safetensors") > 0

    def _get_part_names(self):
        if self.is_safetensors:
            if self.num_parts == 1:  # there's only one .safetensors file
                return ("model.safetensors",)
            return (f"model-{n:05}-of-{self.num_parts:05}.safetensors" for n in range(1, self.num_parts + 1))

        if self.num_parts == 1:  # there's only one .bin file
            return ("pytorch_model.bin",)
        return (f"pytorch_model-{n:05}-of-{self.num_parts:05}.bin" for n in range(1, self.num_parts + 1))

    # used for GPT-2 BPE and WordPiece vocabs
    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.USER_DEFINED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                if tokenizer.added_tokens_decoder[i].special:
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        return tokens, toktypes, tokpre

    # NOTE: this function is generated by convert-hf-to-gguf-update.py
    #       do not modify it manually!
    # ref:  https://github.com/ggerganov/llama.cpp/pull/6920
    def get_vocab_base_pre(self, tokenizer) -> str:
        # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that
        # is specific for the BPE pre-tokenizer used by the model
        # we will use this unique identifier to write a "tokenizer.ggml.pre" entry in the GGUF file which we can
        # use in llama.cpp to implement the same pre-tokenizer

        chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

        chktok = tokenizer.encode(chktxt)
        chkhsh = sha256(str(chktok).encode()).hexdigest()

        logger.debug(f"chktok: {chktok}")
        logger.debug(f"chkhsh: {chkhsh}")

        res = None

        # NOTE: if you get an error here, you need to update the convert-hf-to-gguf-update.py script
        #       or pull the latest version of the model from Huggingface
        #       don't edit the hashes manually!
        if chkhsh == "0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5":
            # ref: https://huggingface.co/meta-llama/Meta-Llama-3-8B
            res = "llama-bpe"
        if chkhsh == "049ecf7629871e3041641907f3de7c733e4dbfdc736f57d882ba0b0845599754":
            # ref: https://huggingface.co/deepseek-ai/deepseek-llm-7b-base
            res = "deepseek-llm"
        if chkhsh == "347715f544604f9118bb75ed199f68779f423cabb20db6de6f31b908d04d7821":
            # ref: https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base
            res = "deepseek-coder"
        if chkhsh == "8aeee3860c56296a157a1fe2fad249ec40aa59b1bb5709f4ade11c4e6fe652ed":
            # ref: https://huggingface.co/tiiuae/falcon-7b
            res = "falcon"
        if chkhsh == "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f":
            # ref: https://huggingface.co/BAAI/bge-small-en-v1.5
            res = "bert-bge"
        if chkhsh == "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166":
            # ref: https://huggingface.co/mosaicml/mpt-7b
            res = "mpt"
        if chkhsh == "35d91631860c815f952d711435f48d356ebac988362536bed955d43bfa436e34":
            # ref: https://huggingface.co/bigcode/starcoder2-3b
            res = "starcoder"
        if chkhsh == "3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454":
            # ref: https://huggingface.co/openai-community/gpt2
            res = "gpt-2"
        if chkhsh == "6221ad2852e85ce96f791f476e0b390cf9b474c9e3d1362f53a24a06dc8220ff":
            # ref: https://huggingface.co/smallcloudai/Refact-1_6-base
            res = "refact"
        if chkhsh == "9c2227e4dd922002fb81bde4fc02b0483ca4f12911410dee2255e4987644e3f8":
            # ref: https://huggingface.co/CohereForAI/c4ai-command-r-v01
            res = "command-r"

        if res is None:
            logger.warning("\n")
            logger.warning("**************************************************************************************")
            logger.warning("** WARNING: The BPE pre-tokenizer was not recognized!")
            logger.warning("**          There are 2 possible reasons for this:")
            logger.warning("**          - the model has not been added to convert-hf-to-gguf-update.py yet")
            logger.warning("**          - the pre-tokenization config has changed upstream")
            logger.warning("**          Check your model files and convert-hf-to-gguf-update.py and update them accordingly.")
            logger.warning("** ref:     https://github.com/ggerganov/llama.cpp/pull/6920")
            logger.warning("**")
            logger.warning(f"** chkhsh:  {chkhsh}")
            logger.warning("**************************************************************************************")
            logger.warning("\n")
            raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")

        logger.debug(f"tokenizer.ggml.pre: {repr(res)}")
        logger.debug(f"chkhsh: {chkhsh}")

        return res

    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_sentencepiece(self):
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / 'tokenizer.model'

        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []

        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"File not found: {tokenizer_path}")

        tokenizer = SentencePieceProcessor(str(tokenizer_path))
        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        for token_id in range(tokenizer.vocab_size()):
            piece = tokenizer.id_to_piece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.get_score(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.is_unknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.is_control(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.is_unused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.is_byte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)

                for key in added_tokens_json:
                    key = key.encode("utf-8")
                    if key not in tokens:
                        tokens.append(key)
                        scores.append(-1000.0)
                        toktypes.append(SentencePieceTokenTypes.USER_DEFINED)

        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            logger.debug(f"Padding vocab with {pad_count} token(s) - [PAD1] through [PAD{pad_count}]")
            for i in range(1, pad_count + 1):
                tokens.append(f"[PAD{i}]")
                scores.append(-1000.0)
                toktypes.append(SentencePieceTokenTypes.UNUSED)

        assert len(tokens) == vocab_size

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_llama_hf(self):
        vocab = LlamaHfVocab(self.dir_model)
        tokens = []
        scores = []
        toktypes = []

        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        assert len(tokens) == vocab.vocab_size

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

# TL1

def process(weight, BM, BY, bm, by, M, K):
    final_weight = []

    # split in row with size of BM (160)
    outer_BM_weights = np.split(weight, (M // BM), axis=0)
    for outer_BM_weight in outer_BM_weights:
        # split in col with size of by (16index * 2 == 32nums)
        outer_BY_weights = np.split(outer_BM_weight, (K // BY), axis=1)
        for outer_BY_weight in outer_BY_weights:
            # split in row with size of bm (32)
            inner_bm_weights = np.split(outer_BY_weight, (BM // bm), axis=0)
            for inner_bm_weight in inner_bm_weights:
                # split in col with size of by (2index * 2 == 4nums)
                inner_by_weights = np.split(inner_bm_weight, (BY // by), axis=1)
                for inner_by_weight in inner_by_weights:
                    # 16 * 6 minor
                    minor_bm_weights = np.split(inner_by_weight, (bm // 16), axis=0)
                    for minor_bm_weight in minor_bm_weights:
                        minor_by_weights = np.split(minor_bm_weight, (by // 4), axis=1)
                        for minor in minor_by_weights:
                            minor_weight = np.split(minor, 2, axis=1)
                            hi_weight = minor_weight[0].astype(np.uint8) << 4
                            lo_weight = minor_weight[1].astype(np.uint8)
                            func_weight = lo_weight + hi_weight
                            final_weight.append(func_weight)

    weight = np.array(final_weight, dtype=np.uint8)
    return weight

# based on t_mac.utils.preprocess_weights
def preprocess_weights(
    w: np.ndarray,
    bits = 2,
    g    = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    M, K = w.shape
    weight = w
    weight = np.where(np.abs(weight) < 1e-6, 0, weight).astype(np.float32)
    weight = np.sign(weight)
    weight_num = np.prod(weight.shape)

    KEMD = 1536
    # outer loop
    BMEMD = 256
    BYEMD = 256

    # inner loop (32row 32num/16index)
    bmEMD = 32
    byEMD = 8

    KGQA = 4096

    BMGQA = 256
    BYGQA = 256

    bmGQA = 32
    byGQA = 8

    weight = np.reshape(weight, (weight_num // 2, 2))
    hi_weight = np.multiply(np.split(weight, 2, axis=1)[0], 3)
    lo_weight = np.split(weight, 2, axis=1)[1]

    weight = np.reshape((hi_weight + lo_weight), weight_num // 2)

    # row-major index
    weight = weight + 4
    weight = np.reshape(weight, (M, K // 2)).astype(np.uint8)

    if K == KEMD:
        weight = process(weight, BMEMD, BYEMD, bmEMD, byEMD, M, K)
    elif K == KGQA:
        weight = process(weight, BMGQA, BYGQA, bmGQA, byGQA, M, K)
    else:
        raise NotImplementedError

    return weight

# class WeightProcessor:
#     def __init__(self, config: dict = {}):
#         self.config = config

#     def process(self, w: np.ndarray, quant_type: gguf.GGMLQuantizationType) -> np.ndarray:
#         """Main entry for processing weights."""
#         M, K = w.shape
#         w = np.sign(np.where(np.abs(w) < 1e-6, 0, w).astype(np.float32))
#         weight_num = np.prod(w.shape)

#         if quant_type == gguf.GGMLQuantizationType.TL1:
#             return self.preprocess_weights_tl1(w, M, K, weight_num)
#         elif quant_type == gguf.GGMLQuantizationType.TL2:
#             return self.preprocess_weights_tl2(w, M, K, weight_num)
#         else:
#             raise ValueError(f"Unsupported mode: {quant_type}")

#     def process_common(self, w: np.ndarray, weight_num: int) -> Tuple[np.ndarray, np.ndarray]:
#         w = np.reshape(w, (weight_num // 2, 2))
#         hi_weight = np.multiply(np.split(w, 2, axis=1)[0], 3)
#         lo_weight = np.split(w, 2, axis=1)[1]
#         w = np.reshape((hi_weight + lo_weight), weight_num // 2)
#         return w, w.astype(np.uint8)

#     def preprocess_weights_tl1(self, w: np.ndarray, M: int, K: int, weight_num: int) -> np.ndarray:
#         KEMD, BMEMD, BYEMD, bmEMD, byEMD = 1536, 256, 256, 32, 8
#         KGQA, BMGQA, BYGQA, bmGQA, byGQA = 4096, 256, 256, 32, 8

#         w, weight = self.process_common(w, weight_num)
#         weight += 4
#         weight = np.reshape(weight, (M, K // 2)).astype(np.uint8)

#         if K == KEMD:
#             return self.process_by_blocks(weight, BMEMD, BYEMD, bmEMD, byEMD, M, K)
#         elif K == KGQA:
#             return self.process_by_blocks(weight, BMGQA, BYGQA, bmGQA, byGQA, M, K)
#         else:
#             raise NotImplementedError

#     def preprocess_weights_tl2(self, w: np.ndarray, M: int, K: int, weight_num: int) -> np.ndarray:
#         BM3, BY3, bm3, by3 = 128, 96, 32, 6
#         BM2, BY2, bm2, by2 = 128, 32, 32, 4

#         if (K % BY3 != 0):
#             slice_k_idx = K - K % BY3
#             three_weight, two_weight = np.split(w, [slice_k_idx], axis=1)
#         else:
#             three_weight, two_weight = w, None

#         final_weight = []

#         self.preprocess_three_weights(three_weight.shape[0],
#                          three_weight.shape[1],
#                          three_weight.shape[0] * three_weight.shape[1],
#                          BM3,
#                          BY3,
#                          bm3,
#                          by3,
#                          three_weight,
#                          final_weight)

#         if (two_weight is not None):
#             self.preprocess_two_weights(  two_weight.shape[0],
#                          two_weight.shape[1],
#                          two_weight.shape[0] * two_weight.shape[1],
#                          BM2,
#                          BY2,
#                          bm2,
#                          by2,
#                          two_weight,
#                          final_weight)

#         weight = np.array(final_weight, dtype=np.uint8)

#         return weight

#     def process_by_blocks(self, weight: np.ndarray, BM: int, BY: int, bm: int, by: int, M: int, K: int) -> np.ndarray:
#         """Block-wise processing used by TL1."""
#         final_weight = []
#         # split in row with size of BM (160)
#         outer_BM_weights = np.split(weight, (M // BM), axis=0)
#         for outer_BM_weight in outer_BM_weights:
#             # split in col with size of by (16index * 2 == 32nums)
#             outer_BY_weights = np.split(outer_BM_weight, (K // BY), axis=1)
#             for outer_BY_weight in outer_BY_weights:
#                 # split in row with size of bm (32)
#                 inner_bm_weights = np.split(outer_BY_weight, (BM // bm), axis=0)
#                 for inner_bm_weight in inner_bm_weights:
#                     # split in col with size of by (2index * 2 == 4nums)
#                     inner_by_weights = np.split(inner_bm_weight, (BY // by), axis=1)
#                     for inner_by_weight in inner_by_weights:
#                         # 16 * 6 minor
#                         minor_bm_weights = np.split(inner_by_weight, (bm // 16), axis=0)
#                         for minor_bm_weight in minor_bm_weights:
#                             minor_by_weights = np.split(minor_bm_weight, (by // 4), axis=1)
#                             for minor in minor_by_weights:
#                                 minor_weight = np.split(minor, 2, axis=1)
#                                 hi_weight = minor_weight[0].astype(np.uint8) << 4
#                                 lo_weight = minor_weight[1].astype(np.uint8)
#                                 func_weight = lo_weight + hi_weight
#                                 final_weight.append(func_weight)
#         return np.array(final_weight, dtype=np.uint8)

#     def preprocess_two_weights(self, M, K, weight_num, BM, BY, bm, by, weight, final_weight):
#         """Processing for TL2 with two weights."""
#         weight = np.reshape(weight, (weight_num // 2, 2))
#         hi_weight = np.multiply(np.split(weight, 2, axis=1)[0], 3)
#         lo_weight = np.split(weight, 2, axis=1)[1]

#         weight = np.reshape((hi_weight + lo_weight), weight_num // 2)

#         # row-major index
#         weight = weight + 4
#         weight = np.reshape(weight, (M, K // 2)).astype(np.uint8)

#         outer_BM_weights = np.split(weight, (M // BM), axis=0)
#         for outer_BM_weight in outer_BM_weights:
#             outer_BY_weights = np.split(outer_BM_weight, (K // BY), axis=1)
#             for outer_BY_weight in outer_BY_weights:
#                 inner_bm_weights = np.split(outer_BY_weight, (BM // bm), axis=0)
#                 for inner_bm_weight in inner_bm_weights:
#                     inner_by_weights = np.split(inner_bm_weight, (BY // by), axis=1)
#                     for inner_by_weight in inner_by_weights:
#                         func_weights = np.split(inner_by_weight, 2, axis=1)

#                         left_weight = func_weights[0]
#                         right_weight = func_weights[1]
#                         hi_weight = left_weight.astype(np.uint8) << 4
#                         lo_weight = right_weight.astype(np.uint8)
#                         func_weight = hi_weight + lo_weight
#                         final_weight.append(func_weight)

#     def preprocess_three_weights(self, M, K, weight_num, BM, BY, bm, by, weight, final_weight):
#         """Processing for TL2 with three weights."""
#         weight = np.reshape(weight, (weight_num // 3, 3))
#         split_weights = np.split(weight, 3, axis=1)
#         first_weight = np.multiply(split_weights[0], 9)
#         second_weight = np.multiply(split_weights[1], 3)
#         third_weight = split_weights[2]

#         weight = np.reshape((first_weight + second_weight + third_weight), weight_num // 3)

#         # row-major index
#         weight = weight + 4
#         weight = np.reshape(weight, (M, K // 3)).astype(np.uint8)

#         outer_BM_weights = np.split(weight, (M // BM), axis=0)
#         for outer_BM_weight in outer_BM_weights:
#             outer_BY_weights = np.split(outer_BM_weight, (K // BY), axis=1)
#             for outer_BY_weight in outer_BY_weights:
#                 inner_bm_weights = np.split(outer_BY_weight, (BM // bm), axis=0)
#                 for inner_bm_weight in inner_bm_weights:
#                     inner_by_weights = np.split(inner_bm_weight, (BY // by), axis=1)
#                     for inner_by_weight in inner_by_weights:
#                         func_weights = np.split(inner_by_weight, 2, axis=1)

#                         left_weight = func_weights[0]
#                         right_weight = func_weights[1]

#                         # Split left and right weights into smaller sub-weights
#                         left_sub_weights = np.split(left_weight, 4, axis=0)
#                         right_sub_weights = np.split(right_weight, 4, axis=0)

#                         # Re-arrange the left and right sub-weights
#                         new_left_weight = np.reshape(
#                             np.concatenate([left_sub_weights[0], left_sub_weights[2], 
#                                             left_sub_weights[1], left_sub_weights[3]], axis=0),
#                             (bm)
#                         )

#                         new_right_weight = np.reshape(
#                             np.concatenate([right_sub_weights[0], right_sub_weights[2], 
#                                             right_sub_weights[1], right_sub_weights[3]], axis=0),
#                             (bm)
#                         )

#                         # Combine high and low weights to get final weight
#                         hi_weight = new_left_weight.astype(np.uint8) << 4
#                         lo_weight = new_right_weight.astype(np.uint8)
#                         func_weight = hi_weight + lo_weight
#                         func_weight = np.reshape(func_weight, bm * by // 6)

#                         final_weight.append(func_weight)

#         return np.array(final_weight, dtype=np.uint8)



def read_model_config(model_dir: str) -> dict[str, Any]:
    config = os.path.join(model_dir, "config.json")
    if not os.path.exists(config):
        raise FileNotFoundError(f"Model config file not found: {config}")
    with open(config, "r") as f:
        return json.load(f)

@Model.register("BitnetForCausalLM")
class BitnetModel(Model):
    model_arch = gguf.MODEL_ARCH.BITNET

    def set_vocab(self):
        self._set_vocab_sentencepiece()
        
    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])

        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        self.gguf_writer.add_rope_scaling_factor(1.0)

    def weight_quant(self, weight):
        dtype = weight.dtype
        weight = weight.float()
        s =  1 / weight.abs().mean().clamp(min=1e-5)
        result = (weight * s).round().clamp(-1, 1) / s
        return result.type(dtype)
    
    def transform_to_lowbit(self, x: np.ndarray, quant_type):
        scale = np.max(np.abs(x))
        # res = np.round(x / scale + 2).astype(np.uint8)
        # res = WeightProcessor().process(x, quant_type)
        res = preprocess_weights(x)
        return res, scale

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # quant weight to low bit (in fp16)
        if name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight", 
                          "down_proj.weight", "up_proj.weight", "gate_proj.weight",
                          "o_proj.weight")):
            data_torch = self.weight_quant(data_torch)

        return [(self.map_tensor_name(name), data_torch)]

    def write_tensors(self):
        max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(".weight,")

        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # use the first number-like part of the tensor name as the block id
            bid = None
            for part in name.split("."):
                if part.isdecimal():
                    bid = int(part)
                    break

            for new_name, data in ((n, d.squeeze().numpy()) for n, d in self.modify_tensors(data_torch, name, bid)):
                data: np.ndarray = data  # type hint
                data_shape = data.shape
                n_dims = len(data.shape)
                data_dtype = data.dtype
                data_qtype: gguf.GGMLQuantizationType | None = None

                # when both are True, f32 should win
                # extra_f32 = self.extra_f32_tensors(name, new_name, bid, n_dims)
                # extra_f16 = self.extra_f16_tensors(name, new_name, bid, n_dims)
                extra_f32 = False
                extra_f16 = False

                # Most of the codebase that takes in 1D tensors or norms only handles F32 tensors
                # Conditions should closely match those in llama_model_quantize_internal in llama.cpp
                extra_f32 = any(cond for cond in (
                    extra_f32,
                    n_dims == 1,
                    new_name.endswith("_norm.weight"),
                ))

                # Some tensor types are always in float32
                extra_f32 = extra_f32 or any(self.match_model_tensor_name(new_name, key, bid) for key in (
                    gguf.MODEL_TENSOR.FFN_GATE_INP,
                    gguf.MODEL_TENSOR.POS_EMBD,
                    gguf.MODEL_TENSOR.TOKEN_TYPES,
                ))

                # if f16 desired, convert any float32 2-dim weight tensors to float16
                extra_f16 = any(cond for cond in (
                    extra_f16,
                    (name.endswith(".weight") and n_dims >= 2),
                ))
                suit_low_bit = not (name.endswith('embed_tokens.weight') or name.endswith('norm.weight'))

                lowbit_scale = None
                if self.ftype != gguf.GGMLQuantizationType.F32 and extra_f16 and not extra_f32:
                    if suit_low_bit and self.ftype in {gguf.GGMLQuantizationType.TL1, gguf.GGMLQuantizationType.TL2}:
                            data, lowbit_scale = self.transform_to_lowbit(data, self.ftype)
                            assert data.dtype == np.uint8 and lowbit_scale.dtype == np.float32
                            data_qtype = self.ftype
                    else:
                        data = data.astype(np.float16) if data_dtype != np.float16 else data
                        data_qtype = gguf.GGMLQuantizationType.F16

                if data_qtype is None:  # by default, convert to float32
                    if data_dtype != np.float32:
                        data = data.astype(np.float32)
                    data_qtype = gguf.GGMLQuantizationType.F32

                shape = data_shape
                # shape = gguf.quant_shape_from_byte_shape(data.shape, data_qtype) if data.dtype == np.uint8 else data.shape
                # reverse shape to make it similar to the internal ggml dimension order
                shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"

                # n_dims is implicit in the shape
                logger.info(f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

                self.gguf_writer.add_tensor(new_name, data, raw_shape=shape, raw_dtype=data_qtype)
                if lowbit_scale is not None:
                    self.gguf_writer.add_tensor(new_name + "_scale", lowbit_scale, raw_dtype=gguf.GGMLQuantizationType.F32)


###### CONVERSION LOGIC ######

ftype_map = {
    "f32": gguf.GGMLQuantizationType.F32,
    "f16": gguf.GGMLQuantizationType.F16,
    "tl1" : gguf.GGMLQuantizationType.TL1,
    "tl2" : gguf.GGMLQuantizationType.TL2,
}

def parse_args() -> argparse.Namespace:
    # TODO: config parse for specific model size
    parser = argparse.ArgumentParser(
        description="Convert a huggingface Bitnet model to a GGML compatible file")
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input",
    )
    parser.add_argument(
        "--outtype", type=str, choices=ftype_map.keys(), default="f32",
        help="output format - use f32 for float32, f16 for float16",
    )
    parser.add_argument("--bigendian", action="store_true", help="model is executed on big endian machine")
    parser.add_argument(
        "model", type=Path,
        help="directory containing model file",
    )
    parser.add_argument("--use-temp-file", action="store_true", help="use the tempfile library while processing (helpful when running out of memory, process killed)")
    parser.add_argument("--model-name", type=str, default=None, help="name of the model")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")

    return parser.parse_args()

def main() -> None:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    dir_model = args.model

    if not dir_model.is_dir():
        logger.error(f'Error: {args.model} is not a directory')
        sys.exit(1)

    if args.outfile is not None:
        fname_out = args.outfile
    else:
        # output in the same directory as the model by default
        fname_out = dir_model / f'ggml-model-{args.outtype}.gguf'

    logger.info(f"Loading model: {dir_model.name}")

    hparams = Model.load_hparams(dir_model)

    with torch.inference_mode():
        model_class = Model.from_model_architecture(hparams["architectures"][0])
        model_instance = model_class(dir_model, ftype_map[args.outtype], fname_out, args.bigendian, args.use_temp_file)

        logger.info("Set model parameters")
        model_instance.set_gguf_parameters()

        logger.info("Set model tokenizer")
        model_instance.set_vocab()

        if args.vocab_only:
            logger.info(f"Exporting model vocab to '{fname_out}'")
            model_instance.write_vocab()
        else:
            logger.info(f"Exporting model to '{fname_out}'")
            model_instance.write()

        logger.info(f"Model successfully exported to '{fname_out}'")


if __name__ == '__main__':
    args = parse_args()

    main()
