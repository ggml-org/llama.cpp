from __future__ import annotations
import json
import os
from .base import (
    ModelBase, TextModel, SentencePieceTokenTypes,
    gguf, torch, logger
)
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("BertModel", "BertForMaskedLM", "CamembertModel", "BertForSequenceClassification")
class BertModel(TextModel):
    model_arch = gguf.MODEL_ARCH.BERT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = None
        if cls_out_labels := self.hparams.get("id2label"):
            if len(cls_out_labels) == 2 and cls_out_labels[0] == "LABEL_0":
                # Remove dummy labels added by AutoConfig
                cls_out_labels = None
        self.cls_out_labels = cls_out_labels

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_causal_attention(False)
        self._try_set_pooling_type()
        if self.cls_out_labels:
            self.gguf_writer.add_classifier_output_labels([v for k, v in sorted(self.cls_out_labels.items())])

    def set_vocab(self):
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.vocab_size = len(tokens)
        # we need this to validate the size of the token_type embeddings
        # though currently we are passing all zeros to the token_type embeddings
        # "Sequence A" or "Sequence B"
        self.gguf_writer.add_token_type_count(self.hparams.get("type_vocab_size", 1))

        # convert to phantom space vocab
        def phantom(tok):
            if tok.startswith("[") and tok.endswith("]"):
                return tok
            if tok.startswith("##"):
                return tok[2:]
            return "\u2581" + tok
        tokens = list(map(phantom, tokens))
        # add vocab to gguf
        self.gguf_writer.add_tokenizer_model("bert")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        # handle special tokens
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        if name.startswith("bert."):
            name = name[5:]
        if name.endswith(".gamma"):
            name = name[:-6] + ".weight"
        if name.endswith(".beta"):
            name = name[:-5] + ".bias"
        # we are only using BERT for embeddings so we don't need the pooling layer
        if name in ("embeddings.position_ids", "pooler.dense.weight", "pooler.dense.bias"):
            return [] # we don't need these
        if name.startswith("cls.predictions"):
            return []
        if name.startswith("cls.seq_relationship"):
            return []
        if self.cls_out_labels:
            # For BertForSequenceClassification (direct projection layer)
            if name == "classifier.weight":
                name = "classifier.out_proj.weight"
            if name == "classifier.bias":
                name = "classifier.out_proj.bias"
        return [(self.map_tensor_name(name), data_torch)]

    def _xlmroberta_tokenizer_init(self) -> None:
        # we need the pad_token_id to know how to chop down position_embd matrix
        if (pad_token_id := self.hparams.get("pad_token_id")) is not None:
            self._position_offset = 1 + pad_token_id
            if "max_position_embeddings" in self.hparams:
                self.hparams["max_position_embeddings"] -= self._position_offset
        else:
            self._position_offset = None

    def _xlmroberta_set_vocab(self) -> None:
        # to avoid TypeError: Descriptors cannot be created directly
        # exception when importing sentencepiece_model_pb2
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        from sentencepiece import SentencePieceProcessor
        from sentencepiece import sentencepiece_model_pb2 as model
        tokenizer_path = self.dir_model / 'sentencepiece.bpe.model'
        tokenizer_json = {}
        tokenizer_config_json = {}
        if not tokenizer_path.is_file():
            tokenizer_path = self.dir_model / 'tokenizer.json'
            tokenizer_config_path = self.dir_model / 'tokenizer_config.json'
            if not tokenizer_path.is_file():
                raise FileNotFoundError(f"File not found: {tokenizer_path}")
            from base64 import b64decode
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
            with open(tokenizer_path, "r", encoding="utf-8") as fp:
                tokenizer_json = json.load(fp)
            if tokenizer_config_path.is_file():
                with open(tokenizer_config_path, "r", encoding="utf-8") as fp:
                    tokenizer_config_json = json.load(fp)
            add_prefix = tokenizer.add_prefix_space
            remove_whitespaces = tokenizer.clean_up_tokenization_spaces
            precompiled_charsmap = b64decode(tokenizer_json["normalizer"]["precompiled_charsmap"])
            vocab_size = max(self.hparams.get("vocab_size", 0), tokenizer.vocab_size)
        else:
            sentencepiece_model = model.ModelProto()  # pyright: ignore[reportAttributeAccessIssue]
            sentencepiece_model.ParseFromString(open(tokenizer_path, "rb").read())
            assert sentencepiece_model.trainer_spec.model_type == 1  # UNIGRAM
            add_prefix = sentencepiece_model.normalizer_spec.add_dummy_prefix
            remove_whitespaces = sentencepiece_model.normalizer_spec.remove_extra_whitespaces
            precompiled_charsmap = sentencepiece_model.normalizer_spec.precompiled_charsmap
            tokenizer = SentencePieceProcessor()
            tokenizer.LoadFromFile(str(tokenizer_path))
            vocab_size = max(self.hparams.get("vocab_size", 0), tokenizer.vocab_size())
        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size
        if isinstance(tokenizer, SentencePieceProcessor):
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
        else:
            added_vocab = tokenizer.get_added_vocab()
            unk_token = tokenizer_config_json.get("unk_token")
            unk_token_id = added_vocab.get(unk_token, tokenizer_json["model"].get("unk_id", 3))
            for token_id in range(tokenizer.vocab_size):
                piece = tokenizer._convert_id_to_token(token_id)
                if (piece := tokenizer._convert_id_to_token(token_id)) is not None:
                    text = piece.encode("utf-8")
                    score = tokenizer_json["model"]["vocab"][token_id][1]
                    toktype = SentencePieceTokenTypes.NORMAL
                    if token_id == unk_token_id:
                        toktype = SentencePieceTokenTypes.UNKNOWN
                    elif token_id in tokenizer.all_special_ids:
                        toktype = SentencePieceTokenTypes.CONTROL
                    elif token_id in added_vocab.values():
                        toktype = SentencePieceTokenTypes.USER_DEFINED
                    # No reliable way to detect this, but jina doesn't have any
                    # elif tokenizer.IsByte(token_id):
                    #     toktype = SentencePieceTokenTypes.BYTE
                    tokens[token_id] = text
                    scores[token_id] = score
                    toktypes[token_id] = toktype
        if isinstance(tokenizer, SentencePieceProcessor):
            # realign tokens (see HF tokenizer code)
            tokens = [b'<s>', b'<pad>', b'</s>', b'<unk>'] + tokens[3:-1]
            scores = [0.0, 0.0, 0.0, 0.0] + scores[3:-1]
            toktypes = [
                SentencePieceTokenTypes.CONTROL,
                SentencePieceTokenTypes.CONTROL,
                SentencePieceTokenTypes.CONTROL,
                SentencePieceTokenTypes.UNKNOWN,
            ] + toktypes[3:-1]
            if self.model_arch == gguf.MODEL_ARCH.NOMIC_BERT_MOE:
                # Add mask token missing from sentencepiece.bpe.model
                tokens[250001] = b'<mask>'
                scores[250001] = 0.0
                toktypes[250001] = SentencePieceTokenTypes.CONTROL
        self.gguf_writer.add_tokenizer_model("t5")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_add_space_prefix(add_prefix)
        self.gguf_writer.add_token_type_count(self.hparams.get("type_vocab_size", 1))
        self.gguf_writer.add_remove_extra_whitespaces(remove_whitespaces)
        if precompiled_charsmap:
            self.gguf_writer.add_precompiled_charsmap(precompiled_charsmap)
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)


@ModelBase.register("DistilBertModel", "DistilBertForMaskedLM", "DistilBertForSequenceClassification")
class DistilBertModel(BertModel):
    model_arch = gguf.MODEL_ARCH.BERT

    def set_gguf_parameters(self):
        self.gguf_writer.add_layer_norm_eps(1e-12)
        logger.info("gguf: layer norm epsilon = 1e-12")
        super().set_gguf_parameters()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("distilbert."):
            name = name[11:]
        # These layers act as MLM head, so we don't need them
        if name.startswith("vocab_"):
            return []
        return super().modify_tensors(data_torch, name, bid)


@ModelBase.register("RobertaModel", "RobertaForSequenceClassification")
class RobertaModel(BertModel):
    model_arch = gguf.MODEL_ARCH.BERT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # we need the pad_token_id to know how to chop down position_embd matrix
        if (pad_token_id := self.hparams.get("pad_token_id")) is not None:
            self._position_offset = 1 + pad_token_id
            if "max_position_embeddings" in self.hparams:
                self.hparams["max_position_embeddings"] -= self._position_offset
        else:
            self._position_offset = None

    def set_vocab(self):
        """Support BPE tokenizers for roberta models"""
        bpe_tok_path = self.dir_model / "tokenizer.json"
        if bpe_tok_path.exists():
            self._set_vocab_gpt2()
            # we need this to validate the size of the token_type embeddings
            # though currently we are passing all zeros to the token_type embeddings
            # "Sequence A" or "Sequence B"
            self.gguf_writer.add_token_type_count(self.hparams.get("type_vocab_size", 1))
        else:
            return super().set_vocab()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # if name starts with "roberta.", remove the prefix
        # e.g. https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main
        if name.startswith("roberta."):
            name = name[8:]
        # position embeddings start at pad_token_id + 1, so just chop down the weight tensor
        if name == "embeddings.position_embeddings.weight":
            if self._position_offset is not None:
                data_torch = data_torch[self._position_offset:,:]
        return super().modify_tensors(data_torch, name, bid)


@ModelBase.register("NomicBertModel")
class NomicBertModel(BertModel):
    model_arch = gguf.MODEL_ARCH.BERT

    def __init__(self, dir_model: Path, ftype: gguf.LlamaFileType, fname_out: Path, **kwargs: Any):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, False)
        self.is_moe = bool(hparams.get("moe_every_n_layers"))
        self.model_arch = gguf.MODEL_ARCH.NOMIC_BERT_MOE if self.is_moe else gguf.MODEL_ARCH.NOMIC_BERT
        super().__init__(dir_model, ftype, fname_out, hparams=hparams, **kwargs)
        self._tokenizer_is_xlmroberta = self._is_tokenizer_xlmroberta()
        if self._tokenizer_is_xlmroberta:
            self._xlmroberta_tokenizer_init()
        npos, mtp = self.hparams["n_positions"], self.hparams.get("max_trained_positions", 2048)
        if npos == 8192 and mtp == 2048:
            self.hparams["n_positions"] = 2048  # nomic-embed-text v1 and v1.5 are trained for 2048 tokens.
        elif npos == 2048 and mtp == 2048:
            self.hparams["n_positions"] = 512   # nomic-embed-text-v2-moe is trained for 512 tokens.
        else:
            raise ValueError(f"unrecognized parameters: n_positions={npos}, max_trained_positions={mtp}")
        assert self.hparams["activation_function"] == "gelu" if self.is_moe else "swiglu"
        # this doesn't do anything in the HF version
        assert self.hparams["causal"] is False
        # no bias tensors unless MoE
        assert self.hparams["qkv_proj_bias"] == self.is_moe
        assert self.hparams["mlp_fc1_bias"]  == self.is_moe
        assert self.hparams["mlp_fc2_bias"]  == self.is_moe
        # norm at end of layer
        assert self.hparams["prenorm"] is False
        # standard RoPE
        assert self.hparams["rotary_emb_fraction"] == 1.0
        assert self.hparams["rotary_emb_interleaved"] is False
        assert self.hparams["rotary_emb_scale_base"] is None

    def set_vocab(self) -> None:
        if self._tokenizer_is_xlmroberta:
            return self._xlmroberta_set_vocab()
        return super().set_vocab()

    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        # If the tensor is an experts bias tensor, skip it by returning an empty list.
        if "mlp.experts.bias" in name:
            return []  # Explicitly return an empty list.
        if "mlp.experts.mlp.w1" in name:
            data_torch = data_torch.view(self.hparams["num_experts"], self.hparams["n_inner"], self.hparams["n_embd"])
            name += ".weight"
        if "mlp.experts.mlp.w2" in name:
            data_torch = data_torch.view(self.hparams["num_experts"], self.hparams["n_inner"], self.hparams["n_embd"])
            data_torch = data_torch.transpose(1, 2)
            name += ".weight"
        return [(self.map_tensor_name(name), data_torch)]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_freq_base(self.hparams["rotary_emb_base"])
        if self.is_moe:
            self.gguf_writer.add_moe_every_n_layers(self.hparams["moe_every_n_layers"])
            self.gguf_writer.add_expert_count(self.hparams["num_experts"])
            self.gguf_writer.add_expert_used_count(self.hparams["moe_top_k"])

    def _is_tokenizer_xlmroberta(self) -> bool:
        with open(self.dir_model / "tokenizer.json") as f:
            tokenizer_json = json.load(f)
        toktyp = tokenizer_json["model"]["type"]
        if toktyp == "Unigram":
            return True
        if toktyp == "WordPiece":
            return False
        raise ValueError(f"unknown tokenizer: {toktyp}")


@ModelBase.register("NeoBERT", "NeoBERTLMHead", "NeoBERTForSequenceClassification")
class NeoBert(BertModel):
    model_arch = gguf.MODEL_ARCH.NEO_BERT

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # NeoBERT uses 2/3 of the intermediate size as feed forward length
        self.gguf_writer.add_feed_forward_length(int(2 * self.hparams["intermediate_size"] / 3))
        self.gguf_writer.add_rope_freq_base(10000.0)  # default value for NeoBERT
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        f_rms_eps = self.hparams.get("norm_eps", 1e-6)  # default value for NeoBERT
        self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
        logger.info(f"gguf: rms norm epsilon = {f_rms_eps}")
        self.gguf_writer.add_pooling_type(gguf.PoolingType.CLS) # https://huggingface.co/chandar-lab/NeoBERT#how-to-use

    def modify_tensors(self, data_torch, name, bid):
        if name.startswith("decoder."):
            return []
        if name.startswith("model."):
            name = name[6:]
        return super().modify_tensors(data_torch, name, bid)


@ModelBase.register("XLMRobertaModel", "XLMRobertaForSequenceClassification")
class XLMRobertaModel(BertModel):
    model_arch = gguf.MODEL_ARCH.BERT
    _lora_files = {}
    _lora_names = []

    def __init__(self, dir_model: Path, ftype: gguf.LlamaFileType, fname_out: Path, **kwargs: Any):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, False)
        if lora_names := hparams.get("lora_adaptations"):
            self._lora_names = lora_names
            self.model_arch = gguf.MODEL_ARCH.JINA_BERT_V3
        super().__init__(dir_model, ftype, fname_out, hparams=hparams, **kwargs)
        self._xlmroberta_tokenizer_init()

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if self._lora_names:
            for name in self._lora_names:
                fname = self.add_prefix_to_filename(self.fname_out, f"lora-{name}-")
                self._lora_files[name] = gguf.GGUFWriter(fname, arch=gguf.MODEL_ARCH_NAMES[self.model_arch], endianess=self.endianess, use_temp_file=self.use_temp_file, dry_run=self.dry_run)
        return super().generate_extra_tensors()

    def set_type(self):
        for lora_writer in self._lora_files.values():
            lora_writer.add_type(gguf.GGUFType.ADAPTER)
            lora_writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
        super().set_type()

    def set_vocab(self):
        self._xlmroberta_set_vocab()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # if name starts with "roberta.", remove the prefix
        # e.g. https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main
        if name.startswith("roberta."):
            name = name[8:]
        # jina-embeddings-v3
        if ".parametrizations." in name:
            name = name.replace(".parametrizations.", ".")
            if name.endswith(".original"):
                name = name[:-9]
        # position embeddings start at pad_token_id + 1, so just chop down the weight tensor
        if name == "embeddings.position_embeddings.weight":
            if self._position_offset is not None:
                data_torch = data_torch[self._position_offset:,:]
        if name.endswith(".0.lora_A") or name.endswith(".0.lora_B"):
            if name.startswith("pooler.dense"):
                return []
            num_loras = data_torch.size(0)
            assert num_loras == len(self._lora_names)
            # Split out each LoRA in their own GGUF
            for i, lora_writer in enumerate(self._lora_files.values()):
                new_name = self.map_tensor_name(name[:-9]) + name[-7:].lower()
                data = data_torch[i, :, :]
                # Transpose/flip token_embd/types into correct shape
                if new_name == "token_embd.weight.lora_b":
                    data = data.T
                elif new_name.startswith("token_types.weight."):
                    new_name = new_name[:-1] + ("a" if new_name[-1:] == "b" else "b")
                lora_writer.add_tensor(new_name, data.float().numpy(), raw_dtype=gguf.GGMLQuantizationType.F32)
            return []
        return super().modify_tensors(data_torch, name, bid)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # jina-embeddings-v3
        if rotary_emb_base := self.hparams.get("rotary_emb_base"):
            self.gguf_writer.add_rope_freq_base(rotary_emb_base)
        lora_alpha = self.hparams.get("lora_alpha")
        if lora_prompt_prefixes := self.hparams.get("task_instructions"):
            assert self._lora_files and all(lora_name in lora_prompt_prefixes for lora_name in self._lora_files.keys())
        for lora_name, lora_writer in self._lora_files.items():
            lora_writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, lora_alpha if lora_alpha is not None else 1.0)
            lora_writer.add_string(gguf.Keys.Adapter.LORA_TASK_NAME, lora_name)
            if lora_prompt_prefixes:
                lora_writer.add_string(gguf.Keys.Adapter.LORA_PROMPT_PREFIX, lora_prompt_prefixes[lora_name])

    def write(self):
        super().write()
        for lora_writer in self._lora_files.values():
            lora_writer.write_header_to_file()
            lora_writer.write_kv_data_to_file()
            lora_writer.write_tensors_to_file(progress=True)
            lora_writer.close()


@ModelBase.register("JinaBertModel", "JinaBertForMaskedLM")
class JinaBertV2Model(BertModel):
    model_arch = gguf.MODEL_ARCH.JINA_BERT_V2

    def set_vocab(self):
        tokenizer_class = 'BertTokenizer'
        with open(self.dir_model / "tokenizer_config.json", "r", encoding="utf-8") as f:
            tokenizer_class = json.load(f)['tokenizer_class']
        if tokenizer_class == 'BertTokenizer':
            super().set_vocab()
        elif tokenizer_class == 'RobertaTokenizer':
            self._set_vocab_gpt2()
            self.gguf_writer.add_token_type_count(2)
        else:
            raise NotImplementedError(f'Tokenizer {tokenizer_class} is not supported for JinaBertModel')
