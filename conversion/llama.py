from __future__ import annotations
import json
import math
from .base import (
    ModelBase, TextModel, gguf, torch, logger, _mistral_common_installed, _mistral_import_error_msg
)
from pathlib import Path
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor

# Import mistral_common types
from mistral_common.tokens.tokenizers.mistral import MistralTokenizerType
from mistral_common.tokens.tokenizers.mistral import MistralVocab
from mistral_common.tokens.tokenizers.mistral import MistralModel


@ModelBase.register("LLaMAForCausalLM", "LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM", "VLlama3ForCausalLM", "LlavaForConditionalGeneration", "VoxtralForConditionalGeneration", "LlamaModel")
class LlamaModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LLAMA
    undo_permute = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # fix for SmolVLM2, missing `num_attention_heads` in config.json
        if self.hf_arch == "VLlama3ForCausalLM":
            self.hparams["num_attention_heads"] = self.hparams.get("num_attention_heads", 32)

    def _set_vocab_mistral(self):
        if not _mistral_common_installed:
            raise ImportError(_mistral_import_error_msg)
        vocab = MistralVocab(self.dir_model)
        logger.info(
            f"Converting tokenizer {vocab.tokenizer_type} of size {vocab.vocab_size}."
        )
        self.gguf_writer.add_tokenizer_model(vocab.gguf_tokenizer_model)
        tokens = []
        scores = []
        toktypes = []
        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)
        assert len(tokens) == vocab.vocab_size, (
            f"token count ({len(tokens)}) != vocab size ({vocab.vocab_size})"
        )
        if vocab.tokenizer_type == MistralTokenizerType.tekken:
            self.gguf_writer.add_tokenizer_pre("tekken")
            self.gguf_writer.add_token_merges(
                vocab.extract_vocab_merges_from_model()
            )
        logger.info(
            f"Setting bos, eos, unk and pad token IDs to {vocab.bos_id}, {vocab.eos_id}, {vocab.unk_id}, {vocab.pad_id}."
        )
        self.gguf_writer.add_bos_token_id(vocab.bos_id)
        self.gguf_writer.add_eos_token_id(vocab.eos_id)
        self.gguf_writer.add_unk_token_id(vocab.unk_id)
        self.gguf_writer.add_pad_token_id(vocab.pad_id)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_vocab_size(vocab.vocab_size)
        self.gguf_writer.add_add_bos_token(True)
        self.gguf_writer.add_add_eos_token(False)
        template_dir = Path(__file__).parent / "models/templates/"
        if not self.is_mistral_format or not self.disable_mistral_community_chat_template:
            # Log only for Mistral format that the official tokenization and detokenization is via `mistral-common`.
            if self.is_mistral_format:
                logger.info(
                    "Using a Mistral community chat template. These templates can be subject to errors in early days or weeks after a release. "
                    "Mistral recommends to use `mistral-common` to perform tokenization and detokenization."
                )
            template = MistralModel.get_community_chat_template(vocab, template_dir, self.is_mistral_format)
            self.gguf_writer.add_chat_template(template)
        else:
            logger.info("Not using a Mistral community chat template. Ensure to perform the tokenization and detokenization via `mistral-common`.")

    def set_vocab(self):
        if self.is_mistral_format:
            return self._set_vocab_mistral()
        path_tekken_json = self.dir_model / "tekken.json"
        path_tokenizer_json = self.dir_model / "tokenizer.json"
        if path_tekken_json.is_file() and not path_tokenizer_json.is_file():
            self._set_vocab_mistral()
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            try:
                self._set_vocab_llama_hf()
            except (FileNotFoundError, TypeError):
                # Llama 3
                self._set_vocab_gpt2()
        # Apply to CodeLlama only (and ignore for Llama 3 with a vocab size of 128256)
        if self.hparams.get("vocab_size", 32000) == 32016:
            special_vocab = gguf.SpecialVocab(
                self.dir_model, load_merges=False,
                special_token_types = ['prefix', 'suffix', 'middle', 'eot']
            )
            special_vocab._set_special_token("prefix", 32007)
            special_vocab._set_special_token("suffix", 32008)
            special_vocab._set_special_token("middle", 32009)
            special_vocab._set_special_token("eot",    32010)
            special_vocab.add_to_gguf(self.gguf_writer)
        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                if "add_prefix_space" in tokenizer_config_json:
                    self.gguf_writer.add_add_space_prefix(tokenizer_config_json["add_prefix_space"])
        # Apply to granite small models only
        if self.hparams.get("vocab_size", 32000) == 49152:
            self.gguf_writer.add_add_bos_token(False)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        if not self.is_mistral_format:
            self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(rope_dim)
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if rope_scaling.get("rope_type", rope_scaling.get("type")) == "linear" and "factor" in rope_scaling:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))
    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.find_hparam(["n_heads", "num_attention_heads"])
        n_kv_head = self.find_hparam(["n_kv_heads", "num_key_value_heads"])
        vision_prefixes = [
            "vision_encoder.",
            "vision_language_adapter.",
            "patch_merger.",
            "pre_mm_projector_norm",
        ]
        is_multimodal_tensor = "vision_tower" in name \
            or "vision_model" in name \
            or "audio_tower" in name \
            or "model.connector" in name \
            or "multi_modal_projector" in name \
            or any(
                name.startswith(prefix)
                for prefix in vision_prefixes
            )
        if is_multimodal_tensor:
            return [] # skip vision tensors
        elif self.hf_arch == "LlamaModel":
            name = "model." + name
        elif name.startswith("model.text_model"):
            name = name.replace("text_model.", "") # for SmolVLM
        elif name.startswith("language_model."):
            name = name.replace("language_model.", "") # for the rest
        if self.undo_permute:
            if name.endswith(("q_proj.weight", "q_proj.bias")):
                data_torch = LlamaModel.permute(data_torch, n_head, n_head)
            if name.endswith(("k_proj.weight", "k_proj.bias")):
                data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)
        # process the experts separately
        if name.find("block_sparse_moe.experts") != -1:
            n_experts = self.hparams["num_local_experts"]
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []
                # merge the experts into a single 3d tensor
                for wid in ["w1", "w2", "w3"]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]
                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"layers.{bid}.feed_forward.experts.{wid}.weight"
                    new_name = self.map_tensor_name(merged_name)
                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []
        return [(self.map_tensor_name(name), data_torch)]

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if rope_scaling := self.find_hparam(["rope_scaling"], optional=True):
            if rope_scaling.get("rope_type", '').lower() == "llama3":
                base = self.hparams.get("rope_theta", 10000.0)
                if (dim := self.hparams.get("head_dim")) is None:
                    dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
                freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                factor = rope_scaling.get("factor", 8.0)
                low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
                high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
                old_context_len = self.hparams.get("original_max_position_embeddings", 8192)
                low_freq_wavelen = old_context_len / low_freq_factor
                high_freq_wavelen = old_context_len / high_freq_factor
                # assert low_freq_wavelen != high_freq_wavelen # Errors for Llama4
                rope_factors = []
                for freq in freqs:
                    wavelen = 2 * math.pi / freq
                    if wavelen < high_freq_wavelen:
                        rope_factors.append(1)
                    elif wavelen > low_freq_wavelen:
                        rope_factors.append(factor)
                    else:
                        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                        rope_factors.append(1 / ((1 - smooth) / factor + smooth))
                yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), torch.tensor(rope_factors, dtype=torch.float32))

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
