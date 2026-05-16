from __future__ import annotations

import json
import math

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register(
    "LLaMAForCausalLM",
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "VLlama3ForCausalLM",
    "LlavaForConditionalGeneration",
    "VoxtralForConditionalGeneration",
    "IQuestCoderForCausalLM",
    "LlamaModel")
class LlamaModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LLAMA
    undo_permute = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # fix for SmolVLM2, missing `num_attention_heads` in config.json
        if self.hf_arch == "VLlama3ForCausalLM":
            self.hparams["num_attention_heads"] = self.hparams.get("num_attention_heads", 32)
        # Mistral consolidated format has no config.json; origin_hf_arch is HF-only.
        if self.is_mistral_format:
            self.origin_hf_arch = None
        else:
            hparams = ModelBase.load_hparams(self.dir_model, is_mistral_format=False)
            self.origin_hf_arch = hparams.get('architectures', [None])[0]

    def set_vocab(self):
        if self.origin_hf_arch == "GlmasrModel":
            return self._set_vocab_glmedge()

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

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    def _repack_nvfp4(self, name: str, weight: Tensor, scale: Tensor, scale2: Tensor, input_scale: Tensor):
        # Mirror the BF16 Q/K RoPE permutation site in modify_tensors; the NVFP4 path bypasses it.
        if self.undo_permute:
            n_head = self.find_hparam(["n_heads", "num_attention_heads"], optional=True)
            n_kv_head = self.find_hparam(["n_kv_heads", "num_key_value_heads"], optional=True)
            if n_head is not None:
                if name.endswith("q_proj.weight"):
                    weight = LlamaModel.permute(weight, n_head, n_head)
                    scale  = LlamaModel.permute(scale, n_head, n_head)
                elif name.endswith("k_proj.weight"):
                    weight = LlamaModel.permute(weight, n_head, n_kv_head)
                    scale  = LlamaModel.permute(scale, n_head, n_kv_head)
        super()._repack_nvfp4(name, weight, scale, scale2, input_scale)

    _experts: list[dict[str, Tensor]] | None = None

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if "text_model." in name:
            name = name.replace("text_model.", "") # for SmolVLM

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.find_hparam(["n_heads", "num_attention_heads"])
        n_kv_head = self.find_hparam(["n_kv_heads", "num_key_value_heads"])

        if self.hf_arch == "LlamaModel":
            name = "model." + name

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
                # merge the experts into a single 3d tensor
                for wid in ["w1", "w2", "w3"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"layers.{bid}.feed_forward.experts.{wid}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if rope_params := self.rope_parameters.get("full_attention", self.rope_parameters):
            if rope_params.get("rope_type", '').lower() == "llama3":
                base = rope_params.get("rope_theta", 10000.0)
                if (dim := self.hparams.get("head_dim")) is None:
                    dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
                freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

                factor = rope_params.get("factor", 8.0)
                low_freq_factor = rope_params.get("low_freq_factor", 1.0)
                high_freq_factor = rope_params.get("high_freq_factor", 4.0)
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


@ModelBase.register("ArceeForCausalLM")
class ArceeModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.ARCEE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self._try_set_pooling_type()


@ModelBase.register(
    "Llama4ForConditionalGeneration",
    "Llama4ForCausalLM",
)
class Llama4Model(LlamaModel):
    model_arch = gguf.MODEL_ARCH.LLAMA4
    undo_permute = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # IMPORTANT: the normal "intermediate_size" is renamed to "intermediate_size_mlp", we need to undo this
        self.hparams["intermediate_size_moe"] = self.hparams["intermediate_size"]
        self.hparams["intermediate_size"] = self.hparams["intermediate_size_mlp"]

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_interleave_moe_layer_step(self.hparams["interleave_moe_layer_step"])
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["intermediate_size_moe"])
        if "layer_types" in self.hparams:
            if all(lt == "full_attention" for lt in self.hparams["layer_types"]):
                # all layers are full attention (for MobileLLM), disable swa
                self.gguf_writer.add_sliding_window(0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        # split the gate_up into gate and up
        if "gate_up_proj" in name:
            name_up = name.replace("gate_up_proj", "up_proj.weight")
            name_gate = name.replace("gate_up_proj", "gate_proj.weight")
            dim_half = data_torch.shape[-1] // 2
            gate_proj_weight, up_proj_weight = data_torch.transpose(-1, -2).split(dim_half, dim=-2)
            yield from super().modify_tensors(gate_proj_weight, name_gate, bid)
            yield from super().modify_tensors(up_proj_weight, name_up, bid)
            return

        if name.endswith("down_proj"):
            name += ".weight"
            data_torch = data_torch.transpose(-1, -2)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("LlamaBidirectionalModel")
class LlamaEmbedNemotronModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.LLAMA_EMBED


@ModelBase.register("SmolLM3ForCausalLM")
class SmolLM3Model(LlamaModel):
    model_arch = gguf.MODEL_ARCH.SMOLLM3


@ModelBase.register("ApertusForCausalLM")
class ApertusModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.APERTUS
    undo_permute = False

    _alpha_n = {}
    _alpha_p = {}
    _beta = {}
    _eps = {}

    def modify_tensors(self, data_torch, name, bid):
        # Handle xIELU activation parameters
        n_layers = self.hparams["num_hidden_layers"]
        if name.endswith(".act_fn.alpha_n"):
            self._alpha_n[bid] = data_torch.to("cpu").float().item()
            if (len(self._alpha_n) == n_layers):
                self.gguf_writer.add_xielu_alpha_n([self._alpha_n[k] for k in sorted(self._alpha_n)])
            return
        if name.endswith(".act_fn.alpha_p"):
            self._alpha_p[bid] = data_torch.to("cpu").float().item()
            if (len(self._alpha_p) == n_layers):
                self.gguf_writer.add_xielu_alpha_p([self._alpha_p[k] for k in sorted(self._alpha_p)])
            return
        if name.endswith(".act_fn.beta"):
            self._beta[bid] = data_torch.to("cpu").float().item()
            if (len(self._beta) == n_layers):
                self.gguf_writer.add_xielu_beta([self._beta[k] for k in sorted(self._beta)])
            return
        if name.endswith(".act_fn.eps"):
            self._eps[bid] = data_torch.to("cpu").float().item()
            if (len(self._eps) == n_layers):
                self.gguf_writer.add_xielu_eps([self._eps[k] for k in sorted(self._eps)])
            return

        yield from super().modify_tensors(data_torch, name, bid)
