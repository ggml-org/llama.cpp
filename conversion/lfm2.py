from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf, jinja_str_or_json, logger

from .gemma import ConformerAudioModel


@ModelBase.register("Lfm2ForCausalLM", "LFM2ForCausalLM")
@ModelBase.example("LiquidAI/LFM2-1.2B", "LiquidAI/LFM2.5-350M")
class LFM2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.LFM2

    def _add_feed_forward_length(self):
        ff_dim = self.find_hparam(["block_ff_dim", "intermediate_size"])
        auto_adjust_ff_dim = self.hparams["block_auto_adjust_ff_dim"]
        ffn_dim_multiplier = self.hparams["block_ffn_dim_multiplier"]
        multiple_of = self.hparams["block_multiple_of"]

        if auto_adjust_ff_dim:
            ff_dim = int(2 * ff_dim / 3)
            # custom dim factor multiplier
            if ffn_dim_multiplier is not None:
                ff_dim = int(ffn_dim_multiplier * ff_dim)
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.gguf_writer.add_feed_forward_length(ff_dim)

    def set_gguf_parameters(self):
        # set num_key_value_heads only for attention layers
        self.hparams["num_key_value_heads"] = [
            self.hparams["num_key_value_heads"] if layer_type != "conv" else 0
            for layer_type in self.hparams["layer_types"]
        ]

        super().set_gguf_parameters()
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        self.gguf_writer.add_shortconv_l_cache(self.hparams["conv_L_cache"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["norm_eps"])
        self._add_feed_forward_length()

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if ConformerAudioModel.is_audio_tensor(name):
            # skip multimodal tensors
            return None

        name = name.replace("lfm.", "model.")      # audio

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # conv op requires 2d tensor
        if 'conv.conv' in name:
            data_torch = data_torch.squeeze(1)

        yield from super().modify_tensors(data_torch, name, bid)


def _is_d1_checkpoint(dir_model: Path) -> bool:
    if not (dir_model / "config.json").is_file():
        return False
    with open(dir_model / "config.json", encoding="utf-8") as f:
        return json.load(f).get("auto_map", {}).get("AutoModel", "").endswith(".D1Model")


@ModelBase.register_hparams_loader(_is_d1_checkpoint)
def _load_d1_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected d1 checkpoint")
    hparams = ModelBase.load_hparams(dir_model, False, guess=False)
    # the mmproj stays LFM2-VL
    hparams["text_config"]["architectures"] = ["D1Model"]
    return hparams


@ModelBase.register("D1Model")
@ModelBase.example("LiquidAI/d1-3b")
class D1Model(LFM2Model):
    model_arch = gguf.MODEL_ARCH.LFM2

    def set_vocab(self):
        super().set_vocab()
        self.gguf_writer.add_chat_template([{"name": "systemone", "template": self._systemone_template()}])

    @staticmethod
    def _systemone_template() -> str:
        # follows prompt.py of the model repo
        description = jinja_str_or_json("o.description")
        choice = (
            "{{ '\\n\\nOptions:\\n' }}"
            "{% for o in options %}{{ o.label }} {% if o.description %}" + description + "{% else %}{{ o.key | replace('_', ' ') }}{% endif %}"
            "{% if not loop.last %}{{ '\\n' }}{% endif %}{% endfor %}"
            "{{ '\\n\\nReply with the option code only.' }}"
        )
        # with criteria, a missing description is written as None
        noul = (
            "{% set ns = namespace(criteria=false) %}{% for o in options %}{% if o.description is not none %}{% set ns.criteria = true %}{% endif %}{% endfor %}"
            "{% if ns.criteria %}"
            "{% for o in options %}{{ '\\nYes: ' if o.key == 'true' else '\\nNo: ' }}"
            "{% if o.description is none %}None{% else %}" + description + "{% endif %}{% endfor %}{% endif %}"
            "{{ '\\n\\nReply with yes or no only.' }}"
        )
        score = (
            "{{ '\\n\\n' }}{% for o in options %}{{ o.key }} " + description + "{{ '\\n' }}{% endfor %}"
            "{{ '\\nReply with a single digit 0-' }}{{ options | length - 1 }}{{ ' only.' }}"
        )
        return (
            "<|startoftext|><|im_start|>user\n"
            "{% for image in images %}{{ image }}{% endfor %}"
            "{% if state is not none %}{% if state is string %}{{ state }}{% else %}{{ state | tojson(indent=2) }}{% endif %}"
            "{{ '\\n\\n\\nQUESTION:\\n' }}{% endif %}"
            + jinja_str_or_json("instructions")
            + "{% if type == 'choice' %}" + choice + "{% elif type == 'noul' %}" + noul + "{% else %}" + score + "{% endif %}"
            "{{ '<|im_end|>\\n<|im_start|>assistant\\n' }}"
        )

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_decision_type(gguf.DecisionType.LFM2_D1)


@ModelBase.register("Lfm2Model", "Lfm2BidirectionalModel", "Lfm2BidirectionalForMaskedLM")
@ModelBase.example("LiquidAI/LFM2.5-ColBERT-350M", "LiquidAI/LFM2.5-Embedding-350M", "LiquidAI/LFM2.5-Encoder-350M", "LiquidAI/LFM2.5-Encoder-230M")
class LFM2ColBertModel(LFM2Model):
    model_arch = gguf.MODEL_ARCH.LFM2
    dense_tensor_name = "dense_2"

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if self.hf_arch in ("Lfm2BidirectionalModel", "Lfm2BidirectionalForMaskedLM"):
            self.gguf_writer.add_causal_attention(False)
        self._try_set_pooling_type()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # masked LM checkpoints use "lfm2." prefix
        name = name.removeprefix("lfm2.")
        if not name.startswith(self.dense_tensor_name):
            name = "model." + name

        yield from super().modify_tensors(data_torch, name, bid)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # optional dense tensor is stored in a separate safetensors file
        from safetensors.torch import load_file
        tensors_file = self.dir_model / "1_Dense" / "model.safetensors"
        if not tensors_file.is_file():
            return
        tensor = load_file(tensors_file)["linear.weight"]
        self.gguf_writer.add_embedding_length_out(tensor.shape[0])
        yield f"{self.dense_tensor_name}.weight", tensor.clone()


def _is_d1_omni_checkpoint(dir_model: Path) -> bool:
    if not (dir_model / "config.json").is_file():
        return False
    with open(dir_model / "config.json", encoding="utf-8") as f:
        return json.load(f).get("model_type") == "d1_omni"


@ModelBase.register_hparams_loader(_is_d1_omni_checkpoint)
def _load_d1_omni_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected d1-omni checkpoint")
    hparams = ModelBase.load_hparams(dir_model, False, guess=False)
    text = hparams["text_config"]
    n_layer, n_layer_head = text["num_hidden_layers"], hparams["head_layers"]
    # the trunk uses the LFM2 FFN sizing, the head blocks are appended with a plain 4x MLP
    n_ff = int(text["block_ffn_dim_multiplier"] * int(2 * text["intermediate_size"] / 3))
    n_ff = text["block_multiple_of"] * ((n_ff + text["block_multiple_of"] - 1) // text["block_multiple_of"])
    text["num_hidden_layers"] = n_layer + n_layer_head
    text["intermediate_size"] = [n_ff] * n_layer + [4 * text["hidden_size"]] * n_layer_head
    text["block_auto_adjust_ff_dim"] = False
    return hparams


@ModelBase.register("D1OmniModel")
@ModelBase.example("LiquidAI/d1-omni-600M")
class D1OmniModel(LFM2Model):
    model_arch = gguf.MODEL_ARCH.LFM2

    # the server cuts the text to these lengths, see server-decision.cpp
    _MAX_LENGTH = 16384
    _IMAGE_TEXT_LENGTH = 896
    _AUDIO_TEXT_LENGTH = 15360

    def set_vocab(self):
        super().set_vocab()
        # the systemone template writes the BOS, after the media
        self.gguf_writer.remove_key(gguf.Keys.Tokenizer.ADD_BOS)
        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_token_type_count(3)  # choice, score, noul
        self.gguf_writer.add_chat_template([{"name": "systemone", "template": self._systemone_template()}])

    @staticmethod
    def _systemone_template() -> str:
        # follows prompt.py of the model repo, the server cuts each marked piece to its token budget
        # the media (images, or an audio clip if audio is true) come first
        description = jinja_str_or_json("o.description")
        has_description = "o.description is not none and o.description != ''"
        yes_no = "{{ 'yes' if o.key == 'true' else 'no' }}"
        option_code = "{% if loop.index0 < 10 %}00{% elif loop.index0 < 100 %}0{% endif %}{{ loop.index0 }}"
        option = (
            "{% if type == 'choice' and audio %}option_" + option_code + ": "
            "{% if " + has_description + " %}" + description + "{% else %}{{ o.key }}{% endif %}"
            "{% elif type == 'choice' %}{{ o.key }}{% if " + has_description + " %}: " + description + "{% endif %}"
            "{% elif type == 'score' %}level {{ o.key }}: " + description
            + "{% elif audio %}{{ o.key }}: " + yes_no
            + "{% else %}{{ o.key }}: {% if " + has_description + " %}" + description
            + "{% elif images and not ns.criteria %}" + yes_no
            + "{% elif o.key == 'true' %}yes, the statement holds"
            "{% else %}no, the statement does not hold{% endif %}{% endif %}"
        )
        state = "{% if state is string %}{{ state }}{% elif state is not none %}{{ state | tojson }}{% elif audio %}{}{% endif %}"
        return (
            "{% set ns = namespace(criteria=false) %}"
            "{% for o in options %}{% if o.description is not none %}{% set ns.criteria = true %}{% endif %}{% endfor %}"
            "{% for image in images %}{{ image }}{% endfor %}{{ sep }}"
            "<|startoftext|><|reserved_7|>{{ sep }}{{ mark_state }}" + state
            + "{{ sep }}{{ mark_question }}<|reserved_8|>" + jinja_str_or_json("instructions")
            + "{% for o in options %}{{ sep }}<|reserved_9|><|mask|>{{ sep }}{{ mark_option }} " + option
            + "{{ sep }}<|reserved_10|>{% endfor %}{{ sep }}<|reserved_11|>"
        )

    def set_gguf_parameters(self):
        lengths = (self.hparams["max_length"], self.hparams["image_text_length"], self.hparams["audio_text_length"])
        if lengths != (self._MAX_LENGTH, self._IMAGE_TEXT_LENGTH, self._AUDIO_TEXT_LENGTH):
            raise ValueError(f"unexpected text lengths: {lengths}")
        n_head, n_layer_head = self.hparams["num_attention_heads"], self.hparams["head_layers"]
        self.hparams["num_key_value_heads"] = [
            self.hparams["num_key_value_heads"] if t != "conv" else 0 for t in self.hparams["layer_types"]
        ] + [n_head] * n_layer_head

        # the head needs per-layer sizes, LFM2Model writes a single feed forward length
        TextModel.set_gguf_parameters(self)
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        self.gguf_writer.add_shortconv_l_cache(self.hparams["conv_L_cache"])
        self.gguf_writer.add_layer_norm_eps(1e-5)  # nn.LayerNorm of the head
        self.gguf_writer.add_causal_attention(False)

        self.gguf_writer.add_decision_type(gguf.DecisionType.LFM2_D1_OMNI)
        self.gguf_writer.add_decision_block_count(n_layer_head)
        # "choice:3-5" -> "choice.3_5", "choice:11+" -> "choice.11"
        for name, value in self.hparams["temperatures"].items():
            self.gguf_writer.add_decision_temperature(name.replace(":", ".").replace("-", "_").rstrip("+"), value)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith(("vision.", "audio.")):
            return None

        name = name.replace("encoder.", "model.", 1) if name.startswith("encoder.") else name
        name = name.replace("head.head.layers.", "head.layers.").replace("in_proj_", "in_proj.")
        name = name.removeprefix("head.") if name.startswith(("head.type_emb", "head.scorer")) else name

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("head.layers.") and bid is not None:
            # the head blocks come after the trunk blocks
            suffix = name.split(".", 3)[3]
            bid += self.block_count - self.hparams["head_layers"]
            name = f"head.layers.{bid}.{suffix}"

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Lfm2MoeForCausalLM")
@ModelBase.example("LiquidAI/LFM2-8B-A1B")
class LFM2MoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LFM2MOE

    def set_gguf_parameters(self):
        # set num_key_value_heads only for attention layers
        self.hparams["num_key_value_heads"] = [
            self.hparams["num_key_value_heads"] if layer_type == "full_attention" else 0
            for layer_type in self.hparams["layer_types"]
        ]

        super().set_gguf_parameters()

        self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])
        self.gguf_writer.add_leading_dense_block_count(self.hparams["num_dense_layers"])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        self.gguf_writer.add_shortconv_l_cache(self.hparams["conv_L_cache"])

    # cache for experts weights for merging
    _experts_cache: dict[int, dict[str, Tensor]] = {}

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.endswith(".expert_bias"):
            name = name.replace(".expert_bias", ".expert_bias.bias")

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # conv op requires 2d tensor
        if 'conv.conv' in name:
            data_torch = data_torch.squeeze(1)

        # merge expert weights
        if 'experts' in name:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            expert_cache = self._experts_cache.setdefault(bid, {})
            expert_cache[name] = data_torch
            expert_weights = ["w1", "w2", "w3"]

            # not enough expert weights to merge
            if len(expert_cache) < n_experts * len(expert_weights):
                return

            for w_name in expert_weights:
                datas: list[Tensor] = []

                for xid in range(n_experts):
                    ename = f"model.layers.{bid}.feed_forward.experts.{xid}.{w_name}.weight"
                    datas.append(expert_cache[ename])
                    del expert_cache[ename]

                data_torch = torch.stack(datas, dim=0)
                merged_name = f"layers.{bid}.feed_forward.experts.{w_name}.weight"

                yield from super().modify_tensors(data_torch, merged_name, bid)

            del self._experts_cache[bid]
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        assert not self._experts_cache


@ModelBase.register("Lfm2VlForConditionalGeneration")
@ModelBase.example("LiquidAI/LFM2-VL-450M")
class LFM2VLModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        # TODO(tarek): for dynamic resolution image_size is not specified, setting here for compatibility
        self.hparams_vision["image_size"] = 256

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.LFM2)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.find_vparam(["layer_norm_eps"]))
        self.gguf_writer.add_vision_projector_scale_factor(self.global_config.get("downsample_factor", 2))
        self.gguf_writer.add_vision_use_gelu(True)
        # python notation, e.g. for vision_feature_layer == -1, we pick last layer -> vision_feature_layers_to_drop = 0
        vision_feature_layers_to_drop = -(self.global_config.get("vision_feature_layer", -1) + 1)
        self.gguf_writer.add_vision_block_count(self.find_vparam(self.n_block_keys) - vision_feature_layers_to_drop)
        # PIL resample enum
        if (resample := self.preprocessor_config.get("resample")) is not None:
            resize_algo = {1: "lanczos", 2: "bilinear", 3: "bicubic"}.get(resample)
            if resize_algo is None:
                raise ValueError(f"unsupported resample: {resample}")
            self.gguf_writer.add_vision_image_resize_algo(resize_algo)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        name = name.replace("model.vision_tower.", "vision_tower.")
        name = name.replace("model.multi_modal_projector.", "multi_modal_projector.")

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "patch_embedding.weight" in name:
            data_torch = data_torch.view(data_torch.shape[0], 16, 16, 3).permute(0, 3, 1, 2)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("D1OmniModel")
@ModelBase.example("LiquidAI/d1-omni-600M")
class D1OmniMmprojModel(ConformerAudioModel):
    has_vision_encoder = True
    has_audio_encoder  = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None and self.hparams_audio is not None
        # dynamic resolution, as LFM2VLModel
        self.hparams_vision["image_size"] = 256
        # the images are normalized to [-1, 1] (vision.py of the model repo)
        self.preprocessor_config = {**self.preprocessor_config, "image_mean": [0.5] * 3, "image_std": [0.5] * 3}
        self.hparams_audio["hidden_size"] = self.hparams_audio["d_model"]
        self.hparams_audio["intermediate_size"] = self.hparams_audio["d_model"] * self.hparams_audio["ff_expansion_factor"]
        self.hparams_audio["num_attention_heads"] = self.hparams_audio["n_heads"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_vision_projector_type(gguf.VisionProjectorType.D1OMNI_V)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.find_vparam(["layer_norm_eps"]))
        self.gguf_writer.add_vision_projector_scale_factor(self.global_config.get("downsample_factor", 2))
        self.gguf_writer.add_vision_use_gelu(True)

        assert self.hparams_audio is not None
        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.D1OMNI_A)
        self.gguf_writer.add_audio_num_mel_bins(self.hparams_audio["feat_in"])
        self.gguf_writer.add_audio_attention_layernorm_eps(1e-5)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith(("encoder.", "head.")):
            return None

        name = name.replace("vision.tower.", "vision_tower.").replace("vision.projector.", "multi_modal_projector.")
        name = name.replace("audio.encoder.", "conformer.")
        # the residual block continues the adapter: norm, linear, gelu, linear, then norm, down, up
        for old, new in (("adapter.norm", 0), ("adapter.linear_1", 1), ("adapter.linear_2", 3),
                         ("residual.ln", 4), ("residual.down", 5), ("residual.up", 6)):
            name = name.replace(f"audio.{old}.", f"audio_adapter.model.{new}.")

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "patch_embedding.weight" in name:
            data_torch = data_torch.view(data_torch.shape[0], 16, 16, 3).permute(0, 3, 1, 2)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Lfm2AudioForConditionalGeneration")
@ModelBase.example("LiquidAI/LFM2.5-Audio-1.5B", "LiquidAI/LFM2-Audio-1.5B")
class LFM2AudioModel(ConformerAudioModel):
    has_vision_encoder = False
    has_audio_encoder = True
    model_name = "Lfm2AudioEncoder"

    def get_audio_config(self) -> dict[str, Any] | None:
        return self.global_config.get("encoder")

    def set_gguf_parameters(self):
        assert self.hparams_audio is not None
        self.hparams_audio["hidden_size"] = self.hparams_audio["d_model"]
        self.hparams_audio["intermediate_size"] = self.hparams_audio["d_model"]
        self.hparams_audio["num_attention_heads"] = self.hparams_audio["n_heads"]
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.LFM2A)
        self.gguf_writer.add_audio_num_mel_bins(self.hparams_audio["feat_in"])
        self.gguf_writer.add_audio_attention_layernorm_eps(1e-5)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # skip language model tensors
        if name.startswith("lfm."):
            return None

        # for training only
        if any(p in name for p in ["audio_loss_weight"]):
            return None

        # for audio output
        if any(p in name for p in ["codebook_offsets", "depth_embeddings", "depth_linear", "depthformer"]):
            return None

        return super().filter_tensors(item)


@ModelBase.register("Lfm25AudioTokenizer")
@ModelBase.example("LiquidAI/LFM2.5-Audio-1.5B")
class LFM25AudioTokenizer(LFM2Model):
    model_arch = gguf.MODEL_ARCH.LFM2

    def set_vocab(self):
        self._set_vocab_none()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_embedding_length_out(self.hparams["output_size"])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # skip language model tensors
        if name == "istft.window" or name.startswith("emb.emb"):
            return None

        if name.startswith("lin"):
            name = name.replace("lin", "dense_2_out")

        return super().filter_tensors((name, gen))
