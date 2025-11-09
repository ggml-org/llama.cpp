from __future__ import annotations
import json
from .base import (
    ModelBase, TextModel, MmprojModel, gguf, torch, logger
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("GemmaForCausalLM")
class GemmaModel(TextModel):
    model_arch = gguf.MODEL_ARCH.GEMMA

    def set_vocab(self):
        self._set_vocab_sentencepiece()
        # TODO: these special tokens should be exported only for the CodeGemma family
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False,
                                          special_token_types = ['prefix', 'suffix', 'middle', 'fsep', 'eot'])
        special_vocab._set_special_token("prefix", 67)
        special_vocab._set_special_token("suffix", 69)
        special_vocab._set_special_token("middle", 68)
        special_vocab._set_special_token("fsep",   70)
        special_vocab._set_special_token("eot",    107)
        special_vocab.chat_template = None  # do not add it twice
        special_vocab.add_to_gguf(self.gguf_writer)
        self.gguf_writer.add_add_space_prefix(False)

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]
        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"] if "num_key_value_heads" in hparams else hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_key_length(hparams["head_dim"])
        self.gguf_writer.add_value_length(hparams["head_dim"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        # lm_head is not used in llama.cpp, while autoawq will include this tensor in model
        # To prevent errors, skip loading lm_head.weight.
        if name == "lm_head.weight":
            logger.debug(f"Skipping get tensor {name!r} in safetensors so that convert can end normally.")
            return []
        # ref: https://github.com/huggingface/transformers/blob/fc37f38915372c15992b540dfcbbe00a916d4fc6/src/transformers/models/gemma/modeling_gemma.py#L89
        if name.endswith("norm.weight"):
            data_torch = data_torch + 1
        return [(self.map_tensor_name(name), data_torch)]


@ModelBase.register("Gemma2ForCausalLM")
class Gemma2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.GEMMA2

    def set_vocab(self):
        self._set_vocab_sentencepiece()
        self.gguf_writer.add_add_space_prefix(False)

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]
        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"] if "num_key_value_heads" in hparams else hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_key_length(hparams["head_dim"])
        self.gguf_writer.add_value_length(hparams["head_dim"])
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_attn_logit_softcapping(
            self.hparams["attn_logit_softcapping"]
        )
        self.gguf_writer.add_final_logit_softcapping(
            self.hparams["final_logit_softcapping"]
        )
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        # lm_head is not used in llama.cpp, while autoawq will include this tensor in model
        # To prevent errors, skip loading lm_head.weight.
        if name == "lm_head.weight":
            logger.debug(f"Skipping get tensor {name!r} in safetensors so that convert can end normally.")
            return []
        # ref: https://github.com/huggingface/transformers/blob/fc37f38915372c15992b540dfcbbe00a916d4fc6/src/transformers/models/gemma/modeling_gemma.py#L89
        if name.endswith("norm.weight"):
            data_torch = data_torch + 1
        return [(self.map_tensor_name(name), data_torch)]


@ModelBase.register("Gemma3ForCausalLM", "Gemma3ForConditionalGeneration")
class Gemma3Model(TextModel):
    model_arch = gguf.MODEL_ARCH.GEMMA3
    norm_shift = 1.0  # Gemma3RMSNorm adds 1.0 to the norm value

    def set_vocab(self):
        self._set_vocab_sentencepiece()
        self.gguf_writer.add_add_space_prefix(False)

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]
        # some default values are not specified in the hparams
        self.gguf_writer.add_context_length(hparams.get("max_position_embeddings", 131072))
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_head_count(hparams.get("num_attention_heads", 8))
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams.get("rms_norm_eps", 1e-6))
        self.gguf_writer.add_key_length(hparams.get("head_dim", 256))
        self.gguf_writer.add_value_length(hparams.get("head_dim", 256))
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_rope_freq_base(hparams.get("rope_theta", 1_000_000.0)) # for global layers
        # attn_logit_softcapping is removed in Gemma3
        assert hparams.get("attn_logit_softcapping") is None
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_head_count_kv(hparams.get("num_key_value_heads", 4))
        if hparams.get("rope_scaling") is not None:
            assert hparams["rope_scaling"]["rope_type"] == "linear"
            # important: this rope_scaling is only applied for global layers, and not used by 1B model
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
            self.gguf_writer.add_rope_scaling_factor(hparams["rope_scaling"]["factor"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        if "language_model." in name:
            name = name.replace("language_model.", "")
        elif name.startswith("multi_modal_projector.") or name.startswith("vision_tower.") \
                or name.startswith("multimodal_projector.") or name.startswith("vision_model."):
            return [] # skip vision tensors
        # remove OOV (out-of-vocabulary) rows in token_embd
        if "embed_tokens.weight" in name:
            vocab = self._create_vocab_sentencepiece()
            tokens = vocab[0]
            data_torch = data_torch[:len(tokens)]
        # ref code in Gemma3RMSNorm
        # output = output * (1.0 + self.weight.float())
        # note: this is not the case on gemma3n
        if name.endswith("norm.weight"):
            data_torch = data_torch + self.norm_shift
        return [(self.map_tensor_name(name), data_torch)]


@ModelBase.register("Gemma3TextModel")
class EmbeddingGemma(Gemma3Model):
    model_arch = gguf.MODEL_ARCH.GEMMA_EMBEDDING
    module_paths = []
    dense_features_dims = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.sentence_transformers_dense_modules:
            # read modules.json to determine if model has Dense layers
            modules_file = self.dir_model / "modules.json"
            if modules_file.is_file():
                with open(modules_file, encoding="utf-8") as modules_json_file:
                    mods = json.load(modules_json_file)
                for mod in mods:
                    if mod["type"] == "sentence_transformers.models.Dense":
                        mod_path = mod["path"]
                        # check if model.safetensors file for Dense layer exists
                        model_tensors_file = self.dir_model / mod_path / "model.safetensors"
                        if model_tensors_file.is_file():
                            self.module_paths.append(mod_path)
                            # read config.json of the Dense layer to get in/out features
                            mod_conf_file = self.dir_model / mod_path / "config.json"
                            if mod_conf_file.is_file():
                                with open(mod_conf_file, encoding="utf-8") as mod_conf_json_file:
                                    mod_conf = json.load(mod_conf_json_file)
                                    # hparams dense_2_feat_out and dense_3_feat_in are required when loading model's dense weights
                                    prefix = self._get_dense_prefix(mod_path)
                                    if mod_conf["in_features"] is not None and mod_conf["out_features"] is not None:
                                        self.dense_features_dims[prefix] = (mod_conf["in_features"], mod_conf["out_features"])

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        from safetensors.torch import load_file
        module_paths = list(self.module_paths)
        for i, module_path in enumerate(module_paths):
            tensors_file = self.dir_model / module_path / "model.safetensors"
            local_tensors = load_file(tensors_file)
            tensor_name = self._get_dense_prefix(module_path)
            for name, local_tensor in local_tensors.items():
                if not name.endswith(".weight"):
                    continue
                orig_name = name.replace("linear", tensor_name)
                name = self.map_tensor_name(orig_name)
                yield name, local_tensor.clone()

    @staticmethod
    def _get_dense_prefix(module_path) -> str:
        """Get the tensor name prefix for the Dense layer from module path."""
        tensor_name = "dense_2" if module_path == "2_Dense" else "dense_3"
        return tensor_name

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # Override the sliding window size as it gets adjusted by the Gemma3TextConfig
        # constructor. We want to use the value from the original model's config.json.
        # ref: https://github.com/huggingface/transformers/pull/40700
        with open(self.dir_model / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            orig_sliding_window = config.get("sliding_window")
            if orig_sliding_window is None:
                raise ValueError("sliding_window not found in model config - this is required for the model")
            logger.info(f"Using original sliding_window from config: {orig_sliding_window} "
                        f"instead of {self.hparams['sliding_window']}")
            self.gguf_writer.add_sliding_window(orig_sliding_window)
        if self.sentence_transformers_dense_modules:
            for dense, dims in self.dense_features_dims.items():
                logger.info(f"Setting dense layer {dense} in/out features to {dims}")
                self.gguf_writer.add_dense_features_dims(dense, dims[0], dims[1])
        self._try_set_pooling_type()


@ModelBase.register("Gemma3ForConditionalGeneration")
class Gemma3VisionModel(MmprojModel):
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.GEMMA3)
        # default values below are taken from HF tranformers code
        self.gguf_writer.add_vision_attention_layernorm_eps(hparams.get("layer_norm_eps", 1e-6))
        self.gguf_writer.add_vision_use_gelu(True)
        # calculate proj_scale_factor (used by tinygemma3 test model)
        image_seq_length = self.preprocessor_config.get("image_seq_length", 256)
        n_per_side = int(image_seq_length ** 0.5)
        image_size = self.hparams["image_size"]
        patch_size = self.hparams["patch_size"]
        proj_scale_factor = (image_size // patch_size) // n_per_side
        if proj_scale_factor > 0 and proj_scale_factor != 4:
            # we only need to write this if it's not the default value
            # in this case, we are converting a test model
            self.gguf_writer.add_vision_projector_scale_factor(proj_scale_factor)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # related to https://github.com/ggml-org/llama.cpp/issues/13025
        if "input_projection" in name:
            return gguf.GGMLQuantizationType.F16
        if ".embeddings." in name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        if "vision_model.head." in name:
            return [] # skip redundant tensors for tinygemma3
        if name.startswith("multi_modal_projector.") or name.startswith("vision_tower.") \
                or name.startswith("multimodal_projector.") or name.startswith("vision_model."):
            # process vision tensors
            name = name.replace("_weight", ".weight")
            # correct norm value ; only this "soft_emb_norm" need to be corrected as it's part of Gemma projector
            # the other norm values are part of SigLIP model, and they are already correct
            # ref code: Gemma3RMSNorm
            if "soft_emb_norm.weight" in name:
                logger.info(f"Correcting norm value for '{name}'")
                data_torch = data_torch + 1
            return [(self.map_tensor_name(name), data_torch)]
        return [] # skip other tensors


@ModelBase.register("Gemma3nForConditionalGeneration")
class Gemma3NModel(Gemma3Model):
    model_arch = gguf.MODEL_ARCH.GEMMA3N
    norm_shift = 0.0 # same value with Gemma3p5RMSNorm scale_shift on python code
    _altup_proj: list[Tensor] = []
    _altup_unembd: list[Tensor] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams["altup_num_inputs"] == 4, "Current conversion only supports 4 altup inputs"
        self._altup_proj = [
            torch.Tensor(), # to be replaced
            torch.Tensor(), # to be replaced
            torch.Tensor(), # to be replaced
        ]
        self._altup_unembd = [
            torch.Tensor(), # to be replaced
            torch.Tensor(), # to be replaced
            torch.Tensor(), # to be replaced
        ]

    def set_vocab(self):
        super().set_vocab()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_altup_active_idx(self.hparams["altup_active_idx"])
        self.gguf_writer.add_altup_num_inputs(self.hparams["altup_num_inputs"])
        self.gguf_writer.add_embedding_length_per_layer_input(self.hparams["hidden_size_per_layer_input"])
        self.gguf_writer.add_shared_kv_layers(self.hparams["num_kv_shared_layers"])
        activation_sparsity_scale = []
        for s in self.hparams["activation_sparsity_pattern"]:
            normal_dist = torch.distributions.normal.Normal(0, 1)
            std_multiplier = normal_dist.icdf(torch.tensor(s, dtype=torch.float32))
            activation_sparsity_scale.append(std_multiplier.item())
        self.gguf_writer.add_activation_sparsity_scale(activation_sparsity_scale)
        sliding_window_pattern = []
        for t in self.hparams["layer_types"]:
            sliding_window_pattern.append(t == "sliding_attention")
        self.gguf_writer.add_sliding_window_pattern(sliding_window_pattern)

    def _stack_matrices(self, matrices: list[Tensor]) -> Tensor | None:
        has_all = all(m.numel() > 0 for m in matrices)
        if not has_all:
            return None
        else:
            return torch.stack(matrices, dim=0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith("_scale"):
            name = name + ".weight"
        # TODO: implement self.prediction_coefs.weight.clamp_(...)
        if "language_model." not in name:
            return [] # skip non-language model tensors
        if "altup_unembed_projections" in name:
            data_torch = data_torch.to(device="cpu")
            if ".0." in name:
                self._altup_unembd[0] = data_torch
            elif ".1." in name:
                self._altup_unembd[1] = data_torch
            elif ".2." in name:
                self._altup_unembd[2] = data_torch
            else:
                raise ValueError(f"Unknown name: {name}")
            out = self._stack_matrices(self._altup_unembd)
            if out is not None:
                return [(self.map_tensor_name("model.altup_unembed_projections.weight"), out)]
            else:
                return []
        if "altup_projections" in name:
            data_torch = data_torch.to(device="cpu")
            if ".0." in name:
                self._altup_proj[0] = data_torch
            elif ".1." in name:
                self._altup_proj[1] = data_torch
            elif ".2." in name:
                self._altup_proj[2] = data_torch
            else:
                raise ValueError(f"Unknown name: {name}")
            out = self._stack_matrices(self._altup_proj)
            if out is not None:
                return [(self.map_tensor_name("model.altup_projections.weight"), out)]
            else:
                return []
        return super().modify_tensors(data_torch, name, bid)
