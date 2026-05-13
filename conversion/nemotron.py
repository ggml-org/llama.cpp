from __future__ import annotations

from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf, logger

from .granite import GraniteHybridModel


@ModelBase.register(
    "NemotronH_Nano_VL_V2",
    "RADIOModel",
)
class NemotronNanoV2VLModel(MmprojModel):
    # ViT-Huge architecture parameters for RADIO v2.5-h
    _vit_hidden_size = 1280
    _vit_intermediate_size = 5120
    _vit_num_layers = 32
    _vit_num_heads = 16

    def get_vision_config(self) -> dict[str, Any] | None:
        # RADIO config doesn't have standard ViT parameters, so they need to be constructed manually
        vision_config = self.global_config.get("vision_config")
        if vision_config is None:
            return None
        # Add ViT-H parameters
        vision_config = {
            **vision_config,
            "hidden_size": self._vit_hidden_size,
            "intermediate_size": self._vit_intermediate_size,
            "num_hidden_layers": self._vit_num_layers,
            "num_attention_heads": self._vit_num_heads,
            "image_size": self.global_config.get("force_image_size", 512),
        }
        return vision_config

    def set_gguf_parameters(self):
        if "image_mean" not in self.preprocessor_config:
            self.preprocessor_config["image_mean"] = [0.485, 0.456, 0.406]
        if "image_std" not in self.preprocessor_config:
            self.preprocessor_config["image_std"] = [0.229, 0.224, 0.225]

        super().set_gguf_parameters()
        hparams = self.global_config
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.NEMOTRON_V2_VL)
        self.gguf_writer.add_vision_attention_layernorm_eps(1e-6)
        self.gguf_writer.add_vision_use_gelu(True)
        downsample_ratio = hparams.get("downsample_ratio", 0.5)
        self.gguf_writer.add_vision_projector_scale_factor(int(1.0 / downsample_ratio))

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".position_embd." in new_name or "pos_embed" in new_name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if "input_conditioner" in name:
            return None

        # mtmd does not support video yet so skip tensors related to video.
        if "radio_model.model.patch_generator.video_embedder" in name:
            return None

        if not name.startswith("vision_model.radio_model.model.") and not name.startswith("mlp1."):
            return None

        if "patch_generator.pos_embed" in name:
            if not name.endswith(".weight"):
                name += ".weight"

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # RADIO's pos_embed doesn't have .weight suffix, but clip.cpp expects it
        if "patch_generator.pos_embed" in name:
            # Downsample position embeddings for fixed 512x512 image size
            import torch.nn.functional as F
            n_embd = self.hparams["hidden_size"]
            image_size = self.global_config.get("force_image_size", 512)
            patch_size = self.hparams["patch_size"]
            target_patches_per_side = image_size // patch_size  # 32
            max_patches_per_side = int((data_torch.shape[1]) ** 0.5)  # 128
            if target_patches_per_side != max_patches_per_side:
                # Reshape to grid, interpolate, flatten back
                data_torch = data_torch.reshape(1, max_patches_per_side, max_patches_per_side, n_embd)
                data_torch = data_torch.permute(0, 3, 1, 2).float()  # [1, n_embd, 128, 128]
                data_torch = F.interpolate(data_torch, size=(target_patches_per_side, target_patches_per_side),
                                           mode='bilinear', align_corners=True)
                data_torch = data_torch.permute(0, 2, 3, 1)  # [1, 32, 32, n_embd]
                data_torch = data_torch.reshape(1, target_patches_per_side * target_patches_per_side, n_embd)

        # Reshape linear patch embedding to conv2d format for ggml_conv_2d
        # From [n_embd, patch_size*patch_size*3] to [n_embd, 3, patch_size, patch_size]
        if "patch_generator.embedder" in name:
            patch_size = self.hparams["patch_size"]
            n_embd = self.hparams["hidden_size"]
            data_torch = data_torch.reshape(n_embd, 3, patch_size, patch_size)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("NemotronForCausalLM")
class NemotronModel(TextModel):
    model_arch = gguf.MODEL_ARCH.NEMOTRON

    def set_vocab(self):
        self._set_vocab_sentencepiece()
        self.gguf_writer.add_pad_token_id(0)
        self.gguf_writer.add_unk_token_id(1)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        f_norm_eps = self.find_hparam(["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon", "norm_eps"])
        self.gguf_writer.add_layer_norm_eps(f_norm_eps)

        # * Partial RoPE
        rot_pct = self.find_hparam(["partial_rotary_factor", "rope_pct", "rope_percent"])
        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        n_head = self.find_hparam(["num_attention_heads", "n_head"])
        self.gguf_writer.add_rope_dimension_count(int(rot_pct * n_embd) // n_head)

        # * RopeScaling for Nemotron
        if "rope_scaling" not in self.hparams or self.hparams["rope_scaling"] is None:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        else:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
            self.gguf_writer.add_rope_scaling_factor(self.hparams["factor"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # * Adding +1 to LayerNorm's weights here to implement layernorm1p w/o changing anything on the GGML engine side
        #   model.layers.{l}.input_layernorm.weight
        #   model.layers.{l}.post_attention_layernorm.weight
        #   model.norm.weight
        if name.endswith("norm.weight"):
            data_torch = data_torch + 1

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("NemotronHForCausalLM")
class NemotronHModel(GraniteHybridModel):
    """Hybrid mamba2/attention model from NVIDIA"""
    model_arch = gguf.MODEL_ARCH.NEMOTRON_H
    is_moe: bool = False

    def __init__(self, *args, **kwargs):
        # We have to determine the correct model architecture (MoE vs non-MoE) before
        # calling the parent __init__. This is because the parent constructor
        # uses self.model_arch to build the tensor name map, and all MoE-specific
        # mappings would be missed if it were called with the default non-MoE arch.
        hparams = ModelBase.load_hparams(args[0], self.is_mistral_format)
        has_moe_params = (
            "num_experts_per_tok" in hparams
            or (isinstance(hparams.get("llm_config"), dict) and "num_experts_per_tok" in hparams["llm_config"])
        )
        if has_moe_params:
            self.model_arch = gguf.MODEL_ARCH.NEMOTRON_H_MOE
            self.is_moe = True

        super().__init__(*args, **kwargs)

        # Save the top-level head_dim for later
        self.head_dim = self.hparams.get("head_dim", self.hparams.get("attention_head_dim"))
        assert self.head_dim is not None, "Could not find the attention head dim in config"

        # Don't use expand to calculate d_inner
        self.d_inner = self.find_hparam(["num_heads"]) * self.d_model

        # Update the ssm / attn / mlp layers
        # M: Mamba2, *: Attention, -: MLP
        # MoE:
        # M: Mamba2, *: Attention, E: Expert
        pattern = self.hparams.get("hybrid_override_pattern") or self.hparams.get("layers_block_type")
        if pattern is None:
            self._ssm_layers = []
            self._mlp_layers = []
        elif isinstance(pattern, str):
            self._ssm_layers = [i for i, val in enumerate(pattern) if val == "M"]
            self._mlp_layers = [i for i, val in enumerate(pattern) if val == ("E" if self.is_moe else "-")]
        else:
            self._ssm_layers = [i for i, val in enumerate(pattern) if val == "mamba"]
            self._mlp_layers = [i for i, val in enumerate(pattern) if val == "moe"]

    def get_attn_layers(self):
        pattern = self.hparams.get("hybrid_override_pattern") or self.hparams.get("layers_block_type")
        if pattern is None:
            return []
        assert len(pattern) == self.block_count, f"Mismatch between pattern ({len(pattern)}) and block_count ({self.block_count})!"
        if isinstance(pattern, str):
            return [i for i, val in enumerate(pattern) if val == "*"]

        return [i for i, val in enumerate(pattern) if val == "attention"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        head_dim = self.head_dim
        if head_dim is None:
            raise ValueError("Could not find the attention head dim in config")
        self.gguf_writer.add_key_length(head_dim)
        self.gguf_writer.add_value_length(head_dim)

        # Set feed_forward_length
        # NOTE: This will trigger an override warning. This is preferable to
        #   duplicating all the parent logic
        if not self.is_moe:
            n_ff = self.find_hparam(["intermediate_size", "n_inner", "hidden_dim"])
            self.gguf_writer.add_feed_forward_length([
                n_ff if i in self._mlp_layers else 0 for i in range(self.block_count)
            ])
        else:
            moe_intermediate_size = self.hparams["moe_intermediate_size"]
            self.gguf_writer.add_feed_forward_length([
                moe_intermediate_size if i in self._mlp_layers else 0 for i in range(self.block_count)
            ])
            self.gguf_writer.add_expert_used_count(self.hparams["num_experts_per_tok"])
            self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])
            self.gguf_writer.add_expert_shared_feed_forward_length(self.hparams["moe_shared_expert_intermediate_size"])
            self.gguf_writer.add_expert_count(self.hparams["n_routed_experts"])
            self.gguf_writer.add_expert_shared_count(self.hparams["n_shared_experts"])
            self.gguf_writer.add_expert_weights_norm(self.hparams["norm_topk_prob"])
            self.gguf_writer.add_expert_weights_scale(self.hparams["routed_scaling_factor"])
            self.gguf_writer.add_expert_group_count(self.hparams["n_group"])

            # number of experts used per token (top-k)
            if (n_experts_used := self.hparams.get("num_experts_per_tok")) is not None:
                self.gguf_writer.add_expert_used_count(n_experts_used)

            if (latent_size := self.hparams.get("moe_latent_size")) is not None:
                self.gguf_writer.add_moe_latent_size(latent_size)

    def set_vocab(self):
        # The NemotronH config uses pattern characters (e.g. '-') that may not
        # be supported by the installed transformers version. AutoTokenizer
        # internally calls AutoConfig which triggers this parsing failure.
        # Using trust_remote_code=True to load the model's own config class.
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)

        # Pad vocab size (from Mamba2Model/GraniteHybridModel)
        self.hparams["pad_vocab_size_multiple"] = 8 # Setting this here since GraniteHybridModel.set_vocab() isn't being invoked now.
        # From Mamba2Model.set_vocab():
        vocab_size = self.hparams["vocab_size"]
        pad_vocab = self.hparams.get("pad_vocab_size_multiple", 16)
        # ref: https://stackoverflow.com/a/17511341/22827863
        vocab_size = -(vocab_size // -pad_vocab) * pad_vocab
        self.hparams["vocab_size"] = vocab_size

        assert max(tokenizer.vocab.values()) < vocab_size  # ty: ignore[unresolved-attribute]

        tokpre = self.get_vocab_base_pre(tokenizer)

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}  # ty: ignore[unresolved-attribute]
        added_vocab = tokenizer.get_added_vocab()  # ty: ignore[unresolved-attribute]

        added_tokens_decoder = tokenizer.added_tokens_decoder  # ty: ignore[unresolved-attribute]

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token: str = reverse_vocab[i]
                if token in added_vocab:
                    if not added_tokens_decoder[i].normalized:
                        previous_token = token
                        token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))  # ty: ignore[unresolved-attribute, invalid-assignment]
                        if previous_token != token:
                            logger.info(f"{repr(previous_token)} is encoded and decoded back to {repr(token)} using AutoTokenizer")

                    if added_tokens_decoder[i].special or self.does_token_look_special(token):
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        token = token.replace(b"\xe2\x96\x81".decode("utf-8"), " ")  # pre-normalize user-defined spaces
                        toktypes.append(gguf.TokenType.USER_DEFINED)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)
                tokens.append(token)

        # From TextModel.set_vocab_gpt2():
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

        # The tokenizer _does_ add a BOS token (via post_processor type
        # TemplateProcessing) but does not set add_bos_token to true in the
        # config, so we need to explicitly override it here.
        if not self.is_moe:
            self.gguf_writer.add_add_bos_token(True)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if self.is_moe and bid is not None:
            # Skip Multi-Token Prediction (MTP) tensors. These are used for
            # for speculative decoding but we don't include them in this model
            # conversion. See https://github.com/ggml-org/llama.cpp/pull/18886
            if name.startswith("mtp."):
                logger.info(f"gguf: Skipping MTP (Speculative) layer: {name}")
                return

            if name.endswith("mixer.gate.e_score_correction.bias"):
                yield from ModelBase.modify_tensors(self, data_torch, name, bid)
                return

            if name.endswith("mixer.dt_bias"):
                new_name = name.replace("dt_bias", "dt.bias")
                yield from ModelBase.modify_tensors(self, data_torch, new_name, bid)
                return

            if name.endswith("mixer.conv1d.weight"):
                squeezed_data = data_torch.squeeze()
                yield from ModelBase.modify_tensors(self, squeezed_data, name, bid)
                return

            if name.endswith("mixer.A_log"):
                transformed_data = -torch.exp(data_torch)
                reshaped_data = transformed_data.squeeze().reshape(-1, 1)
                yield from ModelBase.modify_tensors(self, reshaped_data, name, bid)
                return

            if name.endswith("mixer.D"):
                reshaped_data = data_torch.squeeze().reshape(-1, 1)
                yield from ModelBase.modify_tensors(self, reshaped_data, name, bid)
                return

            if name.endswith("mixer.norm.weight"):
                reshaped_data = data_torch.reshape(self.n_group, -1)
                yield from ModelBase.modify_tensors(self, reshaped_data, name, bid)
                return

            if name.find("mixer.experts") != -1:
                n_experts = self.hparams["n_routed_experts"]
                assert bid is not None

                if self._experts is None:
                    self._experts = [{} for _ in range(self.block_count)]

                self._experts[bid][name] = data_torch

                if len(self._experts[bid]) >= n_experts * 2:
                    # merge the experts into a single tensor
                    for w_name in ["down_proj", "up_proj"]:
                        datas: list[Tensor] = []

                        for xid in range(n_experts):
                            ename = f"backbone.layers.{bid}.mixer.experts.{xid}.{w_name}.weight"
                            datas.append(self._experts[bid][ename])
                            del self._experts[bid][ename]

                        data_torch = torch.stack(datas, dim=0)
                        merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                        yield from ModelBase.modify_tensors(self, data_torch, merged_name, bid)
                    return
                else:
                    return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
