from __future__ import annotations

import json

from typing import Any, Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf, logger

from .qwen import Qwen3Model, Qwen3MoeModel
from .qwenvl import Qwen25AudioModel


@ModelBase.register("Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration", "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration")
class Qwen3VLVisionModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.hparams_vision is None:
            logger.info("No vision config found, skipping vision tensor processing")
            return

        # Compute image_size if not present
        if "image_size" not in self.hparams_vision:
            # For Qwen3VL/Qwen3VLMoe, compute from num_position_embeddings
            num_pos = self.hparams_vision.get("num_position_embeddings", 2304)
            patch_size = self.hparams_vision.get("patch_size", 16)
            # num_position_embeddings = (image_size / patch_size) ** 2
            # So image_size = sqrt(num_position_embeddings) * patch_size
            image_size = int(num_pos**0.5 * patch_size)
            self.hparams_vision["image_size"] = image_size

        # Rename config values for compatibility
        self.hparams_vision["num_attention_heads"] = self.hparams_vision.get("num_heads")
        self.hparams_vision["num_hidden_layers"] = self.hparams_vision.get("depth")

        self.is_deepstack_layers = [False] * int(self.hparams_vision["num_hidden_layers"] or 0)
        for idx in self.hparams_vision.get("deepstack_visual_indexes", []):
            self.is_deepstack_layers[idx] = True

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # in case mixed modalities, the arch will be handled by subclass
        if not self.has_audio_encoder:
            self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.QWEN3VL)
        self.gguf_writer.add_vision_use_gelu(True)

        if self.hparams_vision is not None:
            merge_size = self.hparams_vision.get("spatial_merge_size")
            if merge_size is not None:
                self.gguf_writer.add_vision_spatial_merge_size(int(merge_size))

        # Use text config's rms_norm_eps for vision attention layernorm eps
        rms_norm_eps = self.global_config.get("text_config", {}).get("rms_norm_eps", 1e-6)
        self.gguf_writer.add_vision_attention_layernorm_eps(rms_norm_eps)

        if self.is_deepstack_layers:
            self.gguf_writer.add_vision_is_deepstack_layers(self.is_deepstack_layers)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # Skip text model tensors
        if name.startswith("lm_head."):
            return None

        # Skip MTP tensors
        if name.startswith("mtp."):
            return None

        if name.startswith("model.visual."):
            name = name.replace("model.visual.", "visual.", 1)

        if not name.startswith("visual."):
            return None

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        assert self.hparams_vision is not None

        if name.startswith("visual.deepstack_merger_list."):
            prefix, rest = name.split(".", maxsplit=3)[2:]
            # prefix is the layer index, convert to absolute clip layer index!
            idx = self.hparams_vision.get("deepstack_visual_indexes", [])[int(prefix)]
            target = rest

            tensor_type: gguf.MODEL_TENSOR
            if target.startswith("norm."):
                tensor_type = gguf.MODEL_TENSOR.V_DS_NORM
                suffix = target.split(".", 1)[1]
            elif target.startswith("linear_fc1."):
                tensor_type = gguf.MODEL_TENSOR.V_DS_FC1
                suffix = target.split(".", 1)[1]
            elif target.startswith("linear_fc2."):
                tensor_type = gguf.MODEL_TENSOR.V_DS_FC2
                suffix = target.split(".", 1)[1]
            else:
                raise ValueError(f"Unexpected deepstack tensor: {name}")

            new_name = self.format_tensor_name(tensor_type, idx, suffix=f".{suffix}")
            yield from super().modify_tensors(data_torch, new_name, bid)
            return

        if name.startswith("visual.merger."):
            suffix = name.split(".", 2)[2]
            if suffix.startswith("linear_fc"):
                fc_idx_str, tail = suffix.split(".", 1)
                fc_num = int(fc_idx_str.replace("linear_fc", ""))
                # Qwen3VL has linear_fc1 and linear_fc2
                # Map to indices 0 and 2 (matching Qwen2VL which uses indices 0 and 2)
                if fc_num == 1:
                    fc_idx = 0
                elif fc_num == 2:
                    fc_idx = 2
                else:
                    raise ValueError(f"unexpected fc index {fc_num} in {name}")
                new_name = self.format_tensor_name(gguf.MODEL_TENSOR.V_MMPROJ, fc_idx, suffix=f".{tail}")
            elif suffix.startswith("norm."):
                new_name = self.format_tensor_name(gguf.MODEL_TENSOR.V_POST_NORM, suffix=f".{suffix.split('.', 1)[1]}")
            else:
                raise ValueError(f"Unexpected merger tensor: {name}")
            yield (new_name, data_torch)
            return

        if name == "visual.patch_embed.proj.weight":
            # split Conv3D into Conv2Ds along temporal dimension
            c1, c2, kt, _, _ = data_torch.shape
            del c1, c2
            if kt != 2:
                raise ValueError("Current implementation only supports temporal_patch_size of 2")
            yield (gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH] + ".weight", data_torch[:, :, 0, ...])
            yield (gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH] + ".weight.1", data_torch[:, :, 1, ...])
            return

        if name == "visual.patch_embed.proj.bias":
            # Include the bias - it's used by the C++ code
            yield (gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH] + ".bias", data_torch)
            return

        yield from MmprojModel.modify_tensors(self, data_torch, name, bid)


@ModelBase.register("Qwen3OmniMoeForConditionalGeneration")
class Qwen3OmniMmprojModel(Qwen3VLVisionModel, Qwen25AudioModel):
    has_audio_encoder = True
    has_vision_encoder = True

    def get_vision_config(self) -> dict[str, Any] | None:
        if self.has_vision_encoder:
            return self.global_config["thinker_config"].get("vision_config")
        else:
            return None

    def get_audio_config(self) -> dict[str, Any] | None:
        if self.has_audio_encoder:
            return self.global_config["thinker_config"].get("audio_config")
        else:
            return None

    def set_gguf_parameters(self):
        if self.has_vision_encoder:
            Qwen3VLVisionModel.set_gguf_parameters(self)
            self.gguf_writer.add_clip_vision_projector_type(gguf.VisionProjectorType.QWEN3VL)
        if self.has_audio_encoder:
            Qwen25AudioModel.set_gguf_parameters(self)
            self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.QWEN3A)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # Skip text model tensors
        if name.startswith("lm_head."):
            return None

        # Skip MTP tensors
        if name.startswith("mtp."):
            return None

        if name.startswith("model.visual."):
            name = name.replace("model.visual.", "visual.", 1)

        if "visual." not in name and "audio_tower." not in name:
            return None

        return MmprojModel.filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "visual." in name:
            if not self.has_vision_encoder:
                raise ValueError(f"Model does not have vision encoder, but found tensor {name}")
            # need to transform vision tensor naming, so that modify_tensors() logic can be used correctly
            name = name.replace("thinker.visual.", "model.visual.")
            if ".merger_list." in name:
                name = name.replace(".merger_list.", ".deepstack_merger_list.")
                name = name.replace(".ln_q", ".norm")
                name = name.replace(".mlp.0", ".linear_fc1")
                name = name.replace(".mlp.2", ".linear_fc2")
            elif ".merger." in name:
                name = name.replace(".ln_q", ".norm")
                name = name.replace(".mlp.0", ".linear_fc1")
                name = name.replace(".mlp.2", ".linear_fc2")
            yield from Qwen3VLVisionModel.modify_tensors(self, data_torch, name, bid)
        elif "audio_tower." in name:
            if not self.has_audio_encoder:
                raise ValueError(f"Model does not have audio encoder, but found tensor {name}")
            if "conv2d" in name and name.endswith(".bias"):
                # transform conv2d bias [n_embd] --> [1, 1, n_embd]
                data_torch = data_torch.unsqueeze(-1).unsqueeze(-1)
            yield from Qwen25AudioModel.modify_tensors(self, data_torch, name, bid)


@ModelBase.register("Qwen3ASRForConditionalGeneration")
class Qwen3ASRMmprojModel(Qwen3OmniMmprojModel):
    has_audio_encoder = True
    has_vision_encoder = False


@ModelBase.register("Glm4vForConditionalGeneration", "Glm4vMoeForConditionalGeneration", "GlmOcrForConditionalGeneration")
class Glm4VVisionModel(Qwen3VLVisionModel):
    def set_gguf_parameters(self):
        MmprojModel.set_gguf_parameters(self) # skip Qwen3VLVisionModel parameters
        assert self.hparams_vision is not None
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.GLM4V)

        hidden_act = str(self.hparams_vision.get("hidden_act", "")).lower()
        if hidden_act == "gelu":
            self.gguf_writer.add_vision_use_gelu(True)
        elif hidden_act == "silu":
            self.gguf_writer.add_vision_use_silu(True)

        rms_norm_eps = self.hparams_vision.get("rms_norm_eps", 1e-5)
        self.gguf_writer.add_vision_attention_layernorm_eps(rms_norm_eps)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("visual.merger."):
            yield from ModelBase.modify_tensors(self, data_torch, name, bid)
            return
        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Qwen3VLForConditionalGeneration")
class Qwen3VLTextModel(Qwen3Model):
    model_arch = gguf.MODEL_ARCH.QWEN3VL

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if "thinker_config" in self.hparams:
            vision_config = self.hparams["thinker_config"].get("vision_config", {})
        else:
            vision_config = self.hparams.get("vision_config", {})
        deepstack_layer_num = len(vision_config.get("deepstack_visual_indexes", []))
        self.gguf_writer.add_num_deepstack_layers(deepstack_layer_num)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        name = name.replace("thinker.", "")

        return super().filter_tensors((name, gen))


@ModelBase.register("Qwen3VLMoeForConditionalGeneration")
class Qwen3VLMoeTextModel(Qwen3MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN3VLMOE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        vision_config = self.hparams.get("vision_config", {})
        deepstack_layer_num = len(vision_config.get("deepstack_visual_indexes", []))
        self.gguf_writer.add_num_deepstack_layers(deepstack_layer_num)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        name = name.replace("thinker.", "")

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Qwen3VL has transposed packed tensors, so we treat it differently from general Qwen2MoE packed tensors
        if name.endswith("mlp.experts.down_proj") or name.endswith("mlp.experts.down_proj.weight"):
            mapped = f"{name}.weight" if not name.endswith(".weight") else name
            permuted = data_torch.permute(0, 2, 1).contiguous()
            yield from ModelBase.modify_tensors(self, permuted, mapped, bid)
            return

        if name.endswith("mlp.experts.gate_up_proj") or name.endswith("mlp.experts.gate_up_proj.weight"):
            if data_torch.ndim < 3 or data_torch.shape[-1] % 2 != 0:
                raise ValueError(f"Unexpected gate_up_proj shape for {name}: {tuple(data_torch.shape)}")
            split_dim = data_torch.shape[-1] // 2
            gate = data_torch[..., :split_dim].contiguous()
            up = data_torch[..., split_dim:].contiguous()
            # Input gate/up: (n_expert=128, n_embd=2048, n_ff_exp=768)
            # Want GGML ne: {n_embd, n_ff_exp, n_expert} = {2048, 768, 128}
            # Need PyTorch: (128, 768, 2048) [reversed of GGML]
            # So: permute(0, 2, 1): (128, 2048, 768) -> (128, 768, 2048)
            base_name = name.removesuffix(".weight")
            base = base_name.rsplit('.', 1)[0]
            mapped_gate = f"{base}.gate_proj.weight"
            mapped_up = f"{base}.up_proj.weight"
            perm_gate = gate.permute(0, 2, 1).contiguous()
            perm_up = up.permute(0, 2, 1).contiguous()
            yield from ModelBase.modify_tensors(self, perm_gate, mapped_gate, bid)
            yield from ModelBase.modify_tensors(self, perm_up, mapped_up, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Qwen3OmniMoeForConditionalGeneration")
class Qwen3OmniMoeTextModel(Qwen3VLMoeTextModel):
    model_arch = gguf.MODEL_ARCH.QWEN3VLMOE

    def set_vocab(self):
        super().set_vocab()
        # correct BOS/EOS tokens
        with open(self.dir_model / "tokenizer_config.json", "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
            added_tokens = tokenizer_config.get("added_tokens_decoder", {})
            for token_id, data in added_tokens.items():
                if data.get("content") == "<|im_end|>":
                    self.gguf_writer.add_bos_token_id(int(token_id))
                    self.gguf_writer.add_eos_token_id(int(token_id))
                    break

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_num_deepstack_layers(0)


@ModelBase.register("Qwen3ASRForConditionalGeneration")
class Qwen3ASRTextModel(Qwen3VLTextModel):
    model_arch = gguf.MODEL_ARCH.QWEN3VL

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_num_deepstack_layers(0)

    def set_vocab(self):
        super().set_vocab()
        # fix chat template, use correct chatml format
        self.gguf_writer.add_chat_template("{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
        # correct BOS/EOS tokens
        with open(self.dir_model / "tokenizer_config.json", "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
            added_tokens = tokenizer_config.get("added_tokens_decoder", {})
            for token_id, data in added_tokens.items():
                if data.get("content") == "<|im_end|>":
                    self.gguf_writer.add_bos_token_id(int(token_id))
                    self.gguf_writer.add_eos_token_id(int(token_id))
                    break
