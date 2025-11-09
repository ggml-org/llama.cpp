from __future__ import annotations
from .base import (
    ModelBase, MmprojModel, gguf, logger
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor
from .llama import LlamaModel
import json


@ModelBase.register("LlavaForConditionalGeneration", "Mistral3ForConditionalGeneration", "LightOnOCRForConditionalGeneration", "CogVLMForCausalLM", "JanusForConditionalGeneration")
class LlavaVisionModel(MmprojModel):
    img_break_tok_id = -1
    use_break_tok = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.hparams.get("model_type") == "pixtral":
            # layer_norm_eps is not in config.json, it is hard-coded in modeling_pixtral.py
            self.hparams["layer_norm_eps"] = self.hparams.get("layer_norm_eps", 1e-5)
            if self.use_break_tok:
                self.img_break_tok_id = self.get_token_id("[IMG_BREAK]")
        elif self.is_mistral_format:
            # hparams is already vision config here so norm_eps is only defined in global_config.
            self.hparams["norm_eps"] = self.global_config.get("norm_eps", None)
            assert self.hparams["norm_eps"] is not None, "norm_eps not found in params.json"
            if self.use_break_tok:
                self.img_break_tok_id = self.find_vparam(["image_break_token_id"])
        else:
            raise ValueError(f"Unsupported model type: {self.hparams['model_type']}")
        logger.info(f"Image break token id: {self.img_break_tok_id}")

    def get_token_id(self, token: str) -> int:
        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        with open(tokenizer_config_file, "r", encoding="utf-8") as f:
            added_tokens_decoder = json.load(f)['added_tokens_decoder']
            for id_, token_data in added_tokens_decoder.items():
                if token_data["content"] == token:
                    return int(id_)
        raise ValueError(f"Token '{token}' not found in tokenizer config.")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        if hparams.get("model_type") == "pixtral":
            self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.PIXTRAL)
            self.gguf_writer.add_vision_attention_layernorm_eps(hparams["layer_norm_eps"])
            # hidden_act
            if hparams["hidden_act"] == "silu":
                self.gguf_writer.add_vision_use_silu(True)
            elif hparams["hidden_act"] == "gelu":
                self.gguf_writer.add_vision_use_gelu(True)
            else:
                raise ValueError(f"Unsupported hidden_act: {hparams['hidden_act']}")
            # spatial_merge_size
            if "spatial_merge_size" in self.global_config:
                self.gguf_writer.add_vision_spatial_merge_size(self.global_config["spatial_merge_size"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        n_head = (
            self.hparams["num_attention_heads"] if not self.is_mistral_format else self.find_vparam(["num_attention_heads"])
        )
        n_kv_head = n_head
        valid_prefixes = (
            "multi_modal_projector.",
            "vision_tower.",
            "vision_encoder.",
            "vision_language_adapter.",
            "patch_merger.",
            "pre_mm_projector_norm",
        )
        if any(name.startswith(prefix) for prefix in valid_prefixes):
            # process vision tensors
            if name.endswith(("q_proj.weight", "q_proj.bias")) and not self.is_mistral_format:
                data_torch = LlamaModel.permute(data_torch, n_head, n_head)
            if name.endswith(("k_proj.weight", "k_proj.bias")) and not self.is_mistral_format:
                data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)
            return [(self.map_tensor_name(name), data_torch)]
        embed_key = "embed_tokens.weight" if not self.is_mistral_format else "tok_embeddings.weight"
        if self.img_break_tok_id > 0 and embed_key in name:
            logger.info(f"Extracting [IMG_BREAK] token embedding from {name}")
            # for pixtral model, we need to extract the [IMG_BREAK] token embedding
            img_break_embd = data_torch[self.img_break_tok_id]
            name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_TOK_EMBD_IMG_BREAK]
            return [(self.map_tensor_name(name), img_break_embd)]
        return [] # skip other tensors
