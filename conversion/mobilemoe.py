from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torch import Tensor

from .base import LazyTorchTensor, ModelBase, gguf, logger
from .exaone import Exaone4Model


@ModelBase.register("MobileMoEForCausalLM")
@ModelBase.example("facebook/MobileMoE-S-QAT", "facebook/MobileMoE-M-QAT", "facebook/MobileMoE-L-QAT")
@ModelBase.example("facebook/MobileMoE-S-Base", "facebook/MobileMoE-M-Base", "facebook/MobileMoE-L-Base")
@ModelBase.example("facebook/MobileMoE-S-SFT", "facebook/MobileMoE-M-SFT", "facebook/MobileMoE-L-SFT")
class MobileMoEModel(Exaone4Model):
    model_arch = gguf.MODEL_ARCH.MOBILEMOE

    def __init__(self, dir_model, ftype, fname_out, **kwargs):
        if kwargs.get("hparams") is None:
            kwargs["hparams"] = ModelBase.load_hparams(dir_model, self.is_mistral_format)
        quant = kwargs["hparams"].get("quantization") or {}
        self.is_qat = bool(quant)
        self.exact_qat = self.is_qat and ftype == gguf.LlamaFileType.GUESSED
        super().__init__(dir_model, gguf.LlamaFileType.MOSTLY_Q4_0 if self.exact_qat else ftype, fname_out, **kwargs)
        if self.is_qat and quant.get("format") != "mobilemoe-int4-g32":
            raise ValueError(f"Unsupported MobileMoE quantization format: {quant.get('format')}")
        if self.exact_qat and self.is_big_endian:
            raise ValueError("MobileMoE packed QAT conversion requires little endian output")

    @staticmethod
    def unpack_qat(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
        if packed.dtype != np.uint8 or scales.dtype != np.float16:
            raise ValueError("Expected UINT8 packed weights and FP16 scales")
        if packed.shape[-1] % 16 or scales.shape != (*packed.shape[:-1], packed.shape[-1] // 16):
            raise ValueError("Invalid MobileMoE INT4/g32 weight or scale shape")
        codes = np.stack((packed & 15, packed >> 4), axis=-1).reshape(*packed.shape[:-1], -1)
        return ((codes.astype(np.int16) + 8) % 16 - 8).astype(np.int8)

    @staticmethod
    def pack_qat(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
        # Q4_0 uses offset-binary codes and pairs positions 0/16, not adjacent signed nibbles.
        q = (codes.reshape(-1, 32).astype(np.int16) + 8).astype(np.uint8)
        d = np.ascontiguousarray(scales, dtype="<f2").reshape(-1, 1).view(np.uint8)
        return np.concatenate((d, q[:, :16] | (q[:, 16:] << 4)), axis=1).reshape(*codes.shape[:-1], -1)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        c = self.hparams
        self.gguf_writer.add_rope_dimension_count(c["head_dim"])
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        self.gguf_writer.add_feed_forward_length(c["intermediate_size_mlp"])
        self.gguf_writer.add_expert_feed_forward_length(c["intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(c["intermediate_size_mlp"])
        self.gguf_writer.add_expert_shared_count(1)
        self.gguf_writer.add_expert_weights_scale(c["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(c["norm_topk_prob"])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        if self.exact_qat:
            self.gguf_writer.add_tensor_data_layout("qat_output_grouped_g32")

    def set_vocab(self):
        super().set_vocab()
        # The checkpoint uses the Llama vocabulary with renamed end-of-message and end-of-turn tokens.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        # Fast tokenizer post-processing can differ from the add_bos_token configuration.
        special = tokenizer.encode("a", add_special_tokens=True)
        self.gguf_writer.add_add_bos_token(special[0] == tokenizer.bos_token_id)
        self.gguf_writer.add_add_eos_token(special[-1] == tokenizer.eos_token_id)
        for token, setter in (("<|end_of_text|>", self.gguf_writer.add_eos_token_id),
                              ("<|eot|>", self.gguf_writer.add_eot_token_id),
                              ("<|eom|>", self.gguf_writer.add_eom_token_id)):
            token_id = tokenizer.get_vocab().get(token)
            if token_id is not None:
                setter(token_id)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        if name.endswith(".experts.gate_up_proj"):
            gate, up = data_torch.transpose(-1, -2).split(self.hparams["intermediate_size"], dim=-2)
            yield from super().modify_tensors(gate, name.replace("gate_up_proj", "gate_proj.weight"), bid)
            yield from super().modify_tensors(up, name.replace("gate_up_proj", "up_proj.weight"), bid)
            return
        if name.endswith(".experts.down_proj"):
            data_torch = data_torch.transpose(-1, -2)
            name += ".weight"
        elif name.endswith(("q_proj.weight", "k_proj.weight")):
            heads = self.hparams["num_attention_heads" if name.endswith("q_proj.weight") else "num_key_value_heads"]
            data_torch = data_torch.reshape(heads, self.hparams["head_dim"] // 2, 2, -1).transpose(1, 2).reshape(data_torch.shape)
        elif name.endswith(".expert_bias"):
            name += ".bias"
        yield from super().modify_tensors(data_torch, name, bid)

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        if not self.is_qat or self.exact_qat:
            yield from super().get_tensors()
            return
        consumed = set()
        for name in self.model_tensors:
            if name.endswith((".weight_scale", "_scale")):
                continue
            data = LazyTorchTensor.to_eager(self.model_tensors[name]())
            consumed.add(name)
            if name.endswith((".qweight", "_qweight")):
                expert = name.endswith("_qweight")
                base = name.removesuffix("_qweight" if expert else ".qweight")
                scale_name = base + ("_scale" if expert else ".weight_scale")
                scales = LazyTorchTensor.to_eager(self.model_tensors[scale_name]()).numpy()
                consumed.add(scale_name)
                codes = self.unpack_qat(data.numpy(), scales)
                values = codes.reshape(*scales.shape, 32).astype(np.float32) * scales.astype(np.float32)[..., None]
                data = torch.from_numpy(values.reshape(codes.shape)).to(torch.bfloat16).float()
                name = base + ("_proj" if expert else ".weight")
            yield name, data
        if consumed != set(self.model_tensors):
            raise ValueError(f"Unprocessed MobileMoE tensors: {sorted(set(self.model_tensors) - consumed)}")

    def prepare_tensors(self):
        if not self.exact_qat:
            super().prepare_tensors()
            return
        consumed = set()

        def get(name: str) -> Tensor:
            consumed.add(name)
            return LazyTorchTensor.to_eager(self.model_tensors[name]())

        def emit(name: str, codes: np.ndarray, scales: np.ndarray):
            data = self.pack_qat(codes, scales)
            qtype = gguf.GGMLQuantizationType.Q4_0
            logger.info("%s -> %s", name, qtype.name)
            self.gguf_writer.add_tensor(name, np.ascontiguousarray(data), raw_dtype=qtype)

        for name in self.model_tensors:
            if name in consumed:
                continue
            if name.endswith((".weight_scale", "_scale")):
                continue
            if name.endswith((".qweight", "_qweight")):
                expert = name.endswith("_qweight")
                suffix = "_qweight" if expert else ".qweight"
                base = name.removesuffix(suffix)
                scales = get(base + ("_scale" if expert else ".weight_scale")).numpy()
                codes = self.unpack_qat(get(name).numpy(), scales)
                if expert and base.endswith(".gate_up"):
                    for part, q, s in zip(("gate", "up"), np.split(codes, 2, axis=-1), np.split(scales, 2, axis=-1)):
                        target = self.map_tensor_name(base.removesuffix("gate_up") + part + "_proj.weight")
                        emit(target, q, s)
                else:
                    target = self.map_tensor_name(base + ("_proj.weight" if expert else ".weight"))
                    if name.endswith(("q_proj.qweight", "k_proj.qweight")):
                        heads = self.hparams["num_attention_heads" if name.endswith("q_proj.qweight") else "num_key_value_heads"]

                        def permute(x):
                            return x.reshape(heads, self.hparams["head_dim"] // 2, 2, -1).transpose(0, 2, 1, 3).reshape(x.shape)
                        codes, scales = permute(codes), permute(scales)
                    emit(target, codes, scales)
            else:
                source_name = name + ".bias" if name.endswith(".expert_bias") else name
                target = self.map_tensor_name(source_name)
                tensor = get(name).float().numpy()
                self.gguf_writer.add_tensor(target, tensor)
        if consumed != set(self.model_tensors):
            raise ValueError(f"Unprocessed MobileMoE tensors: {sorted(set(self.model_tensors) - consumed)}")
        for name, tensor in self.generate_extra_tensors():
            self.gguf_writer.add_tensor(name, tensor.numpy())
