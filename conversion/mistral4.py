from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf

from .mistral import MistralModel, MistralMoeModel


@ModelBase.register("MistralLarge4ForConditionalGeneration")
class MistralLarge4Model(MistralMoeModel):
    model_arch = gguf.MODEL_ARCH.MISTRAL_LARGE4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_layer = self.hparams["num_hidden_layers"]
        dsa = self.hparams["dsa"]

        def expand(pattern: str) -> list[str]:
            values = pattern.split(",")
            return [values[i % len(values)] for i in range(n_layer)]

        self.layer_has_rope = [v == "rope" for v in expand(self.hparams.get("nope") or "rope")]
        self.layer_indexer_full = [v == "full" for v in expand(dsa.get("sharing") or "full")]

        kv_sharing = self.hparams.get("kv_sharing") or {}
        if "source_layers_str" in kv_sharing:
            self.layer_kv_source = [int(v) for v in kv_sharing["source_layers_str"].split(",")]
        else:
            assert not kv_sharing, f"unsupported kv_sharing config: {kv_sharing}"
            self.layer_kv_source = list(range(n_layer))
        assert len(self.layer_kv_source) == n_layer

        # note: the rope dim of the indexer must match the main attention
        assert dsa.get("rope_head_dim", -1) in (-1, self.hparams["qk_rope_head_dim"])
        assert dsa.get("match_nope", False), "indexer must follow the NoPE pattern of the main attention"
        assert dsa.get("ssmax_scaling", False), "indexer without SSMax is not supported"
        assert self.hparams.get("q_lora_rank") is None
        assert not self.hparams["yarn"].get("apply_scale", True), "only YaRN without magnitude scaling is supported"

    def dequant_model(self):
        quant_config = self.hparams.get("quantization")
        if quant_config is not None:
            assert quant_config["qformat_weight"] == "fp8_e4m3"
            assert quant_config.get("qscheme_weight") == "BLOCK"
            self.hparams["quantization_config"] = {
                "quant_method": "fp8",
                "weight_block_size": [quant_config["block_size_out"], quant_config["block_size_in"]],
            }
        return super().dequant_model()

    def set_gguf_parameters(self):
        # skip the MistralMoeModel overrides, they apply the deepseek2 YaRN scaling
        super(MistralMoeModel, self).set_gguf_parameters()
        MistralModel.set_mistral_config(self.gguf_writer, self.hparams)

        dsa = self.hparams["dsa"]
        self.gguf_writer.add_indexer_head_count(dsa["n_heads"])
        self.gguf_writer.add_indexer_key_length(dsa["head_dim"])
        self.gguf_writer.add_indexer_top_k(dsa["topk"])
        self.gguf_writer.add_indexer_types(self.layer_indexer_full)
        self.gguf_writer.add_layer_norm_eps(dsa.get("norm_eps", 1e-6))

        self.gguf_writer.add_rope_pattern(self.layer_has_rope)
        self.gguf_writer.add_kv_source_map(self.layer_kv_source)
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
        self.gguf_writer.add_final_logit_softcapping(self.hparams["softmax_tanh_gating"])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # exported separately with --dspark
        if name.startswith("mtp_"):
            return None

        # experts need HF names for the merging logic in DeepseekV2Model
        # note: bare "layers.N.experts" comes from the NVFP4 quantized_layers config
        if ".experts." in name or name.endswith(".experts"):
            name = "model." + name.replace(".experts", ".mlp.experts")
            name = name.replace(".w1.", ".gate_proj.").replace(".w2.", ".down_proj.").replace(".w3.", ".up_proj.")

        # skip the MistralMoeModel renaming, the qscale names are needed for block FP8 dequant
        return super(MistralMoeModel, cls).filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "lerope.alpha":
            # LeRoPE scales inv_freq by exp(alpha), ggml divides theta by the freq factors
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), torch.exp(-data_torch.float()))
            return

        if bid is not None and name.endswith((".ssmax_scaling.s", ".ssmax_scaling.b")):
            tensor = gguf.MODEL_TENSOR.INDEXER_SSMAX if ".indexer." in name else gguf.MODEL_TENSOR.ATTN_SSMAX
            suffix = ".weight" if name.endswith(".s") else ".bias"
            yield (self.format_tensor_name(tensor, bid, suffix), data_torch.float())
            return

        if bid is not None and ".indexer.q_input_norm." in name:
            suffix = ".weight" if name.endswith(".weight") else ".bias"
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_NORM, bid, suffix), data_torch)
            return

        if bid is not None and ".indexer." in name:
            name = "model." + name.replace(".attention.", ".self_attn.")

        # NoPE layers have a larger non-rotary head, so wkv_b is split per layer
        if bid is not None and name.endswith(".wkv_b.weight"):
            n_head = self.hparams["num_attention_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
            if not self.layer_has_rope[bid]:
                qk_nope_head_dim += self.hparams["qk_rope_head_dim"]

            assert data_torch.shape[0] == n_head * (qk_nope_head_dim + v_head_dim)

            kv_b = data_torch.view(n_head, qk_nope_head_dim + v_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)

            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K_B, bid), k_b)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V_B, bid), v_b)
            return

        yield from super().modify_tensors(data_torch, name, bid)


class MistralLarge4DSparkModel(MistralLarge4Model):
    """Mistral Large 4 DSpark draft (mtp_dspark.*) as a DFLASH GGUF, export with --dspark"""
    model_arch = gguf.MODEL_ARCH.DFLASH

    _DSPARK_ROOT: dict[str, gguf.MODEL_TENSOR] = {
        "backbone.target_projection.weight": gguf.MODEL_TENSOR.FC,
        "backbone.target_norm.weight":       gguf.MODEL_TENSOR.ENC_OUTPUT_NORM,
        "backbone.final_norm.weight":        gguf.MODEL_TENSOR.OUTPUT_NORM,
        "backbone.mask_embedding":           gguf.MODEL_TENSOR.DSPARK_MASK_EMBD,
        "autoregressive_head.embedding.weight": gguf.MODEL_TENSOR.DSPARK_MARKOV_W1,
        "autoregressive_head.output.weight":    gguf.MODEL_TENSOR.DSPARK_MARKOV_W2,
        "autoregressive_head.joint_proj.weight": gguf.MODEL_TENSOR.DSPARK_JOINT_PROJ,
        "autoregressive_head.joint_proj.bias":   gguf.MODEL_TENSOR.DSPARK_JOINT_PROJ,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mtp = self.hparams.get("mtp") or {}
        dspark = mtp.get("dspark")
        if dspark is None:
            raise ValueError("Mistral Large 4 DSpark export needs the mtp.dspark config")
        assert dspark.get("head_type") == "rnn", f"unsupported DSpark head type: {dspark.get('head_type')}"
        assert dspark.get("draft_input") == "position_specific", f"unsupported DSpark draft input: {dspark.get('draft_input')}"

        windows = {int(v) for v in (mtp.get("ragged_attention") or "").split(",") if v.strip()}
        assert len(windows) == 1, f"DSpark draft layers must share one sliding window, got {mtp.get('ragged_attention')}"

        self._n_stage = int(mtp["n_layers"])
        self._depth = int(mtp["depth"])
        self._window = windows.pop()
        self._target_layers = [int(v) for v in mtp["target_activation_layers"].split(",")]

        self.layer_has_rope = [True] * self._n_stage

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if not name.startswith("mtp_dspark."):
            return None
        name = name.removeprefix("mtp_dspark.")
        if name.startswith("backbone.layers."):
            return name.removeprefix("backbone."), gen
        return f"dspark.{name}", gen

    def set_vocab(self):
        super().set_vocab()
        # block uses mask_embd, mask token id is not used
        self.gguf_writer.add_mask_token_id(0)

    def set_gguf_parameters(self):
        hp = self.hparams
        w = self.gguf_writer
        w.add_block_count(self._n_stage)
        w.add_context_length(hp["max_position_embeddings"])
        w.add_embedding_length(hp["hidden_size"])
        w.add_feed_forward_length(hp["intermediate_size"])
        w.add_head_count(hp["num_attention_heads"])
        w.add_head_count_kv(1)
        w.add_key_length(hp["kv_lora_rank"] + hp["qk_rope_head_dim"])
        w.add_value_length(hp["kv_lora_rank"])
        w.add_key_length_mla(hp["qk_nope_head_dim"] + hp["qk_rope_head_dim"])
        w.add_value_length_mla(hp["v_head_dim"])
        w.add_kv_lora_rank(hp["kv_lora_rank"])
        w.add_rope_dimension_count(hp["qk_rope_head_dim"])
        w.add_rope_freq_base(float(hp["rope_theta"]))
        w.add_layer_norm_rms_eps(hp["rms_norm_eps"])
        MistralModel.set_mistral_config(w, hp)
        w.add_sliding_window(self._window)
        w.add_final_logit_softcapping(hp["softmax_tanh_gating"])
        w.add_block_size(self._depth)
        # draft reads layer output = input of next layer
        w.add_target_layers([layer + 1 for layer in self._target_layers])
        w.add_sample_from_anchor(True)
        w.add_has_confidence_head(False)
        w.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("dspark."):
            key = name.removeprefix("dspark.")
            if key not in self._DSPARK_ROOT:
                raise ValueError(f"Unexpected DSpark tensor {name!r}")
            suffix = ".bias" if key.endswith(".bias") else ".weight"
            yield (self.format_tensor_name(self._DSPARK_ROOT[key], None, suffix), data_torch)
            return

        yield from super().modify_tensors(data_torch, name, bid)
