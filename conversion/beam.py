from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("BeamForCausalLM")
class BeamModel(TextModel):
    model_arch = gguf.MODEL_ARCH.BEAM

    # reference NoPE query-scale constants; not in config.json
    NOPE_SCALE_POWER  = 0.1
    NOPE_SCALE_LENGTH = 2048

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hp = self.hparams
        rope = hp.get("rope_parameters") or {}
        if rope.get("full_attention") is not None:
            raise ValueError("Beam: expected NoPE full-attention layers (rope_parameters.full_attention = null)")
        if "sliding_attention" in rope:
            swa_rope = rope["sliding_attention"] or {}
        else:
            # flat rope_theta / rope_scaling, already normalized by TextModel
            swa_rope = {k: v for k, v in self.rope_parameters.items() if k not in ("sliding_attention", "full_attention", "type")}
            swa_rope.update(self.rope_parameters.get("sliding_attention") or {})
        if swa_rope.get("rope_type", "default") != "default":
            raise ValueError(f"Beam: unsupported sliding-attention rope type {swa_rope.get('rope_type')!r}")
        # full layers are NoPE, so mirror the sliding rope params for the base class
        swa_rope = {k: v for k, v in swa_rope.items() if not (k == "rope_type" and v == "default")}
        self.rope_parameters = {"sliding_attention": dict(swa_rope), "full_attention": dict(swa_rope)}

        layer_types = hp["layer_types"]
        mlp_types = hp["mlp_layer_types"]
        n_layer = hp["num_hidden_layers"]
        if len(layer_types) != n_layer or len(mlp_types) != n_layer:
            raise ValueError("Beam: layer_types / mlp_layer_types must have num_hidden_layers entries")
        if any(t not in ("sliding_attention", "full_attention") for t in layer_types):
            raise ValueError(f"Beam: unknown layer type in {layer_types}")
        if "sliding_attention" not in layer_types:
            raise ValueError("Beam: expected at least one sliding_attention layer")
        n_dense = 0
        while n_dense < n_layer and mlp_types[n_dense] == "dense":
            n_dense += 1
        if any(t != "sparse" for t in mlp_types[n_dense:]):
            raise ValueError("Beam: llama.cpp supports dense MLP layers only as a leading block")
        self._n_dense_lead = n_dense
        if hp.get("attention_bias", False):
            raise ValueError("Beam: attention_bias=True is not supported")
        self._experts: dict[tuple[int, str], dict[int, Tensor]] = {}

    def set_vocab(self):
        self._set_vocab_gpt2()
        # otherwise EOT is guessed from a hash-ordered scan
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.dir_model)
        ret_id = tok.convert_tokens_to_ids("<|return|>")  # ty: ignore[unresolved-attribute]
        if isinstance(ret_id, int) and ret_id >= 0:
            self.gguf_writer.add_eot_token_id(ret_id)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        head_dim = hp["head_dim"]
        self.gguf_writer.add_rope_dimension_count(head_dim)

        self.gguf_writer.add_sliding_window(hp["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in hp["layer_types"]])

        self.gguf_writer.add_attn_temperature_scale(self.NOPE_SCALE_POWER)
        self.gguf_writer.add_attn_temperature_length(self.NOPE_SCALE_LENGTH)

        self.gguf_writer.add_leading_dense_block_count(self._n_dense_lead)
        self.gguf_writer.add_expert_feed_forward_length(hp["moe_intermediate_size"])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        self.gguf_writer.add_expert_weights_norm(bool(hp.get("norm_topk_prob", True)))
        # equivalent to the reference's post-aggregation scaling
        self.gguf_writer.add_expert_weights_scale(float(hp["routed_scaling_factor"]))

        if (softcap := hp.get("final_logit_softcapping")) is not None:
            self.gguf_writer.add_final_logit_softcapping(float(softcap))

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if name.endswith(".self_attn.gate_bias"):
            name = name.replace(".self_attn.gate_bias", ".self_attn.gate_proj.bias")
        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        hp = self.hparams
        if name.endswith(".self_attn.gate_proj.bias"):
            assert tuple(data_torch.shape) == (hp["num_attention_heads"],), f"{name}: one gate bias per head expected, got {tuple(data_torch.shape)}"
        elif name.endswith(".self_attn.gate_proj.weight"):
            assert tuple(data_torch.shape) == (hp["num_attention_heads"] * hp["head_dim"], hp["hidden_size"]), f"{name}: {tuple(data_torch.shape)}"
        elif name.endswith(".mlp.gate.e_score_correction.bias"):
            assert tuple(data_torch.shape) == (hp["num_experts"],), f"{name}: {tuple(data_torch.shape)}"
        elif name.endswith((".self_attn.q_norm.weight", ".self_attn.k_norm.weight")):
            assert tuple(data_torch.shape) == (hp["head_dim"],), f"{name}: per-head q/k norm expected, got {tuple(data_torch.shape)}"
        elif bid is not None and bid < self._n_dense_lead and ".mlp.experts." in name:
            raise ValueError(f"Beam: expert tensor {name} in a dense layer")
        if ".mlp.experts." in name:
            assert bid is not None
            n_experts = self.hparams["num_experts"]
            parts = name.split(".")
            xid = int(parts[parts.index("experts") + 1])
            proj = parts[parts.index("experts") + 2]
            if proj not in ("gate_proj", "up_proj", "down_proj"):
                raise ValueError(f"Beam: unexpected expert tensor {name}")
            bucket = self._experts.setdefault((bid, proj), {})
            if xid in bucket:
                raise ValueError(f"Beam: duplicate expert tensor {name}")
            bucket[xid] = data_torch
            if len(bucket) < n_experts:
                return
            del self._experts[(bid, proj)]
            data = torch.stack([bucket[e] for e in range(n_experts)], dim=0)
            yield from super().modify_tensors(data, f"model.layers.{bid}.mlp.experts.{proj}.weight", bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts:
            missing = sorted(f"layer {b} {p} ({len(v)} experts)" for (b, p), v in self._experts.items())
            raise ValueError(f"Beam: unprocessed experts: {missing}")
        logger.info("Beam: all routed experts stacked")
