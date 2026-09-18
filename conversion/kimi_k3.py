from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable, Iterator, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf, logger

from .kimi_linear import KimiLinearModel


@ModelBase.register("KimiK3ForConditionalGeneration")
@ModelBase.example("moonshotai/Kimi-K3")
class KimiK3Model(TextModel):
    """
    Kimi-K3 text model (KimiLinearForCausalLM under a `language_model.` prefix).

    Shares the hybrid MLA + KDA skeleton with kimi-linear, but that converter
    cannot load it: K3 adds cross-layer attention residuals, a latent MoE, the
    situ activation, an MLA output gate and a full-rank KDA gate.

    The vision tower and mm_projector are skipped - text only for now.
    """

    model_arch = gguf.MODEL_ARCH.KIMI_K3

    _experts: list[dict[str, Tensor]] | None = None

    # `<x>_res_norm.weight` and `<x>_res_proj.weight` are only used as their
    # elementwise product, so they are fused into one [n_embd] vector here.
    # they arrive apart, so buffer the first one and tag it with its kind.
    _res_parts: dict[str, tuple[str, Tensor]]

    # HF suffix -> (gguf tensor, per-layer?)
    _RES_FUSIONS = {
        "self_attention_res": (gguf.MODEL_TENSOR.ATTN_RES_SCORE, True),
        "mlp_res":            (gguf.MODEL_TENSOR.FFN_RES_SCORE,   True),
        "output_attn_res":    (gguf.MODEL_TENSOR.OUTPUT_RES_SCORE, False),
    }

    # compressed-tensors MXFP4. the `language_model.` prefix is still there, as
    # self.model_tensors is keyed by the raw checkpoint names
    _MXFP4_FORMAT = "mxfp4-pack-quantized"
    _MXFP4_EXPERT_RE = re.compile(
        r"^(?:language_model\.)?model\.layers\.(\d+)"
        r"\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight_packed$"
    )
    _MXFP4_PROJ = {
        "w1": gguf.MODEL_TENSOR.FFN_GATE_EXP,
        "w2": gguf.MODEL_TENSOR.FFN_DOWN_EXP,
        "w3": gguf.MODEL_TENSOR.FFN_UP_EXP,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._res_parts = {}

    def set_vocab(self):
        # K3 has the same TikToken vocab as K2, so kimi-linear's vocab handling works.
        # borrowed, not inherited: the method only touches TextModel members, and K3
        # shares none of kimi-linear's tensor layout.
        KimiLinearModel.set_vocab(self)  # ty: ignore[invalid-argument-type]

        # ...but that forces eos to the tokenizer's eos_id, which is [EOS], the
        # document terminator. K3's config says <|end_of_msg|>, the turn terminator;
        # with [EOS] the generation never stops at the end of a turn.
        if (eos := self.hparams.get("eos_token_id")) is not None:
            logger.info(f"restoring configured eos_token_id {eos} (kimi-linear forces the tokenizer's)")
            self.gguf_writer.add_eos_token_id(eos)

        # K3 renders chats in python (encoding_k3.py) and ships no jinja template,
        # so add the bundled one when the model has none
        if gguf.SpecialVocab(self.dir_model, load_merges=False).chat_template is None:
            template_path = Path(__file__).parent.parent / "models" / "templates" / "Kimi-K3.jinja"
            logger.info(f"gguf: model has no chat template, using {template_path.name}")
            self.gguf_writer.add_chat_template(template_path.read_text(encoding="utf-8"))

    #
    # compressed-tensors MXFP4 -> ggml MXFP4
    #

    def _is_mxfp4_packed(self) -> bool:
        quant_config = self.hparams.get("quantization_config") or {}
        return (quant_config.get("quant_method") == "compressed-tensors"
                and quant_config.get("format") == self._MXFP4_FORMAT)

    def dequant_model(self):
        if not self._is_mxfp4_packed():
            return super().dequant_model()

        # skipping base.py's dequant is only safe if the experts are the only
        # quantized tensors, so check it
        stray = [n for n in self.model_tensors
                 if n.endswith(".weight_packed") and not self._MXFP4_EXPERT_RE.match(n)]
        if stray:
            raise NotImplementedError(
                f"{len(stray)} MXFP4 tensor(s) outside the routed experts, e.g. {stray[0]!r}; "
                "only the routed experts have a repack path"
            )

    def _mxfp4_expert_tensor(self, loaders: list[tuple[Callable[[], Tensor], Callable[[], Tensor]]]):
        """
        One stacked [n_expert, rows, cols] MXFP4 tensor, built lazily.

        gguf_writer holds every added tensor until the final write, so building
        this eagerly (like the DeepSeek-V4 path does) keeps all ~1.38 TB of
        experts in memory. lazy means only the tensor being written is resident.
        """
        # meta shapes, so this does not read any weights
        rows, packed_cols = loaders[0][0]().shape
        n_blocks = (packed_cols * 2) // 32
        byte_shape = (len(loaders), rows, n_blocks * 17)

        def load(fns: list[tuple[Callable[[], Tensor], Callable[[], Tensor]]]) -> np.ndarray:
            out = np.empty(byte_shape, dtype=np.uint8)
            for eid, (packed_fn, scale_fn) in enumerate(fns):
                out[eid] = self.repack_mxfp4_blocks(
                    LazyTorchTensor.to_eager(packed_fn()),
                    LazyTorchTensor.to_eager(scale_fn()),
                )
            return out

        # loaders goes through args, not the closure, so that `func` matches
        # LazyBase's single-argument shape
        return gguf.LazyNumpyTensor(
            meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(np.uint8, byte_shape),
            args=(loaders,),
            func=load,
        )

    def _write_mxfp4_experts(self) -> None:
        n_experts = self.hparams["num_experts"]

        # (bid, wid) -> {expert id: (packed name, scale name)}
        groups: dict[tuple[int, str], dict[int, tuple[str, str]]] = {}
        for name in self.model_tensors:
            m = self._MXFP4_EXPERT_RE.match(name)
            if m is None:
                continue
            bid, eid, wid = int(m.group(1)), int(m.group(2)), m.group(3)
            scale_name = name.removesuffix("_packed") + "_scale"
            if scale_name not in self.model_tensors:
                raise KeyError(f"missing {scale_name} for {name}")
            groups.setdefault((bid, wid), {})[eid] = (name, scale_name)

        consumed: list[str] = []
        for (bid, wid), experts in sorted(groups.items()):
            missing = [e for e in range(n_experts) if e not in experts]
            if missing:
                raise KeyError(
                    f"layer {bid} {wid}: {len(missing)} of {n_experts} experts missing, "
                    f"first is {missing[0]}"
                )
            if len(experts) != n_experts:
                raise KeyError(f"layer {bid} {wid}: {len(experts)} experts, expected {n_experts}")

            loaders = []
            for eid in range(n_experts):
                packed_name, scale_name = experts[eid]
                loaders.append((self.model_tensors[packed_name], self.model_tensors[scale_name]))
                consumed += [packed_name, scale_name]

            data = self._mxfp4_expert_tensor(loaders)
            new_name = self.format_tensor_name(self._MXFP4_PROJ[wid], bid)
            shape = gguf.quant_shape_from_byte_shape(data.shape, gguf.GGMLQuantizationType.MXFP4)
            logger.info(
                f"{new_name}: repacked {n_experts} experts to MXFP4, "
                f"shape = {{{', '.join(str(n) for n in reversed(shape))}}}"
            )
            self.gguf_writer.add_tensor(new_name, data, raw_dtype=gguf.GGMLQuantizationType.MXFP4)

        for name in consumed:
            del self.model_tensors[name]

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # not a generator on purpose: base.py chains this with get_tensors(), so the
        # tensors used here must be removed from model_tensors before that starts
        if self._is_mxfp4_packed():
            self._write_mxfp4_experts()
        return ()

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        for name, data in super().get_tensors():
            if name.startswith(("vision_tower.", "mm_projector.")):
                continue  # text only
            if name.startswith("language_model."):
                name = name[len("language_model."):]
            yield name, data

    def set_gguf_parameters(self):
        # MLA is served as MQA with a single large head, then decompressed
        self.hparams["num_key_value_heads"] = 1

        super().set_gguf_parameters()
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])

        linear_attn_config = self.hparams["linear_attn_config"]

        # n_head_kv == 0 marks a KDA (recurrent) layer. the layer lists are 1-indexed,
        # as KimiLinearConfig.is_kda_layer uses (layer_idx + 1)
        full_attn_layers = linear_attn_config["full_attn_layers"]
        n_kv_heads = [
            self.hparams["num_key_value_heads"] if (il + 1) in full_attn_layers else 0
            for il in range(self.hparams["num_hidden_layers"])
        ]
        assert len(n_kv_heads) == self.hparams["num_hidden_layers"]
        self.gguf_writer.add_head_count_kv(n_kv_heads)

        # --- KDA ---
        self.gguf_writer.add_ssm_conv_kernel(linear_attn_config["short_conv_kernel_size"])
        self.gguf_writer.add_kda_head_dim(linear_attn_config["head_dim"])
        if (lb := linear_attn_config.get("gate_lower_bound")) is not None:
            self.gguf_writer.add_kda_gate_lower_bound(lb)

        # --- MLA ---
        if (q_lora_rank := self.hparams.get("q_lora_rank")) is not None:
            self.gguf_writer.add_q_lora_rank(q_lora_rank)
        kv_lora_rank = self.hparams["kv_lora_rank"]
        self.gguf_writer.add_kv_lora_rank(kv_lora_rank)

        qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
        qk_rope_head_dim = self.hparams["qk_rope_head_dim"]
        v_head_dim = self.hparams["v_head_dim"]
        # K3 is nope-only; qk_rope_head_dim still sizes the un-absorbed part of K
        assert self.hparams.get("mla_use_nope"), "K3 MLA is expected to be nope-only"
        self.gguf_writer.add_rope_dimension_count(qk_rope_head_dim)
        # MLA is served as MQA, so the cache holds the compressed latent
        self.gguf_writer.add_key_length(kv_lora_rank + qk_rope_head_dim)
        self.gguf_writer.add_value_length(kv_lora_rank)
        self.gguf_writer.add_key_length_mla(qk_nope_head_dim + qk_rope_head_dim)
        self.gguf_writer.add_value_length_mla(v_head_dim)

        # --- MoE ---
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(self.hparams["num_shared_experts"])
        self.gguf_writer.add_leading_dense_block_count(self.hparams["first_k_dense_replace"])
        self.gguf_writer.add_expert_weights_scale(self.hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(self.hparams["moe_renormalize"])
        assert self.hparams["moe_router_activation_func"] == "sigmoid"
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        # latent MoE: routed experts live in a down-projected space
        if (latent := self.hparams.get("routed_expert_hidden_size")) is not None:
            self.gguf_writer.add_expert_latent_length(latent)

        # --- situ activation ---
        assert self.hparams["hidden_act"] == "situ", \
            f"unexpected hidden_act {self.hparams['hidden_act']!r}"
        self.gguf_writer.add_activation_situ_beta(self.hparams["activation_situ_beta"])
        self.gguf_writer.add_activation_situ_linear_beta(self.hparams["activation_situ_linear_beta"])

        # --- cross-layer attention residuals ---
        self.gguf_writer.add_attn_res_block_size(self.hparams["attn_res_block_size"])

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            leftover = [k for d in self._experts for k in d.keys()]
            if leftover:
                raise ValueError(f"Unprocessed experts: {leftover}")
        if self._res_parts:
            raise ValueError(f"Unpaired attention-residual tensors: {sorted(self._res_parts)}")
        if self._is_mxfp4_packed():
            # label the file for what it is; prepare_metadata runs after this
            self._is_mxfp4 = True
            self.ftype = gguf.LlamaFileType.MOSTLY_MXFP4_MOE

    def _try_fuse_res(self, data_torch: Tensor, name: str, bid: int | None):
        """
        Pair <x>_res_norm.weight with <x>_res_proj.weight and emit their product.

        Returns None if this is not a res tensor, [] if buffered until its pair.
        """
        for prefix, (tensor_id, per_layer) in self._RES_FUSIONS.items():
            for kind in ("norm", "proj"):
                if not name.endswith(f"{prefix}_{kind}.weight"):
                    continue
                key = f"{prefix}.{bid}"
                other = self._res_parts.pop(key, None)
                if other is None:
                    self._res_parts[key] = (kind, data_torch)
                    return []
                other_kind, other_data = other
                assert other_kind != kind, f"duplicate {kind} for {key}"
                norm = data_torch if kind == "norm" else other_data
                proj = data_torch if kind == "proj" else other_data
                fused = norm.float().flatten() * proj.float().flatten()
                # ".weight" suffix matches the convention map_tensor_name applies
                new_name = (self.format_tensor_name(tensor_id, bid) if per_layer
                            else gguf.TENSOR_NAMES[tensor_id] + ".weight")
                logger.info(f"fused {prefix}_norm * {prefix}_proj -> {new_name}")
                return [(new_name, fused)]
        return None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # --- cross-layer attention residuals: fuse norm * proj ---
        fused = self._try_fuse_res(data_torch, name, bid)
        if fused is not None:
            yield from fused
            return

        # --- KDA conv1d: HF [d_inner, 1, d_conv] -> ggml ne [d_conv, 1, d_inner, 1] ---
        # GGUF reverses the numpy shape on write, so target numpy (1, d_inner, 1, d_conv).
        # conv_step varies fastest in both layouts, so this is a pure reshape.
        if name.endswith((".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")):
            if data_torch.ndim == 3:      # [d_inner, 1, d_conv]
                d_inner, _, d_conv = data_torch.shape
            elif data_torch.ndim == 2:    # [d_inner, d_conv]
                d_inner, d_conv = data_torch.shape
            else:
                raise ValueError(f"unexpected conv1d rank {data_torch.ndim} for {name}")
            data_torch = data_torch.reshape(1, d_inner, 1, d_conv)

        # -exp(A_log) is folded here so the graph does not have to
        if name.endswith(".A_log"):
            n_head = self.hparams["num_attention_heads"]
            data_torch = -torch.exp(data_torch.float()[:n_head])

        # dt_bias -> the name SSM_DT's mapping expects
        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"

        # --- g_proj is two different tensors sharing one HF name ---
        # KDA layers: full-rank gate, [d_inner, n_embd]  (replaces g_a/g_b)
        # MLA layers: output gate,    [n_head*v_head_dim, n_embd]
        # Name-based mapping cannot tell them apart, so resolve by layer type.
        if name.endswith(".self_attn.g_proj.weight"):
            assert bid is not None
            is_kda = (bid + 1) not in self.hparams["linear_attn_config"]["full_attn_layers"]
            tensor_id = gguf.MODEL_TENSOR.SSM_G if is_kda else gguf.MODEL_TENSOR.ATTN_GATE
            yield self.format_tensor_name(tensor_id, bid), data_torch
            return

        # --- routed experts: stack per-expert 2D weights into one 3D tensor ---
        if ".block_sparse_moe.experts." in name:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) < n_experts * 3:
                return

            # w1: gate, w2: down, w3: up
            for wid, tensor_id in (("w1", gguf.MODEL_TENSOR.FFN_GATE_EXP),
                                   ("w2", gguf.MODEL_TENSOR.FFN_DOWN_EXP),
                                   ("w3", gguf.MODEL_TENSOR.FFN_UP_EXP)):
                datas = []
                for xid in range(n_experts):
                    ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight"
                    datas.append(self._experts[bid].pop(ename))
                stacked = torch.stack(datas, dim=0)
                yield from super().modify_tensors(stacked, self.format_tensor_name(tensor_id, bid), bid)
            return

        # --- MLA absorption: split kv_b into k_b (transposed) and v_b ---
        if name.endswith("kv_b_proj.weight"):
            n_head_kv = self.hparams["num_key_value_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
            assert data_torch.shape[0] == n_head_kv * (v_head_dim + qk_nope_head_dim)
            kv_b = data_torch.view(n_head_kv, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)
            yield from super().modify_tensors(k_b, name.replace("kv_b_proj", "k_b_proj"), bid)
            yield from super().modify_tensors(v_b, name.replace("kv_b_proj", "v_b_proj"), bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)
