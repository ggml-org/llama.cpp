from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf, logger

from .qwen import QwenModel


@ModelBase.register("KimiLinearModel", "KimiLinearForCausalLM", "KimiK3ForConditionalGeneration")
class KimiLinearModel(TextModel):
    """Kimi-Linear model with hybrid MLA+KDA architecture"""
    model_arch = gguf.MODEL_ARCH.KIMI_LINEAR

    _experts: list[dict[str, Tensor]] | None = None
    _mxfp4_experts_written = False

    @classmethod
    def filter_tensors(
        cls,
        item: tuple[str, Callable[[], Tensor]],
    ) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        if name.startswith(("vision_tower.", "mm_projector.")):
            return None
        return super().filter_tensors(item)

    @staticmethod
    def _pack_mxfp4_blocks(weight: Tensor, scale: Tensor) -> np.ndarray:
        packed = weight.contiguous().view(torch.uint8)
        scale_u8 = scale.contiguous().view(torch.uint8)
        out_features, packed_cols = packed.shape
        logical_cols = packed_cols * 2
        if logical_cols % 32 != 0:
            raise ValueError(
                f"MXFP4 source row has {logical_cols} values, expected a multiple of 32"
            )
        n_blocks = logical_cols // 32
        if tuple(scale_u8.shape) != (out_features, n_blocks):
            raise ValueError(
                f"MXFP4 scale shape {tuple(scale_u8.shape)} "
                f"does not match {(out_features, n_blocks)}"
            )
        source = packed.reshape(out_features, n_blocks, 16)
        values = torch.stack(
            (source & 0x0F, (source >> 4) & 0x0F), dim=-1
        ).reshape(out_features, n_blocks, 32)
        quants = values[:, :, :16] | (values[:, :, 16:] << 4)
        raw = torch.cat((scale_u8.unsqueeze(-1), quants.to(torch.uint8)), dim=-1)
        return raw.reshape(out_features, n_blocks * 17).cpu().numpy()

    def _write_mxfp4_expert_tensor(
        self,
        bid: int,
        projection: str,
        tensor_key: gguf.MODEL_TENSOR,
    ) -> list[str]:
        n_experts = self.find_hparam(["num_local_experts", "num_experts"])
        data: np.ndarray | None = None
        consumed: list[str] = []
        for expert_id in range(n_experts):
            prefix = (
                f"model.layers.{bid}.block_sparse_moe.experts."
                f"{expert_id}.{projection}"
            )
            weight_name = prefix + ".weight_packed"
            scale_name = prefix + ".weight_scale"
            if weight_name not in self.model_tensors or scale_name not in self.model_tensors:
                raise KeyError(f"Missing routed expert tensors for {prefix}")
            weight = LazyTorchTensor.to_eager(self.model_tensors[weight_name]())
            scale = LazyTorchTensor.to_eager(self.model_tensors[scale_name]())
            packed = self._pack_mxfp4_blocks(weight, scale)
            if data is None:
                data = np.empty((n_experts, *packed.shape), dtype=packed.dtype)
            data[expert_id] = packed
            consumed.extend((weight_name, scale_name))

        assert data is not None
        new_name = self.format_tensor_name(tensor_key, bid)
        shape = gguf.quant_shape_from_byte_shape(
            data.shape, gguf.GGMLQuantizationType.MXFP4
        )
        logger.info(
            f"{new_name}: repacked routed experts to MXFP4, "
            f"shape = {{{', '.join(str(n) for n in reversed(shape))}}}"
        )
        self.gguf_writer.add_tensor(
            new_name, data, raw_dtype=gguf.GGMLQuantizationType.MXFP4
        )
        return consumed

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if not self._is_mxfp4 or self._mxfp4_experts_written:
            return ()

        consumed: list[str] = []
        for bid in range(self.hparams["first_k_dense_replace"], self.block_count):
            consumed.extend(self._write_mxfp4_expert_tensor(
                bid, "w1", gguf.MODEL_TENSOR.FFN_GATE_EXP))
            consumed.extend(self._write_mxfp4_expert_tensor(
                bid, "w2", gguf.MODEL_TENSOR.FFN_DOWN_EXP))
            consumed.extend(self._write_mxfp4_expert_tensor(
                bid, "w3", gguf.MODEL_TENSOR.FFN_UP_EXP))
        for name in consumed:
            del self.model_tensors[name]
        self._mxfp4_experts_written = True
        return ()

    def set_vocab(self):
        try:
            self._set_vocab_gpt2()
            return
        except Exception:
            pass

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        tokpre = self.get_vocab_base_pre(tokenizer)

        if tokpre == "kimi-k2":
            # Build merges list using the approach similar to HunYuanMoE
            merges = []
            vocab = {}
            mergeable_ranks = tokenizer.model._mergeable_ranks  # ty: ignore[unresolved-attribute]
            for token, rank in mergeable_ranks.items():
                vocab[QwenModel.token_bytes_to_string(token)] = rank
                if len(token) == 1:
                    continue
                merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
                if len(merged) == 2:
                    merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))
            # Build token list
            vocab_size = self.hparams["vocab_size"]
            special_tokens = tokenizer.special_tokens  # ty: ignore[unresolved-attribute]
            reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in {**vocab, **special_tokens}.items()}
            tokens: list[str] = []
            toktypes: list[int] = []

            for i in range(vocab_size):
                if i not in reverse_vocab:
                    tokens.append(f"[PAD{i}]")
                    toktypes.append(gguf.TokenType.UNUSED)
                else:
                    token = reverse_vocab[i]
                    tokens.append(token)
                    if i in special_tokens.values():
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        toktypes.append(gguf.TokenType.NORMAL)

            self.gguf_writer.add_tokenizer_model("gpt2")
            self.gguf_writer.add_tokenizer_pre(tokpre)
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_types(toktypes)
            self.gguf_writer.add_token_merges(merges)

            special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
            special_vocab.add_to_gguf(self.gguf_writer)
            # override eos id in config.json with tiktoken eos id
            self.gguf_writer.add_eos_token_id(tokenizer.eos_id)  # ty: ignore[unresolved-attribute]
        else:
            raise NotImplementedError(f"Deepseek pre-tokenizer {tokpre!r} is not supported yet!")

    def set_gguf_parameters(self):
        # note: To enable MLA KV cache, attention needs to be converted into MQA (ie: GQA with 1 group)
        self.hparams["num_key_value_heads"] = 1

        super().set_gguf_parameters()
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])

        # KDA & MLA params
        # Get ssm_d_conv from linear_attn_config.short_conv_kernel_size or ssm_d_conv
        linear_attn_config = self.hparams["linear_attn_config"]
        # n_head == 0 for KDA layers, n_head > 0 for MLA layers
        # full_attention_layers list will be used to distinguish layer type
        _num_kv_heads = list()
        _full_attn_layers = linear_attn_config["full_attn_layers"]
        for il in range(self.hparams["num_hidden_layers"]):
            if il + 1 in _full_attn_layers:
                _num_kv_heads.append(self.hparams["num_key_value_heads"])
            else:
                _num_kv_heads.append(0)
        assert len(_num_kv_heads) == self.hparams["num_hidden_layers"]
        self.gguf_writer.add_head_count_kv(_num_kv_heads)

        if (ssm_d_conv := linear_attn_config.get("short_conv_kernel_size")) is not None:
            self.gguf_writer.add_ssm_conv_kernel(ssm_d_conv)
        if (kda_head_dim := linear_attn_config.get("head_dim")) is not None:
            self.gguf_writer.add_kda_head_dim(kda_head_dim)

        # MLA params - use add_* methods that handle arch substitution
        # Support both HuggingFace naming (q_lora_rank, kv_lora_rank) and internal naming (n_lora_q, n_lora_kv)
        if (q_lora_rank := self.find_hparam(["q_lora_rank", "n_lora_q"], optional=True)) is not None:
            self.gguf_writer.add_q_lora_rank(q_lora_rank)
        # To enable MLA KV cache, MLA needs to be converted into MQA with larger heads, then decompresses to MHA
        kv_lora_rank = self.find_hparam(["kv_lora_rank", "n_lora_kv"], optional=False)
        self.gguf_writer.add_kv_lora_rank(kv_lora_rank)

        # MLA head dimensions
        # Support HuggingFace naming: qk_nope_head_dim, qk_rope_head_dim, v_head_dim
        qk_nope_head_dim = self.hparams.get("qk_nope_head_dim")
        # Rotation - use qk_rope_head_dim for Kimi
        qk_rope_head_dim = self.find_hparam(["qk_rope_head_dim", "n_rot"], optional=False)
        self.gguf_writer.add_rope_dimension_count(qk_rope_head_dim)
        self.gguf_writer.add_key_length(kv_lora_rank + qk_rope_head_dim)
        v_head_dim = self.hparams.get("v_head_dim")

        # Calculate n_embd_head_k_mla = qk_nope_head_dim + qk_rope_head_dim
        if (n_embd_head_k_mla := self.find_hparam(["n_embd_head_k_mla"], optional=True)) is not None:
            self.gguf_writer.add_key_length_mla(n_embd_head_k_mla)
        elif qk_nope_head_dim is not None:
            n_embd_head_k_mla = qk_nope_head_dim + qk_rope_head_dim
            self.gguf_writer.add_key_length_mla(n_embd_head_k_mla)

        # n_embd_head_v_mla = v_head_dim
        if (n_embd_head_v_mla := self.hparams.get("n_embd_head_v_mla")) is not None:
            self.gguf_writer.add_value_length_mla(n_embd_head_v_mla)
        elif v_head_dim is not None:
            self.gguf_writer.add_value_length_mla(v_head_dim)

        # moe_intermediate_size (1024 for Kimi)
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])
        # num_shared_experts (1 for Kimi)
        self.gguf_writer.add_expert_shared_count(self.hparams["num_shared_experts"])
        # first_k_dense_replace (1 for Kimi - first layer uses dense MLP)
        self.gguf_writer.add_leading_dense_block_count(self.hparams["first_k_dense_replace"])
        # Routed scaling factor (expert_weights_scale = 2.446 for Kimi)
        self.gguf_writer.add_expert_weights_scale(self.hparams["routed_scaling_factor"])
        if (latent_size := self.hparams.get("routed_expert_hidden_size")) is not None:
            self.gguf_writer.add_moe_latent_size(latent_size)
        if (block_size := self.hparams.get("attn_res_block_size")) is not None:
            self.gguf_writer.add_attention_residual_block_size(block_size)
        if (hidden_act := self.hparams.get("hidden_act")) is not None:
            self.gguf_writer.add_hidden_act(hidden_act)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        logger.info(f"Processing {name}: shape before = {tuple(data_torch.shape)}")

        # Handle KDA conv1d weights
        # HuggingFace/vLLM stores as [d_inner, d_conv] (2D), memory layout: conv_step changes fastest
        # llama.cpp expects ggml ne = [d_conv, 1, d_inner, 1], memory layout: ne[0]=d_conv changes fastest
        # GGUF reverses numpy shape when writing, so numpy (1, d_inner, 1, d_conv) -> ggml ne = [d_conv, 1, d_inner, 1]
        # Memory layouts match: both have conv_step (d_conv) changing fastest
        if name.endswith((".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")):
            # HF shape: [d_inner, d_conv] e.g. [4096, 4]
            # Target numpy shape: (1, d_inner, 1, d_conv) -> ggml ne = [d_conv, 1, d_inner, 1]
            if data_torch.ndim == 2:
                d_inner, d_conv = data_torch.shape
                # Reshape to (1, d_inner, 1, d_conv) - memory layout preserved (d_conv fastest)
                data_torch = data_torch.reshape(1, d_inner, 1, d_conv)
                logger.info(f"Reshaped conv1d weight {name}: [d_inner={d_inner}, d_conv={d_conv}] -> numpy {tuple(data_torch.shape)} -> ggml ne=[{d_conv}, 1, {d_inner}, 1]")
            elif data_torch.ndim == 3:
                # Already 3D [d_inner, 1, d_conv] from unsqueeze
                d_inner, _, d_conv = data_torch.shape
                data_torch = data_torch.reshape(1, d_inner, 1, d_conv)
                logger.info(f"Reshaped conv1d weight {name}: [d_inner={d_inner}, 1, d_conv={d_conv}] -> numpy {tuple(data_torch.shape)} -> ggml ne=[{d_conv}, 1, {d_inner}, 1]")

        # Handle A_log: iHF stores as [1, 1, num_heads, 1]
        # llama.cpp expects ggml ne = [1, num_heads, 1, 1]
        # GGUF reverses numpy shape: numpy (1, 1, num_heads, 1) -> ggml ne = [1, num_heads, 1, 1]
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch)
        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"
            logger.info("Changed dt_bias to dt_proj.bias")

        # process the experts separately
        if name.find("block_sparse_moe.experts") != -1:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                # w1: gate, w2: down, w3: up
                for wid, tname in [("w1", gguf.MODEL_TENSOR.FFN_GATE_EXP),
                                   ("w2", gguf.MODEL_TENSOR.FFN_DOWN_EXP),
                                   ("w3", gguf.MODEL_TENSOR.FFN_UP_EXP)]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]
                    data_torch = torch.stack(datas, dim=0)
                    new_name = self.format_tensor_name(tname, bid)
                    yield from super().modify_tensors(data_torch, new_name, bid)
            return

        # note: MLA with the absorption optimization, needs these two split and k_b_proj transposed
        if name.endswith("kv_b_proj.weight"):
            name_kb = name.replace("kv_b_proj", "k_b_proj")
            name_vb = name.replace("kv_b_proj", "v_b_proj")
            n_head_kv = self.hparams["num_key_value_heads"]
            v_head_dim = self.find_hparam(["n_embd_head_v_mla", "v_head_dim"], optional=False)
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
            logger.info("Split kv_b n_head_kv %d\n" % n_head_kv)
            assert data_torch.shape[0] == n_head_kv * (v_head_dim + qk_nope_head_dim)
            kv_b = data_torch.view(n_head_kv, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)
            yield from super().modify_tensors(k_b, name_kb, bid)
            yield from super().modify_tensors(v_b, name_vb, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)
