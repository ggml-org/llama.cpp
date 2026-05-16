from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("QWenLMHeadModel")
class QwenModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN

    @staticmethod
    def token_bytes_to_string(b):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode  # ty: ignore[unresolved-import]
        byte_encoder = bytes_to_unicode()
        return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])

    @staticmethod
    def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int | None = None) -> list[bytes]:
        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
        return parts

    def set_vocab(self):
        self._set_vocab_qwen()


@ModelBase.register(
    "Qwen2Model",
    "Qwen2ForCausalLM",
    "Qwen2AudioForConditionalGeneration",
    "KORMoForCausalLM",
    "AudioFlamingo3ForConditionalGeneration",
    "DotsOCRForCausalLM",
)
class Qwen2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN2

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self._try_set_pooling_type()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if self.hf_arch == "Qwen2Model":
            name = f"model.{name}"  # map to Qwen2ForCausalLM tensors
        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Qwen2MoeForCausalLM")
class Qwen2MoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN2MOE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")
        if (shared_expert_intermediate_size := self.hparams.get('shared_expert_intermediate_size')) is not None:
            self.gguf_writer.add_expert_shared_feed_forward_length(shared_expert_intermediate_size)
            logger.info(f"gguf: expert shared feed forward length = {shared_expert_intermediate_size}")

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # handle aggregated expert tensors
        # GGUF stores dimensions reversed from PyTorch, so:
        # PyTorch (A,B,C) -> GGUF writes [C,B,A] -> GGML reads ne={C,B,A}
        # Input shapes from HF: (n_expert, n_ff_exp, n_embd) or (n_expert, n_embd, n_ff_exp)
        # Expected GGML ne: {n_embd, n_ff_exp, n_expert} for gate/up, {n_ff_exp, n_embd, n_expert} for down
        if name.endswith("mlp.experts.down_proj") or name.endswith("mlp.experts.down_proj.weight"):
            mapped = f"{name}.weight" if not name.endswith(".weight") else name
            # HF: [n_expert, n_embd, n_ff] -> GGML: {n_ff, n_embd, n_expert}
            yield from super().modify_tensors(data_torch, mapped, bid)
            return

        if name.endswith("mlp.experts.gate_up_proj") or name.endswith("mlp.experts.gate_up_proj.weight"):
            if data_torch.ndim < 3 or data_torch.shape[-2] % 2 != 0:
                raise ValueError(f"Unexpected gate_up_proj shape for {name}: {tuple(data_torch.shape)}")
            # HF: [n_expert, 2*n_ff, n_embd] -> split on dim=-2
            n_ff = data_torch.shape[-2] // 2
            gate = data_torch[..., :n_ff, :].contiguous()
            up = data_torch[..., n_ff:, :].contiguous()
            # gate/up: [n_expert, n_ff, n_embd] -> GGML: {n_embd, n_ff, n_expert}
            base_name = name.removesuffix(".weight").removesuffix(".gate_up_proj")
            mapped_gate = f"{base_name}.gate_proj.weight"
            mapped_up = f"{base_name}.up_proj.weight"
            yield from super().modify_tensors(gate, mapped_gate, bid)
            yield from super().modify_tensors(up, mapped_up, bid)
            return

        if name.find("experts") != -1:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
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


@ModelBase.register("Qwen3ForCausalLM", "Qwen3Model")
class Qwen3Model(Qwen2Model):
    model_arch = gguf.MODEL_ARCH.QWEN3

    # extra logic for rerank models
    is_rerank: bool = False
    is_tied_embeddings: bool = False
    token_false_id: int | None = None
    token_true_id: int | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # track for intern-s1-mini
        hparams = ModelBase.load_hparams(self.dir_model, is_mistral_format=False)
        self.origin_hf_arch = hparams.get('architectures', [None])[0]

        if self._is_qwen3_reranker():
            self._find_rerank_config()

    def _is_qwen3_reranker(self) -> bool:
        readme_path = self.dir_model / "README.md"
        readme_text = ""
        if readme_path.exists():
            with readme_path.open("r", encoding="utf-8") as f:
                readme_text = f.read()

        name_hints = [
            str(self.dir_model.name),
            str(self.hparams.get("_name_or_path", "")),
            str(self.hparams.get("model_type", "")),
            str(self.origin_hf_arch or ""),
        ]
        name_hints = [hint.lower() for hint in name_hints if hint]

        if "# qwen3-reranker" in readme_text.lower() or "# qwen3-vl-reranker" in readme_text.lower():
            return True

        if any("qwen3-reranker" in hint or "qwen3-vl-reranker" in hint for hint in name_hints):
            return True

        return "sequenceclassification" in (self.origin_hf_arch or "").lower()

    def set_vocab(self):
        # deal with intern-s1-mini
        if self.origin_hf_arch == 'InternS1ForConditionalGeneration':
            self._set_vocab_interns1()
            return

        super().set_vocab()

    def _find_rerank_config(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)

        self.is_rerank = True
        self.is_tied_embeddings = self.hparams.get("tie_word_embeddings", False)
        self.token_false_id = tokenizer.convert_tokens_to_ids("no")  # ty: ignore[unresolved-attribute, invalid-assignment]
        self.token_true_id = tokenizer.convert_tokens_to_ids("yes")  # ty: ignore[unresolved-attribute, invalid-assignment]
        self.sep_token_id = tokenizer.convert_tokens_to_ids("|")  # ty: ignore[unresolved-attribute]

        assert self.token_false_id is not None and self.token_true_id is not None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if self.is_rerank:
            self.gguf_writer.add_pooling_type(gguf.PoolingType.RANK)
            self.gguf_writer.add_classifier_output_labels(["yes", "no"])
            self.gguf_writer.add_chat_template([{
                "name": "rerank",
                "template": "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                            "<|im_start|>user\n<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: {document}<|im_end|>\n"
                            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            }])

    def _get_cls_out_tensor(self, data_torch: Tensor) -> Tensor:
        # extract "yes" and "no" tokens from the output lm_head tensor
        false_row = data_torch[self.token_false_id]
        true_row = data_torch[self.token_true_id]
        return torch.stack([true_row, false_row], dim=0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if self.is_rerank:
            is_tied_head = self.is_tied_embeddings and "embed_tokens" in name
            is_real_head = not self.is_tied_embeddings and "lm_head" in name
            if is_tied_head or is_real_head:
                cls_out_head = (
                    gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.CLS_OUT] + ".weight",
                    self._get_cls_out_tensor(data_torch),
                )
                yield cls_out_head
                if is_tied_head:
                    yield from super().modify_tensors(data_torch, name, bid)
                return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Qwen3MoeForCausalLM")
class Qwen3MoeModel(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN3MOE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hparams = ModelBase.load_hparams(self.dir_model, False)
        self.origin_hf_arch = hparams.get('architectures', [None])[0]

    def set_vocab(self):
        # deal with intern-s1
        if self.origin_hf_arch == 'InternS1ForConditionalGeneration':
            self._set_vocab_interns1()
            return

        super().set_vocab()


@ModelBase.register("Qwen3NextForCausalLM")
class Qwen3NextModel(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN3NEXT

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_ssm_conv_kernel(self.hparams["linear_conv_kernel_dim"])
        self.gguf_writer.add_ssm_state_size(self.hparams["linear_key_head_dim"])
        self.gguf_writer.add_ssm_group_count(self.hparams["linear_num_key_heads"])
        self.gguf_writer.add_ssm_time_step_rank(self.hparams["linear_num_value_heads"])
        self.gguf_writer.add_ssm_inner_size(self.hparams["linear_value_head_dim"] * self.hparams["linear_num_value_heads"])
        self.gguf_writer.add_full_attention_interval(self.hparams.get("full_attention_interval", 4))
        if (rope_dim := self.hparams.get("head_dim")) is None:
            rope_dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rope_dim * self.hparams.get("partial_rotary_factor", 0.25)))

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith("mtp"):
            # ignore MTP layers for now
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch)
        elif name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"
        elif "conv1d" in name:
            data_torch = data_torch.squeeze()
        elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            data_torch = data_torch + 1

        if "in_proj_qkvz.weight" in name:
            # original order:  [q, k, v, z] * head_count
            # corrected order: [q * head_count, k * head_count, v * head_count, z * head_count]
            head_k_dim = self.hparams["linear_key_head_dim"]
            head_v_dim = self.hparams["linear_value_head_dim"]
            num_v_heads = self.hparams["linear_num_value_heads"]
            num_k_heads = self.hparams["linear_num_key_heads"]
            hidden_size = self.hparams["hidden_size"]
            split_arg_list_qkvz = [
                head_k_dim, # q partition
                head_k_dim, # k partition
                (num_v_heads // num_k_heads * head_v_dim), # v partition
                (num_v_heads // num_k_heads * head_v_dim), # z partition
            ]
            # view as (n_embd, head_count, [q+k+v+z])
            data_torch = data_torch.permute(1, 0).contiguous()
            data_torch = data_torch.view(-1, num_k_heads, sum(split_arg_list_qkvz))
            # split into q, k, v, z
            q, k, v, z = torch.split(data_torch, split_arg_list_qkvz, dim=-1)
            # flatten dim + head_count
            q = q.contiguous().view(hidden_size, -1)
            k = k.contiguous().view(hidden_size, -1)
            v = v.contiguous().view(hidden_size, -1)
            z = z.contiguous().view(hidden_size, -1)
            # stack back
            qkv = torch.cat([q, k, v], dim=-1).permute(1, 0).contiguous()
            z = z.permute(1, 0).contiguous()
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_QKV,  bid, ".weight"), qkv)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_GATE, bid, ".weight"), z)
        else:
            yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("RND1")
class RND1Model(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.RND1

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # RND1 specific parameters
        # RND1 uses bidirectional attention
        self.gguf_writer.add_causal_attention(False)

        if (mask_token_id := self.hparams.get("mask_token_id")) is not None:
            self.gguf_writer.add_mask_token_id(mask_token_id)


class _LinearAttentionVReorderBase(Qwen3NextModel):
    model_arch = gguf.MODEL_ARCH.QWEN3NEXT  # overridden by subclasses
    """reorders V heads from grouped to tiled order for ggml broadcast

    see https://github.com/ggml-org/llama.cpp/pull/19468#discussion_r2786394306

    Linear attention may has num_k_heads < num_v_heads. The HF weights store
    V heads grouped by K head: [G0_v0..v{r-1}, G1_v0..v{r-1}, ...].
    ggml binary ops use tiled broadcast: [K0, K1, ..., K0, K1, ...].
    We reorder V heads to tiled order so ggml_repeat can replace the expensive
    interleaved repeat: [G0_v0, G1_v0, ..., G0_v1, G1_v1, ...].
    """

    @staticmethod
    def _reorder_v_heads(tensor: Tensor, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int) -> Tensor:
        """Reorder V heads from grouped (by K head) to tiled order along the given dimension."""
        shape = list(tensor.shape)
        if dim < 0:
            dim += len(shape)
        new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
        tensor = tensor.reshape(*new_shape)
        perm = list(range(len(new_shape)))
        perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
        return tensor.permute(*perm).contiguous().reshape(*shape)

    def _transform_nvfp4_weight(self, name: str, weight: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
        if not name.endswith((
            ".linear_attn.in_proj_qkv.weight",
            ".linear_attn.in_proj_z.weight",
            ".linear_attn.in_proj_a.weight",
            ".linear_attn.in_proj_b.weight",
            ".linear_attn.out_proj.weight",
        )):
            return weight, scale

        num_k_heads = self.hparams["linear_num_key_heads"]
        num_v_heads = self.hparams["linear_num_value_heads"]
        head_k_dim = self.hparams["linear_key_head_dim"]
        head_v_dim = self.hparams["linear_value_head_dim"]
        num_v_per_k = num_v_heads // num_k_heads

        def unpack_nibbles(qs: Tensor) -> Tensor:
            lo = torch.bitwise_and(qs, 0x0F)
            hi = torch.bitwise_right_shift(qs, 4)
            return torch.stack((lo, hi), dim=-1).reshape(*qs.shape[:-1], qs.shape[-1] * 2)

        def pack_nibbles(codes: Tensor) -> Tensor:
            codes = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
            lo = torch.bitwise_and(codes[..., 0], 0x0F)
            hi = torch.bitwise_left_shift(torch.bitwise_and(codes[..., 1], 0x0F), 4)
            return torch.bitwise_or(lo, hi).contiguous()

        def apply_col_perm(qs: Tensor, scales: Tensor, col_perm: Tensor) -> tuple[Tensor, Tensor]:
            assert qs.ndim >= 2
            assert scales.ndim >= 2

            k = qs.shape[-1] * 2
            assert col_perm.numel() == k
            assert k % 16 == 0

            group_cols = col_perm.reshape(-1, 16)
            group_starts = group_cols[:, 0]
            expected = group_starts.unsqueeze(1) + torch.arange(16, dtype=col_perm.dtype)
            assert torch.equal(group_cols, expected)
            assert torch.all(group_starts % 16 == 0)

            group_perm = (group_starts // 16).to(dtype=torch.long)
            expected_groups = torch.arange(scales.shape[-1], dtype=torch.long)
            assert group_perm.numel() == scales.shape[-1]
            assert torch.equal(torch.sort(group_perm).values, expected_groups)

            codes = unpack_nibbles(qs)
            codes = codes.index_select(-1, col_perm.to(device=qs.device, dtype=torch.long))
            qs = pack_nibbles(codes)
            scales = scales.index_select(-1, group_perm.to(device=scales.device))
            return qs, scales

        def reorder_rows(qs: Tensor, scales: Tensor, head_dim: int) -> tuple[Tensor, Tensor]:
            row_perm = self._reorder_v_heads(
                torch.arange(num_v_heads * head_dim, dtype=torch.long).unsqueeze(-1),
                0, num_k_heads, num_v_per_k, head_dim,
            ).squeeze(-1)
            return (
                qs.index_select(0, row_perm.to(device=qs.device)),
                scales.index_select(0, row_perm.to(device=scales.device)),
            )

        if name.endswith(".linear_attn.in_proj_qkv.weight"):
            q_dim = head_k_dim * num_k_heads
            k_dim = head_k_dim * num_k_heads
            q = weight[:q_dim]
            k = weight[q_dim:q_dim + k_dim]
            v = weight[q_dim + k_dim:]
            q_scale = scale[:q_dim]
            k_scale = scale[q_dim:q_dim + k_dim]
            v_scale = scale[q_dim + k_dim:]
            v, v_scale = reorder_rows(v, v_scale, head_v_dim)
            return torch.cat([q, k, v], dim=0), torch.cat([q_scale, k_scale, v_scale], dim=0)

        if name.endswith(".linear_attn.in_proj_z.weight"):
            weight, scale = reorder_rows(weight, scale, head_v_dim)
        elif name.endswith((".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight")):
            weight, scale = reorder_rows(weight, scale, 1)
        elif name.endswith(".linear_attn.out_proj.weight"):
            col_perm = self._reorder_v_heads(
                torch.arange(num_v_heads * head_v_dim, dtype=torch.long).unsqueeze(0),
                1, num_k_heads, num_v_per_k, head_v_dim,
            ).squeeze(0)
            weight, scale = apply_col_perm(weight, scale, col_perm)

        return weight, scale

    def _repack_nvfp4(self, name: str, weight: Tensor, scale: Tensor, scale2: Tensor, input_scale: Tensor):
        weight, scale = self._transform_nvfp4_weight(name, weight, scale)
        super()._repack_nvfp4(name, weight, scale, scale2, input_scale)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        num_k_heads = self.hparams.get("linear_num_key_heads", 0)
        num_v_heads = self.hparams.get("linear_num_value_heads", 0)

        if num_k_heads > 0 and num_v_heads > 0 and num_k_heads != num_v_heads and "linear_attn." in name:
            head_k_dim = self.hparams["linear_key_head_dim"]
            head_v_dim = self.hparams["linear_value_head_dim"]
            num_v_per_k = num_v_heads // num_k_heads

            if ".in_proj_qkv." in name:
                # QKV weight: reorder only the V rows
                q_dim = head_k_dim * num_k_heads
                k_dim = head_k_dim * num_k_heads
                q = data_torch[:q_dim]
                k = data_torch[q_dim:q_dim + k_dim]
                v = data_torch[q_dim + k_dim:]
                v = self._reorder_v_heads(v, 0, num_k_heads, num_v_per_k, head_v_dim)
                data_torch = torch.cat([q, k, v], dim=0)

            elif ".in_proj_z." in name:
                # Z gate weight: reorder rows (num_v_heads * head_v_dim)
                data_torch = self._reorder_v_heads(data_torch, 0, num_k_heads, num_v_per_k, head_v_dim)

            elif ".in_proj_b." in name or ".in_proj_a." in name:
                # Beta/Alpha weight: reorder rows (num_v_heads, head_dim=1)
                data_torch = self._reorder_v_heads(data_torch, 0, num_k_heads, num_v_per_k, 1)

            elif ".A_log" in name or ".dt_bias" in name or ".dt_proj" in name:
                # A_log / dt_bias: 1D parameters with num_v_heads elements
                if data_torch.ndim == 1:
                    data_torch = self._reorder_v_heads(
                        data_torch.unsqueeze(-1), 0, num_k_heads, num_v_per_k, 1
                    ).squeeze(-1)
                else:
                    data_torch = self._reorder_v_heads(data_torch, -1, num_k_heads, num_v_per_k, 1)

            elif ".conv1d" in name:
                # Conv1d kernel: reorder only the V channel portion
                data = data_torch.squeeze()
                qk_channels = head_k_dim * num_k_heads * 2
                qk_part = data[:qk_channels]
                v_part = data[qk_channels:]
                v_part = self._reorder_v_heads(v_part, 0, num_k_heads, num_v_per_k, head_v_dim)
                data_torch = torch.cat([qk_part, v_part], dim=0)

            elif ".out_proj." in name:
                # Out projection weight: reorder columns (input dimension)
                data_torch = self._reorder_v_heads(data_torch, 1, num_k_heads, num_v_per_k, head_v_dim)

        yield from super().modify_tensors(data_torch, name, bid)


class _Qwen35MRopeMixin:
    # Qwen3.5 always applies interleaved MRoPE (see Qwen3_5RotaryEmbedding in transformers);
    # the upstream default mrope_section is [11, 11, 10] and llama.cpp's QWEN35 / QWEN35MOE
    # loaders treat qwen35.rope.dimension_sections as required, so make sure it is always
    # written even when a particular checkpoint omits the field in `rope_parameters`.
    _QWEN35_DEFAULT_MROPE_SECTION = [11, 11, 10, 0]

    gguf_writer: gguf.GGUFWriter
    rope_parameters: dict

    def set_gguf_parameters(self):
        super().set_gguf_parameters()  # ty: ignore[unresolved-attribute]
        if "mrope_section" not in self.rope_parameters:
            self.gguf_writer.add_rope_dimension_sections(self._QWEN35_DEFAULT_MROPE_SECTION)


class _Qwen35MtpMixin:
    """Shared MTP wiring for Qwen3.5/3.6 text variants. The HF config carries
    the MTP block under `mtp_num_hidden_layers` and the tensors under
    `mtp.*`; we extend block_count, emit the nextn metadata key, and remap
    `mtp.*` to the standard layer-indexed nextn naming so the existing
    tensor_map handles them."""

    hparams: dict[str, Any]
    model_arch: gguf.MODEL_ARCH
    gguf_writer: gguf.GGUFWriter
    block_count: int
    tensor_map: gguf.TensorNameMap
    no_mtp: bool
    mtp_only: bool

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_count = self.hparams["num_hidden_layers"]
        if not self.no_mtp:
            self.block_count += self.hparams.get("mtp_num_hidden_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    @classmethod
    def filter_tensors(cls, item):
        name, _ = item
        if name.startswith("mtp."):
            if cls.no_mtp:
                return None
            return item
        if cls.mtp_only:
            canonical = name.replace("language_model.", "")
            keep = canonical in (
                "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
                "embed_tokens.weight", "norm.weight",
            )
            if not keep:
                return None
        return super().filter_tensors(item)  # ty: ignore[unresolved-attribute]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()  # ty: ignore[unresolved-attribute]
        if self.no_mtp:
            return
        if (n := self.hparams.get("mtp_num_hidden_layers", 0)) > 0:
            self.gguf_writer.add_nextn_predict_layers(n)

    def prepare_metadata(self, vocab_only: bool):
        from_dir = self.fname_out.is_dir()
        super().prepare_metadata(vocab_only=vocab_only)  # ty: ignore[unresolved-attribute]

        if not self.mtp_only or not from_dir:
            return

        output_type: str = self.ftype.name.partition("_")[2]  # pyright: ignore[reportAttributeAccessIssue] # ty: ignore[unresolved-attribute]
        fname_default: str = gguf.naming_convention(
            self.metadata.name, self.metadata.basename, self.metadata.finetune,                  # pyright: ignore[reportAttributeAccessIssue] # ty: ignore[unresolved-attribute]
            self.metadata.version, size_label=None, output_type=output_type, model_type=None)    # pyright: ignore[reportAttributeAccessIssue] # ty: ignore[unresolved-attribute]
        self.fname_out = self.fname_out.parent / f"mtp-{fname_default}.gguf"

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("mtp."):
            n_layer = self.hparams["num_hidden_layers"]
            if name.find("layers.") != -1:
                assert bid is not None
                name = name.replace(f"mtp.layers.{bid}", f"model.layers.{bid + n_layer}")
            else:
                remapper = {
                    "mtp.fc":                    "model.layers.{bid}.eh_proj",
                    "mtp.pre_fc_norm_embedding": "model.layers.{bid}.enorm",
                    "mtp.pre_fc_norm_hidden":    "model.layers.{bid}.hnorm",
                    "mtp.norm":                  "model.layers.{bid}.shared_head.norm",
                }
                stem   = Path(name).stem
                suffix = Path(name).suffix
                tmpl   = remapper[stem] + suffix
                for b in range(n_layer, self.block_count):
                    yield from super().modify_tensors(data_torch, tmpl.format(bid=b), b)  # ty: ignore[unresolved-attribute]
                return

        yield from super().modify_tensors(data_torch, name, bid)  # ty: ignore[unresolved-attribute]


@ModelBase.register("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")
class Qwen3_5TextModel(_Qwen35MtpMixin, _Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    model_arch = gguf.MODEL_ARCH.QWEN35


@ModelBase.register("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM")
class Qwen3_5MoeTextModel(_Qwen35MtpMixin, _Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    model_arch = gguf.MODEL_ARCH.QWEN35MOE
