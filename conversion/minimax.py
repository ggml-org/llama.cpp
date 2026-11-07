from __future__ import annotations

import json
import math
import re
from typing import Callable, Iterable, Sequence, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, MmprojModel, gguf, logger
from .qwen import DSparkModel


@ModelBase.register("MiniMaxText01ForCausalLM")
@ModelBase.register("MiniMaxM1ForCausalLM")
@ModelBase.example("MiniMaxAI/MiniMax-Text-01", "MiniMaxAI/MiniMax-M1-40k")
class MiniMaxText01Model(TextModel):
    model_arch = gguf.MODEL_ARCH.MINIMAX01

    def _get_suppress_tokens(self) -> Sequence[int] | None:
        import json
        from transformers import AutoTokenizer
        from .base import LazyTorchTensor

        # check added tokens embeddings in embeddings tensor for zero-valued embeddings
        # they get in the way of the token sampling process and must be suppressed

        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        tokenizer_vocab_size = tokenizer.vocab_size  # ty: ignore[unresolved-attribute]

        with open(self.dir_model / "model.safetensors.index.json", "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        embeddings_tensor_name = "model.embed_tokens.weight"
        embeddings_shard_name = weight_map[embeddings_tensor_name]
        with gguf.utility.SafetensorsLocal(self.dir_model / embeddings_shard_name) as model_shard:
            embeddings_data = model_shard[embeddings_tensor_name]

        embeddings_weights_dtype = LazyTorchTensor._dtype_str_map[embeddings_data.dtype]
        embeddings_weights = torch.from_numpy(embeddings_data.mmap_bytes()).view(embeddings_weights_dtype).reshape(embeddings_data.shape)
        embeddings_vocab_size = embeddings_weights.shape[0]

        embeddings_added_tokens = embeddings_weights[tokenizer_vocab_size:embeddings_vocab_size]
        embeddings_zero_rows = torch.all(embeddings_added_tokens == 0, dim=1)
        tokens_zero_embeddings_ids = (torch.nonzero(embeddings_zero_rows, as_tuple=False).flatten() + tokenizer_vocab_size).tolist()

        return tokens_zero_embeddings_ids

    def set_vocab(self) -> None:
        from pathlib import Path

        self._set_vocab_gpt2()

        for tmpl_file in [
            self.dir_model / "chat_template.jinja",
            Path(__file__).parent.parent / "models" / "templates" / "MiniMax-M1.jinja"
        ]:
            if tmpl_file.is_file():
                self.gguf_writer.add_chat_template(tmpl_file.read_text(encoding="utf-8"))
                logger.info(f"Chat template overridden with {tmpl_file}.")
                break

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        suppress_tokens = self._get_suppress_tokens()
        if suppress_tokens:
            logger.info(f"Suppressing tokens with zero embeddings {suppress_tokens}")
            self.gguf_writer.add_suppress_tokens(suppress_tokens)

        layernorm_full_attention_alpha = self.hparams["layernorm_full_attention_alpha"]
        layernorm_full_attention_beta = self.hparams["layernorm_full_attention_beta"]
        layernorm_linear_attention_alpha = self.hparams["layernorm_linear_attention_alpha"]
        layernorm_linear_attention_beta = self.hparams["layernorm_linear_attention_beta"]
        layernorm_mlp_alpha = self.hparams["layernorm_mlp_alpha"]
        layernorm_mlp_beta = self.hparams["layernorm_mlp_beta"]
        assert layernorm_full_attention_alpha == layernorm_linear_attention_alpha == layernorm_mlp_alpha
        assert layernorm_full_attention_beta == layernorm_linear_attention_beta == layernorm_mlp_beta == 1.0
        # we do not store the layernorm betas as they are all 1.0
        # layernorm alphas are stored as single residual_scale hparam
        self.gguf_writer.add_residual_scale(layernorm_full_attention_alpha)

        self.gguf_writer.add_rope_dimension_count(self.hparams["rotary_dim"])

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # process the experts separately
        if name.find("block_sparse_moe.experts") != -1:
            n_experts = self.hparams["num_local_experts"]

            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for wid in ["w1", "w2", "w3"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"layers.{bid}.feed_forward.experts.{wid}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    yield from super().modify_tensors(data_torch, new_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("MiniMaxM2ForCausalLM")
@ModelBase.example("MiniMaxAI/MiniMax-M2")
class MiniMaxM2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.MINIMAXM2
    _experts_cache: dict[int, dict[str, Tensor]] = {}

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_expert_feed_forward_length(self.find_hparam(["intermediate_size"]))
        self.gguf_writer.add_rope_dimension_count(self.find_hparam(["rotary_dim"]))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        # merge expert weights
        if "block_sparse_moe.experts." in name:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            expert_cache = self._experts_cache.setdefault(bid, {})
            expert_cache[name] = data_torch
            expert_weights = ["w1", "w2", "w3"]

            # not enough expert weights to merge
            if len(expert_cache) < n_experts * len(expert_weights):
                return

            for w_name in expert_weights:
                datas: list[Tensor] = []

                for xid in range(n_experts):
                    ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{w_name}.weight"
                    datas.append(expert_cache[ename])
                    del expert_cache[ename]

                data_torch = torch.stack(datas, dim=0)
                merged_name = f"model.layers.{bid}.block_sparse_moe.experts.{w_name}.weight"
                new_name = self.map_tensor_name(merged_name)
                yield from super().modify_tensors(data_torch, new_name, bid)

            del self._experts_cache[bid]
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration")
@ModelBase.example("MiniMaxAI/MiniMax-M3")
class MiniMaxM3Model(MiniMaxM2Model):
    model_arch = gguf.MODEL_ARCH.MINIMAXM3

    # --outtype auto keeps experts NVFP4 and writes MXFP8 tensors exactly as BF16 (or Q8_0 with --fp8-as-q8)
    _NVFP4_PROJ = {"w1": gguf.MODEL_TENSOR.FFN_GATE_EXP, "w2": gguf.MODEL_TENSOR.FFN_DOWN_EXP, "w3": gguf.MODEL_TENSOR.FFN_UP_EXP}

    def __init__(self, dir_model, ftype, *args, **kwargs):
        self._nvfp4_native = ftype == gguf.LlamaFileType.GUESSED
        super().__init__(dir_model, ftype, *args, **kwargs)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".indexer." in new_name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @staticmethod
    def _dequant_mxfp8(weight: Tensor, scale: Tensor) -> Tensor:
        # exact in fp32 for every E8M0 byte except 255 (NaN)
        rows, cols = weight.shape
        e = scale.view(torch.uint8).to(torch.float64)
        s = torch.exp2(e - 127.0).repeat_interleave(32, dim=1)[:, :cols]
        return (weight.to(torch.float64) * s).to(torch.float32).reshape(rows, cols)

    _e2m1_pair_lut: Tensor | None = None

    @classmethod
    def _dequant_nvfp4_eager(cls, packed: Tensor, scale: Tensor, global_scale: Tensor) -> Tensor:
        # exact: evaluated in fp64, rounded once to fp32
        if cls._e2m1_pair_lut is None:
            v = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
            v16 = v + [-x for x in v]
            # low nibble = element 2i, high nibble = 2i+1
            cls._e2m1_pair_lut = torch.tensor([[v16[b & 15], v16[b >> 4]] for b in range(256)], dtype=torch.float64)
        rows = packed.shape[0]
        vals = cls._e2m1_pair_lut[packed.view(torch.uint8).to(torch.int64)].reshape(rows, -1)
        s = scale.to(torch.float64).repeat_interleave(16, dim=1)
        return (vals * s / global_scale.to(torch.float64)).to(torch.float32)

    @classmethod
    def _dequant_nvfp4(cls, packed: Tensor, scale: Tensor, global_scale: Tensor) -> Tensor:
        if not isinstance(packed, LazyTorchTensor):
            return cls._dequant_nvfp4_eager(packed, scale, global_scale)
        rows, packed_cols = packed.shape
        # a lazy op chain costs ~10 s per expert tensor
        return LazyTorchTensor(meta=LazyTorchTensor.meta_with_dtype_and_shape(torch.float32, (rows, packed_cols * 2)),
                               args=(packed, scale, global_scale), func=cls._dequant_nvfp4_eager)

    @staticmethod
    def _nvfp4_pack_np(packed: Tensor, scale: Tensor) -> np.ndarray:
        # same bytes as ModelBase._nvfp4_pack
        p = packed.contiguous().view(torch.uint8).numpy()
        d = scale.contiguous().view(torch.uint8).numpy() & 0x7F
        rows, packed_cols = p.shape
        n_blocks = packed_cols // 8
        p = p.reshape(rows, n_blocks, 8)
        vals = np.stack((p & 0x0F, p >> 4), axis=-1).reshape(rows, n_blocks, 16)
        qs = vals[:, :, :8] | (vals[:, :, 8:] << 4)
        n_super = n_blocks // 4
        raw = np.concatenate([d.reshape(rows, n_super, 4), qs.reshape(rows, n_super, 32)], axis=-1)
        return raw.reshape(rows, n_super * 36)

    # experts are built lazily and in a thread pool: a serial loop takes hours on the real model
    _EXPERT_THREADS = 16
    _DEQUANT_RAW = {
        gguf.LlamaFileType.ALL_F32:     (gguf.GGMLQuantizationType.F32,  torch.float32,  4),
        gguf.LlamaFileType.MOSTLY_F16:  (gguf.GGMLQuantizationType.F16,  torch.float16,  2),
        gguf.LlamaFileType.MOSTLY_BF16: (gguf.GGMLQuantizationType.BF16, torch.bfloat16, 2),
    }

    def _expert_tensor(self, loaders: list[tuple[Callable[[], Tensor], ...]], native: bool):
        rows, packed_cols = loaders[0][0]().shape
        cols = packed_cols * 2
        if native:
            if cols % 64 != 0:
                raise ValueError(f"NVFP4 expert row has {cols} values, expected a multiple of 64")
            row_bytes = (cols // 64) * 36
        else:
            row_bytes = cols * self._DEQUANT_RAW[self.ftype][2]
        byte_shape = (len(loaders), rows, row_bytes)

        def one(fns) -> np.ndarray:
            t = [LazyTorchTensor.to_eager(f()) for f in fns]
            if native:
                return self._nvfp4_pack_np(t[0], t[1])
            w = self._dequant_nvfp4_eager(t[0], t[1], t[2])
            return w.to(self._DEQUANT_RAW[self.ftype][1]).contiguous().view(torch.uint8).numpy()

        def load(fns_list) -> np.ndarray:
            from concurrent.futures import ThreadPoolExecutor
            out = np.empty(byte_shape, dtype=np.uint8)
            with ThreadPoolExecutor(self._EXPERT_THREADS) as ex:
                for eid, arr in enumerate(ex.map(one, fns_list)):
                    out[eid] = arr
            return out

        return gguf.LazyNumpyTensor(
            meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(np.uint8, byte_shape),
            args=(loaders,),
            func=load,
        )

    def dequant_model(self):
        qc = self.hparams.get("quantization_config") or {}
        if not isinstance(qc, dict) or qc.get("quant_method") != "mxfp8":
            return super().dequant_model()

        if list(qc.get("weight_block_size", [1, 32])) != [1, 32]:
            raise NotImplementedError(f"mxfp8 weight_block_size {qc.get('weight_block_size')!r} is not supported")

        remove: list[str] = []
        experts: dict[tuple[int, str], dict[int, str]] = {}
        n_input_scale = n_input_scale_one = 0

        for name in list(self.model_tensors.keys()):
            if name.endswith(".weight_scale_inv"):
                weight_name = name.removesuffix("_scale_inv")
                if weight_name not in self.model_tensors:
                    raise KeyError(f"{name} has no matching {weight_name}")
                w, s = self.model_tensors[weight_name], self.model_tensors[name]
                self.model_tensors[weight_name] = lambda w=w, s=s: self._dequant_mxfp8(w(), s())
                remove.append(name)
                if self._fp8_as_q8:
                    self._fp8_dequantized.add(weight_name)
            elif name.endswith(".weight_packed"):
                if qc.get("moe_quant_format") != "nvfp4-pack-quantized" or int(qc.get("group_size", 16)) != 16:
                    raise NotImplementedError(f"packed weight {name} with moe_quant_format {qc.get('moe_quant_format')!r}")
                base = name.removesuffix("_packed")            # ...experts.E.wX.weight
                module = base.removesuffix(".weight")
                for aux in (base + "_scale", base + "_global_scale"):
                    if aux not in self.model_tensors:
                        raise KeyError(f"{name}: missing {aux}")
                # trained activation scale; llama.cpp computes its own
                if (module + ".input_global_scale") in self.model_tensors:
                    n_input_scale += 1
                    if float(LazyTorchTensor.to_eager(self.model_tensors[module + ".input_global_scale"]()).float()) == 1.0:
                        n_input_scale_one += 1
                    remove.append(module + ".input_global_scale")
                m = re.search(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])$", module)
                if m is not None and (self._nvfp4_native or self.ftype in self._DEQUANT_RAW):
                    experts.setdefault((int(m.group(1)), m.group(3)), {})[int(m.group(2))] = base
                    continue
                p = self.model_tensors[name]
                s = self.model_tensors[base + "_scale"]
                g = self.model_tensors[base + "_global_scale"]
                self.model_tensors[base] = lambda p=p, s=s, g=g: MiniMaxM3Model._dequant_nvfp4(p(), s(), g())
                remove += [name, base + "_scale", base + "_global_scale"]

        if n_input_scale:
            logger.info(f"dropped {n_input_scale} NVFP4 input_global_scale tensors ({n_input_scale_one} equal to 1.0): "
                        "activation quantization is not part of the converted model")

        n_experts = self.find_hparam(["num_local_experts", "num_experts"])
        for (bid, wid), by_eid in sorted(experts.items()):
            if sorted(by_eid) != list(range(n_experts)):
                raise KeyError(f"layer {bid} {wid}: {len(by_eid)} of {n_experts} NVFP4 experts present")
            loaders = []
            inv_global: list[float] = []
            for eid in range(n_experts):
                base = by_eid[eid]
                loaders.append((self.model_tensors[base + "_packed"], self.model_tensors[base + "_scale"],
                                self.model_tensors[base + "_global_scale"]))
                g = float(LazyTorchTensor.to_eager(self.model_tensors[base + "_global_scale"]()).float())
                if not (g > 0.0 and math.isfinite(g)):
                    raise ValueError(f"{base}_global_scale = {g}")
                inv_global.append(1.0 / g)
                remove += [base + "_packed", base + "_scale", base + "_global_scale"]
            new_name = self.format_tensor_name(self._NVFP4_PROJ[wid], bid)
            qtype = gguf.GGMLQuantizationType.NVFP4 if self._nvfp4_native else self._DEQUANT_RAW[self.ftype][0]
            data = self._expert_tensor(loaders, self._nvfp4_native)
            shape = gguf.quant_shape_from_byte_shape(data.shape, qtype)
            logger.info(f"{new_name}: {n_experts} experts {'kept NVFP4' if self._nvfp4_native else 'dequantized to ' + qtype.name}, "
                        f"shape = {{{', '.join(str(n) for n in reversed(shape))}}}")
            self.gguf_writer.add_tensor(new_name, data, raw_dtype=qtype)
            if self._nvfp4_native:
                # per-expert 1/global, applied by build_moe_ffn
                self.gguf_writer.add_tensor(new_name.replace(".weight", ".scale"), np.array(inv_global, dtype=np.float32))

        for name in remove:
            self.model_tensors.pop(name, None)

    def _raw_text_config(self) -> dict:
        with open(self.dir_model / "config.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw.get("text_config", raw)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_expert_shared_count(self.find_hparam(["n_shared_experts"]))
        self.gguf_writer.add_expert_weights_scale(self.find_hparam(["routed_scaling_factor"]))
        self.gguf_writer.add_expert_weights_norm(True)

        # transformers >= 5.15 folds the sparse config and drops fields: read the raw file
        raw = self._raw_text_config()
        sac = raw.get("sparse_attention_config") or self.find_hparam(["sparse_attention_config"])
        for flat, legacy in (("index_n_heads", "sparse_num_index_heads"), ("index_head_dim", "sparse_index_dim"),
                             ("index_topk_blocks", "sparse_topk_blocks"), ("index_block_size", "sparse_block_size"),
                             ("index_local_blocks", "sparse_local_block")):
            if flat in self.hparams and self.hparams[flat] != sac[legacy]:
                raise ValueError(f"resolved {flat}={self.hparams[flat]} != config {legacy}={sac[legacy]}")
        # only the MSA variant every released config uses is implemented
        if sac.get("sparse_score_type", "max") != "max":
            raise NotImplementedError(f"sparse_score_type {sac['sparse_score_type']!r}")
        if int(sac.get("sparse_init_block", 0)) != 0:
            raise NotImplementedError(f"sparse_init_block {sac['sparse_init_block']}")
        # the loader marks MSA layers by the presence of blk.N.indexer.*
        freq = sac.get("sparse_attention_freq")
        n_layer = self.find_hparam(["num_hidden_layers"])
        if freq is not None:
            if len(freq) != n_layer:
                raise ValueError(f"sparse_attention_freq has {len(freq)} entries, expected {n_layer}")
            disable_v = sac.get("sparse_disable_index_value") or [1] * n_layer
            for il, f in enumerate(freq):
                has_idx = any(f"layers.{il}.self_attn.index_k_proj." in n for n in self.model_tensors)
                if bool(f) != has_idx:
                    raise ValueError(f"layer {il}: sparse_attention_freq={f} but indexer tensors present={has_idx}")
                if f and not disable_v[il]:
                    raise NotImplementedError(f"layer {il}: MSA index value path (sparse_disable_index_value=0)")
        self.gguf_writer.add_indexer_head_count(sac["sparse_num_index_heads"])
        self.gguf_writer.add_indexer_key_length(sac["sparse_index_dim"])
        self.gguf_writer.add_indexer_top_k(sac["sparse_topk_blocks"])
        self.gguf_writer.add_indexer_block_size(sac["sparse_block_size"])
        self.gguf_writer.add_indexer_local_blocks(sac["sparse_local_block"])

        moe_layer_freq = raw.get("moe_layer_freq") or self.find_hparam(["moe_layer_freq"])
        if "mlp_layer_types" in self.hparams:
            assert [t == "sparse" for t in self.hparams["mlp_layer_types"]] == [bool(v) for v in moe_layer_freq]
        n_dense = 0
        for v in moe_layer_freq:
            if v == 0:
                n_dense += 1
            else:
                break
        self.gguf_writer.add_leading_dense_block_count(n_dense)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        # Gemma-style (1 + w) RMSNorm: bake the +1 in so llama.cpp can use plain RMSNorm
        if name.endswith("norm.weight"):
            data_torch = data_torch.to(torch.float32) + 1.0

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("MiniMaxM3SparseForConditionalGeneration", "MiniMaxM3VLForConditionalGeneration")
@ModelBase.example("MiniMaxAI/MiniMax-M3")
class MiniMaxM3VisionModel(MmprojModel):
    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        # keep only the vision-side tensors; text / mtp / sparse-index are dropped
        if not name.startswith(("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")):
            return None
        return super().filter_tensors((name, gen))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.MINIMAXM3)
        self.gguf_writer.add_vision_use_gelu(True)

        # the ViT carries its own LayerNorm eps (text tower uses a different one)
        self.gguf_writer.add_vision_attention_layernorm_eps(
            self.hparams_vision.get("layer_norm_eps", 1e-5)
        )

        comp = self.hparams_vision.get("img_token_compression_config", {})
        merge_size = comp.get("spatial_merge_size", 2)
        self.gguf_writer.add_vision_spatial_merge_size(int(merge_size))

    def modify_tensors(self, data_torch, name, bid):
        assert self.hparams_vision is not None

        # Conv3d patch embed -> Conv2d slices
        if name == "vision_tower.vision_model.embeddings.patch_embedding.weight":
            if data_torch.ndim != 5:
                raise ValueError(f"unexpected patch_embedding rank {data_torch.ndim} for {name}")
            kt = data_torch.shape[2]
            base = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH]
            for t in range(kt):
                suffix = ".weight" if t == 0 else f".weight.{t}"
                yield (base + suffix, data_torch[:, :, t, ...])
            return

        # Permute ViT q/k. HF [Ta Ha Wa | Tb Hb Wb | pad] reorder to [Ta Tb | Ha Hb | Wa Wb | pad].
        for new_name, tensor in super().modify_tensors(data_torch, name, bid):
            if ".attn_q." in new_name or ".attn_k." in new_name:
                tensor = self._permute_vit_qk(tensor, new_name)
            yield new_name, tensor

    def _permute_vit_qk(self, t: "Tensor", new_name: str) -> "Tensor":
        assert self.hparams_vision is not None
        n_head = self.hparams_vision["num_attention_heads"]
        d_head = t.shape[0] // n_head
        axis_dim = 2 * ((2 * (d_head // 2) // 3) // 2)
        ah = axis_dim // 2
        half = 3 * ah
        perm = []
        perm += list(range(0, ah))
        perm += list(range(half, half + ah))
        perm += list(range(ah, 2 * ah))
        perm += list(range(half + ah, half + 2 * ah))
        perm += list(range(2 * ah, 3 * ah))
        perm += list(range(half + 2 * ah, half + 3 * ah))
        perm += list(range(2 * half, d_head))

        assert axis_dim % 2 == 0
        assert 3 * axis_dim <= d_head
        assert len(perm) == d_head
        assert sorted(perm) == list(range(d_head)), "perm is not a bijection of d_head"
        assert t.shape[0] == n_head * d_head, f"{new_name}: {t.shape[0]} != {n_head}*{d_head}"
        assert d_head == 80

        idx = torch.tensor(perm, dtype=torch.long)
        if t.ndim == 2:
            return t.reshape(n_head, d_head, t.shape[1])[:, idx, :].reshape(t.shape)
        return t.reshape(n_head, d_head)[:, idx].reshape(t.shape)


@ModelBase.register("DSparkMiniMaxDraftModel")
@ModelBase.example("MiniMaxAI/MiniMax-M3.1-Flash")
class DSparkMiniMaxModel(DSparkModel):
    # MiniMax-M3.1 DSpark draft; embeddings and lm_head come from the target
    model_arch = gguf.MODEL_ARCH.DFLASH

    _PREFIX = "language_model.model.dspark."

    def __init__(self, dir_model, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            # raw config.json: transformers >= 5.15 drops the DSpark keys
            with open(dir_model / "config.json", "r", encoding="utf-8") as f:
                hparams = json.load(f)
        hparams = {**hparams, **hparams.get("text_config", {})}
        hparams.pop("text_config", None)

        if not hparams.get("dspark_use_dense_ffn", True):
            raise ValueError("MiniMax DSpark: only the dense-FFN draft (dspark_use_dense_ffn) is supported")
        if hparams.get("dspark_hybrid_context_fusion", False):
            raise ValueError("MiniMax DSpark: hybrid_context_fusion is not supported")
        if hparams.get("hidden_act") != "swigluoai":
            raise ValueError(f"MiniMax DSpark: unexpected hidden_act {hparams.get('hidden_act')!r}")
        # llama.cpp's LLM_FFN_SWIGLU_OAI_MOE hard-codes the gpt-oss constants
        if float(hparams.get("swiglu_alpha", 1.702)) != 1.702 or float(hparams.get("swiglu_limit", 7.0)) != 7.0:
            raise ValueError("MiniMax DSpark: swiglu_alpha/limit differ from 1.702/7.0")
        if hparams.get("attention_output_gate", False):
            raise ValueError("MiniMax DSpark: attention_output_gate is not supported")
        if hparams.get("qk_norm_type", "per_head") != "per_head" or not hparams.get("use_qk_norm", True):
            raise ValueError("MiniMax DSpark: expects per-head QK norm")
        if not hparams.get("use_gemma_norm", True):
            raise ValueError("MiniMax DSpark: expects Gemma (1 + w) norms")
        if hparams.get("dspark_markov_head_type", hparams.get("markov_head_type", "vanilla")) != "vanilla":
            raise ValueError("MiniMax DSpark: only the vanilla Markov head is supported")
        layer_types = hparams.get("layer_types") or hparams.get("dflash_config", {}).get("layer_types") or []
        if any(lt != "sliding_attention" for lt in layer_types):
            raise ValueError(f"MiniMax DSpark: expects uniform sliding_attention layers, got {layer_types}")

        # dense FFN width, not the MoE expert width
        hparams["dense_intermediate_size"] = hparams.get("dspark_ffn_hidden_size", hparams["dense_intermediate_size"])
        for k in ("num_local_experts", "num_experts_per_tok", "n_shared_experts", "shared_intermediate_size",
                  "moe_intermediate_size", "num_experts", "scoring_func", "use_routing_bias", "routed_scaling_factor"):
            hparams.pop(k, None)

        dflash_config = dict(hparams.get("dflash_config", {}))
        dflash_config.setdefault("block_size", hparams["dspark_block_size"])
        dflash_config.setdefault("mask_token_id", hparams["dspark_noise_token_id"])
        dflash_config.setdefault("target_layer_ids", hparams["dspark_target_layer_ids"])
        if dflash_config["mask_token_id"] != hparams["dspark_noise_token_id"]:
            raise ValueError("MiniMax DSpark: dflash_config.mask_token_id != dspark_noise_token_id")
        if list(dflash_config["target_layer_ids"]) != list(hparams["dspark_target_layer_ids"]):
            raise ValueError("MiniMax DSpark: dflash_config.target_layer_ids != dspark_target_layer_ids")
        if dflash_config.get("context_source", "target_layers") != "target_layers":
            raise ValueError("MiniMax DSpark: only context_source=target_layers is supported")
        hparams["dflash_config"] = dflash_config
        hparams["markov_rank"] = hparams.get("dspark_markov_rank", hparams.get("markov_rank"))
        hparams["use_sliding_window"] = True
        hparams["layer_types"] = layer_types

        super().__init__(dir_model, *args, hparams=hparams, **kwargs)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if not name.startswith(cls._PREFIX):
            raise ValueError(f"MiniMax DSpark: unexpected tensor {name!r}")
        name = name.removeprefix(cls._PREFIX).replace(".decoder_layer.", ".")
        if name.startswith("final_norm."):
            name = "norm." + name.removeprefix("final_norm.")
        return super().filter_tensors((name, gen))

    def dequant_model(self):
        # MXFP8: F8_E4M3 weight * 2^(E8M0 - 127) per 32 inputs
        quant_config = self.hparams.get("quantization_config") or {}
        if quant_config.get("quant_method") == "mxfp8":
            if list(quant_config.get("weight_block_size", [1, 32])) != [1, 32]:
                raise ValueError(f"MiniMax DSpark: unexpected MXFP8 block {quant_config.get('weight_block_size')}")

            def dequant_mxfp8(w: Tensor, s: Tensor) -> Tensor:
                if w.dtype != torch.float8_e4m3fn or s.dtype != torch.uint8:
                    raise ValueError(f"MiniMax DSpark: MXFP8 expects F8_E4M3 + U8, got {w.dtype} + {s.dtype}")
                if w.shape[-1] != s.shape[-1] * 32 or w.shape[:-1] != s.shape[:-1]:
                    raise ValueError(f"MiniMax DSpark: MXFP8 shape mismatch {tuple(w.shape)} vs {tuple(s.shape)}")
                scale = torch.exp2(s.to(torch.float32) - 127.0)
                out = w.to(torch.float32).reshape(*w.shape[:-1], s.shape[-1], 32) * scale.unsqueeze(-1)
                return out.reshape(w.shape)

            for name in list(self.model_tensors.keys()):
                if name.endswith(".weight_scale_inv"):
                    weight_name = name.removesuffix("_scale_inv")
                    if weight_name not in self.model_tensors:
                        raise ValueError(f"MiniMax DSpark: scale {name!r} has no weight")
                    w = self.model_tensors[weight_name]
                    s = self.model_tensors.pop(name)
                    self.model_tensors[weight_name] = lambda w=w, s=s: dequant_mxfp8(w(), s())
            self.hparams = {k: v for k, v in self.hparams.items() if k != "quantization_config"}
        super().dequant_model()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_hidden_act("swigluoai")

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # F32 in the reference
        if "confidence_head" in name or new_name.startswith("conf_proj"):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # add 1 in fp32: bf16 rounding loses the low bits of w
        if name.endswith("norm.weight"):
            data_torch = data_torch.to(torch.float32) + 1.0
        yield from super().modify_tensors(data_torch, name, bid)
