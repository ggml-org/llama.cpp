from __future__ import annotations

import json
import re

from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf


@ModelBase.register("MiMoV2FlashForCausalLM", "MiMoV2ForCausalLM")
class MimoV2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.MIMO2

    # MiMo V2-Flash, V2.5 and V2.5-Pro all ship 3 trained MTP layers under model.mtp.layers.{0,1,2}.
    # The HF config does not expose the count, so it's hardcoded to match the count found in the safetensors.
    _n_nextn = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.block_count = self.hparams["num_hidden_layers"] + self._n_nextn
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    @staticmethod
    def _tp_aware_qkv_dequant(weight: Tensor, scale_inv: Tensor,
                              n_q: int, n_kv: int, hd: int, vhd: int,
                              bs: int = 128) -> Tensor:
        # MiMo-V2.5 (TP=4) and V2.5-Pro (TP=8) ship qkv_proj sharded across TP
        # ranks; per rank, rows are stacked as [Q_per | K_per | V_per].
        # weight_scale_inv has ceil(rows_per_rank/bs) block-rows per rank (last
        # may extend past rows_per_rank with phantom rows not in the weight).
        # Naive repeat_interleave aligns rank 0 only and mis-applies scales to
        # later ranks once rows_per_rank isn't a multiple of bs.
        # Re-group the per-rank [Q_per|K_per|V_per] rows into a single fused
        # [Q | K | V] tensor matching the un-sharded original layout.
        q_size = n_q * hd
        k_size = n_kv * hd
        v_size = n_kv * vhd
        total_rows = q_size + k_size + v_size
        if weight.shape[0] != total_rows:
            raise ValueError(f"qkv_proj weight rows {weight.shape[0]} != q+k+v {total_rows}")

        # detect TP from scale_inv block count, descending order so larger matches first
        tp = None
        for cand in (8, 4):
            if total_rows % cand != 0:
                continue
            rpr = total_rows // cand
            bpr = (rpr + bs - 1) // bs
            if scale_inv.shape[0] == cand * bpr:
                tp = cand
                break
        if tp is None:
            raise ValueError(
                f"qkv_proj: cannot detect TP - scale_inv rows {scale_inv.shape[0]}, "
                f"q+k+v {total_rows}")

        q_per = q_size // tp
        k_per = k_size // tp
        v_per = v_size // tp
        rows_per_rank = q_per + k_per + v_per
        blocks_per_rank = (rows_per_rank + bs - 1) // bs

        scale_inv = scale_inv.float()
        # per-row scale-row index: rank * blocks_per_rank + (rr_in_rank // bs)
        row_idx = torch.arange(total_rows)
        rr = row_idx % rows_per_rank
        rank = row_idx // rows_per_rank
        scale_row_idx = rank * blocks_per_rank + (rr // bs)
        # gather: (total_rows, n_col_blocks)
        scale_per_row_block = scale_inv[scale_row_idx]
        # expand col-blocks -> cols: each block-col covers `bs` weight cols
        scale_full = scale_per_row_block.repeat_interleave(bs, dim=1)
        # crop to weight col count (in case last col-block isn't full)
        scale_full = scale_full[:, : weight.shape[1]]
        dequant = weight.float() * scale_full

        if tp == 1:
            return dequant

        # Re-group per-rank [Q_per|K_per|V_per] rows into unified [Q | K | V]
        qs, ks, vs = [], [], []
        for r in range(tp):
            base = r * rows_per_rank
            qs.append(dequant[base : base + q_per])
            ks.append(dequant[base + q_per : base + q_per + k_per])
            vs.append(dequant[base + q_per + k_per : base + rows_per_rank])
        return torch.cat(qs + ks + vs, dim=0)

    def dequant_model(self):
        # Capture raw FP8 (weight, scale_inv) lambdas for qkv_proj BEFORE super
        # rewrites them with the existing dequant. Replace super's lambda after
        # it runs so scale_inv removal still happens via the standard path.
        qkv_overrides: dict[str, tuple[Callable, Callable, int]] = {}
        qc = self.hparams.get("quantization_config")
        if isinstance(qc, dict) and qc.get("quant_method") == "fp8":
            pat = re.compile(r"^model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight_scale_inv$")
            for name in list(self.model_tensors.keys()):
                m = pat.match(name)
                if not m:
                    continue
                weight_name = name.removesuffix("_scale_inv")
                if weight_name not in self.model_tensors:
                    continue
                qkv_overrides[weight_name] = (
                    self.model_tensors[weight_name],
                    self.model_tensors[name],
                    int(m.group(1)),
                )

        super().dequant_model()

        if not qkv_overrides:
            return

        n_q = self.hparams["num_attention_heads"]
        hd = self.hparams["head_dim"]
        vhd = self.hparams["v_head_dim"]
        hybrid = self.hparams["hybrid_layer_pattern"]
        n_layer_text = self.hparams["num_hidden_layers"]
        for weight_name, (w_fn, s_fn, bid) in qkv_overrides.items():
            # MTP layers (bid >= n_layer_text) use SWA-style attention dims
            is_swa = True if bid >= n_layer_text else hybrid[bid] == 1
            n_kv = self.hparams["swa_num_key_value_heads" if is_swa else "num_key_value_heads"]
            self.model_tensors[weight_name] = (
                lambda w_fn=w_fn, s_fn=s_fn, n_q=n_q, n_kv=n_kv, hd=hd, vhd=vhd:
                    MimoV2Model._tp_aware_qkv_dequant(w_fn(), s_fn(), n_q, n_kv, hd, vhd)
            )

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        assert self.hparams["swa_head_dim"] == self.hparams["head_dim"]
        assert self.hparams["swa_num_attention_heads"] == self.hparams["num_attention_heads"]
        assert self.hparams["swa_v_head_dim"] == self.hparams["v_head_dim"]
        assert self.hparams["topk_method"] == "noaux_tc"

        n_head_kv = self.hparams["num_key_value_heads"]
        n_head_kv_swa = self.hparams["swa_num_key_value_heads"]
        # Extend the per-layer pattern with SWA entries for the MTP blocks so the
        # runtime arrays (sized to extended block_count) are fully populated.
        hybrid = list(self.hparams["hybrid_layer_pattern"]) + [1] * self._n_nextn
        n_head_kv_arr = [n_head_kv_swa if use_swa == 1 else n_head_kv for use_swa in hybrid]
        self.gguf_writer.add_head_count_kv(n_head_kv_arr)

        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(hybrid)
        self.gguf_writer.add_value_length(self.hparams["v_head_dim"])
        self.gguf_writer.add_expert_count(self.hparams["n_routed_experts"])
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])

        rope_dim = int(self.hparams["head_dim"] * self.rope_parameters["partial_rotary_factor"])
        self.gguf_writer.add_rope_dimension_count(rope_dim)

        self.gguf_writer.add_layer_norm_rms_eps(self.hparams.get("layernorm_epsilon", 1e-5))

        v_scale = self.hparams.get("attention_value_scale")
        if v_scale is not None:
            self.gguf_writer.add_attn_value_scale(float(v_scale))

        self.gguf_writer.add_nextn_predict_layers(self._n_nextn)

    _experts: list[dict[str, Tensor]] | None = None

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if "attention_sink" in name and not name.endswith(".weight"):
            name += ".weight"

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch, name, bid):
        # Remap MTP/NextN tensors to additional layer slots so the standard tensor map handles them.
        # HF: model.mtp.layers.{i}.foo  ->  model.layers.{n_layer_text + i}.foo
        m = re.match(r"^model\.mtp\.layers\.(\d+)\.(.*)$", name)
        if m is not None:
            mtp_idx = int(m.group(1))
            assert mtp_idx < self._n_nextn, f"MTP layer index {mtp_idx} >= _n_nextn ({self._n_nextn})"
            rest = m.group(2)
            n_layer_text = self.hparams["num_hidden_layers"]
            new_bid = n_layer_text + mtp_idx
            name = f"model.layers.{new_bid}.{rest}"
            bid = new_bid

        # process the experts separately
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["gate_proj", "up_proj", "down_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename_to_retrieve = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename_to_retrieve])
                        del self._experts[bid][ename_to_retrieve]

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


@ModelBase.register("MiMoV2ForCausalLM")
class MiMoV2VisionAudioModel(MmprojModel):
    has_audio_encoder = True

    _audio_tok_hparams: dict[str, Any] | None = None
    _rvq_codebook_sizes: list[int] | None = None
    _code_embd: dict[int, Tensor] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        hp = self.hparams_vision

        hp["image_size"] = hp.get("image_size", 560)
        hp["num_attention_heads"] = hp.get("num_heads", 32)
        hp["num_hidden_layers"] = hp.get("depth", 28)

        self.n_q_heads = int(hp["num_heads"])
        self.num_kv_heads = int(hp.get("num_key_value_heads", 8))
        self.head_dim = int(hp.get("qk_channels", 64))
        self.spatial_merge_size = int(hp["spatial_merge_size"])
        # MiMoV2 vision RMSNorm: HF uses getattr(config, "rms_norm_eps", 1e-6) and the
        # field is absent from MiMo-V2.5's vision_config
        self.rms_norm_eps = float(hp.get("rms_norm_eps", 1e-6))

        # fullatt_block_indexes are also reflected in vit_window_attn_types as -1
        self.fullatt_block_indexes = list(hp.get("fullatt_block_indexes") or [])
        self.vit_window_attn_types = list(hp.get("vit_window_attn_types") or [])
        self.visual_token_window_size = int(hp.get("visual_token_window_size", -1))
        self.use_sink = bool(hp.get("use_sink", False))

    def get_audio_config(self) -> dict[str, Any] | None:
        if self._audio_tok_hparams is None:
            path = self.dir_model / "audio_tokenizer" / "config.json"
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # aliases so MmprojModel.find_aparam() / n_block_keys can resolve them
            cfg["hidden_size"] = cfg["d_model"]
            cfg["intermediate_size"] = cfg["encoder_ffn_dim"]
            cfg["num_attention_heads"] = cfg["encoder_attention_heads"]
            self._audio_tok_hparams = cfg
        return self._audio_tok_hparams

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_clip_vision_projector_type(gguf.VisionProjectorType.MIMOVL)
        self.gguf_writer.add_vision_use_silu(True)
        self.gguf_writer.add_vision_head_count_kv(self.num_kv_heads)
        self.gguf_writer.add_vision_spatial_merge_size(self.spatial_merge_size)
        self.gguf_writer.add_uint32(gguf.Keys.ClipVision.WINDOW_SIZE, self.visual_token_window_size)
        self.gguf_writer.add_vision_wa_pattern_mode(self.vit_window_attn_types)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.rms_norm_eps)
        self.gguf_writer.add_vision_min_pixels(int(self.preprocessor_config["min_pixels"]))
        self.gguf_writer.add_vision_max_pixels(int(self.preprocessor_config["max_pixels"]))

        assert self.hparams_audio is not None
        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.MIMO_AUDIO)
        self.gguf_writer.add_audio_num_mel_bins(self.hparams_audio["n_mels"])
        self.gguf_writer.add_audio_attention_layernorm_eps(self.hparams_audio.get("layer_norm_eps", 1e-5))

        assert self._rvq_codebook_sizes is not None
        self.gguf_writer.add_audio_rvq_num_quantizers(len(self._rvq_codebook_sizes))
        self.gguf_writer.add_audio_rvq_codebook_size(self._rvq_codebook_sizes)

        n_layer = self.hparams_audio["encoder_layers"]
        swa_per_block = self.hparams_audio.get("swa_per_block", 1)
        if self.hparams_audio.get("hybrid_attention") and swa_per_block > 1:
            wa_pattern = [0 if i % swa_per_block < swa_per_block - 1 else -1 for i in range(n_layer)]
        else:
            wa_pattern = [-1] * n_layer
        self.gguf_writer.add_audio_wa_pattern_mode(wa_pattern)
        self.gguf_writer.add_audio_window_size(int(self.hparams_audio["encoder_attn_window_size"][0]))

        audio_cfg = self.global_config["audio_config"]
        self.gguf_writer.add_audio_local_block_count(int(audio_cfg["input_local_layers"]))
        self.gguf_writer.add_audio_local_group_size(int(audio_cfg["group_size"]))

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # for audio encoder: keep codebook in F32
        if new_name in (
            gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.A_ENC_RVQ_CODEBOOK] + ".weight",
            gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.A_MM_CODE_EMBD] + ".weight",
        ):
            return gguf.GGMLQuantizationType.F32
        if ("encoder.conv" in name or "encoder.down_sample_layer" in name) and name.endswith(".weight"):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        if name.startswith("visual.") or name.startswith("speech_embeddings.") or name.startswith("audio_encoder."):
            return super().filter_tensors(item)
        return None

    def modify_tensors(self, data_torch, name, bid):
        # Conv3D patch embed: split along the temporal axis (kt=2) into two Conv2D
        # weights that the existing qwen2vl-style two-Conv2D path consumes.
        if name == "visual.patch_embed.proj.weight":
            _, _, kt, _, _ = data_torch.shape
            if kt != 2:
                raise ValueError(f"unexpected temporal_patch_size: {kt}")
            embd_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH]
            yield (embd_name + ".weight",   data_torch[:, :, 0, ...])
            yield (embd_name + ".weight.1", data_torch[:, :, 1, ...])
            return

        if m := re.match(r"^speech_embeddings\.(\d+)\.weight$", name):
            if self._code_embd is None:
                self._code_embd = {}
            self._code_embd[int(m.group(1))] = data_torch

            n_channels = int(self.global_config["audio_config"]["audio_channels"])
            if len(self._code_embd) < n_channels:
                return
            merged = torch.stack([self._code_embd.pop(i) for i in range(n_channels)], dim=0)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_MM_CODE_EMBD), merged)
            return

        if "conv1.bias" in name or "conv2.bias" in name:
            # transpose conv1/conv2 bias so it broadcasts against [n_frames, C_out, 1]
            data_torch = data_torch.unsqueeze(-1)

        if name == "audio_encoder.projection.mlp.0.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_MMPROJ, 1), data_torch)
            return
        if name == "audio_encoder.projection.mlp.2.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_MMPROJ, 2), data_torch)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # note: audio encoder is in its own subdir "audio_tokenizer"
        from safetensors.torch import load_file

        tok_dir = self.dir_model / "audio_tokenizer"
        state_dict = load_file(tok_dir / "model.safetensors")

        codebook_re = re.compile(r"^encoder\.quantizer\.vq\.layers\.(\d+)\._codebook\.embed$")
        codebooks: dict[int, Tensor] = {}

        # EMA/training-only RVQ buffers - not needed for inference (nearest-codebook
        # lookup only reads "_codebook.embed")
        skip_suffixes = (
            "_codebook.cluster_size",
            "_codebook.embed_avg",
            "_codebook.inited",
        )
        for name, tensor in state_dict.items():
            if name.endswith(skip_suffixes):
                continue
            if m := codebook_re.match(name):
                codebooks[int(m.group(1))] = tensor
                continue
            yield name, tensor

        # gather codebooks and merge into 3D tensor, similar to MoE MLP tensors
        n_q = len(codebooks)
        ordered = [codebooks[i] for i in range(n_q)]
        self._rvq_codebook_sizes = [int(cb.shape[0]) for cb in ordered]
        max_bins = max(self._rvq_codebook_sizes)
        dim = ordered[0].shape[1]
        merged = ordered[0].new_zeros(n_q, max_bins, dim)
        for i, cb in enumerate(ordered):
            merged[i, : cb.shape[0], :] = cb

        yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_ENC_RVQ_CODEBOOK), merged)
