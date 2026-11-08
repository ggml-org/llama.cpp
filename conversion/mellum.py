from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("MellumForCausalLM")
@ModelBase.example("JetBrains/Mellum2-12B-A2.5B-Base")
class MellumModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MELLUM
    supports_mtp_export = True

    _n_main_layers: int | None = None

    _experts: list[dict[str, Tensor]] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # number of exported MTP blocks, appended after the trunk layers
        self.n_nextn = 0 if self.no_mtp else (self.find_hparam(["num_nextn_predict_layers", "num_mtp_layers"], optional=True) or 0)
        if self.mtp_only and self.n_nextn == 0:
            raise ValueError("--mtp requested, but the checkpoint has no MTP layers (num_nextn_predict_layers is 0)")
        if self.n_nextn > 0:
            if any(t != "sparse" for t in self.hparams.get("mtp_mlp_layer_types") or []):
                raise ValueError("Mellum MTP layers with dense MLP are not supported")
            self.block_count += self.n_nextn
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def index_tensors(self, remote_hf_model_id: str | None = None):
        type(self)._n_main_layers = self.hparams["num_hidden_layers"]
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem

        if name.startswith("mtp."):
            if cls.no_mtp:
                return None
            assert cls._n_main_layers is not None
            # mtp.layers.{i}.* -> model.layers.{n_main + i}.*, mtp.{fc,norm,...} -> nextn names of the first MTP block
            remapper = {
                "fc":                    "eh_proj",
                "pre_fc_norm_embedding": "enorm",
                "pre_fc_norm_hidden":    "hnorm",
                "norm":                  "shared_head.norm",
            }
            parts = name.split(".", 3)
            if len(parts) == 4 and parts[1] == "layers" and parts[2].isdecimal():
                name = f"model.layers.{cls._n_main_layers + int(parts[2])}.{parts[3]}"
            elif len(parts) == 3 and parts[1] in remapper:
                name = f"model.layers.{cls._n_main_layers}.{remapper[parts[1]]}.{parts[2]}"
        elif cls.mtp_only and name not in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            return None

        return name, gen

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")

        use_sliding_window = self.hparams.get("use_sliding_window")
        sliding_window = self.hparams.get("sliding_window")
        if (use_sliding_window is True or use_sliding_window is None) and sliding_window is not None:
            self.gguf_writer.add_sliding_window(sliding_window)
            logger.info(f"gguf: sliding window = {sliding_window}")
            layer_types = list(self.hparams["layer_types"])
            # MTP blocks default to sliding window attention, as in transformers
            if self.n_nextn > 0:
                layer_types += self.hparams.get("mtp_layer_types") or ["sliding_attention"] * self.n_nextn
            self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in layer_types])
            logger.info(f"gguf: sliding window pattern length = {len(layer_types)}")

        if (norm_topk_prob := self.hparams.get("norm_topk_prob")) is not None:
            self.gguf_writer.add_expert_weights_norm(norm_topk_prob)

        if self.n_nextn > 0:
            self.gguf_writer.add_nextn_predict_layers(self.n_nextn)

    def prepare_metadata(self, vocab_only: bool):
        from_dir = self.fname_out.is_dir()
        super().prepare_metadata(vocab_only=vocab_only)

        if not self.mtp_only or not from_dir:
            return

        output_type: str = self.ftype.name.partition("_")[2]
        fname_default: str = gguf.naming_convention(
            self.metadata.name, self.metadata.basename, self.metadata.finetune,
            self.metadata.version, size_label=None, output_type=output_type, model_type=None)
        self.fname_out = self.fname_out.parent / f"mtp-{fname_default}.gguf"

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.find("experts") != -1:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
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
