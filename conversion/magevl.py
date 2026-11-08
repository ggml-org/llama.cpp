from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf


@ModelBase.register("LlavaOnevision2ForConditionalGeneration")
@ModelBase.example("microsoft/Mage-VL-Cowork")
class MageVLVisionModel(MmprojModel):
    # a rope pair uses 2 different angles: the graph runs 2 ropes on heads permuted to [evens | odds]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None

        self.n_embd = self.find_vparam(["hidden_size"])
        self.n_head = self.find_vparam(["num_attention_heads"])
        self.d_head = self.n_embd // self.n_head

        d_half = self.d_head // 2
        assert d_half % 16 == 0, "the 4:6:6 split of the rope requires d_head to be a multiple of 32"

        # 4:6:6 split for t:h:w
        n_t = 4 * (d_half // 16)
        n_s = 6 * (d_half // 16)
        theta = float(self.find_vparam(["rope_theta"]))
        freq_t = 1.0 / theta ** (torch.arange(n_t, dtype=torch.float64) / n_t)
        freq_s = 1.0 / theta ** (torch.arange(n_s, dtype=torch.float64) / n_s)
        freq = torch.cat([freq_t, freq_s, freq_s])
        self.rope_freq = torch.cat([freq, freq])

        def pair_axis(i: int) -> int:
            dim = (2 * i) % d_half
            return 0 if dim < n_t else 1 if dim < n_t + n_s else 2

        self.rope_pairs = sorted(range(d_half), key=pair_axis)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.MAGEVL)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.find_vparam(["layer_norm_eps"]))
        self.gguf_writer.add_vision_spatial_merge_size(self.find_vparam(["spatial_merge_size"]))
        self.gguf_writer.add_vision_min_pixels(int(self.preprocessor_config["min_pixels"]))
        self.gguf_writer.add_vision_max_pixels(int(self.preprocessor_config["max_pixels"]))

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item

        if not name.startswith("visual."):
            return None

        return super().filter_tensors(item)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # ggml uses theta = pos / rope_freqs
        pairs = torch.tensor(self.rope_pairs)
        freq_a = self.rope_freq[2 * pairs]
        freq_b = self.rope_freq[2 * pairs + 1]
        yield ("visual.rope_freqs.weight", (1.0 / torch.cat([freq_a, freq_b])).to(torch.float32))

    def permute_qk(self, data_torch: Tensor) -> Tensor:
        rows = [2 * i for i in self.rope_pairs] + [2 * i + 1 for i in self.rope_pairs]
        return data_torch.reshape(self.n_head, self.d_head, -1)[:, rows].reshape(data_torch.shape)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if ".self_attn.qkv." in name:
            q, k, v = data_torch.split(self.n_embd, dim=0)
            data_torch = torch.cat([self.permute_qk(q), self.permute_qk(k), v], dim=0)

        yield from super().modify_tensors(data_torch, name, bid)
