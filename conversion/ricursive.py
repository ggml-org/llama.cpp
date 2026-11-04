from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("RicursiveForCausalLM")
@ModelBase.example("ricursiveAI/Ripple-65B-Preview")
class RicursiveModel(TextModel):
    model_arch = gguf.MODEL_ARCH.RICURSIVE

    def set_vocab(self):
        self._set_vocab_gpt2()
        tmpl = self.dir_model / "chat_template.jinja"
        if tmpl.is_file():
            self.gguf_writer.add_chat_template(tmpl.read_text(encoding="utf-8"))
        eos = self.hparams.get("eos_token_id")
        if isinstance(eos, list):
            extra = [e for e in eos if e != self.hparams.get("bos_token_id")]
            if extra:
                self.gguf_writer.add_eot_token_id(extra[0])

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_rope_dimension_count(hparams["head_dim"])

        if (sliding_window := hparams.get("sliding_window")) and "layer_types" in hparams:
            self.gguf_writer.add_sliding_window(sliding_window)
            self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in hparams["layer_types"]])

        if (embed_scale := hparams.get("embed_scale")) is not None:
            self.gguf_writer.add_embedding_scale(float(embed_scale))
        if (clamp := hparams.get("ffn_clamp_threshold")):
            self.gguf_writer.add_swiglu_clamp(float(clamp))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".value_res_lambda"):
            name += ".weight"
        yield from super().modify_tensors(data_torch, name, bid)
