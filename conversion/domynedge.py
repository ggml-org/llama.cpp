from .base import TextModel, ModelBase
import gguf


@ModelBase.register("DomynEdgeForCausalLM")
@ModelBase.example("domyn/Domyn-Edge-Teacher-IF")
class DomynEdgeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DOMYN_EDGE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        position_layers = hparams["position_layers"]  # list of "rope"/"nope" strings per layer
        head_sizes      = hparams["head_sizes"]        # list of int head dims per layer

        # Per-layer rope/SWA pattern: True="rope", False="nope"
        rope_pattern = [v == "rope" for v in position_layers]
        self.gguf_writer.add_rope_pattern(rope_pattern)
        self.gguf_writer.add_sliding_window_pattern(rope_pattern)  # SWA <-> RoPE in domyn-edge

        # window_sizes entries are flash-attn style [left, right] (current token excluded; -1 = full attention),
        # while llama.cpp's n_swa counts the current token, hence the +1 (matches HF's sliding_window=1025)
        swa_window = hparams.get("sliding_window")
        if swa_window is None:
            window_sizes = hparams.get("window_sizes")      # list of [size, offset] pairs per layer
            if window_sizes is None:
                raise KeyError("Neither `sliding_window` nor `window_sizes` found among domyn edge hyperparameters")
            left = next((ws[0] for v, ws in zip(position_layers, window_sizes) if v == "rope"), None)
            swa_window = left + 1 if left is not None and left > 0 else None
        if swa_window is None:
            raise ValueError("No sliding window size found among domyn edge hyperparameters")
        self.gguf_writer.add_sliding_window(swa_window)

        # Head dims differ between full-attention ("nope") and SWA ("rope") layers
        full_head_dims = {h for v, h in zip(position_layers, head_sizes) if v == "nope"}
        swa_head_dims  = {h for v, h in zip(position_layers, head_sizes) if v == "rope"}

        if len(full_head_dims) != 1:
            raise ValueError(f"Expected one head dimension for full attention layers, found {len(full_head_dims)}")
        full_head_dim = full_head_dims.pop()

        self.gguf_writer.add_key_length(full_head_dim)
        self.gguf_writer.add_value_length(full_head_dim)

        if len(swa_head_dims) != 1:
            raise ValueError(f"Expected one head dimension for SWA attention layers, found {len(swa_head_dims)}")
        swa_head_dim = swa_head_dims.pop()

        self.gguf_writer.add_key_length_swa(swa_head_dim)
        self.gguf_writer.add_value_length_swa(swa_head_dim)
        self.gguf_writer.add_rope_dimension_count_swa(swa_head_dim)
