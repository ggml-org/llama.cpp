from .base import TextModel, ModelBase
import gguf


@ModelBase.register("DomynEdgeForCausalLM")
@ModelBase.example("domyn/Domyn-Edge-SFT")
class DomynEdgeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DOMYN_EDGE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        position_layers = hparams["position_layers"]  # list of "rope"/"nope" strings per layer
        window_sizes    = hparams["window_sizes"]      # list of [size, offset] pairs per layer
        head_sizes      = hparams["head_sizes"]        # list of int head dims per layer

        # Per-layer rope/SWA pattern: True="rope", False="nope"
        rope_pattern = [v == "rope" for v in position_layers]
        self.gguf_writer.add_rope_pattern(rope_pattern)
        self.gguf_writer.add_sliding_window_pattern(rope_pattern)  # SWA ↔ RoPE in domyn-edge

        '''# window_sizes entries are [size, offset]; -1 means full/global attention
        swa_window = next((ws[0] for v, ws in zip(position_layers, window_sizes) if v == "rope"), None)
        if swa_window is not None and swa_window > 0:
            self.gguf_writer.add_sliding_window(swa_window)        
        '''
        
        # window_sizes entries are flash-attn style [left, right] (current token excluded; -1 = full attention),
        # while llama.cpp's n_swa counts the current token, hence the +1 (matches HF's sliding_window=1025)
        swa_window = hparams.get("sliding_window")
        if swa_window is None:
            left = next((ws[0] for v, ws in zip(position_layers, window_sizes) if v == "rope"), None)
            swa_window = left + 1 if left is not None and left > 0 else None
        if swa_window is not None:
            self.gguf_writer.add_sliding_window(swa_window)
        
        # Head dims differ between full-attention ("nope") and SWA ("rope") layers
        full_head_dim = next((h for v, h in zip(position_layers, head_sizes) if v == "nope"), None)
        swa_head_dim  = next((h for v, h in zip(position_layers, head_sizes) if v == "rope"), None)

        if full_head_dim is not None:
            self.gguf_writer.add_key_length(full_head_dim)
            self.gguf_writer.add_value_length(full_head_dim)

        if swa_head_dim is not None:
            self.gguf_writer.add_key_length_swa(swa_head_dim)
            self.gguf_writer.add_value_length_swa(swa_head_dim)

