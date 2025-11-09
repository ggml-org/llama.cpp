from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf, torch
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("Rwkv6ForCausalLM")
class Rwkv6Model(TextModel):
    model_arch = gguf.MODEL_ARCH.RWKV6

    def set_vocab(self):
        self._set_vocab_rwkv_world()

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        head_size = self.hparams["head_size"]
        hidden_size = self.hparams["hidden_size"]
        layer_norm_eps = self.hparams["layer_norm_epsilon"]
        rescale_every_n_layers = self.hparams["rescale_every"]
        intermediate_size = self.hparams["intermediate_size"] if self.hparams["intermediate_size"] is not None else int((hidden_size * 3.5) // 32 * 32)
        time_mix_extra_dim = 64 if hidden_size == 4096 else 32
        time_decay_extra_dim = 128 if hidden_size == 4096 else 64
        # RWKV isn't context limited
        self.gguf_writer.add_context_length(1048576)
        self.gguf_writer.add_embedding_length(hidden_size)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_layer_norm_eps(layer_norm_eps)
        self.gguf_writer.add_rescale_every_n_layers(rescale_every_n_layers)
        self.gguf_writer.add_wkv_head_size(head_size)
        self.gguf_writer.add_time_mix_extra_dim(time_mix_extra_dim)
        self.gguf_writer.add_time_decay_extra_dim(time_decay_extra_dim)
        self.gguf_writer.add_feed_forward_length(intermediate_size)
        self.gguf_writer.add_file_type(self.ftype)
        # required by llama.cpp, unused
        self.gguf_writer.add_head_count(0)
    lerp_weights: dict[int, dict[str, Tensor]] = {}

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        new_name = self.map_tensor_name(name)
        if not (new_name.endswith(".weight") or new_name.endswith(".bias")):
            new_name += ".weight"
        if new_name.endswith("time_mix_w1.weight") or new_name.endswith("time_mix_decay_w1.weight") or new_name.endswith("time_mix_decay_w2.weight"):
            data_torch = data_torch.transpose(0, 1)
        if new_name.endswith("time_mix_w2.weight"):
            data_torch = data_torch.permute(0, 2, 1)
        if new_name.endswith("time_mix_decay.weight") or "lerp" in new_name:
            data_torch = data_torch.squeeze()
        try:
            rescale_every_n_layers = self.hparams["rescale_every"]
            if rescale_every_n_layers > 0:
                if new_name.endswith("time_mix_output.weight") or new_name.endswith("channel_mix_value.weight"):
                    data_torch = data_torch.div_(2 ** int(bid // rescale_every_n_layers))
        except KeyError:
            pass
        # concat time_mix_lerp weights to reduce some cpu overhead
        # also reduces the number of tensors in the model
        if bid is not None and "time_mix_lerp" in new_name and "time_mix_lerp_x" not in new_name:
            try:
                self.lerp_weights[bid][new_name] = data_torch
            except KeyError:
                self.lerp_weights[bid] = {new_name: data_torch}
            if all(f"blk.{bid}.time_mix_lerp_{i}.weight" in self.lerp_weights[bid].keys() for i in ["w", "k", "v", "r", "g"]):
                new_name = f"blk.{bid}.time_mix_lerp_fused.weight"
                data = torch.stack([self.lerp_weights[bid][f"blk.{bid}.time_mix_lerp_{i}.weight"].unsqueeze(0) for i in ["w", "k", "v", "r", "g"]], dim=0).unsqueeze(1)
                yield (new_name, data)
            return
        yield (new_name, data_torch)


@ModelBase.register("RWKV6Qwen2ForCausalLM")
class RWKV6Qwen2Model(Rwkv6Model):
    model_arch = gguf.MODEL_ARCH.RWKV6QWEN2

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        num_attention_heads = self.hparams["num_attention_heads"]
        num_key_value_heads = self.hparams["num_key_value_heads"]
        hidden_size = self.hparams["hidden_size"]
        head_size = hidden_size // num_attention_heads
        rms_norm_eps = self.hparams["rms_norm_eps"]
        intermediate_size = self.hparams["intermediate_size"]
        time_mix_extra_dim = self.hparams.get("lora_rank_tokenshift", 64 if hidden_size >= 4096 else 32)
        time_decay_extra_dim = self.hparams.get("lora_rank_decay", 128 if hidden_size >= 4096 else 64)
        # RWKV isn't context limited
        self.gguf_writer.add_context_length(1048576)
        self.gguf_writer.add_embedding_length(hidden_size)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_wkv_head_size(head_size)
        self.gguf_writer.add_time_mix_extra_dim(time_mix_extra_dim)
        self.gguf_writer.add_time_decay_extra_dim(time_decay_extra_dim)
        self.gguf_writer.add_feed_forward_length(intermediate_size)
        self.gguf_writer.add_file_type(self.ftype)
        # special parameters for time_mixing in RWKV6QWEN2
        self.gguf_writer.add_layer_norm_rms_eps(rms_norm_eps)
        self.gguf_writer.add_token_shift_count(1)
        # RWKV6QWEN2 use grouped key/value like GQA
        self.gguf_writer.add_head_count_kv(num_key_value_heads)
        # required by llama.cpp, unused
        self.gguf_writer.add_head_count(0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        for new_name, data in super().modify_tensors(data_torch, name, bid):
            if "time_mix_w1" in new_name or "time_mix_w2" in new_name:
                data = data.view(5, -1, data.shape[-1])
                # rwkv6qwen2 has a different order of rkvwg instead of the original wkvrg
                # permute them here to avoid code changes
                data = torch.stack([data[3], data[1], data[2], data[0], data[4]], dim=0).view(-1, data.shape[-1])
                if "w2" in new_name:
                    data = data.view(5, -1, data.shape[-1])
                yield (new_name, data)
                continue
            yield (new_name, data)


@ModelBase.register("Rwkv7ForCausalLM", "RWKV7ForCausalLM")
class Rwkv7Model(TextModel):
    model_arch = gguf.MODEL_ARCH.RWKV7

    def set_vocab(self):
        self._set_vocab_rwkv_world()

    def calc_lora_rank(self, hidden_size, exponent, multiplier):
        return max(1, round(hidden_size ** exponent * multiplier / 32)) * 32

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        try:
            head_size = self.hparams["head_size"]
            layer_norm_eps = self.hparams["layer_norm_epsilon"]
        except KeyError:
            head_size = self.hparams["head_dim"]
            layer_norm_eps = self.hparams["norm_eps"]
        hidden_size = self.hparams["hidden_size"]
        intermediate_size = self.hparams["intermediate_size"] if self.hparams["intermediate_size"] is not None else (hidden_size * 4)
        # ICLR: In-Context-Learning-Rate
        try:
            lora_rank_decay = self.hparams["lora_rank_decay"] if self.hparams["lora_rank_decay"] is not None else self.calc_lora_rank(hidden_size, 0.5, 1.8)
            lora_rank_iclr = self.hparams["lora_rank_iclr"] if self.hparams["lora_rank_iclr"] is not None else self.calc_lora_rank(hidden_size, 0.5, 1.8)
            lora_rank_value_residual_mix = self.hparams["lora_rank_value_residual_mix"] if self.hparams["lora_rank_value_residual_mix"] is not None else self.calc_lora_rank(hidden_size, 0.5, 1.3)
            lora_rank_gate = self.hparams["lora_rank_gate"] if self.hparams["lora_rank_gate"] is not None else self.calc_lora_rank(hidden_size, 0.8, 0.6)
        except KeyError:
            lora_rank_decay = self.hparams["decay_low_rank_dim"] if self.hparams["decay_low_rank_dim"] is not None else self.calc_lora_rank(hidden_size, 0.5, 1.8)
            lora_rank_iclr = self.hparams["a_low_rank_dim"] if self.hparams["a_low_rank_dim"] is not None else self.calc_lora_rank(hidden_size, 0.5, 1.8)
            lora_rank_value_residual_mix = self.hparams["v_low_rank_dim"] if self.hparams["v_low_rank_dim"] is not None else self.calc_lora_rank(hidden_size, 0.5, 1.3)
            lora_rank_gate = self.hparams["gate_low_rank_dim"] if self.hparams["gate_low_rank_dim"] is not None else self.calc_lora_rank(hidden_size, 0.8, 0.6)
        # RWKV isn't context limited
        self.gguf_writer.add_context_length(1048576)
        self.gguf_writer.add_embedding_length(hidden_size)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_layer_norm_eps(layer_norm_eps)
        self.gguf_writer.add_wkv_head_size(head_size)
        self.gguf_writer.add_decay_lora_rank(lora_rank_decay)
        self.gguf_writer.add_iclr_lora_rank(lora_rank_iclr)
        self.gguf_writer.add_value_residual_mix_lora_rank(lora_rank_value_residual_mix)
        self.gguf_writer.add_gate_lora_rank(lora_rank_gate)
        self.gguf_writer.add_feed_forward_length(intermediate_size)
        self.gguf_writer.add_file_type(self.ftype)
        # required by llama.cpp, unused
        self.gguf_writer.add_head_count(0)
    lerp_weights: dict[int, dict[str, Tensor]] = {}
    lora_needs_transpose: bool = True

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # unify tensor names here to make life easier
        name = name.replace("blocks", "layers").replace("ffn", "feed_forward")
        name = name.replace("self_attn", "attention").replace("attn", "attention")
        name = name.replace("time_mixer.", "")
        # lora layer names in fla-hub's impl
        if "_lora.lora" in name:
            self.lora_needs_transpose = False
        name = name.replace("_lora.lora.0.weight", "1.weight")
        name = name.replace("_lora.lora.2.weight", "2.weight")
        name = name.replace("_lora.lora.2.bias", "0.weight")
        name = name.replace("feed_forward_norm", "ln2")
        name = name.replace("g_norm", "ln_x")
        if "attention.v" in name and "value" not in self.map_tensor_name(name) and bid == 0:
            # some models have dummy v0/v1/v2 on first layer while others don't
            # ignore them all since they are not used
            return
        wkv_has_gate = self.hparams.get("wkv_has_gate", True)
        lerp_list = ["r", "w", "k", "v", "a", "g"] if wkv_has_gate else ["r", "w", "k", "v", "a"]
        if bid is not None and "attention.x_" in name:
            if "attention.x_x" in name:
                # already concatenated
                new_name = f"blk.{bid}.time_mix_lerp_fused.weight"
                data = data_torch.reshape(len(lerp_list), 1, 1, -1)
                yield (new_name, data)
            else:
                try:
                    self.lerp_weights[bid][name] = data_torch
                except KeyError:
                    self.lerp_weights[bid] = {name: data_torch}
                if all(f"model.layers.{bid}.attention.x_{i}" in self.lerp_weights[bid].keys() for i in lerp_list):
                    new_name = f"blk.{bid}.time_mix_lerp_fused.weight"
                    data = torch.stack([self.lerp_weights[bid][f"model.layers.{bid}.attention.x_{i}"] for i in lerp_list], dim=0)
                    yield (new_name, data)
            return
        else:
            data_torch = data_torch.squeeze()
            new_name = self.map_tensor_name(name)
            if not (new_name.endswith(".weight") or new_name.endswith(".bias")):
                new_name += ".weight"
            if self.lora_needs_transpose and any(
                new_name.endswith(t) for t in [
                    "time_mix_w1.weight", "time_mix_w2.weight",
                    "time_mix_a1.weight", "time_mix_a2.weight",
                    "time_mix_v1.weight", "time_mix_v2.weight",
                    "time_mix_g1.weight", "time_mix_g2.weight",
                ]
            ):
                data_torch = data_torch.transpose(0, 1)
            if 'r_k' in new_name:
                data_torch = data_torch.flatten()
            if bid == 0 and "time_mix_a" in new_name:
                # dummy v0/v1/v2 on first layer
                # easist way to make llama happy
                yield (new_name.replace("time_mix_a", "time_mix_v"), data_torch)
            yield (new_name, data_torch)


@ModelBase.register("RwkvHybridForCausalLM")
class ARwkv7Model(Rwkv7Model):
    model_arch = gguf.MODEL_ARCH.ARWKV7

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        hidden_size = self.hparams["hidden_size"]
        head_size = self.hparams["head_size"]
        rms_norm_eps = self.hparams["rms_norm_eps"]
        intermediate_size = self.hparams["intermediate_size"]
        wkv_has_gate = self.hparams["wkv_has_gate"]
        assert self.hparams["wkv_version"] == 7
        # ICLR: In-Context-Learning-Rate
        lora_rank_decay = 64
        lora_rank_iclr = 64
        lora_rank_value_residual_mix = 32
        lora_rank_gate = 128 if wkv_has_gate else 0
        # RWKV isn't context limited
        self.gguf_writer.add_context_length(1048576)
        self.gguf_writer.add_embedding_length(hidden_size)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_layer_norm_rms_eps(rms_norm_eps)
        self.gguf_writer.add_wkv_head_size(head_size)
        self.gguf_writer.add_decay_lora_rank(lora_rank_decay)
        self.gguf_writer.add_iclr_lora_rank(lora_rank_iclr)
        self.gguf_writer.add_value_residual_mix_lora_rank(lora_rank_value_residual_mix)
        self.gguf_writer.add_gate_lora_rank(lora_rank_gate)
        self.gguf_writer.add_feed_forward_length(intermediate_size)
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_token_shift_count(1)
        # required by llama.cpp, unused
        self.gguf_writer.add_head_count(0)
