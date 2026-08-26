from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, gguf, logger

from .qwen import _LinearAttentionVReorderBase, _Qwen35MRopeMixin


@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
@ModelBase.example("Qwen/Qwen4-Exp")
class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    model_arch = gguf.MODEL_ARCH.QWEN4EXP

    # n-gram embedding tables are stored as many row shards, joined into one tensor here.
    # Keyed by (layer, shard) so two PLE layers cannot mix their shards
    _ngram_shards: dict[tuple[int, int], Tensor] | None = None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_hyper_connection_count(self.hparams["hc_count"])

        # the MTP block consumes the target's hyper-connection streams, not the collapsed hidden
        # state, so both files declare the wider row: the target for the nextn read-back and the
        # draft for common/speculative.cpp. Same as DeepSeek-V4, see conversion/deepseek.py.
        self.gguf_writer.add_embedding_length_out(
            self.hparams["hc_count"]*self.hparams["hidden_size"])
        self.gguf_writer.add_hyper_connection_lowrank(self.hparams["hc_lowrank"])

        # the sigmoid output gate of the linear attention layers is hardcoded in the graph
        gate = self.hparams.get("output_gate_type") or self.hparams.get("hidden_act")
        if gate != "sigmoid":
            raise ValueError(f"unsupported output_gate_type {gate!r} (only 'sigmoid' is supported)")

        # the MTP block has no PLE layer (the reference clears ple_layer_ids for it)
        if not self.mtp_only and self.hparams.get("ple_layer_ids"):
            if len(self.hparams["ple_layer_ids"]) != 1:
                raise ValueError("only a single PLE layer is supported")
            self.gguf_writer.add_ple_embedding_length(self.hparams.get("ple_embed_dim") or self.hparams["hidden_size"])
            self.gguf_writer.add_ple_conv_kernel(self.hparams["ple_conv_kernel_size"])

            # the n-gram hash constants are needed on the host, keep them as metadata. They are read
            # here and not in modify_tensors because that already casts int64 buffers to float32.
            self.gguf_writer.add_ple_ngram_multipliers(self._read_u64(".ple_embedding.layer_multipliers"))
            self.gguf_writer.add_ple_ngram_vocab_sizes(self._read_u64(".ple_embedding.ngram_heads_vocab_sizes"))
            self.gguf_writer.add_ple_ngram_offsets(self._read_u64(".ple_embedding.ngram_heads_offsets"))

        # export the layer types instead of letting llama.cpp infer them from
        # full_attention_interval, so an irregular layout cannot misalign silently
        if layer_types := self.hparams.get("layer_types"):
            n_layer = self.hparams["num_hidden_layers"]
            if len(layer_types) != n_layer:
                raise ValueError(f"layer_types has {len(layer_types)} entries, "
                                 f"expected num_hidden_layers ({n_layer})")

            recurrent = []
            for t in layer_types:
                if t == "linear_attention":
                    recurrent.append(True)
                elif t.endswith("sparse_attention"):
                    recurrent.append(False)
                else:
                    raise ValueError(f"unsupported qwen4exp layer type {t!r}")

            # llama.cpp reads this with get_key_or_arr(.., n_layer_all), which wants exactly
            # block_count entries. The MTP block is sparse attention, so it is not recurrent
            recurrent += [False] * (self.block_count - n_layer)
            self.gguf_writer.add_recurrent_layers(recurrent)

        if self.hparams.get("indexer_n_heads") is not None:
            self.gguf_writer.add_indexer_head_count(self.hparams["indexer_n_heads"])
            self.gguf_writer.add_indexer_key_length(self.hparams["indexer_head_dim"])
            self.gguf_writer.add_indexer_top_k     (self.hparams["indexer_budget"])
            self.gguf_writer.add_indexer_block_size(self.hparams["indexer_compress_ratio"])

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        yield from super().generate_extra_tensors()

        # the reference adds the two MTP input projections, and A*e + B*h == [A|B]*concat(e, h),
        # so they join into the single eh_proj the tensor map already knows.
        # ref: conversion/deepseek.py, which joins the DeepSeek-V4 pair the same way
        e_name = "mtp.fc_embedding.weight"
        h_name = "mtp.fc_hidden.weight"

        have_e = e_name in self.model_tensors
        have_h = h_name in self.model_tensors
        if not have_e and not have_h:
            return
        if not have_e or not have_h:
            raise KeyError(f"unpaired MTP input projection: need both {e_name} and {h_name}")

        e = LazyTorchTensor.to_eager(self.model_tensors[e_name]())
        h = LazyTorchTensor.to_eager(self.model_tensors[h_name]())
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_EH_PROJ,
                                       self.hparams["num_hidden_layers"]),
               torch.cat([e, h], dim=1).contiguous())

        del self.model_tensors[e_name]
        del self.model_tensors[h_name]

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # the graph slices the PLE conv weight and multiplies it, which needs F32
        if ".ple_conv1d.weight" in new_name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name = item[0]

        # the vision tower goes to the mmproj file
        if name.startswith("model.visual."):
            return None

        # the MTP block brings its own hyper-connection mixer, which takes the model-level slot of
        # an MTP-only file. Rename it here, before _QwenMtpMixin drops it as a non-MTP tensor.
        if name.startswith("mtp.hyper_connection_mixer."):
            return None if cls.no_mtp else (name.replace("mtp.", "model.", 1), item[1])

        return super().filter_tensors(item)

    def _read_u64(self, suffix: str) -> list[int]:
        name = next((n for n in self.model_tensors if n.endswith(suffix)), None)
        if name is None:
            raise ValueError(f"missing tensor *{suffix}, needed for the PLE n-gram hash")
        return [int(v) for v in LazyTorchTensor.to_eager(self.model_tensors[name]()).tolist()]

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # hash constants, already exported as metadata by set_gguf_parameters
        if ".ple_embedding." in name and not name.endswith(".weight"):
            return

        if ".ple_embedding.ngram_embedding.shard_" in name:
            shard = int(name.rpartition(".ple_embedding.ngram_embedding.shard_")[2].partition(".")[0])
            n_shard = self.hparams["split_ngram_parts"]

            assert bid is not None

            if self._ngram_shards is None:
                self._ngram_shards = {}
            self._ngram_shards[(bid, shard)] = data_torch

            shards = {s: t for (b, s), t in self._ngram_shards.items() if b == bid}
            if len(shards) < n_shard:
                return

            # the shards are joined in shard order, so the numbering has to be 0-based and dense
            if sorted(shards) != list(range(n_shard)):
                raise ValueError(f"layer {bid}: expected n-gram shards 0..{n_shard - 1}, "
                                 f"got {sorted(shards)}")

            logger.info(f"joining {n_shard} n-gram embedding shards")
            rows = sum(shards[i].shape[0] for i in range(n_shard))

            # llama.cpp indexes the joined table as offs[h] + 0..vocab[h]-1, so a table that is
            # too short has to fail here and not hours later as a shape mismatch
            offs = self._read_u64(".ple_embedding.ngram_heads_offsets")
            vocab = self._read_u64(".ple_embedding.ngram_heads_vocab_sizes")
            need = max(o + v for o, v in zip(offs, vocab))
            if rows < need:
                raise ValueError(f"joined n-gram table has {rows} rows, "
                                 f"the hash offsets need {need}")

            joined = torch.empty((rows, data_torch.shape[1]), dtype=data_torch.dtype)
            row = 0
            for i in range(n_shard):
                part = LazyTorchTensor.to_eager(self._ngram_shards.pop((bid, i)))
                joined[row:row + part.shape[0]] = part
                row += part.shape[0]
                del part

            yield (self.format_tensor_name(gguf.MODEL_TENSOR.PLE_NGRAM_EMBD, bid), joined)
            return

        # the QSA indexer packs its query and key projections together
        if name.endswith(".indexer.index_qk_proj.weight"):
            n_q = self.hparams["indexer_n_heads"]*self.hparams["indexer_head_dim"]
            assert bid is not None
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_PROJ, bid), data_torch[:n_q])
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_K_PROJ, bid), data_torch[n_q:])
            return

        # PLE norms are also 1-centered, the base class only catches names ending in "norm.weight"
        if name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight")):
            data_torch = data_torch + 1

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._ngram_shards:
            raise ValueError(f"unprocessed n-gram embedding shards: {sorted(self._ngram_shards)}")
