#!/usr/bin/env python3
# finetune checkpoint --> gguf conversion

import argparse
import gguf
import struct
import numpy as np
from pathlib import Path

# gguf constants
LLM_KV_OPTIMIZER_TYPE = "optimizer.type"
LLM_KV_OPTIMIZER_TYPE_ADAM  = "adam"
LLM_KV_OPTIMIZER_TYPE_LBFGS = "lbfgs"
LLM_KV_OPTIMIZER_FILE_VERSION               = "optimizer.file_version"
LLM_KV_OPTIMIZER_CONVERGENCE_PAST_COUNT     = "optimizer.convergence_past_count"
LLM_KV_OPTIMIZER_PARAMETER_COUNT            = "optimizer.parameter_count"
LLM_KV_OPTIMIZER_ITERATION_COUNT            = "optimizer.iteration_count"
LLM_KV_OPTIMIZER_JUST_INITIALIZED           = "optimizer.just_initialized"
LLM_KV_OPTIMIZER_ADAM_BEST_LOSS             = "optimizer.adam.best_loss"
LLM_KV_OPTIMIZER_ADAM_PREVIOUS_LOSS         = "optimizer.adam.previous_loss"
LLM_KV_OPTIMIZER_ADAM_NO_IMPROVEMENT_COUNT  = "optimizer.adam.no_improvement_count"
LLM_KV_OPTIMIZER_LBFGS_APPROX_HESSIAN_COUNT = "optimizer.lbfgs.approx_hessian_count"
LLM_KV_OPTIMIZER_LBFGS_BEST_LOSS            = "optimizer.lbfgs.best_loss"
LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_STEP     = "optimizer.lbfgs.line_search_step"
LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_J        = "optimizer.lbfgs.line_search_j"
LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_K        = "optimizer.lbfgs.line_search_k"
LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_END      = "optimizer.lbfgs.line_search_end"
LLM_KV_OPTIMIZER_LBFGS_NO_IMPROVEMENT_COUNT = "optimizer.lbfgs.no_improvement_count"

LLM_TENSOR_OPTIMIZER_ADAM_FIRST_MOMENTS    = "optimizer.adam.first_moments"
LLM_TENSOR_OPTIMIZER_ADAM_SECOND_MOMENTS   = "optimizer.adam.second_moments"
LLM_TENSOR_OPTIMIZER_ADAM_PAST_LOSS_VALUES = "optimizer.adam.past_loss_values"

LLM_TENSOR_OPTIMIZER_LBFGS_CURRENT_PARAMETERS  = "optimizer.lbfgs.current_parameters"
LLM_TENSOR_OPTIMIZER_LBFGS_PREVIOUS_PARAMETERS = "optimizer.lbfgs.previous_parameters"
LLM_TENSOR_OPTIMIZER_LBFGS_CURRENT_GRADIENTS   = "optimizer.lbfgs.current_gradients"
LLM_TENSOR_OPTIMIZER_LBFGS_PREVIOUS_GRADIENTS  = "optimizer.lbfgs.previous_gradients"
LLM_TENSOR_OPTIMIZER_LBFGS_SEARCH_DIRECTION    = "optimizer.lbfgs.search_direction"
LLM_TENSOR_OPTIMIZER_LBFGS_PAST_LOSS_VALUES    = "optimizer.lbfgs.past_loss_values"
LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_ALPHA        = "optimizer.lbfgs.memory_alpha"
LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_YS           = "optimizer.lbfgs.memory_ys"
LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_S            = "optimizer.lbfgs.memory_s"
LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_Y            = "optimizer.lbfgs.memory_y"

LLM_KV_TRAINING_TYPE_TRAIN_MODEL   = "train_model"
LLM_KV_TRAINING_TYPE_FINETUNE_LORA = "finetune_lora"
LLM_KV_TRAINING_TYPE               = "training.type"
LLM_KV_TRAINING_FILE_VERSION       = "training.file_version"
LLM_KV_TRAINING_ITERATION_COUNT    = "training.iteration_count"
LLM_KV_TRAINING_SAMPLE_COUNT       = "training.sample_count"
LLM_KV_TRAINING_TOKEN_COUNT        = "training.token_count"

LLM_KV_TRAINING_LORA_RANK_TOKEN_EMBD  = "training.lora.rank.token_embd"
LLM_KV_TRAINING_LORA_RANK_OUTPUT_NORM = "training.lora.rank.output_norm"
LLM_KV_TRAINING_LORA_RANK_OUTPUT      = "training.lora.rank.output"
LLM_KV_TRAINING_LORA_RANK_ATTN_NORM   = "training.lora.rank.attn_norm"
LLM_KV_TRAINING_LORA_RANK_ATTN_Q      = "training.lora.rank.attn_q"
LLM_KV_TRAINING_LORA_RANK_ATTN_K      = "training.lora.rank.attn_k"
LLM_KV_TRAINING_LORA_RANK_ATTN_V      = "training.lora.rank.attn_v"
LLM_KV_TRAINING_LORA_RANK_ATTN_OUT    = "training.lora.rank.attn_output"
LLM_KV_TRAINING_LORA_RANK_FFN_NORM    = "training.lora.rank.ffn_norm"
LLM_KV_TRAINING_LORA_RANK_FFN_GATE    = "training.lora.rank.ffn_gate"
LLM_KV_TRAINING_LORA_RANK_FFN_DOWN    = "training.lora.rank.ffn_down"
LLM_KV_TRAINING_LORA_RANK_FFN_UP      = "training.lora.rank.ffn_up"

class Tensor:
    def __init__(self, dtype='f', ne=None):
        if ne is None:
            ne = []
        self.dtype = dtype
        self.ne = ne
        self.nbytes = 0
        if self.dtype == 'f':
            if len(self.ne) == 0:
                self.nbytes = 0
            else:
                self.nbytes = int(np.product(self.ne)) * 4
        else:
            raise ValueError(f"Unhandled data type '{self.dtype}'")

    def load(self, data, offset):
        nd = struct.unpack('<I', bytes(data[offset:offset + 4]))[0]; offset += 4
        namelen = struct.unpack('<I', bytes(data[offset:offset + 4]))[0]; offset += 4
        dtype = struct.unpack('<I', bytes(data[offset:offset + 4]))[0]; offset += 4

        assert(nd == len(self.ne))
        ne = []
        for d in range(nd):
            n = struct.unpack('<I', bytes(data[offset:offset + 4]))[0]; offset += 4
            ne.append(n)

        if tuple(ne) != tuple(self.ne):
            raise ValueError(f"Tensor.load: Expected number of elements {str(self.ne)} does not match what is read from file {str(ne)}")

        if self.dtype == 'f':
            assert(dtype == 0)
        else:
            raise ValueError(f"Unhandled data type '{self.dtype}'")

        self.name = bytes(data[offset:offset+namelen]); offset += namelen
        # 32-byte alignment
        offset += (0 - offset) & 31
        self.data = data[offset:offset+self.nbytes]
        offset += self.nbytes
        return offset

    def max_storage_size(self):
        result = 0
        result += 4 # nd
        result += 4 # namelen
        result += 4 # dtype
        result += len(self.ne)*8 # ne
        result += 48 # name (maximum as of commit 3b5515bbe0e2224425986ba24f1f5d84aa38dce9)
        result += 31 # 32-byte alignment
        result += self.nbytes
        return result

    def save_gguf(self, gguf_writer, name):
        gguf_writer.add_tensor(
            name=name,
            tensor=self.data,
            raw_shape=np.array(list(reversed(self.ne))),
            raw_dtype=gguf.GGMLQuantizationType.F32)

class OptimizationContext:
    def __init__(self):
        pass

    def load(self, data, offset):
        self.version = struct.unpack('<I', bytes(data[offset:offset + 4]))[0]
        offset += 4

        if self.version != 1:
            raise ValueError('Invalid version of optimization context in checkpoint file')

        self.past    = struct.unpack('<i', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.lbfgs_m = struct.unpack('<i', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.nx      = struct.unpack('N',  bytes(data[offset:offset + 8]))[0];  offset += 8
        self.iter    = struct.unpack('<i', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.just_initialized = bool(struct.unpack('<i', bytes(data[offset:offset + 4]))[0]);  offset += 4

        self.adam_m  = Tensor('f', [self.nx])
        self.adam_v  = Tensor('f', [self.nx])
        self.adam_pf = Tensor('f', [self.past] if self.past > 0 else [])

        self.lbfgs_x    = Tensor('f', [self.nx])
        self.lbfgs_xp   = Tensor('f', [self.nx])
        self.lbfgs_g    = Tensor('f', [self.nx])
        self.lbfgs_gp   = Tensor('f', [self.nx])
        self.lbfgs_d    = Tensor('f', [self.nx])
        self.lbfgs_pf   = Tensor('f', [self.past] if self.past > 0 else [])
        self.lbfgs_lmal = Tensor('f', [self.lbfgs_m])
        self.lbfgs_lmys = Tensor('f', [self.lbfgs_m])
        self.lbfgs_lms  = Tensor('f', [self.nx, self.lbfgs_m])
        self.lbfgs_lmy  = Tensor('f', [self.nx, self.lbfgs_m])

        # forgot to save type in version 1:
        # guess self.type from number of remaining bytes
        size_type_0 = 12 + sum([t.max_storage_size() for t in
                                [self.adam_m, self.adam_v]
                                +([self.adam_pf] if (self.past > 0) else [])])
        size_type_1 = 24 + sum([t.max_storage_size() for t in
                                [self.lbfgs_x, self.lbfgs_xp, self.lbfgs_g,
                                 self.lbfgs_gp, self.lbfgs_d, self.lbfgs_pf,
                                 self.lbfgs_lmal, self.lbfgs_lmys,
                                 self.lbfgs_lms, self.lbfgs_lmy]
                                 +([self.lbfgs_pf] if (self.past > 0) else [])])
        # due to alignment padding the size might not by exact
        # but the difference in size for both types is significant,
        # so we can just use whichever is closest
        remaining = len(data) - offset
        if abs(remaining - size_type_0) < abs(remaining - size_type_1):
            self.type = 0
        else:
            self.type = 1

        if self.type == 0:
            offset = self.adam_m.load(data, offset)
            offset = self.adam_v.load(data, offset)
            offset = self.adam_pf.load(data,offset)

            self.adam_fx_best          = struct.unpack('<f', bytes(data[offset:offset + 4]))[0];  offset += 4
            self.adam_fx_prev          = struct.unpack('<f', bytes(data[offset:offset + 4]))[0];  offset += 4
            self.adam_n_no_improvement = struct.unpack('<i', bytes(data[offset:offset + 4]))[0];  offset += 4

        elif self.type == 1:
            offset = self.lbfgs_x.load(data, offset)
            offset = self.lbfgs_xp.load(data, offset)
            offset = self.lbfgs_g.load(data, offset)
            offset = self.lbfgs_gp.load(data, offset)
            offset = self.lbfgs_d.load(data, offset)
            offset = self.lbfgs_pf.load(data, offset)
            offset = self.lbfgs_lmal.load(data, offset)
            offset = self.lbfgs_lmys.load(data, offset)
            offset = self.lbfgs_lms.load(data, offset)
            offset = self.lbfgs_lmy.load(data, offset)

            self.lbfgs_fx_best          = struct.unpack('<f', bytes(data[offset:offset + 4]))[0];  offset += 4
            self.lbfgs_step             = struct.unpack('<f', bytes(data[offset:offset + 4]))[0];  offset += 4
            self.lbfgs_j                = struct.unpack('<i', bytes(data[offset:offset + 4]))[0];  offset += 4
            self.lbfgs_k                = struct.unpack('<i', bytes(data[offset:offset + 4]))[0];  offset += 4
            self.lbfgs_end              = struct.unpack('<i', bytes(data[offset:offset + 4]))[0];  offset += 4
            self.lbfgs_n_no_improvement = struct.unpack('<i', bytes(data[offset:offset + 4]))[0];  offset += 4

        else:
            raise ValueError(f"Invalid optimizer type '{self.type}'")

        return offset

    def save_gguf(self, gguf_writer):
        gguf_writer.add_uint32(LLM_KV_OPTIMIZER_FILE_VERSION, 0)
        gguf_writer.add_uint32(LLM_KV_OPTIMIZER_CONVERGENCE_PAST_COUNT, self.past)
        gguf_writer.add_uint64(LLM_KV_OPTIMIZER_PARAMETER_COUNT, self.nx)
        gguf_writer.add_uint32(LLM_KV_OPTIMIZER_ITERATION_COUNT, self.iter)
        gguf_writer.add_bool(LLM_KV_OPTIMIZER_JUST_INITIALIZED, self.just_initialized)

        if self.type == 0:
            gguf_writer.add_string(LLM_KV_OPTIMIZER_TYPE, LLM_KV_OPTIMIZER_TYPE_ADAM)
            gguf_writer.add_float32(LLM_KV_OPTIMIZER_ADAM_BEST_LOSS, self.adam_fx_best)
            gguf_writer.add_float32(LLM_KV_OPTIMIZER_ADAM_PREVIOUS_LOSS, self.adam_fx_prev)
            gguf_writer.add_uint32(LLM_KV_OPTIMIZER_ADAM_NO_IMPROVEMENT_COUNT, self.adam_n_no_improvement)

            self.adam_m.save_gguf(gguf_writer, name=LLM_TENSOR_OPTIMIZER_ADAM_FIRST_MOMENTS)
            self.adam_v.save_gguf(gguf_writer, name=LLM_TENSOR_OPTIMIZER_ADAM_SECOND_MOMENTS)
            if self.past > 0:
                self.adam_pf.save_gguf(gguf_writer, name=LLM_TENSOR_OPTIMIZER_ADAM_PAST_LOSS_VALUES)

        elif self.type == 1:
            gguf_writer.add_string(LLM_KV_OPTIMIZER_TYPE, LLM_KV_OPTIMIZER_TYPE_LBFGS)
            gguf_writer.add_uint32(LLM_KV_OPTIMIZER_LBFGS_APPROX_HESSIAN_COUNT, self.lbfgs_m)
            gguf_writer.add_float32(LLM_KV_OPTIMIZER_LBFGS_BEST_LOSS, self.lbfgs_fx_best)
            gguf_writer.add_float32(LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_STEP, self.lbfgs_step)
            gguf_writer.add_int32(LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_J, self.lbfgs_j)
            gguf_writer.add_int32(LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_K, self.lbfgs_k)
            gguf_writer.add_int32(LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_END, self.lbfgs_end)
            gguf_writer.add_uint32(LLM_KV_OPTIMIZER_LBFGS_NO_IMPROVEMENT_COUNT, self.lbfgs_n_no_improvement)

            self.lbfgs_x.save_gguf(gguf_writer, name=LLM_TENSOR_OPTIMIZER_LBFGS_CURRENT_PARAMETERS)
            self.lbfgs_xp.save_gguf(gguf_writer, name=LLM_TENSOR_OPTIMIZER_LBFGS_PREVIOUS_PARAMETERS)
            self.lbfgs_g.save_gguf(gguf_writer, name=LLM_TENSOR_OPTIMIZER_LBFGS_CURRENT_GRADIENTS)
            self.lbfgs_gp.save_gguf(gguf_writer, name=LLM_TENSOR_OPTIMIZER_LBFGS_PREVIOUS_GRADIENTS)
            self.lbfgs_d.save_gguf(gguf_writer, name=LLM_TENSOR_OPTIMIZER_LBFGS_SEARCH_DIRECTION)
            if self.past > 0:
                self.lbfgs_pf.save_gguf(gguf_writer, name=LLM_TENSOR_OPTIMIZER_LBFGS_PAST_LOSS_VALUES)
            self.lbfgs_lmal.save_gguf(gguf_writer, name=LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_ALPHA)
            self.lbfgs_lmys.save_gguf(gguf_writer, name=LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_YS)
            self.lbfgs_lms.save_gguf(gguf_writer, name=LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_S)
            self.lbfgs_lmy.save_gguf(gguf_writer, name=LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_Y)
        else:
            raise ValueError('Unknown optimizer type')

class LoraParams:
    def __init__(self):
        pass

    def load(self, data, offset):
        self.n_rank_attention_norm  = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.n_rank_wq              = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.n_rank_wk              = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.n_rank_wv              = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.n_rank_wo              = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.n_rank_ffn_norm        = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.n_rank_w1              = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.n_rank_w2              = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.n_rank_w3              = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.n_rank_tok_embeddings  = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.n_rank_norm            = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.n_rank_output          = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        return offset

    def save_gguf(self, gguf_writer):
        gguf_writer.add_uint32(LLM_KV_TRAINING_LORA_RANK_TOKEN_EMBD,  self.n_rank_tok_embeddings)
        gguf_writer.add_uint32(LLM_KV_TRAINING_LORA_RANK_OUTPUT_NORM, self.n_rank_norm)
        gguf_writer.add_uint32(LLM_KV_TRAINING_LORA_RANK_OUTPUT,      self.n_rank_output)
        gguf_writer.add_uint32(LLM_KV_TRAINING_LORA_RANK_ATTN_NORM,   self.n_rank_attention_norm)
        gguf_writer.add_uint32(LLM_KV_TRAINING_LORA_RANK_ATTN_Q,      self.n_rank_wq)
        gguf_writer.add_uint32(LLM_KV_TRAINING_LORA_RANK_ATTN_K,      self.n_rank_wk)
        gguf_writer.add_uint32(LLM_KV_TRAINING_LORA_RANK_ATTN_V,      self.n_rank_wv)
        gguf_writer.add_uint32(LLM_KV_TRAINING_LORA_RANK_ATTN_OUT,    self.n_rank_wo)
        gguf_writer.add_uint32(LLM_KV_TRAINING_LORA_RANK_FFN_NORM,    self.n_rank_ffn_norm)
        gguf_writer.add_uint32(LLM_KV_TRAINING_LORA_RANK_FFN_GATE,    self.n_rank_w1)
        gguf_writer.add_uint32(LLM_KV_TRAINING_LORA_RANK_FFN_DOWN,    self.n_rank_w2)
        gguf_writer.add_uint32(LLM_KV_TRAINING_LORA_RANK_FFN_UP,      self.n_rank_w3)

class ModelParams:
    def __init__(self, n_ff = None):
        self.n_ff = n_ff

    def load(self, data, offset):
        self.n_vocab = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.n_embd  = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.n_mult  = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.n_head  = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.n_layer = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        self.n_rot   = struct.unpack('<I', bytes(data[offset:offset + 4]))[0];  offset += 4
        return offset

    def get_n_ff(self):
        if self.n_ff is None:
            # struct my_llama_model::get_n_ff in train-text-from-scratch.cpp commit 3b5515bbe0e2224425986ba24f1f5d84aa38dce9
            return ((2*(4*self.n_embd)//3 + self.n_mult - 1)//self.n_mult)*self.n_mult
        else:
            return self.n_ff

    def save_gguf(self, gguf_writer):
        # self.n_vocab not saved
        gguf_writer.add_embedding_length(self.n_embd)
        gguf_writer.add_head_count(self.n_head)
        gguf_writer.add_block_count(self.n_layer)
        gguf_writer.add_rope_dimension_count(self.n_rot)
        gguf_writer.add_feed_forward_length(self.get_n_ff())

def tensor_name(key, bid=None, suffix=".weight"):
    return gguf.TENSOR_NAMES[key].format(bid=bid) + suffix

class Layer:
    def __init__(self, params, lora_params, bid):
        self.bid = bid
        self.att_norm_a = Tensor('f', [lora_params.n_rank_attention_norm, params.n_embd])
        self.att_norm_b = Tensor('f', [lora_params.n_rank_attention_norm, 1])
        self.wq_a       = Tensor('f', [lora_params.n_rank_wq, params.n_embd])
        self.wq_b       = Tensor('f', [lora_params.n_rank_wq, params.n_embd])
        self.wk_a       = Tensor('f', [lora_params.n_rank_wk, params.n_embd])
        self.wk_b       = Tensor('f', [lora_params.n_rank_wk, params.n_embd])
        self.wv_a       = Tensor('f', [lora_params.n_rank_wv, params.n_embd])
        self.wv_b       = Tensor('f', [lora_params.n_rank_wv, params.n_embd])
        self.wo_a       = Tensor('f', [lora_params.n_rank_wo, params.n_embd])
        self.wo_b       = Tensor('f', [lora_params.n_rank_wo, params.n_embd])
        self.ffn_norm_a = Tensor('f', [lora_params.n_rank_ffn_norm, params.n_embd])
        self.ffn_norm_b = Tensor('f', [lora_params.n_rank_ffn_norm, 1])
        self.w1_a       = Tensor('f', [lora_params.n_rank_w1, params.n_embd])
        self.w1_b       = Tensor('f', [lora_params.n_rank_w1, params.get_n_ff()])
        self.w2_a       = Tensor('f', [lora_params.n_rank_w2, params.get_n_ff()])
        self.w2_b       = Tensor('f', [lora_params.n_rank_w2, params.n_embd])
        self.w3_a       = Tensor('f', [lora_params.n_rank_w3, params.n_embd])
        self.w3_b       = Tensor('f', [lora_params.n_rank_w3, params.get_n_ff()])

    def load(self, data, offset):
        offset = self.att_norm_a.load(data, offset)
        offset = self.att_norm_b.load(data, offset)
        offset = self.wq_a.load(data, offset)
        offset = self.wq_b.load(data, offset)
        offset = self.wk_a.load(data, offset)
        offset = self.wk_b.load(data, offset)
        offset = self.wv_a.load(data, offset)
        offset = self.wv_b.load(data, offset)
        offset = self.wo_a.load(data, offset)
        offset = self.wo_b.load(data, offset)
        offset = self.ffn_norm_a.load(data, offset)
        offset = self.ffn_norm_b.load(data, offset)
        offset = self.w1_a.load(data, offset)
        offset = self.w1_b.load(data, offset)
        offset = self.w2_a.load(data, offset)
        offset = self.w2_b.load(data, offset)
        offset = self.w3_a.load(data, offset)
        offset = self.w3_b.load(data, offset)
        return offset

    def save_gguf(self, gguf_writer):
        self.att_norm_a.save_gguf(gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.ATTN_NORM, self.bid, ".weight.lora_a"))
        self.att_norm_b.save_gguf(gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.ATTN_NORM, self.bid, ".weight.lora_b"))
        self.wq_a.save_gguf      (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.ATTN_Q,    self.bid, ".weight.lora_a"))
        self.wq_b.save_gguf      (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.ATTN_Q,    self.bid, ".weight.lora_b"))
        self.wk_a.save_gguf      (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.ATTN_K,    self.bid, ".weight.lora_a"))
        self.wk_b.save_gguf      (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.ATTN_K,    self.bid, ".weight.lora_b"))
        self.wv_a.save_gguf      (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.ATTN_V,    self.bid, ".weight.lora_a"))
        self.wv_b.save_gguf      (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.ATTN_V,    self.bid, ".weight.lora_b"))
        self.wo_a.save_gguf      (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.ATTN_OUT,  self.bid, ".weight.lora_a"))
        self.wo_b.save_gguf      (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.ATTN_OUT,  self.bid, ".weight.lora_b"))
        self.ffn_norm_a.save_gguf(gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.FFN_NORM,  self.bid, ".weight.lora_a"))
        self.ffn_norm_b.save_gguf(gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.FFN_NORM,  self.bid, ".weight.lora_b"))
        self.w1_a.save_gguf      (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.FFN_GATE,  self.bid, ".weight.lora_a"))
        self.w1_b.save_gguf      (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.FFN_GATE,  self.bid, ".weight.lora_b"))
        self.w2_a.save_gguf      (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.FFN_DOWN,  self.bid, ".weight.lora_a"))
        self.w2_b.save_gguf      (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.FFN_DOWN,  self.bid, ".weight.lora_b"))
        self.w3_a.save_gguf      (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.FFN_UP,    self.bid, ".weight.lora_a"))
        self.w3_b.save_gguf      (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.FFN_UP,    self.bid, ".weight.lora_b"))

class LoraModel:
    def __init__(self, n_ff = None):
        self.params = ModelParams(n_ff = n_ff)
        self.lora_params = LoraParams()
        self.layers = []

    def load(self, data, offset):
        offset = self.params.load(data, offset)
        offset = self.lora_params.load(data, offset)

        self.tok_embd_a = Tensor('f', [self.lora_params.n_rank_tok_embeddings, self.params.n_embd])
        self.tok_embd_b = Tensor('f', [self.lora_params.n_rank_tok_embeddings, self.params.n_vocab])
        self.norm_a     = Tensor('f', [self.lora_params.n_rank_norm, self.params.n_embd])
        self.norm_b     = Tensor('f', [self.lora_params.n_rank_norm, 1])
        self.output_a   = Tensor('f', [self.lora_params.n_rank_output, self.params.n_embd])
        self.output_b   = Tensor('f', [self.lora_params.n_rank_output, self.params.n_vocab])

        offset = self.tok_embd_a.load(data, offset)
        offset = self.tok_embd_b.load(data, offset)
        offset = self.norm_a.load(data, offset)
        offset = self.norm_b.load(data, offset)
        offset = self.output_a.load(data, offset)
        offset = self.output_b.load(data, offset)

        self.layers.clear()
        for bid in range(self.params.n_layer):
            layer = Layer(self.params, self.lora_params, bid)
            offset = layer.load(data, offset)
            self.layers.append(layer)

        return offset

    def save_gguf(self, gguf_writer):
        self.params.save_gguf(gguf_writer)
        self.lora_params.save_gguf(gguf_writer)

        self.tok_embd_a.save_gguf(gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD,  suffix=".weight.lora_a"))
        self.tok_embd_b.save_gguf(gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD,  suffix=".weight.lora_b"))
        self.norm_a.save_gguf    (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.OUTPUT_NORM, suffix=".weight.lora_a"))
        self.norm_b.save_gguf    (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.OUTPUT_NORM, suffix=".weight.lora_b"))
        self.output_a.save_gguf  (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.OUTPUT,      suffix=".weight.lora_a"))
        self.output_b.save_gguf  (gguf_writer, name=tensor_name(gguf.MODEL_TENSOR.OUTPUT,      suffix=".weight.lora_b"))

        for layer in self.layers:
            layer.save_gguf(gguf_writer)

class LoraCheckpoint:
    def __init__(self, n_ff = None):
        self.model = LoraModel(n_ff = n_ff)
        self.opt_ctx = OptimizationContext()

    def load(self, data, offset):
        magic   = bytes(reversed(data[offset:offset + 4])); offset += 4
        if magic != b'ggcl':
            raise ValueError(f"File header magic indicates, that this is no finetune-lora checkpoint file. Expected 'ggcl', Got '{str(magic)}'")

        self.version = struct.unpack('<I', bytes(data[offset:offset + 4]))[0]; offset += 4
        if self.version != 0:
            raise ValueError('Invalid version of checkpoint file')

        self.train_its     = struct.unpack('<I', bytes(data[offset:offset + 4]))[0]; offset += 4
        self.train_samples = struct.unpack('<I', bytes(data[offset:offset + 4]))[0]; offset += 4
        self.train_tokens  = struct.unpack('<I', bytes(data[offset:offset + 4]))[0]; offset += 4

        offset = self.model.load(data, offset)
        offset = self.opt_ctx.load(data, offset)

        return offset

    def save_gguf(self, gguf_writer):
        gguf_writer.add_file_type(gguf.GGMLQuantizationType.F32)
        gguf_writer.add_layer_norm_rms_eps(1e-5)
        gguf_writer.add_uint32(LLM_KV_TRAINING_FILE_VERSION,    0)
        gguf_writer.add_string(LLM_KV_TRAINING_TYPE,            LLM_KV_TRAINING_TYPE_FINETUNE_LORA)
        gguf_writer.add_uint32(LLM_KV_TRAINING_ITERATION_COUNT, self.train_its)
        gguf_writer.add_uint32(LLM_KV_TRAINING_SAMPLE_COUNT,    self.train_samples)
        gguf_writer.add_uint32(LLM_KV_TRAINING_TOKEN_COUNT,     self.train_tokens)
        self.model.save_gguf(gguf_writer)
        self.opt_ctx.save_gguf(gguf_writer)

def handle_args():
    parser = argparse.ArgumentParser(description = 'Convert finetune checkpoints to GGUF')
    parser.add_argument('--input',  '-i', type = Path, help = 'Input finetune checkpoint filename', required=True)
    parser.add_argument('--output', '-o', type = Path, help = 'Output GGUF filename', required=True)
    parser.add_argument('--ff', type = int, help = "Feedforward size, if not provided compute from n_mult. Provide this if you get 'ValueError: Tensor.load: Expected number of elements does not match what is read from file'", required=False)
    return parser.parse_args()

def main():
    cfg = handle_args()
    print(cfg)
    data = np.memmap(cfg.input, mode = 'r')
    chk = LoraCheckpoint(n_ff = cfg.ff)
    offset = 0
    offset = chk.load(data, offset)
    # we should have read all available data
    assert(offset == len(data))

    gguf_writer = gguf.GGUFWriter(cfg.output, gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.LLAMA], use_temp_file = False)
    chk.save_gguf(gguf_writer)
    print("    gguf: write header")
    gguf_writer.write_header_to_file()
    print("    gguf: write metadata")
    gguf_writer.write_kv_data_to_file()
    print("    gguf: write tensors")
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

if __name__ == '__main__':
    main()
