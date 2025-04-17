#pragma once

#include <string>
#include <vector>

#include "ggml.h"

// Allowed tensors for arbitrary quantization with --tensor-type option
static const std::vector<std::string> ALLOWED_TENSOR_TYPE = {
  "attn_k",
  "attn_k_b",
  "attn_kv_a_mqa",
  "attn_kv_b",
  "attn_o",
  "attn_output",
  "attn_q",
  "attn_q_a",
  "attn_q_b",
  "attn_qkv",
  "attn_rel_b",
  "attn_v",
  "attn_v_b",
  "channel_mix_key",
  "channel_mix_receptance",
  "channel_mix_value",
  "cls",
  "cls.output",
  "conv1",
  "conv1d",
  "conv2",
  "cross_attn_k",
  "cross_attn_o",
  "cross_attn_q",
  "cross_attn_rel_b",
  "cross_attn_v",
  "dw",
  "ffn_down",
  "ffn_down_exps",
  "ffn_down_shexp",
  "ffn_gate",
  "ffn_gate_exps",
  "ffn_gate_shexp",
  "ffn_up",
  "ffn_up_exps",
  "ffn_up_shexp",
  "pw1",
  "pw1",
  "ssm_a",
  "ssm_conv1d",
  "ssm_dt",
  "ssm_in",
  "ssm_out",
  "ssm_x",
  "time_mix_gate",
  "time_mix_key",
  "time_mix_output",
  "time_mix_receptance",
  "time_mix_value",
  "token_types"
};

// Quantization types
struct tensor_quantization {
  std::string name;
  ggml_type quant = GGML_TYPE_COUNT;
};
