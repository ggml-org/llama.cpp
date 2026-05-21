#include "models.h"

#include "ggml.h"
#include "llama-memory-recurrent.h"

#include <cmath>

/*
 * zaya.py ref: L52-81 (ResidualScaling class)
 *
 * class ResidualScaling(nn.Module):
 *     def __init__(self, config, layer_n, ...):
 *         self.not_first_layer = (layer_n != 0)
 *         self.hidden_states_scale = torch.nn.Parameter(torch.ones(config.hidden_size))
 *         self.hidden_states_bias  = torch.nn.Parameter(torch.zeros(config.hidden_size))
 *         if self.not_first_layer:
 *             self.residual_scale = torch.nn.Parameter(torch.ones(config.hidden_size))
 *             self.residual_bias  = torch.nn.Parameter(torch.zeros(config.hidden_size))
 *
 *     def forward(self, residual, hidden_states):
 *         hidden_states = (hidden_states.float() + hs_bias) * hs_scale
 *         if self.not_first_layer and residual is not None:
 *             residual = (residual.float() + res_bias) * res_scale
 *         return residual, hidden_states
 */

void llama_model_zaya::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL, hparams.ssm_d_conv);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);

    const uint32_t n_qk = (hparams.n_head() + hparams.n_head_kv()) * hparams.n_embd_head_k();
    hparams.ssm_d_inner = 2*n_qk + hparams.n_embd; // CCA conv state + delayed value stream state
    hparams.ssm_d_state = 1;
    hparams.ssm_n_group = 0;

    /*
     * zaya.py ref: L575-602 (layer alternation)
     *
     * for layer_n in range(config.num_hidden_layers):
     *     if layer_n % 2 == 1:
     *         self.layers.append(ZayaDecoderMLPLayer(...))   # MoE layer
     *     else:
     *         self.layers.append(ZayaDecoderATTLayer(...))   # Attention layer
     */
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = (i % 2) == 0;
    }

    switch (hparams.n_layer) {
        case 80: type = LLM_TYPE_8B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_zaya::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    /*
     * zaya.py ref: L569-573
     *
     * self.embed_tokens = VocabParallelEmbedding(
     *     self.vocab_size, config.hidden_size, org_num_embeddings=config.vocab_size)
     */
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    /*
     * zaya.py ref: L608-613
     *
     * if (config.normalization == "RMSNorm"):
     *     self.final_norm = RMSNorm(self.config.hidden_size, eps=config.norm_epsilon)
     * elif (config.normalization == "LayerNorm"):
     *     self.final_norm = nn.LayerNorm(self.config.hidden_size, eps=config.norm_epsilon)
     */
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    /*
     * zaya.py ref: L729-743
     *
     * self.lm_head = ParallelLMHead(self.unpadded_vocab_size, config.hidden_size, ...)
     * if self.config.tie_word_embeddings:
     *     self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
     */
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = tok_embd;  // tied weights
    }

    /*
     * zaya.py ref: L605-606 (final ResidualScaling after all layers)
     *
     * if self.config.scale_residual_merge:
     *     self.res_scale = ResidualScaling(config, config.num_hidden_layers)
     */
    zaya_res_scale_hs    = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS_FINAL,    "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    zaya_res_scale_hs_b  = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS_FINAL,    "bias"),   {n_embd}, TENSOR_NOT_REQUIRED);
    zaya_res_scale_res   = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES_FINAL,   "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    zaya_res_scale_res_b = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES_FINAL,   "bias"),   {n_embd}, TENSOR_NOT_REQUIRED);

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t d_conv      = hparams.ssm_d_conv;
    const int64_t n_ff_exp    = hparams.n_ff_exp;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t n_head    = hparams.n_head(i);
        const int64_t n_head_kv = hparams.n_head_kv(i);
        const int64_t n_embd_q  = n_head    * n_embd_head;
        const int64_t n_embd_k  = n_head_kv * n_embd_head;
        const int64_t n_qk      = n_embd_q + n_embd_k;
        const int64_t n_groups  = n_head + n_head_kv;
        const int64_t n_ff      = hparams.n_ff(i);
        const int64_t n_expert  = hparams.n_expert;

        /*
         * zaya.py ref: L212-217 (ZayaDecoderATTLayer input_norm)
         * zaya.py ref: L508-513 (ZayaDecoderMLPLayer input_norm)
         *
         * if (config.normalization == "RMSNorm"):
         *     self.input_norm = RMSNorm(self.config.hidden_size, eps=config.norm_epsilon)
         * elif (config.normalization == "LayerNorm"):
         *     self.input_norm = nn.LayerNorm(self.config.hidden_size, eps=config.norm_epsilon)
         */
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // CCA attention layers (even indices only)
        if (i % 2 == 0) {
            /*
             * zaya.py ref: L98-184 (ZayaAttention class)
             *
             * self.q_dim = cca_num_q_heads * head_dim
             * self.k_dim = cca_num_k_heads * head_dim
             * self.v_dim = cca_num_k_heads * head_dim
             *
             * self.qkv = CCA(config, cca_num_k_heads, cca_num_q_heads, cca_num_heads, ...)
             * self.o_proj = ReplicatedLinear(cca_num_q_heads * head_dim, hidden_size, ...)
             * self.attn = Attention(cca_num_q_heads, head_dim, scale, cca_num_k_heads, ...)
             * self.rotary_emb = get_rope(head_size=head_dim, ..., partial_rotary_factor=0.5)
             */

            /*
             * zaya.py ref: L125-138 (CCA layer for Q, K projections)
             *
             * self.qkv = CCA(...)
             * output_qkv = torch.zeros((hidden_states.shape[0], self.qkv_dim), ...)
             * self.qkv(hidden_states, output_qkv)
             * q, k, v = output_qkv.split([self.q_dim, self.k_dim, self.v_dim], dim=-1)
             */
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_q}, 0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_k}, 0);

            /*
             * zaya.py ref: CCA.py - value projections (val_proj1, val_proj2)
             *
             * V1 = val_proj1(x)
             * V2 = val_proj2(x_delayed)
             * V = concat(V1, V2)
             */
            layer.cca_val_proj1 = create_tensor(tn(LLM_TENSOR_CCA_VAL_PROJ1, "weight", i),
                {n_embd, n_embd_k / 2}, 0);
            layer.cca_val_proj2 = create_tensor(tn(LLM_TENSOR_CCA_VAL_PROJ2, "weight", i),
                {n_embd, n_embd_k / 2}, 0);

            /*
             * zaya.py ref: L139-144
             *
             * self.o_proj = ReplicatedLinear(self.cca_num_q_heads * self.head_dim,
             *                                self.hidden_size, bias=self.config.attention_bias, ...)
             */
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q, n_embd}, 0);

            /*
             * zaya.py ref: CCA.py - depthwise conv on QK
             *
             * conv_dw applied to [Q, K] concatenated
             */
            layer.cca_conv_dw   = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, n_qk}, 0);
            layer.cca_conv_dw_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i), {n_qk}, TENSOR_NOT_REQUIRED);

            /*
             * zaya.py ref: CCA.py - grouped conv on QK
             *
             * conv_grp applied after dw conv, with n_groups = n_head + n_head_kv
             */
            layer.cca_conv_grp   = create_tensor(tn(LLM_TENSOR_CCA_CONV_GRP, "weight", i),
                {d_conv, n_qk / n_groups, n_qk}, 0);
            layer.cca_conv_grp_b = create_tensor(tn(LLM_TENSOR_CCA_CONV_GRP, "bias", i), {n_qk}, 0);

            /*
             * zaya.py ref: CCA.py - K scaling after L2 norm
             *
             * Kcur = Kcur * cca_k_scale
             */
            layer.cca_k_scale = create_tensor(tn(LLM_TENSOR_CCA_K_SCALE, "weight", i), {n_head_kv}, 0);
        }

        /*
         * zaya.py ref: L52-81, L219-220, L515-516 (per-layer ResidualScaling)
         *
         * if self.config.scale_residual_merge:
         *     self.res_scale = ResidualScaling(config, layer_n)
         *
         * hidden_states = (hidden_states.float() + hs_bias) * hs_scale
         * residual = (residual.float() + res_bias) * res_scale
         */
        layer.res_scale_hs   = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS, "weight", i), {n_embd}, 0);
        layer.res_scale_hs_b = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.res_scale_res  = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.res_scale_res_b = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);

        // MoE layers (odd indices)
        if (i % 2 == 1) {
            /*
             * zaya.py ref: L251-380 (ZayaRouter class)
             *
             * self.down_proj = ReplicatedLinear(self.hidden_size, self.mlp_expansion, bias=True, ...)
             * self.rmsnorm_eda = RMSNorm(self.mlp_expansion, eps=ln_eps)
             * self.router_states_scale = nn.Parameter(torch.ones(self.mlp_expansion))  // EDA scale
             * self.router_mlp = nn.Sequential(
             *     ReplicatedLinear(D, D, bias=True, ...),
             *     nn.GELU(),
             *     ReplicatedLinear(D, D, bias=True, ...),
             *     nn.GELU(),
             *     ReplicatedLinear(D, E, bias=False, ...),
             * )
             * self.register_buffer("balancing_biases", torch.zeros(self.num_experts, dtype=torch.float32))
             */

            /*
             * zaya.py ref: L291
             *
             * self.down_proj = ReplicatedLinear(self.hidden_size, self.mlp_expansion, bias=True, ...)
             */
            layer.zaya_router_down   = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i),
                {n_embd, n_ff_exp}, 0);
            layer.zaya_router_down_b = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "bias", i),
                {n_ff_exp}, TENSOR_NOT_REQUIRED);

            /*
             * zaya.py ref: L298-299
             *
             * self.rmsnorm_eda = RMSNorm(self.mlp_expansion, eps=ln_eps)
             */
            layer.zaya_router_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i),
                {n_ff_exp}, 0);

            /*
             * zaya.py ref: L305-314 (router MLP layers 0, 2, 4)
             *
             * self.router_mlp = nn.Sequential(
             *     ReplicatedLinear(D, D, bias=True, ...),   // mlp0
             *     self.non_linearity,                        // GELU
             *     ReplicatedLinear(D, D, bias=True, ...),   // mlp2
             *     self.non_linearity,                        // GELU
             *     ReplicatedLinear(D, E, bias=False, ...),  // mlp4
             * )
             */
            layer.zaya_router_mlp0   = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i),
                {n_ff_exp, n_ff_exp}, 0);
            layer.zaya_router_mlp0_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i),
                {n_ff_exp}, TENSOR_NOT_REQUIRED);
            layer.zaya_router_mlp2   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP2, "weight", i),
                {n_ff_exp, n_ff_exp}, 0);
            layer.zaya_router_mlp2_b = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP2, "bias", i),
                {n_ff_exp}, TENSOR_NOT_REQUIRED);
            layer.zaya_router_mlp4   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP4, "weight", i),
                {n_ff_exp, n_expert + 1}, 0);

            /*
             * zaya.py ref: L317-319
             *
             * self.register_buffer("balancing_biases", torch.zeros(self.num_experts, dtype=torch.float32))
             * if self.use_mod:
             *     self.balancing_biases[-1] = -1.0
             */
            layer.zaya_router_biases = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_BIASES, "weight", i),
                {n_expert + 1}, TENSOR_NOT_REQUIRED);

            /*
             * zaya.py ref: L302-303
             *
             * self.router_states_scale = nn.Parameter(torch.ones(self.mlp_expansion))
             */
            layer.zaya_router_eda_scale = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_EDA_SCALE, "weight", i),
                {n_ff_exp}, TENSOR_NOT_REQUIRED);

            /*
             * zaya.py ref: L435-446 (FusedMoE experts)
             *
             * self.experts = FusedMoE(
             *     num_experts=self.num_moe_experts,
             *     top_k=self.topk,
             *     hidden_size=config.hidden_size,
             *     intermediate_size=ffn_hidden_size // 2,
             *     reduce_results=False,
             *     renormalize=False,
             *     custom_routing_function=_custom_routing_fn,
             *     activation="silu",
             *     ...
             * )
             */
            create_tensor_gate_up_exps(layer, i, n_embd, n_ff, n_expert, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i),
                {n_ff, n_embd, n_expert}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_zaya::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_zaya::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_expert    = hparams.n_expert;
    const int64_t n_seqs      = ubatch.n_seqs;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(n_tokens % n_seqs == 0);

    const int64_t n_seq_tokens = n_tokens / n_seqs;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    /*
     * zaya.py ref: L638-641 (ZayaModel.forward)
     *
     * if inputs_embeds is None:
     *     inputs_embeds = self.embed_tokens(input_ids)
     * residual = None
     * hidden_states = inputs_embeds
     * prev_router_hidden_states = None
     */
    inpL = build_inp_embd(model.tok_embd);

    auto * inp = build_inp_mem_hybrid();
    auto * inp_recr = inp->get_recr();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * residual    = nullptr;
    ggml_tensor * prev_router = nullptr;

    /*
     * zaya.py ref: L71-81 (ResidualScaling.forward)
     *
     * hidden_states = (hidden_states.float() + hs_bias) * hs_scale
     * if self.not_first_layer and residual is not None:
     *     residual = (residual.float() + res_bias) * res_scale
     * return residual, hidden_states
     */
    const auto apply_res_scale = [&](ggml_tensor * x, ggml_tensor * scale, ggml_tensor * bias, const char * name, int il) {
        if (scale == nullptr) {
            return x;
        }
        if (bias != nullptr) {
            x = ggml_add(ctx0, x, bias);
        }
        x = ggml_mul(ctx0, x, scale);
        cb(x, name, il);
        return x;
    };

    /*
     * zaya.py ref: L644-651 (ZayaModel.forward layer loop)
     *
     * for layer_n, decoder_layer in enumerate(self.layers):
     *     hidden_states, residual, prev_router_hidden_states = decoder_layer(
     *         hidden_states, residual, positions, layer_n, prev_router_hidden_states)
     */
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);
        const int64_t n_embd_q  = n_head    * n_embd_head;
        const int64_t n_embd_k  = n_head_kv * n_embd_head;
        const int64_t n_qk      = n_embd_q + n_embd_k;
        const int64_t n_groups  = n_head + n_head_kv;
        const int64_t n_gqa     = n_head / n_head_kv;

        /*
         * zaya.py ref: L234-241 (ZayaDecoderATTLayer.forward)
         * zaya.py ref: L530-537 (ZayaDecoderMLPLayer.forward)
         *
         * if self.config.scale_residual_merge:
         *     residual, hidden_states = self.res_scale(residual, hidden_states)
         * if residual is not None:
         *     residual = residual.float() + hidden_states.float()
         * else:
         *     residual = hidden_states.float()
         * hidden_states = _apply_norm_with_fp32_residual(self.input_norm, residual, layer_input_dtype)
         */
        ggml_tensor * hidden_states = apply_res_scale(inpL, layer.res_scale_hs, layer.res_scale_hs_b, "res_scale_hs", il);
        if (residual != nullptr) {
            residual = apply_res_scale(residual, layer.res_scale_res, layer.res_scale_res_b, "res_scale_res", il);
            residual = ggml_add(ctx0, hidden_states, residual);
        } else {
            residual = hidden_states;
        }
        cb(residual, "residual", il);

        /*
         * zaya.py ref: L84-95 (_apply_norm_with_fp32_residual)
         * zaya.py ref: L240-241, L536-537
         *
         * if isinstance(norm, RMSNorm):
         *     if residual.dtype != norm.weight.dtype:
         *         hidden_states = norm.forward_native(residual)
         *     else:
         *         hidden_states = norm(residual)
         *     return hidden_states.to(target_dtype)
         */
        cur = build_norm(residual, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "input_norm", il);

        if (il % 2 == 0) {
            // ===== CCA Attention =====
            /*
             * zaya.py ref: L98-184 (ZayaAttention)
             * zaya.py ref: L171-184 (ZayaAttention.forward)
             *
             * def forward(self, hidden_states, position_ids):
             *     output_qkv = torch.zeros((hidden_states.shape[0], self.qkv_dim), ...)
             *     self.qkv(hidden_states, output_qkv)
             *     q, k, v = output_qkv.split([self.q_dim, self.k_dim, self.v_dim], dim=-1)
             *     q, k = self.rotary_emb(position_ids, q, k)
             *     attn_output = self.attn(q, k, v)
             *     attn_output = self.o_proj(attn_output)
             *     return attn_output
             */

            const int64_t conv_state_size = 2*n_qk;
            const int64_t cca_state_size  = conv_state_size + n_embd;
            GGML_ASSERT((int64_t) hparams.n_embd_s() == cca_state_size);

            /*
             * zaya.py ref: CCA.py - recurrent state management
             *
             * CCA maintains conv_state and prev_hs in recurrent memory
             */
            ggml_tensor * cca_state_all = inp_recr->mctx->get_s_l(il);
            ggml_tensor * cca_state     = build_rs(inp_recr, cca_state_all, hparams.n_embd_s(), n_seqs);
            cb(cca_state, "cca_state", il);

            ggml_tensor * conv_state = ggml_view_3d(ctx0, cca_state, 2, n_qk, n_seqs,
                    2*ggml_element_size(cca_state),
                    cca_state->nb[1],
                    0);
            cb(conv_state, "cca_conv_state", il);

            ggml_tensor * prev_hs = ggml_view_2d(ctx0, cca_state, n_embd, n_seqs,
                    cca_state->nb[1],
                    conv_state_size*ggml_element_size(cca_state));
            cb(prev_hs, "cca_prev_hs", il);

            /*
             * zaya.py ref: L177-179
             *
             * output_qkv = torch.zeros((hidden_states.shape[0], self.qkv_dim), ...)
             * self.qkv(hidden_states, output_qkv)
             * q, k, v = output_qkv.split([self.q_dim, self.k_dim, self.v_dim], dim=-1)
             */
            ggml_tensor * Qraw = ggml_mul_mat(ctx0, layer.wq, cur);
            cb(Qraw, "Qraw", il);
            ggml_tensor * Kraw = ggml_mul_mat(ctx0, layer.wk, cur);
            cb(Kraw, "Kraw", il);

            /*
             * zaya.py ref: CCA.py - delayed hidden state stream for val_proj2
             *
             * During decode: comes from recurrent state
             * During prefill: one-token shift of current sequence
             *
             * hs_d = concat(prev_hs_last, cur[:-1])  along seq dimension
             */
            ggml_tensor * cur_state_src = ggml_cont(ctx0, cur);
            ggml_tensor * cur_seq = ggml_reshape_3d(ctx0, cur_state_src, n_embd, n_seq_tokens, n_seqs);

            ggml_tensor * hs_d = ggml_reshape_3d(ctx0, ggml_cont(ctx0, prev_hs), n_embd, 1, n_seqs);
            if (n_seq_tokens > 1) {
                ggml_tensor * cur_shift = ggml_view_3d(ctx0, cur_seq, n_embd, n_seq_tokens - 1, n_seqs,
                        cur_seq->nb[1],
                        cur_seq->nb[2],
                        0);
                hs_d = ggml_concat(ctx0, hs_d, cur_shift, 1);
            }
            hs_d = ggml_reshape_2d(ctx0, ggml_cont(ctx0, hs_d), n_embd, n_tokens);
            cb(hs_d, "cca_hs_d", il);

            /*
             * zaya.py ref: CCA.py - V projection
             *
             * V1 = val_proj1(cur)
             * V2 = val_proj2(hs_d)
             * Vcur = concat(V1, V2, dim=0)
             */
            ggml_tensor * V1 = ggml_mul_mat(ctx0, layer.cca_val_proj1, cur);
            cb(V1, "V1", il);
            ggml_tensor * V2 = ggml_mul_mat(ctx0, layer.cca_val_proj2, hs_d);
            cb(V2, "V2", il);
            ggml_tensor * Vcur = ggml_concat(ctx0, V1, V2, 0);
            cb(Vcur, "Vcur", il);

            /*
             * zaya.py ref: CCA.py - QK concatenation for conv
             *
             * QKraw = concat(Qraw, Kraw, dim=0)
             */
            ggml_tensor * QKraw = ggml_concat(ctx0, Qraw, Kraw, 0);
            cb(QKraw, "QKraw", il);

            /*
             * zaya.py ref: CCA.py - qk_mean computation
             *
             * Qpre: [n_embd_head, n_head, n_tokens]
             * Kpre: [n_embd_head, n_head_kv, n_tokens]
             * Kpre_grouped = repeat(Kpre, n_gqa times along head dim)
             * qk_mean_q = (Qpre + Kpre_rep) * 0.5
             *
             * Qgroup = group Q by GQA, mean across group
             * qk_mean_k = (Qmean + Kpre) * 0.5
             */
            ggml_tensor * Qpre = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Qraw), n_embd_head, n_head, n_tokens);
            ggml_tensor * Kpre = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Kraw), n_embd_head, n_head_kv, n_tokens);

            ggml_tensor * Kpre_grouped = ggml_reshape_4d(ctx0, Kpre, n_embd_head, 1, n_head_kv, n_tokens);
            Kpre_grouped = ggml_repeat_4d(ctx0, Kpre_grouped, n_embd_head, n_gqa, n_head_kv, n_tokens);
            ggml_tensor * Kpre_rep = ggml_reshape_3d(ctx0, Kpre_grouped, n_embd_head, n_head, n_tokens);
            ggml_tensor * qk_mean_q = ggml_scale(ctx0, ggml_add(ctx0, Qpre, Kpre_rep), 0.5f);
            cb(qk_mean_q, "qk_mean_q", il);

            ggml_tensor * Qgroup = ggml_reshape_4d(ctx0, Qpre, n_embd_head, n_gqa, n_head_kv, n_tokens);
            Qgroup = ggml_permute(ctx0, Qgroup, 1, 0, 2, 3);
            Qgroup = ggml_cont(ctx0, Qgroup);
            ggml_tensor * Qmean = ggml_mean(ctx0, Qgroup);
            Qmean = ggml_reshape_3d(ctx0, Qmean, n_embd_head, n_head_kv, n_tokens);
            ggml_tensor * qk_mean_k = ggml_scale(ctx0, ggml_add(ctx0, Qmean, Kpre), 0.5f);
            cb(qk_mean_k, "qk_mean_k", il);

            /*
             * zaya.py ref: CCA.py - conv state update
             *
             * conv_input = concat(conv_state, QKraw_reshaped, dim=0)
             * last_conv_states = conv_input[-2:]  (last 2 positions for state update)
             */
            ggml_tensor * QKraw_t = ggml_cont(ctx0, ggml_transpose(ctx0, QKraw));
            QKraw_t = ggml_reshape_3d(ctx0, QKraw_t, n_seq_tokens, n_qk, n_seqs);

            ggml_tensor * conv_input = ggml_concat(ctx0, conv_state, QKraw_t, 0);
            cb(conv_input, "cca_conv_input", il);

            ggml_tensor * last_conv_states = ggml_view_3d(ctx0, conv_input, 2, n_qk, n_seqs,
                    conv_input->nb[1],
                    conv_input->nb[2],
                    n_seq_tokens*conv_input->nb[0]);
            cb(last_conv_states, "cca_last_conv_states", il);

            /*
             * zaya.py ref: CCA.py - recurrent state write-back
             *
             * Update conv_state and prev_hs in recurrent memory for next step
             */
            const auto kv_head = inp_recr->mctx->get_head();
            ggml_tensor * conv_state_update_target = ggml_view_2d(ctx0, cca_state_all, conv_state_size, n_seqs,
                    cca_state_all->nb[1],
                    kv_head*cca_state_size*ggml_element_size(cca_state_all));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv_states, conv_state_update_target));

            ggml_tensor * last_hs = ggml_view_2d(ctx0, cur_seq, n_embd, n_seqs,
                    cur_seq->nb[2],
                    (n_seq_tokens - 1)*cur_seq->nb[1]);
            ggml_tensor * prev_hs_update_target = ggml_view_2d(ctx0, cca_state_all, n_embd, n_seqs,
                    cca_state_all->nb[1],
                    (kv_head*cca_state_size + conv_state_size)*ggml_element_size(cca_state_all));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_hs, prev_hs_update_target));

            /*
             * zaya.py ref: CCA.py - depthwise conv
             *
             * QK = ssm_conv(conv_input, conv_dw) + conv_dw_b
             */
            ggml_tensor * conv_dw = layer.cca_conv_dw;
            if (conv_dw->type != GGML_TYPE_F32) {
                conv_dw = ggml_cont(ctx0, ggml_cast(ctx0, conv_dw, GGML_TYPE_F32));
            }
            ggml_tensor * QK = ggml_ssm_conv(ctx0, conv_input, conv_dw);
            QK = ggml_cont(ctx0, ggml_permute(ctx0, QK, 1, 0, 2, 3));
            if (layer.cca_conv_dw_b) {
                QK = ggml_add(ctx0, QK, ggml_reshape_3d(ctx0, layer.cca_conv_dw_b, 1, n_qk, 1));
            }
            cb(QK, "QK_dw", il);

            /*
             * zaya.py ref: CCA.py - grouped conv
             *
             * QK = conv_1d_grouped(QK, conv_grp, n_groups) + conv_grp_b
             */
            ggml_tensor * conv_grp = layer.cca_conv_grp;
            if (conv_grp->type != GGML_TYPE_F16) {
                conv_grp = ggml_cont(ctx0, ggml_cast(ctx0, conv_grp, GGML_TYPE_F16));
            }
            QK = ggml_conv_1d_grouped(ctx0, conv_grp, QK, 1, 0, 1, n_groups);
            QK = ggml_add(ctx0, QK, ggml_reshape_3d(ctx0, layer.cca_conv_grp_b, 1, n_qk, 1));
            cb(QK, "QK_grp", il);

            QK = ggml_cont(ctx0, ggml_permute(ctx0, QK, 1, 0, 2, 3));
            QK = ggml_reshape_2d(ctx0, QK, n_qk, n_tokens);

            ggml_tensor * Q_conv = ggml_view_2d(ctx0, QK, n_embd_q, n_tokens, QK->nb[1], 0);
            ggml_tensor * K_conv = ggml_view_2d(ctx0, QK, n_embd_k, n_tokens, QK->nb[1], n_embd_q*ggml_element_size(QK));

            ggml_tensor * Qcur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Q_conv), n_embd_head, n_head, n_tokens);
            ggml_tensor * Kcur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, K_conv), n_embd_head, n_head_kv, n_tokens);

            /*
             * zaya.py ref: CCA.py - add qk_mean back to Q, K
             *
             * Qcur = Qcur + qk_mean_q
             * Kcur = Kcur + qk_mean_k
             */
            Qcur = ggml_add(ctx0, Qcur, qk_mean_q);
            Kcur = ggml_add(ctx0, Kcur, qk_mean_k);

            /*
             * zaya.py ref: CCA.py - L2 normalization and scaling
             *
             * Qcur = l2_norm(Qcur) * sqrt(n_embd_head)
             * Kcur = l2_norm(Kcur) * sqrt(n_embd_head) * cca_k_scale
             */
            Qcur = ggml_scale(ctx0, ggml_l2_norm(ctx0, Qcur, 1e-12f), sqrtf((float) n_embd_head));
            Kcur = ggml_scale(ctx0, ggml_l2_norm(ctx0, Kcur, 1e-12f), sqrtf((float) n_embd_head));
            Kcur = ggml_mul(ctx0, Kcur, ggml_reshape_3d(ctx0, layer.cca_k_scale, 1, n_head_kv, 1));
            cb(Qcur, "Qcur_pre_rope", il);
            cb(Kcur, "Kcur_pre_rope", il);

            /*
             * zaya.py ref: L155-164 (rotary embedding)
             *
             * self.rotary_emb = get_rope(
             *     head_size=self.head_dim,
             *     max_position=config.max_position_embeddings,
             *     is_neox_style=True,
             *     rope_parameters={"rope_theta": config.rope_theta, "rope_type": "default", "partial_rotary_factor": 0.5},
             * )
             * q, k = self.rotary_emb(position_ids, q, k)
             */
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);

            Vcur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Vcur), n_embd_head, n_head_kv, n_tokens);

            /*
             * zaya.py ref: L146-153, L181-182 (Attention + output projection)
             *
             * self.attn = Attention(self.cca_num_q_heads, self.head_dim, self.scale, self.cca_num_k_heads, ...)
             * attn_output = self.attn(q, k, v)
             * attn_output = self.o_proj(attn_output)
             */
            cur = build_attn(inp->get_attn(), layer.wo, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                1.0f / sqrtf((float) n_embd_head), il);
            cb(cur, "attn_out", il);

        } else {
            // ===== MoE Layer =====
            /*
             * zaya.py ref: L481-541 (ZayaDecoderMLPLayer)
             * zaya.py ref: L382-479 (ZayaBlock)
             * zaya.py ref: L251-380 (ZayaRouter)
             *
             * def forward(self, hidden_states, residual, position_ids, layer_n, prev_router_hidden_states):
             *     if self.config.scale_residual_merge:
             *         residual, hidden_states = self.res_scale(residual, hidden_states)
             *     residual = residual.float() + hidden_states.float()
             *     hidden_states = _apply_norm_with_fp32_residual(self.input_norm, residual, layer_input_dtype)
             *     hidden_states, prev_router_hidden_states = self.zaya_block(hidden_states, prev_router_hidden_states)
             *     return hidden_states, residual, prev_router_hidden_states
             */

            /*
             * zaya.py ref: L321-380 (ZayaRouter.forward)
             *
             * hs = self.down_proj(hidden_states)
             * if self.use_eda and (prev_router_hidden_states is not None):
             *     hs = hs + prev_router_hidden_states * self.router_states_scale
             * router_hidden_states_next = hs[-S:].clone()
             * hs_norm = self.rmsnorm_eda(hs)
             * logits = self.router_mlp(hs_norm)  // Linear->GELU->Linear->GELU->Linear
             * expert_prob = torch.softmax(logits, dim=-1, dtype=torch.float32)
             * biased = expert_prob.detach().to(torch.float32) + self.balancing_biases
             * _, expert_choice_t = torch.topk(biased, self.topk, dim=-1)
             * route_prob = torch.gather(expert_prob, dim=1, index=expert_choice_t)
             * return route_prob_flat, expert_choice_flat, router_hidden_states_next
             */

            /*
             * zaya.py ref: L343
             *
             * hs = self.down_proj(hidden_states)
             */
            ggml_tensor * router_h = ggml_mul_mat(ctx0, layer.zaya_router_down, cur);
            router_h = ggml_add(ctx0, router_h, layer.zaya_router_down_b);
            cb(router_h, "router_down", il);

            /*
             * zaya.py ref: L344-345
             *
             * if self.use_eda and (prev_router_hidden_states is not None):
             *     hs = hs + prev_router_hidden_states * self.router_states_scale
             */
            if (prev_router != nullptr && layer.zaya_router_eda_scale != nullptr) {
                router_h = ggml_add(ctx0, router_h, ggml_mul(ctx0, prev_router, layer.zaya_router_eda_scale));
                cb(router_h, "router_eda", il);
            }

            prev_router = router_h;  // zaya.py ref: L348 (router_hidden_states_next)

            /*
             * zaya.py ref: L351
             *
             * hs_norm = self.rmsnorm_eda(hs)
             */
            router_h = build_norm(router_h, layer.zaya_router_norm, nullptr, LLM_NORM_RMS, il);
            cb(router_h, "router_norm", il);

            /*
             * zaya.py ref: L305-314, L354
             *
             * logits = self.router_mlp(hs_norm)
             * self.router_mlp = nn.Sequential(
             *     ReplicatedLinear(D, D, bias=True, ...),   // mlp0
             *     nn.GELU(),
             *     ReplicatedLinear(D, D, bias=True, ...),   // mlp2
             *     nn.GELU(),
             *     ReplicatedLinear(D, E, bias=False, ...),  // mlp4
             * )
             */
            router_h = ggml_mul_mat(ctx0, layer.zaya_router_mlp0, router_h);
            router_h = ggml_add(ctx0, router_h, layer.zaya_router_mlp0_b);
            router_h = ggml_gelu(ctx0, router_h);
            cb(router_h, "router_mlp0", il);

            router_h = ggml_mul_mat(ctx0, layer.zaya_router_mlp2, router_h);
            router_h = ggml_add(ctx0, router_h, layer.zaya_router_mlp2_b);
            router_h = ggml_gelu(ctx0, router_h);
            cb(router_h, "router_mlp2", il);

            router_h = ggml_mul_mat(ctx0, layer.zaya_router_mlp4, router_h);
            cb(router_h, "router_logits", il);

            /*
             * zaya.py ref: L355-359
             *
             * expert_prob = torch.softmax(logits, dim=-1, dtype=torch.float32)
             * biased = expert_prob.detach().to(torch.float32) + self.balancing_biases
             * _, expert_choice_t = torch.topk(biased, self.topk, dim=-1)
             */
            ggml_tensor * router_probs = ggml_soft_max(ctx0, router_h);
            cb(router_probs, "router_probs", il);

            /*
             * zaya.py ref: L387-389 (MOD skip expert handling)
             *
             * gate_probs = router_probs[:, :n_expert]  // exclude skip expert from routing
             */
            ggml_tensor * gate_probs = ggml_cont(ctx0,
                    ggml_view_2d(ctx0, router_probs, n_expert, n_tokens, router_probs->nb[1], 0));
            cb(gate_probs, "gate_probs", il);

            /*
             * zaya.py ref: L317-319, L362-363
             *
             * self.register_buffer("balancing_biases", torch.zeros(self.num_experts, dtype=torch.float32))
             * biased = expert_prob.detach().to(torch.float32) + self.balancing_biases
             */
            ggml_tensor * expert_biases = nullptr;
            if (layer.zaya_router_biases != nullptr) {
                expert_biases = ggml_view_1d(ctx0, layer.zaya_router_biases, n_expert, 0);
            }

            /*
             * zaya.py ref: L448-479 (ZayaBlock.forward - MoE execution)
             *
             * probs, indices, router_hidden_states_out = self.router(hidden_states, prev_router_hidden_states)
             * if self.config.zaya_use_mod:
             *     clamped_indices = torch.clamp(indices, min=0, max=self.num_moe_experts - 1)
             *     packed_logits = torch.cat([probs, clamped_indices.to(probs.dtype)], dim=-1)
             *     hidden_states_experts = self.experts(hidden_states, packed_logits)
             *     hidden_states_mod = hidden_states * probs
             *     mod_mask = (indices != self.num_moe_experts)
             *     hidden_states = (mod_mask * hidden_states_experts) + ((~mod_mask) * hidden_states_mod)
             * else:
             *     packed_logits = torch.cat([probs, indices.to(probs.dtype)], dim=-1)
             *     hidden_states = self.experts(hidden_states, packed_logits)
             */
            cur = build_moe_ffn(cur,
                /* gate_inp */        nullptr,
                /* up_exps */         nullptr,
                /* gate_exps */       nullptr,
                /* down_exps */       layer.ffn_down_exps,
                /* exp_probs_b */     expert_biases,
                /* n_expert */        n_expert,
                /* n_expert_used */   hparams.n_expert_used,
                /* type_op */         LLM_FFN_SILU,
                /* norm_w */          false,
                /* w_scale */         1.0f,
                /* gating_op */       LLAMA_EXPERT_GATING_FUNC_TYPE_NONE,
                /* il */              il,
                /* probs_in */        gate_probs,
                /* gate_up_exps */    layer.ffn_gate_up_exps);
            cb(cur, "moe_out", il);
        }

        inpL = cur;
    }

    /*
     * zaya.py ref: L653-664 (ZayaModel.forward - final residual + norm)
     *
     * if self.config.scale_residual_merge:
     *     residual, hidden_states = self.res_scale(residual, hidden_states)
     * if residual is not None:
     *     hidden_states = hidden_states.float() + residual.float()
     * else:
     *     hidden_states = hidden_states.float()
     * hidden_states = _apply_norm_with_fp32_residual(self.final_norm, hidden_states, final_input_dtype)
     */
    ggml_tensor * final_hidden = apply_res_scale(inpL, model.zaya_res_scale_hs, model.zaya_res_scale_hs_b, "final_res_scale_hs", -1);
    if (residual != nullptr) {
        residual = apply_res_scale(residual, model.zaya_res_scale_res, model.zaya_res_scale_res_b, "final_res_scale_res", -1);
        cur = ggml_add(ctx0, final_hidden, residual);
    } else {
        cur = final_hidden;
    }
    cb(cur, "final_residual", -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    /*
     * zaya.py ref: L608-613 (final norm)
     *
     * if (config.normalization == "RMSNorm"):
     *     self.final_norm = RMSNorm(self.config.hidden_size, eps=config.norm_epsilon)
     */
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    /*
     * zaya.py ref: L729-746, L769-782 (lm_head + logits_processor)
     *
     * self.lm_head = ParallelLMHead(self.unpadded_vocab_size, config.hidden_size, ...)
     * logits = self.logits_processor(self.lm_head, hidden_states)
     */
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
