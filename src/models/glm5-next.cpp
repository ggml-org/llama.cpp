#include "models.h"
#include "llama-memory-hybrid-idx.h"

// GLM5-Next (GLM-5.3-Flash): hybrid KDA (linear) + nope MLA with a k-pool DSA indexer,
// mHC residual streams, DeepSeek-style MoE.

void llama_model_glm5_next::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,     hparams.f_norm_eps, false);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,             hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                hparams.n_embd_head_kda);
    ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND,        hparams.kda_gate_lower_bound, false);

    // the MLA cache holds the compressed latent
    hparams.n_embd_head_v_full = hparams.n_lora_kv;

    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all);

    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (i >= hparams.n_layer()) {
            hparams.n_head_kv_arr[i] = 1;
        }
        hparams.is_recr_impl[i] = i < hparams.n_layer() && hparams.n_head_kv(i) == 0;
    }

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,  hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func, false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,   hparams.swiglu_clamp_exp,   hparams.n_layer_all, false);
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, hparams.n_layer_all, false)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }

    // DSA indexer with k-pool compression
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,        hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,        hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,             hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KPOOL,             hparams.indexer_kpool);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, hparams.indexer_kpool_select_tail, false);
    GGML_ASSERT(hparams.indexer_kpool > 1 && hparams.indexer_top_k % hparams.indexer_kpool == 0);
    std::fill(hparams.is_indexer_full_impl.begin(), hparams.is_indexer_full_impl.end(), 1);
    ml.get_key_or_arr(LLM_KV_ATTENTION_INDEXER_TYPES, hparams.is_indexer_full_impl, hparams.n_layer(), false);

    // mHC
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    GGML_ASSERT(hparams.dsv4_hc_mult == 4 && "mHC with hc_mult != 4 is not supported");

    switch (hparams.n_layer()) {
        case 45: type = LLM_TYPE_320B_A18B; break; // GLM-5.3-Flash
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_glm5_next::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_mix_dim = (2 + hc)*hc;

    // the NextN block is loaded but only used by the MTP graph (TODO)
    const std::string mtp_probe = "blk." + std::to_string(n_layer) + ".nextn.eh_proj.weight";
    const bool trunk_only = (n_layer_nextn > 0) && (ml.get_weight(mtp_probe.c_str()) == nullptr);
    int mtp_flags = trunk_only ? TENSOR_NOT_REQUIRED : 0;
    if (!ml.load_mtp) {
        mtp_flags |= TENSOR_SKIP;
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];

        const int flags = (i >= n_layer) ? mtp_flags : 0;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, flags);

        if (i < n_layer) {
            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc*n_embd, hc_mix_dim}, 0);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc*n_embd, hc_mix_dim}, 0);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);
        }

        const int64_t head_dim = hparams.n_embd_head_kda;
        const int64_t d_conv   = hparams.ssm_d_conv;
        const int64_t d_inner  = head_dim * n_head;

        if (hparams.is_recr(i)) {
            auto conv = [&](llm_tensor tid) {
                ggml_tensor * t = create_tensor(tn(tid, "weight", i), {d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
                return t ? t : create_tensor(tn(tid, "weight", i), {d_conv, 1, d_inner}, 0);
            };
            layer.ssm_q_conv = conv(LLM_TENSOR_SSM_CONV1D_Q);
            layer.ssm_k_conv = conv(LLM_TENSOR_SSM_CONV1D_K);
            layer.ssm_v_conv = conv(LLM_TENSOR_SSM_CONV1D_V);

            create_tensor_qkv(layer, i, n_embd, d_inner, d_inner, d_inner, 0);

            layer.ssm_f_a  = create_tensor(tn(LLM_TENSOR_SSM_F_A,  "weight", i), {n_embd, head_dim}, 0);
            layer.ssm_f_b  = create_tensor(tn(LLM_TENSOR_SSM_F_B,  "weight", i), {head_dim, d_inner}, 0);
            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd, n_head}, 0);

            layer.ssm_a    = create_tensor(tn(LLM_TENSOR_SSM_A, i), {n_head}, 0);
            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner}, 0);

            layer.ssm_g_a    = create_tensor(tn(LLM_TENSOR_SSM_G_A,  "weight", i), {n_embd, head_dim}, 0);
            layer.ssm_g_b    = create_tensor(tn(LLM_TENSOR_SSM_G_B,  "weight", i), {head_dim, d_inner}, 0);
            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {head_dim}, 0);
            layer.wo         = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {d_inner, n_embd}, 0);
        } else {
            const int64_t q_lora_rank      = hparams.n_lora_q;
            const int64_t kv_lora_rank     = hparams.n_lora_kv;
            const int64_t n_embd_head_k    = hparams.n_embd_head_k_mla();
            const int64_t n_embd_head_v    = hparams.n_embd_head_v_mla();
            const int64_t qk_rope_head_dim = hparams.n_rot();
            const int64_t qk_nope_head_dim = n_embd_head_k - qk_rope_head_dim;

            layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, flags);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);

            layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, flags);
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k}, flags);

            layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + qk_rope_head_dim}, flags);
            layer.wk_b      = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {qk_nope_head_dim, kv_lora_rank, n_head}, flags);
            layer.wv_b      = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v, n_head}, flags);
            layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v, n_embd}, flags);

            const int64_t n_indexer_head = hparams.indexer_n_head;
            const int64_t n_embd_indexer = hparams.indexer_head_size;
            const int64_t kpool          = hparams.indexer_kpool;

            const bool full = i >= n_layer || hparams.is_indexer_full(i);
            const int  iflags = flags | (full ? 0 : TENSOR_NOT_REQUIRED);

            layer.indexer_k_norm     = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,     "weight", i), {n_embd_indexer}, iflags);
            layer.indexer_k_norm_b   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,     "bias",   i), {n_embd_indexer}, iflags);
            layer.indexer_proj       = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,       "weight", i), {n_embd, n_indexer_head}, iflags);
            layer.indexer_attn_k     = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,     "weight", i), {n_embd, n_embd_indexer}, iflags);
            layer.indexer_attn_q_b   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B,   "weight", i), {q_lora_rank, n_indexer_head * n_embd_indexer}, iflags);
            layer.indexer_kpool_gate = create_tensor(tn(LLM_TENSOR_INDEXER_KPOOL_GATE, "weight", i), {n_embd, n_embd_indexer}, iflags);
            layer.indexer_kpool_ape  = create_tensor(tn(LLM_TENSOR_INDEXER_KPOOL_APE,  "weight", i), {n_embd_indexer, kpool}, iflags);
        }

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        } else {
            const int64_t n_ff_exp        = hparams.n_ff_exp;
            const int64_t n_expert_shared = hparams.n_expert_shared;

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, flags);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
        }

        if (i >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab}, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), {n_embd, n_vocab}, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, flags | TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_glm5_next::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// Mean over the hyper-connection streams
static ggml_tensor * glm5_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx, acc, 1.0f/hc);
}

// Causal conv1d over one of Q/K/V
static ggml_tensor * glm5_conv1d(ggml_cgraph * gf, ggml_context * ctx0,
                                 ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
                                 int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
                                 int64_t d_conv, int64_t head_dim, int64_t n_head,
                                 int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head) {
    const int64_t d_inner         = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total  = 3 * conv_state_size;

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1)   * ggml_element_size(conv_state_all),
        n_embd_r_total * ggml_element_size(conv_state_all),
        qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
    ggml_tensor * x_3d   = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);
    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    ggml_tensor * last_conv_x = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
        conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, last_conv_x,
            ggml_view_3d(ctx0, conv_states_all, d_conv - 1, d_inner, n_seqs,
                (d_conv - 1)   * ggml_element_size(conv_states_all),
                n_embd_r_total * ggml_element_size(conv_states_all),
                (kv_head * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);
    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}


// K-pool indexer inputs
class llama_model_glm5_next::llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(const llama_memory_hybrid_idx_context * mctx, uint32_t kpool) : mctx(mctx), kpool(kpool) {}
    virtual ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_kpool(pool_idxs, pool_mask, tail_idxs, cell_pool, ubatch, kpool);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        bool res = true;

        res &= k_idxs->ne[0]    == params.ubatch.n_tokens;
        res &= pool_idxs->ne[1] == mctx->get_n_kpool(kpool);
        res &= pool_mask->ne[1] == params.ubatch.n_tokens;
        res &= tail_idxs->ne[1] == params.ubatch.n_tokens;
        res &= cell_pool->ne[0] == idx->get_n_kv();

        return res;
    }

    ggml_tensor * k_idxs    = nullptr; // I64     [n_tokens]
    ggml_tensor * pool_idxs = nullptr; // I32     [kpool, n_pool]
    ggml_tensor * pool_mask = nullptr; // F32/F16 [n_pool, n_tokens]
    ggml_tensor * tail_idxs = nullptr; // I32     [kpool - 1, n_tokens]
    ggml_tensor * cell_pool = nullptr; // I32     [n_kv]

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t kpool;
};

llama_model_glm5_next::llm_graph_input_kpool * llama_model_glm5_next::graph::build_inp_kpool(const llama_memory_hybrid_idx_context * mctx_hyb) {
    const auto * mctx_idx = mctx_hyb->get_idx();
    GGML_ASSERT(mctx_idx != nullptr);

    const uint32_t kpool  = hparams.indexer_kpool;
    const uint32_t n_pool = mctx_hyb->get_n_kpool(kpool);
    const uint32_t n_kv   = mctx_idx->get_n_kv();

    // the fused lightning indexer wants an f16 mask
    const auto type_mask = cparams.fused_lid ? GGML_TYPE_F16 : GGML_TYPE_F32;

    auto inp = std::make_unique<llm_graph_input_kpool>(mctx_hyb, kpool);

    inp->k_idxs    = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    inp->pool_idxs = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, n_pool);
    inp->pool_mask = ggml_new_tensor_2d(ctx0, type_mask, n_pool, n_tokens);
    inp->tail_idxs = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool - 1, n_tokens);
    inp->cell_pool = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_kv);
    ggml_set_input(inp->pool_idxs);
    ggml_set_input(inp->pool_mask);
    ggml_set_input(inp->tail_idxs);
    ggml_set_input(inp->cell_pool);

    return (llm_graph_input_kpool *) res->add_input(std::move(inp));
}

llama_model_glm5_next::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {

    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    cb(inp, "inp_embd", -1);

    // recurrent state + K-only MLA cache through the generic hybrid input, plus the indexer cache
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(mctx);

    auto * inp_hyb   = build_inp_mem_hybrid_k();
    auto * inp_rs    = inp_hyb->get_recr();
    auto * inp_attn  = inp_hyb->get_attn();
    auto * inp_kpool = build_inp_kpool(mctx_hyb);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t n_head_kda   = hparams.n_head();
    const int64_t head_dim     = hparams.n_embd_head_kda;
    const int64_t d_conv       = hparams.ssm_d_conv;
    const int64_t d_inner      = n_head_kda * head_dim;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    ggml_tensor * prev_sel = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_hc_pre(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_kda_layer(cur, layer, inp_rs, d_conv, head_dim, n_head_kda,
                                  d_inner, n_seq_tokens, n_seqs, il);
        } else {
            cur = build_dsa_layer(cur, layer, mctx_hyb, inp_attn, inp_kpool, &prev_sel, il);
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp,   nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    // narrow to the output tokens, then collapse the streams
    // Unmasked nextn embeddings need all rows.
    const bool narrow_early = inp_out_ids && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked);
    if (narrow_early) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    cur = glm5_hc_mean(ctx0, inpL);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    // the post-norm hidden state feeds the draft head
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (inp_out_ids && !narrow_early) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// KDA layer, g_a/g_b output gate

ggml_tensor * llama_model_glm5_next::graph::build_kda_layer(
        ggml_tensor * cur, const llama_layer & layer, llm_graph_input_rs * inp_rs,
        int64_t d_conv, int64_t head_dim, int64_t n_head_kda,
        int64_t d_inner, int64_t n_seq_tokens, int64_t n_seqs, int il) {

    const auto * mctx_cur = inp_rs->mctx;
    const auto   kv_head  = mctx_cur->get_head();

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * conv_state_all  = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);

    ggml_tensor * Qcur = glm5_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head);
    ggml_tensor * Kcur = glm5_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head);
    ggml_tensor * Vcur = glm5_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head);
    cb(Qcur, "kda_q_conv", il);
    cb(Kcur, "kda_k_conv", il);
    cb(Vcur, "kda_v_conv", il);

    // Decay gate, ssm_a holds -exp(A_log)
    ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
    ggml_tensor * g1  = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
    g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);

    ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head_kda, 1);

    if (hparams.kda_gate_lower_bound > -INFINITY) {
        g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head_kda, n_tokens);
        g1 = ggml_mul(ctx0, g1, A);
        g1 = ggml_sigmoid(ctx0, ggml_scale(ctx0, g1, -1.0f));
        g1 = ggml_scale(ctx0, g1, hparams.kda_gate_lower_bound);
    } else {
        g1 = ggml_softplus(ctx0, g1);
        g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head_kda, n_tokens);
        g1 = ggml_mul(ctx0, g1, A);
    }
    cb(g1, "kda_g1", il);

    g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head_kda, n_seq_tokens, n_seqs);

    ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
    beta = ggml_reshape_4d(ctx0, beta, 1, n_head_kda, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "kda_beta", il);

    ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head_kda, n_seqs);

    const float eps = hparams.f_norm_rms_eps;
    Qcur = ggml_l2_norm(ctx0, Qcur, eps);
    Kcur = ggml_l2_norm(ctx0, Kcur, eps);

    auto attn_out = build_delta_net(Qcur, Kcur, Vcur, g1, beta, state, il);

    ggml_tensor * output    = ggml_cont(ctx0, attn_out.first);
    ggml_tensor * new_state = attn_out.second;
    cb(output, "kda_scan_out", il);

    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, new_state,
            ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                         kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

    // output gate, then RMSNorm(o) * Sigmoid(g2)
    ggml_tensor * g_a = ggml_mul_mat(ctx0, layer.ssm_g_a, cur);
    ggml_tensor * g2  = ggml_mul_mat(ctx0, layer.ssm_g_b, g_a);
    g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head_kda, n_tokens);

    ggml_tensor * o      = ggml_reshape_3d(ctx0, output, head_dim, n_head_kda, n_tokens);
    ggml_tensor * normed = build_norm(o, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
    cb(g2, "kda_g2", il);
    cb(normed, "kda_normed", il);
    ggml_tensor * gated = ggml_mul(ctx0, normed, ggml_sigmoid(ctx0, g2));

    gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
    cur   = ggml_mul_mat(ctx0, layer.wo, gated);
    cb(cur, "kda_out", il);

    return cur;
}

// Scores pools of kpool consecutive tokens, expands the selected pools and the incomplete tail into an additive mask

ggml_tensor * llama_model_glm5_next::graph::build_kpool_select(
        ggml_tensor * cur, ggml_tensor * qr, const llama_layer & layer,
        const llama_memory_hybrid_idx_context * mctx_hyb, llm_graph_input_kpool * inp_kpool, int il) {

    const auto * mctx_lid = mctx_hyb->get_idx();

    const int64_t n_indexer_head = hparams.indexer_n_head;
    const int64_t n_embd_indexer = hparams.indexer_head_size;
    const int64_t kpool          = hparams.indexer_kpool;
    const int64_t n_pool         = inp_kpool->pool_idxs->ne[1];

    // queries
    ggml_tensor * iq = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, qr);
    iq = ggml_reshape_3d(ctx0, iq, n_embd_indexer, n_indexer_head, n_tokens);
    cb(iq, "indexer_q", il);

    // Per-token key and pool gate scores, cached together
    ggml_tensor * ik = ggml_mul_mat(ctx0, layer.indexer_attn_k, cur);
    ik = build_norm(ik, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, il);
    cb(ik, "indexer_k", il);

    ggml_tensor * ig = ggml_mul_mat(ctx0, layer.indexer_kpool_gate, cur);
    cb(ig, "indexer_gate", il);

    ggml_tensor * packed = ggml_concat(ctx0, ik, ig, 0);
    packed = ggml_reshape_3d(ctx0, packed, 2*n_embd_indexer, 1, n_tokens);
    ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, packed, inp_kpool->k_idxs, il));

    ggml_tensor * k_all = mctx_lid->get_k(ctx0, il);
    GGML_ASSERT(k_all->ne[3] == 1 && "TODO: k-pool indexer with multiple streams");
    const int64_t n_kv = k_all->ne[2];
    k_all = ggml_view_2d(ctx0, k_all, 2*n_embd_indexer, n_kv, k_all->nb[2], 0);

    // Gather the member of every pool
    ggml_tensor * rows = ggml_get_rows(ctx0, k_all, ggml_reshape_1d(ctx0, inp_kpool->pool_idxs, kpool*n_pool));
    rows = ggml_reshape_3d(ctx0, rows, 2*n_embd_indexer, kpool, n_pool);

    ggml_tensor * pk = ggml_view_3d(ctx0, rows, n_embd_indexer, kpool, n_pool, rows->nb[1], rows->nb[2], 0);
    ggml_tensor * pg = ggml_view_3d(ctx0, rows, n_embd_indexer, kpool, n_pool, rows->nb[1], rows->nb[2], ggml_row_size(rows->type, n_embd_indexer));

    ggml_tensor * logits = ggml_add(ctx0, pg, layer.indexer_kpool_ape);
    logits = ggml_cont(ctx0, ggml_permute(ctx0, logits, 1, 0, 2, 3)); // [kpool, head_dim, n_pool]
    ggml_tensor * probs = ggml_soft_max(ctx0, logits);

    pk = ggml_cont(ctx0, ggml_permute(ctx0, pk, 1, 0, 2, 3));
    ggml_tensor * pooled = ggml_sum_rows(ctx0, ggml_mul(ctx0, probs, pk)); // [1, head_dim, n_pool]
    pooled = ggml_reshape_3d(ctx0, pooled, n_embd_indexer, 1, n_pool);
    cb(pooled, "indexer_pool_k", il);

    ggml_tensor * weights = ggml_mul_mat(ctx0, layer.indexer_proj, cur);
    weights = ggml_scale(ctx0, weights, 1.0f / sqrtf(float(n_embd_indexer * n_indexer_head)));
    cb(weights, "indexer_weights", il);

    ggml_tensor * score = nullptr;
    if (cparams.fused_lid) {
        score = ggml_lightning_indexer(ctx0, iq, pooled, weights, inp_kpool->pool_mask);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
    } else {
        ggml_tensor * q_p = ggml_permute(ctx0, iq, 0, 2, 1, 3);     // [head_dim, n_tokens, n_head]
        ggml_tensor * k_p = ggml_permute(ctx0, pooled, 0, 2, 1, 3); // [head_dim, n_pool, 1]

        ggml_tensor * kq = ggml_mul_mat(ctx0, k_p, q_p);            // [n_pool, n_tokens, n_head]
        kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));   // [n_head, n_tokens, n_pool]
        score = ggml_relu(ctx0, kq);
        score = ggml_mul(ctx0, score, weights);
        score = ggml_sum_rows(ctx0, score);                          // [1, n_tokens, n_pool]
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3)); // [n_pool, n_tokens, 1]
        score = ggml_add(ctx0, score, inp_kpool->pool_mask);
    }
    cb(score, "indexer_score", il);

    const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool);
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, score, n_top_pool)); // [n_top_pool, n_tokens]
    cb(top_k, "indexer_top_k", il);

    // Pool-level selection mask, -inf everywhere except the selected pools
    ggml_tensor * pool_sel = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_pool, n_tokens);
    pool_sel = ggml_fill(ctx0, pool_sel, -INFINITY);
    ggml_tensor * zeros = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_top_pool, n_tokens), 0.0f);
    pool_sel = ggml_set_rows(ctx0, pool_sel, zeros, ggml_reshape_3d(ctx0, top_k, n_top_pool, n_tokens, 1));
    pool_sel = ggml_reshape_2d(ctx0, pool_sel, n_pool, n_tokens);

    // Expand to the cells
    ggml_tensor * pool_sel_t = ggml_cont(ctx0, ggml_transpose(ctx0, pool_sel));            // [n_tokens, n_pool]
    ggml_tensor * sel_t      = ggml_get_rows(ctx0, pool_sel_t, inp_kpool->cell_pool);          // [n_tokens, n_kv]
    ggml_tensor * sel        = ggml_cont(ctx0, ggml_transpose(ctx0, sel_t));                 // [n_kv, n_tokens]

    if (hparams.indexer_kpool_select_tail) {
        ggml_tensor * pad = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens), -INFINITY);
        sel = ggml_concat(ctx0, sel, pad, 0);                                                // [n_kv + 1, n_tokens]
        sel = ggml_reshape_3d(ctx0, sel, 1, n_kv + 1, n_tokens);
        ggml_tensor * tzeros = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, kpool - 1, n_tokens), 0.0f);
        sel = ggml_set_rows(ctx0, sel, tzeros, ggml_reshape_3d(ctx0, inp_kpool->tail_idxs, kpool - 1, n_tokens, 1));
        sel = ggml_view_2d(ctx0, sel, n_kv, n_tokens, sel->nb[2], 0);
    }
    cb(sel, "indexer_sel", il);

    return sel;
}

// Nope MLA layer with sparse attention over the indexer selection

ggml_tensor * llama_model_glm5_next::graph::build_dsa_layer(
        ggml_tensor * cur, const llama_layer & layer,
        const llama_memory_hybrid_idx_context * mctx_hyb, llm_graph_input_attn_k * inp_attn,
        llm_graph_input_kpool * inp_kpool, ggml_tensor ** prev_sel, int il) {

    const auto * mctx_mla = mctx_hyb->get_attn();

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t kv_lora_rank      = hparams.n_lora_kv;
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - hparams.n_rot();
    const float   kq_scale = 1.0f / sqrtf((float) n_embd_head_k_mla);

    GGML_ASSERT(hparams.n_rot() == 0 && "GLM5-Next MLA is nope-only");

    ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "q_resid", il);

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens);

    ggml_tensor * kv_cmpr = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    cb(kv_cmpr, "kv_cmpr", il);

    // absorb wk_b so the cache holds only the latent
    ggml_tensor * q_absorbed = ggml_permute(ctx0, q, 0, 2, 1, 3);
    q_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_absorbed);
    q_absorbed = ggml_permute(ctx0, q_absorbed, 0, 2, 1, 3);
    cb(q_absorbed, "q_absorbed", il);

    ggml_tensor * sel = nullptr;
    if (il >= (int) hparams.n_layer() || hparams.is_indexer_full(il)) { // the NextN block always has a full indexer
        sel = build_kpool_select(cur, qr, layer, mctx_hyb, inp_kpool, il);
        *prev_sel = sel;
    } else {
        GGML_ASSERT(*prev_sel != nullptr && "shared indexer layer must follow a full indexer layer");
        sel = *prev_sel;
    }

    ggml_build_forward_expand(gf, q_absorbed);
    ggml_build_forward_expand(gf, kv_cmpr);
    ggml_build_forward_expand(gf, mctx_mla->cpy_k(ctx0, kv_cmpr, inp_attn->get_k_idxs(), il));

    // Combine the causal mask with the indexer selection
    ggml_tensor * kq_mask = inp_attn->get_kq_mask();
    ggml_tensor * mask = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
    mask = ggml_add(ctx0, ggml_reshape_2d(ctx0, mask, mask->ne[0], mask->ne[1]), sel);
    if (kq_mask->type != GGML_TYPE_F32) {
        mask = ggml_cast(ctx0, mask, kq_mask->type);
    }
    mask = ggml_reshape_4d(ctx0, mask, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);
    cb(mask, "kq_mask_dsa", il);

    ggml_tensor * k = mctx_mla->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, kv_lora_rank, k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * out = build_attn_mha(q_absorbed, k, v, nullptr, mask, nullptr, layer.wv_b, kq_scale, il);
    cb(out, "kqv_out", il);

    out = ggml_mul_mat(ctx0, layer.wo, out);
    cb(out, "attn_out", il);

    return out;
}

// Nextn draft head. The Nextn block is a DSA layer.

llama_model_glm5_next::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : graph(model, params, no_trunk_t{}) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "GLM5-Next MTP supports a single NextN block");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset == 0);
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm && "MTP block tensors missing, convert without --no-mtp");

    // token and previous hidden state inputs
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");
    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // K-only MLA cache plus the indexer cache, no recurrent layers here
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(mctx);

    auto * inp_hyb   = build_inp_mem_hybrid_k();
    auto * inp_attn  = inp_hyb->get_attn();
    auto * inp_kpool = build_inp_kpool(mctx_hyb);

    ggml_build_forward_expand(gf, inp_hyb->get_recr()->s_copy);

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * cur = ggml_mul_mat(ctx0, layer.nextn.eh_proj, ggml_concat(ctx0, e_norm, h_norm, 0));
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    ggml_tensor * prev_sel = nullptr;
    cur = build_dsa_layer(cur, layer, mctx_hyb, inp_attn, inp_kpool, &prev_sel, il);
    cb(cur, "mtp_attn_out", il);

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il);
    cb(moe_out, "mtp_ffn_moe_out", il);

    ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp,   nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = ggml_add(ctx0, ggml_add(ctx0, moe_out, ffn_shexp), ffn_inp);
    cb(cur, "mtp_post_ffn", il);

    // shared_head.norm, then the post-norm hidden state.
    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm ? layer.nextn.shared_head_norm : model.output_norm;
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    cur = ggml_mul_mat(ctx0, head_w, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
