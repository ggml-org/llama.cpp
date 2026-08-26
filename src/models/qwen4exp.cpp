#include "models.h"
#include "llama-impl.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cmath>
#include <vector>

// Qwen4-Exp: Qwen3.5-MoE trunk (gated delta-net + gated full attention + MoE) on top of
// hyper-connections, plus a hashed n-gram per-layer embedding (PLE) on selected linear layers.
//
// The full-attention layers use QSA: an indexer scores blocks of `indexer_block_size` keys and
// the attention reads only the top `indexer_top_k`/`indexer_block_size` blocks plus the trailing
// incomplete block. Blocks are anchored to cache cells - see llm_graph_input_qsa.

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // linear attention (gated delta net)
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,   hparams.hc_count);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOWRANK, hparams.hc_lowrank);
    GGML_ASSERT(hparams.hc_count > 1);

    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);

    // the MTP block reads the hyper-connection streams, not the collapsed hidden state, so the
    // nextn read-back and common/speculative.cpp move hc_count*n_embd per token
    hparams.n_embd_out_impl = hparams.hc_count*hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head,     false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size,   false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k,       false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE, hparams.indexer_block_size,  false);

    // per-layer embedding (PLE)
    hparams.n_embd_ple = hparams.n_embd;
    ml.get_key(LLM_KV_PLE_EMBD_LENGTH, hparams.n_embd_ple, false);
    ml.get_key(LLM_KV_PLE_CONV_KERNEL, hparams.ple_conv_kernel, false);
    // the n-gram hash constants are needed on the host, they come in as metadata
    auto load_u64 = [&ml](llm_kv kid, auto & dst) -> uint32_t {
        std::vector<uint64_t> tmp;
        if (!ml.get_arr(kid, tmp, false)) {
            return 0;
        }
        GGML_ASSERT(tmp.size() <= dst.size());
        std::copy(tmp.begin(), tmp.end(), dst.begin());
        return tmp.size();
    };

    hparams.ple_ngram_size  = load_u64(LLM_KV_PLE_NGRAM_MULTIPLIERS, hparams.ple_ngram_mult);
    hparams.ple_ngram_heads = load_u64(LLM_KV_PLE_NGRAM_VOCAB_SIZES, hparams.ple_ngram_vocab);
    GGML_ASSERT(hparams.ple_ngram_heads == load_u64(LLM_KV_PLE_NGRAM_OFFSETS, hparams.ple_ngram_offs));

    if (hparams.ple_ngram_size > 0) {
        // the hash spreads the heads over the 2..n_ngram grams and gives each head one slice of
        // the PLE row. Without these the head loop divides by zero or truncates
        GGML_ASSERT(hparams.ple_ngram_size > 1 && "PLE needs at least 2-grams");
        GGML_ASSERT(hparams.ple_ngram_heads > 0 && "PLE needs at least one n-gram head");
        GGML_ASSERT(hparams.ple_ngram_heads % (hparams.ple_ngram_size - 1) == 0 &&
                "PLE n-gram heads must be a multiple of ple_ngram_size - 1");
        GGML_ASSERT(hparams.n_embd_ple % hparams.ple_ngram_heads == 0 &&
                "PLE embedding length must be a multiple of the n-gram heads");
    }

    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i + 1) % full_attn_interval != 0;
        }
    }

    // the MTP block is a sparse-attention layer, but the interval above covers n_layer_all and
    // would call it recurrent. Nothing reads it there today, so say the truth before it does
    for (uint32_t i = hparams.n_layer(); i < hparams.n_layer_all; ++i) {
        hparams.is_recr_impl[i] = false;
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_122B_A10B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc      = hparams.hc_count;
    const int64_t hc_dim  = hc * n_embd;
    const int64_t n_ngram = hparams.ple_ngram_size;
    const int64_t n_head_ngram = hparams.ple_ngram_heads;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // the hyper-connection mixer replaces the usual output norm
    hc_mix_norm = create_tensor(tn(LLM_TENSOR_HC_MIX_NORM, "weight"), { hc_dim }, 0);
    hc_mix_down = create_tensor(tn(LLM_TENSOR_HC_MIX_DOWN, "weight"), { hc_dim, hparams.hc_lowrank }, 0);
    hc_mix_up   = create_tensor(tn(LLM_TENSOR_HC_MIX_UP,   "weight"), { hparams.hc_lowrank, hc_dim }, 0);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    uint32_t n_embd_r_ple = 0;  // PLE conv taps  -> the second recurrent state's r
    uint32_t n_embd_s_ple = 0;  // n-gram history -> its s

    // an MTP-only file carries just the MTP block, see conversion --mtp
    const bool mtp_only = hparams.n_layer_nextn > 0 && ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr;

    for (int il = 0; il < (mtp_only ? 0 : n_layer); ++il) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp   ? hparams.n_ff_exp   : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { hc_dim }, 0);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hparams.hc_lowrank }, 0);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hparams.hc_lowrank, hc_dim }, 0);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, 0);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { hc_dim }, 0);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hparams.hc_lowrank }, 0);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hparams.hc_lowrank, hc_dim }, 0);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, 0);

        if (!hparams.is_recr(il)) {
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, 0);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, 0);

            // QSA indexer, skipped when the file has no indexer at all
            const int64_t n_embd_idx = hparams.indexer_head_size;
            const int      idx_flags = hparams.has_qsa() ? 0 : (TENSOR_NOT_REQUIRED | TENSOR_SKIP);
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, n_embd_idx * hparams.indexer_n_head }, idx_flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, n_embd_idx }, idx_flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { n_embd_idx }, idx_flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { n_embd_idx }, idx_flags);
        } else {
            layer.wqkv      = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, conv_dim }, 0);
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, 0);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, 0);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { n_v_heads }, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,          il), { n_v_heads }, 0);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, 0);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, 0);
        }

        if (n_ngram > 0) {
            const int64_t n_embd_ngram = hparams.n_embd_ple / n_head_ngram;

            // the hash reads row offs[h] + 0..vocab[h]-1 per head, so the table has to be at
            // least this tall
            int64_t n_row_ngram = 0;
            for (int64_t h = 0; h < n_head_ngram; ++h) {
                n_row_ngram = std::max(n_row_ngram,
                        (int64_t) (hparams.ple_ngram_offs[h] + hparams.ple_ngram_vocab[h]));
            }

            // the checkpoint pads the table and does not export the divisor, so read the real
            // height off the file instead of assuming one
            const auto * w_ngram = ml.get_weight(tn(LLM_TENSOR_PLE_NGRAM_EMBD, "weight", il).str().c_str());
            if (w_ngram) {
                GGML_ASSERT(w_ngram->tensor->ne[1] >= n_row_ngram &&
                        "PLE n-gram table is shorter than the hash offsets need");
                n_row_ngram = w_ngram->tensor->ne[1];
            }

            layer.ple_ngram_embd = create_tensor(tn(LLM_TENSOR_PLE_NGRAM_EMBD, "weight", il), { n_embd_ngram, n_row_ngram }, TENSOR_NOT_REQUIRED);

            if (layer.ple_ngram_embd) {
                layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { hparams.n_embd_ple, hc_dim }, 0);
                layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { hparams.n_embd_ple, n_embd }, 0);
                layer.ple_key_norm   = create_tensor(tn(LLM_TENSOR_PLE_KEY_NORM,   "weight", il), { hc_dim }, 0);
                layer.ple_query_norm = create_tensor(tn(LLM_TENSOR_PLE_QUERY_NORM, "weight", il), { hc_dim }, 0);
                layer.ple_conv_norm  = create_tensor(tn(LLM_TENSOR_PLE_CONV_NORM,  "weight", il), { hc_dim }, 0);
                layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, 0);

                GGML_ASSERT(hparams.is_recr(il) && "PLE is only supported on linear attention layers");
                n_embd_r_ple = (hparams.ple_conv_kernel - 1) * n_ngram * hc_dim;
                n_embd_s_ple = n_ngram - 1;

                GGML_ASSERT(hparams.il_2nd < 0 && "only one PLE layer is supported");
                hparams.il_2nd = il;
            }
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, 0);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, 0);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, 0);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, 0);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, 0);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, 0);
    }

    // the MTP block is one sparse-attention layer with its own input fusion. It has no PLE and no
    // linear attention, so it is built here instead of in the loop above.
    for (int il = n_layer; il < n_layer + (int) hparams.n_layer_nextn; ++il) {
        auto & layer = layers[il];

        const int flags = ml.load_mtp ? 0 : TENSOR_SKIP;

        const int64_t n_ff_exp   = hparams.n_ff_exp   ? hparams.n_ff_exp   : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { hc_dim }, flags);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hparams.hc_lowrank }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hparams.hc_lowrank, hc_dim }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { hc_dim }, flags);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hparams.hc_lowrank }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hparams.hc_lowrank, hc_dim }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

        // the MTP attention runs dense, so the indexer tensors are loaded but unused
        const int64_t n_embd_idx = hparams.indexer_head_size;
        const int      idx_flags = flags | TENSOR_NOT_REQUIRED | TENSOR_SKIP;
        layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, n_embd_idx * hparams.indexer_n_head }, idx_flags);
        layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, n_embd_idx }, idx_flags);
        layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { n_embd_idx }, idx_flags);
        layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { n_embd_idx }, idx_flags);

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);

        // hnorm is over the whole hyper-connection row, not per stream
        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), { n_embd }, flags);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), { hc_dim }, flags);
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);
    }

    // PLE keeps a state of its own, so the delta-net row stays exactly the delta-net conv state
    hparams.n_embd_r_2nd = n_embd_r_ple;
    hparams.n_embd_s_2nd = n_embd_s_ple;
}


// A second recurrent state needs its own graph input. The plain llm_graph_input_rs cannot be
// used as a top-level input here: its can_reuse() casts params.mctx straight to a recurrent
// context, which is wrong when the context is hybrid - on a reused graph it would rebind to
// garbage. Rebind from the hybrid's second state instead. ref: llm_graph_input_mem_hybrid
class llm_graph_input_rs_2nd : public llm_graph_input_rs {
public:
    using llm_graph_input_rs::llm_graph_input_rs;
    virtual ~llm_graph_input_rs_2nd() = default;

    bool can_reuse(const llm_graph_params & params) override {
        const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(params.mctx)->get_ple();

        mctx = mctx_cur;

        bool res = true;

        res &= s_copy->ne[0]       == mctx_cur->get_n_rs();
        res &= s_copy_main->ne[0]  == params.ubatch.n_seqs;
        res &= s_copy_extra->ne[0] == mctx_cur->get_n_rs() - params.ubatch.n_seqs;

        res &= head == mctx_cur->get_head();
        res &= rs_z == mctx_cur->get_rs_z();

        return res;
    }
};

// QSA: the indexer scores blocks of `indexer_block_size` consecutive *positions*, so the host says
// which blocks each query may see and where every position lives in the caches. Selecting in
// position space keeps this correct whatever cells find_slot handed out.
class llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_context * mctx, int64_t ratio, uint32_t pad) :
        mctx(mctx), ratio(ratio), pad(pad) {}
    virtual ~llm_graph_input_qsa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

        const int64_t ns = params.cparams.kv_unified ? 1 : params.ubatch.n_seqs_unq;

        n_kv_lid = mctx->get_lid()->get_n_kv();

        if (pos_cell_f && (pos_cell_f->ne[0] != (int64_t) mctx->get_n_pos(pad, params.cparams.n_seq_max) ||
                           pos_cell_f->ne[1] != ns)) {
            return false;
        }

        return pos_cell->ne[0] == (int64_t) mctx->get_n_pos(pad, params.cparams.n_seq_max) &&
               pos_cell->ne[1] == ns &&
               (!cell_blk || cell_blk->ne[0] == (int64_t) mctx->get_attn()->get_n_kv()) &&
               blk_mask->ne[1] == (int64_t) params.ubatch.n_tokens/ns &&
               blk_mask->ne[3] == ns;
    }

    ggml_tensor * k_idxs   = nullptr; // I64 [n_tokens]            destination cells of this ubatch
    ggml_tensor * pos_cell = nullptr; // I32 [n_pos, ns]           position -> indexer cell
    ggml_tensor * cell_blk = nullptr; // I32 [n_kv, ns]            attention cell -> position block
    ggml_tensor * blk_mask = nullptr; // F16 [n_blk, n_tps, 1, ns] 0 / -inf, block complete and visible
    ggml_tensor * blk_cur  = nullptr; // I32 [1, n_tps, 1, ns]     the query's own block, always read

    // only for the decode gather, see build_qsa_gather()
    ggml_tensor * pos_cell_f = nullptr; // F32 [n_pos, ns] position -> attention cell, as values
    ggml_tensor * pos_vis    = nullptr; // F32 [n_pos, ns] 0 / -inf, the query may read this position

    const llama_memory_hybrid_context * mctx;
    const int64_t ratio;
    const uint32_t pad;

    // rows of the indexer cache the graph views, see set_input_pos_cell()
    int64_t n_kv_lid = 0;
};

void llm_graph_input_qsa::set_input(const llama_ubatch * ubatch) {
    mctx->get_lid()->set_input_k_idxs(k_idxs, ubatch);
    mctx->set_input_pos_cell(pos_cell, ubatch, true, n_kv_lid);
    if (cell_blk) {
        mctx->set_input_cell_blk(cell_blk, ubatch, ratio);
    }

    const int64_t n_blk    = blk_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    std::vector<ggml_fp16_t> mask(n_blk*n_tokens, ggml_fp32_to_fp16(-INFINITY));
    std::vector<int32_t>     cur (n_tokens, 0);

    for (int64_t i = 0; i < n_tokens; ++i) {
        const int64_t n_vis  = ubatch->pos[i] + 1;
        const int64_t n_full = n_vis/ratio;

        // only complete blocks compete in the top-k, as in the reference
        for (int64_t b = 0; b < std::min(n_full, n_blk); ++b) {
            mask[i*n_blk + b] = ggml_fp32_to_fp16(0.0f);
        }

        // the trailing incomplete block is always read on top of the top-k. Opening all of it is
        // fine - the cells past the query are closed again by the causal mask. When the last block
        // is complete there is no tail, so point at the last block instead: no cell maps to it, so
        // opening it does nothing.
        cur[i] = (int32_t) (n_vis%ratio != 0 ? std::min(n_full, n_blk - 1) : n_blk - 1);
    }

    ggml_backend_tensor_set(blk_mask, mask.data(), 0, ggml_nbytes(blk_mask));
    ggml_backend_tensor_set(blk_cur,  cur.data(),  0, ggml_nbytes(blk_cur));

    if (pos_cell_f) {
        // the gather reads whole blocks, so the positions past the query have to be closed here.
        // Unmapped positions default to cell 0, which holds real but unrelated keys - they are only
        // ever past the query, so the same test covers both
        mctx->set_input_pos_cell(pos_cell_f, ubatch, false, mctx->get_attn()->get_n_kv());

        const int64_t n_pos = pos_vis->ne[0];
        const int64_t ns    = pos_vis->ne[1];

        std::vector<float> vis(n_pos*ns, -INFINITY);

        for (int64_t s = 0; s < ns; ++s) {
            const llama_pos p1 = ubatch->pos[s*(n_tokens/ns) + n_tokens/ns - 1];

            for (int64_t p = 0; p <= p1 && p < n_pos; ++p) {
                vis[s*n_pos + p] = 0.0f;
            }
        }

        ggml_backend_tensor_set(pos_vis, vis.data(), 0, ggml_nbytes(pos_vis));
    }

}

// Computes the n-gram embedding rows on the host: the hash needs 64-bit integer math, and the
// previous tokens of a sequence come from the tail of its recurrent state row.
class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_hparams & hparams, const llama_memory_recurrent_context * mctx,
            llama_token tok_eos, int il) :
        hparams(hparams), mctx(mctx), tok_eos(tok_eos), il(il) {}
    virtual ~llm_graph_input_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx)->get_ple();

        return ngram_ids->ne[0] == (int64_t) params.ubatch.n_tokens*hparams.ple_ngram_heads;
    }

    ggml_tensor * ngram_ids = nullptr;  // I32 [n_head_ngram*n_tokens] rows of the n-gram table
    ggml_tensor * tokens    = nullptr;  // F32 [1, n_tokens] token ids kept as the next history

    const llama_hparams & hparams;
    const llama_memory_recurrent_context * mctx;
    const llama_token tok_eos;
    const int         il;
};

void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    // media chunks arrive as embeddings without token ids, so eos is hashed there. The reference
    // hashes the media placeholder token instead - gemma4 has the same gap in build_inp_per_layer.
    // TODO: needs the placeholder id to reach the ubatch, see llm_graph_context::build_inp_embd
    auto token_at = [ubatch, this](int64_t i) {
        return ubatch->token ? ubatch->token[i] : tok_eos;
    };

    const int64_t n_ngram      = hparams.ple_ngram_size;
    const int64_t n_head_ngram = hparams.ple_ngram_heads;
    const int64_t n_per_ngram  = n_head_ngram / (n_ngram - 1);
    const int64_t n_seqs       = ubatch->n_seqs;
    const int64_t n_seq_tokens = ubatch->n_seq_tokens;

    ggml_tensor * s_l = mctx->get_s_l(il);

    std::vector<int32_t> ids(n_head_ngram*ubatch->n_tokens);
    std::vector<float>   toks(ubatch->n_tokens);

    // history of the (n_ngram - 1) tokens before this ubatch, eos where a sequence starts
    std::vector<llama_token> hist(n_ngram - 1, tok_eos);
    std::vector<float>       hist_f(n_ngram - 1);

    std::vector<llama_token> shifted(n_ngram);

    for (int64_t s = 0; s < n_seqs; ++s) {
        std::fill(hist.begin(), hist.end(), tok_eos);

        // how many tokens the state row already holds. This reads pos as the history length,
        // which context shift breaks: it renumbers pos and leaves the state alone. Context shift
        // is off by default and mmproj forces it off, so it cannot happen today
        const int64_t n_prev = ubatch->pos[s*n_seq_tokens];

        if (n_prev > 0) {
            const size_t row = mctx->s_copy(s);
            ggml_backend_tensor_get(s_l, hist_f.data(),
                    row*hist_f.size()*sizeof(float), hist_f.size()*sizeof(float));
            // the state row starts as zeros, so only the last n_prev entries hold tokens
            for (int64_t k = 0; k < std::min(n_ngram - 1, n_prev); ++k) {
                hist[k] = (llama_token) hist_f[n_ngram - 2 - k];
            }
        }

        for (int64_t t = 0; t < n_seq_tokens; ++t) {
            const int64_t i = s*n_seq_tokens + t;

            // tokens at t, t-1, t-2 ...; a sequence restarts after an eos, so anything at or
            // before the closest previous eos reads as eos
            bool cut = false;
            for (int64_t k = 0; k < n_ngram; ++k) {
                const int64_t src = t - k;
                const llama_token tok = cut ? tok_eos
                    : src >= 0 ? token_at(s*n_seq_tokens + src)
                    : hist[-src - 1];
                shifted[k] = tok;
                cut = cut || (k > 0 && tok == tok_eos);
            }

            toks[i] = (float) token_at(i);

            for (int64_t n = 2; n <= n_ngram; ++n) {
                // unsigned: the products overflow by design, and the remainder has to stay
                // non-negative to be a row index
                uint64_t mixed = (uint64_t) shifted[0]*hparams.ple_ngram_mult[0];
                for (int64_t k = 1; k < n; ++k) {
                    mixed ^= (uint64_t) shifted[k]*hparams.ple_ngram_mult[k];
                }

                const int64_t h0 = (n - 2)*n_per_ngram;
                for (int64_t h = h0; h < h0 + n_per_ngram; ++h) {
                    ids[i*n_head_ngram + h] =
                        (int32_t) (mixed % hparams.ple_ngram_vocab[h] + hparams.ple_ngram_offs[h]);
                }
            }
        }
    }

    ggml_backend_tensor_set(ngram_ids, ids.data(),  0, ids.size() *ggml_element_size(ngram_ids));
    ggml_backend_tensor_set(tokens,    toks.data(), 0, toks.size()*ggml_element_size(tokens));
}

// The MTP block predicts the next-next token from the target's hyper-connection streams and the
// embedding of the token the target just produced. One sparse-attention layer, no PLE.
// ref: sglang qwen4_exp_mtp.py, and src/models/deepseek4.cpp for the same shape of MTP block
llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : graph(model, params, true) {
    // one block also keeps t_h_nextn below correct: the draft context reads it masked, and only
    // a chained head would put non-output tokens in a draft batch. See common/speculative.cpp
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN4EXP MTP supports a single MTP block");
    GGML_ASSERT(ubatch.token && "QWEN4EXP MTP requires token input");

    const int64_t hc     = hparams.hc_count;
    const int64_t n_embd = hparams.n_embd;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) (n_embd*hc) && "QWEN4EXP MTP hidden width mismatch");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_out());

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    // the reference norms the whole hyper-connection row and only then splits it into streams,
    // so hnorm is hc_count*n_embd wide - unlike deepseek4, which norms each stream
    ggml_tensor * h_norm = build_norm(h, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    h_norm = ggml_reshape_3d(ctx0, h_norm, n_embd, hc, n_tokens);
    cb(h_norm, "mtp_hnorm", il);

    // the embedding term is shared by every stream
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    // fc_embedding(e) + fc_hidden(h) is one projection of the concatenation, see conversion
    ggml_tensor * inpL = build_lora_mm(layer.nextn.eh_proj,
            ggml_concat(ctx0, e_norm, h_norm, 0), layer.nextn.eh_proj_s);
    cb(inpL, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur    = build_hc_pre(inpL,
            layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject, &inject, il);
    cb(cur, "mtp_hc_attn_pre", il);

    // the draft runs the attention dense: a QSA indexer would need its own cache, and one layer
    // spends its time reading weights, not attending - see the measurements in the notes
    cur = build_layer_attn(inp_attn, nullptr, cur, inp_pos, sections, il);

    inpL = build_hc_post(inpL, cur, inject, il);
    cb(inpL, "mtp_hc_attn_post", il);

    cur = build_hc_pre(inpL,
            layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject, &inject, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_layer_ffn(cur, il);
    cb(cur, "mtp_ffn_out", il);

    inpL = build_hc_post(inpL, cur, inject, il);
    cb(inpL, "mtp_l_out", il);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, hc*n_embd, n_tokens);

    // chained heads read the streams back, as the trunk hands them over
    res->t_h_nextn = flat;

    if (inp_out_ids) {
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    cur = build_hc_pre(inpL, model.hc_mix_norm, model.hc_mix_down, model.hc_mix_up, nullptr, nullptr, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }

    return std::make_unique<graph>(*this, params);
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, true) {
    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    const int64_t hc = hparams.hc_count;

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inp = build_inp_embd(model.tok_embd);

    auto * inp_mem = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // every hyper-connection stream starts as a copy of the token embedding
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    llm_graph_input_qsa * inp_qsa = build_qsa_inp();

    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_context *>(mctx);

    llm_graph_input_ple * inp_ple = nullptr;
    for (int il = 0; il < n_layer; ++il) {
        if (!model.layers[il].ple_ngram_embd) {
            continue;
        }

        // the hash constants are derived per PLE layer, only one such layer is supported
        GGML_ASSERT(!inp_ple && "only one PLE layer is supported");

        const int64_t n_head_ngram = hparams.ple_ngram_heads;

        // eos pads the history and cuts it at sequence breaks, so it has to exist
        GGML_ASSERT(model.vocab.token_eos() != LLAMA_TOKEN_NULL &&
                "PLE needs eos to pad the n-gram history");

        auto ple = std::make_unique<llm_graph_input_ple>(model.hparams, mctx_hyb->get_ple(),
                model.vocab.token_eos(), il);

        ple->ngram_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_head_ngram*n_tokens);
        ple->tokens    = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
        ggml_set_input(ple->ngram_ids);
        ggml_set_input(ple->tokens);

        inp_ple = (llm_graph_input_ple *) res->add_input(std::move(ple));
    }

    // PLE has a recurrent state of its own, so the delta-net row below stays exactly the
    // delta-net conv state and both can use the shared build_conv_state
    llm_graph_input_rs * inp_ple_rs = nullptr;
    if (inp_ple) {
        const auto * mctx_ple = mctx_hyb->get_ple();

        auto rs = std::make_unique<llm_graph_input_rs_2nd>(mctx_ple);

        const int64_t n_rs = mctx_ple->get_n_rs();

        rs->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
        ggml_set_input(rs->s_copy);

        rs->s_copy_main  = ggml_view_1d(ctx0, rs->s_copy, ubatch.n_seqs, 0);
        rs->s_copy_extra = ggml_view_1d(ctx0, rs->s_copy, n_rs - ubatch.n_seqs,
                ubatch.n_seqs*rs->s_copy->nb[0]);

        rs->head = mctx_ple->get_head();
        rs->rs_z = mctx_ple->get_rs_z();

        inp_ple_rs = (llm_graph_input_rs *) res->add_input(std::move(rs));
    }

    for (int il = 0; il < n_layer; ++il) {
        if (model.layers[il].ple_ngram_embd) {
            inpL = ggml_add(ctx0, inpL,
                    build_ple(inpL, inp_ple->ngram_ids, inp_ple->tokens, inp_ple_rs, il));
            cb(inpL, "ple_out", il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur    = build_hc_pre(inpL,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject, &inject, il);
        cb(cur, "hc_attn_pre", il);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp_mem->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp_mem->get_attn(), inp_qsa, cur, inp_pos, sections, il);
        }

        inpL = build_hc_post(inpL, cur, inject, il);
        cb(inpL, "hc_attn_post", il);

        cur = build_hc_pre(inpL,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject, &inject, il);
        cb(cur, "hc_ffn_pre", il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(inpL, cur, inject, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, hc*n_embd, n_tokens);

    // the MTP block consumes the streams, so hand them over before the mixer collapses them. The
    // read-back covers every token, so it has to see the rows before the output masking below.
    res->t_h_nextn = flat;

    if (inp_out_ids) {
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    // the mixer without an injection gate is also the final norm
    ggml_tensor * cur = build_hc_pre(inpL, model.hc_mix_norm, model.hc_mix_down, model.hc_mix_up, nullptr, nullptr, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_norm(
        ggml_tensor * x,
        ggml_tensor * w,
        int           il) {
    ggml_tensor * cur = ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps);
    cur = ggml_mul(ctx0, cur, ggml_reshape_2d(ctx0, w, n_embd, hparams.hc_count));
    cb(cur, "hc_norm", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_pre(
         ggml_tensor * x,
         ggml_tensor * norm,
         ggml_tensor * down,
         ggml_tensor * up,
         ggml_tensor * inject,
        ggml_tensor ** inject_out,
                 int   il) {
    const int64_t hc = hparams.hc_count;
    const int64_t nt = x->ne[2];

    ggml_tensor * xn   = build_hc_norm(x, norm, il);
    ggml_tensor * flat = ggml_reshape_2d(ctx0, xn, hc*n_embd, nt);

    ggml_tensor * mix = build_lora_mm(down, flat);
    mix = ggml_silu(ctx0, ggml_scale(ctx0, mix, 1.0f/hc));
    mix = ggml_sigmoid(ctx0, build_lora_mm(up, mix));
    mix = ggml_reshape_3d(ctx0, mix, n_embd, hc, nt);
    cb(mix, "hc_mix", il);

    // mean over the streams
    ggml_tensor * cur = ggml_mul(ctx0, mix, xn);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    cur = ggml_mean(ctx0, cur);
    cur = ggml_reshape_2d(ctx0, cur, n_embd, nt);

    if (inject_out) {
        ggml_tensor * w = build_lora_mm(inject, flat);
        w = ggml_scale(ctx0, ggml_sigmoid(ctx0, ggml_scale(ctx0, w, 1.0f/hc)), 2.0f);
        cb(w, "hc_inject", il);
        *inject_out = w;
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_post(
        ggml_tensor * residual,
        ggml_tensor * y,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.hc_count;
    const int64_t nt = y->ne[1];

    ggml_tensor * cur = ggml_reshape_3d(ctx0, y, n_embd, 1, nt);
    cur = ggml_repeat_4d(ctx0, cur, n_embd, hc, nt, 1);
    cur = ggml_mul(ctx0, cur, ggml_reshape_3d(ctx0, inject, 1, hc, nt));
    cb(cur, "hc_injected", il);

    return ggml_add(ctx0, residual, cur);
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        ggml_tensor *        x,
        ggml_tensor *        ngram_ids,
        ggml_tensor *        tokens,
        llm_graph_input_rs * inp,
        int                  il) {
    const auto & layer = model.layers[il];

    const int64_t hc      = hparams.hc_count;
    const int64_t hc_dim  = hc*n_embd;
    const int64_t n_ngram = hparams.ple_ngram_size;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * emb = ggml_get_rows(ctx0, layer.ple_ngram_embd, ngram_ids);
    emb = ggml_reshape_2d(ctx0, emb, hparams.n_embd_ple, n_tokens);
    cb(emb, "ple_embd", il);

    ggml_tensor * key = build_lora_mm(layer.ple_key, emb);
    key = build_hc_norm(ggml_reshape_3d(ctx0, key, n_embd, hc, n_tokens), layer.ple_key_norm, il);

    ggml_tensor * val   = build_lora_mm(layer.ple_value, emb);
    ggml_tensor * query = build_hc_norm(x, layer.ple_query_norm, il);

    ggml_tensor * gate = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    gate = ggml_scale(ctx0, gate, 1.0f/sqrtf((float) n_embd));

    // signed square root keeps the gate in a narrow range
    gate = ggml_mul(ctx0,
            ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, gate), 1e-6f, INFINITY)),
            ggml_sgn(ctx0, gate));
    gate = ggml_sigmoid(ctx0, gate);
    cb(gate, "ple_gate", il);

    ggml_tensor * cur = ggml_reshape_3d(ctx0, val, n_embd, 1, n_tokens);
    cur = ggml_repeat_4d(ctx0, cur, n_embd, hc, n_tokens, 1);
    cur = ggml_mul(ctx0, cur, gate);
    cb(cur, "ple_value", il);

    // dilated depthwise convolution over the normalized values
    const int64_t n_kernel = layer.ple_conv1d->ne[0];
    const int64_t n_taps   = (n_kernel - 1)*n_ngram;
    const int64_t n_gdn    = (hparams.ssm_d_conv - 1)*(hparams.ssm_d_inner + 2*hparams.ssm_n_group*hparams.ssm_d_state);

    GGML_UNUSED(n_gdn);
    ggml_tensor * conv_in = build_conv_state(inp, inp->mctx->get_r_l(il),
            ggml_reshape_3d(ctx0, build_hc_norm(cur, layer.ple_conv_norm, il), hc_dim, n_seq_tokens, n_seqs),
            n_taps + 1, hc_dim, il);

    const size_t es_in = ggml_element_size(conv_in);
    const size_t es_w  = ggml_element_size(layer.ple_conv1d);

    ggml_tensor * conv = nullptr;
    for (int64_t k = 0; k < n_kernel; ++k) {
        ggml_tensor * w = ggml_view_2d(ctx0, layer.ple_conv1d, 1, hc_dim, layer.ple_conv1d->nb[1], k*es_w);
        ggml_tensor * v = ggml_cont(ctx0, ggml_view_3d(ctx0, conv_in, n_seq_tokens, hc_dim, n_seqs,
                    conv_in->nb[1], conv_in->nb[2], k*n_ngram*es_in));
        v = ggml_mul(ctx0, v, w);

        conv = conv ? ggml_add(ctx0, conv, v) : v;
    }
    conv = ggml_silu(ctx0, conv);
    conv = ggml_cont(ctx0, ggml_transpose(ctx0, conv));
    cb(conv, "ple_conv", il);

    // remember the tokens of this ubatch as the n-gram history of the next one. These are token
    // ids, not a conv state, so build_conv_state does not apply - it reshapes the whole row
    {
        ggml_tensor * s_all = inp->mctx->get_s_l(il);

        // gather the source rows directly: build_rs would also clear a row and propagate the
        // extra states, which is what a delta-net state needs but not what this is - the history
        // is moved by hand below, and the clearing would wipe it on every decode step
        ggml_tensor * s_row = ggml_get_rows(ctx0,
                ggml_reshape_2d(ctx0, s_all, n_ngram - 1, s_all->ne[1]), inp->s_copy_main);

        ggml_tensor * hist = ggml_reshape_3d(ctx0, s_row, n_ngram - 1, 1, n_seqs);
        ggml_tensor * both = ggml_concat(ctx0, hist,
                ggml_transpose(ctx0, ggml_reshape_3d(ctx0, tokens, 1, n_seq_tokens, n_seqs)), 0);

        const size_t es = ggml_element_size(s_all);

        ggml_tensor * tail = ggml_view_3d(ctx0, both, n_ngram - 1, 1, n_seqs,
                both->nb[1], both->nb[2], (both->ne[0] - (n_ngram - 1))*ggml_element_size(both));

        ggml_tensor * dst = ggml_view_3d(ctx0, s_all, n_ngram - 1, 1, n_seqs,
                (n_ngram - 1)*es, (n_ngram - 1)*es,
                (inp->mctx->get_head()*(n_ngram - 1))*es);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, tail, dst));
    }

    return ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, conv, n_embd, hc, n_tokens));
}

// No indexer cache means QSA cannot run for this context (see llama_model::create_memory), and
// the sparse layers fall back to dense.
llm_graph_input_qsa * llama_model_qwen4exp::graph::build_qsa_inp() {
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_context *>(mctx);

    if (!hparams.has_qsa() || !mctx_hyb->get_lid()) {
        return nullptr;
    }

    const int64_t  ratio = hparams.indexer_block_size;
    // round 256 up to a whole number of blocks; GGML_PAD would need ratio to be a power of two
    const uint32_t pad   = ((256u + (uint32_t) ratio - 1)/(uint32_t) ratio)*(uint32_t) ratio;
    const int64_t  n_pos = mctx_hyb->get_n_pos(pad, cparams.n_seq_max);
    const int64_t  ns    = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;
    const int64_t  n_tps = n_tokens/ns;

    auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, ratio, pad);

    qsa->n_kv_lid = mctx_hyb->get_lid()->get_n_kv();

    // one query per stream is what lets the attention read a gathered buffer: the keys of a flash
    // attention call are shared by all of its queries, and each query picks its own blocks. Longer
    // ubatches - a prefill, or a speculative verification batch - take the mask path below
    const bool gather = n_tps == 1 && cparams.flash_attn;

    qsa->k_idxs   = mctx_hyb->get_lid()->build_input_k_idxs(ctx0, ubatch);
    qsa->pos_cell = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_pos, ns);
    qsa->blk_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, n_pos/ratio, n_tps, 1, ns);
    qsa->blk_cur  = ggml_new_tensor_4d(ctx0, GGML_TYPE_I32, 1,           n_tps, 1, ns);

    ggml_set_input(qsa->pos_cell);
    ggml_set_input(qsa->blk_mask);
    ggml_set_input(qsa->blk_cur);

    // only the mask path walks the cells, and an input that nothing reads gets no buffer
    if (!gather) {
        qsa->cell_blk = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, mctx_hyb->get_attn()->get_n_kv(), ns);
        ggml_set_input(qsa->cell_blk);
    }

    // The gather indexes V by cell, so it needs V stored untransposed, which is what flash
    // attention asks for anyway (llama_kv_cache is built with v_trans = !flash_attn)
    if (gather) {
        qsa->pos_cell_f = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pos, ns);
        qsa->pos_vis    = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pos, ns);

        ggml_set_input(qsa->pos_cell_f);
        ggml_set_input(qsa->pos_vis);
    }

    return (llm_graph_input_qsa *) res->add_input(std::move(qsa));
}

// Block scores, shared by the mask and the gather path: relu(q . k_blk) summed over the indexer
// heads, with the incomplete and invisible blocks already closed by blk_mask.
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_scores(
        llm_graph_input_qsa * inp_qsa,
        ggml_tensor *         cur,
        ggml_tensor *         inp_pos,
        int *                 sections,
        int                   il) {
    const auto & layer = model.layers[il];

    const int64_t n_idx = hparams.indexer_head_size;
    const int64_t n_ih  = hparams.indexer_n_head;
    const int64_t ratio = hparams.indexer_block_size;
    const int64_t n_blk = inp_qsa->blk_mask->ne[0];
    const int64_t n_tps = inp_qsa->blk_mask->ne[1];
    const int64_t ns    = inp_qsa->blk_mask->ne[3];

    // indexer queries: normed, then the same partial IMRoPE as the attention queries
    ggml_tensor * q = build_lora_mm(layer.index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, n_idx, n_ih, n_tokens);
    q = build_norm(q, layer.index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "qsa_q", il);

    // indexer keys are cached raw: the norm and the RoPE come after the block pooling
    ggml_tensor * k = build_lora_mm(layer.index_k_proj, cur);
    k = ggml_reshape_3d(ctx0, k, n_idx, 1, n_tokens);
    ggml_build_forward_expand(gf, inp_qsa->mctx->get_lid()->cpy_k(ctx0, k, inp_qsa->k_idxs, il));

    // read the cached keys by position, so a block always covers the same `ratio` positions
    ggml_tensor * kc = inp_qsa->mctx->get_lid()->get_k(ctx0, il);
    kc = ggml_view_3d(ctx0, kc, n_idx, kc->ne[2], ns, kc->nb[2], kc->nb[3], 0);

    ggml_tensor * kb = ggml_get_rows(ctx0, kc, inp_qsa->pos_cell);
    kb = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, kb, n_idx, ratio, n_blk*ns), 1, 0, 2, 3));
    kb = ggml_reshape_3d(ctx0, ggml_mean(ctx0, kb), n_idx, 1, n_blk*ns);
    kb = build_norm(kb, layer.index_k_norm, nullptr, LLM_NORM_RMS, il);

    // a block covers positions [b*ratio, b*ratio + ratio), so it is anchored to b*ratio
    ggml_tensor * bpos = ggml_repeat_4d(ctx0,
            ggml_arange(ctx0, 0.0f, (float) n_blk*ratio, (float) ratio),
            n_blk*ns*hparams.n_pos_per_embd(), 1, 1, 1);
    bpos = ggml_cast(ctx0, bpos, GGML_TYPE_I32);
    kb = ggml_rope_multi(ctx0, kb, bpos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(kb, "qsa_k_blk", il);

    // score = sum_h relu(q_h . k_blk)/sqrt(n_idx), the indexer weights are prescaled
    ggml_tensor * w = ggml_scale(ctx0,
            ggml_repeat_4d(ctx0, ggml_arange(ctx0, 1.0f, 2.0f, 1.0f), n_ih, n_tps, 1, ns),
            1.0f/sqrtf((float) n_idx));
    ggml_tensor * sc = ggml_lightning_indexer(ctx0,
            ggml_reshape_4d(ctx0, q,  n_idx, n_ih, n_tps, ns),
            ggml_reshape_4d(ctx0, kb, n_idx, 1,   n_blk, ns),
            w, inp_qsa->blk_mask);
    cb(sc, "qsa_score", il);

    return sc;
}

// The mask path: turn the block scores into the kq mask of the cells the attention may read. Every
// cell is walked whatever the mask says - see build_qsa_gather() for the decode path.
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_mask(
        llm_graph_input_qsa * inp_qsa,
        ggml_tensor *         sc,
        ggml_tensor *         kq_mask,
        int                   il) {
    const int64_t ratio = hparams.indexer_block_size;
    const int64_t n_blk = inp_qsa->blk_mask->ne[0];
    const int64_t n_tps = inp_qsa->blk_mask->ne[1];
    const int64_t ns    = inp_qsa->blk_mask->ne[3];
    const int64_t n_kv  = inp_qsa->cell_blk->ne[0];
    const int64_t n_top = std::min(n_blk, (int64_t) hparams.indexer_top_k/ratio);

    // fully masked block axis, then open the top-k blocks and the query's own block
    ggml_tensor * bm = ggml_fill(ctx0,
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_blk, n_tps*ns), -INFINITY);

    ggml_tensor * idx = ggml_top_k(ctx0, sc, n_top);
    bm = ggml_set_rows(ctx0, bm,
            ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_top, n_tps*ns), 0.0f),
            ggml_reshape_2d(ctx0, idx, n_top, n_tps*ns));
    bm = ggml_set_rows(ctx0, bm,
            ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tps*ns), 0.0f),
            ggml_reshape_2d(ctx0, inp_qsa->blk_cur, 1, n_tps*ns));

    // block -> cell: gather the block mask with the block each cell belongs to
    bm = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, bm, n_blk, n_tps, ns), 1, 0, 2, 3));

    ggml_tensor * mask = ggml_get_rows(ctx0, bm, inp_qsa->cell_blk);
    mask = ggml_cont(ctx0, ggml_permute(ctx0, mask, 1, 0, 2, 3));
    mask = ggml_reshape_4d(ctx0, mask, n_kv, n_tps, 1, ns);

    // the causal mask also closes empty cells and the cells of the other sequences
    mask = ggml_add(ctx0, mask,
            kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32));
    if (kq_mask->type != GGML_TYPE_F32) {
        mask = ggml_cast(ctx0, mask, kq_mask->type);
    }
    cb(mask, "qsa_kq_mask", il);

    return mask;
}

// The decode path: gather the selected keys and values into one compact buffer and attend over
// that, instead of masking the whole cache. The CUDA flash attention only trims the tail of the KV
// range (flash_attn_mask_to_KV_max scans back to the first tile that is not all -inf), so a mask
// with holes in it costs exactly as much as dense attention. One query per stream is what makes
// this possible: the keys a flash attention call reads are shared by all of its queries, and each
// query picks its own blocks.
// ref: src/models/minimax-m3.cpp, the msa_decode branch
void llama_model_qwen4exp::graph::build_qsa_gather(
        llm_graph_input_qsa * inp_qsa,
        ggml_tensor *         sc,
        ggml_tensor *         k_all,
        ggml_tensor *         v_all,
        ggml_tensor *         kq_mask,
        ggml_tensor **        k_out,
        ggml_tensor **        v_out,
        ggml_tensor **        m_out) {
    const int64_t ratio = hparams.indexer_block_size;
    const int64_t n_blk = inp_qsa->blk_mask->ne[0];
    const int64_t ns    = inp_qsa->blk_mask->ne[3];
    const int64_t n_pos = inp_qsa->pos_cell_f->ne[0];
    const int64_t n_top = std::min(n_blk, (int64_t) hparams.indexer_top_k/ratio);

    const int64_t d_head = k_all->ne[0];
    const int64_t n_hkv  = k_all->ne[1];
    const int64_t n_kv   = k_all->ne[2];

    // The query's own block has to be read on top of the top-k. Do not append it: below the indexer
    // budget the top-k reaches into the masked blocks and can pick that same block, and a key
    // gathered twice takes two shares of the softmax. Score it above everything instead, so it is
    // always rank 0 of a top-k of n_top + 1 - the same visible set as the mask path, no duplicates.
    // ref: the bias input of src/models/minimax-m3.cpp
    // n_top can already be every block, and top_k cannot ask for more rows than there are. Then
    // the own block is still rank 0 and the rest is all of them, which is what the mask path opens
    const int64_t n_sel = std::min(n_blk, n_top + 1);
    const int64_t n_g   = ratio*n_sel;

    ggml_tensor * scb = ggml_cont(ctx0, ggml_permute(ctx0,
            ggml_reshape_3d(ctx0, sc, n_blk, 1, ns), 1, 0, 2, 3));
    scb = ggml_set_rows(ctx0, scb,
            ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, ns), 1e30f),
            ggml_reshape_2d(ctx0, inp_qsa->blk_cur, 1, ns));
    scb = ggml_cont(ctx0, ggml_permute(ctx0, scb, 1, 0, 2, 3));

    ggml_tensor * sel = ggml_reshape_3d(ctx0,
            ggml_cast(ctx0, ggml_top_k(ctx0, ggml_reshape_3d(ctx0, scb, n_blk, 1, ns), n_sel),
                    GGML_TYPE_F32), n_sel, 1, ns);

    // block -> the `ratio` positions it covers: tj[t, b] = sel[b]*ratio + t
    ggml_tensor * tj = ggml_add(ctx0,
            ggml_repeat_4d(ctx0, ggml_scale(ctx0, ggml_reshape_4d(ctx0, sel, 1, n_sel, 1, ns), (float) ratio),
                    ratio, n_sel, 1, ns),
            ggml_reshape_3d(ctx0, ggml_arange(ctx0, 0.0f, (float) ratio, 1.0f), ratio, 1, 1));
    ggml_tensor * tokj = ggml_cast(ctx0, ggml_reshape_2d(ctx0, tj, n_g, ns), GGML_TYPE_I32);

    // position -> cell. The map has to be read as values here, so it is the F32 copy
    ggml_tensor * cs = ggml_get_rows(ctx0,
            ggml_reshape_3d(ctx0, inp_qsa->pos_cell_f, 1, n_pos, ns), tokj);

    // cell -> the n_hkv rows of that cell: tr[h, l] = cs[l]*n_hkv + h
    ggml_tensor * tr = ggml_add(ctx0,
            ggml_repeat_4d(ctx0, ggml_scale(ctx0, ggml_reshape_4d(ctx0, cs, 1, n_g, ns, 1), (float) n_hkv),
                    n_hkv, n_g, ns, 1),
            ggml_reshape_2d(ctx0, ggml_arange(ctx0, 0.0f, (float) n_hkv, 1.0f), n_hkv, 1));
    ggml_tensor * tokr = ggml_cast(ctx0, ggml_reshape_2d(ctx0, tr, n_hkv*n_g, ns), GGML_TYPE_I32);

    ggml_tensor * k3 = ggml_view_3d(ctx0, k_all, d_head, n_hkv*n_kv, ns, k_all->nb[1], k_all->nb[3], 0);
    ggml_tensor * v3 = ggml_view_3d(ctx0, v_all, d_head, n_hkv*n_kv, ns, v_all->nb[1], v_all->nb[3], 0);

    ggml_tensor * kg = ggml_get_rows(ctx0, k3, tokr);
    ggml_tensor * vg = ggml_get_rows(ctx0, v3, tokr);

    // the rows came out with the heads fastest, so this is already the layout the cache hands to
    // build_attn_mha - one gathered position where a cell used to be
    ggml_tensor * kfa = ggml_reshape_4d(ctx0, kg, d_head, n_hkv, n_g, ns);
    ggml_tensor * vfa = ggml_reshape_4d(ctx0, vg, d_head, n_hkv, n_g, ns);

    // a quantized cache is gathered as it is stored, and flash attention wants F16
    if (ggml_is_quantized(kfa->type)) { kfa = ggml_cast(ctx0, kfa, GGML_TYPE_F16); }
    if (ggml_is_quantized(vfa->type)) { vfa = ggml_cast(ctx0, vfa, GGML_TYPE_F16); }

    // two terms. pos_vis closes the positions the query may not read, which the gather can pick up
    // because it reads whole blocks - the trailing one runs past the query and those positions have
    // no cell, so they would otherwise land on cell 0. The causal mask then closes the empty cells
    // and the cells of the other sequences, which position space knows nothing about
    ggml_tensor * mg = ggml_get_rows(ctx0,
            ggml_reshape_3d(ctx0, inp_qsa->pos_vis, 1, n_pos, ns), tokj);

    ggml_tensor * km = ggml_view_3d(ctx0, kq_mask, kq_mask->ne[0], 1, ns,
            kq_mask->nb[1], kq_mask->nb[3], 0);
    km = ggml_cont(ctx0, ggml_permute(ctx0, km, 1, 0, 2, 3));
    if (km->type != GGML_TYPE_F32) {
        km = ggml_cast(ctx0, km, GGML_TYPE_F32);
    }

    ggml_tensor * tokc = ggml_cast(ctx0, ggml_reshape_2d(ctx0, cs, n_g, ns), GGML_TYPE_I32);

    mg = ggml_add(ctx0, mg, ggml_get_rows(ctx0, km, tokc));

    *k_out = kfa;
    *v_out = vfa;
    *m_out = ggml_cast(ctx0, ggml_reshape_4d(ctx0, mg, n_g, 1, 1, ns), GGML_TYPE_F16);
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        llm_graph_input_qsa *     inp_qsa,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    // a single Q projection produces the query and the output gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (inp_qsa == nullptr) {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    } else {
        // same as build_attn() above, except that the mask only lets the selected cells through
        // ref: llm_graph_context::build_attn() for the k_dsa input
        const auto * mctx = inp->mctx;

        // a quantized KV cache rotates K/V before storing them, see llm_graph_context::build_attn
        if (inp->self_k_rot) {
            Qcur = llama_mul_mat_hadamard(ctx0, Qcur, inp->self_k_rot);
            Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp->self_k_rot);
        }
        if (inp->self_v_rot) {
            Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp->self_v_rot);
        }

        ggml_build_forward_expand(gf, mctx->cpy_k(ctx0, Kcur, inp->get_k_idxs(), il));
        ggml_build_forward_expand(gf, mctx->cpy_v(ctx0, Vcur, inp->get_v_idxs(), il));

        ggml_tensor * sc = build_qsa_scores(inp_qsa, cur, inp_pos, sections, il);

        if (inp_qsa->pos_cell_f) {
            ggml_tensor * kg = nullptr;
            ggml_tensor * vg = nullptr;
            ggml_tensor * mg = nullptr;

            build_qsa_gather(inp_qsa, sc, mctx->get_k(ctx0, il), mctx->get_v(ctx0, il),
                    inp->get_kq_mask(), &kg, &vg, &mg);

            cur = build_attn_mha(Qcur, kg, vg, nullptr, mg, nullptr, nullptr, kq_scale, il);
        } else {
            ggml_tensor * mask = build_qsa_mask(inp_qsa, sc, inp->get_kq_mask(), il);

            cur = build_attn_mha(Qcur, mctx->get_k(ctx0, il), mctx->get_v(ctx0, il),
                    nullptr, mask, nullptr, nullptr, kq_scale, il);
        }

        if (inp->self_v_rot) {
            cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
        }
    }
    cb(cur, "attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, cur, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, cur, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * gate = ggml_mul(ctx0, ggml_softplus(ctx0, ggml_add(ctx0, alpha, model.layers[il].ssm_dt)),
            model.layers[il].ssm_a);
    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(gate, "gate", il);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2*num_k_heads*head_k_dim;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed,
            conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);

    ggml_tensor * conv_qkv_mix = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, conv_kernel));
    cb(conv_qkv_mix, "conv_output_silu", il);

    const int64_t qkv_dim = head_k_dim*num_k_heads*2 + head_v_dim*num_v_heads;
    const int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim), nb1_qkv, nb1_qkv*n_seq_tokens, 0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim), nb1_qkv, nb1_qkv*n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, head_k_dim*num_k_heads));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim), nb1_qkv, nb1_qkv*n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2*head_k_dim*num_k_heads));

    q_conv = ggml_l2_norm(ctx0, q_conv, hparams.f_norm_rms_eps);
    k_conv = ggml_l2_norm(ctx0, k_conv, hparams.f_norm_rms_eps);

    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // gated output norm, Qwen4-Exp uses a sigmoid gate here
    ggml_tensor * z_4d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    output = ggml_mul(ctx0, build_norm(output, model.layers[il].ssm_norm, nullptr, LLM_NORM_RMS, il),
            ggml_sigmoid(ctx0, z_4d));

    output = ggml_reshape_3d(ctx0, output, head_v_dim*num_v_heads, n_seq_tokens, n_seqs);

    cur = build_lora_mm(model.layers[il].ssm_out, output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    return ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens*n_seqs);
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    ggml_tensor * ffn_shexp =
        build_ffn(cur,
            model.layers[il].ffn_up_shexp,   NULL, model.layers[il].ffn_up_shexp_s,
            model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
            model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "ffn_shexp", il);

    ggml_tensor * shared_gate = ggml_sigmoid(ctx0, build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur));
    cb(shared_gate, "shared_expert_gate_sigmoid", il);

    return ggml_add(ctx0, moe_out, ggml_mul(ctx0, ffn_shexp, shared_gate));
}
