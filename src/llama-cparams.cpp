#include "llama-cparams.h"

size_t llama_max_parallel_sequences(void) {
    return LLAMA_MAX_SEQ;
}

bool llama_cparams::is_same(const llama_cparams & other) const {
    return
        n_ctx               == other.n_ctx               &&
        n_batch             == other.n_batch             &&
        n_ubatch            == other.n_ubatch            &&
        n_seq_max           == other.n_seq_max           &&
        n_threads           == other.n_threads           &&
        n_threads_batch     == other.n_threads_batch     &&
        rope_freq_base      == other.rope_freq_base      &&
        rope_freq_scale     == other.rope_freq_scale     &&
        n_ctx_orig_yarn     == other.n_ctx_orig_yarn     &&
        yarn_ext_factor     == other.yarn_ext_factor     &&
        yarn_attn_factor    == other.yarn_attn_factor    &&
        yarn_beta_fast      == other.yarn_beta_fast      &&
        yarn_beta_slow      == other.yarn_beta_slow      &&
        defrag_thold        == other.defrag_thold        &&
        embeddings          == other.embeddings          &&
        causal_attn         == other.causal_attn         &&
        offload_kqv         == other.offload_kqv         &&
        flash_attn          == other.flash_attn          &&
        no_perf             == other.no_perf             &&
        warmup              == other.warmup              &&
        op_offload          == other.op_offload          &&
        graph_reuse         == other.graph_reuse         &&
        pooling_type        == other.pooling_type        &&
        cb_eval             == other.cb_eval             &&
        cb_eval_user_data   == other.cb_eval_user_data;
}
