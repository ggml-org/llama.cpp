#include "../llama-model.h"
#include "../llama-graph.h"

#include "llm_build_lfm2.h"
#include <cmath>

llm_build_lfm2::llm_build_lfm2(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    cb(cur, "model.embed_tokens", -1);

    ggml_tensor * inp_pos     = build_inp_pos();
    auto        * inp_hybrid  = build_inp_mem_hybrid();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        auto * prev_cur = cur;
        cur = build_norm(cur, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "model.layers.{}.operator_norm", il);

        // TODO: implement recurrent/attention logic inline
        // cur = hparams.is_recurrent(il) ?
        //     build_shortconv_block(cur, inp_hybrid->get_recr(), il) :
        //     build_attn_block(cur, inp_pos, inp_hybrid->get_attn(), il) ;

        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0,      cur, inp_out_ids);
            prev_cur = ggml_get_rows(ctx0, prev_cur, inp_out_ids);
        }
;
        cur = ggml_add(ctx0, prev_cur, cur);
        // TODO: implement feed_forward inline
        // cur = ggml_add(ctx0, cur, build_feed_forward(cur, il));
    }
;
    cur = build_norm(cur, model.tok_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "model.embedding_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "lm_head", -1);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
;