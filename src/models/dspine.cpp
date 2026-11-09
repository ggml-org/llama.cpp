#include "models.h"

#include "llama-impl.h"

#include <cmath>

// DSpine packs [top_k candidates, top_k x top_k transition scores] into one n_embd row per block position
void llama_model_dflash::load_hparams_dspine(llama_model_loader & ml) {
    ml.get_key(LLM_KV_DSPINE_EPS, hparams.dspine_eps, false);
    bool causal = false;
    ml.get_key(LLM_KV_ATTENTION_CAUSAL, causal, false);
    const uint32_t top_k = hparams.dspine_head_topk;
    if (!causal || !std::isfinite(hparams.dspine_eps) || !(hparams.dspine_eps > 0.0f) || top_k < 2 || top_k > 64 || (top_k & (top_k - 1)) != 0 || hparams.dflash_block_size < 2 ||
            hparams.dflash_selector_top_k > 0 || (uint64_t) top_k * (top_k + 1) > hparams.n_embd) {
        throw std::runtime_error("invalid DSpine DFlash metadata");
    }
}

void llama_model_dflash::load_tensors_dspine() {
    LLAMA_LOAD_LOCALS;

    const struct ggml_tensor * codebook_meta = ml->get_tensor_meta("dspine_codebook.weight");
    const struct ggml_tensor * gate_meta     = ml->get_tensor_meta("dspine_gate.weight");
    const bool conv_meta = hparams.dflash_conv_kernel_size > 0 && hparams.dflash_conv_group_size > 0;
    // the bridge and lattice are built on the plain DFlash layers only
    if (!codebook_meta || !gate_meta || d2t || conv_meta || backbone != BACKBONE_BASE) {
        throw std::runtime_error("DSpine draft needs its code bridge, a plain DFlash decoder and the full target vocabulary");
    }
    const int64_t code_size = codebook_meta->ne[0];
    const int64_t rank      = gate_meta->ne[1];
    // checked before any tensor is created: the buffer probe of a zero-size weight aborts
    if (code_size <= 0 || rank <= 0 || n_vocab < (int64_t) hparams.dspine_head_topk) {
        throw std::runtime_error("invalid DSpine code bridge size");
    }

    dspine_codebook  = create_tensor(tn(LLM_TENSOR_DSPINE_CODEBOOK,  "weight"), { code_size, n_vocab }, 0);
    dspine_gate      = create_tensor(tn(LLM_TENSOR_DSPINE_GATE,      "weight"), { code_size, rank }, 0);
    dspine_gate_b    = create_tensor(tn(LLM_TENSOR_DSPINE_GATE,      "bias"),   { rank }, 0);
    dspine_pred_gate = create_tensor(tn(LLM_TENSOR_DSPINE_PRED_GATE, "weight"), { code_size, rank }, 0);
    dspine_prev      = create_tensor(tn(LLM_TENSOR_DSPINE_PREV,      "weight"), { code_size, rank }, 0);
    for (int i = 0; i < n_layer; ++i) {
        layers[i].dspine_read  = create_tensor(tn(LLM_TENSOR_DSPINE_READ,  "weight", i), { n_embd, code_size }, 0);
        layers[i].dspine_write = create_tensor(tn(LLM_TENSOR_DSPINE_WRITE, "weight", i), { rank, n_embd }, 0);
    }

    LLAMA_LOG_INFO("%s: DSpine code bridge: code size = %lld, rank = %lld, head top-k = %u\n", __func__,
            (long long) code_size, (long long) rank, hparams.dspine_head_topk);
}

// rows [beg, beg + n) of every block of a [ne0, n_rows, n_blocks] tensor
static ggml_tensor * dspine_block_rows(ggml_context * ctx0, ggml_tensor * x, int64_t beg, int64_t n) {
    return ggml_cont(ctx0, ggml_view_3d(ctx0, x, x->ne[0], n, x->ne[2], x->nb[1], x->nb[2], beg * x->nb[1]));
}

// sqrt(code_size) * codebook[anchor] for every block: [code_size, n_blocks]
static ggml_tensor * build_dspine_anchor_codes(ggml_context * ctx0, const llama_model & model, ggml_tensor * tokens, int64_t n_rows, int64_t n_blocks) {
    ggml_tensor * anchors = ggml_cont_1d(ctx0, ggml_view_2d(ctx0, tokens, 1, n_blocks, n_rows * tokens->nb[0], 0), n_blocks);
    return ggml_scale(ctx0, ggml_get_rows(ctx0, model.dspine_codebook, anchors), sqrtf((float) model.dspine_codebook->ne[0]));
}

// DSpine code write after a layer: block row t > 0 gets a message gated by its own code and the code of row t - 1
ggml_tensor * llama_model_dflash::graph_dspine::build_layer_out(ggml_tensor * raw, int il) {
    if (!anchor_codes) {
        return raw;
    }

    const auto & layer = model.layers[il];

    const int64_t code_size = anchor_codes->ne[0];
    const int64_t n_blocks  = anchor_codes->ne[1];
    const int64_t n_rows    = n_tokens / n_blocks;
    const int64_t rank      = model.dspine_gate->ne[1];
    const float   eps       = hparams.dspine_eps;

    // the reference adds eps after the square root, so this is neither ggml_rms_norm nor ggml_l2_norm
    ggml_tensor * rms        = ggml_sqrt(ctx0, ggml_scale(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, raw)), 1.0f / n_embd));
    ggml_tensor * normalized = ggml_div(ctx0, raw, ggml_scale_bias(ctx0, rms, 1.0f, eps));

    ggml_tensor * proj  = build_lora_mm(layer.dspine_read, normalized);
    ggml_tensor * codes = ggml_div(ctx0, ggml_scale(ctx0, proj, sqrtf((float) code_size)),
            ggml_scale_bias(ctx0, ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, proj))), 1.0f, eps));

    // row 0 of each block carries the exact code of the anchor token
    codes = ggml_concat(ctx0, ggml_reshape_3d(ctx0, anchor_codes, code_size, 1, n_blocks),
            dspine_block_rows(ctx0, ggml_reshape_3d(ctx0, codes, code_size, n_rows, n_blocks), 1, n_rows - 1), 1);

    ggml_tensor * query  = build_lora_mm(model.dspine_gate,      codes);
    ggml_tensor * sender = build_lora_mm(model.dspine_pred_gate, codes);
    ggml_tensor * value  = build_lora_mm(model.dspine_prev,      codes);

    // move the sender terms one row down inside each block: row 0 gets a zero message
    const auto shift = [&](ggml_tensor * x) {
        return ggml_pad_ext(ctx0, dspine_block_rows(ctx0, x, 0, n_rows - 1), 0, 0, 1, 0, 0, 0, 0, 0);
    };
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_add(ctx0, ggml_add(ctx0, query, shift(sender)), model.dspine_gate_b));
    ggml_tensor * msg  = ggml_reshape_2d(ctx0, ggml_mul(ctx0, gate, shift(value)), rank, n_tokens);

    ggml_tensor * delta = build_lora_mm(layer.dspine_write, msg);

    write_raw   = raw;
    write_rms   = rms;
    write_query = ggml_reshape_2d(ctx0, query, rank, n_tokens);

    ggml_tensor * cur = ggml_add(ctx0, raw, ggml_mul(ctx0, delta, rms));
    cb(cur, "dspine_out", il);

    return cur;
}

// DSpine refinement: redo the last code write with the exact code of each predecessor candidate, then score
// the top_k candidates of each block row. Packed as [candidates, top_k x top_k scores] for the CPU-side walk.
void llama_model_dflash::graph_dspine::build_lattice(ggml_tensor * output, ggml_tensor * output_s) {
    const auto & last = model.layers[hparams.n_layer() - 1];

    const int64_t code_size = anchor_codes->ne[0];
    const int64_t n_blocks  = anchor_codes->ne[1];
    const int64_t n_rows    = n_tokens / n_blocks;
    const int64_t n_pos     = n_rows - 1; // block rows 1..n_rows-1 are drafts
    const int64_t rank      = model.dspine_gate->ne[1];
    const int64_t top_k     = hparams.dspine_head_topk;

    ggml_tensor * cand     = ggml_reshape_3d(ctx0, ggml_top_k(ctx0, res->t_logits, top_k), top_k, n_rows, n_blocks);
    ggml_tensor * cand_pos = dspine_block_rows(ctx0, cand, 1, n_pos);

    // predecessor codes: the anchor for row 1, the candidates of row t - 1 for row t
    ggml_tensor * codes = ggml_repeat_4d(ctx0, ggml_reshape_4d(ctx0, anchor_codes, code_size, 1, 1, n_blocks), code_size, top_k, 1, n_blocks);
    if (n_pos > 1) {
        ggml_tensor * ids  = ggml_reshape_1d(ctx0, dspine_block_rows(ctx0, cand, 1, n_pos - 1), top_k * (n_pos - 1) * n_blocks);
        ggml_tensor * prev = ggml_scale(ctx0, ggml_get_rows(ctx0, model.dspine_codebook, ids), sqrtf((float) code_size));
        codes = ggml_concat(ctx0, codes, ggml_reshape_4d(ctx0, prev, code_size, top_k, n_pos - 1, n_blocks), 2);
    }

    ggml_tensor * query = ggml_reshape_4d(ctx0, dspine_block_rows(ctx0, ggml_reshape_3d(ctx0, write_query, rank, n_rows, n_blocks), 1, n_pos), rank, 1, n_pos, n_blocks);
    ggml_tensor * raw   = ggml_reshape_4d(ctx0, dspine_block_rows(ctx0, ggml_reshape_3d(ctx0, write_raw, n_embd, n_rows, n_blocks), 1, n_pos), n_embd, 1, n_pos, n_blocks);
    ggml_tensor * rms   = ggml_reshape_4d(ctx0, dspine_block_rows(ctx0, ggml_reshape_3d(ctx0, write_rms, 1, n_rows, n_blocks), 1, n_pos), 1, 1, n_pos, n_blocks);

    // the CPU repack kernels take no 4D src1; merging only the position and block dims keeps ne11 = top_k for the GPU kernels
    const int64_t n_mats   = n_pos * n_blocks;
    ggml_tensor * codes_3d = ggml_reshape_3d(ctx0, codes, code_size, top_k, n_mats);
    ggml_tensor * sender   = ggml_reshape_4d(ctx0, build_lora_mm(model.dspine_pred_gate, codes_3d), rank, top_k, n_pos, n_blocks);
    ggml_tensor * value    = ggml_reshape_4d(ctx0, build_lora_mm(model.dspine_prev,      codes_3d), rank, top_k, n_pos, n_blocks);

    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_add(ctx0, ggml_add(ctx0, sender, query), model.dspine_gate_b));
    ggml_tensor * msg  = ggml_reshape_3d(ctx0, ggml_mul(ctx0, gate, value), rank, top_k, n_mats);

    ggml_tensor * delta   = ggml_reshape_4d(ctx0, build_lora_mm(last.dspine_write, msg), n_embd, top_k, n_pos, n_blocks);
    ggml_tensor * refined = ggml_add(ctx0, ggml_mul(ctx0, delta, rms), raw);
    refined = build_norm(refined, model.output_norm, NULL, LLM_NORM_RMS, -1);

    ggml_tensor * head = ggml_get_rows(ctx0, output, ggml_reshape_1d(ctx0, cand_pos, top_k * n_pos * n_blocks));
    head = ggml_reshape_4d(ctx0, head, n_embd, top_k, n_pos, n_blocks);

    // scores[candidate, predecessor] of each draft row, computed like the logits in build_lora_mm
    ggml_tensor * scores = ggml_mul_mat(ctx0, head, build_hadamard_input(output, refined));
    if (output_s) {
        scores = ggml_mul(ctx0, scores, output_s);
    }

    ggml_tensor * packed = ggml_concat(ctx0, ggml_cast(ctx0, cand_pos, GGML_TYPE_F32),
            ggml_reshape_3d(ctx0, scores, top_k * top_k, n_pos, n_blocks), 0);
    packed = ggml_pad_ext(ctx0, packed, 0, n_embd - top_k * (top_k + 1), 1, 0, 0, 0, 0, 0);
    packed = ggml_reshape_2d(ctx0, packed, n_embd, n_tokens);
    cb(packed, "dspine_lattice", -1);
    res->t_h_nextn = packed;
    ggml_build_forward_expand(gf, packed);
}

llama_model_dflash::graph_dspine::graph_dspine(const llm_graph_params & params) : graph_base(params) {
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd || ubatch.embd_row) {
        build_inject();
        return;
    }

    ggml_tensor * inp_tokens = build_inp_tokens();

    // DSpine writes run on whole [anchor, mask...] blocks only, so graph reservation keeps the plain decoder
    const int64_t n_blocks = ubatch.n_seqs_unq;
    const int64_t n_rows   = n_blocks > 0 ? n_tokens / n_blocks : 0;
    if (n_blocks > 0 && n_tokens % n_blocks == 0 && n_rows >= 2 && n_rows <= (int64_t) hparams.dflash_block_size) {
        anchor_codes = build_dspine_anchor_codes(ctx0, model, inp_tokens, n_rows, n_blocks);
    }

    ggml_tensor * cur = build_layers(inp_tokens);

    // DSpine looks up the codebook rows of the head's candidates
    const auto [output, output_s] = get_lm_head(*this, model);
    if (output->ne[1] != model.dspine_codebook->ne[1]) {
        throw std::runtime_error("DSpine draft codebook does not match the target vocabulary");
    }
    // a repacked head (a CPU buffer that is not host) has a backend layout that get_rows cannot read
    ggml_backend_buffer_type_t head_buft = output->buffer ? ggml_backend_buffer_get_type(output->buffer) : nullptr;
    ggml_backend_dev_t         head_dev  = head_buft ? ggml_backend_buft_get_device(head_buft) : nullptr;
    if (head_dev && !ggml_backend_buft_is_host(head_buft) && ggml_backend_dev_type(head_dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        throw std::runtime_error("DSpine draft cannot read the rows of a repacked target head, load the target with --no-repack");
    }

    build_logits(cur);

    if (anchor_codes) {
        build_lattice(output, output_s);
    }
}
