// KV cache state cases
//
#include "test-case.h"

#include "test-log.h"
#include "test-util.h"

#include "common.h"

#include "../../src/llama-ext.h" // llama_model_set_attn_rot

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

//
// reference run
//

// - decode all but the last token, saving the state to disk
// - decode the last token
// - generate n_predict tokens
// the logits it produces are the reference of every case with needs_baseline()
struct case_baseline : ctx_test {
    const char * name() const override { return "baseline"; }

    const char * description() const override {
        return "reference generation over the whole prompt, saves the state before the last token";
    }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_seq_max = 2;

        auto ctx  = mr.ctx(opts);
        auto smpl = mr.sampler();
        if (!ctx) {
            return case_result::fail();
        }

        auto n_past = 0;
        if (!common_prompt_batch_decode(ctx.get(), mr.tokens, (int) mr.tokens.size(), n_past, mr.params.n_batch, mr.params.out_file, true)) {
            TLOG_ERR("%s: failed to decode the prompt\n", __func__);
            return case_result::fail();
        }

        generation_result result;
        if (!generate_tokens(ctx.get(), smpl.get(), n_past, mr.params.n_predict, 0, result)) {
            return case_result::fail();
        }
        if (result.empty()) {
            return case_result::fail();
        }

        mr.baseline      = std::move(result);
        mr.have_baseline = true;

        return case_result::pass();
    }
};

//
// sequences
//

// - decode the same prefix into two sequences
// - remove sequence 0
// - verify that sequence 1 remains unchanged
struct case_seq_rm_isolated : ctx_test {
    const char * name() const override { return "seq_rm"; }

    const char * description() const override {
        return "removing one sequence leaves the state of the other sequences untouched";
    }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_seq_max  = 2;
        opts.kv_unified = 1;

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        const size_t n_tokens = std::min<size_t>(mr.tokens.size(), 128);
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            if (!decode_seq(ctx.get(), mr.tokens, 0, (llama_pos) n_tokens, seq_id, true)) {
                TLOG_ERR("%s: failed to decode the prompt for sequence %d\n", __func__, seq_id);
                return case_result::fail();
            }
        }

        std::vector<uint8_t> before;
        if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, before)) {
            return case_result::fail();
        }

        if (!seq_rm(ctx.get(), 0)) {
            return case_result::fail();
        }

        std::vector<uint8_t> after;
        if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, after)) {
            return case_result::fail();
        }

        CTX_REQUIRE(blobs_equal(before, after), "removing seq 0 changed the state of seq 1");

        return case_result::pass();
    }
};

// - load the state saved by the baseline case
// - replay its last token
// - generate again and compare against the baseline logits
struct case_state_load : ctx_test {
    const char * name() const override { return "state_load"; }

    const char * description() const override {
        return "a saved session replays to the same logits as the run that saved it";
    }

    bool needs_baseline() const override { return true; }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_seq_max = 2;

        auto ctx  = mr.ctx(opts);
        auto smpl = mr.sampler();
        if (!ctx) {
            return case_result::fail();
        }

        int n_past = 0;
        if (!load_state_replay_last(ctx.get(), mr.params.out_file, mr.tokens, n_past)) {
            return case_result::fail();
        }

        if (!generate_tokens_compare(ctx.get(), smpl.get(), n_past, mr.params.n_predict, 0, mr.baseline, NMSE_THRESHOLD)) {
            return case_result::fail();
        }

        return case_result::pass();
    }
};

// - load the state saved by the baseline case and replay its last token
// - copy sequence 0 to sequence 1 over the given io path, clearing the cache in between
// - continue on sequence 1 and compare against the baseline logits
struct case_seq_cp : ctx_test {
    explicit case_seq_cp(bool on_device_) : on_device(on_device_) {}

    const char * name() const override { return on_device ? "cp_d" : "cp_h"; }

    const char * description() const override {
        return on_device ? "copying a sequence over the device io path keeps its state"
                         : "copying a sequence over the host io path keeps its state";
    }

    bool needs_baseline() const override { return true; }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_seq_max = 2;

        auto ctx  = mr.ctx(opts);
        auto smpl = mr.sampler();
        if (!ctx) {
            return case_result::fail();
        }

        const uint32_t flags = on_device ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : LLAMA_STATE_SEQ_FLAGS_NONE;
        TLOG_INF("io path: %s\n", on_device ? "device" : "host");

        int n_past = 0;
        if (!load_state_replay_last(ctx.get(), mr.params.out_file, mr.tokens, n_past)) {
            return case_result::fail();
        }

        {
            std::vector<uint8_t> seq_store;
            if (!get_seq_state(ctx.get(), 0, flags, seq_store)) {
                return case_result::fail();
            }
            TLOG_TRC("%s: seq 0 copied, %zu bytes\n", __func__, seq_store.size());

            llama_memory_clear(llama_get_memory(ctx.get()), true);
            TLOG_TRC("%s: kv cache cleared\n", __func__);

            if (!set_seq_state(ctx.get(), 1, flags, seq_store)) {
                return case_result::fail();
            }
            TLOG_TRC("%s: seq 1 restored, %zu bytes\n", __func__, seq_store.size());
        }

        if (!generate_tokens_compare(ctx.get(), smpl.get(), n_past, mr.params.n_predict, 1, mr.baseline, NMSE_THRESHOLD)) {
            return case_result::fail();
        }

        return case_result::pass();
    }

private:
    bool on_device;
};

// - decode the same prefix on two sequences, interleaving seq 0 cells between the seq 1 cells
// - save the seq 1 state, free the interleaved seq 0 cells, and restore via the given io path
// - the restore destination is non-contiguous: scatter reads are batched per contiguous run
// - save again on the host and compare the two blobs byte for byte
struct case_seq_cp_scatter : ctx_test {
    explicit case_seq_cp_scatter(bool on_device_) : on_device(on_device_) {}

    const char * name() const override { return on_device ? "cp_d_s" : "cp_h_s"; }

    const char * description() const override {
        return "saving and restoring a sequence with a fragmented (scattered) cell range";
    }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_seq_max  = 2;
        opts.kv_unified = 1;

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        const uint32_t flags = on_device ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : LLAMA_STATE_SEQ_FLAGS_NONE;
        TLOG_INF("io path: %s\n", on_device ? "device" : "host");

        // seq 0 cells 0,1,4 interleave the seq 1 cells 2,3,5
        if (!decode_one(ctx.get(), mr.tokens[0], 0, 0) ||
            !decode_one(ctx.get(), mr.tokens[1], 1, 0) ||
            !decode_one(ctx.get(), mr.tokens[0], 0, 1) ||
            !decode_one(ctx.get(), mr.tokens[1], 1, 1) ||
            !decode_one(ctx.get(), mr.tokens[2], 2, 0) ||
            !decode_one(ctx.get(), mr.tokens[2], 2, 1)) {
            TLOG_ERR("%s: failed to build the interleaved state\n", __func__);
            return case_result::fail();
        }

        // host blob: contains the KV data, used for the byte-for-byte comparison
        std::vector<uint8_t> state_before;
        if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, state_before)) {
            return case_result::fail();
        }

        // save via the io path under test
        std::vector<uint8_t> state_save;
        if (!get_seq_state(ctx.get(), 1, flags, state_save)) {
            return case_result::fail();
        }
        TLOG_TRC("%s: seq 1 saved via %s, %zu bytes\n", __func__, on_device ? "device" : "host", state_save.size());

        // free seq 0's cells so the ring is fragmented: the restore destination (seq 1's interleaved cells) stays non-contiguous
        if (!seq_rm(ctx.get(), 0)) {
            return case_result::fail();
        }

        // restore via the io path under test
        if (!set_seq_state(ctx.get(), 1, flags, state_save)) {
            return case_result::fail();
        }
        TLOG_TRC("%s: seq 1 restored via %s, %zu bytes\n", __func__, on_device ? "device" : "host", state_save.size());

        std::vector<uint8_t> state_after;
        if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, state_after)) {
            return case_result::fail();
        }

        // the blob is serialized in sequence cell order, so identical bytes iff the restore wrote the same KV
        CTX_REQUIRE(blobs_equal(state_before, state_after), "the restored KV state is not byte-identical to the saved state");

        return case_result::pass();
    }

private:
    bool on_device;
};

//
// sessions
//

// - save a sequence, erase it, restore it, save it again
// - the two blobs must be identical
// compares blobs rather than generated text: a partially restored cell still decodes to plausible tokens
struct case_state_roundtrip : ctx_test {
    const char * name() const override { return "rt"; }

    const char * description() const override {
        return "save, erase, restore, save again - the two blobs must be identical";
    }

    case_result run(model_run & mr) const override {
        auto ctx = mr.ctx();
        if (!ctx) {
            return case_result::fail();
        }

        common_batch batch = common_batch_get_one(ctx.get(), mr.tokens);
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            TLOG_ERR("%s: failed to decode the prompt\n", __func__);
            return case_result::fail();
        }

        std::vector<uint8_t> blob_a;
        if (!get_seq_state(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE, blob_a)) {
            return case_result::fail();
        }

        if (!seq_rm(ctx.get(), 0)) {
            return case_result::fail();
        }

        if (!set_seq_state(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE, blob_a)) {
            TLOG_ERR("%s: failed to restore seq 0\n", __func__);
            return case_result::fail();
        }

        std::vector<uint8_t> blob_b;
        if (!get_seq_state(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE, blob_b)) {
            return case_result::fail();
        }

        CTX_REQUIRE(blobs_equal(blob_a, blob_b), "the state changed across a save / erase / restore cycle");

        return case_result::pass();
    }
};

// the partial state holds what seq_rm cannot roll back: recurrent states and SWA windows
struct case_state_partial_size : ctx_test {
    const char * name() const override { return "ps"; }

    const char * description() const override {
        return "the partial state is not larger than the full state and, without SWA, does not grow with the tokens";
    }

    case_result run(model_run & mr) const override {
        auto ctx = mr.ctx();
        if (!ctx) {
            return case_result::fail();
        }

        const llama_pos n_tokens = mr.tokens.size();
        const llama_pos n_half   = n_tokens / 2;

        if (!decode_seq(ctx.get(), mr.tokens, 0, n_half, 0, true)) {
            return case_result::fail();
        }

        const size_t partial_a = llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        const size_t full_a    = llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE);

        if (!decode_seq(ctx.get(), mr.tokens, n_half, n_tokens, 0, true)) {
            return case_result::fail();
        }

        const size_t partial_b = llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        const size_t full_b    = llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE);

        TLOG_INF("%s: partial %zu -> %zu, full %zu -> %zu\n", __func__, partial_a, partial_b, full_a, full_b);

        CTX_REQUIRE(partial_a <= full_a && partial_b <= full_b, "the partial state is larger than the full state");

        if (llama_model_n_swa(mr.model) == 0) {
            CTX_REQUIRE(partial_a == partial_b, "the partial state grows with the tokens");
        }

        return case_result::pass();
    }
};

// a failed restore must leave the sequence empty and must not change the logits of other sequences
struct case_state_restore_failure : ctx_test {
    const char * name() const override { return "rf"; }

    const char * description() const override {
        return "restoring a corrupted state fails and leaves the other sequences untouched";
    }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_seq_max  = 4;
        opts.kv_unified = 1;

        // without flash attention, corrupted data left behind by the restore shows up as NaN logits on the other sequences
        opts.fa = LLAMA_FLASH_ATTN_TYPE_DISABLED;

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        llama_memory_t mem = llama_get_memory(ctx.get());
        if (mem == nullptr) {
            TLOG_INF("no memory to test\n");
            return case_result::pass();
        }

        const auto decode = [&](const llama_tokens & inp, llama_seq_id seq_id, std::vector<float> * logits_out) {
            if (!decode_tokens(ctx.get(), inp, seq_id)) {
                TLOG_ERR("%s: failed to decode on sequence %d\n", __func__, seq_id);
                return false;
            }

            if (logits_out && !get_current_logits(ctx.get(), *logits_out)) {
                TLOG_ERR("%s: failed to get logits\n", __func__);
                return false;
            }

            return true;
        };

        const llama_tokens tokens_save  (mr.tokens.begin(), mr.tokens.begin() + std::min<size_t>(24, mr.tokens.size()));
        const llama_tokens tokens_verify(mr.tokens.end() - std::min<size_t>(8, mr.tokens.size()), mr.tokens.end());

        // the cases share a working directory, so the state file is named after the model
        const std::string path = "state-restore-failure." + mr.name + ".tmp.bin";

        llama_memory_clear(mem, true);

        std::vector<float> baseline;
        if (!decode(tokens_verify, 1, &baseline)) {
            return case_result::fail();
        }

        const std::vector<std::pair<const char *, std::function<bool()>>> sub_cases = {
            { "buffer", [&]() {
                std::vector<uint8_t> state(llama_state_seq_get_size(ctx.get(), 0));
                GGML_ASSERT(llama_state_seq_get_data(ctx.get(), state.data(), state.size(), 0) == state.size());
                llama_memory_seq_rm(mem, 0, -1, -1);

                if (!corrupt_state(state)) {
                    return false;
                }

                return llama_state_seq_set_data(ctx.get(), state.data(), state.size(), 0) == 0;
            }},
            { "file", [&]() {
                GGML_ASSERT(llama_state_seq_save_file(ctx.get(), path.c_str(), 0, tokens_save.data(), tokens_save.size()) > 0);
                llama_memory_seq_rm(mem, 0, -1, -1);

                std::vector<uint8_t> data;
                {
                    std::ifstream f(path, std::ios::binary);
                    data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
                }

                if (!corrupt_state(data)) {
                    std::remove(path.c_str());
                    return false;
                }

                {
                    std::ofstream f(path, std::ios::binary);
                    f.write((const char *) data.data(), data.size());
                }

                llama_tokens tokens_out(tokens_save.size());
                size_t n_token_count = 0;
                const size_t nread = llama_state_seq_load_file(ctx.get(), path.c_str(), 0, tokens_out.data(), tokens_out.size(), &n_token_count);
                std::remove(path.c_str());

                return nread == 0;
            }},
        };

        for (const auto & [name, restore_failed] : sub_cases) {
            llama_memory_clear(mem, true);

            if (!decode(tokens_save, 0, nullptr)) {
                return case_result::fail();
            }

            CTX_REQUIRE(restore_failed(), std::string(name) + ": restoring a corrupted state did not fail");

            CTX_REQUIRE(llama_memory_seq_pos_max(mem, 0) == -1,
                    std::string(name) + ": the sequence is not empty after a failed restore");

            std::vector<float> logits;
            if (!decode(tokens_verify, 1, &logits)) {
                return case_result::fail();
            }

            const logits_diff diff = compare_logits(baseline.data(), logits.data(), logits.size());

            CTX_REQUIRE(diff.n_nan == 0 && diff.diff_max <= 1e-6f,
                    std::string(name) + ": the logits changed after a failed restore (max diff = " +
                        std::to_string(diff.diff_max) + ", nan = " + std::to_string(diff.n_nan) + ")");

            TLOG_TRC("%s: %s: logits match (max diff = %g)\n", __func__, name, (double) diff.diff_max);
        }

        return case_result::pass();
    }
};

// a KV state saved with attention rotation enabled must restore only into a context with the
// same setting
// note: rotation is only active for quantized KV caches with a head size that is a multiple of
//       64, for other models the restore into the rotation-disabled context is valid and the
//       test passes vacuously
struct case_state_rotation : ctx_test {
    const char * name() const override { return "rot"; }

    const char * description() const override {
        return "a state saved with attention rotation restores only into a matching context";
    }

    case_result run(model_run & mr) const override {
        const auto make_context = [&](ggml_type type_k, ggml_type type_v, bool disable_rotation) {
            // a setting of the model, not of the process - a sweep that runs several models at a
            //   time must not let this leak into the contexts of another model
            llama_model_set_attn_rot(mr.model, !disable_rotation);

            ctx_opts opts;
            opts.n_ctx    = 32;
            opts.n_batch  = 1;
            opts.n_ubatch = 1;
            opts.type_k   = type_k;
            opts.type_v   = type_v;
            opts.fa       = LLAMA_FLASH_ATTN_TYPE_ENABLED;

            // probing whether this KV type is supported is expected to fail for some models
            opts.quiet = true;

            return mr.ctx(opts);
        };

        std::vector<std::pair<ggml_type, ggml_type>> type_pairs;
        for (const auto & types : { std::pair{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0} }) {
            if (make_context(types.first, types.second, false)) {
                type_pairs.push_back(types);
            }
        }
        if (type_pairs.empty()) {
            return case_result::skip("no supported quantized KV cache type combination");
        }

        bool success = true;
        for (const auto & types : type_pairs) {
            auto src = make_context(types.first, types.second, false);
            if (!src) {
                TLOG_ERR("%s: failed to create the source context\n", __func__);
                success = false;
                break;
            }

            llama_token token = 0;
            if (llama_decode(src.get(), llama_batch_get_one(&token, 1))) {
                TLOG_ERR("%s: failed to decode a token\n", __func__);
                success = false;
                break;
            }

            std::vector<uint8_t> state;
            if (!get_seq_state(src.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE, state)) {
                if (state.empty()) {
                    continue; // no KV state to test
                }
                success = false;
                break;
            }

            auto matching = make_context(types.first, types.second, false);
            if (!matching || !set_seq_state(matching.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE, state)) {
                TLOG_ERR("%s: failed to restore a matching rotation\n", __func__);
                success = false;
                break;
            }

            auto mismatched = make_context(types.first, types.second, true);
            if (!mismatched) {
                TLOG_ERR("%s: failed to create a mismatched rotation context\n", __func__);
                success = false;
                break;
            }
            // a model that does not use attention rotation accepts the mismatched restore
            if (set_seq_state(mismatched.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE, state)) {
                TLOG_TRC("%s: state restored into rotation-disabled context, model does not use attention rotation\n", __func__);
            }
        }

        llama_model_set_attn_rot(mr.model, true); // the default, the cases that follow rely on it

        if (!success) {
            return case_result::fail();
        }

        return case_result::pass();
    }
};

// restore a sequence into a fragmented cache: the slots freed by the removed sequence are
// scattered between live cells of the other sequences, so the restore cannot claim one
// contiguous block
// the restored state must come back byte for byte, and the sequence must continue to behave
// exactly as the same sequence in a context where it was never removed
// ref: https://github.com/ggml-org/llama.cpp/issues/17527
struct case_state_restore_fragmented : ctx_test {
    const char * name() const override { return "frag"; }

    const char * description() const override {
        return "restore a sequence into a cache fragmented by the removal of another sequence";
    }

    std::string skip_reason(model_run & mr) const override {
        if (arch_reserves_single_seq(mr.model)) {
            return "the interleaved batch is a multi-seq graph";
        }

        return ctx_test::skip_reason(mr);
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_seq_max  = 3;
        opts.kv_unified = 1;
        opts.n_batch    = 3*70;
        opts.n_ubatch   = 3*70; // the interleaved prompt is a single multi-seq batch

        // ctx removes and restores seq 1, ctx_keep keeps it untouched as the reference
        llama_context_ptr ctx      = mr.ctx(opts);
        llama_context_ptr ctx_keep = mr.ctx(opts);
        if (!ctx || !ctx_keep) {
            return case_result::fail();
        }

        const llama_pos n_tokens = (llama_pos) std::min<size_t>(70, mr.tokens.size());

        // interleave the 3 sequences: 01201230123... - seq 1 owns every third cell
        const auto decode_interleaved = [&](llama_context * cur) {
            common_batch batch(cur);
            for (llama_pos i = 0; i < n_tokens; i++) {
                for (llama_seq_id s = 0; s < 3; ++s) {
                    batch.add(mr.tokens[i], i, s, false);
                }
            }
            batch.set_output(batch.size() - 1, true);

            return llama_process(cur, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
        };

        if (!decode_interleaved(ctx.get()) || !decode_interleaved(ctx_keep.get())) {
            TLOG_ERR("%s: failed to decode the interleaved prompt\n", __func__);
            return case_result::fail();
        }

        TLOG_INF("%s: processed the prompt on seq 0, 1, 2 (%d tokens each)\n", __func__, n_tokens);

        std::vector<uint8_t> state_before;
        if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, state_before)) {
            return case_result::fail();
        }

        // clearing seq 1 leaves holes where its cells were - no contiguous block large enough
        // for the seq 1 state is left in the cache
        llama_memory_seq_rm(llama_get_memory(ctx.get()), 1, -1, -1);

        if (!set_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, state_before)) {
            TLOG_ERR("%s: failed to restore the seq state into a fragmented cache\n", __func__);
            return case_result::fail();
        }

        TLOG_INF("%s: restored seq 1 into a fragmented cache, %zu bytes\n", __func__, state_before.size());

        // the restore has to be lossless - the cells the sequence ends up in are irrelevant, its
        //   state has to come back exactly as it was saved
        std::vector<uint8_t> state_after;
        if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, state_after)) {
            return case_result::fail();
        }

        CTX_REQUIRE(blobs_equal(state_before, state_after), "restoring into a fragmented cache changed the state of seq 1");

        // the restored sequence must continue exactly as the untouched one - decode the same token
        //   on seq 1 in both contexts and compare the logits they produce
        const llama_token token = gen_token(n_vocab, 1, n_tokens);

        if (!decode_one(ctx.get(), token, n_tokens, 1) || !decode_one(ctx_keep.get(), token, n_tokens, 1)) {
            TLOG_ERR("%s: failed to decode with the restored state\n", __func__);
            return case_result::fail();
        }

        const float * logits      = llama_get_logits(ctx.get());
        const float * logits_keep = llama_get_logits(ctx_keep.get());
        if (!logits || !logits_keep) {
            TLOG_ERR("%s: missing logits after the restore\n", __func__);
            return case_result::fail();
        }

        const logits_diff diff = compare_logits(logits, logits_keep, n_vocab);

        CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD,
                "the logits of the restored seq 1 differ from the reference: nmse " + std::to_string(diff.nmse()));

        TLOG_INF("%s: restored seq 1 continued identically to the reference, nmse %g\n", __func__, diff.nmse());

        return case_result::pass();
    }
};

//
// no_alloc
//

// - load a second instance of the model with no_alloc, so the tensor data stays on disk
// - create a context from it, which reserves the compute graphs over the dummy buffers
// - the memory use must still be reported, and nothing real may be allocated for the memory
struct case_noalloc : ctx_test {
    const char * name() const override { return "noalloc"; }

    const char * description() const override {
        return "with no_alloc the memory use is simulated: non-zero breakdown, null buffer bases";
    }

    case_result run(model_run & mr) const override {
        const std::string & path = mr.params.model.path;
        CTX_REQUIRE(!path.empty(), "the model path is not known");

        llama_model_params mparams = llama_model_default_params();
        mparams.no_alloc  = true;
        mparams.load_mode = LLAMA_LOAD_MODE_NONE;

        llama_model_ptr model(llama_model_load_from_file(path.c_str(), mparams));
        CTX_REQUIRE(model, "failed to load the model with no_alloc");

        auto ctx = init_ctx(model.get(), common_context_params_to_llama(mr.params), 0);
        CTX_REQUIRE(ctx, "failed to create the context with no_alloc");

        size_t n_ctx_mem = 0;
        for (const auto & [_, data] : llama_get_memory_breakdown(ctx.get())) {
            n_ctx_mem += data.context;
        }

        // models without a memory module (BERT, diffusion) have a null memory
        if (llama_get_memory(ctx.get()) != nullptr) {
            CTX_REQUIRE(n_ctx_mem > 0, "the simulated memory breakdown is zero");
        } else {
            CTX_REQUIRE(n_ctx_mem == 0, "a model without memory reported context memory");
        }

        for (ggml_backend_buffer_t buf : llama_memory_get_buffers(llama_get_memory(ctx.get()))) {
            CTX_REQUIRE(ggml_backend_buffer_get_base(buf) == nullptr, "a memory buffer was really allocated");
        }

        return case_result::pass();
    }
};

std::vector<ctx_test_ptr> make_kv_cases() {
    std::vector<ctx_test_ptr> cases;

    cases.push_back(std::make_unique<case_baseline>());
    cases.push_back(std::make_unique<case_seq_rm_isolated>());
    cases.push_back(std::make_unique<case_state_load>());
    cases.push_back(std::make_unique<case_seq_cp>(false));
    cases.push_back(std::make_unique<case_seq_cp>(true));
    cases.push_back(std::make_unique<case_seq_cp_scatter>(false));
    cases.push_back(std::make_unique<case_seq_cp_scatter>(true));
    cases.push_back(std::make_unique<case_state_roundtrip>());
    cases.push_back(std::make_unique<case_state_partial_size>());
    cases.push_back(std::make_unique<case_state_restore_failure>());
    cases.push_back(std::make_unique<case_state_rotation>());
    cases.push_back(std::make_unique<case_state_restore_fragmented>());
    cases.push_back(std::make_unique<case_noalloc>());

    return cases;
}
