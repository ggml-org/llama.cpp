#include "../tools/mtmd/mtmd-session.h"
#include "../tools/mtmd/mtmd-workspace.h"
#include "../tools/mtmd/mtmd-projector-storage.h"
#include "../src/llama-context-memory.h"
#include "testing.h"

#include <stdexcept>
#include <cmath>
#include <cstring>
#include <functional>
#if !defined(_WIN32) || !defined(LLAMA_SHARED)
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-kv-stream-model.h"
#include "../src/llama-io.h"
#include "../ggml/src/ggml-backend-impl.h"

struct vision_restore_fault {
    using factory = ggml_backend_buffer_t (*)(ggml_backend_buffer_t,size_t,size_t);
    inline static vision_restore_fault * active = nullptr;
    ggml_backend_buffer_t parent;
    factory original;
    size_t calls = 0;
    explicit vision_restore_fault(ggml_backend_buffer_t parent) : parent(parent),original(parent->view_buffer) {}
    void arm() {
        GGML_ASSERT(!active); active = this;
        parent->view_buffer = [](ggml_backend_buffer_t parent,size_t offset,size_t size) {
            if (++active->calls == 1) return ggml_backend_buffer_t(nullptr);
            return active->original(parent,offset,size);
        };
    }
    ~vision_restore_fault() {
        parent->view_buffer = original;
        active = nullptr;
    }
};

struct session_recurrent_snapshot : llama_io_write_i {
    std::vector<uint8_t> bytes;
    void write(const void * data, size_t count) override {
        const auto * source = static_cast<const uint8_t *>(data);
        bytes.insert(bytes.end(), source, source + count);
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t count) override {
        const size_t first = bytes.size();
        bytes.resize(first + count);
        ggml_backend_tensor_get(tensor, bytes.data() + first, offset, count);
    }
    size_t n_bytes() override { return bytes.size(); }
};
#endif

struct session_fixture : mtmd_session_backend {
    mtmd::input_chunks_ptr input{mtmd_test_create_input_chunks()};
    const mtmd_input_chunk * text = mtmd_input_chunks_get(input.get(), 0);
    const mtmd_input_chunk * image = mtmd_input_chunks_get(input.get(), 1);
    mtmd::input_chunk_ptr second{mtmd_input_chunk_copy(image)};
    std::vector<const mtmd_input_chunk *> prompt{text, image, text, second.get(), text};
    std::vector<std::string> calls;
    std::vector<mtmd_embedding_view> held;
    bool batchable = true;
    bool bad_outputs = false;
    int32_t encode_error = 0;
    int32_t prefill_error = 0;
    bool throw_from_encode = false;
    bool throw_from_prefill = false;
    size_t output_width = 4;
    std::function<void()> on_validate;
    std::function<void()> on_prefill;
    mtmd_session_plan * cancel_from_encode = nullptr;
    size_t tokens = 10;

    int32_t validate_batch(const std::vector<const mtmd_input_chunk *> & entries,
            const mtmd_input_chunk * chunk) override {
        if (on_validate) on_validate();
        return mtmd_batch_validate_chunk(entries, chunk, batchable, SIZE_MAX);
    }
    int32_t encode(const std::vector<const mtmd_input_chunk *> & chunks,
            std::vector<mtmd_embedding_view> & output) override {
        calls.push_back("encode" + std::to_string(chunks.size()));
        if (throw_from_encode) throw std::runtime_error("injected encode failure");
        if (encode_error) return encode_error;
        std::vector<mtmd_embedding_chunk> ranges;
        size_t count = 0;
        for (const auto * chunk : chunks) {
            size_t rows = mtmd_input_chunk_get_n_tokens(chunk);
            count += rows;
            ranges.push_back({chunk, rows});
        }
        std::vector<float> values(count * output_width, 3.f);
        mtmd_embedding_output owner;
        if (!owner.publish(ranges, output_width, values)) return -1;
        for (const auto * chunk : chunks) {
            mtmd_embedding_view view;
            if (!owner.acquire(chunk, view)) return -1;
            output.push_back(view);
        }
        if (bad_outputs) output.clear();
        if (cancel_from_encode) cancel_from_encode->cancel();
        return 0;
    }
    int32_t prefill(const mtmd_input_chunk * chunk, const mtmd_embedding_view * embedding) override {
        if (throw_from_prefill) throw std::runtime_error("injected prefill failure");
        if (embedding) {
            if (embedding->data()[0] != 3.f) throw std::runtime_error("invalid embedding");
            calls.push_back(chunk == image ? "image1" : "image2");
            held.push_back(*embedding);
        } else calls.push_back("text");
        if (prefill_error) return prefill_error;
        if (on_prefill) on_prefill();
        tokens += mtmd_input_chunk_get_n_tokens(chunk);
        return 0;
    }
};

static void finish(testing & t, mtmd_session_plan & plan, session_fixture & f) {
    while (plan.status() == mtmd_session_status::ready) {
        if (!t.assert_equal(0, plan.advance(f))) return;
    }
    t.assert_true(plan.status() == mtmd_session_status::complete);
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr, * projector_path = nullptr;
    bool cuda = false;
    size_t stream_pool_mib = 0;
    size_t shared_budget_mib = 0;
    size_t prefix_repetitions = 6000;
    size_t continuation_tokens = 16;
    bool resident_control = false;
    bool shared_vision = false;
    bool phase_arena = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--cuda")) cuda = true;
        else if (!std::strcmp(argv[i], "--stream-pool-mib") && i + 1 < argc) stream_pool_mib = std::stoull(argv[++i]);
        else if (!std::strcmp(argv[i], "--shared-budget-mib") && i + 1 < argc) shared_budget_mib = std::stoull(argv[++i]);
        else if (!std::strcmp(argv[i], "--resident-control")) resident_control = true;
        else if (!std::strcmp(argv[i], "--shared-vision")) shared_vision = true;
        else if (!std::strcmp(argv[i], "--phase-vision")) phase_arena = shared_vision = true;
        else if (!std::strcmp(argv[i], "--prefix-repetitions") && i + 1 < argc) prefix_repetitions = std::stoull(argv[++i]);
        else if (!std::strcmp(argv[i], "--continuation-tokens") && i + 1 < argc) continuation_tokens = std::stoull(argv[++i]);
        else if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!std::strcmp(argv[i], "--mmproj") && i + 1 < argc) projector_path = argv[++i];
        else return 2;
    }
    const bool adaptive = stream_pool_mib || shared_budget_mib;
    const bool suspended_vision = shared_vision && shared_budget_mib;
    if ((adaptive && !cuda) || (stream_pool_mib && shared_budget_mib) || prefix_repetitions > 6000 ||
            !continuation_tokens || continuation_tokens > 16) return 2;
    const auto evaluate = phase_arena ? mtmd_session_eval_chunks_arena :
        shared_vision ? mtmd_session_eval_chunks_shared : mtmd_session_eval_chunks;
    testing t;
    t.test("future_images_are_encoded_together_but_prefilled_in_prompt_order", [](testing & t) {
        session_fixture f;
        mtmd_session_plan plan;
        if (!t.assert_true(plan.prepare(f.prompt, 4, f))) return;
        t.assert_true(f.calls.empty());
        t.assert_equal(size_t(6), plan.steps().size());
        t.assert_true(plan.steps()[1].phase == mtmd_session_phase::vision_encode);
        finish(t, plan, f);
        t.assert_true(f.calls == std::vector<std::string>({"text", "encode2", "image1", "text", "image2", "text"}));
        t.assert_equal(size_t(537), f.tokens);
        const auto count = f.calls.size();
        t.assert_equal(0, plan.advance(f));
        t.assert_equal(count, f.calls.size());
    });
    t.test("incompatible_or_unbatchable_images_get_separate_encode_steps", [](testing & t) {
        session_fixture f;
        f.batchable = false;
        mtmd_session_plan plan;
        if (!t.assert_true(plan.prepare(f.prompt, 4, f))) return;
        t.assert_equal(size_t(7), plan.steps().size());
        finish(t, plan, f);
        t.assert_true(f.calls == std::vector<std::string>({"text", "encode1", "image1", "text", "encode1", "image2", "text"}));
    });
    t.test("incompatible_shapes_are_not_skipped_to_batch_later_images", [](testing & t) {
        session_fixture f;
        size_t bytes = 0;
        if (!t.assert_equal(0, mtmd_input_chunk_save(f.second.get(), nullptr, 0, &bytes))) return;
        std::vector<char> encoded(bytes);
        if (!t.assert_equal(0, mtmd_input_chunk_save(f.second.get(), encoded.data(), bytes, nullptr))) return;
        // Version 1 starts image geometry after version, type, empty text count, and image flag.
        const size_t nx_offset = sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint8_t);
        const uint32_t nx = 8;
        std::memcpy(encoded.data() + nx_offset, &nx, sizeof(nx));
        f.second.reset(mtmd_input_chunk_load(encoded.data(), encoded.size()));
        if (!t.assert_true(bool(f.second))) return;
        f.prompt[3] = f.second.get();
        f.prompt.push_back(f.image);
        mtmd_session_plan plan;
        if (!t.assert_true(plan.prepare(f.prompt, 4, f))) return;
        finish(t, plan, f);
        t.assert_true(f.calls == std::vector<std::string>({"text", "encode1", "image1", "text",
            "encode1", "image2", "text", "encode1", "image1"}));
    });
    t.test("followup_plan_preserves_existing_target_progress", [](testing & t) {
        session_fixture f;
        mtmd_session_plan plan;
        if (!t.assert_true(plan.prepare({f.text, f.image, f.text}, 4, f))) return;
        finish(t, plan, f);
        const auto before = f.tokens;
        f.calls.clear();
        if (!t.assert_true(plan.prepare({f.text, f.second.get(), f.text}, 4, f))) return;
        finish(t, plan, f);
        t.assert_equal(before + 266, f.tokens);
        t.assert_true(f.calls == std::vector<std::string>({"text", "encode1", "image2", "text"}));
        t.assert_equal(3.f, f.held[0].data()[0]);
    });
    t.test("encode_failure_never_runs_that_image_or_later_text", [](testing & t) {
        session_fixture f;
        mtmd_session_plan plan;
        f.encode_error = 23;
        if (!t.assert_true(plan.prepare(f.prompt, 4, f))) return;
        if (!t.assert_equal(0, plan.advance(f))) return;
        t.assert_equal(23, plan.advance(f));
        t.assert_true(plan.status() == mtmd_session_status::failed);
        t.assert_true(f.calls == std::vector<std::string>({"text", "encode2"}));
        const auto count = f.calls.size();
        t.assert_true(plan.advance(f) != 0);
        t.assert_equal(count, f.calls.size());
        t.assert_true(!plan.prepare(f.prompt, 4, f));
    });
    t.test("partial_or_invalid_encoder_outputs_fail_before_prefill", [](testing & t) {
        session_fixture f;
        mtmd_session_plan plan;
        f.bad_outputs = true;
        if (!t.assert_true(plan.prepare(f.prompt, 4, f))) return;
        if (!t.assert_equal(0, plan.advance(f))) return;
        t.assert_true(plan.advance(f) != 0);
        t.assert_true(plan.status() == mtmd_session_status::failed);
        t.assert_true(f.held.empty());
    });
    t.test("cancellation_releases_future_embeddings_without_advancing_text", [](testing & t) {
        session_fixture f;
        mtmd_session_plan plan;
        if (!t.assert_true(plan.prepare(f.prompt, 4, f))) return;
        for (int i = 0; i < 3; ++i) if (!t.assert_equal(0, plan.advance(f))) return;
        plan.cancel();
        t.assert_true(plan.status() == mtmd_session_status::cancelled);
        const auto count = f.calls.size();
        t.assert_true(plan.advance(f) != 0);
        t.assert_equal(count, f.calls.size());
        t.assert_equal(3.f, f.held[0].data()[0]);
        mtmd_session_plan another;
        if (!t.assert_true(another.prepare(f.prompt, 4, f))) return;
        f.cancel_from_encode = &another;
        if (!t.assert_equal(0, another.advance(f))) return;
        t.assert_true(another.advance(f) != 0);
        t.assert_true(another.status() == mtmd_session_status::cancelled);
    });
    t.test("invalid_inputs_and_speculation_are_rejected_without_execution", [](testing & t) {
        session_fixture f;
        mtmd_session_plan plan;
        t.assert_true(!plan.prepare({nullptr}, 4, f));
        t.assert_true(!plan.prepare(f.prompt, 0, f));
        t.assert_true(!plan.prepare(f.prompt, SIZE_MAX, f));
        t.assert_true(!plan.prepare(f.prompt, 4, f, true));
        t.assert_true(f.calls.empty());
        if (!t.assert_true(plan.prepare(f.prompt, 4, f))) return;
        const auto count = plan.steps().size();
        t.assert_true(!plan.prepare({f.text}, 4, f));
        t.assert_equal(count, plan.steps().size());
        plan.cancel();
        t.assert_true(plan.prepare({}, 0, f));
        t.assert_true(plan.status() == mtmd_session_status::complete);
    });
    t.test("another_backend_cannot_consume_the_prepared_session", [](testing & t) {
        session_fixture f, other;
        mtmd_session_plan plan;
        if (!t.assert_true(plan.prepare(f.prompt, 4, f))) return;
        t.assert_true(plan.advance(other) != 0);
        t.assert_true(other.calls.empty());
        t.assert_equal(size_t(0), plan.next_step());
        finish(t, plan, f);
        plan.cancel();
        t.assert_true(plan.status() == mtmd_session_status::complete);
    });
    t.test("wrong_embedding_width_and_backend_exceptions_are_fail_closed", [](testing & t) {
        for (int mode = 0; mode < 3; ++mode) {
            session_fixture f;
            mtmd_session_plan plan;
            f.output_width = mode == 0 ? 8 : 4;
            f.throw_from_encode = mode == 1;
            if (!t.assert_true(plan.prepare(f.prompt, 4, f))) return;
            if (!t.assert_equal(0, plan.advance(f))) return;
            if (mode == 2) {
                if (!t.assert_equal(0, plan.advance(f))) return;
                f.throw_from_prefill = true;
            }
            t.assert_true(plan.advance(f) != 0);
            t.assert_true(plan.status() == mtmd_session_status::failed);
            t.assert_true(f.held.empty());
        }
    });
    t.test("reentry_and_cancellation_during_callbacks_preserve_ownership", [](testing & t) {
        session_fixture f;
        mtmd_session_plan plan;
        if (!t.assert_true(plan.prepare(f.prompt, 4, f))) return;
        bool reentered = false;
        f.on_prefill = [&] {
            reentered = plan.advance(f) == 0 || plan.prepare(f.prompt, 4, f);
        };
        if (!t.assert_equal(0, plan.advance(f))) return;
        t.assert_true(!reentered);
        if (!t.assert_equal(0, plan.advance(f))) return;
        f.on_prefill = [&] {
            plan.cancel();
            t.assert_equal(3.f, f.held.back().data()[0]);
        };
        t.assert_true(plan.advance(f) != 0);
        t.assert_true(plan.status() == mtmd_session_status::cancelled);
        t.assert_true(f.calls == std::vector<std::string>({"text", "encode2", "image1"}));
        f.on_prefill = {};
        f.calls.clear();
        if (!t.assert_true(plan.prepare({f.text}, 0, f))) return;
        finish(t, plan, f);
        const auto tokens = f.tokens;
        f.calls.clear();
        f.on_validate = [&] { plan.cancel(); };
        t.assert_true(!plan.prepare(f.prompt, 4, f));
        t.assert_true(plan.status() == mtmd_session_status::cancelled);
        t.assert_true(f.calls.empty());
        t.assert_equal(tokens, f.tokens);
    });
    t.test("pure_text_and_empty_plans_need_no_projector_work", [](testing & t) {
        session_fixture f;
        mtmd_session_plan plan;
        t.assert_true(plan.advance(f) != 0);
        if (!t.assert_true(plan.prepare({f.text, f.text}, 0, f))) return;
        finish(t, plan, f);
        t.assert_true(f.calls == std::vector<std::string>({"text", "text"}));
        f.calls.clear();
        if (!t.assert_true(plan.prepare({}, 0, f))) return;
        t.assert_true(plan.status() == mtmd_session_status::complete);
        t.assert_equal(0, plan.advance(f));
        t.assert_true(f.calls.empty());
        llama_pos unchanged = 37;
        t.assert_true(mtmd_session_eval_chunks(nullptr, nullptr, nullptr, 0, 0, 0, false, &unchanged) != 0);
        t.assert_equal(llama_pos(37), unchanged);
    });
    if (model_path && projector_path) t.test("real_text_image_sessions_and_followups_match_ordinary_helpers", [&](testing & t) {
        auto log = [](ggml_log_level level, const char * text, void *) {
            if (level == GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
        };
        llama_log_set(log, nullptr);
        mtmd_helper_log_set(log, nullptr);
        ggml_backend_load_all();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = cuda ? 999 : 0;
        model_params.load_mtp = false;
        llama_model_ptr model(llama_model_load_from_file(model_path, model_params));
        if (!t.assert_true(bool(model))) return;
        auto context_params = llama_context_default_params();
        context_params.n_ctx = adaptive ? 8192 : 1024;
        context_params.n_batch = context_params.n_ubatch = 64;
        context_params.n_seq_max = 1;
        context_params.n_rs_seq = 0;
        context_params.n_threads = context_params.n_threads_batch = 4;
        context_params.type_k = GGML_TYPE_Q8_0;
        context_params.type_v = GGML_TYPE_Q4_0;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context_ptr target(llama_init_from_model(model.get(), context_params));
        if (!t.assert_true(bool(target))) return;
        auto params = mtmd_context_params_default();
        struct vision_probe {
            llama_context * target = nullptr;
            size_t paused_calls = 0;
            size_t phase_bytes = 0;
            bool fail_once = false;
            std::function<void()> on_encode;
        } probe;
        params.cb_eval_user_data = &probe;
        params.cb_eval = [](ggml_tensor *, bool ask, void * data) {
            auto & probe = *static_cast<vision_probe *>(data);
            auto * owner = probe.target ? llama_context_compute_memory(probe.target) : nullptr;
            if (ask && owner && owner->kv_device_suspended()) ++probe.paused_calls;
            if (ask && owner && owner->kv_device_suspended()) {
                llama_context_memory_diagnostics d;
                if (owner->diagnostics(d)) probe.phase_bytes = std::max(probe.phase_bytes,d.borrowed_phase_bytes);
            }
            if (ask && probe.on_encode) probe.on_encode();
            if (ask && probe.fail_once) {
                probe.fail_once = false;
                throw std::runtime_error("injected vision execution interruption");
            }
            return false;
        };
        params.use_gpu = cuda; params.warmup = false; params.n_threads = 4;
        params.image_min_tokens = 64; params.image_max_tokens = 256;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        mtmd::context_ptr vision(mtmd_init_from_file(projector_path, model.get(), params));
        if (!t.assert_true(bool(vision))) return;
        const uint32_t image_size = adaptive ? 512 : 256;
        std::vector<uint8_t> pixels(size_t(image_size) * image_size * 3);
        for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = uint8_t((i * 7 + i / 101) % 256);
        mtmd::bitmap_ptr first(mtmd_bitmap_init(image_size, image_size, pixels.data()));
        for (auto & value : pixels) value = 255 - value;
        mtmd::bitmap_ptr second(mtmd_bitmap_init(image_size, image_size, pixels.data()));
        const auto marker = std::string(mtmd_get_marker(vision.get()));
        auto tokenize = [&](const std::string & prompt, std::vector<const mtmd_bitmap *> images) {
            mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
            const mtmd_input_text text{prompt.data(), prompt.size(), true, true};
            if (mtmd_tokenize(vision.get(), chunks.get(), &text, images.data(), images.size()) != 0) chunks.reset();
            return chunks;
        };
        std::string background;
        if (adaptive) for (size_t i = 0; i < prefix_repetitions; ++i) background += " green";
        auto prefix = tokenize(background + " Describe the patterns: " + marker + " Compare with: " + marker + " One-word summary: ",
            {first.get(), second.get()});
        auto followup = tokenize("Now consider this image: " + marker + " Describe one difference: ", {second.get()});
        if (!t.assert_true(bool(prefix) && bool(followup))) return;
        auto logits = [&]() {
            auto * data = llama_get_logits_ith(target.get(), -1);
            const size_t count = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
            return std::vector<float>(data, data + count);
        };
        auto snapshot = [&]() {
            std::vector<uint8_t> bytes(llama_state_get_size(target.get()));
            const size_t count = llama_state_get_data(target.get(), bytes.data(), bytes.size());
            bytes.resize(count);
            return bytes;
        };
        auto compare = [&](const std::vector<float> & expected, const char * label) {
            const auto actual = logits();
            float error = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) return false;
                error = std::max(error, std::abs(actual[i] - expected[i]));
            }
            t.out << label << " max_logit_error=" << error << '\n';
            return error == 0;
        };
        llama_pos baseline_position = 0;
        if (!t.assert_equal(0, mtmd_helper_eval_chunks(vision.get(), target.get(), prefix.get(), 0, 0, 32, true,
                &baseline_position))) return;
        auto expected_prefix = logits();
#if !defined(_WIN32) || !defined(LLAMA_SHARED)
        session_recurrent_snapshot initial_recurrent;
        static_cast<llama_memory_hybrid *>(llama_get_memory(target.get()))->get_mem_recr()->state_write(initial_recurrent, 0, 0);
#endif
        auto baseline_state = snapshot();
        if (!t.assert_true(!baseline_state.empty())) return;
        const auto prefix_position = baseline_position;
        if (!t.assert_equal(0, mtmd_helper_eval_chunks(vision.get(), target.get(), followup.get(), baseline_position, 0, 32, true,
                &baseline_position))) return;
        auto expected_followup = logits();
#if !defined(_WIN32) || !defined(LLAMA_SHARED)
        session_recurrent_snapshot followup_recurrent;
        static_cast<llama_memory_hybrid *>(llama_get_memory(target.get()))->get_mem_recr()->state_write(followup_recurrent, 0, 0);
#endif
        auto generate = [&](llama_pos position) {
            std::vector<llama_token> tokens;
            for (size_t i = 0; i < continuation_tokens; ++i) {
                auto values = logits();
                llama_token token = llama_token(std::max_element(values.begin(), values.end()) - values.begin());
                tokens.push_back(token);
                llama_pos positions[4] = {position, position, position, position};
                int32_t n_seq = 1;
                llama_seq_id seq = 0, * seq_ptr = &seq;
                int8_t output = 1;
                llama_batch batch{1, &token, nullptr, positions, &n_seq, &seq_ptr, &output};
                llama_set_kv_stream_decode(target.get(), true);
                if (llama_decode(target.get(), batch) != 0) { tokens.clear(); return tokens; }
                ++position;
            }
            return tokens;
        };
        auto baseline_tokens = generate(baseline_position);
        if (!t.assert_equal(continuation_tokens, baseline_tokens.size())) return;
        const auto first_followup_position = baseline_position;
        llama_pos after_decode_position = baseline_position;
        std::vector<float> expected_after_decode;
        std::vector<llama_token> expected_after_decode_tokens;
#if !defined(_WIN32) || !defined(LLAMA_SHARED)
        session_recurrent_snapshot expected_after_decode_recurrent;
#endif
        if (shared_vision) {
            if (!t.assert_equal(0, mtmd_helper_eval_chunks(vision.get(), target.get(), followup.get(),
                    baseline_position + continuation_tokens, 0, 32, true, &after_decode_position))) return;
            expected_after_decode = logits();
#if !defined(_WIN32) || !defined(LLAMA_SHARED)
            static_cast<llama_memory_hybrid *>(llama_get_memory(target.get()))->get_mem_recr()->state_write(expected_after_decode_recurrent, 0, 0);
#endif
            expected_after_decode_tokens = generate(after_decode_position);
            if (!t.assert_equal(continuation_tokens, expected_after_decode_tokens.size())) return;
        }
        target.reset();
        if (phase_arena) {
            vision.reset();
            vision.reset(mtmd_init_from_file_deferred(projector_path,model.get(),params));
            if (!t.assert_true(bool(vision) && !mtmd_acquire_projector_weights(vision.get()))) return;
        }
        context_params.kv_stream_pool_bytes = stream_pool_mib * 1048576;
        context_params.shared_device_memory_bytes = shared_budget_mib * 1048576;
        target.reset(llama_init_from_model(model.get(), context_params));
        if (!t.assert_true(bool(target))) return;
        if (shared_vision) probe.target = target.get();
        llama_pos planned_position = 0;
        if (!t.assert_equal(0, evaluate(vision.get(), target.get(), prefix.get(), 0, 0, 32, true,
                &planned_position))) return;
        t.assert_equal(prefix_position, planned_position);
        t.assert_true(compare(expected_prefix, "initial"));
        if (suspended_vision) {
            t.assert_true(probe.paused_calls > 0);
            auto * owner = llama_context_compute_memory(target.get());
            t.assert_true(owner && owner->valid() && !owner->kv_device_suspended() && owner->serial_ready());
            if (phase_arena) {
                t.assert_true(!mtmd_acquire_projector_weights(vision.get()));
                mtmd::batch_ptr inspect(mtmd_batch_init(vision.get()));
                for (size_t i = 0; i < mtmd_input_chunks_size(prefix.get()); ++i) {
                    auto * entry = mtmd_input_chunks_get(prefix.get(),i);
                    if (mtmd_input_chunk_get_type(entry) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                        if (!t.assert_equal(0,mtmd_batch_add_chunk(inspect.get(),entry))) return;
                        break;
                    }
                }
                mtmd_vision_phase_requirements plan;
                if (!t.assert_true(mtmd_batch_measure_vision_phase(inspect.get(),owner->shared_parent(),plan))) return;
                t.assert_true(probe.phase_bytes >= plan.weight_bytes+plan.device_compute_bytes && probe.phase_bytes <= shared_budget_mib*1048576);
                t.out << "vision phase: weights=" << plan.weight_bytes/1048576.0 << " MiB, device compute="
                      << plan.device_compute_bytes/1048576.0 << " MiB, host compute outside=" << plan.host_compute_bytes/1048576.0
                      << " MiB, peak grants=" << probe.phase_bytes/1048576.0 << " MiB\n";
            }
        }
#if !defined(_WIN32) || !defined(LLAMA_SHARED)
        auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(target.get()));
        auto * stream = hybrid->get_mem_attn()->get_kv_stream();
        session_recurrent_snapshot actual_recurrent;
        hybrid->get_mem_recr()->state_write(actual_recurrent, 0, 0);
        t.assert_true(actual_recurrent.bytes == initial_recurrent.bytes);
        if (stream) {
            size_t physical_rows = 0;
            for (size_t i = 0; i < mtmd_input_chunks_size(prefix.get()); ++i)
                physical_rows += mtmd_input_chunk_get_n_tokens(mtmd_input_chunks_get(prefix.get(), i));
            t.assert_equal(physical_rows, stream->tokens());
            t.assert_true(physical_rows > size_t(planned_position));
        }
#endif
        auto planned_state = snapshot();
        if (!t.assert_true(!planned_state.empty())) return;
        if (suspended_vision) {
            // Earlier text stays committed when vision is interrupted; no embedding prefill follows it.
            probe.fail_once = true;
            llama_pos interrupted_position = planned_position;
            t.assert_true(evaluate(vision.get(),target.get(),followup.get(),planned_position,0,32,true,
                &interrupted_position) != 0);
            t.assert_true(!probe.fail_once && interrupted_position >= planned_position);
            auto * owner = llama_context_compute_memory(target.get());
            t.assert_true(owner && owner->valid() && !owner->kv_device_suspended() && owner->serial_ready());
            t.assert_true(mtmd_release_compute_workspace(vision.get()));
            t.assert_equal(planned_state.size(),llama_state_set_data(target.get(),planned_state.data(),planned_state.size()));
            t.assert_true(snapshot() == planned_state);
#if !defined(_WIN32) || !defined(LLAMA_SHARED)
            {
                // Arm only after vision has acquired its grants, so the fault reaches target restoration.
                vision_restore_fault fault(owner->shared_parent());
                bool armed = false;
                probe.on_encode = [&] { if (!armed) { armed = true; fault.arm(); } };
                interrupted_position = planned_position;
                t.assert_true(evaluate(vision.get(),target.get(),followup.get(),planned_position,0,32,true,
                    &interrupted_position) != 0);
                probe.on_encode = {};
                t.assert_true(armed && fault.calls > 0 && owner->valid() && owner->kv_device_suspended());
                t.assert_true(!owner->serial_ready());
            }
            if (!t.assert_true(llama_context_resume_kv_device(target.get(),llama_memory_text_phase::prefill))) return;
            t.assert_equal(planned_state.size(),llama_state_set_data(target.get(),planned_state.data(),planned_state.size()));
            t.assert_true(snapshot() == planned_state);
#endif
        }
#if !defined(_WIN32) || !defined(LLAMA_SHARED)
        if (stream) {
            auto * cache = hybrid->get_mem_attn();
            const auto & cells = cache->get_cells(0);
            size_t first = 0;
            for (size_t i = 1; i < stream->tokens(); ++i) {
                if (cells.pos_get(uint32_t(i)) == cells.pos_get(uint32_t(i - 1))) { first = i - 1; break; }
            }
            if (!t.assert_true(first != 0)) return;
            const auto position = cells.pos_get(uint32_t(first));
            while (first && cells.pos_get(uint32_t(first - 1)) == position) --first;
            t.assert_true(!cache->seq_rm(0, position, position + 1));
            t.assert_true(snapshot() == planned_state);
            // Exercise attention-only suffix addressing, then restore recurrent and KV state together.
            if (!t.assert_true(cache->seq_rm(0, position, -1))) return;
            t.assert_equal(first, stream->tokens());
            t.assert_equal(first, size_t(cells.get_used()));
            t.assert_equal(planned_state.size(), llama_state_set_data(target.get(), planned_state.data(), planned_state.size()));
            t.assert_true(snapshot() == planned_state);
        }
#endif
        if (adaptive) {
            mtmd::batch_ptr image_batch(mtmd_batch_init(vision.get()));
            const mtmd_input_chunk * image_chunk = nullptr;
            for (size_t i = 0; i < mtmd_input_chunks_size(prefix.get()); ++i) {
                auto * chunk = mtmd_input_chunks_get(prefix.get(), i);
                if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) { image_chunk = chunk; break; }
            }
            if (!t.assert_equal(0, mtmd_batch_add_chunk(image_batch.get(), image_chunk))) return;
            if (phase_arena) {
                if (!t.assert_equal(0,mtmd_batch_encode_arena(vision.get(),image_batch.get(),target.get()))) return;
            } else if (shared_vision) {
                auto * parent = llama_context_compute_memory(target.get());
                std::vector<ggml_backend_memory_workspace_group> groups;
                if (!t.assert_true(parent && parent->shared_parent() &&
                        mtmd_batch_measure_compute_workspace(image_batch.get(), groups,
                            ggml_backend_buffer_get_type(parent->shared_parent())) &&
                        mtmd_borrow_compute_workspace(vision.get(), *parent))) return;
            }
            if (!phase_arena && !t.assert_equal(0, mtmd_batch_encode(image_batch.get()))) return;
            mtmd_embedding_view image_view;
            if (!t.assert_true(mtmd_batch_acquire_output_embd(image_batch.get(), image_chunk, image_view))) return;
            if (shared_vision && !phase_arena && !t.assert_true(mtmd_release_compute_workspace(vision.get()))) return;
            llama_pos coordinates[4] = {-1, 0, 0, 0};
            int32_t n_seq = 1;
            llama_seq_id seq = 0, * seq_ptr = &seq;
            int8_t output = 0;
            llama_batch invalid_batch{1, nullptr, const_cast<float *>(image_view.data()), coordinates, &n_seq, &seq_ptr, &output};
            t.assert_true(llama_decode(target.get(), invalid_batch) != 0);
            t.assert_true(snapshot() == planned_state);
            coordinates[0] = coordinates[1] = coordinates[2] = planned_position;
            llama_set_kv_stream_decode(target.get(), true);
            t.assert_true(llama_decode(target.get(), invalid_batch) != 0);
            t.assert_true(snapshot() == planned_state);
            llama_set_kv_stream_decode(target.get(), false);
        }
        // Both whole-context and sequence checkpoint routes must preserve image position metadata.
        if (!t.assert_equal(baseline_state.size(), llama_state_set_data(target.get(), baseline_state.data(), baseline_state.size()))) return;
        if (!t.assert_equal(planned_state.size(), llama_state_set_data(target.get(), planned_state.data(), planned_state.size()))) return;
        std::vector<uint8_t> sequence_state(llama_state_seq_get_size(target.get(), 0));
        if (!t.assert_equal(sequence_state.size(), llama_state_seq_get_data(target.get(), sequence_state.data(), sequence_state.size(), 0))) return;
        if (!t.assert_equal(sequence_state.size(), llama_state_seq_set_data(target.get(), sequence_state.data(), sequence_state.size(), 0))) return;
        if (!t.assert_equal(0, evaluate(vision.get(), target.get(), followup.get(), planned_position, 0, 32, true,
                &planned_position))) return;
        t.assert_equal(first_followup_position, planned_position);
        t.assert_true(compare(expected_followup, "followup"));
#if !defined(_WIN32) || !defined(LLAMA_SHARED)
        session_recurrent_snapshot actual_followup;
        hybrid->get_mem_recr()->state_write(actual_followup, 0, 0);
        t.assert_true(actual_followup.bytes == followup_recurrent.bytes);
#endif
        auto planned_tokens = generate(planned_position);
        t.assert_equal(continuation_tokens, planned_tokens.size());
        t.assert_true(planned_tokens == baseline_tokens);
        if (shared_vision) {
            // Reclaim the populated decode pool for vision, then rebuild a fresh decode layout.
            auto * parent = llama_context_compute_memory(target.get());
            mtmd::batch_ptr image_batch(mtmd_batch_init(vision.get()));
            for (size_t i = 0; i < mtmd_input_chunks_size(followup.get()); ++i) {
                auto * chunk = mtmd_input_chunks_get(followup.get(), i);
                if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE &&
                        !t.assert_equal(0, mtmd_batch_add_chunk(image_batch.get(), chunk))) return;
            }
            auto retained_state = snapshot();
            std::vector<ggml_backend_memory_workspace_group> groups;
            if (phase_arena) {
                auto tiny = ggml_backend_buffer_ptr(ggml_backend_buffer_view(parent->shared_parent(),0,1048576));
                mtmd_vision_phase_requirements untouched; untouched.weight_bytes = 123;
                t.assert_true(!mtmd_batch_measure_vision_phase(image_batch.get(),tiny.get(),untouched));
                t.assert_equal(size_t(123),untouched.weight_bytes);
                t.assert_true(snapshot() == retained_state && !parent->kv_device_suspended());
                size_t upload_calls = 0;
                auto cancel_upload = [](float,void * data) { return ++*static_cast<size_t *>(data) == 1; };
                t.assert_true(mtmd_batch_encode_arena(vision.get(),image_batch.get(),target.get(),cancel_upload,&upload_calls) != 0);
                t.assert_equal(size_t(2),upload_calls);
                t.assert_true(!parent->kv_device_suspended() && !mtmd_acquire_projector_weights(vision.get()));
                t.assert_true(snapshot() == retained_state);
                if (!t.assert_equal(0,mtmd_batch_encode_arena(vision.get(),image_batch.get(),target.get()))) return;
                t.assert_true(!mtmd_acquire_projector_weights(vision.get()));
            } else {
                if (!t.assert_true(parent && parent->shared_parent() &&
                        mtmd_batch_measure_compute_workspace(image_batch.get(), groups,
                            ggml_backend_buffer_get_type(parent->shared_parent())) &&
                        (!suspended_vision || llama_context_suspend_kv_device(target.get())) &&
                        mtmd_borrow_compute_workspace(vision.get(), *parent))) return;
                if (suspended_vision) t.assert_true(!llama_context_resume_kv_device(target.get(),llama_memory_text_phase::decode));
                if (!t.assert_equal(0, mtmd_batch_encode(image_batch.get())) ||
                        !t.assert_true(mtmd_release_compute_workspace(vision.get()))) return;
                if (suspended_vision) {
                    t.assert_true(parent->kv_device_suspended());
                    if (!t.assert_true(llama_context_resume_kv_device(target.get(),llama_memory_text_phase::decode))) return;
                }
            }
            t.assert_true(snapshot() == retained_state);
            if (!t.assert_equal(0, evaluate(vision.get(), target.get(), followup.get(), planned_position + continuation_tokens,
                    0, 32, true, &planned_position))) return;
            t.assert_equal(after_decode_position, planned_position);
            t.assert_true(compare(expected_after_decode, "after-decode image"));
#if !defined(_WIN32) || !defined(LLAMA_SHARED)
            session_recurrent_snapshot actual_after_decode;
            hybrid->get_mem_recr()->state_write(actual_after_decode, 0, 0);
            t.assert_true(actual_after_decode.bytes == expected_after_decode_recurrent.bytes);
#endif
            t.assert_true(generate(planned_position) == expected_after_decode_tokens);
        }
#if !defined(_WIN32) || !defined(LLAMA_SHARED)
        if (stream) {
            llama_kv_stream_runtime_diagnostics diagnostics;
            if (!t.assert_true(stream->runtime_diagnostics(diagnostics))) return;
            t.out << "stream pool=" << stream_pool_mib << " MiB, active_pages=" << diagnostics.active_pages
                  << ", resident_pages=" << diagnostics.resident_pages_per_layer
                  << ", ring_slots=" << diagnostics.ring_slots << ", streaming=" << diagnostics.streaming_active
                  << ", granted_pool=" << diagnostics.pool_bytes / 1048576.0 << " MiB\n";
            t.assert_true(diagnostics.streaming_active != resident_control);
            t.assert_equal(stream->tokens(), size_t(hybrid->get_mem_attn()->get_cells(0).get_used()));
        }
#endif
        mtmd::bitmap_ptr missing_pixels(mtmd_bitmap_init(256, 256, nullptr));
        auto invalid = tokenize("Safe prefix: " + marker + " This text must not be decoded after the failed image.",
            {missing_pixels.get()});
        if (!t.assert_true(bool(invalid))) return;
        const llama_pos start = planned_position + continuation_tokens;
        llama_pos expected_failure_position = start;
        for (size_t i = 0; i < mtmd_input_chunks_size(invalid.get()); ++i) {
            auto * chunk = mtmd_input_chunks_get(invalid.get(), i);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) break;
            expected_failure_position += mtmd_input_chunk_get_n_pos(chunk);
        }
        llama_pos stopped_position = start;
        t.assert_true(evaluate(vision.get(), target.get(), invalid.get(), start, 0, 32, true,
            &stopped_position) != 0);
        llama_synchronize(target.get());
        t.assert_equal(expected_failure_position, stopped_position);
        t.assert_equal(stopped_position - 1, llama_memory_seq_pos_max(llama_get_memory(target.get()), 0));
    });
    return t.summary();
}
