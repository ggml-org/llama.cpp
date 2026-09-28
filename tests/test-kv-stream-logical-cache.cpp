#include "../src/llama-kv-stream-logical-cache.h"
#include "testing.h"
#include "../src/llama-kv-stream-writer.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <cstring>
#include <vector>

using operand = ggml_kv_stream_operand;

static std::shared_ptr<llama_kv_stream_host> host(uint64_t id, ggml_type k = GGML_TYPE_Q8_0, ggml_type v = GGML_TYPE_Q4_0) {
    llama_kv_stream_host_config config;
    config.cache_id = id;
    config.shape = {k, v, 32, 32, 1, 8, 128};
    config.capabilities = {
        {k, true, true, true, true},
        {v, true, true, true, true},
        true, true,
    };
    config.context_tokens = 17;
    config.layers = 1;
    return llama_kv_stream_host::create(config, ggml_backend_cpu_buffer_type());
}

static bool append(testing & t, llama_kv_stream_logical_cache & cache, size_t count, uint8_t value) {
    const auto backing = cache.host();
    const size_t k_bytes = backing->layout().k_token_bytes*count;
    const size_t v_bytes = backing->layout().v_token_bytes*count;
    std::vector<uint8_t> k(k_bytes, value), v(v_bytes, value + 1);
    llama_kv_stream_write write;
    const size_t first = cache.tokens();
    if (!t.assert_true(cache.begin(count))) return false;
    if (!t.assert_true(cache.content()->prepare({
            {0, operand::k, first*backing->layout().k_token_bytes, k.data(), k.size()},
            {0, operand::v, first*backing->layout().v_token_bytes, v.data(), v.size()}}, write))) return false;
    if (!t.assert_true(cache.publish_host(write))) return false;
    t.assert_equal(first + count, cache.frontiers().host);
    t.assert_equal(size_t(0), cache.frontiers().device);
    t.assert_equal(first, cache.tokens());
    if (!t.assert_true(cache.finish())) return false;
    t.assert_equal(first + count, cache.tokens());
    return true;
}

struct generated_source {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;

    generated_source(ggml_backend_t backend, size_t rows) {
        context.reset(ggml_init({65536, nullptr, true}));
        GGML_ASSERT(context);
        k = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 32, int64_t(rows));
        v = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 32, int64_t(rows));
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        GGML_ASSERT(buffer);
        std::vector<float> keys(rows*32, 1.25f), values(rows*32, 2.5f);
        ggml_backend_tensor_set(k, keys.data(), 0, keys.size()*sizeof(float));
        ggml_backend_tensor_set(v, values.data(), 0, values.size()*sizeof(float));
    }
};

int main() {
    testing t;
    t.test("suffix_truncation_preserves_clean_resident_rows_but_restore_invalidates_them", [](testing & t) {
        auto cache = llama_kv_stream_logical_cache::create(host(202));
        if (!t.assert_true(cache && append(t,*cache,4,0x20))) return;
        const auto content=cache->content();
        t.assert_true(content->flush({{0,operand::k,0,4},{0,operand::v,0,4}},[](const auto &) { return true; }));
        t.assert_true(cache->truncate(2));
        bool dirty=true;
        t.assert_true(content->dirty({0,operand::k,0,2},dirty));
        t.assert_true(!dirty);
        t.assert_true(content->dirty({0,operand::v,2,2},dirty));
        t.assert_true(dirty);
        t.assert_true(cache->restore(cache->checkpoint()));
        t.assert_true(content->dirty({0,operand::k,0,2},dirty));
        t.assert_true(dirty);
    });
    t.test("target_and_mtp_have_distinct_host_bytes_and_publication_revisions", [](testing & t) {
        auto target = llama_kv_stream_logical_cache::create(host(101));
        auto mtp = llama_kv_stream_logical_cache::create(host(202));
        if (!t.assert_true(target && mtp)) return;
        t.assert_true(target->host() != mtp->host());
        t.assert_true(target->content() != mtp->content());
        t.assert_equal(uint64_t(101), target->identity().id);
        t.assert_equal(uint64_t(202), mtp->identity().id);
        t.assert_true(append(t, *mtp, 3, 0x31));
        t.assert_equal(size_t(0), target->tokens());
        t.assert_equal(size_t(3), mtp->tokens());
        llama_kv_stream_host_layer target_layer, mtp_layer;
        t.assert_true(target->host()->layer(0, target_layer) && mtp->host()->layer(0, mtp_layer));
        t.assert_equal(uint8_t(0), static_cast<uint8_t *>(target_layer.k)[0]);
        t.assert_equal(uint8_t(0x31), static_cast<uint8_t *>(mtp_layer.k)[0]);
        t.assert_equal(uint8_t(0x32), static_cast<uint8_t *>(mtp_layer.v)[0]);
        t.assert_true(mtp->identity().generation > target->identity().generation);
    });

    t.test("cancel_and_truncate_invalidate_stale_writes_without_touching_other_cache", [](testing & t) {
        auto target = llama_kv_stream_logical_cache::create(host(101));
        auto mtp = llama_kv_stream_logical_cache::create(host(202));
        if (!t.assert_true(target && mtp && append(t, *mtp, 4, 0x20))) return;
        const auto before = mtp->identity();
        if (!t.assert_true(mtp->begin(2))) return;
        uint8_t value = 9;
        llama_kv_stream_write stale;
        if (!t.assert_true(mtp->content()->prepare({{0, operand::k, 0, &value, 1}}, stale))) return;
        t.assert_true(mtp->cancel());
        t.assert_equal(size_t(4), mtp->tokens());
        t.assert_true(!mtp->content()->commit(stale));
        t.assert_true(mtp->identity().generation > before.generation);
        t.assert_true(mtp->truncate(1));
        t.assert_equal(size_t(1), mtp->tokens());
        t.assert_true(!mtp->truncate(2));
        t.assert_equal(size_t(0), target->tokens());
        t.assert_true(append(t, *mtp, 2, 0x40));
        t.assert_equal(size_t(3), mtp->tokens());
    });

    t.test("external_save_restore_checks_cache_identity_and_rebuilds_frontiers", [](testing & t) {
        auto mtp = llama_kv_stream_logical_cache::create(host(202));
        if (!t.assert_true(mtp && append(t, *mtp, 2, 0x50))) return;
        const auto saved = mtp->checkpoint();
        llama_kv_stream_host_layer layer;
        if (!t.assert_true(mtp->host()->layer(0, layer))) return;
        std::vector<uint8_t> saved_bytes(mtp->host()->bytes());
        std::memcpy(saved_bytes.data(), layer.k, saved_bytes.size());
        if (!t.assert_true(append(t, *mtp, 2, 0x70))) return;
        std::memcpy(layer.k, saved_bytes.data(), saved_bytes.size());
        const auto foreign = llama_kv_stream_logical_cache::create(host(303));
        if (!t.assert_true(bool(foreign))) return;
        t.assert_true(!foreign->restore(saved));
        t.assert_true(mtp->restore(saved));
        t.assert_equal(size_t(2), mtp->tokens());
        t.assert_equal(size_t(2), mtp->frontiers().reserved);
        t.assert_equal(size_t(0), mtp->frontiers().device);
        t.assert_equal(size_t(2), mtp->frontiers().host);
        t.assert_equal(size_t(2), mtp->frontiers().committed);
        t.assert_true(std::memcmp(layer.k, saved_bytes.data(), saved_bytes.size()) == 0);
        t.assert_true(mtp->identity().generation > saved.generation);
        t.assert_true(append(t, *mtp, 1, 0x60));
    });
    t.test("partial_or_out_of_range_kv_cannot_publish_a_token_frontier", [](testing & t) {
        auto mtp = llama_kv_stream_logical_cache::create(host(404));
        if (!t.assert_true(bool(mtp) && mtp->begin(2))) return;
        const auto backing = mtp->host();
        const size_t k_stride = backing->layout().k_token_bytes;
        const size_t v_stride = backing->layout().v_token_bytes;
        std::vector<uint8_t> k(k_stride*2, 1), v(v_stride*2, 2);
        llama_kv_stream_write partial, extra, complete;
        t.assert_true(mtp->content()->prepare({{0, operand::k, 0, k.data(), k.size()}}, partial));
        t.assert_true(!mtp->publish_host(partial));
        t.assert_equal(size_t(0), mtp->frontiers().host);
        t.assert_true(mtp->content()->prepare({
            {0, operand::k, 0, k.data(), k.size()},
            {0, operand::v, v_stride, v.data(), v_stride}}, extra));
        t.assert_true(!mtp->publish_host(extra));
        t.assert_true(mtp->content()->prepare({
            {0, operand::k, 0, k.data(), k.size()},
            {0, operand::v, 0, v.data(), v.size()}}, complete));
        t.assert_true(mtp->publish_host(complete));
        t.assert_true(mtp->finish());
        t.assert_equal(size_t(2), mtp->tokens());
    });
    t.test("generated_prompt_chunks_publish_only_after_writer_completion", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096));
        auto backing = host(505, GGML_TYPE_F16, GGML_TYPE_F16);
        if (!t.assert_true(bool(backend) && bool(scratch) && bool(backing))) return;
        auto cache = llama_kv_stream_logical_cache::create(backing);
        auto unique = llama_kv_stream_writer::create(backend.get(), scratch.get(), 4096, backing->config().shape, 9);
        if (!t.assert_true(bool(cache) && bool(unique))) return;
        std::shared_ptr<llama_kv_stream_writer> writer(std::move(unique));
        generated_source first(backend.get(), 9);
        if (!t.assert_true(cache->begin_generated(writer, first.k, first.v))) return;
        t.assert_equal(size_t(9), cache->frontiers().reserved);
        t.assert_equal(size_t(0), cache->frontiers().host);
        t.assert_equal(size_t(0), cache->tokens());
        if (!t.assert_true(cache->complete_generated())) return;
        t.assert_equal(size_t(9), cache->tokens());
        t.assert_equal(size_t(0), cache->frontiers().device);
        generated_source second(backend.get(), 8);
        if (!t.assert_true(cache->begin_generated(writer, second.k, second.v) &&
                cache->complete_generated())) return;
        t.assert_equal(size_t(17), cache->tokens());
        llama_kv_stream_host_layer planes;
        if (!t.assert_true(backing->layer(0, planes))) return;
        const auto * keys = static_cast<const ggml_fp16_t *>(planes.k);
        const auto * values = static_cast<const ggml_fp16_t *>(planes.v);
        for (size_t row : {size_t(0), size_t(8), size_t(9), size_t(16)}) {
            t.assert_equal(ggml_fp32_to_fp16(1.25f), keys[row*32]);
            t.assert_equal(ggml_fp32_to_fp16(2.5f), values[row*32]);
        }
    });
    t.test("cancel_generated_append_drains_before_reusing_host_rows", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096));
        auto backing = host(606, GGML_TYPE_F16, GGML_TYPE_F16);
        if (!t.assert_true(bool(backend) && bool(scratch) && bool(backing))) return;
        auto cache = llama_kv_stream_logical_cache::create(backing);
        auto unique = llama_kv_stream_writer::create(backend.get(), scratch.get(), 4096, backing->config().shape, 9);
        if (!t.assert_true(bool(cache) && bool(unique))) return;
        std::shared_ptr<llama_kv_stream_writer> writer(std::move(unique));
        generated_source source(backend.get(), 3);
        if (!t.assert_true(cache->begin_generated(writer, source.k, source.v))) return;
        t.assert_equal(size_t(0), cache->frontiers().host);
        t.assert_true(cache->cancel());
        t.assert_equal(size_t(0), cache->tokens());
        t.assert_true(cache->begin_generated(writer, source.k, source.v));
        t.assert_true(cache->complete_generated());
        t.assert_equal(size_t(3), cache->tokens());
    });
    t.test("wide_generated_prefill_crosses_page_and_writer_tile_boundaries", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096));
        auto config = host(707, GGML_TYPE_F16, GGML_TYPE_F16)->config();
        config.context_tokens = 513;
        config.shape.page_tokens = 256;
        auto backing = llama_kv_stream_host::create(config, ggml_backend_cpu_buffer_type());
        if (!t.assert_true(bool(backend) && bool(scratch) && bool(backing))) return;
        auto cache = llama_kv_stream_logical_cache::create(backing);
        auto unique = llama_kv_stream_writer::create(backend.get(), scratch.get(), 4096, config.shape, 257);
        if (!t.assert_true(bool(cache) && bool(unique))) return;
        std::shared_ptr<llama_kv_stream_writer> writer(std::move(unique));
        generated_source first(backend.get(), 257);
        auto wrong_shape = config.shape;
        wrong_shape.type_k = GGML_TYPE_Q8_0;
        wrong_shape.type_v = GGML_TYPE_Q4_0;
        ggml_backend_buffer_ptr wrong_scratch(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096));
        if (!t.assert_true(bool(wrong_scratch))) return;
        auto wrong_unique = llama_kv_stream_writer::create(backend.get(), wrong_scratch.get(), 4096, wrong_shape, 257);
        if (!t.assert_true(bool(wrong_unique))) return;
        std::shared_ptr<llama_kv_stream_writer> wrong_writer(std::move(wrong_unique));
        t.assert_true(!cache->begin_generated(wrong_writer, first.k, first.v));
        t.assert_equal(size_t(0), cache->frontiers().reserved);
        generated_source wrong(backend.get(), 3);
        t.assert_true(!cache->begin_generated(writer, first.k, wrong.v));
        t.assert_equal(size_t(0), cache->tokens());
        if (!t.assert_true(cache->begin_generated(writer, first.k, first.v))) return;
        t.assert_equal(size_t(0), cache->frontiers().host);
        if (!t.assert_true(cache->complete_generated())) return;
        t.assert_true(writer->stats().graph_submissions > 2);
        generated_source second(backend.get(), 256);
        if (!t.assert_true(cache->begin_generated(writer, second.k, second.v) &&
                cache->complete_generated())) return;
        t.assert_equal(size_t(513), cache->tokens());
        t.assert_equal(size_t(0), cache->frontiers().device);
        llama_kv_stream_host_layer planes;
        if (!t.assert_true(backing->layer(0, planes))) return;
        const auto * keys = static_cast<const ggml_fp16_t *>(planes.k);
        const auto * values = static_cast<const ggml_fp16_t *>(planes.v);
        for (size_t row : {size_t(0), size_t(255), size_t(256), size_t(257), size_t(512)}) {
            t.assert_equal(ggml_fp32_to_fp16(1.25f), keys[row*32]);
            t.assert_equal(ggml_fp32_to_fp16(2.5f), values[row*32]);
        }
    });
    return t.summary();
}
