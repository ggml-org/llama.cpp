#include "../tools/mtmd/mtmd-embeddings.h"
#include "testing.h"

#include <cstring>
#include <numeric>

struct embedding_fixture {
    mtmd::input_chunks_ptr input{mtmd_test_create_input_chunks()};
    const mtmd_input_chunk * first = mtmd_input_chunks_get(input.get(), 1);
    mtmd::input_chunk_ptr second{mtmd_input_chunk_copy(first)};
    std::vector<mtmd_embedding_chunk> chunks{{first, 2}, {second.get(), 3}};
    std::vector<float> values = std::vector<float>(20);
    embedding_fixture() { std::iota(values.begin(), values.end(), 1.f); }
};

int main() {
    testing t;
    t.test("complete_batch_publication_moves_storage_and_maps_each_chunk", [](testing & t) {
        embedding_fixture f;
        mtmd_embedding_output owner;
        mtmd_embedding_view first, second;
        t.assert_true(!owner.acquire(f.first, first));
        const float * allocation = f.values.data();
        if (!t.assert_true(owner.publish(f.chunks, 4, f.values))) return;
        t.assert_true(f.values.empty());
        if (!t.assert_true(owner.acquire(f.first, first) && owner.acquire(f.second.get(), second))) return;
        t.assert_true(first.data() == allocation);
        t.assert_true(second.data() == allocation + 8);
        t.assert_equal(size_t(2), first.n_tokens());
        t.assert_equal(size_t(3), second.n_tokens());
        t.assert_equal(size_t(4), second.n_embd());
        t.assert_true(owner.borrow(f.first) == first.data());
        t.assert_equal(20.f, second.data()[11]);
    });
    t.test("checked_slices_preserve_token_order_and_cannot_cross_chunks", [](testing & t) {
        embedding_fixture f;
        mtmd_embedding_output owner;
        mtmd_embedding_view image, prefix, suffix;
        if (!t.assert_true(owner.publish(f.chunks, 4, f.values) && owner.acquire(f.second.get(), image))) return;
        if (!t.assert_true(image.slice(0, 1, prefix) && image.slice(1, 2, suffix))) return;
        t.assert_equal(9.f, prefix.data()[0]);
        t.assert_equal(13.f, suffix.data()[0]);
        t.assert_equal(20.f, suffix.data()[7]);
        const auto * saved = suffix.data();
        for (auto range : {std::pair{3u, 1u}, std::pair{2u, 2u}, std::pair{0u, 0u}}) {
            t.assert_true(!image.slice(range.first, range.second, suffix));
            t.assert_true(suffix.data() == saved);
        }
        t.assert_true(!image.slice(SIZE_MAX, 1, suffix));
        t.assert_true(!image.slice(1, SIZE_MAX, suffix));
        t.assert_true(image.slice(1, 2, image));
        t.assert_equal(size_t(2), image.n_tokens());
        t.assert_true(image.slice(1, 1, image));
        t.assert_equal(17.f, image.data()[0]);
    });
    t.test("replacement_and_cancel_keep_readers_until_their_final_use", [](testing & t) {
        embedding_fixture f;
        mtmd_embedding_view later;
        std::vector<float> expected(f.values.begin() + 8, f.values.end());
        {
            mtmd_embedding_output owner;
            if (!t.assert_true(owner.publish(f.chunks, 4, f.values) && owner.acquire(f.second.get(), later))) return;
            std::vector<float> next(20, -5.f);
            if (!t.assert_true(owner.publish(f.chunks, 4, next))) return;
            t.assert_equal(-5.f, owner.borrow(f.second.get())[0]);
            owner.clear();
            owner.clear();
            t.assert_true(owner.borrow(f.first) == nullptr);
            t.assert_true(!owner.acquire(f.first, later));
        }
        f.second.reset();
        f.input.reset();
        t.assert_true(std::equal(expected.begin(), expected.end(), later.data()));
        mtmd_embedding_view tail;
        if (!t.assert_true(later.slice(2, 1, tail))) return;
        later = {};
        t.assert_equal(17.f, tail.data()[0]);
        tail = {};
        t.assert_true(tail.data() == nullptr);
        t.assert_equal(size_t(0), tail.n_tokens());
        t.assert_equal(size_t(0), tail.n_embd());
    });
    t.test("invalid_publication_preserves_prior_generation_and_input_values", [](testing & t) {
        embedding_fixture f;
        mtmd_embedding_output owner;
        if (!t.assert_true(owner.publish(f.chunks, 4, f.values))) return;
        std::vector<float> candidate(20, 9.f);
        const auto * saved = owner.borrow(f.first);
        for (auto chunks : {std::vector<mtmd_embedding_chunk>{},
                std::vector<mtmd_embedding_chunk>{{nullptr, 5}},
                std::vector<mtmd_embedding_chunk>{{f.first, 0}},
                std::vector<mtmd_embedding_chunk>{{f.first, SIZE_MAX}},
                std::vector<mtmd_embedding_chunk>{{f.first, SIZE_MAX / 4}, {f.second.get(), SIZE_MAX / 4}}}) {
            t.assert_true(!owner.publish(chunks, 4, candidate));
            t.assert_true(candidate.size() == 20 && candidate[0] == 9.f);
            t.assert_true(owner.borrow(f.first) == saved);
        }
        t.assert_true(!owner.publish(f.chunks, 0, candidate));
        t.assert_true(!owner.publish(f.chunks, SIZE_MAX, candidate));
        candidate.resize(19);
        t.assert_true(!owner.publish(f.chunks, 4, candidate));
        candidate.resize(21);
        t.assert_true(!owner.publish(f.chunks, 4, candidate));
        mtmd_embedding_view output;
        if (!t.assert_true(owner.acquire(f.first, output))) return;
        t.assert_true(!owner.acquire(nullptr, output));
        t.assert_true(!owner.acquire(mtmd_input_chunks_get(f.input.get(), 0), output));
        t.assert_true(output.data() == saved);
    });
    t.test("copied_and_moved_views_keep_independent_read_pins", [](testing & t) {
        embedding_fixture f;
        mtmd_embedding_output owner;
        mtmd_embedding_view first;
        if (!t.assert_true(owner.publish(f.chunks, 4, f.values) && owner.acquire(f.first, first))) return;
        auto copied = first;
        auto moved = std::move(first);
        t.assert_true(first.data() == nullptr);
        t.assert_equal(size_t(0), first.n_tokens());
        t.assert_equal(size_t(0), first.n_embd());
        owner.clear();
        moved = {};
        t.assert_equal(1.f, copied.data()[0]);
        t.assert_equal(8.f, copied.data()[7]);
        first = std::move(copied);
        t.assert_equal(size_t(2), first.n_tokens());
        t.assert_true(copied.data() == nullptr);
        t.assert_equal(size_t(0), copied.n_tokens());
    });
    t.test("empty_view_and_cancelled_batch_are_safe", [](testing & t) {
        mtmd_embedding_view empty, output;
        t.assert_true(empty.data() == nullptr);
        t.assert_true(!empty.slice(0, 1, output));
        t.assert_true(!mtmd_batch_acquire_output_embd(nullptr, nullptr, output));
        mtmd_batch_clear_output_embd(nullptr);
        mtmd::batch_ptr batch(mtmd_batch_init(nullptr));
        t.assert_true(!mtmd_batch_acquire_output_embd(batch.get(), nullptr, output));
        mtmd_batch_clear_output_embd(batch.get());
    });
    t.test("compatible_batches_keep_model_limits_and_incompatible_chunks_are_rejected", [](testing & t) {
        embedding_fixture f;
        const auto rows = mtmd_input_chunk_get_n_tokens(f.first);
        t.assert_equal(0, mtmd_batch_validate_chunk({}, f.first, false, 0));
        t.assert_equal(0, mtmd_batch_validate_chunk({f.first}, f.second.get(), true, 2 * rows));
        t.assert_equal(2, mtmd_batch_validate_chunk({f.first}, f.second.get(), true, 2 * rows - 1));
        t.assert_equal(2, mtmd_batch_validate_chunk({f.first}, f.second.get(), false, SIZE_MAX));
        t.assert_equal(1, mtmd_batch_validate_chunk({f.first}, nullptr, true, SIZE_MAX));
        t.assert_equal(1, mtmd_batch_validate_chunk({}, mtmd_input_chunks_get(f.input.get(), 0), true, SIZE_MAX));
        size_t bytes = 0;
        if (!t.assert_equal(0, mtmd_input_chunk_save(f.first, nullptr, 0, &bytes))) return;
        std::vector<char> encoded(bytes);
        if (!t.assert_equal(0, mtmd_input_chunk_save(f.first, encoded.data(), bytes, nullptr))) return;
        // Version 1 image payload follows version, type, empty text count, and image-present byte.
        uint64_t version = 0, text_count = 1;
        std::memcpy(&version, encoded.data(), sizeof(version));
        std::memcpy(&text_count, encoded.data() + sizeof(version) + sizeof(uint32_t), sizeof(text_count));
        if (!t.assert_equal(uint64_t(1), version) || !t.assert_equal(uint64_t(0), text_count)) return;
        const size_t nx_offset = sizeof(version) + sizeof(uint32_t) + sizeof(text_count) + sizeof(uint8_t);
        const uint32_t nx = 8;
        std::memcpy(encoded.data() + nx_offset, &nx, sizeof(nx));
        mtmd::input_chunk_ptr different(mtmd_input_chunk_load(encoded.data(), encoded.size()));
        if (!t.assert_true(bool(different))) return;
        t.assert_equal(3, mtmd_batch_validate_chunk({f.first}, different.get(), true, SIZE_MAX));
    });
    return t.summary();
}
