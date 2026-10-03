#include "../src/llama-kv-stream-positions.h"
#include "testing.h"

#include <limits>

struct coordinate_fixture {
    llama_token tokens[3] = {1, 2, 3};
    float embeddings[3] = {0, 0, 0};
    llama_pos positions[12] = {8, 8, 8, 8, 8, 9, 8, 9, 8, 0, 0, 0};
    int32_t counts[3] = {1, 1, 1};
    llama_seq_id seq = 0;
    llama_seq_id * sequences[3] = {&seq, &seq, &seq};
    llama_batch batch{3, nullptr, embeddings, positions, counts, sequences, nullptr};
    llama_kv_stream_append_coordinates coordinates{100, 256, 8, 4, false, false};
};

static llama_kv_cells cells_with(std::initializer_list<llama_pos> positions) {
    llama_kv_cells cells;
    cells.resize(32);
    uint32_t row = 0;
    for (auto position : positions) {
        cells.pos_set(row, position);
        cells.ext_set(row, {llama_pos(row % 2), llama_pos(row / 2)});
        cells.seq_add(row++, 0);
    }
    return cells;
}

int main() {
    testing t;
    t.test("image_rows_append_physically_despite_overlapping_model_positions", [](testing & t) {
        coordinate_fixture f;
        t.assert_true(llama_kv_stream_validate_append(f.batch, f.coordinates));
        f.positions[2] = 20;
        t.assert_true(llama_kv_stream_validate_append(f.batch, f.coordinates));
        f.coordinates.first_cell = 254;
        t.assert_true(!llama_kv_stream_validate_append(f.batch, f.coordinates));
    });
    t.test("later_text_and_implicit_positions_use_the_model_frontier", [](testing & t) {
        coordinate_fixture f;
        f.batch.embd = nullptr; f.batch.token = f.tokens;
        f.positions[0] = 20; f.positions[1] = 21; f.positions[2] = 24;
        t.assert_true(llama_kv_stream_validate_append(f.batch, f.coordinates));
        f.batch.pos = nullptr;
        t.assert_true(llama_kv_stream_validate_append(f.batch, f.coordinates));
        f.coordinates.previous_position = std::numeric_limits<llama_pos>::max() - 1;
        t.assert_true(!llama_kv_stream_validate_append(f.batch, f.coordinates));
        f.coordinates.previous_position = std::numeric_limits<llama_pos>::max() - 3;
        t.assert_true(!llama_kv_stream_validate_append(f.batch, f.coordinates));
    });
    t.test("decode_and_mtp_do_not_silently_admit_image_inputs", [](testing & t) {
        coordinate_fixture f;
        f.coordinates.decode = true;
        t.assert_true(!llama_kv_stream_validate_append(f.batch, f.coordinates));
        f.coordinates.decode = false; f.coordinates.linear_required = true;
        t.assert_true(!llama_kv_stream_validate_append(f.batch, f.coordinates));
        f.batch.embd = nullptr; f.batch.token = f.tokens;
        f.positions[0] = 100; f.positions[1] = 101; f.positions[2] = 102;
        t.assert_true(llama_kv_stream_validate_append(f.batch, f.coordinates));
    });
    t.test("malformed_coordinates_are_rejected_before_model_work", [](testing & t) {
        coordinate_fixture f;
        f.positions[0] = -1;
        t.assert_true(!llama_kv_stream_validate_append(f.batch, f.coordinates));
        f.positions[0] = 8; f.positions[1] = 7;
        t.assert_true(!llama_kv_stream_validate_append(f.batch, f.coordinates));
        f.positions[1] = 8; f.positions[4] = -1;
        t.assert_true(!llama_kv_stream_validate_append(f.batch, f.coordinates));
        f.positions[4] = 8; f.batch.pos = nullptr;
        t.assert_true(!llama_kv_stream_validate_append(f.batch, f.coordinates));
        f.batch.pos = f.positions; f.batch.token = f.tokens;
        t.assert_true(!llama_kv_stream_validate_append(f.batch, f.coordinates));
        f.batch.token = nullptr; f.counts[0] = 2;
        t.assert_true(!llama_kv_stream_validate_append(f.batch, f.coordinates));
        f.counts[0] = 1; f.seq = 1;
        t.assert_true(!llama_kv_stream_validate_append(f.batch, f.coordinates));
        f.seq = 0; f.batch.n_seq_id = nullptr;
        t.assert_true(!llama_kv_stream_validate_append(f.batch, f.coordinates));
    });
    t.test("restored_cells_keep_positions_and_spatial_metadata", [](testing & t) {
        auto cells = cells_with({0, 1, 2, 2, 2, 10, 11});
        t.assert_true(llama_kv_stream_validate_cells(cells, 7, false));
        t.assert_true(!llama_kv_stream_validate_cells(cells, 7, true));
        t.assert_equal(llama_pos(2), cells.pos_get(4));
        t.assert_equal(llama_pos(2), cells.ext_get(4).y);
        cells.rm(3);
        t.assert_true(!llama_kv_stream_validate_cells(cells, 7, false));
        auto decreasing = cells_with({0, 3, 2});
        t.assert_true(!llama_kv_stream_validate_cells(decreasing, 3, false));
    });
    t.test("removal_maps_position_ranges_to_cell_suffixes", [](testing & t) {
        auto cells = cells_with({0, 1, 2, 2, 2, 10, 11});
        size_t retained = 99;
        t.assert_true(llama_kv_stream_suffix(cells, 7, 10, -1, retained));
        t.assert_equal(size_t(5), retained);
        t.assert_true(llama_kv_stream_suffix(cells, 7, 2, -1, retained));
        t.assert_equal(size_t(2), retained);
        retained = 99;
        t.assert_true(!llama_kv_stream_suffix(cells, 7, 2, 3, retained));
        t.assert_equal(size_t(99), retained);
        t.assert_true(llama_kv_stream_suffix(cells, 7, 99, -1, retained));
        t.assert_equal(size_t(7), retained);
        t.assert_true(llama_kv_stream_suffix(cells, 7, -1, -1, retained));
        t.assert_equal(size_t(0), retained);
        auto maximum = cells_with({std::numeric_limits<llama_pos>::max()});
        t.assert_true(llama_kv_stream_suffix(maximum, 1, 0, -1, retained));
        t.assert_equal(size_t(0), retained);
    });
    t.test("binary_suffix_lookup_matches_interval_removal_for_duplicate_and_gapped_positions", [](testing & t) {
        for (uint32_t count = 1; count <= 64; count += 7) {
            llama_kv_cells cells;
            cells.resize(count);
            llama_pos position = 0;
            for (uint32_t i = 0; i < count; ++i) {
                position += i % 3 ? llama_pos(1 + i % 5) : 0;
                cells.pos_set(i, position); cells.seq_add(i, 0);
            }
            if (!t.assert_true(llama_kv_stream_validate_cells(cells, count, false))) return;
            for (llama_pos begin = -1; begin < position + 3; begin += 3) {
                for (llama_pos end = -1; end < position + 3; end += 5) {
                    size_t keep = count;
                    bool valid = true;
                    for (uint32_t i = 0; i < count; ++i) {
                        const auto value = cells.pos_get(i);
                        const bool remove = value >= std::max(llama_pos(0), begin) && (end < 0 || value < end);
                        if (remove && keep == count) keep = i;
                        if (!remove && keep != count) valid = false;
                    }
                    size_t actual = SIZE_MAX;
                    t.assert_equal(valid, llama_kv_stream_suffix(cells, count, begin, end, actual));
                    t.assert_equal(valid ? keep : SIZE_MAX, actual);
                }
            }
        }
    });
    return t.summary();
}
