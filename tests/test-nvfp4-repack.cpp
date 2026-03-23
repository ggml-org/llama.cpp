#include "ggml.h"
#include "ggml-cuda/repack_nvfp4.cuh"

#include <cstdio>
#include <cstring>
#include <vector>

static void set_q4(uint8_t * qs, int idx, uint8_t value) {
    uint8_t & byte = qs[idx / 2];

    if (idx & 1) {
        byte = (uint8_t) ((byte & 0x0f) | ((value & 0x0f) << 4));
    } else {
        byte = (uint8_t) ((byte & 0xf0) | (value & 0x0f));
    }
}

static void fill_block(block_nvfp4 & blk, uint32_t seed, int row, int block) {
    memset(&blk, 0, sizeof(blk));

    for (int i = 0; i < QK_NVFP4; ++i) {
        const uint32_t value = seed + (uint32_t) row * 19U + (uint32_t) block * 11U + (uint32_t) i * 5U + (uint32_t) (i / 3);
        set_q4(blk.qs, i, (uint8_t) (value & 0x0f));
    }

    for (int i = 0; i < QK_NVFP4 / QK_NVFP4_SUB; ++i) {
        const uint32_t value = seed + 0x31U + (uint32_t) row * 7U + (uint32_t) block * 13U + (uint32_t) i * 9U;
        blk.d[i] = (uint8_t) (0x30 + value % 0x30);
    }
}

static void fill_rows(std::vector<uint8_t> & rows, int64_t ne0, int64_t nrows, uint32_t seed) {
    const size_t row_size = ggml_row_size(GGML_TYPE_NVFP4, ne0);
    const int blocks_per_row = (int) (ne0 / QK_NVFP4);

    rows.assign(row_size * nrows, 0);

    for (int64_t row = 0; row < nrows; ++row) {
        block_nvfp4 * dst = (block_nvfp4 *) (rows.data() + row * row_size);
        for (int block = 0; block < blocks_per_row; ++block) {
            fill_block(dst[block], seed, (int) row, block);
        }
    }
}

static void fill_layout_row(std::vector<uint8_t> & rows) {
    rows.assign(ggml_row_size(GGML_TYPE_NVFP4, QK_K), 0);

    block_nvfp4 * dst = (block_nvfp4 *) rows.data();
    for (int lane = 0; lane < 4; ++lane) {
        for (int i = 0; i < 32; ++i) {
            dst[lane].qs[i] = (uint8_t) ((lane * 0x31 + i * 0x17 + 0x12) & 0xff);
        }
        for (int i = 0; i < 4; ++i) {
            dst[lane].d[i] = (uint8_t) (0x31 + lane * 0x0d + i * 0x09);
        }
    }
}

static bool expect_equal(const char * name, const std::vector<uint8_t> & expected, const std::vector<uint8_t> & actual) {
    if (expected.size() != actual.size()) {
        std::printf("%s: size mismatch (%zu != %zu)\n", name, expected.size(), actual.size());
        return false;
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            std::printf("%s: first mismatch at byte %zu (expected 0x%02x, got 0x%02x)\n",
                    name, i, expected[i], actual[i]);
            return false;
        }
    }

    return true;
}

static bool check_roundtrip(const char * name, int64_t ne0, int64_t nrows, uint32_t seed) {
    const size_t row_size = ggml_row_size(GGML_TYPE_NVFP4, ne0);

    std::vector<uint8_t> input;
    std::vector<uint8_t> packed(row_size * nrows);
    std::vector<uint8_t> output(row_size * nrows);

    fill_rows(input, ne0, nrows, seed);
    ggml_cuda_repack_rows_nvfp4(ne0, nrows, input.data(), packed.data());
    ggml_cuda_unpack_rows_nvfp4(ne0, nrows, packed.data(), output.data());

    return expect_equal(name, input, output);
}

static bool check_partial_patch(const char * name, int64_t ne0, int64_t nrows, size_t offset, size_t size, uint32_t seed) {
    const size_t row_size = ggml_row_size(GGML_TYPE_NVFP4, ne0);
    const size_t total_size = row_size * nrows;

    std::vector<uint8_t> input;
    std::vector<uint8_t> expected;
    std::vector<uint8_t> packed(total_size);
    std::vector<uint8_t> output(total_size);
    std::vector<uint8_t> patch(size);

    fill_rows(input, ne0, nrows, seed);
    expected = input;

    for (size_t i = 0; i < patch.size(); ++i) {
        patch[i] = (uint8_t) (seed + 0x5bU + (uint32_t) i * 23U + (uint32_t) (i / 5));
    }

    memcpy(expected.data() + offset, patch.data(), size);

    ggml_cuda_repack_rows_nvfp4(ne0, nrows, input.data(), packed.data());

    const size_t aligned_offset = offset / row_size * row_size;
    const size_t aligned_end = (offset + size + row_size - 1) / row_size * row_size;
    const size_t aligned_size = aligned_end - aligned_offset;
    const int64_t aligned_rows = (int64_t) (aligned_size / row_size);
    const size_t inner_offset = offset - aligned_offset;

    std::vector<uint8_t> rows(aligned_size);
    ggml_cuda_unpack_rows_nvfp4(ne0, aligned_rows, packed.data() + aligned_offset, rows.data());
    memcpy(rows.data() + inner_offset, patch.data(), size);
    ggml_cuda_repack_rows_nvfp4(ne0, aligned_rows, rows.data(), packed.data() + aligned_offset);
    ggml_cuda_unpack_rows_nvfp4(ne0, nrows, packed.data(), output.data());

    return expect_equal(name, expected, output);
}

static bool check_4x64_to_256_layout(const char * name, uint32_t seed) {
    GGML_UNUSED(seed);

    std::vector<uint8_t> input;
    std::vector<uint8_t> packed(ggml_row_size(GGML_TYPE_NVFP4, QK_K));

    fill_layout_row(input);
    ggml_cuda_repack_rows_nvfp4(QK_K, 1, input.data(), packed.data());

    const block_nvfp4 * src = (const block_nvfp4 *) input.data();
    for (int lane = 0; lane < 4; ++lane) {
        for (int pack = 0; pack < 8; ++pack) {
            uint32_t got = 0;
            memcpy(&got, packed.data() + lane * 32 + pack * sizeof(got), sizeof(got));
            const uint32_t expected = ggml_cuda_nvfp4_pack(src[lane].qs, pack);
            if (got != expected) {
                std::printf("%s: lane %d pack %d mismatch (expected 0x%08x, got 0x%08x)\n", name, lane, pack, expected, got);
                return false;
            }
        }

        if (memcmp(packed.data() + 128 + lane * 4, src[lane].d, 4) != 0) {
            std::printf("%s: lane %d scales mismatch\n", name, lane);
            return false;
        }
    }

    std::printf("%s:\n", name);
    std::printf("  4 x block_nvfp4 (64 weights each) -> 1 x block_nvfp4_cuda (256 weights total)\n");
    for (int lane = 0; lane < 4; ++lane) {
        std::printf("  block_nvfp4[%d] qs:", lane);
        for (int i = 0; i < 32; ++i) {
            std::printf(" %02x", src[lane].qs[i]);
        }
        std::printf("\n");
    }
    for (int lane = 0; lane < 4; ++lane) {
        std::printf("  block_nvfp4[%d] scales:", lane);
        for (int i = 0; i < 4; ++i) {
            std::printf(" %02x", src[lane].d[i]);
        }
        std::printf("\n");
    }

    std::printf("  block_nvfp4_cuda:\n");
    for (int lane = 0; lane < 4; ++lane) {
        std::printf("  lane %d qs @0x%02x:", lane, lane * 32);
        for (int pack = 0; pack < 8; ++pack) {
            uint32_t word = 0;
            memcpy(&word, packed.data() + lane * 32 + pack * sizeof(word), sizeof(word));
            std::printf(" %08x", word);
        }
        std::printf("\n");
    }
    for (int lane = 0; lane < 4; ++lane) {
        std::printf("  lane %d scales @0x%02x:", lane, 128 + lane * 4);
        for (int i = 0; i < 4; ++i) {
            std::printf(" %02x", packed[128 + lane * 4 + i]);
        }
        std::printf("\n");
    }
    return true;
}

int main() {
    int total = 0;
    int passed = 0;

    const struct { const char * name; int64_t ne0, nrows; uint32_t seed; } roundtrip_cases[] = {
        { "roundtrip-ne0-64", 64, 3, 0x1001U }, { "roundtrip-ne0-128", 128, 3, 0x1002U },
        { "roundtrip-ne0-192", 192, 3, 0x1003U }, { "roundtrip-ne0-256", 256, 2, 0x1004U },
        { "roundtrip-ne0-320", 320, 3, 0x1005U },
    };

    for (const auto & test : roundtrip_cases) {
        total += 1;
        passed += check_roundtrip(test.name, test.ne0, test.nrows, test.seed);
    }

    total += 1;
    passed += check_partial_patch("partial-cross-row-128", 128, 3, ggml_row_size(GGML_TYPE_NVFP4, 128) - 11, 27, 0x2001U);

    total += 1;
    passed += check_partial_patch("partial-cross-row-320", 320, 2, ggml_row_size(GGML_TYPE_NVFP4, 320) - 19, 41, 0x2002U);

    total += 1;
    passed += check_4x64_to_256_layout("layout-4x64-to-256", 0x3001U);

    std::printf("test-nvfp4-repack: %d/%d passed\n", passed, total);
    return passed == total ? 0 : 1;
}
