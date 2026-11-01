#include "../src/llama-mmap.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct test_file {
    std::filesystem::path directory;
    std::string path;

    test_file() {
        std::random_device random;
        for (int i = 0; i < 100; ++i) {
            auto candidate = std::filesystem::temp_directory_path() / ("llama-file-read-" + std::to_string(random()));
            if (std::filesystem::create_directory(candidate)) {
                directory = std::move(candidate);
                path = (directory / "weights.bin").u8string();
                return;
            }
        }
        throw std::runtime_error("failed to create test directory");
    }

    ~test_file() {
        std::error_code error;
        std::filesystem::remove(directory / "weights.bin", error);
        std::filesystem::remove(directory, error);
    }
};

static void check_reads(llama_file & file, const std::vector<uint8_t> & expected, size_t seed) {
    const size_t alignment = file.read_alignment_at();
    std::vector<uint8_t> storage(128 * 1024 + 2 * alignment);
    auto * buffer = (uint8_t *) ((uintptr_t(storage.data()) + alignment - 1) & ~(alignment - 1));
    for (size_t i = 0; i < 64; ++i) {
        const size_t count = 1 + (i * 7919 + seed) % (64 * 1024);
        const size_t offset = i % 3 == 0 ? expected.size() - count : (i * 31 + seed) % (expected.size() - count);
        const size_t start = offset & ~(alignment - 1);
        const size_t end = (offset + count + alignment - 1) & ~(alignment - 1);
        file.read_raw_at(buffer, end - start, start);
        GGML_ASSERT(std::memcmp(buffer + offset - start, expected.data() + offset, count) == 0);
    }
    // Unaligned requests must use the buffered fallback when a direct handle is open.
    file.read_raw_at(buffer + 1, 17, 3);
    GGML_ASSERT(std::memcmp(buffer + 1, expected.data() + 3, 17) == 0);
    file.read_raw_at(nullptr, 0, expected.size());
}

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 1 || (argc == 2 && std::string(argv[1]) == "--require-direct"));
    const bool require_direct = argc == 2;
    test_file temporary;
    std::vector<uint8_t> expected(256 * 1024 + 13);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = (i * 17 + i / 13) % 251;
    }
    {
        std::ofstream output(std::filesystem::u8path(temporary.path), std::ios::binary);
        output.write((const char *) expected.data(), expected.size());
        output.close();
        GGML_ASSERT(output.good());
    }
    for (bool direct : {false, true}) {
        llama_file file(temporary.path.c_str(), "rb");
        if (direct) {
            file.prepare_read_at();
            file.prepare_read_at();
            GGML_ASSERT(!require_direct || file.read_alignment_at() > 1);
        }
        std::vector<std::future<void>> readers;
        for (size_t i = 0; i < 4; ++i) {
            readers.emplace_back(std::async(std::launch::async, [&file, &expected, i] {
                check_reads(file, expected, i);
            }));
        }
        for (auto & reader : readers) {
            reader.get();
        }
        bool rejected = false;
        try {
            file.read_raw_at(expected.data(), 1, expected.size() + 1);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        GGML_ASSERT(rejected);
        std::printf("PASS concurrent %s reads, alignment %zu, unaligned fallback, EOF tails and bounds\n",
                    direct ? "prepared" : "buffered", file.read_alignment_at());
    }
    {
        llama_file file(temporary.path.c_str(), "rb");
        file.prepare_read_at();
        const size_t alignment = file.read_alignment_at();
        std::vector<uint8_t> storage(2 * alignment);
        auto * buffer = (uint8_t *) ((uintptr_t(storage.data()) + alignment - 1) & ~(alignment - 1));
        std::filesystem::resize_file(std::filesystem::u8path(temporary.path), 0);
        bool rejected = false;
        try {
            file.read_raw_at(buffer, alignment, 0);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        GGML_ASSERT(rejected);
        std::puts("PASS truncated file fails instead of uploading stale bytes");
    }
}
