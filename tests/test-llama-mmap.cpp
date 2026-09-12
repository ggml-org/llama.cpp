#include "testing.h"

#include "../src/llama-mmap.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

static constexpr size_t test_file_size = LLAMA_FILE_IO_CHUNK_SIZE + 3*1024*1024 + 1;
static constexpr size_t test_io_size   = 1024*1024;

static uint8_t test_byte(size_t offset) {
    return static_cast<uint8_t>((offset * 131u) ^ (offset >> 11));
}

static bool matches(const void * data, size_t offset, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        if (bytes[i] != test_byte(offset + i)) {
            return false;
        }
    }
    return true;
}

struct temporary_file {
    std::filesystem::path path;

    temporary_file() {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() / ("test-llama-mmap-" + std::to_string(now) + ".bin");
    }

    ~temporary_file() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

struct aligned_buffer {
    void * data = nullptr;

    aligned_buffer(size_t alignment, size_t size) {
        alignment = std::max(alignment, alignof(std::max_align_t));
#if defined(_WIN32)
        data = _aligned_malloc(size, alignment);
#else
        if (posix_memalign(&data, alignment, size) != 0) {
            data = nullptr;
        }
#endif
    }

    ~aligned_buffer() {
#if defined(_WIN32)
        _aligned_free(data);
#else
        free(data);
#endif
    }

    aligned_buffer(const aligned_buffer &) = delete;
    aligned_buffer & operator=(const aligned_buffer &) = delete;
};

static void write_test_file(const std::filesystem::path & path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to create test file");
    }

    std::vector<uint8_t> buffer(test_io_size);
    for (size_t offset = 0; offset < test_file_size; offset += buffer.size()) {
        const size_t size = std::min(buffer.size(), test_file_size - offset);
        for (size_t i = 0; i < size; ++i) {
            buffer[i] = test_byte(offset + i);
        }
        file.write(reinterpret_cast<const char *>(buffer.data()), size);
        if (!file) {
            throw std::runtime_error("failed to write test file");
        }
    }
}

static void check_file(testing & t, const std::filesystem::path & path, bool use_direct_io) {
    llama_file file(path.string().c_str(), "rb", use_direct_io);
    t.assert_equal(test_file_size, file.size());
    t.assert_true("read alignment is nonzero", file.read_alignment() != 0);

    const size_t alignment = file.read_alignment();
    const std::vector<std::pair<size_t, size_t>> ranges = {
        { 0, 1 },
        { 1, 31 },
        { alignment - 1, alignment + 3 },
        { LLAMA_FILE_IO_CHUNK_SIZE - 11, 27 },
        { test_file_size - 31, 31 },
        {0, test_file_size}
    };
    for (const auto & range : ranges) {
        std::vector<uint8_t> buffer(range.second);
        file.seek(range.first, SEEK_SET);
        file.read_raw(buffer.data(), buffer.size());
        t.assert_true("read expected bytes", matches(buffer.data(), range.first, buffer.size()));
        t.assert_equal(range.first + range.second, file.tell());
    }

    std::vector<uint8_t> buffer(test_io_size);
    for (size_t offset = 0; offset < test_file_size; offset += buffer.size()) {
        const size_t size = std::min(buffer.size(), test_file_size - offset);
        file.seek(offset, SEEK_SET);
        file.read_raw(buffer.data(), size);
        t.assert_true("scan expected bytes", matches(buffer.data(), offset, size));
    }

    const size_t unsafe_size = ((LLAMA_FILE_IO_CHUNK_SIZE + 3 * alignment - 1) / alignment) * alignment;
    aligned_buffer buffer_aligned(alignment, unsafe_size);
    t.assert_true("allocate aligned buffer", buffer_aligned.data != nullptr);
    if (buffer_aligned.data != nullptr) {
        const size_t offset = 7 * alignment;
        file.seek(offset, SEEK_SET);
        file.read_raw_unsafe(buffer_aligned.data, unsafe_size);
        t.assert_true("unsafe read expected bytes", matches(buffer_aligned.data, offset, unsafe_size));
        t.assert_equal(offset + unsafe_size, file.tell());

        const size_t final_offset = (test_file_size - 1) & ~(alignment - 1);
        file.seek(final_offset, SEEK_SET);
        file.read_raw_unsafe(buffer_aligned.data, alignment);
        t.assert_true("unsafe EOF read expected bytes", matches(buffer_aligned.data, final_offset, test_file_size - final_offset));
        t.assert_equal(test_file_size, file.tell());
    }
}

static void check_mmap(testing & t, const std::filesystem::path & path) {
    if (!llama_mmap::SUPPORTED) {
        return;
    }

    llama_file file(path.string().c_str(), "rb");
    llama_mmap mapping(&file, 0, false);
    t.assert_equal(test_file_size, mapping.size());
    const auto * data = static_cast<const uint8_t *>(mapping.addr());
    t.assert_true("mapping has an address", data != nullptr);
    if (data != nullptr) {
        t.assert_true("first byte", matches(data, 0, 1));
        t.assert_true("chunk boundary", matches(data + LLAMA_FILE_IO_CHUNK_SIZE - 11, LLAMA_FILE_IO_CHUNK_SIZE - 11, 27));
        t.assert_true("last bytes", matches(data + test_file_size - 31, test_file_size - 31, 31));
    }
}

int main(int argc, char * argv[]) {
    testing t;
    temporary_file test_file;
    t.test("write_test_file", [&] (testing &) { write_test_file(test_file.path); });

    t.test("llama_file.buffered", [&] (testing & t) { check_file(t, test_file.path, false); });
    t.test("llama_file.direct", [&] (testing & t) { check_file(t, test_file.path, true); });
    t.test("llama_mmap", [&] (testing & t) { check_mmap(t, test_file.path); });
    return t.summary();
}
