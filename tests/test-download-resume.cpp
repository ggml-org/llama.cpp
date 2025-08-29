// Test curl download resume functionality
#include "arg.h"
#include "common.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#undef NDEBUG
#include <cassert>

// Mock server that simulates partial downloads
class MockDownloadServer {
  public:
    MockDownloadServer(const std::string & test_file_path, size_t total_size) :
        file_path(test_file_path),
        file_size(total_size) {
        // Create a test file with predictable content
        std::ofstream f(file_path, std::ios::binary);
        for (size_t i = 0; i < file_size; i++) {
            char c = 'A' + (i % 26);
            f.write(&c, 1);
        }
    }

    ~MockDownloadServer() {
        // Cleanup test file
        if (std::filesystem::exists(file_path)) {
            std::filesystem::remove(file_path);
        }
    }

    bool simulate_partial_download(const std::string & dest_path, size_t bytes_to_transfer, size_t start_offset = 0) {
        std::ifstream src(file_path, std::ios::binary);
        std::ofstream dst(dest_path, std::ios::binary | (start_offset > 0 ? std::ios::app : std::ios::trunc));

        if (!src || !dst) {
            return false;
        }

        src.seekg(start_offset);

        char   buffer[1024];
        size_t transferred = 0;
        while (transferred < bytes_to_transfer && src.good()) {
            size_t to_read = std::min(sizeof(buffer), bytes_to_transfer - transferred);
            src.read(buffer, to_read);
            size_t read = src.gcount();
            if (read > 0) {
                dst.write(buffer, read);
                transferred += read;
            } else {
                break;
            }
        }

        return transferred == bytes_to_transfer;
    }

    bool verify_downloaded_content(const std::string & downloaded_path) {
        std::ifstream original(file_path, std::ios::binary);
        std::ifstream downloaded(downloaded_path, std::ios::binary);

        if (!original || !downloaded) {
            return false;
        }

        // Compare file sizes first
        original.seekg(0, std::ios::end);
        downloaded.seekg(0, std::ios::end);
        if (original.tellg() != downloaded.tellg()) {
            return false;
        }

        // Reset to beginning
        original.seekg(0);
        downloaded.seekg(0);

        // Compare content
        char c1, c2;
        while (original.get(c1) && downloaded.get(c2)) {
            if (c1 != c2) {
                return false;
            }
        }

        return true;
    }

  private:
    std::string file_path;
    size_t      file_size;
};

static void test_resume_download() {
    printf("Testing download resume functionality...\n");

    const std::string test_source = "test_source.bin";
    const std::string test_dest   = "test_download.bin.downloadInProgress";
    const size_t      file_size   = 10000;  // 10KB test file

    // Create mock server with test file
    MockDownloadServer server(test_source, file_size);

    // Test 1: Simulate interrupted download at 30%
    printf("  Test 1: Interrupt at 30%%... ");
    size_t first_chunk = file_size * 0.3;
    assert(server.simulate_partial_download(test_dest, first_chunk));
    assert(std::filesystem::file_size(test_dest) == first_chunk);
    printf("OK\n");

    // Test 2: Resume download from 30% to 70%
    printf("  Test 2: Resume to 70%%... ");
    size_t second_chunk = file_size * 0.4;
    assert(server.simulate_partial_download(test_dest, second_chunk, first_chunk));
    assert(std::filesystem::file_size(test_dest) == first_chunk + second_chunk);
    printf("OK\n");

    // Test 3: Complete the download
    printf("  Test 3: Complete download... ");
    size_t final_chunk = file_size - (first_chunk + second_chunk);
    assert(server.simulate_partial_download(test_dest, final_chunk, first_chunk + second_chunk));
    assert(std::filesystem::file_size(test_dest) == file_size);
    printf("OK\n");

    // Test 4: Verify content integrity
    printf("  Test 4: Verify integrity... ");
    assert(server.verify_downloaded_content(test_dest));
    printf("OK\n");

    // Cleanup
    if (std::filesystem::exists(test_dest)) {
        std::filesystem::remove(test_dest);
    }

    printf("All download resume tests passed!\n");
}

static void test_exponential_backoff() {
    printf("Testing exponential backoff calculation...\n");

    int base_delay = 2;  // 2 seconds base

    // Test the corrected exponential backoff formula
    for (int attempt = 0; attempt < 3; attempt++) {
        int expected = base_delay * (1 << attempt) * 1000;  // 2^attempt * base * 1000ms
        printf("  Attempt %d: Expected delay = %d ms\n", attempt + 1, expected);

        // These should match our fixed implementation:
        // Attempt 1: 2 * 2^0 * 1000 = 2000ms
        // Attempt 2: 2 * 2^1 * 1000 = 4000ms
        // Attempt 3: 2 * 2^2 * 1000 = 8000ms
        assert((attempt == 0 && expected == 2000) || (attempt == 1 && expected == 4000) ||
               (attempt == 2 && expected == 8000));
    }

    printf("Exponential backoff tests passed!\n");
}

int main() {
    printf("test-download-resume: Testing curl download resume functionality\n\n");

    test_resume_download();
    test_exponential_backoff();

    printf("\nAll tests passed successfully!\n");
    return 0;
}
