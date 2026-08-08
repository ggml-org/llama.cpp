#include "common.h"
#include <cassert>
#include <string>
#include <filesystem>
#include <iostream>

int main() {
    namespace fs = std::filesystem;
    fs::path base_dir = fs::temp_directory_path() / "llama_test_common_sandbox";
    fs::create_directories(base_dir);

    std::string base = fs::canonical(base_dir).string();
    std::cout << "test-common: base directory is " << base << "\n";

    // Positive test cases
    assert(true  == fs_validate_path_in_directory("safe.txt", base));
    assert(true  == fs_validate_path_in_directory("sub/safe.txt", base));
    assert(true  == fs_validate_path_in_directory("sub/dir/safe.txt", base));

    // Negative test cases: empty filenames
    assert(false == fs_validate_path_in_directory("", base));

    // Negative test cases: only dots
    assert(false == fs_validate_path_in_directory(".", base));
    assert(false == fs_validate_path_in_directory("..", base));
    assert(false == fs_validate_path_in_directory("sub/..", base));
    assert(false == fs_validate_path_in_directory("../safe.txt", base));
    assert(false == fs_validate_path_in_directory("sub/../../safe.txt", base));

    // Negative test cases: absolute paths (Unix)
    assert(false == fs_validate_path_in_directory("/etc/passwd", base));
    assert(false == fs_validate_path_in_directory("/absolute/path", base));

    // Negative test cases: absolute paths (Windows & platform specific)
    assert(false == fs_validate_path_in_directory("C:/etc/passwd", base));
    assert(false == fs_validate_path_in_directory("C:\\etc\\passwd", base));
    assert(false == fs_validate_path_in_directory("\\\\server\\share\\file", base));
    assert(false == fs_validate_path_in_directory("\\etc\\passwd", base));

    // Negative test cases: directory traversal via Unicode equivalents or other tricks
    assert(false == fs_validate_path_in_directory("safe.txt:", base));

    // Cleanup
    fs::remove_all(base_dir);

    std::cout << "test-common: all tests passed successfully!\n";
    return 0;
}
