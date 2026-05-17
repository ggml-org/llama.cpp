#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

class llm_moe_file_source {
  public:
    struct tensor_info {
        uint64_t file_offset;
        size_t   nbytes;
    };

    // Open and parse GGUF tensor index. Idempotent.
    bool open(const std::string & path);

    // Lookup tensor by name. Returns nullptr if absent or not open.
    const tensor_info * lookup(const std::string & name);

    // Read bytes from file. Returns true on success.
    bool pread_into(void * dst, uint64_t file_offset, size_t nbytes);

    int  fd();
    bool is_open();

    ~llm_moe_file_source();

  private:
    bool parse_locked(const std::string & path);

    std::mutex                                   mtx;
    int                                          fd_ = -1;
    std::string                                  path_;
    std::unordered_map<std::string, tensor_info> tensors;
};

// Singleton accessor.
llm_moe_file_source & moe_file_source();