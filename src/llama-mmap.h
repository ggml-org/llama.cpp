#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include "uint8-buff-stream.h"
#include <future>
#include <string>
#include <map>

struct llama_file;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_file {
    virtual ~llama_file() = default;

    virtual size_t tell() const = 0;
    virtual size_t size() const = 0;
    virtual int file_id() const = 0;

    virtual void seek(size_t offset, int whence) const = 0;

    virtual void read_raw(void * ptr, size_t len) const = 0;
    virtual uint32_t read_u32() const = 0;

    virtual void write_raw(const void * ptr, size_t len) const = 0;
    virtual void write_u32(uint32_t val) const = 0;
};

struct llama_file_disk : public llama_file {
    llama_file_disk(const char * fname, const char * mode);
    ~llama_file_disk() override;

    size_t tell() const override;
    size_t size() const override;
    int file_id() const override;

    void seek(size_t offset, int whence) const override;

    void read_raw(void * ptr, size_t len) const override;
    uint32_t read_u32() const override;

    void write_raw(const void * ptr, size_t len) const override;
    void write_u32(uint32_t val) const override;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

template <bool Writable> struct llama_file_buffer : public llama_file {
    llama_file_buffer(std::unique_ptr<std::basic_streambuf<uint8_t>> && streambuf);

    ~llama_file_buffer() override;

    size_t tell() const override;
    size_t size() const override;

    /// @return -1 to indicate this is not a real file descriptor
    int file_id() const override;

    void seek(size_t offset, int whence) const override;

    void     read_raw(void * ptr, size_t len) const override;
    uint32_t read_u32() const override;

    /// @throw std::runtime_error if the buffer is read-only
    void write_raw(const void * ptr, size_t len) const override;

    /// @throw std::runtime_error if the buffer is read-only
    void write_u32(uint32_t val) const override;

    std::unique_ptr<std::basic_streambuf<uint8_t>> streambuf;
};

template <bool Writable> struct llama_future_file_buffer {
    /// @brief A file buffer object whose operations will block
    /// until the given promise key is set with a file buffer.
    /// @param promise_key The key to use for the promise (e.g. a file path).
    /// @param context The context to use for the promise, used to distinguish same promise key (e.g. for a same file opened twice).
    llama_future_file_buffer(const std::string & promise_key, const std::string & context);

    // Delete copy constructor and copy assignment operator
    llama_future_file_buffer(const llama_future_file_buffer &)             = delete;
    llama_future_file_buffer & operator=(const llama_future_file_buffer &) = delete;

    llama_future_file_buffer(llama_future_file_buffer && other) noexcept;
    llama_future_file_buffer & operator=(llama_future_file_buffer && other) noexcept;

    ~llama_future_file_buffer();

    /// @brief Sets the given key and context with a file buffer so that
    /// operations can resume/start.
    static bool fulfill_promise(const std::string & promise_key, const std::string & context,
                                std::unique_ptr<llama_file_buffer<Writable>> && value);

    /// @brief Waits for future buffer or obtains current if already
    /// fulfilled and moves the future contents outside the registry.
    std::unique_ptr<llama_file_buffer<Writable>> extract() const;

  private:
    typename std::map<std::string, std::promise<std::unique_ptr<llama_file_buffer<Writable>>>>::iterator
                                                                      file_buffer_promise_iterator;
    mutable std::future<std::unique_ptr<llama_file_buffer<Writable>>> file_buffer_future;
    mutable std::unique_ptr<llama_file_buffer<Writable>>              file_buffer;
};

// Type aliases for convenience
using llama_file_buffer_ro = llama_file_buffer<false>;
using llama_file_buffer_rw = llama_file_buffer<true>;
using llama_future_file_buffer_ro = llama_future_file_buffer<false>;
using llama_future_file_buffer_rw = llama_future_file_buffer<true>;

struct llama_mmap {
    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false);
    ~llama_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mlock {
    llama_mlock();
    ~llama_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t llama_path_max();
