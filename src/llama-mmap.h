#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <cstdio>

struct llama_file;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_file {
    llama_file(const char * fname, const char * mode, bool use_direct_io = false);
    llama_file(FILE * file);
    ~llama_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);

    // Positional read: does not move the file pointer, so callers need not seek first
    // and two threads may read one file without fighting over a shared position.
    // Returns the bytes actually read, which is short only at end of file.
    //
    // This is the RAW form. Under direct I/O the caller is responsible for alignment
    // of all three of offset, length and the address of `ptr` - unlike read_raw(),
    // which bounces through an aligned buffer and accepts anything.
    //
    // worker_id names a caller that reads concurrently with other callers, and it is
    // what makes those callers actually concurrent on Windows. Removing the race on the
    // file position was not enough: measured 2026-08-03, several threads on one handle
    // reach 1.01x the throughput of a single thread at queue depth 8, while a private
    // handle each reaches 2.22x. Windows serialises on the file OBJECT.
    //
    // Pass a dense index starting at 0 - one per thread, stable for that thread's life.
    // -1, or an index the pool does not cover, reads through the shared handle: correct,
    // simply serialised. Callers that do not read concurrently pass nothing.
    size_t read_raw_at(void * ptr, size_t len, size_t offset, int worker_id = -1);

    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;

    // How many private per-worker handles this file holds. 0 whenever direct I/O is not
    // in effect, and 0 on POSIX, where pread on one descriptor is already parallel-safe.
    //
    // Exists so the pool can be asserted rather than assumed: a pool that failed to open
    // and a pool that works look identical from the outside - both read the right bytes,
    // one of them just never scales.
    size_t direct_io_handles() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

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
