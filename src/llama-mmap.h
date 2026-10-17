#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cstdio>

// staging buffer size for direct I/O reads, 64MB works well for NVMe drives
#define LLAMA_DIRECT_IO_BUFFER_SIZE (64 * 1024 * 1024)

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
    const std::string & name() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mmap {
    // list of [first, last) byte ranges within a file
    using ranges = std::vector<std::pair<size_t, size_t>>;

    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false,
               const ranges & lazy_ranges = {});
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

struct llama_memory_range {
    const void * addr;
    size_t size;
};

using llama_memory_ranges = std::vector<llama_memory_range>;

// Prefetch the host pages covering these memory ranges.
void llama_prefetch(llama_memory_ranges mr);

// Copy byte ranges of a host buffer that may be a lazily read file mapping: `copy(offs, size)` is called for
// each run in order, split into chunks, while the pages of the following chunks are hinted to the kernel
// (POSIX_MADV_WILLNEED, in pieces no larger than the kernel's per-call readahead cap) so that the reads from disk
// overlap the copy instead of faulting in one page at a time. The hints cover the chunk being copied and
// `lookahead` bytes beyond it, across run boundaries. With `prefetch` false the runs are copied unchanged.
struct llama_byte_range {
    size_t offs;
    size_t size;
};
// `hint(offs, size)` issues one hint piece; nullptr means llama_prefetch (tests record the order of hints and copies)
void llama_copy_ranges_prefetched(const void * base, const std::vector<llama_byte_range> & runs, bool prefetch,
        size_t chunk, size_t lookahead, const std::function<void(size_t offs, size_t size)> & copy,
        const std::function<void(size_t offs, size_t size)> * hint = nullptr);

size_t llama_path_max();

// Direct reads of lazily mapped tensors (the prompt pass of MoE op offload): the bytes of a tensor's expert runs are
// read with O_DIRECT into a page-aligned buffer by a pool of reader threads, bypassing the page cache, so the pass
// neither evicts the experts decode has cached nor is limited by the kernel's readahead path.
//
// A fill is one buffer's worth of segments gathered across runs: each segment occupies the aligned extent
// [align_down(f), align_up(f + size)) of its run bytes, f = base + offs its offset in the file, placed at `buf_offs`
// in the buffer, so the requested bytes start at buf_offs + f % align.
struct llama_fill_segment {
    size_t offs;     // byte offset within the tensor
    size_t size;     // requested bytes
    size_t buf_offs; // offset in the fill buffer of the aligned extent
};

struct llama_fill {
    std::vector<llama_fill_segment> segs;
    size_t bytes = 0; // aligned bytes used in the buffer
};

// Pack `runs` (tensor-relative, in order) into fills of at most `capacity` aligned bytes each. A run larger than
// a buffer is split at absolute multiples of `split` (relative to `base` = the tensor's offset in its file, so the
// pieces line up with the file's 2 MiB stripes); `align` must be a power of two <= `split`.
std::vector<llama_fill> llama_gather_fills(const std::vector<llama_byte_range> & runs, size_t capacity, size_t align,
        size_t base, size_t split = 2u*1024*1024);

struct llama_direct_reader {
    llama_direct_reader();
    ~llama_direct_reader();
    llama_direct_reader(const llama_direct_reader &) = delete;

    // Open `paths` for direct reads and probe one aligned read into `probe_buf` (>= alignment() bytes, aligned).
    // Returns enabled(): false when the platform, filesystem or buffer does not support O_DIRECT (logged once).
    bool init(const std::vector<std::string> & paths, void * probe_buf, size_t probe_size, int n_threads, size_t piece);
    bool   enabled() const;
    size_t alignment() const; // offset/length/buffer alignment of the reads (512..4096)

    // Hits: pieces whose pages are all resident in the mapping are copied from memory instead of read from disk.
    // Only available when the calling user owns (or may write) every file: mincore reports every page resident
    // otherwise. Decided in init(); `want` enables it.
    bool hits_enabled() const;
    void set_hits(bool want);

    struct ticket { uint64_t id = 0; };

    // Queue the reads of one fill of tensor `file_idx` (file offset `file_offs` of the tensor, mapped at `map`)
    // into `buf` (aligned, >= fill.bytes). Pieces are split at absolute `piece` multiples of the file offset.
    ticket submit(const llama_fill & fill, int file_idx, size_t file_offs, const void * map, void * buf);
    // Wait for all pieces of the fill; false if any read failed (the buffer contents are then undefined and the
    // caller must produce the fill another way). Tickets are waited in submission order.
    bool wait(ticket t);
    // Wait for everything queued and join the threads.
    void stop();

    struct counters {
        uint64_t bytes_read  = 0; // from disk
        uint64_t bytes_hit   = 0; // copied from resident pages
        uint64_t pieces_read = 0;
        uint64_t pieces_hit  = 0;
        uint64_t errors      = 0;
    };
    counters get_counters() const; // totals since init
    void     reset_counters();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
