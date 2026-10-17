// llama_gather_fills + llama_direct_reader: the direct (O_DIRECT) reads of expert runs must deliver exactly the
// requested bytes - fragmented runs gathered into one buffer, unaligned run starts, a run ending on the last byte
// of a file whose size is not a sector multiple - without putting a single page into the page cache, and the
// hit mode must copy resident pieces from memory while leaving the rest to disk.

#include "../src/llama-mmap.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/vfs.h>
#include <unistd.h>

static int n_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

static size_t resident_pages(const void * addr, size_t size, size_t page) {
    const size_t n = (size + page - 1) / page;
    std::vector<unsigned char> vec(n);
    if (mincore(const_cast<void *>(addr), size, vec.data()) != 0) {
        return (size_t) -1;
    }
    size_t r = 0;
    for (auto v : vec) {
        r += v & 1;
    }
    return r;
}

static void * aligned_alloc_or_die(size_t align, size_t size) {
    void * p = nullptr;
    if (posix_memalign(&p, align, size) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        exit(1);
    }
    return p;
}

// read `runs` of the tensor at `base` (file offset) through the real gather + reader into `dst` (tensor image)
// returns false if any fill failed
static bool read_runs(llama_direct_reader & rd, const std::vector<llama_byte_range> & runs, size_t base,
        const void * map, size_t capacity, void * buf, std::vector<char> & dst, size_t * n_fills = nullptr) {
    const size_t align = rd.alignment();
    const auto fills = llama_gather_fills(runs, capacity, align, base);
    if (n_fills) {
        *n_fills = fills.size();
    }
    bool ok = true;
    for (const auto & f : fills) {
        CHECK(f.bytes <= capacity, "fill of %zu bytes exceeds the capacity %zu", f.bytes, capacity);
        auto t = rd.submit(f, 0, base, map, buf);
        if (!rd.wait(t)) {
            ok = false;
            continue;
        }
        for (const auto & s : f.segs) {
            memcpy(dst.data() + s.offs, (const char *) buf + s.buf_offs + (base + s.offs) % align, s.size);
        }
    }
    return ok;
}

int main() {
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const size_t MB   = 1024*1024;

    // 48 MiB + 32 B: the end of the file is not a sector multiple, like the model shards
    const size_t size = 48*MB + 32;

    const char * d = getenv("TMPDIR");
    std::string path = std::string(d && *d ? d : ".") + "/test-copy-direct-XXXXXX";
    const int fd = mkstemp(path.data());
    if (fd < 0) {
        fprintf(stderr, "mkstemp failed\n");
        return 1;
    }
    // memory-backed filesystems accept O_DIRECT on recent kernels but never drop their pages: nothing to test there
    {
        struct statfs sf{};
        if (fstatfs(fd, &sf) == 0 && (sf.f_type == 0x01021994 /* TMPFS_MAGIC */ || sf.f_type == 0x858458f6 /* RAMFS_MAGIC */)) {
            fprintf(stderr, "%s is on tmpfs/ramfs: skipping\n", path.c_str());
            unlink(path.c_str());
            return 77;
        }
    }
    std::vector<char> content(size);
    {
        uint32_t x = 0x9E3779B9u;
        for (size_t i = 0; i < size; ++i) {
            x = x*1664525u + 1013904223u;
            content[i] = (char) (x >> 24);
        }
        if (pwrite(fd, content.data(), size, 0) != (ssize_t) size) {
            fprintf(stderr, "pwrite failed\n");
            unlink(path.c_str());
            return 1;
        }
    }
    fdatasync(fd);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);

    void * map = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "mmap failed\n");
        unlink(path.c_str());
        return 1;
    }
    posix_madvise(map, size, POSIX_MADV_RANDOM); // like the lazy expert ranges
    // precondition of every residency assertion below: the filesystem drops the pages we asked it to drop
    if (resident_pages(map, size, page) != 0) {
        fprintf(stderr, "page cache of %s cannot be evicted here: skipping\n", path.c_str());
        unlink(path.c_str());
        return 77;
    }

    const size_t capacity = 8*MB + 2*MB; // small ring buffer so that the runs below need several fills
    void * buf = aligned_alloc_or_die(page, capacity);

    llama_direct_reader rd;
    if (!rd.init({ path }, buf, capacity, 8, 2*MB)) {
        fprintf(stderr, "O_DIRECT not available on %s: skipping\n", path.c_str());
        unlink(path.c_str());
        return 77;
    }
    const size_t align = rd.alignment();
    CHECK(align >= 512 && align <= 4096 && (align & (align - 1)) == 0, "alignment %zu", align);

    // --- 1. the "tensor" starts 100 bytes into the file; runs as in test-copy-prefetch plus one ending on the
    //        last byte of the file; an unaligned run start; a run larger than the buffer (split at 2 MiB multiples)
    {
        const size_t base = 100;
        const size_t tsize = size - base;
        const std::vector<llama_byte_range> runs = {
            { 1*MB, 10*MB },                      // larger than the buffer
            { 20*MB, 3*MB + 512 },                // the 512-byte MMQ padding
            { 30*MB + 100, 2*MB },                // unaligned start
            { tsize - 5*MB - 7, 5*MB + 7 },       // ends on the last byte of the file
        };
        std::vector<char> got(tsize, 0);
        size_t n_fills = 0;
        const bool ok = read_runs(rd, runs, base, (const char *) map + base, capacity, buf, got, &n_fills);
        CHECK(ok, "a fill failed");
        for (const auto & r : runs) {
            CHECK(memcmp(got.data() + r.offs, content.data() + base + r.offs, r.size) == 0, "run at %zu differs", r.offs);
        }
        // bytes outside the runs were never written
        CHECK(got[0] == 0 && got[15*MB] == 0, "bytes outside the runs were written");
        CHECK(n_fills >= 3, "%zu fills", n_fills);
        // nothing went through the page cache
        const size_t res = resident_pages(map, size, page);
        CHECK(res == 0, "%zu pages resident after direct reads", res);
    }

    // --- 2. fragmented runs (2 MiB every 6 MiB) are gathered: few fills, bytes exact
    {
        const size_t base = 4096;
        std::vector<llama_byte_range> runs;
        for (size_t o = 0; o + 2*MB <= size - base - 6*MB; o += 6*MB) {
            runs.push_back({ o, 2*MB });
        }
        std::vector<char> got(size - base, 0);
        size_t n_fills = 0;
        CHECK(read_runs(rd, runs, base, (const char *) map + base, capacity, buf, got, &n_fills), "a fill failed");
        size_t total = 0;
        for (const auto & r : runs) {
            total += r.size;
            CHECK(memcmp(got.data() + r.offs, content.data() + base + r.offs, r.size) == 0, "run at %zu differs", r.offs);
        }
        CHECK(n_fills <= (total + 8*MB - 1) / (8*MB) + 1, "%zu fills for %zu runs", n_fills, runs.size());
        CHECK(resident_pages(map, size, page) == 0, "pages resident after fragmented reads");
    }

    // --- 3. one thread gives the same bytes
    {
        llama_direct_reader rd1;
        CHECK(rd1.init({ path }, buf, capacity, 1, 2*MB), "single-thread init");
        const size_t base = 100;
        const std::vector<llama_byte_range> runs = { { 1*MB, 10*MB }, { 30*MB + 100, 2*MB } };
        std::vector<char> got(size - base, 0);
        CHECK(read_runs(rd1, runs, base, (const char *) map + base, capacity, buf, got), "a fill failed");
        for (const auto & r : runs) {
            CHECK(memcmp(got.data() + r.offs, content.data() + base + r.offs, r.size) == 0, "run at %zu differs (1 thread)", r.offs);
        }
        rd1.stop();
    }

    // --- 4. hits: pieces whose pages are resident are copied from memory; the others read from disk; the
    //        resident pages stay resident and the others stay out of the cache
    {
        llama_direct_reader rdh;
        rdh.set_hits(true);
        CHECK(rdh.init({ path }, buf, capacity, 4, 2*MB), "hit-mode init");
        if (rdh.hits_enabled()) {
            // fault in the second 2 MiB stripe of the file (pages 512..1023)
            volatile char sink = 0;
            for (size_t o = 2*MB; o < 4*MB; o += page) {
                sink += ((const char *) map)[o];
            }
            (void) sink;
            CHECK(resident_pages((const char *) map + 2*MB, 2*MB, page) == 2*MB/page, "pre-faulted pages not resident");

            const size_t base = 0;
            const std::vector<llama_byte_range> runs = { { 0, 8*MB } };
            std::vector<char> got(size, 0);
            rdh.reset_counters();
            CHECK(read_runs(rdh, runs, base, map, capacity, buf, got), "a fill failed (hits)");
            CHECK(memcmp(got.data(), content.data(), 8*MB) == 0, "hit-mode bytes differ");
            const auto c = rdh.get_counters();
            CHECK(c.pieces_hit == 1 && c.bytes_hit == 2*MB, "hits: %llu pieces, %llu bytes", (unsigned long long) c.pieces_hit, (unsigned long long) c.bytes_hit);
            CHECK(c.pieces_read == 3 && c.bytes_read == 6*MB, "reads: %llu pieces, %llu bytes", (unsigned long long) c.pieces_read, (unsigned long long) c.bytes_read);
            CHECK(resident_pages((const char *) map + 2*MB, 2*MB, page) == 2*MB/page, "pre-faulted pages evicted");
            CHECK(resident_pages((const char *) map + 4*MB, 4*MB, page) == 0, "direct reads filled the cache");
            posix_madvise(map, size, POSIX_MADV_DONTNEED);
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        } else {
            fprintf(stderr, "hit detection unavailable (not the file owner): hit case skipped\n");
        }
        rdh.stop();
    }

    // --- 5. the gather alone: capacity respected, segments placed at aligned offsets, phases kept
    {
        const size_t base = 100;
        const std::vector<llama_byte_range> runs = { { 0, 3*MB }, { 3*MB + 1, 1 }, { 20*MB, 30*MB } };
        const auto fills = llama_gather_fills(runs, capacity, align, base);
        size_t covered = 0;
        for (const auto & f : fills) {
            CHECK(f.bytes <= capacity, "fill too large");
            size_t end = 0;
            for (const auto & s : f.segs) {
                CHECK(s.buf_offs % align == 0 && s.buf_offs >= end, "segment placement");
                end = s.buf_offs + ((base + s.offs + s.size + align - 1) / align) * align - ((base + s.offs) / align) * align;
                covered += s.size;
            }
            CHECK(end == f.bytes, "fill bytes %zu vs %zu", f.bytes, end);
        }
        CHECK(covered == 3*MB + 1 + 30*MB, "covered %zu", covered);
    }

    // --- 6. the init probe decides up front: a buffer that is not aligned for O_DIRECT, or a file that cannot be
    //        opened, leave the reader disabled (and the caller on the mapping path)
    {
        llama_direct_reader rdm;
        CHECK(!rdm.init({ path }, (char *) buf + 256, capacity - 256, 2, 2*MB) && !rdm.enabled(), "misaligned probe buffer accepted");
        llama_direct_reader rdn;
        CHECK(!rdn.init({ path + ".missing" }, buf, capacity, 2, 2*MB) && !rdn.enabled(), "missing file accepted");
        llama_direct_reader rdp;
        CHECK(!rdp.init({ path }, buf, capacity, 2, 2*MB + 256) && !rdp.enabled(), "unaligned piece size accepted");
    }

    rd.stop();
    munmap(map, size);
    close(fd);
    unlink(path.c_str());
    free(buf);

    if (n_fail) {
        fprintf(stderr, "%d failure(s)\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
