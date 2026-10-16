// llama_copy_ranges_prefetched: the copy callback must see exactly the requested runs, chunked, in order, and
// the pages of every run must have been hinted (read into the page cache) ahead of the copy - checked through
// page-cache residency of a file mapping whose pages were evicted first, with a copy callback that never reads
// the source bytes (so residency can only come from the hints)

#include "../src/llama-mmap.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

static int n_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

static bool page_resident(const void * addr) {
    unsigned char v = 0;
    return mincore(const_cast<void *>(addr), 1, &v) == 0 && (v & 1);
}

int main() {
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const size_t MB   = 1024*1024;

    // a 48 MiB file: runs at [1 MiB, 11 MiB), [20 MiB, 36.5 MiB), [40 MiB + 100, 40 MiB + 100 + 3 MiB); the rest untouched
    const size_t size = 48*MB;
    const std::vector<llama_byte_range> runs = { { 1*MB, 10*MB }, { 20*MB, 16*MB + 512*1024 }, { 40*MB + 100, 3*MB } };

    const char * d = getenv("TMPDIR");
    std::string path = std::string(d && *d ? d : ".") + "/test-copy-prefetch-XXXXXX";
    const int fd = mkstemp(path.data());
    if (fd < 0) {
        fprintf(stderr, "mkstemp failed\n");
        return 1;
    }
    unlink(path.c_str());
    {
        std::vector<char> buf(size, 7);
        if (pwrite(fd, buf.data(), size, 0) != (ssize_t) size) {
            fprintf(stderr, "pwrite failed\n");
            return 1;
        }
    }
    fdatasync(fd);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);

    void * map = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "mmap failed\n");
        return 1;
    }
    posix_madvise(map, size, POSIX_MADV_RANDOM); // like the lazy expert ranges: no readahead on faults

    bool evicted = true;
    for (size_t o = 0; o < size; o += page) {
        evicted = evicted && !page_resident((char *) map + o);
    }
    if (!evicted) {
        printf("SKIPPED: file pages could not be evicted (tmpfs?)\n");
        return 77;
    }

    // the copy records its calls and never touches the source
    std::vector<llama_byte_range> calls;
    auto copy = [&](size_t offs, size_t n) { calls.push_back({ offs, n }); };

    const size_t chunk = 4*MB, lookahead = 8*MB;
    llama_copy_ranges_prefetched(map, runs, true, chunk, lookahead, copy);

    // 1. exactly the runs, split into chunks, in order
    std::vector<llama_byte_range> expect;
    for (const auto & r : runs) {
        for (size_t off = 0; off < r.size; off += chunk) {
            expect.push_back({ r.offs + off, std::min(chunk, r.size - off) });
        }
    }
    CHECK(calls.size() == expect.size(), "%zu copy calls, expected %zu", calls.size(), expect.size());
    for (size_t i = 0; i < calls.size() && i < expect.size(); ++i) {
        CHECK(calls[i].offs == expect[i].offs && calls[i].size == expect[i].size,
                "call %zu: [%zu, +%zu) expected [%zu, +%zu)", i, calls[i].offs, calls[i].size, expect[i].offs, expect[i].size);
    }

    // 2. every page of every run was hinted (reads are asynchronous: allow them to land), nothing outside
    for (int tries = 0; tries < 200; ++tries) {
        bool all = true;
        for (const auto & r : runs) {
            for (size_t o = r.offs & ~(page - 1); o < r.offs + r.size; o += page) {
                all = all && page_resident((char *) map + o);
            }
        }
        if (all) {
            break;
        }
        usleep(10000);
    }
    // (a kernel may read a little past a hinted range with large folios: pages within one 2 MiB piece after a
    // run's end are tolerated, anything further away must not have been read)
    size_t missing = 0, outside = 0;
    for (size_t o = 0; o < size; o += page) {
        bool in_run = false, near_run = false;
        for (const auto & r : runs) {
            in_run   = in_run   || (o + page > r.offs && o < r.offs + r.size);
            near_run = near_run || (o + page > r.offs && o < r.offs + r.size + 2*MB);
        }
        const bool res = page_resident((char *) map + o);
        missing += in_run && !res;
        outside += !near_run && res;
    }
    CHECK(missing == 0, "%zu pages of the runs were not prefetched", missing);
    CHECK(outside == 0, "%zu pages away from the runs were read", outside);

    // 3. order: when a chunk is copied, it and `lookahead` bytes beyond it (in run order) have been hinted, in
    //    pieces no larger than 2 MiB; the hints never run further ahead than that
    {
        size_t hinted = 0;       // bytes hinted so far (hints come in run order)
        size_t max_piece = 0;
        size_t copied = 0;
        bool order_ok = true;
        std::function<void(size_t, size_t)> rec = [&](size_t /*offs*/, size_t n) {
            hinted += n;
            max_piece = std::max(max_piece, n);
        };
        auto copy2 = [&](size_t /*offs*/, size_t n) {
            const size_t total = 29*MB + 512*1024;
            const size_t need  = std::min(total, copied + n + lookahead);
            order_ok = order_ok && hinted >= need && hinted <= need + 2*MB;
            copied += n;
        };
        llama_copy_ranges_prefetched(map, runs, true, chunk, lookahead, copy2, &rec);
        CHECK(order_ok, "hints must cover the chunk being copied plus the lookahead, and no more");
        CHECK(max_piece <= 2*MB, "hint pieces must be <= 2 MiB, largest %zu", max_piece);
        CHECK(hinted == 29*MB + 512*1024, "all run bytes hinted once, got %zu", hinted);
    }

    // 4. prefetch off: same calls, no hints
    munmap(map, size);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    map = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    posix_madvise(map, size, POSIX_MADV_RANDOM);
    calls.clear();
    llama_copy_ranges_prefetched(map, runs, false, chunk, lookahead, copy);
    CHECK(calls.size() == runs.size(), "prefetch off: %zu calls, expected one per run (%zu)", calls.size(), runs.size());
    usleep(50000);
    size_t resident = 0;
    for (size_t o = 0; o < size; o += page) {
        resident += page_resident((char *) map + o);
    }
    CHECK(resident == 0, "prefetch off must not read anything, %zu pages resident", resident);

    munmap(map, size);
    close(fd);

    printf("%s\n", n_fail ? "FAILED" : "OK");
    return n_fail ? 1 : 0;
}
