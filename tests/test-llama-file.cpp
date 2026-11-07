// Covers llama_file's read path, which until now no test touched at all: grepping
// tests/ for load_mode found only argument parsing (test-arg-parser.cpp), load
// cancellation (test-model-load-cancel.cpp) and quantisation statistics
// (test-quantize-stats.cpp). Nothing went red when the reading itself broke.
//
// The cases that MUST fail are the point of this file. A test that only calls
// functions and checks they return proves that they return.
//
// llama_file is internal and the llama library does not export it - see the comment
// at tests/CMakeLists.txt above the block of tests disabled on Windows for that very
// reason - so this test compiles llama-mmap.cpp in directly rather than linking it.

#include "llama-mmap.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#endif

// Deliberately NOT a multiple of any plausible sector size. All four model files on
// this machine end mid-sector, so an aligned read running past EOF is the normal case
// rather than an edge one, and a test file ending on a boundary would never reach it.
static const size_t FILE_SIZE  = 1024 * 1024 + 1234;
static const size_t SECTOR     = 4096;

static int g_failed = 0;

// Whether a direct-I/O open is expected to succeed at all on this platform.
//
// llama_file implements an unbuffered path on Windows (FILE_FLAG_NO_BUFFERING) and on
// Linux (O_DIRECT). Everywhere else the constructor has nothing to open unbuffered and
// correctly falls back to buffered I/O, so failing the run there would report a defect
// where the behaviour is right. On the two platforms that do have the path, that same
// fallback is exactly the defect these cases exist to catch, so it must stay a failure
// there - skipping everywhere would turn this file into a test that cannot go red.
#if defined(_WIN32) || defined(__linux__)
#  define DIRECT_IO_EXPECTED 1
#else
#  define DIRECT_IO_EXPECTED 0
#endif

// The buffer address must be sector-aligned for unbuffered reads, exactly as much as
// the offset and the length must be. Handing a std::vector to read_raw_at is what
// killed the first attempt at this test with 0xC0000409 and no output at all.
struct aligned_buffer {
    size_t   size;
    uint8_t * data;

    explicit aligned_buffer(size_t raw_size) {
        size = (raw_size + SECTOR - 1) & ~(SECTOR - 1);
#ifdef _WIN32
        data = (uint8_t *) _aligned_malloc(size, SECTOR);
#else
        data = nullptr;
        if (posix_memalign((void **) &data, SECTOR, size) != 0) {
            data = nullptr;
        }
#endif
        if (data == nullptr) {
            throw std::bad_alloc();
        }
        std::memset(data, 0, size);
    }

    ~aligned_buffer() {
        if (data) {
#ifdef _WIN32
            _aligned_free(data);
#else
            free(data);
#endif
        }
    }

    aligned_buffer(const aligned_buffer &) = delete;
    aligned_buffer & operator=(const aligned_buffer &) = delete;
};

static void check(bool condition, const char * what) {
    if (condition) {
        printf("  ok    %s\n", what);
    } else {
        printf("  FAIL  %s\n", what);
        g_failed++;
    }
}

static uint8_t pattern_byte(size_t i) {
    // The first term alone repeats every 256 bytes, and 256 divides every sector size in
    // play - so two sector-ALIGNED reads of the same length used to return byte-identical
    // data no matter which sector they came from. Every unbuffered read in this file is
    // aligned by definition, which made the pattern unable to tell one sector from
    // another: a reader that ignored the offset it was given would have passed.
    //
    // Found on 2026-08-03 by the worker-slot case, which compared offset 2*SECTOR against
    // the bytes of 5*SECTOR and found them equal. That looked like a defect in the handle
    // pool and was a defect in this function.
    //
    // The second term changes once per sector and 17 is odd, so it is invertible modulo
    // 256 and two sectors fewer than 256 apart can never carry the same bytes.
    return (uint8_t) ((i * 31 + 7 + (i >> 12) * 17) & 0xFF);
}

static bool write_test_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    std::vector<uint8_t> buf(FILE_SIZE);
    for (size_t i = 0; i < FILE_SIZE; i++) {
        buf[i] = pattern_byte(i);
    }
    const size_t written = fwrite(buf.data(), 1, FILE_SIZE, f);
    fclose(f);
    return written == FILE_SIZE;
}

int main() {
    // Unbuffered stdout, and it is not cosmetic: when this test crashed the buffered
    // output was never flushed, so the failure looked like it happened before main()
    // started. Every line below has to be on screen the moment it is written, or the
    // next crash is just as blind as the last one.
    setvbuf(stdout, NULL, _IONBF, 0);

    // A RELATIVE path on purpose. The first attempt could not derive a volume from one
    // and fell back to buffered I/O without saying so, which made every case below
    // silently skip while the run still reported PASS.
    const std::string path = "test-llama-file.tmp";

    if (!write_test_file(path)) {
        printf("SETUP ERROR: could not write %s\n", path.c_str());
        return 2;
    }
    printf("test file  %s, %zu bytes (%zu past the last %zu boundary)\n\n",
           path.c_str(), FILE_SIZE, FILE_SIZE % SECTOR, SECTOR);

    // -----------------------------------------------------------------------------
    // Case 1: has_direct_io() must say FALSE when nobody asked for direct I/O. On
    // Windows this returned a hard true on every path - it answered a question it had
    // never been in a position to answer.
    // -----------------------------------------------------------------------------
    printf("buffered open, use_direct_io = false:\n");
    {
        llama_file f(path.c_str(), "rb");
        check(!f.has_direct_io(),
              "has_direct_io() is false when direct I/O was never requested");
        check(f.read_alignment() == 1, "read_alignment() is 1 on a buffered file");
        check(f.size() == FILE_SIZE, "size() matches the file on disk");

        aligned_buffer got(64);
        const size_t n = f.read_raw_at(got.data, 64, 1000);
        bool match = (n == 64);
        for (size_t i = 0; i < 64 && match; i++) {
            match = got.data[i] == pattern_byte(1000 + i);
        }
        check(match, "read_raw_at returns the bytes at the requested offset");

        // The failing half of the same check. If a wrong expectation also passed, the
        // check above would only prove that the function returns.
        bool wrong_matches = true;
        for (size_t i = 0; i < 64; i++) {
            if (got.data[i] != pattern_byte(2000 + i)) {
                wrong_matches = false;
                break;
            }
        }
        check(!wrong_matches,
              "the same read does NOT match the bytes of a different offset");
    }

    // -----------------------------------------------------------------------------
    // Case 2: the direct-I/O open. Windows accepted use_direct_io and dropped it;
    // FILE_FLAG_NO_BUFFERING appeared nowhere in llama-mmap.cpp.
    // -----------------------------------------------------------------------------
    printf("\nunbuffered open, use_direct_io = true:\n");
    {
        llama_file f(path.c_str(), "rb", true);

        if (!f.has_direct_io()) {
            // Reporting PASS here would be worse than having no test: a green light for
            // something nobody checked. The first version of this file did exactly that
            // and hid a real defect for a whole run. On a platform without an unbuffered
            // path the same state is the documented behaviour, so it is reported and
            // skipped rather than counted as a failure.
#if DIRECT_IO_EXPECTED
            printf("  FAIL  direct I/O was requested and is not in effect\n");
            printf("        The constructor fell back to buffered I/O, so none of the\n");
            printf("        cases this test exists for were exercised.\n");
            g_failed++;
#else
            printf("  SKIP  this platform has no unbuffered read path in llama_file\n");
            printf("        The fallback to buffered I/O is correct here, so the\n");
            printf("        direct-I/O cases do not apply.\n");
#endif
        } else {
            const size_t align = f.read_alignment();
            check(align > 1, "read_alignment() reports the device's sector size, not 1");
            printf("        alignment = %zu\n", align);
            check(f.size() == FILE_SIZE, "size() matches the file on disk");

            // An unaligned offset AND an unaligned length - exactly what direct I/O
            // refuses at the syscall level. This works only if the bounce buffer does
            // its job, and the destination here may be an ordinary vector: read_raw
            // copies out of its own aligned buffer.
            std::vector<uint8_t> plain(777);
            bool match = false;
            try {
                f.seek(333, SEEK_SET);
                f.read_raw(plain.data(), plain.size());
                match = true;
                for (size_t i = 0; i < plain.size(); i++) {
                    if (plain[i] != pattern_byte(333 + i)) {
                        match = false;
                        break;
                    }
                }
            } catch (const std::exception & e) {
                // Catching is not politeness, it is the measurement. An uncaught C++
                // exception ends in MSVC's abort(), which raises
                // __fastfail(FAST_FAIL_FATAL_APP_EXIT) - reported as 0xC0000409, the
                // same code a stack buffer overrun produces. Without this handler the
                // failure is indistinguishable from memory corruption, and that is
                // exactly how it was misread once already.
                printf("  EXCEPTION during unaligned read_raw: %s\n", e.what());
            }
            check(match, "read_raw handles an unaligned offset and length");

            // -------------------------------------------------------------------
            // Case 3: the read whose aligned end lies past the logical EOF.
            // Measured 2026-08-03 on all four model files: ReadFile returns TRUE and
            // reports exactly the bytes up to EOF. This pins that behaviour so a
            // future change cannot quietly turn it into a throw or a full count.
            //
            // Offset, length and buffer address are all sector-aligned here. This is
            // the raw positional read; it does not bounce.
            // -------------------------------------------------------------------
            const size_t last_aligned = (FILE_SIZE / SECTOR) * SECTOR;
            aligned_buffer tail(SECTOR);
            const size_t n = f.read_raw_at(tail.data, SECTOR, last_aligned);

            check(n == FILE_SIZE - last_aligned,
                  "a read past EOF reports exactly the bytes up to the end of file");
            printf("        offset %zu, asked %zu, got %zu (file ends %zu in)\n",
                   last_aligned, SECTOR, n, FILE_SIZE - last_aligned);

            // The failing half: it must NOT report the full request. If it did, the
            // caller would copy uninitialised bytes into a tensor and never know.
            check(n != SECTOR,
                  "it does NOT report the full request when the file ends first");

            bool tail_match = (n == FILE_SIZE - last_aligned);
            for (size_t i = 0; i < n && tail_match; i++) {
                tail_match = tail.data[i] == pattern_byte(last_aligned + i);
            }
            check(tail_match, "the bytes it did return are the right ones");

            // Wholly past the end, and the offset is sector-aligned like every other
            // unbuffered read - an unaligned one here is what took the first attempt
            // down together with the unaligned buffer.
            const size_t beyond_offset = last_aligned + 4 * SECTOR;
            aligned_buffer beyond(SECTOR);
            const size_t none = f.read_raw_at(beyond.data, SECTOR, beyond_offset);
            check(none == 0, "a read starting entirely past EOF returns 0 bytes");
        }
    }

    // -----------------------------------------------------------------------------
    // Case 4: the per-worker handle pool.
    //
    // This is the one case whose absence cannot be seen in the bytes. A pool that never
    // opened reads exactly what a working pool reads - it just serialises every thread
    // behind one file object, which is what was measured at 1.01x against 2.22x on
    // 2026-08-03. So the pool is asserted by COUNT first and by behaviour second;
    // checking only that the reads come back right would pass on the old code.
    // -----------------------------------------------------------------------------
    printf("\nper-worker handle pool:\n");
    {
        llama_file buffered(path.c_str(), "rb");
        check(buffered.direct_io_handles() == 0,
              "a buffered file holds no private handles");

        llama_file f(path.c_str(), "rb", true);
        if (!f.has_direct_io()) {
#if DIRECT_IO_EXPECTED
            printf("  FAIL  direct I/O not in effect, the pool cannot be checked\n");
            g_failed++;
#else
            printf("  SKIP  no unbuffered read path on this platform, no pool to check\n");
#endif
        } else {
            const size_t n_handles = f.direct_io_handles();
            printf("        private handles = %zu\n", n_handles);

            // The failing case for this stage. On the build before the pool this is 0,
            // and the whole section below would otherwise still be green.
            check(n_handles > 1,
                  "direct I/O opened more than one handle on the file");

            // Every slot must read the file, not just slot 0. An off-by-one in the bounds
            // check would leave the last worker silently on the shared handle - correct
            // bytes, no concurrency, and nothing to see.
            bool all_slots_ok = true;
            for (size_t slot = 0; slot < n_handles; slot++) {
                aligned_buffer b(SECTOR);
                const size_t n = f.read_raw_at(b.data, SECTOR, 2 * SECTOR, (int) slot);
                if (n != SECTOR) {
                    all_slots_ok = false;
                    break;
                }
                for (size_t i = 0; i < SECTOR; i++) {
                    if (b.data[i] != pattern_byte(2 * SECTOR + i)) {
                        all_slots_ok = false;
                        break;
                    }
                }
                if (!all_slots_ok) {
                    break;
                }
            }
            check(all_slots_ok,
                  "every worker slot returns the bytes at the requested offset");

            // The failing half again: a slot that read the right bytes must not also
            // match a different offset, or the loop above proves only that it returned.
            aligned_buffer probe(SECTOR);
            f.read_raw_at(probe.data, SECTOR, 2 * SECTOR, 0);
            bool wrong_matches = true;
            for (size_t i = 0; i < SECTOR; i++) {
                if (probe.data[i] != pattern_byte(5 * SECTOR + i)) {
                    wrong_matches = false;
                    break;
                }
            }
            check(!wrong_matches,
                  "a worker-slot read does NOT match the bytes of a different offset");

            // Out of range in both directions. A caller that knows nothing about pools
            // must land on the shared handle rather than past the array - serialised is
            // an acceptable answer here, reading somebody else's memory is not.
            bool oob_ok = true;
            for (int bad : { -5, (int) n_handles, 999 }) {
                aligned_buffer b(SECTOR);
                const size_t n = f.read_raw_at(b.data, SECTOR, 3 * SECTOR, bad);
                if (n != SECTOR) {
                    oob_ok = false;
                    break;
                }
                for (size_t i = 0; i < SECTOR; i++) {
                    if (b.data[i] != pattern_byte(3 * SECTOR + i)) {
                        oob_ok = false;
                        break;
                    }
                }
                if (!oob_ok) {
                    break;
                }
            }
            check(oob_ok,
                  "an out-of-range worker id falls back to the shared handle and still reads");

            // All slots at once. This does not measure throughput - the file is a
            // megabyte and lives in RAM by now - it checks that nothing is shared that
            // should not be. If the OVERLAPPED structure or the byte counter were held
            // per FILE instead of per call, this is where the offsets would cross.
            const size_t n_threads = n_handles < 8 ? n_handles : 8;
            std::vector<int> ok((size_t) n_threads, 1);
            std::vector<std::thread> ts;
            for (size_t t = 0; t < n_threads; t++) {
                ts.emplace_back([&f, &ok, t]() {
                    // Each thread reads its OWN offset. Reading the same one everywhere
                    // would pass even if every thread ignored the offset it was given.
                    const size_t off = (t + 4) * SECTOR;
                    for (int rep = 0; rep < 16; rep++) {
                        aligned_buffer b(SECTOR);
                        if (f.read_raw_at(b.data, SECTOR, off, (int) t) != SECTOR) {
                            ok[t] = 0;
                            return;
                        }
                        for (size_t i = 0; i < SECTOR; i++) {
                            if (b.data[i] != pattern_byte(off + i)) {
                                ok[t] = 0;
                                return;
                            }
                        }
                    }
                });
            }
            for (auto & t : ts) {
                t.join();
            }
            bool concurrent_ok = true;
            for (size_t t = 0; t < n_threads; t++) {
                if (!ok[t]) {
                    concurrent_ok = false;
                }
            }
            check(concurrent_ok,
                  "all slots read their own offsets correctly at the same time");
        }
    }

    remove(path.c_str());

    printf("\n");
    if (g_failed != 0) {
        printf("RESULT: FAIL - %d check(s) failed\n", g_failed);
        return 1;
    }
    printf("RESULT: PASS - all checks behaved as required\n");
    return 0;
}
