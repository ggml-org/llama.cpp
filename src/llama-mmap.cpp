#include "llama-mmap.h"

#include "llama-impl.h"

#include "ggml.h"

#include <cstring>
#include <climits>
#include <stdexcept>
#include <cerrno>
#include <algorithm>

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #include <fcntl.h>
        #include <sys/stat.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
        #endif
        #if defined(_POSIX_MEMLOCK_RANGE)
            #include <sys/resource.h>
        #endif
    #endif
#endif

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
    // IOCTL_STORAGE_QUERY_PROPERTY and STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR: the device
    // is asked for its sector sizes rather than told what they are.
    #include <winioctl.h>
    // _aligned_malloc / _aligned_free, the Windows counterpart to posix_memalign.
    #include <malloc.h>
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#ifdef _WIN32
#    define llama_mmap_ftell _ftelli64
#    define llama_mmap_fseek _fseeki64
#else
#    define llama_mmap_ftell ftello
#    define llama_mmap_fseek fseeko
#endif

// TODO: consider moving to llama-impl.h if needed in more places
#if defined(_WIN32)
static std::string llama_format_win_err(DWORD err) {
    LPSTR buf;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
}
#endif

// llama_file

struct llama_file::impl {
#if defined(_WIN32)
    HANDLE fp_win32 = INVALID_HANDLE_VALUE;

    // True only when this class opened the handle itself with CreateFileW, which is
    // exactly when direct I/O is in effect. On the buffered path fp_win32 is derived
    // from the CRT stream and belongs to it - closing it here would be a double close.
    // The destructor asks about OWNERSHIP, not about mode, which is why this is not
    // called is_direct_io.
    bool owns_handle = false;

    // One private handle per concurrent reader. This is the whole point of the pool, and
    // it is not an optimisation of the OVERLAPPED read - it is the thing the OVERLAPPED
    // read could not buy.
    //
    // Measured 2026-08-03 on this machine, 12.75 MiB blocks, FILE_FLAG_NO_BUFFERING,
    // same file and offsets throughout: one shared handle read through an OVERLAPPED
    // offset reaches 1.01x at queue depth 8; one handle per thread reaches 2.22x. The
    // same pair through SetFilePointerEx gives 0.98x and 2.19x. The read mechanism makes
    // no difference at all; sharing the handle makes all of it. Windows serialises on the
    // file object, not on the file pointer.
    //
    // The count is a ceiling on worker ids, not a throughput setting. Saturation sits at
    // depth 8: depth 64 measured 2.15x against 2.22x at depth 8, so slots past the eighth
    // buy no bandwidth. They are there so that a caller running more readers than the
    // saturation point still gets a private handle for each of them - a worker id at or
    // past n_pool falls back to the shared handle and takes the serialisation with it,
    // which is the one failure this pool exists to prevent. 18 is that headroom, and the
    // price for it is 18 open handles on one file.
    //
    // n_pool counts what was actually opened, always contiguous from 0. That is why the
    // array needs no sentinel: slots at or past n_pool were never opened, so the
    // destructor closes exactly [0, n_pool) and an aggregate initialiser that fills with
    // NULL rather than INVALID_HANDLE_VALUE cannot turn into a CloseHandle(NULL).
    static constexpr int POOL_SLOTS = 18;
    HANDLE h_pool[POOL_SLOTS] = {};
    int    n_pool = 0;

    // The file position under direct I/O, kept here instead of in the kernel.
    //
    // Measured 2026-08-03: FILE_FLAG_NO_BUFFERING makes SetFilePointerEx reject any
    // unaligned position outright with ERROR_INVALID_PARAMETER. The POSIX branch does
    // not have this problem - lseek positions freely and only read() must be aligned -
    // which is why read_aligned_chunk, copied from there, could not work here at all.
    //
    // So seek() writes to this variable and every read goes through read_raw_at, whose
    // OVERLAPPED offset never touches the kernel's pointer. Note that this does NOT
    // lift the alignment requirement: the OVERLAPPED offset must be sector-aligned too.
    // It only removes the requirement from POSITIONING, which is what was fatal.
    size_t logical_pos = 0;

    std::string fname;

    std::string GetErrorMessageWin32(DWORD error_code) const {
        std::string ret;
        LPSTR lpMsgBuf = NULL;
        DWORD bufLen = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                    NULL, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&lpMsgBuf, 0, NULL);
        if (!bufLen) {
            ret = format("Win32 error code: %lx", error_code);
        } else {
            ret = lpMsgBuf;
            LocalFree(lpMsgBuf);
        }

        return ret;
    }

    impl(const char * fname, const char * mode, const bool use_direct_io = false) : fname(fname) {
        // Try unbuffered I/O for read only, mirroring what the POSIX branch does with
        // O_DIRECT. Until now this parameter was accepted and dropped: the constructor
        // took use_direct_io, ignored it, and has_direct_io() reported true regardless.
        // That combination is a promise the code did not keep.
        if (use_direct_io && std::strcmp(mode, "rb") == 0) {
            if (init_direct()) {
                return;
            }
            LLAMA_LOG_WARN("Failed to open file '%s' unbuffered: %s. Falling back to buffered I/O\n",
                           fname, GetErrorMessageWin32(GetLastError()).c_str());
        }
        init_fp(mode);
    }

    // Ask the device for its sector sizes. Returns 0 when the question cannot be
    // answered, which makes the caller fall back to buffered I/O WITH A WARNING.
    //
    // Deliberately no default of 4096. That constant is written three times across this
    // ecosystem and on this drive it happens to be right - measured 2026-08-03, 512
    // logical and 4096 physical. Correct by luck is not correct by construction, and a
    // silent default would turn "the device never answered" into something that looks
    // like an answer.
    //
    // The alignment DUTY of FILE_FLAG_NO_BUFFERING is the LOGICAL sector size; the
    // PHYSICAL one is Microsoft's performance recommendation. Reading aligned to the
    // logical size on a 512e drive is permitted and costs read-modify-write on the
    // controller, so the physical size is what we take.
    size_t query_sector_size(char drive_letter) const {
        char volume[] = "\\\\.\\X:";
        volume[4] = drive_letter;

        // Access 0 asks for metadata only; this needs no elevation.
        HANDLE hv = CreateFileA(volume, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                                OPEN_EXISTING, 0, NULL);
        if (hv == INVALID_HANDLE_VALUE) {
            return 0;
        }

        STORAGE_PROPERTY_QUERY query = {};
        query.PropertyId = StorageAccessAlignmentProperty;
        query.QueryType  = PropertyStandardQuery;

        STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR desc = {};
        DWORD returned = 0;
        const BOOL ok = DeviceIoControl(hv, IOCTL_STORAGE_QUERY_PROPERTY,
                                        &query, sizeof(query),
                                        &desc, sizeof(desc), &returned, NULL);
        CloseHandle(hv);

        if (!ok || returned < sizeof(desc) || desc.BytesPerPhysicalSector == 0) {
            return 0;
        }
        return (size_t) desc.BytesPerPhysicalSector;
    }

    bool init_direct() {
        // UTF-8 to UTF-16. ggml_fopen does the same conversion, but its helper is
        // static inside ggml.c and there is no exported equivalent.
        const int wlen = MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, NULL, 0);
        if (wlen == 0) {
            return false;
        }
        std::vector<wchar_t> wname(wlen);
        if (MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, wname.data(), wlen) == 0) {
            return false;
        }

        // Resolve to an absolute path BEFORE deriving the volume. The sector size can
        // only be asked of a drive letter, and a relative path has none - so a relative
        // path used to make direct I/O silently unavailable and fall back to buffered
        // reads. The loader happens to pass absolute paths, which is luck rather than
        // construction; the test below passes a relative one and is how this was found.
        wchar_t abs_path[MAX_PATH];
        const DWORD abs_len = GetFullPathNameW(wname.data(), MAX_PATH, abs_path, NULL);
        if (abs_len == 0 || abs_len >= MAX_PATH) {
            return false;
        }
        if (abs_path[1] != L':') {
            // A UNC path has no volume to ask, so the sector size stays unknown and
            // unbuffered reads cannot be aligned safely.
            SetLastError(ERROR_NOT_SUPPORTED);
            return false;
        }

        HANDLE h = CreateFileW(abs_path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
        if (h == INVALID_HANDLE_VALUE) {
            return false;
        }

        const size_t sector = query_sector_size((char) abs_path[0]);
        if (sector == 0) {
            CloseHandle(h);
            SetLastError(ERROR_NOT_SUPPORTED);
            return false;
        }

        LARGE_INTEGER li;
        if (!GetFileSizeEx(h, &li)) {
            const DWORD err = GetLastError();
            CloseHandle(h);
            SetLastError(err);
            return false;
        }

        fp_win32    = h;
        owns_handle = true;
        alignment   = sector;
        size        = (size_t) li.QuadPart;

        // Open the private handles here: single-threaded, after the path, the flags and
        // the sector size are all known good, and before anyone can read.
        //
        // A short pool is NOT fatal. read_raw_at falls back to the shared handle for any
        // slot that is missing, which costs throughput and correctness nothing. It is
        // logged rather than swallowed, because a pool that came up empty reads exactly
        // the same bytes as one that works - it simply never scales, and that is the
        // failure shape this file keeps finding.
        //
        // FILE_SHARE_READ, the same as the handle above. Win32 grants the second open
        // only if its access is permitted by every existing handle's share mode AND its
        // own share mode permits every existing handle's access; read-only on both sides
        // satisfies that. Asking for FILE_SHARE_WRITE here as well would still be granted
        // and would still mean two different sharing contracts on one file.
        for (int i = 0; i < POOL_SLOTS; i++) {
            const HANDLE hp = CreateFileW(abs_path, GENERIC_READ, FILE_SHARE_READ, NULL,
                                          OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
            if (hp == INVALID_HANDLE_VALUE) {
                LLAMA_LOG_WARN("%s: opened %d of %d private read handles for '%s': %s. "
                               "Concurrent reads fall back to the shared handle and will not scale\n",
                               __func__, i, POOL_SLOTS, fname.c_str(),
                               GetErrorMessageWin32(GetLastError()).c_str());
                break;
            }
            h_pool[n_pool++] = hp;
        }
        return true;
    }

    void init_fp(const char * mode) {
        fp = ggml_fopen(fname.c_str(), mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname.c_str(), strerror(errno)));
        }
        fp_win32 = (HANDLE) _get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    impl(FILE * file) : fname("(file*)"), owns_fp(false) {
        fp = file;
        fp_win32 = (HANDLE) _get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        if (owns_handle) {
            return logical_pos;
        }

        LARGE_INTEGER li;
        li.QuadPart = 0;
        BOOL ret = SetFilePointerEx(fp_win32, li, &li, FILE_CURRENT);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }

        return li.QuadPart;
    }

    void seek(size_t offset, int whence) {
        static_assert(SEEK_SET == FILE_BEGIN, "SEEK_SET != FILE_BEGIN");
        static_assert(SEEK_CUR == FILE_CURRENT, "SEEK_CUR != FILE_CURRENT");
        static_assert(SEEK_END == FILE_END, "SEEK_END != FILE_END");

        if (owns_handle) {
            // Never ask the kernel. See the note on logical_pos: an unaligned position
            // is refused outright here, and refusing to seek is not something callers
            // expect from seek().
            switch (whence) {
                case SEEK_SET: logical_pos = offset;                break;
                case SEEK_CUR: logical_pos += offset;               break;
                case SEEK_END: logical_pos = size + offset;         break;
                default:
                    throw std::runtime_error(format("seek error: bad whence %d", whence));
            }
            return;
        }

        LARGE_INTEGER li;
        li.QuadPart = offset;
        BOOL ret = SetFilePointerEx(fp_win32, li, NULL, whence);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }
    }

    void read_raw_unsafe(void * ptr, size_t len) {
        if (owns_handle) {
            // Direct I/O never uses the kernel's file pointer - see logical_pos.
            const size_t got = read_raw_at(ptr, len, logical_pos);
            if (got < len) {
                // End of file. An aligned request necessarily overshoots a file whose
                // size is not a sector multiple, and none of the four model files here
                // ends on a boundary, so this is the normal case rather than an edge
                // one. Measured 2026-08-03 on all four: ReadFile returns TRUE and
                // reports exactly the bytes up to the logical EOF. Zero the padding and
                // carry on; the caller knows the tensor size and trims it. Same
                // behaviour as the POSIX branch.
                std::memset(reinterpret_cast<char*>(ptr) + got, 0, len - got);
            }
            logical_pos += got;
            return;
        }

        size_t bytes_read = 0;
        while (bytes_read < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_read, 64*1024*1024);
            DWORD chunk_read = 0;
            BOOL result = ReadFile(fp_win32, reinterpret_cast<char*>(ptr) + bytes_read, chunk_size, &chunk_read, NULL);
            if (!result) {
                throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_read < chunk_size || chunk_read == 0) {
                // A short count at the end of the file. Under direct I/O this is the
                // NORMAL case, not an edge one: an aligned request necessarily runs past
                // a file whose size is not a sector multiple, and none of the four model
                // files on this machine ends on a sector boundary.
                //
                // Measured 2026-08-03 on all four: with FILE_FLAG_NO_BUFFERING, ReadFile
                // returns TRUE and reports exactly the bytes up to the logical EOF. The
                // padding is zeroed and the caller - which knows the tensor size and
                // trims the padding itself - carries on. This mirrors the POSIX branch.
                //
                // Reached only on the buffered path now, which never asks for more than
                // it wants - so a short count here is still a real failure, exactly as
                // it was before this file learned about direct I/O.
                throw std::runtime_error("unexpectedly reached end of file");
            }

            bytes_read += chunk_read;
        }
    }

    // Read `size_to_read` bytes from the current position, coping with an offset or
    // length that direct I/O will not accept. Reads the enclosing sector-aligned range
    // into an aligned bounce buffer and copies out the part that was asked for.
    //
    // Lives here as an implementation detail, not in the header. The header used to
    // declare read_aligned_chunk() publicly while no llama_file method of that name
    // existed on any platform and nobody called it - a promise with nothing behind it.
    void read_aligned_chunk(void * dest, size_t size_to_read) {
        const size_t offset                = tell();
        const size_t aligned_offset        = offset & ~(alignment - 1);
        const size_t offset_from_alignment = offset - aligned_offset;
        const size_t bytes_to_read         = (offset_from_alignment + size_to_read + alignment - 1) & ~(alignment - 1);

        // The buffer ADDRESS has to be sector-aligned too, not just the offset and the
        // length. Missing that is what turned the first attempt at this into a
        // 0xC0000409 with no output at all.
        void * raw_buffer = _aligned_malloc(bytes_to_read, alignment);
        if (raw_buffer == nullptr) {
            throw std::runtime_error(format("_aligned_malloc of %zu bytes failed", bytes_to_read));
        }

        struct aligned_buffer_deleter {
            void operator()(void * p) const { _aligned_free(p); }
        };
        std::unique_ptr<void, aligned_buffer_deleter> buffer(raw_buffer);

        // Reads at the ALIGNED offset directly, without moving any file pointer. The
        // earlier form seek(aligned_offset) + read was copied from POSIX and could not
        // work here: the caller's unaligned position had already been rejected by the
        // kernel before this function was ever reached.
        const size_t got = read_raw_at(buffer.get(), bytes_to_read, aligned_offset);
        if (got < offset_from_alignment + size_to_read) {
            throw std::runtime_error("unexpectedly reached end of file");
        }

        std::memcpy(dest, reinterpret_cast<char*>(buffer.get()) + offset_from_alignment, size_to_read);

        // The logical position advances by what the caller asked for, not by what the
        // bounce buffer had to read around it.
        logical_pos = offset + size_to_read;
    }

    void read_raw(void * ptr, size_t len) {
        if (has_direct_io()) {
            read_aligned_chunk(ptr, len);
        } else {
            read_raw_unsafe(ptr, len);
        }
    }

    // The positional read Windows never had in llama_file. The offset travels in the
    // OVERLAPPED structure, so no seek is needed and the shared file pointer is not
    // touched - which is what makes it SAFE to call from several threads on one handle.
    //
    // Safe was never the same as concurrent, and that distinction cost a measurement to
    // establish. Passing OVERLAPPED to a handle opened without FILE_FLAG_OVERLAPPED does
    // not make the read asynchronous: measured 2026-08-03, one shared handle across 8
    // threads reaches 1.01x the throughput of a single thread whether the offset comes
    // from the structure or from SetFilePointerEx. A private handle each reaches 2.22x.
    //
    // Hence worker_id. It selects this caller's own handle out of the pool, and no lock
    // guards that selection because none is needed: each index is read by exactly one
    // thread, and the array is filled once in init_direct before any reader exists.
    size_t read_raw_at(void * ptr, size_t len, size_t offset, int worker_id = -1) {
        // Anything the pool does not cover - a negative id, an id past what opened, or a
        // buffered file that has no pool at all - reads through the shared handle. That
        // is correct and merely serialised, which is the right way round: a caller that
        // knows nothing about pools must not be able to index past the array.
        const HANDLE h_read = (worker_id >= 0 && worker_id < n_pool) ? h_pool[worker_id] : fp_win32;

        size_t total = 0;
        while (total < len) {
            const size_t chunk_size = std::min<size_t>(len - total, 64*1024*1024);
            const size_t pos = offset + total;

            OVERLAPPED ov = {};
            ov.Offset     = (DWORD) (pos & 0xFFFFFFFFull);
            ov.OffsetHigh = (DWORD) (pos >> 32);

            DWORD chunk_read = 0;
            if (!ReadFile(h_read, reinterpret_cast<char*>(ptr) + total, (DWORD) chunk_size, &chunk_read, &ov)) {
                const DWORD err = GetLastError();
                if (err == ERROR_HANDLE_EOF) {
                    return total;
                }
                throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(err).c_str()));
            }
            total += chunk_read;
            if (chunk_read < chunk_size) {
                // Short read means end of file - see the note in read_raw_unsafe.
                return total;
            }
        }
        return total;
    }

    uint32_t read_u32() {
        uint32_t val;
        read_raw(&val, sizeof(val));
        return val;
    }

    void write_raw(const void * ptr, size_t len) const {
        size_t bytes_written = 0;
        while (bytes_written < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_written, 64*1024*1024);
            DWORD chunk_written = 0;
            BOOL result = WriteFile(fp_win32, reinterpret_cast<char const*>(ptr) + bytes_written, chunk_size, &chunk_written, NULL);
            if (!result) {
                throw std::runtime_error(format("write error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_written < chunk_size || chunk_written == 0) {
                throw std::runtime_error("unexpectedly failed to write bytes");
            }

            bytes_written += chunk_written;
        }
    }

    void write_u32(uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    // Used to return true unconditionally, on a branch that never opened anything
    // unbuffered. The single in-tree caller sits in the POSIX branch, so the lie was
    // harmless in this repository and would only have been found by external code
    // asking the question. Any caller that logs "O_DIRECT in effect" on the strength of
    // this answer would have printed it on Windows while reading through the page cache.
    //
    // Asks about ownership AND alignment, so it stays honest when the constructor fell
    // back to buffered I/O.
    bool has_direct_io() const {
        return owns_handle && alignment > 1;
    }

    size_t direct_io_handles() const {
        return (size_t) n_pool;
    }

    ~impl() {
        // The pool first. These handles are always ours - nothing else can hold them,
        // since they are opened in init_direct and handed out by value nowhere. Closing
        // exactly n_pool of them is what lets the array go without a sentinel: the slots
        // from n_pool upward were never opened, so they are NULL rather than
        // INVALID_HANDLE_VALUE and must not reach CloseHandle at all.
        for (int i = 0; i < n_pool; i++) {
            CloseHandle(h_pool[i]);
        }
        n_pool = 0;

        if (owns_handle) {
            if (fp_win32 != INVALID_HANDLE_VALUE) {
                CloseHandle(fp_win32);
            }
        } else if (fp && owns_fp) {
            // Buffered path: the CRT stream owns fp_win32. Closing both would be a
            // double close, and closing fp when owns_fp is false would close a stream
            // that belongs to the caller.
            std::fclose(fp);
        }
    }
#else
    impl(const char * fname, const char * mode, [[maybe_unused]] const bool use_direct_io = false) : fname(fname) {
#ifdef __linux__
        // Try unbuffered I/O for read only
        if (use_direct_io && std::strcmp(mode, "rb") == 0) {
            if (init_fd()) {
                return;
            }
            LLAMA_LOG_WARN("Failed to open file '%s' with error: %s. Falling back to buffered I/O",
                           fname, strerror(errno));
        }
#endif
        init_fp(mode);
    }

#ifdef __linux__
    bool init_fd() {
        fd = open(fname.c_str(), O_RDONLY | O_DIRECT);

        if (fd != -1) {
            struct stat file_stats{};
            fstat(fd, &file_stats);

            size = file_stats.st_size;
            alignment = file_stats.st_blksize;

            off_t ret = lseek(fd, 0, SEEK_SET);
            if (ret == -1) {
                throw std::runtime_error(format("seek error: %s", strerror(errno)));
            }
            return true;
        }
        return false;
    }
#endif

    void init_fp(const char * mode) {
        fp = ggml_fopen(fname.c_str(), mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname.c_str(), strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    impl(FILE * file) : fname("(file*)"), owns_fp(false) {
        fp = file;
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        if (fd == -1) {
            off_t ret = llama_mmap_ftell(fp);
            if (ret == -1) {
                throw std::runtime_error(format("ftell error: %s", strerror(errno)));
            }

            return (size_t) ret;
        }

        off_t pos = lseek(fd, 0, SEEK_CUR);
        if (pos == -1) {
            throw std::runtime_error(format("lseek error: %s", strerror(errno)));
        }
        return (size_t) pos;
    }

    void seek(size_t offset, int whence) const {
        off_t ret = 0;
        if (fd == -1) {
            ret = llama_mmap_fseek(fp, offset, whence);
        } else {
            ret = lseek(fd, offset, whence);
        }
        if (ret == -1) {
            throw std::runtime_error(format("seek error: %s", strerror(errno)));
        }
    }

    void read_raw_unsafe(void * ptr, size_t len) {
        if (len == 0) {
            return;
        }
        errno = 0;
        if (fd == -1) {
            const size_t curr_off = tell();
            const size_t to_read = std::min(len, size - curr_off);

            std::size_t ret = std::fread(ptr, to_read, 1, fp);
            if (ferror(fp)) {
                throw std::runtime_error(format("read error: %s", strerror(errno)));
            }
            if (to_read > 0 && ret != 1) {
                throw std::runtime_error("unexpectedly reached end of file");
            }
        } else {
            size_t bytes_read = 0;
            while (bytes_read < len) {
                const size_t to_read = len - bytes_read;
                ssize_t ret = ::read(fd, reinterpret_cast<char *>(ptr) + bytes_read, to_read);

                if (ret == -1) {
                    if (errno == EINTR) {
                        continue;  // Interrupted by signal, retry
                    }
                    // Fallback to std::fread in case the DMA controller cannot access the buffer
                    if (errno == EFAULT || errno == EINVAL) {
                        LLAMA_LOG_WARN("%s: Falling back to buffered IO due to %s\n", __func__, strerror(errno));
                        auto curr_off = tell();
                        close(fd);
                        fd = -1;
                        alignment = 1;
                        init_fp("rb");
                        seek(curr_off, SEEK_SET);
                        read_raw_unsafe(ptr, len);
                        return;
                    }
                    throw std::runtime_error(format("read error: %s", strerror(errno)));
                }
                if (ret == 0) {
                    // EOF: allow if this read was only pulling alignment padding past file end
                    off_t pos = lseek(fd, 0, SEEK_CUR);
                    if (pos != -1 && (size_t) pos == size) {
                        std::memset(reinterpret_cast<char *>(ptr) + bytes_read, 0, len - bytes_read);
                        return;
                    }
                    throw std::runtime_error("unexpectedly reached end of file");
                }

                bytes_read += (size_t) ret;
            }
        }
    }

    void read_aligned_chunk(void * dest, size_t size) {
        size_t offset = tell();
        off_t aligned_offset = offset & ~(alignment - 1);
        off_t offset_from_alignment = offset - aligned_offset;
        size_t bytes_to_read = (offset_from_alignment + size + alignment - 1) & ~(alignment - 1);

        void * raw_buffer = nullptr;
        int ret = posix_memalign(&raw_buffer, alignment, bytes_to_read);
        if (ret != 0) {
            throw std::runtime_error(format("posix_memalign failed with error %d", ret));
        }

        struct aligned_buffer_deleter {
            void operator()(void * p) const { free(p); }
        };
        std::unique_ptr<void, aligned_buffer_deleter> buffer(raw_buffer);

        seek(aligned_offset, SEEK_SET);
        read_raw_unsafe(buffer.get(), bytes_to_read);

        uintptr_t actual_data = reinterpret_cast<uintptr_t>(buffer.get()) + offset_from_alignment;
        memcpy(dest, reinterpret_cast<void *>(actual_data), size);
    }

    void read_raw(void * ptr, size_t len) {
        if (has_direct_io()) {
            read_aligned_chunk(ptr, len);
        } else {
            read_raw_unsafe(ptr, len);
        }
    }

    // Positional read; see the Windows counterpart for what it is for. pread does not
    // move the file pointer, which is what lets two threads share one descriptor.
    size_t read_raw_at(void * ptr, size_t len, size_t offset, int worker_id = -1) {
        // POSIX needs no handle pool. pread carries its offset in the call and does not
        // hold the file object while it runs, so several threads on ONE descriptor are
        // already concurrent. Windows has no such call, which is why the pool exists over
        // there and not here. The parameter is accepted on both so that a caller need not
        // know which platform it is compiled for.
        (void) worker_id;

#if defined(fileno)
        const int use_fd = (fd != -1) ? fd : fileno(fp);
#else
        const int use_fd = (fd != -1) ? fd : ::fileno(fp);
#endif
        size_t total = 0;
        while (total < len) {
            const ssize_t ret = ::pread(use_fd, reinterpret_cast<char *>(ptr) + total,
                                        len - total, (off_t) (offset + total));
            if (ret == -1) {
                if (errno == EINTR) {
                    continue;  // Interrupted by signal, retry
                }
                throw std::runtime_error(format("read error: %s", strerror(errno)));
            }
            if (ret == 0) {
                return total;  // end of file
            }
            total += (size_t) ret;
        }
        return total;
    }

    uint32_t read_u32() {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    bool has_direct_io() const {
        return fd != -1 && alignment > 1;
    }

    // Always zero here, and that is the honest answer rather than a stub: this branch
    // holds no private handles because pread does not need them. A caller comparing the
    // number against its thread count learns the right thing on both platforms.
    size_t direct_io_handles() const {
        return 0;
    }

    ~impl() {
        if (fd != -1) {
            close(fd);
        } else if (owns_fp) {
            std::fclose(fp);
        }
    }
    int fd = -1;
    std::string fname;
#endif

    size_t read_alignment() const {
        return alignment;
    }

    size_t alignment = 1;

    FILE * fp{};
    size_t size{};
    bool owns_fp = true;
};

llama_file::llama_file(const char * fname, const char * mode, const bool use_direct_io) :
    pimpl(std::make_unique<impl>(fname, mode, use_direct_io)) {}

llama_file::llama_file(FILE * file) : pimpl(std::make_unique<impl>(file)) {}

llama_file::~llama_file() = default;

size_t llama_file::tell() const { return pimpl->tell(); }
size_t llama_file::size() const { return pimpl->size; }

size_t llama_file::read_alignment() const { return pimpl->read_alignment(); }
bool llama_file::has_direct_io() const { return pimpl->has_direct_io(); }

int llama_file::file_id() const {
#ifdef _WIN32
    if (pimpl->owns_handle) {
        // A direct-I/O file has no CRT descriptor, and a HANDLE cannot be squeezed
        // through an int: it is a 64-bit pointer, so a cast drops the upper half and
        // yields something that still looks like a valid handle and reads zero bytes
        // without raising. That exact truncation is easy to hit through any FFI layer
        // that defaults a handle-returning call to a 32-bit result type.
        //
        // An exception rather than GGML_ASSERT on purpose: an assert can be compiled
        // out by build configuration, and a guard that disappears in release is no
        // guard at all. Failing here is safe because the two are mutually exclusive by
        // construction - the model loader only opens files unbuffered for
        // LLAMA_LOAD_MODE_DIRECT_IO, and that mode does not use mmap.
        throw std::runtime_error("file_id() is not available on a direct-I/O handle on Windows");
    }
    return _fileno(pimpl->fp);
#else
    if (pimpl->fd != -1) {
        return pimpl->fd;
    }
#if defined(fileno)
    return fileno(pimpl->fp);
#else
    return ::fileno(pimpl->fp);
#endif
#endif
}

void llama_file::seek(size_t offset, int whence) const { pimpl->seek(offset, whence); }
void llama_file::read_raw(void * ptr, size_t len) { pimpl->read_raw(ptr, len); }

// The #ifdef that used to sit here routed Windows to read_raw, because the Windows
// impl had no read_raw_unsafe at all. It has one now - the same loop as before plus
// the end-of-file case - so the buffered path is unchanged.
void llama_file::read_raw_unsafe(void * ptr, size_t len) { pimpl->read_raw_unsafe(ptr, len); }

size_t llama_file::read_raw_at(void * ptr, size_t len, size_t offset, int worker_id) {
    return pimpl->read_raw_at(ptr, len, offset, worker_id);
}

size_t llama_file::direct_io_handles() const { return pimpl->direct_io_handles(); }

uint32_t llama_file::read_u32() { return pimpl->read_u32(); }

void llama_file::write_raw(const void * ptr, size_t len) const { pimpl->write_raw(ptr, len); }
void llama_file::write_u32(uint32_t val) const { pimpl->write_u32(val); }

// llama_mmap

struct llama_mmap::impl {
#ifdef _POSIX_MAPPED_FILES
    std::vector<std::pair<size_t, size_t>> mapped_fragments;

    impl(struct llama_file * file, size_t prefetch, bool numa) {
        size = file->size();
        int fd = file->file_id();
        int flags = MAP_SHARED;
        if (numa) { prefetch = 0; }
#ifdef __linux__
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
            LLAMA_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                    strerror(errno));
        }
        if (prefetch) { flags |= MAP_POPULATE; }
#endif
        addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        if (prefetch > 0) {
            if (posix_madvise(addr, std::min(file->size(), prefetch), POSIX_MADV_WILLNEED)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n",
                        strerror(errno));
            }
        }
        if (numa) {
            if (posix_madvise(addr, file->size(), POSIX_MADV_RANDOM)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                        strerror(errno));
            }
        }

        mapped_fragments.emplace_back(0, file->size());
    }

    static void align_range(size_t * first, size_t * last, size_t page_size) {
        size_t offset_in_page = *first & (page_size - 1);
        size_t offset_to_page = offset_in_page == 0 ? 0 : page_size - offset_in_page;
        *first += offset_to_page;

        *last = *last & ~(page_size - 1);

        if (*last <= *first) {
            *last = *first;
        }
    }

    void unmap_fragment(size_t first, size_t last) {
        int page_size = sysconf(_SC_PAGESIZE);
        align_range(&first, &last, page_size);
        size_t len = last - first;

        if (len == 0) {
            return;
        }

        GGML_ASSERT(first % page_size == 0);
        GGML_ASSERT(last % page_size == 0);
        GGML_ASSERT(last > first);

        void * next_page_start = (uint8_t *) addr + first;

        if (munmap(next_page_start, len)) {
            LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
        }

        std::vector<std::pair<size_t, size_t>> new_mapped_fragments;
        for (const auto & frag : mapped_fragments) {
            if (frag.first < first && frag.second > last) {
                new_mapped_fragments.emplace_back(frag.first, first);
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first < first && frag.second > first) {
                new_mapped_fragments.emplace_back(frag.first, first);
            } else if (frag.first < last && frag.second > last) {
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first >= first && frag.second <= last) {
            } else {
                new_mapped_fragments.push_back(frag);
            }
        }
        mapped_fragments = std::move(new_mapped_fragments);
    }

    ~impl() {
        for (const auto & frag : mapped_fragments) {
            if (munmap((char *) addr + frag.first, frag.second - frag.first)) {
                LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
            }
        }
    }
#elif defined(_WIN32)
    HANDLE hMapping = nullptr;

    impl(struct llama_file * file, size_t prefetch, bool numa) {
        GGML_UNUSED(numa);

        size = file->size();

        HANDLE hFile = (HANDLE) _get_osfhandle(file->file_id());

        hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

        if (hMapping == NULL) {
            DWORD error = GetLastError();
            throw std::runtime_error(format("CreateFileMappingA failed: %s", llama_format_win_err(error).c_str()));
        }

        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        DWORD error = GetLastError();

        if (addr == NULL) {
            CloseHandle(hMapping);
            throw std::runtime_error(format("MapViewOfFile failed: %s", llama_format_win_err(error).c_str()));
        }

        if (prefetch > 0) {
#if _WIN32_WINNT >= 0x602
            BOOL (WINAPI *pPrefetchVirtualMemory) (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

            pPrefetchVirtualMemory = (decltype(pPrefetchVirtualMemory))(void *) GetProcAddress(hKernel32, "PrefetchVirtualMemory");

            if (pPrefetchVirtualMemory) {
                WIN32_MEMORY_RANGE_ENTRY range;
                range.VirtualAddress = addr;
                range.NumberOfBytes = (SIZE_T) std::min(size, prefetch);
                if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                    LLAMA_LOG_WARN("warning: PrefetchVirtualMemory failed: %s\n",
                            llama_format_win_err(GetLastError()).c_str());
                }
            }
#else
            LLAMA_LOG_DEBUG("skipping PrefetchVirtualMemory because _WIN32_WINNT < 0x602\n");
#endif
        }
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);
    }

    ~impl() {
        if (hMapping) {
            if (addr) {
                if (!UnmapViewOfFile(addr)) {
                    LLAMA_LOG_WARN("warning: UnmapViewOfFile failed: %s\n",
                            llama_format_win_err(GetLastError()).c_str());
                }
            }
            if (!CloseHandle(hMapping)) {
                LLAMA_LOG_WARN("warning: CloseHandle failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
            }
        }
    }
#else
    impl(struct llama_file * file, size_t prefetch, bool numa) {
        GGML_UNUSED(file);
        GGML_UNUSED(prefetch);
        GGML_UNUSED(numa);

        throw std::runtime_error("mmap not supported");
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);

        throw std::runtime_error("mmap not supported");
    }
#endif

    void * addr;
    size_t size;
};

llama_mmap::llama_mmap(struct llama_file * file, size_t prefetch, bool numa) : pimpl(std::make_unique<impl>(file, prefetch, numa)) {}
llama_mmap::~llama_mmap() = default;

size_t llama_mmap::size() const { return pimpl->size; }
void * llama_mmap::addr() const { return pimpl->addr; }

void llama_mmap::unmap_fragment(size_t first, size_t last) { pimpl->unmap_fragment(first, last); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool llama_mmap::SUPPORTED  = true;
#else
const bool llama_mmap::SUPPORTED  = false;
#endif

// llama_mlock

struct llama_mlock::impl {
#ifdef _POSIX_MEMLOCK_RANGE
    static size_t lock_granularity() {
        return (size_t) sysconf(_SC_PAGESIZE);
    }

    bool raw_lock(const void * addr, size_t size) const {
        if (!mlock(addr, size)) {
            return true;
        }

#ifdef __APPLE__
#define MLOCK_SUGGESTION \
        "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
        "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MEMLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION \
        "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n"
#endif

        char* errmsg = std::strerror(errno);
        bool suggest = (errno == ENOMEM);
#if defined(TARGET_OS_VISION) || defined(TARGET_OS_TV) || defined(_AIX) || defined(__HAIKU__)
        // visionOS/tvOS/Haiku don't support RLIMIT_MEMLOCK
        // Skip resource limit checks on these platforms
        suggest = false;
#else
        struct rlimit lock_limit;
        if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
            suggest = false;
        }
        if (suggest && ((uint64_t)lock_limit.rlim_max > (uint64_t)lock_limit.rlim_cur + size)) {
            suggest = false;
        }
#endif

        LLAMA_LOG_WARN("warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
                size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
        return false;
    }

    static void raw_unlock(void * addr, size_t size) {
        if (munlock(addr, size)) {
            LLAMA_LOG_WARN("warning: failed to munlock buffer: %s\n", std::strerror(errno));
        }
    }
#elif defined(_WIN32)
    static size_t lock_granularity() {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (size_t) si.dwPageSize;
    }

    bool raw_lock(void * ptr, size_t len) const {
        for (int tries = 1; ; tries++) {
            if (VirtualLock(ptr, len)) {
                return true;
            }
            if (tries == 2) {
                LLAMA_LOG_WARN("warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n",
                    len, size, llama_format_win_err(GetLastError()).c_str());
                return false;
            }

            SIZE_T min_ws_size, max_ws_size;
            if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
                LLAMA_LOG_WARN("warning: GetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
            size_t increment = len + 1048576;
            min_ws_size += increment;
            max_ws_size += increment;
            if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size, max_ws_size)) {
                LLAMA_LOG_WARN("warning: SetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
        }
    }

    static void raw_unlock(void * ptr, size_t len) {
        if (!VirtualUnlock(ptr, len)) {
            LLAMA_LOG_WARN("warning: failed to VirtualUnlock buffer: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static size_t lock_granularity() {
        return (size_t) 65536;
    }

    bool raw_lock(const void * addr, size_t len) const {
        LLAMA_LOG_WARN("warning: mlock not supported on this system\n");
        return false;
    }

    static void raw_unlock(const void * addr, size_t len) {}
#endif

    impl() : addr(NULL), size(0), failed_already(false) {}

    void init(void * ptr) {
        GGML_ASSERT(addr == NULL && size == 0);
        addr = ptr;
    }

    void grow_to(size_t target_size) {
        GGML_ASSERT(addr);
        if (failed_already) {
            return;
        }
        size_t granularity = lock_granularity();
        target_size = (target_size + granularity - 1) & ~(granularity - 1);
        if (target_size > size) {
            if (raw_lock((uint8_t *) addr + size, target_size - size)) {
                size = target_size;
            } else {
                failed_already = true;
            }
        }
    }

    void * addr;
    size_t size;

    bool failed_already;
};

llama_mlock::llama_mlock() : pimpl(std::make_unique<impl>()) {}
llama_mlock::~llama_mlock() = default;

void llama_mlock::init(void * ptr) { pimpl->init(ptr); }
void llama_mlock::grow_to(size_t target_size) { pimpl->grow_to(target_size); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool llama_mlock::SUPPORTED = true;
#else
const bool llama_mlock::SUPPORTED = false;
#endif

size_t llama_path_max() {
    return PATH_MAX;
}
