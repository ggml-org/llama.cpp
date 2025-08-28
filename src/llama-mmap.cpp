#include "llama-mmap.h"

#include "llama-impl.h"
#include "uint8-buff-stream.h"

#include "ggml.h"

#include <cstring>
#include <climits>
#include <stdexcept>
#include <cerrno>
#include <algorithm>
#include <map>

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
            #include <fcntl.h>
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
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
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

struct llama_file_disk::impl {
#if defined(_WIN32)
    HANDLE fp_win32;
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

    impl(const char * fname, const char * mode) {
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        fp_win32 = (HANDLE) _get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        LARGE_INTEGER li;
        li.QuadPart = 0;
        BOOL ret = SetFilePointerEx(fp_win32, li, &li, FILE_CURRENT);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }

        return li.QuadPart;
    }

    void seek(size_t offset, int whence) const {
        static_assert(SEEK_SET == FILE_BEGIN, "SEEK_SET != FILE_BEGIN");
        static_assert(SEEK_CUR == FILE_CURRENT, "SEEK_CUR != FILE_CURRENT");
        static_assert(SEEK_END == FILE_END, "SEEK_END != FILE_END");

        LARGE_INTEGER li;
        li.QuadPart = offset;
        BOOL ret = SetFilePointerEx(fp_win32, li, NULL, whence);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }
    }

    void read_raw(void * ptr, size_t len) const {
        size_t bytes_read = 0;
        while (bytes_read < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_read, 64*1024*1024);
            DWORD chunk_read = 0;
            BOOL result = ReadFile(fp_win32, reinterpret_cast<char*>(ptr) + bytes_read, chunk_size, &chunk_read, NULL);
            if (!result) {
                throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_read < chunk_size || chunk_read == 0) {
                throw std::runtime_error("unexpectedly reached end of file");
            }

            bytes_read += chunk_read;
        }
    }

    uint32_t read_u32() const {
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

    ~impl() {
        if (fp) {
            std::fclose(fp);
        }
    }
#else
    impl(const char * fname, const char * mode) {
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
// TODO: this ifdef is never true?
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        if (ret == -1) {
            throw std::runtime_error(format("ftell error: %s", strerror(errno)));
        }

        return (size_t) ret;
    }

    void seek(size_t offset, int whence) const {
// TODO: this ifdef is never true?
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        if (ret != 0) {
            throw std::runtime_error(format("seek error: %s", strerror(errno)));
        }
    }

    void read_raw(void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, len, 1, fp);
        if (ferror(fp)) {
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1) {
            throw std::runtime_error("unexpectedly reached end of file");
        }
    }

    uint32_t read_u32() const {
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

    ~impl() {
        if (fp) {
            std::fclose(fp);
        }
    }
#endif

    FILE * fp;
    size_t size;
};

llama_file_disk::llama_file_disk(const char * fname, const char * mode) : pimpl(std::make_unique<impl>(fname, mode)) {}
llama_file_disk::~llama_file_disk() = default;

size_t llama_file_disk::tell() const { return pimpl->tell(); }
size_t llama_file_disk::size() const { return pimpl->size; }

int llama_file_disk::file_id() const {
#ifdef _WIN32
    return _fileno(pimpl->fp);
#else
#if defined(fileno)
    return fileno(pimpl->fp);
#else
    return ::fileno(pimpl->fp);
#endif
#endif
}

void llama_file_disk::seek(size_t offset, int whence) const { pimpl->seek(offset, whence); }
void llama_file_disk::read_raw(void * ptr, size_t len) const { pimpl->read_raw(ptr, len); }

uint32_t llama_file_disk::read_u32() const { return pimpl->read_u32(); }

void llama_file_disk::write_raw(const void * ptr, size_t len) const { pimpl->write_raw(ptr, len); }
void llama_file_disk::write_u32(uint32_t val) const { pimpl->write_u32(val); }

template <bool Writable>
llama_file_buffer<Writable>::llama_file_buffer(std::unique_ptr<std::basic_streambuf<uint8_t>> && streambuf) :
    streambuf(std::move(streambuf)) {}

template <bool Writable> llama_file_buffer<Writable>::~llama_file_buffer() = default;

template <bool Writable> size_t llama_file_buffer<Writable>::tell() const {
    return streambuf->pubseekoff(0, std::ios_base::cur);
}

template <bool Writable> size_t llama_file_buffer<Writable>::size() const {
    auto current_pos = streambuf->pubseekoff(0, std::ios_base::cur);
    auto end_pos     = streambuf->pubseekoff(0, std::ios_base::end);
    streambuf->pubseekpos(current_pos);
    return end_pos;
}

template <bool Writable> int llama_file_buffer<Writable>::file_id() const {
    return -1;
}

template <bool Writable> void llama_file_buffer<Writable>::seek(size_t offset, int whence) const {
    static std::map<int, std::ios_base::seekdir> whence_to_dir = {
        { SEEK_SET, std::ios_base::beg },
        { SEEK_CUR, std::ios_base::cur },
        { SEEK_END, std::ios_base::end }
    };
    auto result = streambuf->pubseekoff(offset, whence_to_dir.at(whence));
    if (result == std::streampos(-1)) {
        throw std::runtime_error("seek failed");
    }
}

template <bool Writable> void llama_file_buffer<Writable>::read_raw(void * ptr, size_t len) const {
    auto bytes_read = streambuf->sgetn(static_cast<uint8_t *>(ptr), len);
    if (bytes_read != static_cast<std::streamsize>(len)) {
        throw std::runtime_error("read beyond end of buffer");
    }
}

template <bool Writable> uint32_t llama_file_buffer<Writable>::read_u32() const {
    uint32_t val;
    read_raw(&val, sizeof(val));
    return val;
}

template <> void llama_file_buffer<false>::write_raw([[maybe_unused]] const void * ptr, size_t len) const {
    if (len > 0) {
        throw std::runtime_error("buffer is not writable");
    }
}

template <> void llama_file_buffer<false>::write_u32(uint32_t val) const {
    if (val > 0) {
        // Cannot directly set [[noreturn]] for a function since it was defined without it.
        throw std::runtime_error("buffer is not writable");
    }
}

template <> void llama_file_buffer<true>::write_raw(const void * ptr, size_t len) const {
    auto bytes_written = streambuf->sputn(static_cast<const uint8_t *>(ptr), len);
    if (bytes_written != static_cast<std::streamsize>(len)) {
        throw std::runtime_error("write beyond end of buffer");
    }
}

template <> void llama_file_buffer<true>::write_u32(uint32_t val) const {
    write_raw(&val, sizeof(val));
}

// Explicit instantiations
template struct llama_file_buffer<false>;
template struct llama_file_buffer<true>;

// llama_future_file_buffer implementation

namespace {
std::string final_key(const std::string & promise_key, const std::string & context) {
    return promise_key + ":" + context;
}

std::mutex promise_registry_mutex;

std::map<std::string, std::promise<std::unique_ptr<llama_file_buffer<false>>>> promise_registry_ro;
std::map<std::string, std::promise<std::unique_ptr<llama_file_buffer<true>>>>  promise_registry_rw;

template <bool Writable>
std::map<std::string, std::promise<std::unique_ptr<llama_file_buffer<Writable>>>> & promise_registry() {
    if constexpr (Writable) {
        return promise_registry_rw;
    } else {
        return promise_registry_ro;
    }
}

/// @brief Ensures a promise exists in the registry for the given key.
/// If it doesn't exist, creates it. Returns an iterator to the promise.
/// Thread-safe.
template <bool Writable>
typename std::map<std::string, std::promise<std::unique_ptr<llama_file_buffer<Writable>>>>::iterator
ensure_promise_registry(const std::string & key) {
    std::lock_guard<std::mutex> lock(promise_registry_mutex);
    auto                        it = promise_registry<Writable>().find(key);
    if (it != promise_registry<Writable>().end()) {
        return it;
    }
    auto result =
        promise_registry<Writable>().emplace(key, std::promise<std::unique_ptr<llama_file_buffer<Writable>>>());
    LLAMA_LOG_CMAKE_DEBUG("%s: created future file buffer %p for %s\n", __func__, (void *) &(*it), key.c_str());
    return result.first;
}
}  // namespace

template <bool Writable>
llama_future_file_buffer<Writable>::llama_future_file_buffer(const std::string & promise_key,
                                                             const std::string & context) :
    file_buffer_future(),
    file_buffer() {
    std::string key              = final_key(promise_key, context);
    file_buffer_promise_iterator = ensure_promise_registry<Writable>(key);
    file_buffer_future           = file_buffer_promise_iterator->second.get_future();
}

template <bool Writable>
llama_future_file_buffer<Writable>::llama_future_file_buffer(llama_future_file_buffer && other) noexcept :
    file_buffer_promise_iterator(std::move(other.file_buffer_promise_iterator)),
    file_buffer_future(std::move(other.file_buffer_future)),
    file_buffer(std::move(other.file_buffer)) {
    // Set the other object's iterator to end() to mark it as moved from
    // to avoid early erasure at destruction of the moved other object
    other.file_buffer_promise_iterator = promise_registry<Writable>().end();
}

template <bool Writable>
llama_future_file_buffer<Writable> & llama_future_file_buffer<Writable>::operator=(
    llama_future_file_buffer && other) noexcept {
    if (this != &other) {
        file_buffer_promise_iterator       = std::move(other.file_buffer_promise_iterator);
        file_buffer_future                 = std::move(other.file_buffer_future);
        file_buffer                        = std::move(other.file_buffer);
        other.file_buffer_promise_iterator = promise_registry<Writable>().end();
    }
    return *this;
}

template <bool Writable> llama_future_file_buffer<Writable>::~llama_future_file_buffer() {
    std::lock_guard<std::mutex> lock(promise_registry_mutex);
    if (file_buffer_promise_iterator != promise_registry<Writable>().end()) {
        promise_registry<Writable>().erase(file_buffer_promise_iterator);
    }
}

template <bool Writable>
bool llama_future_file_buffer<Writable>::fulfill_promise(const std::string & promise_key, const std::string & context,
                                                         std::unique_ptr<llama_file_buffer<Writable>> && value) {
    std::string key = final_key(promise_key, context);
    auto        it  = ensure_promise_registry<Writable>(key);
    if (it != promise_registry<Writable>().end()) {
        LLAMA_LOG_CMAKE_DEBUG("fulfilling future file buffer %p for %s\n", (void *) &(*it), key.c_str());
        it->second.set_value(std::move(value));
        return true;
    }
    return false;
}

template <bool Writable>
std::unique_ptr<llama_file_buffer<Writable>> llama_future_file_buffer<Writable>::extract() const {
    if (file_buffer) {
        return std::move(file_buffer);
    }

    auto future_result = file_buffer_future.get();
    file_buffer        = std::move(future_result);
    return std::move(file_buffer);
}

// Explicit instantiations for llama_future_file_buffer
template struct llama_future_file_buffer<false>;
template struct llama_future_file_buffer<true>;

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
    impl(struct llama_file * file, size_t prefetch, bool numa) {
        GGML_UNUSED(numa);

        size = file->size();

        HANDLE hFile = (HANDLE) _get_osfhandle(file->file_id());

        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

        if (hMapping == NULL) {
            DWORD error = GetLastError();
            throw std::runtime_error(format("CreateFileMappingA failed: %s", llama_format_win_err(error).c_str()));
        }

        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        DWORD error = GetLastError();
        CloseHandle(hMapping);

        if (addr == NULL) {
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
        if (!UnmapViewOfFile(addr)) {
            LLAMA_LOG_WARN("warning: UnmapViewOfFile failed: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
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
#if defined(TARGET_OS_VISION) || defined(TARGET_OS_TV) || defined(_AIX)
        // visionOS/tvOS dont't support RLIMIT_MEMLOCK
        // Skip resource limit checks on visionOS/tvOS
        suggest = false;
#else
        struct rlimit lock_limit;
        if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
            suggest = false;
        }
        if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size)) {
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
