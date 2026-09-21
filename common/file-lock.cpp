#include "file-lock.h"

#include "log.h"

#include <chrono>
#include <filesystem>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <sys/file.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#endif

common_file_lock::common_file_lock(const std::string & lock_path) {
    this->lock_path = lock_path;
}

#if defined(_WIN32)
// the \\?\ prefix (for paths longer than 255 chars) is only accepted on absolute paths
// with backslash separators: normalize first, then check the length of the absolute form
static std::wstring as_extended_path(const std::string & path) {
    // convert from the narrow encoding std::filesystem uses on this system
    int wlen = MultiByteToWideChar(CP_ACP, 0, path.c_str(), (int) path.size(), nullptr, 0);
    if (wlen <= 0) {
        return std::wstring(path.begin(), path.end()); // best effort for exotic encodings
    }
    std::wstring wide(wlen, L'\0');
    MultiByteToWideChar(CP_ACP, 0, path.c_str(), (int) path.size(), wide.data(), wlen);

    if (wide.rfind(L"\\\\?\\", 0) == 0) {
        return wide;
    }

    // GetFullPathNameW resolves relative paths and normalizes separators
    DWORD size = GetFullPathNameW(wide.c_str(), 0, nullptr, nullptr);
    if (size == 0) {
        return wide;
    }
    std::wstring full(size, L'\0');
    DWORD written = GetFullPathNameW(wide.c_str(), size, full.data(), nullptr);
    if (written == 0 || written >= size) {
        return wide;
    }
    full.resize(written);

    if (full.size() <= 255) {
        return wide; // short enough: open the path exactly as given
    }
    if (full.rfind(L"\\\\", 0) == 0) {
        return L"\\\\?\\UNC\\" + full.substr(2); // \\server\share -> \\?\UNC\server\share
    }
    return L"\\\\?\\" + full;
}
#endif

common_file_lock::status common_file_lock::try_lock() {
    if (lock_path.empty()) {
        return status::acquired; // no lock path, nothing to lock
    }
#if defined(_WIN32)
    if (fd < 0) {
        // opened with a wide path: extended paths cannot go through the narrow CRT
        std::wstring lock_path_w = as_extended_path(lock_path);
        fd = _wopen(lock_path_w.c_str(), _O_RDWR | _O_CREAT | _O_BINARY | _O_NOINHERIT, _S_IREAD | _S_IWRITE);
        if (fd < 0) {
            LOG_ERR("%s: cannot open lock file: %s\n", __func__, lock_path.c_str());
            return status::error;
        }
    }
    // lock the first byte, the file itself stays empty
    OVERLAPPED ov = {};
    locked = LockFileEx((HANDLE) _get_osfhandle(fd),
                        LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &ov);
    if (locked) {
        return status::acquired;
    }
    if (GetLastError() == ERROR_LOCK_VIOLATION) {
        return status::busy;
    }
    LOG_ERR("%s: file locking failed (%d)\n", __func__, (int) GetLastError());
    return status::error;
#else
    if (fd < 0) {
        int open_flags = O_RDWR | O_CREAT | O_CLOEXEC;
#ifdef O_NOFOLLOW
        open_flags |= O_NOFOLLOW;
#endif
        // group-writable, so users sharing a cache can wait on the lock
        fd = open(lock_path.c_str(), open_flags, 0664);
        if (fd < 0) {
            LOG_ERR("%s: cannot open lock file: %s\n", __func__, lock_path.c_str());
            return status::error;
        }
        // open() filters the mode through the umask
        (void) fchmod(fd, 0664);
    }

    if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
        locked = true;
        return status::acquired;
    }
    // EACCES is reported as contention on some filesystems
    if (errno == EWOULDBLOCK || errno == EAGAIN || errno == EACCES) {
        return status::busy;
    }
    // flock is not implemented (some NFS/FUSE mounts): report it instead of racing
    LOG_ERR("%s: file locking failed (%s)\n", __func__, strerror(errno));
    return status::error;
#endif
}

bool common_file_lock::acquire(const std::function<bool()> & keep_waiting) {
    // the lock file is opened by try_lock, its parent must exist by then
    if (!lock_path.empty()) {
        std::error_code ec;
        auto lock_dir = std::filesystem::path(lock_path).parent_path();
        if (!lock_dir.empty()) {
            std::filesystem::create_directories(lock_dir, ec);
            if (ec) {
                LOG_ERR("%s: unable to create lock directory '%s': %s\n", __func__, lock_dir.string().c_str(), ec.message().c_str());
                return false;
            }
        }
    }

    while (true) {
        switch (try_lock()) {
            case status::acquired:
                return true;
            case status::busy:
                if (keep_waiting && !keep_waiting()) {
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            case status::error:
                return false;
        }
    }
}

// the lock file is never deleted: another process could then create a fresh
// file at the same path and lock that
void common_file_lock::close() {
#if defined(_WIN32)
    if (fd >= 0) {
        if (locked) {
            OVERLAPPED ov = {};
            UnlockFileEx((HANDLE) _get_osfhandle(fd), 0, 1, 0, &ov);
        }
        _close(fd);
    }
#else
    if (fd >= 0) {
        if (locked) {
            flock(fd, LOCK_UN);
        }
        ::close(fd);
    }
#endif
    fd     = -1;
    locked = false;
}

common_file_lock::~common_file_lock() {
    close();
}
