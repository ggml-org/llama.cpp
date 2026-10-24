#pragma once

#include <functional>
#include <string>

// Cross-process exclusive lock (flock on POSIX, LockFileEx on Windows).
// Lazy: acquire() opens the lock file, so a read-only cache works if the lock is never acquired.
class common_file_lock {
public:
    common_file_lock(const std::string & lock_path);
    ~common_file_lock();

    // take the lock, waiting while another process holds it;
    // keep_waiting is polled before each retry, false cancels the wait.
    // Also returns false when locking is unavailable.
    bool acquire(const std::function<bool()> & keep_waiting = {});

    void close();

    common_file_lock(const common_file_lock &) = delete;
    common_file_lock & operator=(const common_file_lock &) = delete;

private:
    enum class status {
        acquired,
        busy,  // another process holds the lock
        error, // file locking is unavailable: no exclusion is possible
    };

    status try_lock();

    std::string lock_path;

    int  fd     = -1;
    bool locked = false;
};
