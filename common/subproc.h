#pragma once

#include <atomic>
#include <cstdio>
#include <string>
#include <vector>

#ifdef LLAMA_SUBPROCESS
#include <sheredom/subprocess.h>
#else
// dummy values to allow compilation when subprocess is disabled
struct subprocess_s {};
static constexpr int subprocess_option_no_window = 0;
static constexpr int subprocess_option_combined_stdout_stderr = 0;
static constexpr int subprocess_option_inherit_environment = 0;
static constexpr int subprocess_option_search_user_path = 0;
#endif

// RAII-style wrapper around https://github.com/sheredom/subprocess.h,
// exposing method calls instead of free functions operating on subprocess_s.
struct common_subproc {
    common_subproc() = default;
    ~common_subproc();

    common_subproc(const common_subproc &) = delete;
    common_subproc & operator=(const common_subproc &) = delete;

    // spawn a child process; if env is non-empty it replaces the child's environment
    // (do not combine with subprocess_option_inherit_environment)
    bool create(
            const std::vector<std::string> & args,
            int options,
            const std::vector<std::string> & env = {},
            const char * cwd = nullptr);

    bool alive();

    // true if LLAMA_SUBPROCESS was enabled at build time; when false, create() always fails
    static bool is_supported();

    FILE * stdin_file();
    FILE * stdout_file();
    FILE * stderr_file();

    // close stdin and detach it from the process, so a later join()/destroy() won't double-close it;
    // use this after writing all input to signal EOF to the child while it's still running
    void close_stdin();

    void terminate();

    // wait for the process to exit, release the underlying handle and return its exit code
    int join();

private:
    subprocess_s proc {};
    std::atomic<bool> is_created{false};

    bool has_handle() const;
};
