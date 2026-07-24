#pragma once

#include <sheredom/subprocess.h>

#include <cstdio>
#include <string>
#include <vector>

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

    FILE * stdin_file();
    FILE * stdout_file();
    FILE * stderr_file();

    void terminate();

    // wait for the process to exit, release the underlying handle and return its exit code
    int join();

private:
    subprocess_s proc {};
    bool is_created = false;

    bool has_handle() const;
};
