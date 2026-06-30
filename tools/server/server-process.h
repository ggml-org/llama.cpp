//
// Created by Morgan Funtowicz on 6/19/2026.
//

#ifndef LLAMA_CPP_SERVER_PROCESS_H
#define LLAMA_CPP_SERVER_PROCESS_H

#include <cstdio>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/types.h>  // pid_t
#endif

struct server_process {
    server_process() = default;

    server_process(const server_process &) = delete;
    server_process & operator=(const server_process &) = delete;
    server_process(server_process && o) noexcept;
    server_process & operator=(server_process && o) noexcept;

    ~server_process() {
        terminate();
        join();
    }

    [[nodiscard]] FILE * get_stdin() const { return fStdin; }
    [[nodiscard]] FILE * get_stdout() const { return fStdout; }

    void close_stdin() {
        if (fStdin) {
            fclose(fStdin);
        }
        fStdin = nullptr;
    }

    void close_stdout() {
        if (fStdout) {
            fclose(fStdout);
        }
        fStdout = nullptr;
    }

    [[nodiscard]] bool is_alive() const;
    void terminate() const;
    int join();
    int run(const std::vector<std::string> & args, const std::vector<std::string> & env);
private:
#ifdef _WIN32
    HANDLE hHandle = nullptr;
#else
    pid_t pid    = 0;
    int   status = 0;
    bool  joined = false;
#endif

    FILE * fStdin  = nullptr;
    FILE * fStdout = nullptr;
};
#endif  //LLAMA_CPP_SERVER_PROCESS_H
