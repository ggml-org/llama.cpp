// local llama-server process management for llama-cli
//
// when no --server-base is given, the CLI spawns a llama-server child process
// and talks to it over HTTP; the child lifetime is managed the same way the
// server router mode manages model instances:
// - the child is spawned with LLAMA_SERVER_ROUTER_PORT set, which makes it
//   watch its stdin and exit on EOF, so no orphan is left behind if the CLI
//   dies unexpectedly
// - the parent reads the child's stdout and waits for the ready line
// - on stop, the parent sends an exit command on stdin and force-kills the
//   child after a timeout

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct subprocess_s;

struct cli_server {
    ~cli_server();

    // spawn llama-server (located next to the current executable) with the
    // given args on a free port; if pass_output is true, the child output is
    // forwarded to stderr; returns false on failure (see last_error)
    bool start(const std::vector<std::string> & args, bool pass_output);

    // wait until the child reports it is ready to accept requests
    // returns false if the child exited or is_aborted returned true
    bool wait_ready(const std::function<bool()> & is_aborted);

    // gracefully stop the child process (force-kill after a timeout)
    void stop();

    bool alive() const { return started && !exited; }

    std::string address() const;

    // last lines of child output, for error reporting
    std::string recent_output() const;

    std::string last_error;
    int port = 0;

private:
    std::shared_ptr<subprocess_s> subproc;
    FILE * child_stdin = nullptr;
    std::thread log_thread;
    bool started     = false;
    bool pass_output = false;

    std::atomic<bool> ready{false};
    std::atomic<bool> exited{false};

    mutable std::mutex mtx;
    std::condition_variable cv;
    std::deque<std::string> output_lines;
};
