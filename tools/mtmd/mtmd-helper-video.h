#pragma once

// Internal video decoder process ownership; not part of the public helper API.
#include <cstdint>
#include <vector>

#ifdef MTMD_VIDEO
#include "sheredom/subprocess.h"
#include <thread>

// Owns the child and its pipe threads until they have all been joined.
struct mtmd_helper_subprocess {
    struct subprocess_s proc = {};
    bool created = false;
    bool alive = false;
    std::thread feeder;
    std::thread stderr_reader;

    enum class frame_result { frame, eof, error };

    mtmd_helper_subprocess() = default;
    mtmd_helper_subprocess(const mtmd_helper_subprocess &) = delete;
    mtmd_helper_subprocess & operator=(const mtmd_helper_subprocess &) = delete;
    ~mtmd_helper_subprocess() { stop(); }

    int start(const char * const command[], int options);
    int finish();
    void stop();
    FILE * stdout_pipe() { return subprocess_stdout(&proc); }
    void start_feeder(const std::vector<uint8_t> & buf);
    frame_result read_frame(std::vector<uint8_t> & frame);
};
#endif
