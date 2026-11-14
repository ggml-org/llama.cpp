// Stub stage used to measure the transport without a model in the loop.
//
// It speaks both protocols the real stages speak: the stdin/stdout framing and
// the shared-memory ring. `--work-us N` fakes the compute so the measurement
// can separate transport cost from stage cost.
//
//   ring-stub --depth-pipe [--ring-fd N] [--work-us N]
#include "m5_ring.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

int main(int argc, char ** argv) {
    int ring_fd = -1, work_us = 0;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--ring-fd") == 0 && i + 1 < argc) ring_fd = atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--work-us") == 0 && i + 1 < argc) work_us = atoi(argv[++i]);
    }

    std::vector<float> row(512, 0.0f);
    for (int i = 0; i < 512; i++) row[i] = (float) i * 0.001f;

    auto burn = [&]() {
        if (work_us <= 0) return;
        const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(work_us);
        while (std::chrono::steady_clock::now() < until) { }
    };

    if (ring_fd >= 0) {
        m5::ring::Server server;
        if (!server.attach(ring_fd)) { fprintf(stderr, "stub: cannot attach\n"); return 1; }
        for (;;) {
            int rows = 0;
            if (server.next(rows) != 0) break;
            if (rows > 0) {
                burn();
                memcpy(server.reply(), row.data(), sizeof(float) * 512);
            }
            server.publish();   // also acknowledges a reset (rows == 0)
        }
        return 0;
    }

    uint32_t count = 0;
    while (fread(&count, sizeof(count), 1, stdin) == 1) {
        if (count == 0) continue;
        std::vector<float> in((size_t) count * 512);
        if (fread(in.data(), sizeof(float), in.size(), stdin) != in.size()) return 1;
        burn();
        if (fwrite(row.data(), sizeof(float), 512, stdout) != 512) return 1;
        fflush(stdout);
    }
    return 0;
}
