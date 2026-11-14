// Measures what the shared-memory ring actually saves against the pipes.
//
// Both transports run the same stub stage (ring_stub) so the only variable is
// the exchange itself. `--work-us` fakes the stage compute: with 0 the loop is
// pure transport, with 3700 it resembles the real depth core (a 1-row decode
// on the HTP), where transport is expected to disappear into the noise.
#include "m5_ring.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// One depth frame: 1 reset + 15 requests, of which the first carries 2 rows.
constexpr int kCallsPerFrame = 16;
constexpr int kFrameCount = 20;

struct Pipe {
    pid_t pid = -1;
    int to = -1, from = -1;

    bool start(const std::string & command) {
        int in[2], out[2];
        if (pipe(in) != 0 || pipe(out) != 0) return false;
        pid = fork();
        if (pid == 0) {
            dup2(in[0], STDIN_FILENO);
            dup2(out[1], STDOUT_FILENO);
            ::close(in[0]); ::close(in[1]); ::close(out[0]); ::close(out[1]);
            execl("/system/bin/sh", "sh", "-c", command.c_str(), (char *) nullptr);
            _exit(127);
        }
        ::close(in[0]);
        ::close(out[1]);
        to = in[1];
        from = out[0];
        return true;
    }
    bool call(int rows, const float * data, float * out) {
        int32_t count = rows;
        if (write(to, &count, 4) != 4) return false;
        if (rows > 0 && write(to, data, sizeof(float) * 512 * (size_t) rows) != (ssize_t) (sizeof(float) * 512 * rows)) return false;
        // A reset carries no reply on this transport, exactly like the engine.
        if (rows == 0) return true;
        size_t want = sizeof(float) * 512;
        char * p = (char *) out;
        while (want) {
            ssize_t n = read(from, p, want);
            if (n <= 0) return false;
            p += n;
            want -= (size_t) n;
        }
        return true;
    }
    void close() {
        if (to >= 0) ::close(to);
        if (from >= 0) ::close(from);
        if (pid > 0) { kill(pid, SIGKILL); waitpid(pid, nullptr, 0); }
        to = from = -1;
        pid = -1;
    }
};

}  // namespace

int main(int argc, char ** argv) {
    int work_us = 0;
    std::string stub = "./ring-stub";
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--work-us") == 0 && i + 1 < argc) work_us = atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--stub") == 0 && i + 1 < argc) stub = argv[++i];
    }
    const std::string tail = " --depth-pipe --work-us " + std::to_string(work_us);

    std::vector<float> two(2 * 512), reply(512);
    for (size_t i = 0; i < two.size(); i++) two[i] = (float) i * 0.5f;

    // pipe transport
    Pipe pipe_stage;
    if (!pipe_stage.start(stub + tail)) { fprintf(stderr, "cannot start stub\n"); return 1; }
    double t0 = now_ms();
    for (int frame = 0; frame < kFrameCount; frame++) {
        if (!pipe_stage.call(0, two.data(), reply.data())) { fprintf(stderr, "pipe reset failed\n"); return 1; }
        for (int call = 1; call < kCallsPerFrame; call++) {
            const int rows = call == 1 ? 2 : 1;
            if (!pipe_stage.call(rows, two.data(), reply.data())) { fprintf(stderr, "pipe call failed\n"); return 1; }
        }
    }
    const double pipe_ms = now_ms() - t0;
    pipe_stage.close();

    // ring transport
    m5::ring::Client client;
    const int fd = client.create(2, 512);
    if (fd < 0) { fprintf(stderr, "memfd unavailable\n"); return 1; }
    Pipe ring_stage;
    if (!ring_stage.start(stub + tail + " --ring-fd " + std::to_string(fd))) { fprintf(stderr, "cannot start stub\n"); return 1; }
    if (!client.reset(ring_stage.pid)) { fprintf(stderr, "ring probe failed\n"); return 1; }
    t0 = now_ms();
    for (int frame = 0; frame < kFrameCount; frame++) {
        for (int call = 0; call < kCallsPerFrame; call++) {
            const int rows = call == 0 ? 0 : (call == 1 ? 2 : 1);
            const bool ok = rows == 0 ? client.reset(ring_stage.pid)
                                      : client.call(two.data(), rows, reply.data(), ring_stage.pid);
            if (!ok) { fprintf(stderr, "ring call failed\n"); return 1; }
        }
    }
    const double ring_ms = now_ms() - t0;
    ring_stage.close();
    client.close();

    const int calls = kCallsPerFrame * kFrameCount;
    const double per_call = (double) calls;
    printf("work_us=%d calls=%d\n", work_us, calls);
    printf("  pipe  %8.2f ms total  %7.3f us/call\n", pipe_ms, pipe_ms * 1000.0 / per_call);
    printf("  ring  %8.2f ms total  %7.3f us/call\n", ring_ms, ring_ms * 1000.0 / per_call);
    printf("  saved %8.2f ms  %6.3f us/call  (%.1f%%)\n", pipe_ms - ring_ms,
           (pipe_ms - ring_ms) * 1000.0 / per_call,
           pipe_ms > 0 ? 100.0 * (pipe_ms - ring_ms) / pipe_ms : 0.0);
    return 0;
}
