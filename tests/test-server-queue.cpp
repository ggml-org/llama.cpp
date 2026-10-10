// Regression test for https://github.com/ggml-org/llama.cpp/issues/29689
//
// A request that passes wait_until_no_sleep() while the server is awake must
// prevent the server from entering the sleeping state until the request has
// finished posting its task. Otherwise:
//
//  1. the main loop falls asleep with the task stranded in the queue: the
//     client hangs forever waiting for a response, or
//  2. the sleep callbacks destroy server state (e.g. unload the model) while
//     the request is still using it (e.g. tokenizing): use-after-free crash.
//
// The test drives server_queue directly with a tiny idle timeout, so no
// timing-sensitive network requests are needed. Note: plain assert() is not
// used because the build may define NDEBUG; failures return non-zero instead.

#include "server-queue.h"

#include <chrono>
#include <cstdio>
#include <thread>

static void sleep_ms(int64_t ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// wait up to timeout_ms for cond() to become true, return the final value
template <typename F>
static bool wait_for(F cond, int64_t timeout_ms) {
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - t0).count() < timeout_ms) {
        if (cond()) {
            return true;
        }
        sleep_ms(10);
    }
    return cond();
}

static int run_tests(server_queue & queue) {
    // idle timeout of 50 ms: without the in-flight guard the server would
    // fall asleep almost immediately once the queue is empty
    if (!wait_for([&]() { return queue.is_sleeping(); }, 2000)) {
        printf("FAIL: server should fall asleep when idle and nothing is in-flight\n");
        return 1;
    }
    printf("ok: server sleeps when idle\n");

    // wake it back up the way a request would
    queue.wait_until_no_sleep();
    if (!wait_for([&]() { return !queue.is_sleeping(); }, 2000)) {
        printf("FAIL: server should wake up on wait_until_no_sleep()\n");
        return 1;
    }
    printf("ok: server wakes up on wait_until_no_sleep()\n");

    // now hold the in-flight claim (as a request between wait_until_no_sleep()
    // and post() would) and verify the server does NOT fall back asleep,
    // even though the idle timer keeps expiring.
    // note: the loop re-evaluates the sleeping condition at most once per
    // second (max_wait_time), so the window must comfortably exceed that.
    queue.wait_until_no_sleep(); // +1 in-flight
    bool ever_slept = false;
    for (int i = 0; i < 25; i++) {
        sleep_ms(100);
        if (queue.is_sleeping()) {
            ever_slept = true;
            break;
        }
    }
    if (ever_slept) {
        printf("FAIL: BUG #29689: server fell asleep while a request was in-flight\n");
        return 1;
    }
    printf("ok: server stays awake while a request is in-flight\n");

    // releasing the claim must allow sleep again (i.e. the guard does not
    // permanently disable the sleeping state)
    queue.release_inflight(); // back to the 1 claim from the wake above
    queue.release_inflight(); // fully released
    if (!wait_for([&]() { return queue.is_sleeping(); }, 2000)) {
        printf("FAIL: server should fall asleep again once nothing is in-flight\n");
        return 1;
    }
    printf("ok: server sleeps again after the in-flight claim is released\n");

    return 0;
}

int main() {
    server_queue queue;
    queue.on_update_slots([](){});

    std::thread loop_thread([&]() { queue.start_loop(50); });

    const int rc = run_tests(queue);

    queue.terminate();
    loop_thread.join();

    if (rc == 0) {
        printf("all server_queue sleep-race tests passed\n");
    }
    return rc;
}
