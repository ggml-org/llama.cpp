// Unit tests for the E-core pump state machine
// (common/ane-pump.h). The pump drives the lock-free per-
// function dispatch flow:
//
//   IDLE -> INPUT_READY -> ANE_BUSY -> OUTPUT_READY -> IDLE
//
// Each transition uses atomic CAS. The test exercises:
//
//   1. The state starts at IDLE after init.
//   2. signal_input_ready CASes IDLE -> INPUT_READY.
//   3. signal_input_ready fails when state is not IDLE.
//   4. run() drives INPUT_READY -> ANE_BUSY -> OUTPUT_READY
//      -> IDLE and increments the completions counter.
//   5. run() fails when state is not INPUT_READY.
//   6. The submission counter increments on each
//      signal_input_ready; the completions counter increments
//      on each successful run().
//   7. Multiple SPSC cycles (host signals, pump runs, host
//      signals again) maintain the lock-free contract.

#include "ane-pump.h"

#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg); \
        ++g_failures; \
    } else { \
        std::fprintf(stdout, "ok   %s\n", msg); \
    } \
} while (0)

// Build a small synthetic manifest with one function, two input
// slots, one output slot. The slot content doesn't matter for
// the state machine test; we only need slot ids to resolve.
ane_state_layout_v1_t build_manifest() {
    ane_state_layout_v1_t m = {};
    m.version = 1;
    std::strncpy(m.bundle_name, "pump-test", sizeof(m.bundle_name) - 1);
    m.state_size_bytes = 1024 * 1024;
    m.model_type = ANE_MODEL_TYPE_ML_PROGRAM;
    m.n_slots = 3;
    {
        auto & s = m.slots[0];
        std::strncpy(s.name, "x", sizeof(s.name) - 1);
        s.kind = ANE_SLOT_KIND_INPUT;
        s.dtype = ANE_DTYPE_F32;
        s.n_dim = 1;
        s.shape[0] = 4;
        s.offset = 0;
        s.size_bytes = 16;
    }
    {
        auto & s = m.slots[1];
        std::strncpy(s.name, "y", sizeof(s.name) - 1);
        s.kind = ANE_SLOT_KIND_INPUT;
        s.dtype = ANE_DTYPE_I32;
        s.n_dim = 1;
        s.shape[0] = 2;
        s.offset = 16;
        s.size_bytes = 8;
    }
    {
        auto & s = m.slots[2];
        std::strncpy(s.name, "z", sizeof(s.name) - 1);
        s.kind = ANE_SLOT_KIND_OUTPUT;
        s.dtype = ANE_DTYPE_F32;
        s.n_dim = 1;
        s.shape[0] = 4;
        s.offset = 32;
        s.size_bytes = 16;
    }
    m.n_functions = 1;
    {
        auto & f = m.functions[0];
        std::strncpy(f.name, "main", sizeof(f.name) - 1);
        f.role = ANE_ROLE_MATMUL;
        f.bucket = 0;
        f.stateful = false;
        f.use_ane = true;
        std::strncpy(f.core_ml_function_name, "main", sizeof(f.core_ml_function_name) - 1);
        f.n_inputs = 2;
        f.input_slot_ids[0] = 0;
        f.input_slot_ids[1] = 1;
        f.n_outputs = 1;
        f.output_slot_ids[0] = 2;
    }
    return m;
}

// No-op submit callback for the state machine test. The pump
// doesn't need Core ML for the lock-free test; it just needs a
// callable that returns true. The program/instance pointers
// are not dereferenced.
bool noop_submit(common_ane_mtp_program & /*program*/,
                 common_ane_compute_instance & /*instance*/,
                 void * /*context*/) {
    return true;
}

}  // namespace

int main() {
    std::fprintf(stdout, "E-core pump state machine test\n");
    const ane_state_layout_v1_t manifest = build_manifest();

    // Test 1: init sets state to IDLE
    {
        common_ane_pump pump;
        const bool ok = ane_pump::init(pump, manifest, 0);
        CHECK(ok, "init returns true for valid function_id");
        CHECK(pump.state.load() == ANE_PUMP_IDLE,
              "init sets state to IDLE");
        CHECK(pump.input_slot_ids.size() == 2,
              "init resolves 2 input slot ids");
        CHECK(pump.input_slot_ids[0] == 0, "first input slot id is 0");
        CHECK(pump.input_slot_ids[1] == 1, "second input slot id is 1");
        CHECK(pump.output_slot_ids.size() == 1,
              "init resolves 1 output slot id");
        CHECK(pump.output_slot_ids[0] == 2, "first output slot id is 2");
        CHECK(pump.submission_counter.load() == 0,
              "submission counter starts at 0");
        CHECK(pump.completions.load() == 0, "completions starts at 0");
    }

    // Test 2: signal_input_ready succeeds from IDLE
    {
        common_ane_pump pump;
        ane_pump::init(pump, manifest, 0);
        const bool ok = ane_pump::signal_input_ready(pump);
        CHECK(ok, "signal_input_ready succeeds from IDLE");
        CHECK(pump.state.load() == ANE_PUMP_INPUT_READY,
              "state is INPUT_READY after signal");
        CHECK(pump.submission_counter.load() == 1,
              "submission counter is 1 after one signal");
    }

    // Test 3: signal_input_ready fails when not in IDLE
    {
        common_ane_pump pump;
        ane_pump::init(pump, manifest, 0);
        ane_pump::signal_input_ready(pump);
        const bool ok = ane_pump::signal_input_ready(pump);
        CHECK(!ok, "signal_input_ready fails when state is INPUT_READY");
        CHECK(pump.state.load() == ANE_PUMP_INPUT_READY,
              "state is still INPUT_READY after failed signal");
        CHECK(pump.submission_counter.load() == 1,
              "submission counter is unchanged after failed signal");
    }

    // Test 4: run() drives INPUT_READY -> ANE_BUSY -> OUTPUT_READY
    // -> IDLE and increments completions
    {
        common_ane_pump pump;
        ane_pump::init(pump, manifest, 0);
        ane_pump::signal_input_ready(pump);
        // The program/instance pointers are unused by noop_submit;
        // null is safe.
        const bool ok = ane_pump::run(pump,
                *static_cast<common_ane_mtp_program *>(nullptr),
                *static_cast<common_ane_compute_instance *>(nullptr),
                noop_submit, nullptr);
        CHECK(ok, "run succeeds from INPUT_READY");
        CHECK(pump.state.load() == ANE_PUMP_IDLE,
              "state returns to IDLE after run");
        CHECK(pump.completions.load() == 1,
              "completions counter is 1 after one run");
    }

    // Test 5: run() fails when not in INPUT_READY
    {
        common_ane_pump pump;
        ane_pump::init(pump, manifest, 0);
        const bool ok = ane_pump::run(pump,
                *static_cast<common_ane_mtp_program *>(nullptr),
                *static_cast<common_ane_compute_instance *>(nullptr),
                noop_submit, nullptr);
        CHECK(!ok, "run fails when state is IDLE");
        CHECK(pump.state.load() == ANE_PUMP_IDLE,
              "state is still IDLE after failed run");
        CHECK(pump.completions.load() == 0,
              "completions counter is 0 after failed run");
    }

    // Test 6: SPSC cycle. 1000 cycles of host-signal + pump-run.
    // The lock-free contract must hold for all 1000.
    {
        common_ane_pump pump;
        ane_pump::init(pump, manifest, 0);
        const uint64_t n_cycles = 1000;
        for (uint64_t i = 0; i < n_cycles; ++i) {
            if (!ane_pump::signal_input_ready(pump)) {
                std::fprintf(stderr,
                        "FAIL: signal failed at cycle %llu\n",
                        (unsigned long long) i);
                ++g_failures;
                break;
            }
            if (!ane_pump::run(pump,
                    *static_cast<common_ane_mtp_program *>(nullptr),
                    *static_cast<common_ane_compute_instance *>(nullptr),
                    noop_submit, nullptr)) {
                std::fprintf(stderr,
                        "FAIL: run failed at cycle %llu\n",
                        (unsigned long long) i);
                ++g_failures;
                break;
            }
        }
        CHECK(pump.completions.load() == n_cycles,
              "completions counter matches cycle count");
        CHECK(pump.submission_counter.load() == n_cycles,
              "submission counter matches cycle count");
        CHECK(pump.state.load() == ANE_PUMP_IDLE,
              "state is IDLE after all cycles");
    }

    // Test 7: lock-free SPSC across two threads. The host
    // (producer) signals input_ready; the pump thread
    // (consumer) calls run. Run for 10000 cycles across
    // threads; the lock-free contract must hold.
    {
        common_ane_pump pump;
        ane_pump::init(pump, manifest, 0);
        std::atomic<bool> stop{false};
        std::thread pump_thread([&]() {
            while (!stop.load(std::memory_order_acquire)) {
                if (pump.state.load(std::memory_order_acquire) ==
                        ANE_PUMP_INPUT_READY) {
                    ane_pump::run(pump,
                        *static_cast<common_ane_mtp_program *>(nullptr),
                        *static_cast<common_ane_compute_instance *>(nullptr),
                        noop_submit, nullptr);
                } else {
                    std::this_thread::yield();
                }
            }
            // Drain any final cycle
            while (pump.state.load(std::memory_order_acquire) !=
                    ANE_PUMP_IDLE) {
                if (pump.state.load(std::memory_order_acquire) ==
                        ANE_PUMP_INPUT_READY) {
                    ane_pump::run(pump,
                        *static_cast<common_ane_mtp_program *>(nullptr),
                        *static_cast<common_ane_compute_instance *>(nullptr),
                        noop_submit, nullptr);
                } else {
                    std::this_thread::yield();
                }
            }
        });
        const uint64_t n_signals = 10000;
        for (uint64_t i = 0; i < n_signals; ++i) {
            // Spin until the pump returns to IDLE before signaling.
            ane_pump::wait_idle(pump);
            const bool ok = ane_pump::signal_input_ready(pump);
            if (!ok) {
                std::fprintf(stderr,
                        "FAIL: cross-thread signal failed at %llu\n",
                        (unsigned long long) i);
                ++g_failures;
                break;
            }
        }
        // Wait for the pump to drain the last signal.
        ane_pump::wait_idle(pump);
        stop.store(true, std::memory_order_release);
        pump_thread.join();
        CHECK(pump.completions.load() == n_signals,
              "cross-thread completions matches signal count");
        CHECK(pump.submission_counter.load() == n_signals,
              "cross-thread submission counter matches signal count");
        CHECK(pump.state.load() == ANE_PUMP_IDLE,
              "cross-thread state is IDLE after all signals");
    }

    if (g_failures == 0) {
        std::fprintf(stdout, "ALL PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "FAILED: %d failures\n", g_failures);
    return 1;
}
