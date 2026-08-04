// ane-pump.mm — E-core pump implementation.
//
// The pump is the lock-free orchestrator for one multifunction
// function. See ane-pump.h for the architecture pivot rationale.
//
// Implementation notes:
//
//   - The state is a single std::atomic<uint32_t>. All
//     transitions use compare_exchange_weak (the weak variant is
//     preferred on Apple Silicon where LL/SC is the underlying
//     primitive; the spurious-failure retry loop is faster than
//     compare_exchange_strong's stronger CAS).
//
//   - The host signals input readiness via ane_pump_signal_input_ready
//     which CASes IDLE -> INPUT_READY. The host must wait for
//     the pump to return to IDLE before signaling again; this
//     serializes the host with the pump (single-producer
//     single-consumer lock-free).
//
//   - The pump (caller of ane_pump_run) drives INPUT_READY ->
//     ANE_BUSY -> OUTPUT_READY -> IDLE. The host's
//     signal_input_ready can race with the pump returning to
//     IDLE; the CAS ensures only one transition lands.
//
//   - The submit_fn callback is invoked with state == ANE_BUSY
//     and is responsible for invoking Core ML's
//     predictionFromFeatures with the manifest's input slot
//     MLMultiArrays (already pinned at load) and the output slot
//     MLMultiArrays as outputBackings. The default
//     implementation in the dispatch path uses the existing
//     dispatch_pinned_function helper.
//
//   - The submission_counter and completions counters are
//     monotonic. Producers can use them to detect "the pump
//     just completed a submission" without polling state; this
//     is the basis for the W7 MTLSharedEvent handoff (the
//     counter is also exposed as the event value).

#include "ane-pump.h"
#include "ane-mtp.h"

#include <thread>

// Forward declaration of the program's instance handle. The
// pump's run() takes a pointer to the program + the resolved
// instance; the implementation in this file does not need to
// see the program's internals (the submit_fn callback handles
// the Core ML call).

namespace ane_pump {

bool init(common_ane_pump & pump,
         const ane_state_layout_v1_t & manifest,
         uint32_t function_id) {
    if (function_id >= manifest.n_functions) {
        return false;
    }
    const ane_function_v1_t & function = manifest.functions[function_id];
    pump.function_id = function_id;
    pump.input_slot_ids.clear();
    pump.output_slot_ids.clear();
    pump.input_slot_ids.reserve(function.n_inputs);
    pump.output_slot_ids.reserve(function.n_outputs);
    for (uint32_t i = 0; i < function.n_inputs; ++i) {
        pump.input_slot_ids.push_back(function.input_slot_ids[i]);
    }
    for (uint32_t i = 0; i < function.n_outputs; ++i) {
        pump.output_slot_ids.push_back(function.output_slot_ids[i]);
    }
    pump.state.store(ANE_PUMP_IDLE, std::memory_order_release);
    pump.submission_counter.store(0, std::memory_order_release);
    pump.completions.store(0, std::memory_order_release);
    return true;
}

bool signal_input_ready(common_ane_pump & pump) {
    uint32_t expected = ANE_PUMP_IDLE;
    if (!pump.state.compare_exchange_strong(
            expected, ANE_PUMP_INPUT_READY,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return false;  // pump is busy
    }
    pump.submission_counter.fetch_add(
        1, std::memory_order_acq_rel);
    return true;
}

bool run(common_ane_pump & pump,
        common_ane_mtp_program & program,
        common_ane_compute_instance & instance,
        submit_fn submit,
        signal_fn signal,
        void * context) {
    // INPUT_READY -> ANE_BUSY
    uint32_t expected = ANE_PUMP_INPUT_READY;
    if (!pump.state.compare_exchange_strong(
            expected, ANE_PUMP_ANE_BUSY,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return false;  // host hasn't signaled; not our turn
    }
    // Submit the prediction. The submit_fn is expected to be
    // synchronous (it returns when the prediction is complete);
    // for the current Core ML public API this is the only mode
    // (the model.predictionFromFeatures:options:error: path is
    // synchronous).
    const bool submit_ok = submit(program, instance, context);
    if (!submit_ok) {
        // Revert to IDLE so the host can retry. The completions
        // counter is not incremented (this is a failed
        // submission).
        pump.state.store(ANE_PUMP_IDLE, std::memory_order_release);
        return false;
    }
    // ANE_BUSY -> OUTPUT_READY (atomic, the public release of
    // the output data). Downstream consumers (Metal, the host
    // when reading outputs) can now safely read the pinned
    // slots.
    expected = ANE_PUMP_ANE_BUSY;
    if (!pump.state.compare_exchange_strong(
            expected, ANE_PUMP_OUTPUT_READY,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        // Should be impossible: only the pump transitions
        // ANE_BUSY. Log and continue.
        return false;
    }
    // W7: signal downstream consumers. The signal_fn is the
    // per-slot MTLSharedEvent signaller; the value is the
    // pump's monotonic completion counter so consumers can
    // strict-order their reads. The signal is invoked AFTER
    // the ANE_BUSY -> OUTPUT_READY transition so consumers
    // observing the signaled value are guaranteed to see the
    // data plane state (the IOSurface bytes the ANE wrote).
    if (signal != nullptr) {
        const uint64_t value = pump.completions.load(
            std::memory_order_acquire) + 1;
        signal(program, pump.function_id, value, context);
    }
    // OUTPUT_READY -> IDLE. The host's next signal_input_ready
    // will be the IDLE -> INPUT_READY transition.
    expected = ANE_PUMP_OUTPUT_READY;
    if (!pump.state.compare_exchange_strong(
            expected, ANE_PUMP_IDLE,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return false;
    }
    pump.completions.fetch_add(1, std::memory_order_acq_rel);
    return true;
}

uint32_t wait_idle(common_ane_pump & pump) {
    // Spin with a tiny pause. The pump returns to IDLE quickly
    // after a successful submission (microseconds); a backoff
    // would be premature here. For longer waits, the caller
    // should use a dispatch_semaphore signalled by the pump
    // (W7).
    while (pump.state.load(std::memory_order_acquire) != ANE_PUMP_IDLE) {
        std::this_thread::yield();
    }
    return ANE_PUMP_IDLE;
}

}  // namespace ane_pump
