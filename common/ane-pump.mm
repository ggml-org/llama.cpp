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

#import <Foundation/Foundation.h>
#include <pthread.h>
#include <thread>

// Forward declaration of the program's instance handle. The
// pump's run() takes a pointer to the program + the resolved
// instance; the implementation in this file does not need to
// see the program's internals (the submit_fn callback handles
// the Core ML call).

namespace ane_pump {

// Forward-declare from pthread/qos.h. The QOS class API is
// available on macOS 10.10+ via pthread/qos.h; we use the
// modern pthread_set_qos_class_self_np / pthread_get_qos_class_np
// pair. QOS_CLASS_BACKGROUND is the lowest tier; the OS
// schedules these threads on the E-cores (low-power cluster)
// on Apple Silicon. This is the runtime payoff for the pump:
// the ANE dispatch runs on an E-core, leaving the P-cores free
// for the host's main thread.
static void pin_current_thread_to_ecore() {
    qos_class_t qos = QOS_CLASS_BACKGROUND;
    pthread_set_qos_class_self_np(qos, 0);
}

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
    // Create the per-pump E-core dispatch queue. The queue is
    // serial: only one Core ML prediction per pump at a time
    // (ANE itself is single-threaded; concurrent predictions
    // would just serialize on the ANE anyway). The thread that
    // services this queue sets its QoS to QOS_CLASS_BACKGROUND
    // before running any block; the OS places the thread on an
    // E-core on Apple Silicon.
    if (pump.ecore_queue == nullptr) {
        const char * label = "org.ggml.llama.ane.pump.ecore";
        // Create the queue first; then dispatch a one-shot
        // block on it to set the QoS. Subsequent blocks inherit
        // the QoS of the worker thread (QOS_CLASS_BACKGROUND
        // persists on a serial GCD queue's worker).
        dispatch_queue_t q = dispatch_queue_create(label,
                dispatch_queue_attr_make_with_qos_class(
                    DISPATCH_QUEUE_SERIAL, QOS_CLASS_BACKGROUND, 0));
        if (q == nullptr) {
            return false;
        }
        // Set the QoS on the worker thread. dispatch_sync on
        // the queue blocks until the block runs; the block runs
        // on the queue's worker thread, which is the thread we
        // want to pin. We use sync (not async) because init is
        // a one-time cost and we want the QoS to be in place
        // before the pump is used.
        dispatch_sync(q, ^{
            pin_current_thread_to_ecore();
        });
        // Store as a void* with __bridge_retained; ARC
        // increments the refcount. The free() path uses
        // __bridge_transfer to release it. The dispatch_release
        // call below is removed because ARC manages the lifetime
        // via the strong reference held by the pump struct.
        pump.ecore_queue = (__bridge_retained void *) q;
        pump.ecore_queue_ready = true;
    }
    return true;
}

void free(common_ane_pump & pump) {
    if (pump.ecore_queue_ready) {
        dispatch_queue_t q = (__bridge_transfer dispatch_queue_t)
            pump.ecore_queue;
        // Drain the queue before releasing: wait for any
        // in-flight block to finish. dispatch_sync on a queue
        // is sufficient (the queue is serial; one in-flight
        // block at a time).
        dispatch_sync(q, ^{});  // no-op barrier
        // ARC releases the strong reference at the end of this
        // scope (the local q goes out of scope; the pump's
        // void* is nulled below).
        pump.ecore_queue = nullptr;
        pump.ecore_queue_ready = false;
    }
    pump.state.store(ANE_PUMP_IDLE, std::memory_order_release);
}

int ecore_qos_class(const common_ane_pump & pump) {
    if (!pump.ecore_queue_ready) {
        return -1;
    }
    qos_class_t qos = QOS_CLASS_UNSPECIFIED;
    int rel = 0;
    const int rc = pthread_get_qos_class_np(
        pthread_self(), &qos, &rel);
    if (rc != 0) {
        return -1;
    }
    return (int) qos;
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
    if (!pump.ecore_queue_ready) {
        return false;
    }
    // INPUT_READY -> ANE_BUSY
    uint32_t expected = ANE_PUMP_INPUT_READY;
    if (!pump.state.compare_exchange_strong(
            expected, ANE_PUMP_ANE_BUSY,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return false;  // host hasn't signaled; not our turn
    }
    // Run the submit_fn on the E-core thread. dispatch_sync on
    // a serial queue blocks until the block completes; the
    // block runs on the queue's worker thread, which has
    // QOS_CLASS_BACKGROUND affinity (set at init). This is
    // where the runtime payoff lives: the Core ML prediction
    // runs on the E-core, off the critical dispatch path on
    // the main thread.
    dispatch_queue_t q = (__bridge dispatch_queue_t) pump.ecore_queue;
    __block bool submit_ok = false;
    dispatch_sync(q, ^{
        submit_ok = submit(program, instance, context);
    });
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
        return false;
    }
    // W7: signal downstream consumers. The signal_fn is the
    // per-slot MTLSharedEvent signaller; the value is the
    // pump's monotonic completion counter so consumers can
    // strict-order their reads. The signal is invoked AFTER
    // the ANE_BUSY -> OUTPUT_READY transition so consumers
    // observing the signaled value are guaranteed to see the
    // data plane state (the IOSurface bytes the ANE wrote).
    //
    // The signal itself runs on the E-core thread (off the
    // critical path). The signal is fast (ggml_mtl_shared_event_signal
    // is a single setSignaledValue: call) so the extra
    // dispatch_sync is negligible.
    if (signal != nullptr) {
        const uint64_t value = pump.completions.load(
            std::memory_order_acquire) + 1;
        dispatch_sync(q, ^{
            signal(program, pump.function_id, value, context);
        });
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
