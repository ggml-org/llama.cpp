// ane-pump.h — E-core pump for the multifunction .mlmodelc dispatch.
//
// The architecture pivot is: the multifunction .mlmodelc is loaded
// stateless, the IOSurface-backed state is the canonical state,
// and the dispatch contract is a lock-free state machine driven
// by a per-function pump. The pump is the E-core pump
// (low-power background core) that runs the dispatch flow:
//
//   IDLE                 host can write inputs to the pinned slots
//      |                 (caller calls ane_pump_signal_input_ready)
//      v
//   INPUT_READY          pump thread is awakened; it submits the
//      |                 Core ML prediction with outputBackings =
//      v                 pinned output slots
//   ANE_BUSY             Core ML is processing; pump waits via the
//      |                 MLFeatureProvider completion path
//      v
//   OUTPUT_READY         Core ML wrote outputs into the pinned
//      |                 slots; pump signals the per-slot
//      v                 MTLSharedEvent for downstream (Metal)
//   IDLE                 pump returns to idle; the next host
//                        transition restarts the cycle.
//
// State transitions use atomic CAS so the host (producer) and
// the pump thread (consumer) can run lock-free. The data plane
// is the IOSurface-backed state; the control plane is the atomic
// state and the per-slot MTLSharedEvent.
//
// The pump is per-function: one pump per multifunction function
// (prefill_s128, mtp_predict, dflash_b4, hybrid_b4, etc.). Each
// function has its own state machine; the host signals input
// readiness to one pump at a time.
//
// The pump itself is not a thread; the caller invokes
// ane_pump_run() to drive one cycle (which submits the prediction
// and waits for completion synchronously). The E-core pinning is
// a follow-on: when the caller runs ane_pump_run() from a
// dispatch queue with a thread affinity set to an E-core
// (QoS_CLASS_BACKGROUND or pthread_set_qos_class_self_np), the
// pump inherits that affinity. The state machine is what matters
// for the lock-free contract; the E-core is the runtime payoff
// (off the critical dispatch path on the main thread).

#pragma once

#include "ane-state.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// Pump state. The state is an unsigned int (atomic) so the
// CAS primitive is well-defined on all Apple Silicon targets.
// All states except the destination are checked under CAS;
// the destination is the only valid next state for the
// transition.
typedef enum {
    ANE_PUMP_IDLE         = 0,
    ANE_PUMP_INPUT_READY  = 1,
    ANE_PUMP_ANE_BUSY     = 2,
    ANE_PUMP_OUTPUT_READY = 3,
} ane_pump_state_t;

// Per-function pump. Owns the lock-free state machine and the
// resolved slot metadata for one multifunction function. The
// pump does NOT own the IOSurface; the IOSurface is owned by
// the program (common_ane_mtp_program::state_iosurface). The
// pump just transitions states and (in W7) signals per-slot
// MTLSharedEvent handles for downstream consumers.
struct common_ane_mtp_program;
struct common_ane_compute_instance;
typedef struct common_ane_pump {
    // The bound function (index into manifest.functions[]). Set
    // at init; never changes.
    uint32_t function_id;
    // The bound function's resolved input slot ids (cached at
    // init from the manifest; the host-side set_pinned_input
    // helpers resolve by name today but the pump can resolve
    // once and drive by id).
    std::vector<uint32_t> input_slot_ids;
    // The bound function's resolved output slot ids. The pump
    // sets MLPredictionOptions.outputBackings from these in
    // ane_pump_run.
    std::vector<uint32_t> output_slot_ids;
    // The atomic state. The host transitions IDLE -> INPUT_READY
    // via ane_pump_signal_input_ready. The pump transitions
    // INPUT_READY -> ANE_BUSY -> OUTPUT_READY -> IDLE inside
    // ane_pump_run.
    std::atomic<uint32_t> state;
    // Submission counter (incremented by the host on each
    // INPUT_READY transition; the pump records the counter
    // value at submission time so a racing host can detect
    // "the pump is processing the Nth submission"). The
    // counter is monotonically increasing; both producer and
    // consumer read it as uint64_t.
    std::atomic<uint64_t> submission_counter;
    // Total submissions completed (incremented by the pump
    // when the OUTPUT_READY -> IDLE transition lands).
    std::atomic<uint64_t> completions;
    // Per-pump dispatch queue. Created at init; the thread that
    // services this queue has QOS_CLASS_BACKGROUND affinity
    // (E-core). The dispatch path submits ane_pump::run work
    // to this queue (via dispatch_sync) so the actual Core ML
    // prediction runs on the E-core, off the critical dispatch
    // path. The queue is freed in ane_pump::free.
    // Default-initialized to nullptr so a fresh common_ane_pump
    // starts with no queue (the init path sets it). Without the
    // initializer the field reads stack garbage on first use.
    void * ecore_queue = nullptr;
    // True when the E-core queue has been created.
    bool ecore_queue_ready = false;
} common_ane_pump;

#ifdef __cplusplus
}
#endif

// C++ API. The functions are implemented in common/ane-pump.mm
// and operate on a common_ane_pump embedded in the program
// (one per multifunction function).
namespace ane_pump {

// Initialize the pump for one function. Resolves the function's
// input/output slot ids from the manifest; sets state to IDLE;
// creates the per-pump dispatch queue pinned to a low-power
// background QoS class (E-core affinity via QOS_CLASS_BACKGROUND).
// Returns true on success, false if function_id is out of
// range or the manifest is malformed.
//
// The per-pump queue is what ane_pump::run uses to execute
// the submit_fn callback; routing the dispatch through this
// queue (rather than the program's serial queue) is the
// runtime payoff: the pump's work runs off the critical
// dispatch path on the main thread.
bool init(common_ane_pump & pump,
         const ane_state_layout_v1_t & manifest,
         uint32_t function_id);

// Tear down the pump: signal IDLE -> SHUTDOWN transition (not
// strictly necessary; the destructor frees the queue), free
// the per-pump dispatch queue. After free() the pump must
// not be used.
void free(common_ane_pump & pump);

// Read the QoS class of the E-core thread from the current
// thread. Returns QOS_CLASS_BACKGROUND (or whatever was set
// at init). Returns -1 if the pump has no E-core queue or
// if pthread_get_qos_class_np fails. Used by the test to
// verify the E-core affinity is in place.
int ecore_qos_class(const common_ane_pump & pump);

// Host-side: signal that inputs are written and the pump can
// submit. CASes IDLE -> INPUT_READY. Returns true on success,
// false if the pump was not in IDLE (the caller must wait for
// the pump to return to IDLE before signaling).
bool signal_input_ready(common_ane_pump & pump);

// Pump-side: drive one cycle. CASes INPUT_READY -> ANE_BUSY,
// submits the Core ML prediction, CASes ANE_BUSY -> OUTPUT_READY,
// signals downstream, CASes OUTPUT_READY -> IDLE. Returns true
// on success, false if the pump was not in INPUT_READY at entry
// or the prediction failed. The submission callback receives
// the pump + the program + the function's instance; the
// default callback uses program->queue + dispatch_pinned_function.
// When the callback returns true, the pump advances to
// OUTPUT_READY; on false, the pump reverts to IDLE and the
// error is recorded.
//
// `instance` is the resolved MLModel for the function; it
// must outlive the pump (it does, by virtue of being owned by
// the program). The callback is invoked with the host inputs
// already in the pinned state (the host called
// signal_input_ready before this).
using submit_fn = bool (*)(
    common_ane_mtp_program & program,
    common_ane_compute_instance & instance,
    void * context);

// W7: signal-side callback. Invoked after the pump transitions
// ANE_BUSY -> OUTPUT_READY. The callback is expected to signal
// the per-slot MTLSharedEvent handles so downstream consumers
// (Metal via ggml_mtl_shared_event_encode_wait) can read the
// output slots. `value` is the pump's monotonic completion
// counter; pass it as the event value so a Metal consumer can
// waitUntilSignaledValue:value and observe a strict ordering.
// May be nullptr (the pump no-ops on the signal).
using signal_fn = void (*)(
    common_ane_mtp_program & program,
    uint32_t function_id,
    uint64_t value,
    void * context);

bool run(common_ane_pump & pump,
        common_ane_mtp_program & program,
        common_ane_compute_instance & instance,
        submit_fn submit,
        signal_fn signal,
        void * context);

// Block until the pump returns to IDLE. Spins with a small
// pause; for production, the caller should be the next
// dispatcher that wants to claim the pump. Returns the
// final state (always IDLE on a normal return).
uint32_t wait_idle(common_ane_pump & pump);

}  // namespace ane_pump
