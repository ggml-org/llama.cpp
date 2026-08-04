// Note: this description is outdated
//
// An interface allowing to compute ggml_cgraph with Metal
//
// This is a fully functional interface that extends ggml with GPU support for Apple devices.
// A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, etc.)
//
// How it works?
//
// As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
// use ggml_metal_graph_compute() (or ggml_vulkan_graph_compute(), etc.)
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device memory with the ggml_metal_add_buffer() function. This mapping is
// used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the ggml_metal_set_tensor() and ggml_metal_get_tensor() functions.
//

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

//
// backend API
// user-code should use only these functions
//

// TODO: remove in the future
GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);

GGML_BACKEND_API bool ggml_backend_is_metal(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data);

// helper to check if the device supports a specific family
// ideally, the user code should be doing these checks
// ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
GGML_BACKEND_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);

// capture all command buffers committed the next time `ggml_backend_graph_compute` is called
GGML_BACKEND_API void ggml_backend_metal_capture_next_compute(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);

//
// lock-free control plane: cross-backend MTLSharedEvent
//
// A shared event is a counter that the CPU and Metal backends (and the
// CPU-side ANE sequencer) can read, increment, and wait on. Per the
// prism-engine SharedEventContract pattern: every event protects one
// IOSurface slot (the data-plane handoff); the producer signals the
// event after it publishes its output, the consumer waits for the
// event before it reads the slot.
//
// On the CPU side: signal/wait/try_wait are blocking or non-blocking
// host calls. On the Metal side: the underlying MTLSharedEvent can be
// encoded into an MTLCommandBuffer (encodeWaitForEvent: /
// encodeSignalEvent:) for fully on-GPU synchronization. The ANE leg
// is sequenced through the CPU (ANE itself does not consume
// MTLSharedEvent); the dispatch loop signals the event after ANE
// returns and waits on it before dispatching the next ANE call.
//
typedef struct ggml_mtl_shared_event * ggml_mtl_shared_event_t;

// Create a new shared event with an initial signaled value of 0.
// Returns nullptr if MTLCreateSystemDefaultDevice fails (non-Apple
// or no Metal-capable hardware).
GGML_BACKEND_API ggml_mtl_shared_event_t ggml_mtl_shared_event_new(void);

// Release the event. The underlying MTLSharedEvent is released; any
// pending waits become invalid.
GGML_BACKEND_API void ggml_mtl_shared_event_free(ggml_mtl_shared_event_t event);

// Set the signaled value (any thread). Equivalent to
// [event setSignaledValue:value] on the underlying MTLSharedEvent.
GGML_BACKEND_API void ggml_mtl_shared_event_signal(ggml_mtl_shared_event_t event, uint64_t value);

// Block until the event reaches at least `value`. Equivalent to
// [event waitUntilSignaledValue:value] on the underlying MTLSharedEvent.
GGML_BACKEND_API void ggml_mtl_shared_event_wait(ggml_mtl_shared_event_t event, uint64_t value);

// Non-blocking check. Returns true if the event has reached `value`,
// false otherwise.
GGML_BACKEND_API bool ggml_mtl_shared_event_try_wait(ggml_mtl_shared_event_t event, uint64_t value);

// Read the current signaled value (any thread).
GGML_BACKEND_API uint64_t ggml_mtl_shared_event_get_value(ggml_mtl_shared_event_t event);

// Get the underlying MTLSharedEvent* for use in a Metal command
// buffer (encodeWaitForEvent:value: / encodeSignalEvent:value:).
// The returned pointer is opaque to the caller; pass it back to
// ggml_mtl_shared_event_encode_wait / _signal helpers. The pointer
// is owned by the ggml_mtl_shared_event_t and is invalidated when
// the event is freed.
GGML_BACKEND_API void * ggml_mtl_shared_event_get_mtl_event(ggml_mtl_shared_event_t event);

//
// Dispatch API: tie events to a Metal command buffer.
//
// The dispatch layer composes lock-free CPU/Metal/ANE handoffs. The
// pattern: the producer (a Metal kernel, an ANE dispatch, or a CPU
// write) signals a shared event after it publishes to its output
// IOSurface slot. The consumer encodes a wait for that event value
// into its own command buffer (Metal) or a CPU-side spin (ANE leg).
//
// `cmd_buf` is an opaque `MTLCommandBuffer*` obtained from a Metal
// backend. The dispatch helpers are the only sanctioned way to
// encode a wait/signal on a shared event; callers should not call
// the underlying Metal APIs directly because the helpers also
// record the wait/signal for the dispatch planner's bookkeeping.
//

// Encode a wait for `value` into the Metal command buffer `cmd_buf`.
// Consumer-side: call this BEFORE the consumer's work. The command
// buffer's execution will block on the GPU timeline until the event
// reaches `value`.
GGML_BACKEND_API void ggml_mtl_shared_event_encode_wait(
        ggml_mtl_shared_event_t event,
        void * cmd_buf,
        uint64_t value);

// Encode a signal at `value` into the Metal command buffer `cmd_buf`.
// Producer-side: call this AFTER the producer's work. The signal
// is committed when the command buffer reaches this point in its
// execution.
GGML_BACKEND_API void ggml_mtl_shared_event_encode_signal(
        ggml_mtl_shared_event_t event,
        void * cmd_buf,
        uint64_t value);

#ifdef __cplusplus
}
#endif
