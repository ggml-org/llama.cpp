// ane-state.h — multifunction .mlmodelc state-layout manifest (v1)
//
// The multifunction ANE bundle (prefill_sN, mtp_predict, dflash_bN,
// hybrid_bN) is stateless at the Core ML level: every function is
// invoked via [model predictionFromFeatures:options:error:] with no
// MLState, and any "state" the functions appear to have lives in
// IOSurface-mapped slots declared in this manifest.
//
// The conversion tool (tools/ane-mtp/state_layout.py) emits a
// ane_state_layout_v1.json next to the .mlmodelc; the runtime
// (common/ane-mtp.mm, ggml/src/ggml-ane/ggml-ane.mm) reads the
// JSON, allocates one big IOSurface for the state, and pins each
// declared slot to a deterministic offset. The IOSurface is shared
// across ANE, Metal, and CPU (zero-copy) and is the canonical
// state for the bundle. Lock-free coordination between producers
// and consumers is via MTLSharedEvent + per-slot atomic flags.
//
// Versioning: bump ANE_STATE_LAYOUT_VERSION when the binary layout
// changes. The runtime rejects unknown versions; the conversion
// tool refuses to write a version it doesn't recognize.
//
// The JSON form is the source of truth; the C struct is a
// deserialized view. An on-disk .mlmodelc may carry the JSON
// alongside (e.g. in the parent directory or a sidecar), or the
// runtime may accept it via a separate path argument.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ANE_STATE_LAYOUT_VERSION 1
#define ANE_STATE_SLOT_NAME_MAX 64
#define ANE_STATE_FUNCTION_NAME_MAX 64
#define ANE_STATE_BUNDLE_NAME_MAX 128
#define ANE_STATE_SLOTS_MAX 64
#define ANE_STATE_FUNCTIONS_MAX 32
#define ANE_STATE_DEPS_MAX 128
#define ANE_STATE_SLOT_IO_MAX 8

typedef enum {
    // The function reads this slot on every call. The runtime
    // guarantees the slot's content is published (via
    // ggml_mtl_shared_event_signal or equivalent) before the
    // function is dispatched. ANE reads it zero-copy.
    ANE_SLOT_KIND_INPUT  = 0,

    // The function writes this slot as the result of prediction.
    // The runtime passes an IOSurface-backed MLMultiArray via
    // MLPredictionOptions.outputBackings so Core ML writes
    // directly into our IOSurface (zero-copy, no result memcpy).
    ANE_SLOT_KIND_OUTPUT = 1,

    // Persistent state. The function reads the previous value,
    // computes a new value, and writes the new value back. The
    // K/V cache lives in slots of this kind. Producers and
    // consumers across the dependency graph share the slot;
    // lock-free coordination is per-slot via an atomic
    // generation counter.
    ANE_SLOT_KIND_STATE  = 2,

    // Transient buffer. The runtime allocates a fresh IOSurface
    // region (or reuses a scratch pool slot) per call. Used
    // for intermediate activations that don't survive past
    // one function's invocation.
    ANE_SLOT_KIND_SCRATCH = 3,
} ane_slot_kind_t;

typedef enum {
    ANE_DTYPE_F32 = 0,
    ANE_DTYPE_F16 = 1,
    ANE_DTYPE_I32 = 2,
} ane_dtype_t;

typedef enum {
    ANE_ROLE_UNKNOWN  = 0,
    // Whole-layer prefill slab. Reads token_ids, positions;
    // writes hidden_states, key_states, value_states.
    ANE_ROLE_PREFILL  = 1,
    // Multi-token prediction. Reads h_nextn, token_ids,
    // positions; writes top_token, confidence, next_hidden.
    ANE_ROLE_MTP      = 2,
    // DFlash draft block. Reads target_features, token_ids,
    // positions; writes draft_tokens, confidence.
    ANE_ROLE_DFLASH   = 3,
    // Candidate arbitration. Reads both dflash and mtp outputs;
    // writes selected_source, agreement.
    ANE_ROLE_HYBRID   = 4,
    // K/V scatter into state. Implemented as a memcpy on the
    // E-core pump, NOT a Core ML function (the manifest may
    // carry a Core ML function for CPU-only fallback but the
    // runtime prefers the memcpy path).
    ANE_ROLE_SYNC     = 5,
    // K/V clear. Implemented as a memset on the E-core pump.
    ANE_ROLE_RESET    = 6,
    // Generic matmul (the W0/W1 spike's path). Stateless;
    // input "x" and output "y" only.
    ANE_ROLE_MATMUL   = 7,
    // Per-row RMSNorm. Stateless, one input row and one output
    // row of the same length (the W2 body-op spike's path).
    ANE_ROLE_RMS_NORM = 8,
    // Row softmax. Stateless, one input row and one output row
    // of the same length.
    ANE_ROLE_SOFT_MAX = 9,
    // Rotary position embedding (gemma 4 variant). Reads the
    // query/key tensor and the positions; writes the rotated
    // tensor in place.
    ANE_ROLE_ROPE     = 10,
    // Gated linear unit (gemma 4 split-form: input is the
    // concatenated [gate | up] row, output is gate_act * up).
    // Stateless.
    ANE_ROLE_GLU      = 11,
    // Embedding lookup (get_rows). Reads the embedding matrix
    // and a vector of token ids; writes the looked-up rows.
    ANE_ROLE_GET_ROWS = 12,
} ane_role_t;

// Core ML model spec type. Determines whether
// MLModelConfiguration.functionName is settable at load time.
// The W0 spike's matmul is NeuralNetwork (functionName MUST be
// nil); the multifunction prefill/MTP/DFlash bundles are ML
// Program (functionName required to pick which named function
// to bind). The conversion tool sets this on manifest emit and
// the runtime reads it to pick the load path.
typedef enum {
    ANE_MODEL_TYPE_NEURAL_NETWORK = 0,
    ANE_MODEL_TYPE_ML_PROGRAM     = 1,
} ane_model_type_t;

typedef struct {
    char            name[ANE_STATE_SLOT_NAME_MAX];
    ane_slot_kind_t kind;
    ane_dtype_t     dtype;
    uint32_t        n_dim;
    uint32_t        shape[4];
    // Byte offset in the state IOSurface. The runtime aligns
    // each slot to 16 KB (ANE page size); the conversion tool
    // emits already-aligned offsets.
    size_t          offset;
    // Byte size, padded to 16 bytes (ANE constraint; matches
    // ggml-ane.mm's GGML_ANE_PAGE = 16 KB minimum page; the
    // per-slot alignment is 16 bytes for SIMD safety).
    size_t          size_bytes;
} ane_slot_v1_t;

typedef struct {
    char          name[ANE_STATE_FUNCTION_NAME_MAX];
    ane_role_t    role;
    // Sequence or batch bucket. For prefill_sN this is N.
    // For matmul this is the output dim. 0 means non-bucketed.
    uint32_t      bucket;
    // True if the function reads or writes any STATE-kind slot.
    // False for stateless functions (the W0 matmul).
    bool          stateful;
    // Slot ids (index into the slots[] array).
    uint32_t      n_inputs;
    uint32_t      input_slot_ids[ANE_STATE_SLOT_IO_MAX];
    uint32_t      n_outputs;
    uint32_t      output_slot_ids[ANE_STATE_SLOT_IO_MAX];
    // Optional: a Core ML function name to invoke. For
    // multifunction bundles this is the function's declared
    // name (e.g., "prefill_s32"). For single-function bundles
    // (the W0 case) this is the default function name "main".
    // The sync/reset roles may leave this empty; they are
    // implemented as memcpy/memset on the pump and do not
    // invoke a Core ML function.
    char          core_ml_function_name[ANE_STATE_FUNCTION_NAME_MAX];
    // True iff this function should be executed on the ANE.
    // sync/reset roles are CPU-only on the E-core pump and
    // set this to false.
    bool          use_ane;
} ane_function_v1_t;

// Directed edge: function `producer` writes `slot_id`; some
// other function (in the same layout) reads it. The runtime
// uses this list to build the per-slot dependency graph for
// the E-core pump's lock-free state machine.
typedef struct {
    uint32_t producer_function_id;  // index into functions[]
    uint32_t slot_id;               // index into slots[]
    // For multi-consumer slots, list each consumer here too.
    // (K/V cache slots typically have 3-4 consumers: prefill,
    // mtp, dflash, hybrid.) The runtime builds the fan-in
    // fan-out graph from this.
    uint32_t consumer_function_ids[ANE_STATE_FUNCTIONS_MAX];
    uint32_t n_consumers;
} ane_function_dep_t;

typedef struct {
    uint32_t           version;
    char               bundle_name[ANE_STATE_BUNDLE_NAME_MAX];
    // Total byte size of the state IOSurface. The runtime
    // allocates one IOSurface of this size at load and pins
    // all slots inside it.
    size_t             state_size_bytes;
    // Core ML spec type of the underlying .mlmodelc. The
    // runtime uses this to decide whether to set
    // MLModelConfiguration.functionName at load (only legal
    // for ANE_MODEL_TYPE_ML_PROGRAM).
    ane_model_type_t   model_type;
    uint32_t           n_slots;
    ane_slot_v1_t      slots[ANE_STATE_SLOTS_MAX];
    uint32_t           n_functions;
    ane_function_v1_t  functions[ANE_STATE_FUNCTIONS_MAX];
    uint32_t           n_deps;
    ane_function_dep_t deps[ANE_STATE_DEPS_MAX];
} ane_state_layout_v1_t;

#ifdef __cplusplus
}
#endif
