#pragma once

#include "ggml-backend.h"

// the MoE cache computes the experts that are not in the cache within one graph, with the publish, join and merge ops below
// the publish op writes the input and the CPU experts of a layer to mapped host memory, then bumps the request counter
// a CPU thread computes these experts, writes the result and sets the completed counter
// the merge op waits for the completed counter and adds the CPU result to the result of the device
//
// when the device can read the host experts, the publish op picks the CPU misses like the host plan and caches the other misses itself:
// each takes the next victim slot and is copied there from host memory, the host replays the slots given to the selected experts after the graph
// without a CPU share there are no CPU misses and no merge op

#define GGML_MOE_CPU_OK      0
#define GGML_MOE_CPU_STOP    1 // set by the host, the device does not wait for the CPU
#define GGML_MOE_CPU_TIMEOUT 2 // set by the device, the CPU did not answer in time

#define GGML_MOE_CPU_MAX_BANKS 4

// the ClassLRU of the MoE cache: lists of the frequency classes and the half-life of the CRF in tokens
#define GGML_MOE_CPU_N_CLASS        8
#define GGML_MOE_CPU_CRF_HALF_LIFE  128.0f
// the decay of the CRF is 0 from this many tokens on: 2^-64 times a CRF below 2^8 does not change the CRF + 1 of the host
#define GGML_MOE_CPU_N_DECAY        8192

// the counters are on separate cache lines
struct ggml_moe_cpu_request {
    uint32_t request;   // set by the device after it wrote the request
    uint32_t pad0[15];
    uint32_t completed; // set to request by the CPU after it wrote the result
    uint32_t pad1[15];
    uint32_t status;
    int32_t  layer;     // layer of the request
    int32_t  n_cpu;     // experts of the CPU in the request, each layer publishes a request, also without experts of the CPU
    int32_t  n_tokens;  // tokens in the request
    uint32_t pad2[12];
};

// the parameters of the publish and merge ops of a layer, in host memory, the pointers are device pointers
struct ggml_moe_cpu_layer {
    struct ggml_moe_cpu_request * req;
    int32_t * ids;      // [n_used, n_tokens] experts of the CPU, -1 for the experts of the device
    float   * input;    // [n_embd, n_tokens]
    float   * output;   // [n_embd, n_used, n_tokens] result of the CPU
    int32_t * trace;    // [n_used, n_tokens] experts selected by this layer in the last graph
    int32_t   layer;
    int32_t   zero_slot; // the k-th expert of a token reads the zero slot zero_slot + k if it is an expert of the CPU
    uint64_t  timeout_ns;

    // the device caches the misses that the CPU does not compute with the ClassLRU of the host, lru is NULL otherwise
    // lru: [n_slots] key of each slot (layer*n_expert + expert, -1 if empty), [n_slots] prev, [n_slots] next, [n_slots] class list of each slot,
    //      [n_class] head of each list, [n_class] tail, number of slots that hold an expert, then the misses cached by the current layer: count, (expert, slot) pairs,
    //      then at 2*n_ids_max ints after the miss count the hits of the current layer for their deferred bookkeeping: count, n_tokens, (expert, slot) pairs
    int32_t *    lru;
    int32_t *    maps;      // [n_expert, n_layer] slot maps of all the layers, -1 if not cached
    float *      crf;       // [n_expert, n_layer]
    int32_t *    last_use;  // [n_expert, n_layer] clock of the layer at the last use
    int32_t *    clock;     // [n_layer] tokens seen by each layer
    const float * decay;    // [GGML_MOE_CPU_N_DECAY] decay of the CRF after d tokens, then [GGML_MOE_CPU_N_CLASS] least CRF of each class, computed by the host
    uint32_t *   carry;     // [n_layer] fraction of a CPU miss carried over by each layer, as in the host plan
    uint32_t *   carry_log; // a copy in host memory
    int32_t *    slots_log; // [n_used, n_tokens] slots given to the selected experts, in host memory
    uint32_t     share_q;   // share of the misses computed on the CPU once the cache is full, in 1/65536
    int32_t      n_slots;
    int32_t      n_zero;    // zero slots, 0 without a CPU share
    int32_t      n_expert;  // row size of maps
    int32_t      n_ids_max; // most selected experts of one ubatch (n_used*n_tokens), sizes the miss and hit lists in lru
    int32_t      n_banks;
    char *       bank[GGML_MOE_CPU_MAX_BANKS];
    const char * host[GGML_MOE_CPU_MAX_BANKS]; // host experts of the layer, read by the device
    int64_t      size[GGML_MOE_CPU_MAX_BANKS]; // bytes of one expert
};

struct ggml_moe_cpu_misses_api {
    // mapped host memory, returns the device pointer and sets *host, nullptr on failure
    void * (*alloc)(ggml_backend_t backend, size_t size, void ** host);
    void   (*free)(void * host);

    // true if the device can read the host memory at ptr
    bool   (*host_readable)(const void * ptr);

    // the address of the data of a tensor (on the device or in readable host memory) for the parameters of the ops, nullptr on failure
    // NULL: the data pointer
    void * (*addr)(ggml_backend_t backend, const struct ggml_tensor * t);

    // the most tokens for which MUL_MAT_ID with experts of this type writes zero rows for negative ids,
    // so that the misses can be computed in a second pass (see join), 0 for none, NULL: no join op
    int    (*max_two_pass)(ggml_backend_t backend, enum ggml_type type);

    bool cpu_part; // a CPU share of the misses: the merge op and the CPU requests

    // the ops are GGML_OP_CUSTOM ops (ggml_custom_4d) with one of these as fun and the ggml_moe_cpu_layer of the layer as userdata,
    // the backend recognizes and computes them, the CPU backend aborts if it gets one
    //
    // publish: args ids   [n_used, n_tokens] i32, selected experts
    //               cur   [n_embd, n_tokens] f32, input of the experts
    //               map   [n_expert]         i32, view of the device slot maps at the layer, slot of each expert, -1 if not cached
    //               state [1 + n_layer]      i32, request counters in device memory
    //          res  [n_used, n_tokens] i32, slot of each selected expert, zero slots for the experts of the CPU
    //               two pass: [n_used, n_tokens, 2], the slots of the hits and the zero slots of the CPU (-1 for the misses that the device caches),
    //               then the slots of the misses that the device caches (-1 for the others)
    // join:    args slots [n_used, n_tokens] i32, the second plane of publish
    //          res  the same slots, once the misses are copied
    // merge:   args experts [n_embd, n_used, n_tokens] f32, result of the cached experts, the experts of the CPU read zero slots
    //               slots, the result of publish, the first plane with two passes
    //               state
    //          res  experts plus the result of the CPU
    ggml_custom_op_t publish;
    ggml_custom_op_t join;
    ggml_custom_op_t merge;
};

typedef const struct ggml_moe_cpu_misses_api * (*ggml_moe_cpu_misses_get_api_t)(void);
