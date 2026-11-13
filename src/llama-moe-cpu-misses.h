#pragma once

#include <stdint.h>

// the MoE cache computes the experts that are not in the cache within one graph, with its own kernels (llama-moe-cpu-misses.cu)
// the CUDA backend runs them in GGML_OP_CUSTOM nodes (see ggml_cuda_kernel in ggml-cuda.h)
// this file is shared by the kernels and the host (llama-moe-cache.cpp)
//
// the publish kernel writes the input and the CPU experts of a layer to pinned host memory, then bumps the request counter
// a CPU thread computes these experts, writes the result and sets the completed counter
// the merge kernel waits for the completed counter and adds the CPU result to the result of the device
//
// when the device can read the host experts, the publish kernel picks the CPU misses like the host plan and caches the other misses itself:
// each takes the next victim slot and is copied there from host memory, the host replays the slots given to the selected experts after the graph
// without a CPU share there are no CPU misses and no merge kernel

#define LLAMA_MOE_CPU_OK      0
#define LLAMA_MOE_CPU_STOP    1 // set by the host, the device does not wait for the CPU
#define LLAMA_MOE_CPU_TIMEOUT 2 // set by the device, the CPU did not answer in time

#define LLAMA_MOE_CPU_MAX_BANKS 4

#define LLAMA_MOE_CPU_BLOCK_SIZE 256
// blocks per bank of the miss copy: PCIe saturates with a few KiB in flight, more blocks only take DRAM bandwidth from the kernels that run beside it
#define LLAMA_MOE_CPU_COPY_BLOCKS 1

// the counters are on separate cache lines
struct llama_moe_cpu_request {
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

// the parameters of the kernels of a layer, the pointers are device pointers
struct llama_moe_cpu_layer {
    struct llama_moe_cpu_request * req;
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
    const float * decay;    // [n_decay] decay of the CRF after d tokens, then [n_class] least CRF of each class, computed by the host
    uint32_t *   carry;     // [n_layer] fraction of a CPU miss carried over by each layer, as in the host plan
    uint32_t *   carry_log; // a copy in host memory
    int32_t *    slots_log; // [n_used, n_tokens] slots given to the selected experts, in host memory
    uint32_t     share_q;   // share of the misses computed on the CPU once the cache is full, in 1/65536
    int32_t      n_class;   // frequency classes of the ClassLRU, at most 32 (a lane of a warp searches each list)
    int32_t      n_decay;   // the decay of the CRF is 0 from n_decay tokens on
    int32_t      n_slots;
    int32_t      n_zero;    // zero slots, 0 without a CPU share
    int32_t      n_expert;  // row size of maps
    int32_t      n_ids_max; // most selected experts of one ubatch (n_used*n_tokens), sizes the miss and hit lists in lru
    int32_t      n_banks;
    char *       bank[LLAMA_MOE_CPU_MAX_BANKS];
    const char * host[LLAMA_MOE_CPU_MAX_BANKS]; // host experts of the layer, read by the device
    int64_t      size[LLAMA_MOE_CPU_MAX_BANKS]; // bytes of one expert
};

// the last argument of each kernel, after the data of the node and of its arguments
//
// llama_moe_cpu_publish, llama_moe_cpu_publish_dev (the device caches the misses):
//          args ids   [n_used, n_tokens] i32, selected experts
//               cur   [n_embd, n_tokens] f32, input of the experts
//               map   [n_expert]         i32, view of the device slot maps at the layer, slot of each expert, -1 if not cached
//               state [1 + n_layer]      i32, request counters in device memory
//          res  [n_used, n_tokens] i32, slot of each selected expert, zero slots for the experts of the CPU
//               two pass: [n_used, n_tokens, 2], the slots of the hits and the zero slots of the CPU (-1 for the misses that the device caches),
//               then the slots of the misses that the device caches (-1 for the others)
// llama_moe_cpu_copy: copies the misses that publish_dev cached to their slots, after publish_dev (its argument)
// llama_moe_cpu_book: the bookkeeping of publish_dev that its slots do not need, after publish_dev (its argument)
//          with two passes both run on side streams, next to the hits pass
// llama_moe_cpu_join (two pass):
//          args slots [n_used, n_tokens] i32, the second plane of publish_dev, then the copy and book nodes
//          res  the same slots, once the misses are copied
// llama_moe_cpu_merge:
//          args experts [n_embd, n_used, n_tokens] f32, result of the cached experts, the experts of the CPU read zero slots
//               slots, the result of publish, the first plane with two passes
//               state
//          res  experts plus the result of the CPU
struct llama_moe_cpu_kernel_params {
    struct llama_moe_cpu_layer layer;
    int32_t n_used;   // experts per token
    int32_t n_tokens;
    int32_t n_embd;   // publish: elements of cur, merge: row size of experts
    int32_t two_pass;
};
