#pragma once

//
// Runtime speculative-decoding engine for the Tessera app (extern-C ABI).
//
// This is the PRODUCTION generation path: trunk + drafter loaded once,
// generation driven by the common_speculative_begin/draft/process/accept
// loop (the same live-loop API llama-cli and llama-server use), with
// optional per-step telemetry records emitted through a callback.
//
// Records are schema-identical to llama-imatrix --telemetry-out
// (llama.tessera.spec.v1) plus the additive fields "provenance":"runtime"
// and "sid":"<session uuid>". See docs/tessera-runtime-traces-design.md
// sections 5 and 8.
//
// The caller owns the backend lifetime: llama_backend_init() must have
// been called before tessera_rt_load() (the CLlama shim does this when it
// resolves libllama).
//

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tessera_rt tessera_rt;

typedef void (*tessera_rt_token_cb)(const char * piece, int32_t token_id,
                                    void * ud);
typedef void (*tessera_rt_trace_cb)(const char * jsonl_line, void * ud);

// Load trunk + drafter, build contexts and the spec handle.
// draft_max: max drafted tokens per step.
// Returns NULL on failure; tessera_rt_last_error() carries the reason.
tessera_rt * tessera_rt_load(const char * trunk_path,
                             const char * draft_path,
                             uint32_t n_ctx,
                             int32_t  n_threads,
                             int32_t  n_gpu_layers,
                             int32_t  draft_max);

// Tokenize + decode the prompt, then generate with spec decoding.
// telemetry_topk: 0 = no trace emission (cheap path); > 0 = emit one
// spec.v1 record per spec step through on_trace. on_trace may be NULL.
// max_tokens <= 0 means no generation limit.
// Returns tokens generated, or -1 on error.
int32_t tessera_rt_generate(tessera_rt * rt,
                            const char * prompt,
                            int32_t max_tokens,
                            int32_t telemetry_topk,
                            tessera_rt_token_cb on_token,
                            tessera_rt_trace_cb on_trace,
                            void * ud);

void tessera_rt_free(tessera_rt * rt);
const char * tessera_rt_last_error(void);

#ifdef __cplusplus
}
#endif
