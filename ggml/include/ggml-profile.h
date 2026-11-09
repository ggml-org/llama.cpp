#pragma once

#include "ggml.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GGML_BUILD_PROFILE
extern int g_current_token_idx;
void ggml_profile_init_trace_file(void);
void ggml_profile_flush_trace(void);
void ggml_set_current_token_idx(int idx);
void ggml_trace_log_begin(const char * name, const char * cat, const char * args);
void ggml_trace_log_end(const char * name, const char * cat, const char * args);
GGML_API void ggml_profile_log_op_begin(struct ggml_tensor * tensor, int ith, int nth);
GGML_API void ggml_profile_log_op_end(struct ggml_tensor * tensor, int ith, int nth);
GGML_API void ggml_profile_log_spacemit_op_begin(struct ggml_tensor * tensor, int ith, int nth);
GGML_API void ggml_profile_log_spacemit_op_end(struct ggml_tensor * tensor, int ith, int nth);
void ggml_profile_flush_tls(void);
#else
#define g_current_token_idx 0
#define ggml_profile_init_trace_file() ((void) 0)
#define ggml_profile_flush_trace() ((void) 0)
#define ggml_set_current_token_idx(idx) ((void) (idx))
#define ggml_trace_log_begin(name, cat, args) ((void) 0)
#define ggml_trace_log_end(name, cat, args) ((void) 0)
#define ggml_profile_log_op_begin(tensor, ith, nth) ((void) 0)
#define ggml_profile_log_op_end(tensor, ith, nth) ((void) 0)
#define ggml_profile_log_spacemit_op_begin(tensor, ith, nth) ((void) 0)
#define ggml_profile_log_spacemit_op_end(tensor, ith, nth) ((void) 0)
#define ggml_profile_flush_tls() ((void) 0)
#endif

#ifdef __cplusplus
}
#endif
