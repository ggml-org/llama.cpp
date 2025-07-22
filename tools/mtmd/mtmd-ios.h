#ifndef MTMD_IOS_H
#define MTMD_IOS_H

#include <string>

#include "ggml.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

// Context structure
typedef struct mtmd_ios_context mtmd_ios_context;

// Parameters structure (C++ only)
typedef struct mtmd_ios_params {
    std::string model_path;
    std::string mmproj_path;
    int         n_predict;
    int         n_ctx;
    int         n_threads;
    float       temperature;
    bool        use_gpu;
    bool        mmproj_use_gpu;
    bool        warmup;
} mtmd_ios_params;

// Loop return value structure (C++ only)
typedef struct {
    char * token;
    bool   is_end;
} mtmd_ios_token;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize, returns 0 on success, -1 on failure
// Parameters:
// params: parameters
mtmd_ios_context * mtmd_ios_init(const mtmd_ios_params * params);

// Free resources
// Parameters:
// ctx: context
void mtmd_ios_free(mtmd_ios_context * ctx);

// Get default parameters
mtmd_ios_params mtmd_ios_params_default(void);

// Prefill image, returns 0 on success, -1 on failure
// Parameters:
// ctx: context
// image_path: image path
int mtmd_ios_prefill_image(mtmd_ios_context * ctx, const std::string & image_path);

// Prefill text, returns 0 on success, -1 on failure
// Parameters:
// ctx: context
// text: text
// role: role
int mtmd_ios_prefill_text(mtmd_ios_context * ctx, const std::string & text, const std::string & role);

// Loop, returns 0 on success, -1 on failure
// Parameters:
// ctx: context
mtmd_ios_token mtmd_ios_loop(mtmd_ios_context * ctx);

// Get last error message
// Parameters:
// ctx: context
const char * mtmd_ios_get_last_error(mtmd_ios_context * ctx);

// Free string
// Parameters:
// str: string
void mtmd_ios_string_free(char * str);

// Clean kv-cache
// Parameters:
// ctx: context
bool mtmd_ios_clean_kv_cache(mtmd_ios_context * ctx);

#ifdef __cplusplus
}
#endif

#endif
