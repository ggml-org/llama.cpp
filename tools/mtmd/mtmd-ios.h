#ifndef MTMD_IOS_H
#define MTMD_IOS_H

#include "mtmd.h"
#include "mtmd-helper.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "chat.h"

#include <string>
#include <vector>
#include <functional>
#include <memory>

#ifdef __cplusplus
extern "C" {
#endif

struct mtmd_ios_context;
struct mtmd_ios_params;

struct mtmd_ios_params {
    const char* model_path;
    const char* mmproj_path;
    
    int n_predict;
    int n_ctx;
    int n_threads;
    float temperature;
    
    bool use_gpu;
    bool mmproj_use_gpu;
    bool warmup;
};



mtmd_ios_context* mtmd_ios_init(const mtmd_ios_params* params);
void mtmd_ios_free(mtmd_ios_context* ctx);

mtmd_ios_params mtmd_ios_params_default(void);

int mtmd_ios_prefill_image(mtmd_ios_context* ctx, const char* image_path);
int mtmd_ios_prefill_text(mtmd_ios_context* ctx, const char* text, const char* role);

typedef struct {
    char* token;
    bool is_end;
} mtmd_ios_token;

mtmd_ios_token mtmd_ios_loop(mtmd_ios_context* ctx);

const char* mtmd_ios_get_last_error(mtmd_ios_context* ctx);

void mtmd_ios_string_free(char* str);

#ifdef __cplusplus
}
#endif

#endif 