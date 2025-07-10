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
struct mtmd_ios_message;

struct mtmd_ios_params {
    const char* model_path;
    const char* mmproj_path;
    
    int n_predict;
    int n_ctx;
    int n_threads;
    float temperature;
    
    bool use_gpu;
    bool mmproj_use_gpu;
};

struct mtmd_ios_message {
    const char* role;
    const char* content;
    const unsigned char** image_buffers;
    size_t* image_sizes;
    int n_images;
    const unsigned char** audio_buffers;
    size_t* audio_sizes;
    int n_audios;
};

mtmd_ios_context* mtmd_ios_init(const mtmd_ios_params* params);
void mtmd_ios_free(mtmd_ios_context* ctx);

mtmd_ios_params mtmd_ios_params_default(void);

char* mtmd_ios_generate(mtmd_ios_context* ctx, const mtmd_ios_message* message);

const char* mtmd_ios_get_last_error(mtmd_ios_context* ctx);

void mtmd_ios_string_free(char* str);

#ifdef __cplusplus
}
#endif

#endif 