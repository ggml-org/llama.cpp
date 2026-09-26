#pragma once

// shared internal utilities for the mtmd-helper-*.cpp translation units
// (mtmd-helper.cpp, mtmd-helper-gen.cpp)
// NOT part of the public mtmd-helper.h API

#include "ggml.h"
#include "llama.h"
#include "llama-cpp.h"
#include "mtmd.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <vector>

//
// logging
//

struct mtmd_helper_logger {
    ggml_log_callback default_callback = [](ggml_log_level level, const char * text, void * user_data) {
        (void) level;
        (void) user_data;
        fputs(text, stderr);
        fflush(stderr);
    };

    ggml_log_callback log_callback = default_callback;
    void * log_callback_user_data;

    void log_v(enum ggml_log_level level, const char * format, va_list args) {
        if (format == NULL) {
            return;
        }
        va_list args_copy;
        va_copy(args_copy, args);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            log_callback(level, buffer, log_callback_user_data);
        } else {
            char * buffer2 = (char *) calloc(len + 1, sizeof(char));
            vsnprintf(buffer2, len + 1, format, args_copy);
            buffer2[len] = 0;
            log_callback(level, buffer2, log_callback_user_data);
            free(buffer2);
        }
        va_end(args_copy);
    }

    void log(enum ggml_log_level level, const char * format, ...) {
        va_list args;
        va_start(args, format);
        log_v(level, format, args);
        va_end(args);
    }
};

// inline, so all TUs including this header share one instance
inline mtmd_helper_logger g_logger;

#define LOG_DBG(...) g_logger.log(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_INF(...) g_logger.log(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) g_logger.log(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) g_logger.log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

//
// embd batch
//

// helper struct to make working with embd batch easier
// note: this will be removed after llama_batch_ext refactoring
struct decode_embd_batch {
    int n_pos_per_embd;
    int n_mmproj_embd;
    int32_t n_tokens;
    const float * embd;              // [n_tokens, n_mmproj_embd], not owned
    std::vector<llama_pos> pos;      // [n_pos_per_embd, n_tokens], section-major
    std::vector<llama_pos> pos_view; // sliced positions of the last get_view()
    std::vector<int8_t>    logits;
    llama_seq_id seq_id = 0;

    llama_batch_ext_ptr batch; // rendered sub-batch, see render()

    decode_embd_batch(const float * embd, int32_t n_tokens, int n_pos_per_embd, int n_mmproj_embd)
            : n_pos_per_embd(n_pos_per_embd), n_mmproj_embd(n_mmproj_embd), n_tokens(n_tokens), embd(embd) {
        GGML_ASSERT(n_tokens > 0 && n_pos_per_embd > 0 && n_mmproj_embd > 0);
        pos   .resize((size_t) n_tokens * (size_t) n_pos_per_embd);
        logits.resize(n_tokens, 0);
    }

    void set_position_normal(llama_pos pos_0, llama_seq_id seq_id) {
        this->seq_id = seq_id;
        for (int i = 0; i < n_tokens; i++) {
            pos[i] = pos_0 + i;
        }
    }

    // M-RoPE for image
    void set_position_mrope_2d(const std::vector<mtmd_decoder_pos> & rel_pos, llama_seq_id seq_id) {
        GGML_ASSERT(n_pos_per_embd == 4);
        GGML_ASSERT(!rel_pos.empty() && (int32_t)rel_pos.size() == n_tokens);
        this->seq_id = seq_id;
        for (int32_t i = 0; i < n_tokens; i++) {
            const size_t idx = (size_t) i;
            const size_t n   = (size_t) n_tokens;
            pos[idx        ] = rel_pos[i].t;
            pos[idx + n    ] = rel_pos[i].y;
            pos[idx + n * 2] = rel_pos[i].x;
            pos[idx + n * 3] = rel_pos[i].z;
        }
    }

    // M-RoPE for audio
    void set_position_mrope_1d(llama_pos pos_0, llama_seq_id seq_id) {
        GGML_ASSERT(n_pos_per_embd == 4);
        this->seq_id = seq_id;
        for (int i = 0; i < n_tokens; i++) {
            const size_t idx = (size_t) i;
            const size_t n   = (size_t) n_tokens;
            pos[idx        ] = pos_0 + i;
            pos[idx + n    ] = pos_0 + i;
            pos[idx + n * 2] = pos_0 + i;
            pos[idx + n * 3] = pos_0 + i;
        }
    }

    // describe the entries [offset, offset + n) with section-major positions
    mtmd_helper_embd_batch get_view(int offset, int n) {
        GGML_ASSERT(offset >= 0 && n > 0 && offset + n <= n_tokens);
        pos_view.clear();
        pos_view.reserve((size_t) n * (size_t) n_pos_per_embd);
        for (int j = 0; j < n_pos_per_embd; j++) {
            const size_t src = (size_t) j * (size_t) n_tokens + (size_t) offset;
            pos_view.insert(pos_view.end(), pos.data() + src, pos.data() + src + n);
        }
        return {
            /*n_tokens =*/ n,
            /*embd     =*/ embd + (size_t) offset * n_mmproj_embd,
            /*n_embd   =*/ n_mmproj_embd,
            /*pos      =*/ pos_view.data(),
            /*n_pos    =*/ n_pos_per_embd,
            /*seq_id   =*/ seq_id,
        };
    }

    // render the entries [offset, offset + n) into a batch owned by this object, ready for llama_process()
    llama_batch_ext * render(llama_context * lctx, int offset, int n) {
        GGML_ASSERT(offset >= 0 && n > 0 && offset + n <= n_tokens);
        if (!batch) {
            batch.reset(llama_batch_ext_init(lctx));
        }
        llama_batch_ext_clear(batch.get());
        for (int i = offset; i < offset + n; i++) {
            const llama_embd e = { embd + (size_t) i * n_mmproj_embd, 1, (size_t) n_mmproj_embd };
            const int32_t idx = llama_batch_ext_add_embd(batch.get(), seq_id, e);
            GGML_ASSERT(idx >= 0);

            llama_pos p[GGML_MROPE_SECTIONS] = { 0, 0, 0, 0 };
            for (int j = 0; j < n_pos_per_embd; j++) {
                p[j] = pos[(size_t) j * (size_t) n_tokens + (size_t) i];
            }
            llama_batch_ext_set_pos(batch.get(), idx, p);

            if (logits[i]) {
                llama_batch_ext_set_output_logits(batch.get(), idx, true);
            }
        }
        return batch.get();
    }
};
