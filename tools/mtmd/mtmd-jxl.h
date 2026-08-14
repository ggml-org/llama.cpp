#pragma once

// JPEG XL decode helpers for libmtmd.
// Guarded so llama.cpp still builds when MTMD_JXL is off.

#ifdef MTMD_JXL

#include <jxl/decode.h>
#include <jxl/thread_parallel_runner.h>

#include <stdlib.h>
#include <string.h>

// Returns true if buf/len begins with a JPEG XL signature (codestream or container).
static bool jxl_is_jxl(const unsigned char * buf, size_t len) {
    const JxlSignature sig = JxlSignatureCheck(buf, len);
    return sig == JXL_SIG_CODESTREAM || sig == JXL_SIG_CONTAINER;
}

// Decodes the first frame of a JXL image to 8-bit packed RGB (no alpha).
// Caller frees *out with free(). Returns false on failure.
static bool jxl_decode_rgb(const unsigned char * buf, size_t len,
                           unsigned char ** out, int * width, int * height) {
    *out = nullptr;
    *width = 0;
    *height = 0;

    if (!buf || len == 0 || !jxl_is_jxl(buf, len)) {
        return false;
    }

    JxlDecoder * dec = JxlDecoderCreate(nullptr);
    if (!dec) {
        return false;
    }

    void * runner = JxlThreadParallelRunnerCreate(
        nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());
    if (runner) {
        if (JxlDecoderSetParallelRunner(dec, JxlThreadParallelRunner, runner) != JXL_DEC_SUCCESS) {
            JxlThreadParallelRunnerDestroy(runner);
            runner = nullptr;
        }
    }

    bool ok = false;
    unsigned char * pixels = nullptr;

    if (JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) != JXL_DEC_SUCCESS) {
        goto cleanup;
    }

    if (JxlDecoderSetInput(dec, buf, len) != JXL_DEC_SUCCESS) {
        goto cleanup;
    }
    JxlDecoderCloseInput(dec);

    for (;;) {
        const JxlDecoderStatus status = JxlDecoderProcessInput(dec);

        if (status == JXL_DEC_ERROR || status == JXL_DEC_NEED_MORE_INPUT) {
            break;
        }

        if (status == JXL_DEC_SUCCESS) {
            ok = pixels != nullptr;
            break;
        }

        if (status == JXL_DEC_BASIC_INFO) {
            JxlBasicInfo info;
            if (JxlDecoderGetBasicInfo(dec, &info) != JXL_DEC_SUCCESS) {
                break;
            }
            if (info.xsize == 0 || info.ysize == 0) {
                break;
            }
            *width  = (int) info.xsize;
            *height = (int) info.ysize;
            // UINT8 output is already nonlinear sRGB unless a custom CMS is set.
            continue;
        }

        if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
            const JxlPixelFormat fmt = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
            size_t buf_size = 0;
            if (JxlDecoderImageOutBufferSize(dec, &fmt, &buf_size) != JXL_DEC_SUCCESS) {
                break;
            }
            pixels = (unsigned char *) malloc(buf_size);
            if (!pixels) {
                break;
            }
            if (JxlDecoderSetImageOutBuffer(dec, &fmt, pixels, buf_size) != JXL_DEC_SUCCESS) {
                break;
            }
            continue;
        }

        if (status == JXL_DEC_FULL_IMAGE) {
            // First frame only; ignore subsequent animation frames.
            ok = pixels != nullptr;
            break;
        }
    }

cleanup:
    if (!ok && pixels) {
        free(pixels);
        pixels = nullptr;
        *width = 0;
        *height = 0;
    }
    *out = pixels;
    JxlDecoderDestroy(dec);
    if (runner) {
        JxlThreadParallelRunnerDestroy(runner);
    }
    return ok && *out != nullptr && *width > 0 && *height > 0;
}

#endif // MTMD_JXL
