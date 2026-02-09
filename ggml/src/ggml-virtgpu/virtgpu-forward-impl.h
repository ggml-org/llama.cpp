#include "virtgpu.h"

#include "ggml-remoting.h"
#include "backend/shared/apir_backend.h"
#include "backend/shared/apir_cs_ggml.h"

#include "ggml-backend-impl.h"

#define REMOTE_CALL_PREPARE(gpu_dev_name, encoder_name, apir_command_type__)                               \
    do {                                                                                                   \
        int32_t forward_flag = (int32_t) apir_command_type__;                                              \
        encoder_name         = remote_call_prepare(gpu_dev_name, APIR_COMMAND_TYPE_FORWARD, forward_flag); \
        if (!encoder_name) {                                                                               \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);                       \
        }                                                                                                  \
    } while (0)

#define REMOTE_CALL(gpu_dev_name, encoder_name, decoder_name, ret_name)                                           \
    do {                                                                                                          \
        ret_name = (ApirForwardReturnCode) remote_call(gpu_dev_name, encoder_name, &decoder_name, 0, NULL);       \
        if (!decoder_name) {                                                                                      \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to kick the remote call", __func__);                                         \
        }                                                                                                         \
        if (ret_name < APIR_FORWARD_BASE_INDEX) {                                                                 \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to forward the API call: %s: code %d", __func__,                             \
                       apir_forward_error(ret_name), ret_name);                                                   \
        }                                                                                                         \
        ret_name = (ApirForwardReturnCode) (ret_name - APIR_FORWARD_BASE_INDEX);                                  \
    } while (0)
