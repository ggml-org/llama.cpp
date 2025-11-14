#include "ggml-backend-impl.h"
#include "ggml-remoting.h"
#include "virtgpu.h"
#include "../ggml-remotingbackend/shared/apir_backend.h"
#include "../ggml-remotingbackend/shared/venus_cs_ggml.h"

#define CACHED
//  printf("INFO: ### found response in the cache %s\n", __func__)o


#define REMOTE_CALL_PREPARE(gpu_dev_name, encoder_name, apir_command_type__)		\
  do {									\
    int32_t forward_flag = (int32_t) apir_command_type__;		\
    encoder_name = remote_call_prepare(gpu_dev_name, APIR_COMMAND_TYPE_Forward, forward_flag); \
    if (!encoder_name) {							\
      FATAL("%s: failed to prepare the remote call encoder :/", __func__); \
    }									\
  } while(0)


#define REMOTE_CALL(gpu_dev_name, encoder_name, decoder_name, ret_name) \
  do {									\
    ret_name = (ApirForwardReturnCode) remote_call(gpu_dev_name, encoder_name, &decoder_name, 0, NULL); \
    if (!decoder_name) {						\
      FATAL("%s: failed to kick the remote call :/", __func__);		\
    }									\
    if (ret_name < APIR_FORWARD_BASE_INDEX) {				\
      FATAL("%s: failed to forward the API call: %s: code %d", __func__, apir_forward_error(ret_name), ret_name); \
    }									\
    ret_name = (ApirForwardReturnCode) (ret_name - APIR_FORWARD_BASE_INDEX); \
  } while(0)
