#include "ggml-backend-impl.h"
#include "ggml-remoting.h"
#include "virtgpu.h"
#include "/Users/kevinpouget/remoting/llama_cpp/src/ggml/src/ggml-remotingbackend/shared/apir_backend.h"
#include "/Users/kevinpouget/remoting/llama_cpp/src/ggml/src/ggml-remotingbackend/shared/venus_cs_ggml.h"

#define CACHED
//  printf("INFO: ### found response in the cache %s\n", __func__)o


#define REMOTE_CALL_PREPARE(gpu_dev_name, encoder_name, apir_command_type__)		\
  do {									\
    int32_t forward_flag = (int32_t) apir_command_type__;		\
    encoder_name = remote_call_prepare(gpu_dev_name, VIRGL_APIR_COMMAND_TYPE_Forward, forward_flag); \
    if (!encoder) {							\
      FATAL("%s: failed to prepare the remote call encoder :/", __func__); \
    }									\
  } while(0)

#define REMOTE_CALL(gpu_dev_name, encoder_name, decoder_name) \
  do {							      \
    decoder_name = remote_call(gpu_dev_name, encoder_name);   \
    if (!decoder) {					      \
      FATAL("%s: failed to kick the remote call :/", __func__); \
    }								      \
  } while(0)

#define REMOTE_CALL_FINISH(gpu_dev_name, encoder_name, decoder_name)	\
  do {									\
    int32_t ret = remote_call_finish(encoder_name, decoder_name);	\
    if (ret != 0) {							\
      FATAL("%s: failed to forward the API call (code=%d):/", __func__, ret); \
    }									\
  } while(0)
