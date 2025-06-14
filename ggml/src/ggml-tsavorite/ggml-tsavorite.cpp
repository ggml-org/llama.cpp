// -----------------------------------------------------------------------------n
// Copyright (c) 2023 Tsavorite Scalable Intelligence, Inc . All rights reserved.
//
//
// This file is the confidential and proprietary property of
// Tsavorite Scalable Intelligence, Inc
//
// Possession or use of this file requires a written license from
// Tsavorite Scalable Intelligence, Inc

/******************************************************************************
 * File: ggml-tsavorite.cpp
 * Author TSI Inc
 *
 * Description:
 * ***************************************************************************/

#include "ggml-tsavorite.h"
#include <unistd.h>
#include <inttypes.h>
#include <math.h>
#include <string>
#include <iostream>
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "HostShimCAPI.h"
#include "tsi-rt/utils/Profiler.h"

using namespace std;
namespace tsirt = ::tsi::runtime;
typedef struct _txe_device_t *txe_device_s;
typedef struct _txe_compute_pipeline_state_t *txe_compute_pipeline_state_s;
FILE *tsi_op_log_file;
uint64_t num_of_op;

#ifdef USE_COMMAND_BUFFERS
typedef struct _txe_command_queue_t *txe_command_queue_s;
typedef struct _txe_dispatch_queue_t *txe_dispatch_queue_s;
typedef struct _txe_command_buffer_t *txe_command_buffer_s;
#endif /* USE_COMMAND_BUFFERS */
typedef struct ggml_backend_tsavorite_buffer ggml_backend_tsavorite_buffer_s;

struct _txe_device_t {
  char name[100];
  uint32_t max_buf_len;
  size_t recommended_max_working_set_size;
  size_t current_allocated_size;
  int reserved;
  struct _stats {
    struct _op_run_count {
      // Each Kernel operation belong to one tensor. Below count will increment for each Node Tensor
      uint64_t total_tensor_count;
      // This counter increment whenever kernel call are  made
      uint64_t num_of_kernel_call;
      // below field count all tensors whose num of elements are larger than  kernel number of
      // elements
      uint64_t num_of_tensor_spilt;
      // For Any application below field maintain smallest tensor num of elem
      uint64_t min_num_of_elem;
      // For Any application below field maintain largest tensor num of elem
      uint64_t max_num_of_elem;
    } op_run_count[GGML_TSAVORITE_KERNEL_TYPE_COUNT];
  } stats;
};

struct _txe_compute_pipeline_state_t {
  void (*_mlir_fptr_2_input)(void *, void *, void *);
  void (*_mlir_fptr_1_input)(void *, void *);
  std::string kernel_name;
  int reserved;
};

#ifdef USE_COMMAND_BUFFERS
struct _txe_command_queue_t {
  int reserved;
};

struct _txe_dispatch_queue_t {
  int reserved;
};

struct _txe_command_buffer_t {
  int reserved;
};
#endif /* USE_COMMAND_BUFFERS */

static txe_device_s tsi_system_default_device_create();

// kernels

struct ggml_tsavorite_kernel {
  txe_compute_pipeline_state_s pipeline;
};

struct ggml_backend_tsavorite_context {
#ifdef USE_COMMAND_BUFFERS
  txe_command_queue_s queue;

  txe_dispatch_queue_s d_queue;
#endif /* USE_COMMAND_BUFFERS */

  struct ggml_tsavorite_kernel kernels[GGML_TSAVORITE_KERNEL_TYPE_COUNT];

  // capture state
  bool capture_next_compute;
  bool capture_started;

  // command buffer state
  int n_cb;       // number of extra threads used to submit the command buffers
  int n_nodes_0;  // number of nodes submitted by the main thread
  int n_nodes_1;  // remaining number of nodes submitted by the n_cb threads
  int n_nodes_per_cb;

  struct ggml_cgraph *gf;

  // the callback given to the thread pool
  // void (^encode_async)(size_t ith);

#ifdef USE_COMMAND_BUFFERS
  // n_cb command buffers + 1 used by the main thread
  txe_command_buffer_s command_buffers[GGML_TSAVORITE_MAX_COMMAND_BUFFERS + 1];
#endif /* USE_COMMAND_BUFFERS */

  // abort ggml_tsavorite_graph_compute if callback returns true
  ggml_abort_callback abort_callback;
  void *abort_callback_data;

  // picking CPU compute example
  int n_threads;
  ggml_threadpool_t threadpool;

  uint8_t *work_data;
  size_t work_size;
};

// global

// initialized in ggml_backend_tsavorite_reg
static struct ggml_backend_reg g_ggml_backend_tsavorite_reg;
static struct ggml_backend_device g_ggml_backend_tsavorite_device;

// information about a tSavorite device
// note: assumes single GPU device - the default one
// Need to Add Support for multiple GPU devices
static struct ggml_backend_tsavorite_device_context {
  txe_device_s device;
  int ref_count;

  char name[128];
} g_ggml_ctx_dev_main = {
    /*.device                  =*/tsi_nil,
    /*.ref_count               =*/0,
    /*.name                    =*/"",
};

// temporarily defined here for compatibility between ggml-backend and the old API

struct ggml_backend_tsavorite_buffer {
  void *data;
  size_t size;
};

struct ggml_backend_tsavorite_buffer_context {
  void *all_data;
  size_t all_size;
  bool owned;

  // multiple buffers are used only to avoid the maximum buffer size limitation when using mmap
  int n_buffers;
  ggml_backend_tsavorite_buffer_s buffers[GGML_TSAVORITE_MAX_BUFFERS];
};

static txe_device_s tsi_system_default_device_create() {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  txe_device_s device = (txe_device_s)malloc(sizeof(struct _txe_device_t));
  device->max_buf_len = TSAVORITE_DEVICE_MAX_BUF_LEN;
  device->recommended_max_working_set_size = TSAVORITE_DEVICE_MAX_BUF_LEN;
  device->current_allocated_size = 0;
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return device;
}

static void tsi_device_free(txe_device_s device) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  free(device);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return;
}

#ifdef USE_COMMAND_BUFFERS
static txe_command_queue_s tsi_command_queue_create() {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  txe_command_queue_s cqueue = (txe_command_queue_s)malloc(sizeof(struct _txe_command_queue_t));
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return cqueue;
}

static txe_dispatch_queue_s tsi_dispatch_queue_create() {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  txe_dispatch_queue_s dqueue = (txe_dispatch_queue_s)malloc(sizeof(struct _txe_dispatch_queue_t));
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return dqueue;
}

static void tsi_command_queue_free(txe_command_queue_s cqueue) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  if (cqueue)
    free(cqueue);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return;
}

static void tsi_dispatch_queue_free(txe_dispatch_queue_s dqueue) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  if (dqueue)
    free(dqueue);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return;
}
#endif /* USE_COMMAND_BUFFERS */

static void tsi_buffer_free(void *data) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  if (data)
    free(data);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return;
}

static bool tsi_log_setup() {
  tsi_op_log_file = fopen("tsi-op.txt", "w+");
  if (tsi_op_log_file == NULL) {
    printf("Error Creating or opening log file\n");
    return false;
  }
  return true;
}

void ggml_tsi_log_tensor_data(tensor_log log_data) {
  if (!log_data.log_file) {
    GGML_TSAVORITE_LOG_ERROR("%s: error: log file Cant be NULL\n", __func__);
    return;
  }

  switch (log_data.data_type) {
  case GGML_TSAVORITE_TENSOR_HEADER:
    fprintf(log_data.log_file, "\n\n");
    fprintf(log_data.log_file, "#############################################################\n");
    fprintf(log_data.log_file,
            "Tensor Number %ld and Type %d \n leaf1  len %d, leaf2 len %d, Node len %d\n",
            log_data.num_of_op, log_data.kernel_type, log_data.leaf1_len, log_data.leaf2_len,
            log_data.node_len);
    fprintf(log_data.log_file, "############################################################\n");
    fprintf(log_data.log_file, "\n\n");
    fflush(log_data.log_file);
    return;
  case GGML_TSAVORITE_TENSOR_LEAF1:
    fprintf(log_data.log_file, "\n---------------------------------------------------\n");
    fprintf(log_data.log_file, "leaf1 Detail:\n");
    break;
  case GGML_TSAVORITE_TENSOR_LEAF2:
    fprintf(log_data.log_file, "\n---------------------------------------------------\n");
    fprintf(log_data.log_file, "leaf2 Detail:\n");
    break;
  case GGML_TSAVORITE_TENSOR_NODE:
    fprintf(log_data.log_file, "\n---------------------------------------------------\n");
    fprintf(log_data.log_file, "Node Detail:\n");
    break;
  case GGML_TSAVORITE_TENSOR_END_DATA:
    fprintf(log_data.log_file, "DONE WITH THIS OPERATION %ld\n", log_data.num_of_op);
    fprintf(log_data.log_file, "############################################################\n");
    fprintf(log_data.log_file, "\n\n");
    fflush(log_data.log_file);
    return;
  default:
    GGML_TSAVORITE_LOG_ERROR("%s: error: Invalid Data Type Passed\n", __func__);
    return;
  }
  if (!log_data.tensor) {
    GGML_TSAVORITE_LOG_ERROR("%s: error: tensor pointer is  NULL\n", __func__);
    return;
  }
  float *p;
  int64_t count = (log_data.tensor->ne[0]) * (log_data.tensor->ne[1]) * (log_data.tensor->ne[2]) *
                  (log_data.tensor->ne[3]);
  p = (float *)log_data.tensor->data;
  if ((!p) || (count == 0)) {
    fprintf(log_data.log_file, "\n\n");
    fprintf(log_data.log_file, "Tensor Data is Empty");
    fprintf(log_data.log_file, "\n---------------------------------------------------\n");
    fprintf(log_data.log_file, "\n\n");
    fflush(log_data.log_file);
    return;
  }
  fprintf(tsi_op_log_file, "%.16f ", p[0]);
  for (int64_t ii = 1; ii < count; ++ii) {
    if (!(ii % 4))
      fprintf(log_data.log_file, "\n");
    fprintf(log_data.log_file, "%.16f ", p[ii]);
  }
  fprintf(log_data.log_file, "\n\n");
  fprintf(log_data.log_file, "\n---------------------------------------------------\n");
  fflush(log_data.log_file);
  return;
}

static void ggml_tsavorite_disp_stats(struct ggml_backend_tsavorite_context *ctx,
                                      txe_device_s device) {
  if (!ctx || !device) {
    GGML_TSAVORITE_LOG_ERROR(
        "At %s Either backend context or device or both are NULL, hence cant display Stats",
        __func__);
    return;
  }
  for (int i = 0; i < GGML_TSAVORITE_KERNEL_TYPE_COUNT; ++i) {
    if (!ctx->kernels[i].pipeline)
      continue;
    GGML_TSAVORITE_LOG_CONT(
        "\n %s Operation, total tensor: %lu  Number of Kernel Call: %lu  Number of tensor got "
        "spilt: %lu Min Num of Elem %lu Max Num of Elem %lu \n",
        ctx->kernels[i].pipeline->kernel_name.c_str(),
        device->stats.op_run_count[i].total_tensor_count,
        device->stats.op_run_count[i].num_of_kernel_call,
        device->stats.op_run_count[i].num_of_tensor_spilt,
        device->stats.op_run_count[i].min_num_of_elem,
        device->stats.op_run_count[i].max_num_of_elem);
  }
  return;
}

static void _mlir_ciface_txe_add_test (void *src0, void *src1, void *res)
{
    // MemRefDescriptor
    if (!src0 || !src1 || !res)
        return;

    const int Rank = MEM_REF_DESCRIPTOR_RANK;
    MemRefDescriptor<Rank> *srcP0, *srcP1, *nodeP;
    srcP0 = (MemRefDescriptor<Rank> *)src0;
    srcP1 = (MemRefDescriptor<Rank> *)src1;
    nodeP = (MemRefDescriptor<Rank> *)res;

    uint32_t count = srcP0->shape[Rank - 1];
    float *s0      = (float*)srcP0->data;
    float *s1      = (float*)srcP1->data;
    float *n       = (float*)nodeP->data;

    for(uint32_t i=0; i < count; ++i)
        n[i] = s0[i] + s1[i];
    //printf("\n Calling mlir_add cpu function-5 \n");
    return;
}

static void _mlir_ciface_txe_mult_test (void *src0, void *src1, void *res)
{
    // MemRefDescriptor
    if (!src0 || !src1 || !res)
        return;

    const int Rank = MEM_REF_DESCRIPTOR_RANK;
    MemRefDescriptor<Rank> *srcP0, *srcP1, *nodeP;
    srcP0 = (MemRefDescriptor<Rank> *)src0;
    srcP1 = (MemRefDescriptor<Rank> *)src1;
    nodeP = (MemRefDescriptor<Rank> *)res;

    uint32_t count = srcP0->shape[Rank - 1];
    float *s0      = (float*)srcP0->data;
    float *s1      = (float*)srcP1->data;
    float *n       = (float*)nodeP->data;

    for(uint32_t i=0; i < count; ++i)
        n[i] = s0[i]*s1[i];
    return;
}

static txe_compute_pipeline_state_s tsi_kernel_setup(enum ggml_tsavorite_kernel_type kernel_type) {
  txe_compute_pipeline_state_s kernel_pipeline =
      (txe_compute_pipeline_state_s)calloc(1, sizeof(struct _txe_compute_pipeline_state_t));
  bool flag = false;
  if (!kernel_pipeline) {
    GGML_TSAVORITE_LOG_ERROR("Calloc failing while setting up kernel");
    return NULL;
  }
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  switch (kernel_type) {
      case GGML_TSAVORITE_KERNEL_TYPE_ADD:
          if (ggml_tsavorite_kernel_mode_flag == GGML_TSAVORITE_KERNEL_MODE_CPU)
              kernel_pipeline->_mlir_fptr_2_input = &_mlir_ciface_txe_add_test;
          else
              kernel_pipeline->_mlir_fptr_2_input = &_mlir_ciface_txe_add_host;
          kernel_pipeline->kernel_name = "TXE_ADD";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_SUB:
          kernel_pipeline->_mlir_fptr_2_input = &_mlir_ciface_txe_sub_host;
          kernel_pipeline->kernel_name = "TXE_SUB";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_MULT:
          if (ggml_tsavorite_kernel_mode_flag == GGML_TSAVORITE_KERNEL_MODE_CPU)
              kernel_pipeline->_mlir_fptr_2_input = &_mlir_ciface_txe_mult_test;
          else
              kernel_pipeline->_mlir_fptr_2_input = &_mlir_ciface_txe_mult_host;
          kernel_pipeline->kernel_name = "TXE_MULT";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_DIV:
          kernel_pipeline->_mlir_fptr_2_input = &_mlir_ciface_txe_div_host;
          kernel_pipeline->kernel_name = "TXE_DIV";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_SQRT:
          kernel_pipeline->_mlir_fptr_1_input = &_mlir_ciface_txe_sqrt_host;
          kernel_pipeline->kernel_name = "TXE_SQRT";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_NEG:
          kernel_pipeline->_mlir_fptr_1_input = &_mlir_ciface_txe_neg_host;
          kernel_pipeline->kernel_name = "TXE_NEG";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_ABS:
          kernel_pipeline->_mlir_fptr_1_input = &_mlir_ciface_txe_abs_host;
          kernel_pipeline->kernel_name = "TXE_ABS";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_SIN:
          kernel_pipeline->_mlir_fptr_1_input = &_mlir_ciface_txe_sin_host;
          kernel_pipeline->kernel_name = "TXE_SIN";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_SIGMOID:
          kernel_pipeline->_mlir_fptr_1_input = &_mlir_ciface_txe_sigmoid_host;
          kernel_pipeline->kernel_name = "TXE_SIGMOID";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_SILU:
          kernel_pipeline->_mlir_fptr_1_input = &_mlir_ciface_txe_silu_host;
          kernel_pipeline->kernel_name = "TXE_SILU";
          flag = true;
          break;
      default:
          break;
  }
  if (!flag) {
    GGML_TSAVORITE_LOG_INFO("Kernel %d not supported \n", kernel_type);
    if (kernel_pipeline) {
      free(kernel_pipeline);
      kernel_pipeline = NULL;
    }
  }
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return kernel_pipeline;
}

static void tsi_kernel_release(txe_compute_pipeline_state_s kernel_pipeline) {
  // clear kernel_pipeline
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  if (kernel_pipeline) {
    free(kernel_pipeline);
  }
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return;
}

// acquire
static txe_device_s
ggml_backend_tsavorite_device_acq(struct ggml_backend_tsavorite_device_context *ctx) {
  assert(ctx != NULL);
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  if (ctx->device == tsi_nil) {
    ctx->device = tsi_system_default_device_create();
    snprintf(ctx->name, sizeof("txe"), "txe");
  }

  ctx->ref_count++;
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return ctx->device;
}

// release
static void ggml_backend_tsavorite_device_rel(struct ggml_backend_tsavorite_device_context *ctx) {
  assert(ctx != NULL);
  assert(ctx->ref_count > 0);
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  ctx->ref_count--;

  // Need to define function txe_device_free
  if (ctx->ref_count == 0) {
    tsi_device_free(ctx->device);
    ctx->device = tsi_nil;
  }
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
}

// We will use Unified Memory this memory is used for buffer
static void *ggml_tsavorite_host_malloc(size_t n) {
  void *data = NULL;
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  GGML_TSAVORITE_LOG_INFO("\n Allocating memory from tsi_alloc with size  %ld \n", n);
  data = tsi_alloc(n);
  GGML_TSAVORITE_LOG_CONT("\n Allocating memory from tsi_alloc with size  %ld starting memory %p\n",
                          n, data);

  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return data;
}

static struct ggml_backend_tsavorite_context *ggml_tsavorite_init(ggml_backend_dev_t dev) {
  GGML_TSAVORITE_LOG_INFO("%s: Start\n", __func__);
  // Open a file named "tsi-op.txt" in the current directory for writing
  num_of_op = 0;

  if (tsi_log_setup() == false)
    return NULL;

  std::string mainProfilerName = "GGML Tsavorite ";
  tsirt::utils::TSIProfiler::initialize();
  tsirt::utils::TSIScopedProfiler mainProfiler(mainProfilerName);
  
  // TSI Run time Initalization
  tsi_initialize(NUM_OF_TXES, NULL);

  // init context
  struct ggml_backend_tsavorite_context *ctx = (struct ggml_backend_tsavorite_context *)calloc(
      1, sizeof(struct ggml_backend_tsavorite_context));
  struct ggml_backend_tsavorite_device_context *ctx_dev =
      (struct ggml_backend_tsavorite_device_context *)dev->context;

  // setup the devie context
  txe_device_s device = ggml_backend_tsavorite_device_acq(ctx_dev);
  GGML_TSAVORITE_LOG_INFO("%s: picking default device: %s\n", __func__, device->name);
  for (uint32_t op = GGML_TSAVORITE_KERNEL_TYPE_ADD; op < GGML_TSAVORITE_KERNEL_TYPE_COUNT; ++op) {
    device->stats.op_run_count[op].total_tensor_count = 0;
    device->stats.op_run_count[op].num_of_kernel_call = 0;
    device->stats.op_run_count[op].num_of_tensor_spilt = 0;
    device->stats.op_run_count[op].min_num_of_elem = 0;
    device->stats.op_run_count[op].max_num_of_elem = 0;
  }
  ctx->n_threads = GGML_DEFAULT_N_THREADS;
  ctx->threadpool = NULL;
  ctx->work_data = NULL;
  ctx->work_size = 0;
  ctx->abort_callback = NULL;
  ctx->abort_callback_data = NULL;

  // We dont need it for now, we will revisit
#ifdef USE_COMMAND_BUFFERS
  // setting up backend context
  ctx->queue = tsi_command_queue_create();
  ctx->d_queue = tsi_dispatch_queue_create();
#endif /* USE_COMMAND_BUFFERS */

  ctx->capture_next_compute = false;
  ctx->capture_started = false;

  ctx->gf = tsi_nil;
  // ctx->encode_async = tsi_nil;

#ifdef USE_COMMAND_BUFFERS
  for (int i = 0; i < GGML_TSAVORITE_MAX_COMMAND_BUFFERS; ++i) {
    ctx->command_buffers[i] = tsi_nil;
  }
#endif /* USE_COMMAND_BUFFERS */

  // load TSavorite kernels
  {
    for (int i = 0; i < GGML_TSAVORITE_KERNEL_TYPE_COUNT; ++i) {
      ctx->kernels[i].pipeline = tsi_nil;
    }

#define GGML_TSAVORITE_KERNEL(e, supported)                                                        \
  if (supported) {                                                                                 \
    ctx->kernels[e].pipeline = tsi_kernel_setup(e);                                                \
    GGML_TSAVORITE_LOG_INFO(" TSAVORITE SUPPORTED KERNEL ");                                       \
  } else {                                                                                         \
    GGML_TSAVORITE_LOG_WARN("%s: skipping %-40s (not supported)\n", __func__, "kernel_" #e);       \
  }

    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_ADD,                true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_SUB,                true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_MULT,               true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_DIV,                true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_SQRT,               true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_NEG,                true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_ABS,                true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_SIN,                true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_SIGMOID,            true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_SILU,               true);
  }

  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return ctx;
}

static void ggml_tsavorite_free(struct ggml_backend_tsavorite_context *ctx) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  for (int i = 0; i < GGML_TSAVORITE_KERNEL_TYPE_COUNT; ++i) {
    if (ctx->kernels[i].pipeline) {
      tsi_kernel_release(ctx->kernels[i].pipeline);
      ctx->kernels[i].pipeline = tsi_nil;
    }
  }

  // Block_release(ctx->encode_async);
  //
#ifdef USE_COMMAND_BUFFERS
  tsi_command_queue_free(ctx->queue);

  tsi_dispatch_queue_free(ctx->d_queue);
#endif /* USE_COMMAND_BUFFERS */

  free(ctx);

  // TSI run time free
  GGML_TSAVORITE_LOG_INFO("\n Calling tsi_finalize \n");
  // delay to allow any file operations to complete for runtime

  GGML_TSAVORITE_LOG_INFO("Delaying tsi_finalize for 2 sec");
  sleep(2);
  tsi_finalize();
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  tsirt::utils::TSIProfiler::finalize();
  std::cout << "\nGGML Tsavorite Profiling Results:" << std::endl;
  std::cout << tsirt::utils::TSIProfiler::getFormattedResults(
                   /*truncateFuncNames*/ true)
            << std::endl;
}

#if 0
// finds the tSavorite buffer that contains the tensor data on the TXE device unified memory
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// tSavorite buffer based on the host memory pointer
//
static ggml_backend_tsavorite_buffer_s ggml_tsavorite_get_buffer(struct ggml_tensor * t, size_t * offs) {
    // GGML_TSAVORITE_LOG_INFO("%s: data tensor '%16s', offs_data = %8ld, offs_eval = %8ld, offs_cach = %8ld\n", __func__, t->name, offs_data, offs_eval, offs_cach);
    GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

    const int64_t tsize = ggml_nbytes(t);

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    struct ggml_backend_tsavorite_buffer_context * buf_ctx = (struct ggml_backend_tsavorite_buffer_context *) buffer->context;

    // find the view that contains the tensor fully
    for (int i = 0; i < buf_ctx->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) buf_ctx->buffers[i].data;

        // GGML_TSAVORITE_LOG_INFO("ioffs = %10ld, tsize = %10ld, sum = %10ld, buf_ctx->buffers[%d].size = %10ld\n", ioffs, tsize, ioffs + tsize, i, buf_ctx->buffers[i].size);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) buf_ctx->buffers[i].size) {
            *offs = (size_t) ioffs;

            // GGML_TSAVORITE_LOG_INFO("%s: tensor '%16s', offs = %8ld\n", __func__, t->name, *offs);
            GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

            return buf_ctx->buffers[i];
        }
    }

    GGML_TSAVORITE_LOG_ERROR("%s: error: tensor '%s' buffer is tsi_nil\n", __func__, t->name);
    GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

    return tsi_nil;
}
#endif

static bool ggml_tsavorite_supports_op(const struct ggml_backend_tsavorite_device_context *ctx_dev,
                                       const struct ggml_tensor *op) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  if (!ctx_dev)
    return false;
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  for (size_t i = 0, n = 3; i < n; ++i) {
    if (op->src[i] != NULL && op->src[i]->type != GGML_TYPE_F32) {
      return false;
    }
  }

  if (op->type != GGML_TYPE_F32)
    return false;
  switch (op->op) {
  case GGML_OP_NONE:
  case GGML_OP_ADD:
  case GGML_OP_SUB:
  case GGML_OP_MUL:
  case GGML_OP_DIV:
  case GGML_OP_SQRT:
  case GGML_OP_SIN:
    break;
  case GGML_OP_UNARY:
    switch (ggml_get_unary_op(op)) {
    case GGML_UNARY_OP_NEG:
    case GGML_UNARY_OP_ABS:
    case GGML_UNARY_OP_SIGMOID:
    case GGML_UNARY_OP_SILU:
      break;
    default:
      return false;
    }
    break;
  default:
    return false;
  }
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return true;
}

/*
static void ggml_tsavorite_encode_node(
                        ggml_backend_t   backend,
                                   int   idx,
          tsi_command_encoder   encoder) {
}
*/

static void ggml_tsavorite_decompose_unary_kernel_sin(uint32_t num_elem, ggml_tensor *src) {
  float *p = (float *)(src->data);
  for (uint32_t i = 0; i < num_elem; ++i) {
    *p = (*p) / (2 * M_PI);
    ++p;
  }
  return;
}

static void ggml_tsavorite_decompose_unary_kernel(uint32_t num_elem, ggml_tensor *src,
                                                  ggml_tensor *node) {
  switch (node->op) {
  case GGML_OP_SIN:
    ggml_tsavorite_decompose_unary_kernel_sin(num_elem, src);
    break;
  default:
    break;
  }
  return;
}

// nodes are intermediate which has multiple src tensors & operation
// Here we create multiple thread
// Each Thread run the command buffer & pick Tensor and execute and get the result back base on
// async or sync all Compute wil finish all tensors execution
static enum ggml_status ggml_tsavorite_graph_compute(ggml_backend_t backend,
                                                     struct ggml_cgraph *cgraph) {
#if 0
    GGML_LOG_INFO("Start %s\n", __func__);
    struct ggml_backend_tsavorite_context        * ctx     = backend->context;
    struct ggml_backend_tsavorite_device_context * ctx_dev = backend->device->context;

    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = 128;

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // submit the ggml compute graph to the TXE by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel

    GGML_LOG_INFO("End %s\n", __func__);
    return GGML_STATUS_SUCCESS;
#endif

  struct ggml_backend_tsavorite_context *ctx =
      (struct ggml_backend_tsavorite_context *)backend->context;
  if (!ctx) {
    GGML_LOG_ERROR("\n backend ctx is NULL \n");
    return GGML_STATUS_FAILED;
  }

#if 0
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, ctx->n_threads, ctx->threadpool);

    if (ctx->work_size < cplan.work_size) {
        delete[] ctx->work_data;
        ctx->work_data = new uint8_t[cplan.work_size];
        if (ctx->work_data == NULL) {
            ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
        ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)ctx->work_data;

    cplan.abort_callback      = ctx->abort_callback;
    cplan.abort_callback_data = ctx->abort_callback_data;
#endif

  txe_device_s device = ggml_backend_tsavorite_device_acq(
      (struct ggml_backend_tsavorite_device_context *)backend->device->context);

  if (!device) {
    GGML_TSAVORITE_LOG_ERROR("\n tsavorite device is NULL \n");
    return GGML_STATUS_FAILED;
  }
  // MemRefDescriptor
  const int Rank = MEM_REF_DESCRIPTOR_RANK;
  MemRefDescriptor<Rank> *srcP0, *srcP1, *nodeP;
  struct ggml_tensor *src0, *src1, *node;
  uint32_t num_elem_src0, num_elem_src1, num_elem_node;
  enum ggml_tsavorite_kernel_type kernel_type;
  // This variable not needed since src0 or node will have max elem size
  //  and src1 size will min elem size
  uint64_t max_num_of_elem, min_num_of_elem;
  enum ggml_tsavorite_input_tensors_count num_of_input_tensors;
  tensor_log log_data;

  for (int i = 0; i < cgraph->n_nodes; i++) {
    node = cgraph->nodes[i];
    src0 = node->src[0];
    src1 = node->src[1];
    min_num_of_elem = 0;
    max_num_of_elem = 0;

    switch (node->op) {
    case GGML_OP_ADD:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_ADD;
      num_of_input_tensors = TSAVORITE_TWO_INPUT_TENSORS;
      break;
    case GGML_OP_SUB:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_SUB;
      num_of_input_tensors = TSAVORITE_TWO_INPUT_TENSORS;
      break;
    case GGML_OP_MUL:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_MULT;
      num_of_input_tensors = TSAVORITE_TWO_INPUT_TENSORS;
      break;
    case GGML_OP_DIV:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_DIV;
      num_of_input_tensors = TSAVORITE_TWO_INPUT_TENSORS;
      break;
    case GGML_OP_SQRT:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_SQRT;
      num_of_input_tensors = TSAVORITE_UNARY_INPUT_TENSORS;
      break;
    case GGML_OP_SIN:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_SIN;
      num_of_input_tensors = TSAVORITE_UNARY_INPUT_TENSORS;
      break;
    case GGML_OP_UNARY:
      switch (ggml_get_unary_op(node)) {
      case GGML_UNARY_OP_NEG:
        kernel_type = GGML_TSAVORITE_KERNEL_TYPE_NEG;
        num_of_input_tensors = TSAVORITE_UNARY_INPUT_TENSORS;
        break;
      case GGML_UNARY_OP_ABS:
        kernel_type = GGML_TSAVORITE_KERNEL_TYPE_ABS;
        num_of_input_tensors = TSAVORITE_UNARY_INPUT_TENSORS;
        break;
      case GGML_UNARY_OP_SIGMOID:
        kernel_type = GGML_TSAVORITE_KERNEL_TYPE_SIGMOID;
        num_of_input_tensors = TSAVORITE_UNARY_INPUT_TENSORS;
        break;
      case GGML_UNARY_OP_SILU:
        kernel_type = GGML_TSAVORITE_KERNEL_TYPE_SILU;
        num_of_input_tensors = TSAVORITE_UNARY_INPUT_TENSORS;
        break;
      default:
        ggml_backend_tsavorite_device_rel(
            (struct ggml_backend_tsavorite_device_context *)backend->device->context);
        return GGML_STATUS_ABORTED;
      }
      break;
    default:
      ggml_backend_tsavorite_device_rel(
          (struct ggml_backend_tsavorite_device_context *)backend->device->context);
      return GGML_STATUS_ABORTED;
    }

    if (!ctx->kernels[kernel_type].pipeline ||
        (!ctx->kernels[kernel_type].pipeline->_mlir_fptr_2_input &&
         !ctx->kernels[kernel_type].pipeline->_mlir_fptr_1_input)) {
      GGML_TSAVORITE_LOG_ERROR("Kernel Type %d, not supported \n", kernel_type);
      return GGML_STATUS_ABORTED;
    }
    ++num_of_op;

    if (num_of_input_tensors == TSAVORITE_TWO_INPUT_TENSORS) {
      if (node->src[0] && node->src[1]) {
        if (!src0->data || !src1->data || !node->data) {
          GGML_TSAVORITE_LOG_ERROR(
              "One of tensor Data doesnt have memory leaf1 %p, leaf2 %p, node %p \n", src0->data,
              src1->data, node->data);
          ggml_backend_tsavorite_device_rel(
              (struct ggml_backend_tsavorite_device_context *)backend->device->context);
          return GGML_STATUS_ABORTED;
        }
        srcP0 = (MemRefDescriptor<Rank> *)src0->data;
        srcP1 = (MemRefDescriptor<Rank> *)src1->data;
        nodeP = (MemRefDescriptor<Rank> *)node->data;
        // This is for tsavorite MemRef Header hence getting header
        --srcP0;
        --srcP1;
        --nodeP;
        srcP0->data = srcP0->base = src0->data;
        srcP1->data = srcP1->base = src1->data;
        nodeP->data = nodeP->base = node->data;
        // offset & shape size will be update base on Tensor Size
        // TSAVORITE KERNEL CAN Take max of TSAVORITE_KERNEL_SIZE
        // Hence we need to load tensor  data at multiple iteration
        // for large Tensor Dataset
        srcP0->offset = 0;
        srcP1->offset = 0;
        nodeP->offset = 0;

        // currently _mlir_ as restriction to hold max of 64 elements, we need to spilt the work if
        // its more than 64, i will address this at future PR Initalizing num_elem
        num_elem_src0 = 1;
        for (int i = 0; i < GGML_MAX_DIMS && src0->nb[i] != 0; ++i)
          num_elem_src0 *= src0->ne[i];

        num_elem_src1 = 1;
        for (int i = 0; i < GGML_MAX_DIMS && src1->nb[i] != 0; ++i)
          num_elem_src1 *= src1->ne[i];

        num_elem_node = 1;
        for (int i = 0; i < GGML_MAX_DIMS && node->nb[i] != 0; ++i)
          num_elem_node *= node->ne[i];

        if (!num_elem_src0 || !num_elem_src1 || !num_elem_node) {
          GGML_TSAVORITE_LOG_ERROR("\nOne or more of Tensor length is zero of kernel_type %d\n",
                                   kernel_type);
          return GGML_STATUS_ABORTED;
        }

        min_num_of_elem = max_num_of_elem = num_elem_src0;

        if (min_num_of_elem > num_elem_src1)
          min_num_of_elem = num_elem_src1;
        if (min_num_of_elem > num_elem_node)
          min_num_of_elem = num_elem_node;

        if (max_num_of_elem < num_elem_src1)
          max_num_of_elem = num_elem_src1;
        if (max_num_of_elem < num_elem_node)
          max_num_of_elem = num_elem_node;

        if (ggml_tsavorite_log_type_val == GGML_TSAVORITE_LOG_DEBUG) {
          bzero((char *)&log_data, sizeof(log_data));
          log_data.leaf1_len = num_elem_src0;
          log_data.leaf2_len = num_elem_src1;
          log_data.node_len = num_elem_node;
          log_data.log_file = tsi_op_log_file;
          log_data.num_of_op = num_of_op;
          log_data.kernel_type = kernel_type;

          log_data.data_type = GGML_TSAVORITE_TENSOR_HEADER;
          ggml_tsi_log_tensor_data(log_data);

          log_data.data_type = GGML_TSAVORITE_TENSOR_LEAF1;
          log_data.tensor = src0;
          ggml_tsi_log_tensor_data(log_data);

          log_data.data_type = GGML_TSAVORITE_TENSOR_LEAF2;
          log_data.tensor = src1;
          ggml_tsi_log_tensor_data(log_data);
        }

        ggml_tensor *dst = node;
        const int nr = ggml_nrows(src0);

        GGML_TENSOR_BINARY_OP_LOCALS

        for (int ir = 0; ir < nr; ++ir) {
          const int64_t i03 = ir / (ne02 * ne01);
          const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
          const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

          const int64_t i13 = i03 % ne13;
          const int64_t i12 = i02 % ne12;
          const int64_t i11 = i01 % ne11;
          const int64_t nr0 = ne00 / ne10;

          float *dst_ptr = (float *)((char *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
          float *src0_ptr = (float *)((char *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
          float *src1_ptr = (float *)((char *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

          for (int64_t r = 0; r < nr0; ++r) {
            // While loop is added to  handle the scenario when kernel number of elements
            // less than ggml tensor number of elements.GGML tensor number of elements decided
            // base on application like llama.cpp. Currently we have build Kernel elements
            // statically hence we have MACRO: TSAVORITE_KERNEL_SIZE to track this
            int count = 0;
            while (count < ne10) {
              int kernel_size;
              srcP1->data =  srcP1->base = (void *)(src1_ptr + count);
              srcP0->data =  srcP0->base = (void *)(src0_ptr + r * ne10 + count);
              nodeP->data =  nodeP->base = (void *)(dst_ptr + r * ne10 + count);
              if ((count + TSAVORITE_KERNEL_SIZE) > ne10)
                kernel_size = ne10 - count;
              else
                kernel_size = TSAVORITE_KERNEL_SIZE;
              count += kernel_size;
              srcP0->shape[Rank - 1]   = kernel_size;
              srcP1->shape[Rank - 1]   = kernel_size;
              nodeP->shape[Rank - 1]   = kernel_size;
              srcP0->strides[Rank - 1] = 0;
              srcP1->strides[Rank - 1] = 0;
              nodeP->strides[Rank - 1] = 0;
              // kernel call
              ctx->kernels[kernel_type].pipeline->_mlir_fptr_2_input(srcP0, srcP1, nodeP);
              ++device->stats.op_run_count[kernel_type].num_of_kernel_call;
            }
          }
        }

        if (ggml_tsavorite_log_type_val == GGML_TSAVORITE_LOG_DEBUG) {
          log_data.data_type = GGML_TSAVORITE_TENSOR_NODE;
          log_data.tensor = node;
          ggml_tsi_log_tensor_data(log_data);

          log_data.data_type = GGML_TSAVORITE_TENSOR_END_DATA;
          log_data.tensor = NULL;
          ggml_tsi_log_tensor_data(log_data);
        }
      }
    }

    if (num_of_input_tensors == TSAVORITE_UNARY_INPUT_TENSORS) {
      if (node->src[0]) {
        if (!src0->data || !node->data) {
          GGML_TSAVORITE_LOG_ERROR(
              "input or output tensor Data doesnt have memory leaf %p,  node %p \n", src0->data,
              node->data);
          ggml_backend_tsavorite_device_rel(
              (struct ggml_backend_tsavorite_device_context *)backend->device->context);
          return GGML_STATUS_ABORTED;
        }
        srcP0 = (MemRefDescriptor<Rank> *)src0->data;
        nodeP = (MemRefDescriptor<Rank> *)node->data;
        // This is for tsavorite MemRef Header hence getting header
        --srcP0;
        --nodeP;
        srcP0->data = srcP0->base = src0->data;
        nodeP->data = nodeP->base = node->data;
        // offset & shape size will be update base on Tensor Size
        // TSAVORITE KERNEL CAN Take max of TSAVORITE_KERNEL_SIZE
        // Hence we need to load tensor  data at multiple iteration
        // for large Tensor Dataset
        srcP0->offset = 0;
        nodeP->offset = 0;

        // currently _mlir_ as restriction to hold max of 64 elements, we need to spilt the work if
        // its more than 64, i will address this at future PR Initalizing num_elem
        num_elem_src0 = 1;
        for (int i = 0; i < GGML_MAX_DIMS && src0->nb[i] != 0; ++i)
          num_elem_src0 *= src0->ne[i];
        max_num_of_elem = min_num_of_elem = num_elem_src0;

        if (ggml_tsavorite_log_type_val == GGML_TSAVORITE_LOG_DEBUG) {
          bzero((char *)&log_data, sizeof(log_data));
          log_data.leaf1_len = num_elem_src0;
          log_data.leaf2_len = 0;
          log_data.node_len = num_elem_src0;
          log_data.log_file = tsi_op_log_file;
          log_data.num_of_op = num_of_op;
          log_data.kernel_type = kernel_type;

          log_data.data_type = GGML_TSAVORITE_TENSOR_HEADER;
          ggml_tsi_log_tensor_data(log_data);

          log_data.data_type = GGML_TSAVORITE_TENSOR_LEAF1;
          log_data.tensor = src0;
          ggml_tsi_log_tensor_data(log_data);
        }
        // While loop is added to  handle the scenario when kernel number of elements
        // less than ggml tensor number of elements.GGML tensor number of elements decided
        // base on application like llama.cpp. Currently we have build Kernel elements statically
        // hence we have MACRO: TSAVORITE_KERNEL_SIZE to track this
        uint32_t count = 0;

        if (node->op == GGML_OP_SIN) {
          ggml_tsavorite_decompose_unary_kernel(num_elem_src0, src0, node);
        }
        while (count < num_elem_src0) {
          int kernel_size;
          srcP0->data = srcP0->base = (void *)((float *)src0->data + count);
          nodeP->data = nodeP->base = (void *)((float *)node->data + count);
          if ((count + TSAVORITE_KERNEL_SIZE) > num_elem_src0)
            kernel_size = num_elem_src0 - count;
          else
            kernel_size = TSAVORITE_KERNEL_SIZE;
          count += kernel_size;
          srcP0->shape[Rank - 1]    = kernel_size;
          nodeP->shape[Rank - 1]    = kernel_size;
          srcP0->strides[Rank - 1]  = 0;
          nodeP->strides[Rank - 1]  = 0;
          // kernel call
          ctx->kernels[kernel_type].pipeline->_mlir_fptr_1_input(srcP0, nodeP);
          ++device->stats.op_run_count[kernel_type].num_of_kernel_call;
        }

        if (ggml_tsavorite_log_type_val == GGML_TSAVORITE_LOG_DEBUG) {
          log_data.data_type = GGML_TSAVORITE_TENSOR_NODE;
          log_data.tensor = node;
          ggml_tsi_log_tensor_data(log_data);

          log_data.data_type = GGML_TSAVORITE_TENSOR_END_DATA;
          log_data.tensor = NULL;
          ggml_tsi_log_tensor_data(log_data);
        }
      }
    }
    if (min_num_of_elem > 0) {
      ++device->stats.op_run_count[kernel_type].total_tensor_count;

      if (min_num_of_elem > TSAVORITE_KERNEL_SIZE)
        ++device->stats.op_run_count[kernel_type].num_of_tensor_spilt;

      if (!(device->stats.op_run_count[kernel_type].min_num_of_elem) ||
          device->stats.op_run_count[kernel_type].min_num_of_elem > min_num_of_elem)
        device->stats.op_run_count[kernel_type].min_num_of_elem = min_num_of_elem;

      if (!(device->stats.op_run_count[kernel_type].max_num_of_elem) ||
          device->stats.op_run_count[kernel_type].max_num_of_elem < max_num_of_elem)
        device->stats.op_run_count[kernel_type].max_num_of_elem = max_num_of_elem;
    }
  }

  // This this need to implement correctly when we have mixture of CPU and accelerator operation
  // return ggml_graph_compute(cgraph, &cplan);
  ggml_backend_tsavorite_device_rel(
      (struct ggml_backend_tsavorite_device_context *)backend->device->context);
  return GGML_STATUS_SUCCESS;

  GGML_UNUSED(backend);
}

////////////////////////////////////////////////////////////////////////////////

// backend interface

#if 0
static const char * ggml_backend_tsavorite_buffer_get_name(ggml_backend_buffer_t buffer) {
    GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
    GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
    return "tSavorite";

    TSI_UNUSED(buffer);
}
#endif

static void ggml_backend_tsavorite_buffer_free_buffer(ggml_backend_buffer_t buffer) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  struct ggml_backend_tsavorite_buffer_context *ctx =
      (struct ggml_backend_tsavorite_buffer_context *)buffer->context;

#if 0
    // ctx->all_data & tsi_buffer_free(ctx->buffers[i].data and same memory and created by tsi_alloc
    // tsi_finalize called when ggml call backend free all memory
    // this fucntion called when ggml free backend particular buffer, currently we cant provide this support
    // and just return NoOps
    // But at end there is no memory leak but memory can grow since we free at last once backend is shutdown
    // We need to revisit this hence i kept the stuff under if 0
    for (int i = 0; i < ctx->n_buffers; i++) {
        tsi_buffer_free(ctx->buffers[i].data);
    }
    ggml_backend_tsavorite_device_rel((struct ggml_backend_tsavorite_device_context *)buffer->buft->device->context);

    if (ctx->owned) {
        free(ctx->all_data);
    }
#endif

  free(ctx);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
}

static void *ggml_backend_tsavorite_buffer_get_base(ggml_backend_buffer_t buffer) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  struct ggml_backend_tsavorite_buffer_context *ctx =
      (struct ggml_backend_tsavorite_buffer_context *)buffer->context;

  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return ctx->all_data;
}

static ggml_status ggml_backend_tsavorite_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                      struct ggml_tensor *tensor) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  const int Rank = MEM_REF_DESCRIPTOR_RANK;
  MemRefDescriptor<Rank> tensor_data_header;
  tensor->data = (void *)(sizeof(tensor_data_header) + (char *)tensor->data);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return GGML_STATUS_SUCCESS;

  TSI_UNUSED(buffer);
}

static void ggml_backend_tsavorite_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                        struct ggml_tensor *tensor, uint8_t value,
                                                        size_t offset, size_t size) {
  if (!tensor || !tensor->data) {
    GGML_TSAVORITE_LOG_ERROR("\n tensor or data cant be null under func: %s\n", __func__);
    return;
  }
  memset((char *)tensor->data + offset, value, size);

  GGML_UNUSED(buffer);
}

static void ggml_backend_tsavorite_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                     struct ggml_tensor *tensor, const void *data,
                                                     size_t offset, size_t size) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  memcpy((char *)tensor->data + offset, data, size);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  TSI_UNUSED(buffer);
}

static void ggml_backend_tsavorite_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                     const struct ggml_tensor *tensor, void *data,
                                                     size_t offset, size_t size) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  memcpy(data, (const char *)tensor->data + offset, size);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  TSI_UNUSED(buffer);
}

static bool ggml_backend_tsavorite_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                     const struct ggml_tensor *src,
                                                     struct ggml_tensor *dst) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  if (ggml_backend_buffer_is_host(src->buffer)) {
    memcpy(dst->data, src->data, (ggml_nbytes(src)));
    return true;
  }
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return false;

  TSI_UNUSED(buffer);
}

static void ggml_backend_tsavorite_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  struct ggml_backend_tsavorite_buffer_context *ctx =
      (struct ggml_backend_tsavorite_buffer_context *)buffer->context;
  if (!ctx || !ctx->all_data) {
    GGML_TSAVORITE_LOG_ERROR("\n ctx or all_data cant be null under func: %s\n", __func__);
    return;
  }
  memset((char *)ctx->all_data, value, ctx->all_size);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
}

static struct ggml_backend_buffer_i ggml_backend_tsavorite_buffer_i = {
    /* .free_buffer     = */ ggml_backend_tsavorite_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_tsavorite_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_tsavorite_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_tsavorite_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_tsavorite_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_tsavorite_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_tsavorite_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_tsavorite_buffer_clear,
    /* .reset           = */ NULL,
};

// default buffer type

static const char *ggml_backend_tsavorite_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return "tsavorite";

  TSI_UNUSED(buft);
}

static void ggml_backend_tsavorite_log_allocated_size(txe_device_s device, size_t size_aligned) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
#ifndef GGML_TSAVORITE_NDEBUG
#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
  GGML_TSAVORITE_LOG_INFO("%s: allocated buffer, size = %8.2f MiB, (%8.2f)\n", __func__,
                          size_aligned / 1024.0 / 1024.0,
                          device.currentAllocatedSize / 1024.0 / 1024.0);
#endif
#endif
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  TSI_UNUSED(device);
  TSI_UNUSED(size_aligned);
}

static ggml_backend_buffer_t
ggml_backend_tsavorite_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  struct ggml_backend_tsavorite_buffer_context *ctx =
      (struct ggml_backend_tsavorite_buffer_context *)calloc(
          1, sizeof(struct ggml_backend_tsavorite_buffer_context));

  const size_t size_page = sysconf(_SC_PAGESIZE);
  GGML_TSAVORITE_LOG_CONT(
      "ggml_backend_tsavorite_buffer_type_alloc_buffer is called from llama data Loader \n");

  size_t size_aligned = size;
  if ((size_aligned % size_page) != 0) {
    size_aligned += (size_page - (size_aligned % size_page));
  }

  txe_device_s device = ggml_backend_tsavorite_device_acq(
      (struct ggml_backend_tsavorite_device_context *)buft->device->context);
  if (!device)
    return NULL;

  ctx->all_data = ggml_tsavorite_host_malloc(size_aligned);
  ctx->all_size = size_aligned;
  ctx->owned = true;
  ctx->n_buffers = 1;
  GGML_TSAVORITE_LOG_INFO("\n\n\n\n  Memory Starting address %p and size %ld \n\n\n", ctx->all_data,
                          ctx->all_size);

  if (ctx->all_data != NULL) {
    GGML_TSAVORITE_LOG_CONT("\nAddress of Newly Created BUffer %p and size %ld \n", ctx->all_data,
                            ctx->all_size);
    if (ggml_tsavorite_log_type_val == GGML_TSAVORITE_LOG_DEBUG) {
      fprintf(tsi_op_log_file, "Address of Newly Created BUffer %p and size %ld \n", ctx->all_data,
              ctx->all_size);
    }
    ctx->buffers[0].data = NULL;
    ctx->buffers[0].data = ctx->all_data;
    ctx->buffers[0].size = size;
    memset((char *)ctx->all_data, 0, ctx->all_size);
  }

  if (size_aligned > 0 && (ctx->all_data == NULL)) {
    GGML_TSAVORITE_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__,
                             size_aligned / 1024.0 / 1024.0);
    free(ctx);
    ggml_backend_tsavorite_device_rel(
        (struct ggml_backend_tsavorite_device_context *)buft->device->context);
    return NULL;
  }

  // ggml_backend_tsavorite_log_allocated_size(device, size_aligned);
  device->current_allocated_size += ctx->all_size;
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return ggml_backend_buffer_init(buft, ggml_backend_tsavorite_buffer_i, ctx, size);
}

static size_t ggml_backend_tsavorite_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return 32;
  TSI_UNUSED(buft);
}

static size_t ggml_backend_tsavorite_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  txe_device_s device = ggml_backend_tsavorite_device_acq(
      (struct ggml_backend_tsavorite_device_context *)buft->device->context);
  const size_t max_size = device->max_buf_len;
  ggml_backend_tsavorite_device_rel(
      (struct ggml_backend_tsavorite_device_context *)buft->device->context);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return max_size;

  TSI_UNUSED(buft);
}

static size_t ggml_backend_tsavorite_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                                const struct ggml_tensor *tensor) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  txe_device_s device = ggml_backend_tsavorite_device_acq(
      (struct ggml_backend_tsavorite_device_context *)buft->device->context);
  if (!device) {
    GGML_TSAVORITE_LOG_ERROR("\n tsavorite device is NULL \n");
    return 0;
  }
  const int Rank = MEM_REF_DESCRIPTOR_RANK;
  MemRefDescriptor<Rank> tensor_data_header;
  ggml_backend_tsavorite_device_rel(
      (struct ggml_backend_tsavorite_device_context *)buft->device->context);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  GGML_TSAVORITE_LOG_INFO(
      "\n\n\n\n Calculating---- Alloc ----Size header %lu  and data %lu \n\n\n\n ",
      sizeof(tensor_data_header), ggml_nbytes(tensor));

  return (sizeof(tensor_data_header) + ggml_nbytes(tensor));

  TSI_UNUSED(buft);
}

static bool ggml_backend_tsavorite_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  // For Now CPU is loading all data and then copy some tensor to Tsavorite Backend
  // Once we have most of Operation supported by Tsavorite
  // We will figure out to make tsavorite Backend also host
  return false;

  TSI_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_tsavorite_buffer_type(void) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  static struct ggml_backend_buffer_type ggml_backend_buffer_type_tsavorite = {
      /* .iface = */ {
          /* .get_name         = */ ggml_backend_tsavorite_buffer_type_get_name,
          /* .alloc_buffer     = */ ggml_backend_tsavorite_buffer_type_alloc_buffer,
          /* .get_alignment    = */ ggml_backend_tsavorite_buffer_type_get_alignment,
          /* .get_max_size     = */ ggml_backend_tsavorite_buffer_type_get_max_size,
          /* .get_alloc_size   = */
          ggml_backend_tsavorite_buffer_type_get_alloc_size,  // defaults to ggml_nbytes
          /* .is_host          = */ ggml_backend_tsavorite_buffer_type_is_host,
      },
      /* .device  = */ &g_ggml_backend_tsavorite_device,
      /* .context = */ NULL,
  };
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return &ggml_backend_buffer_type_tsavorite;
}

// backend

static const char *ggml_backend_tsavorite_name(ggml_backend_t backend) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return "Tsavorite";

  TSI_UNUSED(backend);
}

static void ggml_backend_tsavorite_free(ggml_backend_t backend) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  if (!backend || !backend->context || !backend->device || !backend->device->context) {
    GGML_TSAVORITE_LOG_ERROR("At %s One of more pointer among: Backend, backend_context, "
                             "device_context or device are NULL",
                             __func__);
    return;
  }
  struct ggml_backend_tsavorite_context *ctx =
      (struct ggml_backend_tsavorite_context *)backend->context;
  struct ggml_backend_tsavorite_device_context *ctx_dev =
      (struct ggml_backend_tsavorite_device_context *)backend->device->context;
  ggml_tsavorite_disp_stats(ctx, ctx_dev->device);

  ggml_backend_tsavorite_device_rel(ctx_dev);
  ggml_tsavorite_free(ctx);

  free(backend);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
}

static void ggml_backend_tsavorite_synchronize(ggml_backend_t backend) {
// We need to implement ASYN  Method to take output of tensor data to input of other Tensor
// We will evaluate and implement at later PR
#ifdef SYNC_DEBUG
  usleep(100000);
#endif /* SYNC_DEBUG */
  TSI_UNUSED(backend);
}

static ggml_backend_buffer_type_t
ggml_backend_tsavorite_get_default_buffer_type(ggml_backend_t backend) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return ggml_backend_tsavorite_buffer_type();

  TSI_UNUSED(backend);
}

static enum ggml_status ggml_backend_tsavorite_graph_compute(ggml_backend_t backend,
                                                             struct ggml_cgraph *cgraph) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return ggml_tsavorite_graph_compute(backend, cgraph);
}

static void ggml_backend_tsavorite_set_n_cb(ggml_backend_t backend, int n_cb) {
  // GGML_ASSERT(ggml_backend_is_tsavorite(backend));
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  struct ggml_backend_tsavorite_context *ctx =
      (struct ggml_backend_tsavorite_context *)backend->context;

  if (ctx->n_cb != n_cb) {
    ctx->n_cb = MIN(n_cb, GGML_TSAVORITE_MAX_COMMAND_BUFFERS);

    if (ctx->n_cb > 2) {
      GGML_TSAVORITE_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade "
                              "the performance in some cases\n",
                              __func__, n_cb);
    }
  }

#if 0
    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }
#endif
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
}

static struct ggml_backend_i ggml_backend_tsavorite_i = {
    /* .get_name                = */ ggml_backend_tsavorite_name,
    /* .free                    = */ ggml_backend_tsavorite_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_tsavorite_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_tsavorite_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static ggml_guid_t ggml_backend_tsavorite_guid(void) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  static ggml_guid guid = {0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed,
                           0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6};
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return &guid;
}

// This need to be removed in the future
ggml_backend_t ggml_backend_tsavorite_init(void) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_tsavorite_reg(), 0);

  struct ggml_backend_tsavorite_context *ctx = ggml_tsavorite_init(dev);
  if (ctx == NULL) {
    GGML_TSAVORITE_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
    return NULL;
  }

  ggml_backend_t backend = (ggml_backend_t)malloc(sizeof(struct ggml_backend));
  if (backend) {
    backend->guid = ggml_backend_tsavorite_guid();
    backend->iface = ggml_backend_tsavorite_i;
    backend->device = dev;
    backend->context = ctx;
  }
  // Will enable later
  // ggml_backend_tsavorite_set_n_cb(backend, 1);

  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return backend;
}

bool ggml_backend_is_tsavorite(ggml_backend_t backend) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_tsavorite_guid());
}

void ggml_backend_tsavorite_set_abort_callback(ggml_backend_t backend,
                                               ggml_abort_callback abort_callback,
                                               void *user_data) {
  GGML_ASSERT(ggml_backend_is_tsavorite(backend));
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  struct ggml_backend_tsavorite_context *ctx =
      (struct ggml_backend_tsavorite_context *)backend->context;

  ctx->abort_callback = abort_callback;
  ctx->abort_callback_data = user_data;
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
}

void ggml_backend_tsavorite_capture_next_compute(ggml_backend_t backend) {
  GGML_ASSERT(ggml_backend_is_tsavorite(backend));
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  struct ggml_backend_tsavorite_context *ctx =
      (struct ggml_backend_tsavorite_context *)backend->context;
  ctx->capture_next_compute = true;
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
}

// backend device

static const char *ggml_backend_tsavorite_device_get_name(ggml_backend_dev_t dev) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return "Tsavorite";

  GGML_UNUSED(dev);
}

static const char *ggml_backend_tsavorite_device_get_description(ggml_backend_dev_t dev) {
  // acq/rel just to populate ctx->name in case it hasn't been done yet
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  struct ggml_backend_tsavorite_device_context *ctx_dev =
      (struct ggml_backend_tsavorite_device_context *)dev->context;
  ggml_backend_tsavorite_device_acq(ctx_dev);
  ggml_backend_tsavorite_device_rel(ctx_dev);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return ctx_dev->name;
}

static void ggml_backend_tsavorite_device_get_memory(ggml_backend_dev_t dev, size_t *free,
                                                     size_t *total) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  if (!dev || !free || !total) {
    GGML_TSAVORITE_LOG_INFO("One of more pointers(dev, free, total) are NULL\n");
    GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
    return;
  }
  *total = 0;
  *total = 0;
  struct ggml_backend_tsavorite_device_context *ctx_dev =
      (struct ggml_backend_tsavorite_device_context *)dev->context;
  if (ctx_dev) {
    txe_device_s device = ggml_backend_tsavorite_device_acq(ctx_dev);
    *total = device->recommended_max_working_set_size;
    *free = *total - device->current_allocated_size;
    GGML_TSAVORITE_LOG_CONT("\n TXE Device MEMORY Summary total %lu and free %lu \n", *total,
                            *free);
    ggml_backend_tsavorite_device_rel(ctx_dev);
  }
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return;
}

// Currently We are setting our TXE accerator at GPU Type
static enum ggml_backend_dev_type ggml_backend_tsavorite_device_get_type(ggml_backend_dev_t dev) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return GGML_BACKEND_DEVICE_TYPE_GPU;

  GGML_UNUSED(dev);
}

// Need to understand the scope of this API since this is not used
// // use by Structure llama_model_loader
// func llm_load_tensors
// structure lama_new_context_with_model
static void ggml_backend_tsavorite_device_get_props(ggml_backend_dev_t dev,
                                                    struct ggml_backend_dev_props *props) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  props->name = ggml_backend_tsavorite_device_get_name(dev);
  props->description = ggml_backend_tsavorite_device_get_description(dev);
  props->type = ggml_backend_tsavorite_device_get_type(dev);
  ggml_backend_tsavorite_device_get_memory(dev, &props->memory_free, &props->memory_total);

  if (props) {
    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = true;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
  }
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
}

static ggml_backend_t ggml_backend_tsavorite_device_init(ggml_backend_dev_t dev,
                                                         const char *params) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  struct ggml_backend_tsavorite_context *ctx = ggml_tsavorite_init(dev);
  if (ctx == NULL) {
    GGML_TSAVORITE_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
    return NULL;
  }

  ggml_backend_t backend = (ggml_backend_t)malloc(sizeof(struct ggml_backend));

  if (backend) {
    backend->guid = ggml_backend_tsavorite_guid();
    backend->iface = ggml_backend_tsavorite_i;
    backend->device = dev;
    backend->context = ctx;
  }

  ggml_backend_tsavorite_set_n_cb(backend, 1);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return backend;

  GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t
ggml_backend_tsavorite_device_get_buffer_type(ggml_backend_dev_t dev) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return ggml_backend_tsavorite_buffer_type();

  GGML_UNUSED(dev);
}

// Currently for llama.cpp model below API it seems not used
// llama.cpp is using as part llm_load_tensors
// buffer_from_host_ptr_supported
// is_default_buft
// else they will be using
// ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
// Need to revist when we will look at buffer section implementation
static ggml_backend_buffer_t ggml_backend_tsavorite_device_buffer_from_ptr(ggml_backend_dev_t dev,
                                                                           void *ptr, size_t size,
                                                                           size_t max_tensor_size) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  struct ggml_backend_tsavorite_buffer_context *ctx =
      (struct ggml_backend_tsavorite_buffer_context *)calloc(
          1, sizeof(struct ggml_backend_tsavorite_buffer_context));

  ctx->all_data = ptr;
  ctx->all_size = size;
  ctx->owned = false;
  ctx->n_buffers = 0;

  const size_t size_page = sysconf(_SC_PAGESIZE);

  // page-align the data ptr
  {
    const uintptr_t offs = (uintptr_t)ptr % size_page;
    ptr = (void *)((char *)ptr - offs);
    size += offs;
  }

  size_t size_aligned = size;
  if ((size_aligned % size_page) != 0) {
    size_aligned += (size_page - (size_aligned % size_page));
  }

  struct ggml_backend_tsavorite_device_context *ctx_dev =
      (struct ggml_backend_tsavorite_device_context *)dev->context;
  txe_device_s device = ggml_backend_tsavorite_device_acq(ctx_dev);

  // the buffer fits into the max buffer size allowed by the device
  if (size_aligned <= device->max_buf_len) {
    ctx->buffers[ctx->n_buffers].data = ptr;
    ctx->buffers[ctx->n_buffers].size = size;

    // ggml_backend_tsavorite_log_allocated_size(device, size_aligned);

    ++ctx->n_buffers;
  } else {
    // this overlap between the views will guarantee that the tensor with the maximum size will
    // fully fit into one of the views
    const size_t size_ovlp = ((max_tensor_size + size_page - 1) / size_page + 1) *
                             size_page;  // round-up 2 pages just in case
    const size_t size_step = device->max_buf_len - size_ovlp;
    const size_t size_view = device->max_buf_len;

    for (size_t i = 0; i < size; i += size_step) {
      const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

      ctx->buffers[ctx->n_buffers].data = (void *)((uint8_t *)ptr + i);
      ctx->buffers[ctx->n_buffers].size = size_step_aligned;

      // ggml_backend_tsavorite_log_allocated_size(device, size_step_aligned);

      if (i + size_step < size) {
        GGML_TSAVORITE_LOG_INFO("\n");
      }

      ++ctx->n_buffers;
    }
  }
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return ggml_backend_buffer_init(ggml_backend_tsavorite_buffer_type(),
                                  ggml_backend_tsavorite_buffer_i, ctx, size);
}

// llama_build_graph -> ggml_backend_supports_op -> gml_backend_dev_supports_op
// basically if true then it will call ggml_backend_sched_set_tensor_backend(lctx.sched.get(), cur,
// backend.get()); here is cur is tensor
static bool ggml_backend_tsavorite_device_supports_op(ggml_backend_dev_t dev,
                                                      const struct ggml_tensor *op) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  struct ggml_backend_tsavorite_device_context *ctx_dev =
      (struct ggml_backend_tsavorite_device_context *)dev->context;

  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return ggml_tsavorite_supports_op(ctx_dev, op);
}

// template<typename F>
// static bool buft_supported(ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev, F & fn) {}
//  ggml_backend_dev_supports_op(dev, op_tensor);
static bool ggml_backend_tsavorite_device_supports_buft(ggml_backend_dev_t dev,
                                                        ggml_backend_buffer_type_t buft) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return buft->iface.get_name == ggml_backend_tsavorite_buffer_type_get_name;

  TSI_UNUSED(dev);
}

// // returns the backend that should be used for the node based on the current locations
// ggml_backend_sched_backend_id_from_cur  -> ggml_backend_offload_op ->
static bool ggml_backend_tsavorite_device_offload_op(ggml_backend_dev_t dev,
                                                     const struct ggml_tensor *op) {
  if (op->type != GGML_TYPE_F32)
    return false;
  switch (op->op) {
  // case GGML_OP_NONE:
  case GGML_OP_ADD:
  case GGML_OP_SUB:
  case GGML_OP_DIV:
  case GGML_OP_MUL:
  case GGML_OP_SQRT:
  case GGML_OP_SIN:
    break;
  case GGML_OP_UNARY:
    switch (ggml_get_unary_op(op)) {
    case GGML_UNARY_OP_NEG:
    case GGML_UNARY_OP_ABS:
    case GGML_UNARY_OP_SIGMOID:
    case GGML_UNARY_OP_SILU:
      break;
    default:
      return false;
    }
    break;
  default:
    return false;
  }
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return true;
  TSI_UNUSED(dev);
}
#ifdef SYNC_DEBUG
static void ggml_backend_tsavorite_device_synchronize(ggml_backend_dev_t dev,
                                                      ggml_backend_event_t event) {
  usleep(100);
  TSI_UNUSED(dev);
  TSI_UNUSED(event);
}
#endif /* SYNC_DEBUG */

static struct ggml_backend_device_i ggml_backend_tsavorite_device_i = {
    /* .get_name             = */ ggml_backend_tsavorite_device_get_name,
    /* .get_description      = */ ggml_backend_tsavorite_device_get_description,
    /* .get_memory           = */ ggml_backend_tsavorite_device_get_memory,
    /* .get_type             = */ ggml_backend_tsavorite_device_get_type,
    /* .get_props            = */ ggml_backend_tsavorite_device_get_props,
    /* .init_backend         = */ ggml_backend_tsavorite_device_init,
    /* .get_buffer_type      = */ ggml_backend_tsavorite_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_tsavorite_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_tsavorite_device_supports_op,
    /* .supports_buft        = */ ggml_backend_tsavorite_device_supports_buft,
    /* .offload_op           = */ ggml_backend_tsavorite_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend registry

static const char *ggml_backend_tsavorite_reg_get_name(ggml_backend_reg_t reg) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return "Tsavorite";

  GGML_UNUSED(reg);
}

static size_t ggml_backend_tsavorite_reg_device_count(ggml_backend_reg_t reg) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return 1;

  GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_tsavorite_reg_device_get(ggml_backend_reg_t reg,
                                                                size_t index) {
  GGML_ASSERT(index == 0);
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return &g_ggml_backend_tsavorite_device;

  GGML_UNUSED(reg);
  GGML_UNUSED(index);
}

static struct ggml_backend_reg_i ggml_backend_tsavorite_reg_i = {
    /* .get_name         = */ ggml_backend_tsavorite_reg_get_name,
    /* .device_count     = */ ggml_backend_tsavorite_reg_device_count,
    /* .device_get       = */ ggml_backend_tsavorite_reg_device_get,
    /* .get_proc_address = */ NULL,
};


ggml_backend_reg_t ggml_backend_tsavorite_reg(void) {
  ggml_tsavorite_log_type_val = GGML_TSAVORITE_LOG_NONE;
  ggml_tsavorite_kernel_mode_flag = GGML_TSAVORITE_KERNEL_MODE_MLIR;
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  g_ggml_backend_tsavorite_reg.iface = ggml_backend_tsavorite_reg_i;
  g_ggml_backend_tsavorite_reg.context = NULL;

  g_ggml_backend_tsavorite_device.iface = ggml_backend_tsavorite_device_i;
  g_ggml_backend_tsavorite_device.reg = &g_ggml_backend_tsavorite_reg;
  g_ggml_backend_tsavorite_device.context = &g_ggml_ctx_dev_main;
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return &g_ggml_backend_tsavorite_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_tsavorite_reg)
