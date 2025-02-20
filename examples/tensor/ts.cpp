#include "trace_driver.h"
#include <iostream>
#include <iomanip>
#include <chrono>

#define NR_DPUS 8
#define NR_LAYER 2
#define DPU_BINARY "./dpu/gemv_dpu"

void fp_table_init(void) {
  for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
            }
}

int gemv_dpu_kernel(struct pim_context *context, struct ggml_tensor * w, struct ggml_tensor * in_q, struct ggml_tensor * res) {
  uint32_t pim_offset = 0;
  struct dpu_set_t dpu;

  //ggml_table_f32_f16 tbl is transferred to pim
  DPU_ASSERT(dpu_broadcast_to(context->dpu_set, DPU_MRAM_HEAP_POINTER_NAME, pim_offset, (void *)(ggml_table_f32_f16), sizeof(ggml_table_f32_f16), DPU_XFER_DEFAULT));
  pim_offset += sizeof(ggml_table_f32_f16);

  std::cout << "ggml_table_f32_f16 len = " << sizeof(ggml_table_f32_f16) << std::endl;

  // Transfer pim_metadata into DPUs
  context->pim_metadata.layer_num = NR_LAYER;
  context->pim_metadata.weight_type = (uint16_t)(w->type);

  //ne[1] is row num,ne[0] is col num ?
  context->pim_metadata.rows_per_dpu = w->ne[1] / NR_DPUS;
  context->pim_metadata.rest_rows = w->ne[1] % NR_DPUS;
  GGML_ASSERT(context->pim_metadata.rest_rows == 0);

  context->pim_metadata.layer_len = w->nb[1] * (context->pim_metadata.rows_per_dpu);
  context->pim_metadata.input_offset = sizeof(ggml_table_f32_f16) + sizeof(struct pim_meta) + context->pim_metadata.layer_len * NR_LAYER;

  std::cout << "layer_num = " << NR_LAYER << ", weight_type = " << (uint16_t)(w->type) << ", rows_per_dpu = " << w->ne[1] / NR_DPUS << ", rest_rows = " << w->ne[1] % NR_DPUS << ", layer_len = " << context->pim_metadata.layer_len << ", input_offset = " << context->pim_metadata.input_offset << std::endl;

  //Todo: NR_DPUS contexts are dispatched to different dpus(rest row is different on different dpu)
  DPU_ASSERT(dpu_broadcast_to(context->dpu_set, DPU_MRAM_HEAP_POINTER_NAME, pim_offset, &(context->pim_metadata), sizeof(struct pim_meta), DPU_XFER_DEFAULT));
  pim_offset += sizeof(struct pim_meta);

  // Transfer weight into DPUs
  uint32_t layer_len = context->pim_metadata.layer_len;
  uint32_t i;
  for (uint32_t layeridx = 0; layeridx < NR_LAYER; layeridx++) {
    uint32_t size_per_row = w->nb[1];
    // row is send to dpu
    DPU_FOREACH(context->dpu_set, dpu, i) {
      uint32_t prev_rows_dpu = i * context->pim_metadata.rows_per_dpu;

      // every dpu's data
      DPU_ASSERT(dpu_prepare_xfer(dpu, ((unsigned char *)w->data) + prev_rows_dpu * size_per_row));
    }

    DPU_ASSERT(dpu_push_xfer(context->dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, pim_offset + layer_len * layeridx, layer_len, DPU_XFER_DEFAULT));
  }

  // Transfer input into DPUs
  pim_matrix_des input_descript;
  input_descript.type = (int32_t)in_q->type;
  input_descript.layerid = 0; // TODO: set this value from 0 to NR_LAYER - 1 as you like
  memcpy(input_descript.ne, in_q->ne, sizeof(in_q->ne));

  uint32_t input_offset = context->pim_metadata.input_offset;
  // broadcast input metadata
  DPU_ASSERT(dpu_broadcast_to(context->dpu_set, DPU_MRAM_HEAP_POINTER_NAME, input_offset, &input_descript, sizeof(pim_matrix_des), DPU_XFER_DEFAULT));
  input_offset += sizeof(pim_matrix_des);

  // broadcast input data
  uint32_t bclen = ggml_row_size(in_q->type, in_q->ne[0])*in_q->ne[1]*in_q->ne[2]*in_q->ne[3];
  DPU_ASSERT(dpu_broadcast_to(context->dpu_set, DPU_MRAM_HEAP_POINTER_NAME, input_offset, in_q->data, bclen, DPU_XFER_DEFAULT));
  input_offset += bclen;

  std::chrono::high_resolution_clock::time_point ex_tp1 = std::chrono::high_resolution_clock::now();
  // Launch DPU kernel
  DPU_ASSERT(dpu_launch(context->dpu_set, DPU_SYNCHRONOUS));
  std::chrono::high_resolution_clock::time_point ex_tp2 = std::chrono::high_resolution_clock::now();

  std::chrono::duration<size_t, std::nano> dur = ex_tp2 - ex_tp1;

  std::cout << "执行用时：" << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count() << " ms" << std::endl;

  // Check results
  float *mul_mat_res = (float *)res->data;
  DPU_FOREACH(context->dpu_set, dpu, i) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, mul_mat_res + i * context->pim_metadata.rows_per_dpu*in_q->ne[1]));
  }
  DPU_ASSERT(dpu_push_xfer(context->dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_offset, context->pim_metadata.rows_per_dpu*in_q->ne[1]*sizeof(float), DPU_XFER_DEFAULT));

  return 0;
}

int main(int argc, char** argv) {
  // init fp table for fp16 dump
  fp_table_init();

  // WQ-PIM allocate dpu
  struct pim_context *pqcontext = (struct pim_context *)malloc(sizeof(struct pim_context));
  memset(pqcontext,0,sizeof(struct pim_context));
  DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &pqcontext->dpu_set));
  DPU_ASSERT(dpu_load(pqcontext->dpu_set, DPU_BINARY, NULL));

  const char* filenamea = "tensor-files/a.tensor";
  const char* filenameb = "tensor-files/b.tensor";
  const char* filenamebq = "tensor-files/b_quant.tensor";
  const char* filenamec = "tensor-files/c.tensor";
  const char* filenamec_p = "tensor-files/c_pim.tensor";
  struct ggml_tensor * ts_a = tensor_import(filenamea);
  struct ggml_tensor * ts_b = tensor_import(filenameb);
  struct ggml_tensor * ts_bq = tensor_import(filenamebq);
  struct ggml_tensor * ts_c = tensor_import(filenamec);
  struct ggml_tensor * ts_c_pim = tensor_import(filenamec_p);
  // std::cout<<"ts_a:"<<std::endl;
  // dump_tensor(ts_a, stdout);
  // std::cout<<"ts_b:"<<std::endl;
  // dump_tensor(ts_b, stdout);
  // std::cout<<"ts_bq:"<<std::endl;
  // dump_tensor(ts_bq, stdout);
  // std::cout<<"ts_c:"<<std::endl;
  // dump_tensor(ts_c, stdout);
  // std::cout<<"ts_c_pim:"<<std::endl;
  // dump_tensor(ts_c_pim, stdout);

  std::cout << "ts_a: " << std::endl;
  print_tensor(ts_a, stdout);
  std::cout << "ts_b: " << std::endl;
  print_tensor(ts_b, stdout);

  gemv_dpu_kernel(pqcontext, ts_a, ts_bq, ts_c_pim);
  // std::cout<<"ts_c_pim calculated by DPUs:"<<std::endl;
  // dump_tensor(ts_c_pim, stdout);

  float first_res = mul_add_q4_0_q8_0(ts_a, ts_bq);
  std::cout<<"first element: "<<std::fixed << std::setprecision(6)<<first_res<<std::endl;

  std::cout << "error between c and c_pim:" << std::endl;
  compare_tensor(ts_c, ts_c_pim);
  return 0;
}
