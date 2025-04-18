#include "trace_driver.h"
#include <iostream>
#include <iomanip>
#include <chrono>

#define NR_DPUS 512
#define NR_LAYER 2
#define DPU_BINARY "./dpu/gemv_dpu"

int16_t mul_table_int4_int8[1<<4][1<<8];

void fp_table_init(void) {
  for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
            }
}

void mul_table_int4_int8_init(void) {
  for(int i = 0; i < (1 << 4); ++i){
    for(int j = 0; j< (1 << 8); ++j){
      mul_table_int4_int8[i][j] = (i - 8) * (j + INT8_MIN);
    }
  }
}

#ifdef PIM_KERNEL
int gemv_dpu_kernel(struct pim_context *context, struct ggml_tensor * w, struct ggml_tensor * in_q, struct ggml_tensor * res) {
  uint32_t pim_offset = 0;
  struct dpu_set_t dpu;

  std::chrono::high_resolution_clock::time_point ex_tp1 = std::chrono::high_resolution_clock::now();

  DPU_ASSERT(dpu_broadcast_to(context->dpu_set, "mul_table_int4_int8", 0, (void *)(mul_table_int4_int8), sizeof(mul_table_int4_int8), DPU_XFER_DEFAULT));
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

  // std::cout << "layer_num = " << NR_LAYER << ", weight_type = " << (uint16_t)(w->type) << ", rows_per_dpu = " << w->ne[1] / NR_DPUS << ", rest_rows = " << w->ne[1] % NR_DPUS << ", layer_len = " << context->pim_metadata.layer_len << ", input_offset = " << context->pim_metadata.input_offset << std::endl;

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

  std::chrono::high_resolution_clock::time_point ex_tp2 = std::chrono::high_resolution_clock::now();

  std::chrono::duration<size_t, std::nano> dur = ex_tp2 - ex_tp1;

  std::cout << "dpu: w传输用时：" << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count() << " ms" << std::endl;
  std::cout << "dpu: w传输用时：" << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() << " us" << std::endl;

  ex_tp1 = std::chrono::high_resolution_clock::now();

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

  ex_tp2 = std::chrono::high_resolution_clock::now();

  dur = ex_tp2 - ex_tp1;

  std::cout << "dpu: in_q传输用时：" << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count() << " ms" << std::endl;
  std::cout << "dpu: in_q传输用时：" << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() << " us" << std::endl;

  ex_tp1 = std::chrono::high_resolution_clock::now();
  // Launch DPU kernel
  DPU_ASSERT(dpu_launch(context->dpu_set, DPU_SYNCHRONOUS));
  ex_tp2 = std::chrono::high_resolution_clock::now();

  dur = ex_tp2 - ex_tp1;

  // std::cout << "执行用时：" << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count() << " ms" << std::endl;
  std::cout << "执行用时：" << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() << " us" << std::endl;

  // Check results
  float *mul_mat_res = (float *)res->data;
  ex_tp1 = std::chrono::high_resolution_clock::now();
  DPU_FOREACH(context->dpu_set, dpu, i) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, mul_mat_res + i * context->pim_metadata.rows_per_dpu*in_q->ne[1]));
  }
  DPU_ASSERT(dpu_push_xfer(context->dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_offset, context->pim_metadata.rows_per_dpu*in_q->ne[1]*sizeof(float), DPU_XFER_DEFAULT));
  ex_tp2 = std::chrono::high_resolution_clock::now();

  dur = ex_tp2 - ex_tp1;

  std::cout << "传回结果用时：" << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() << " us" << std::endl;
  return 0;
}
#endif


void gemv_cpu_kernel(struct pim_context *context, struct ggml_tensor * w, struct ggml_tensor * in_q, struct ggml_tensor * res_comp) {

  // 初始化上下文
  ggml_init_params params = {.mem_size = 256*1024*1024};
  ggml_context* ctx = ggml_init(params);

  // 创建tensor
  ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 4096, 4096);
  ggml_tensor* B = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 4096, 1);

  assert(A->ne[0] == w->ne[0] && A->ne[1] == w->ne[1] && A->ne[2] == w->ne[2] && A->ne[3] == w->ne[3]);
  assert(B->ne[0] == in_q->ne[0] && B->ne[1] == in_q->ne[1] && B->ne[2] == in_q->ne[2] && B->ne[3] == in_q->ne[3]);

  memcpy(A->data, w->data, ggml_nbytes(w));
  memcpy(B->data, in_q->data, ggml_nbytes(in_q));

  // 构建计算图
  ggml_tensor* C = ggml_mul_mat(ctx, A, B);
  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, C);

  std::chrono::high_resolution_clock::time_point ex_tp1 = std::chrono::high_resolution_clock::now();
  // 执行计算
  ggml_graph_compute_with_ctx(ctx, gf, 64); // 使用4线程
  std::chrono::high_resolution_clock::time_point ex_tp2 = std::chrono::high_resolution_clock::now();

  std::chrono::duration<size_t, std::nano> dur = ex_tp2 - ex_tp1;

  std::cout << "执行用时：" << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() << " us" << std::endl;
  std::cout << "执行用时：" << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count() << " ms" << std::endl;

  
  // 保存结果
  print_tensor(C, stdout);

  std::cout << "error between cpu and dpu before gemv:" << std::endl;
  compare_tensor(C, res_comp);
  
  // 释放资源
  ggml_free(ctx);
}

int main(int argc, char** argv) {
  // init fp table for fp16 dump
  fp_table_init();
  mul_table_int4_int8_init();

#ifdef PIM_KERNEL
  // WQ-PIM allocate dpu
  struct pim_context *pqcontext = (struct pim_context *)malloc(sizeof(struct pim_context));
  memset(pqcontext,0,sizeof(struct pim_context));
  DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &pqcontext->dpu_set));
  DPU_ASSERT(dpu_load(pqcontext->dpu_set, DPU_BINARY, NULL));

  const char* filenamea   = "tensor-files/a.tensor";
  const char* filenameb   = "tensor-files/b.tensor";
  const char* filenamebq  = "tensor-files/b_quant.tensor";
  const char* filenamec   = "tensor-files/c.tensor";
  const char* filenamec_p = "tensor-files/c_pim.tensor";
  struct ggml_tensor * ts_a     = tensor_import(filenamea);
  struct ggml_tensor * ts_b     = tensor_import(filenameb);
  struct ggml_tensor * ts_bq    = tensor_import(filenamebq);
  struct ggml_tensor * ts_c     = tensor_import(filenamec);
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
// #define IS_CONTIGUOUS(t, tn) {\
//   if(ggml_is_contiguous(t)) printf("%s is contiguous\n", tn);\
// }\

//   IS_CONTIGUOUS(ts_a, "ts_a");
//   IS_CONTIGUOUS(ts_b, "ts_b");
//   IS_CONTIGUOUS(ts_bq, "ts_bq");
//   IS_CONTIGUOUS(ts_c, "ts_c");
//   IS_CONTIGUOUS(ts_c_pim, "ts_ac_pim");
// #undef IS_CONTIGUOUS


  // dpu code 
  gemv_dpu_kernel(pqcontext, ts_a, ts_bq, ts_c_pim);
  std::cout << "error between c and c_pim:" << std::endl;
  compare_tensor(ts_c, ts_c_pim);

  // cpu code
  // gemv_cpu_kernel(pqcontext, ts_a, ts_bq, ts_c_pim);

  // float first_res = mul_add_q4_0_q8_0(ts_a, ts_bq);
  // std::cout<<"first element: "<<std::fixed << std::setprecision(6)<<first_res<<std::endl;

#endif
  return 0;
}
