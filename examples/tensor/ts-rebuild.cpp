#include "trace_driver.h"
#include <iostream>
#include <iomanip>
#include <chrono>

extern "C" {
#include "../../PIM-tensorStore/host/pim_llm.h"
}


#define NR_DPUS 512
#define NR_LAYER 2
#define DPU_BINARY "./PIM-tensorStore/build/dpu_task"
#define PIM_KERNEL

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
int gemv_dpu_kernel(struct dpu_set_t dpu_set, struct ggml_tensor * w, struct ggml_tensor * in_q, struct ggml_tensor * res) {
  struct dpu_set_t dpu;

  std::chrono::high_resolution_clock::time_point ex_tp1 = std::chrono::high_resolution_clock::now();

  DPU_ASSERT(dpu_broadcast_to(dpu_set, "mul_table_int4_int8", 0, (void *)(mul_table_int4_int8), sizeof(mul_table_int4_int8), DPU_XFER_DEFAULT));
  //ggml_table_f32_f16 tbl is transferred to pim

  all_dpu_mm_reset();
  remote_ptr table_f32_f16_pim_ptr = all_dpu_alloc(sizeof(ggml_table_f32_f16));
  assert(table_f32_f16_pim_ptr.dpu_id == ALL_DPU && table_f32_f16_pim_ptr.dpu_addr == FREE_STORAGE_OFFSET);
  dpu_broadcast_direct(dpu_set, table_f32_f16_pim_ptr, (void *)(ggml_table_f32_f16), sizeof(ggml_table_f32_f16));
  // DPU_ASSERT(dpu_broadcast_to(dpu_set, "table_f32_f16", 0, (void *)(ggml_table_f32_f16), sizeof(ggml_table_f32_f16), DPU_XFER_DEFAULT));
  std::cout << "ggml_table_f32_f16 len = " << sizeof(ggml_table_f32_f16) << std::endl;

  assert(w->ne[1] % NR_DPUS == 0);

  remote_ptr w_pim_ptr = all_dpu_alloc(w->nb[1] * (w->ne[1] / NR_DPUS));
  assert(w_pim_ptr.dpu_id == ALL_DPU && w_pim_ptr.dpu_addr == FREE_STORAGE_OFFSET + sizeof(ggml_table_f32_f16));

  void *src_w_ptrs[NR_DPUS];
  for (int i = 0; i < NR_DPUS; i++)
  {
    src_w_ptrs[i] = (void *)((unsigned char *)w->data + i * w->nb[1] * (w->ne[1] / NR_DPUS));
  }

  dpu_send_direct(dpu_set, w_pim_ptr, src_w_ptrs, w->nb[1] * (w->ne[1] / NR_DPUS));

  std::chrono::high_resolution_clock::time_point ex_tp2 = std::chrono::high_resolution_clock::now();

  std::chrono::duration<size_t, std::nano> dur = ex_tp2 - ex_tp1;

  std::cout << "dpu: w传输用时：" << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count() << " ms" << std::endl;
  std::cout << "dpu: w传输用时：" << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() << " us" << std::endl;

  ex_tp1 = std::chrono::high_resolution_clock::now();

  msg_block_des msg_gemv;
  printf("%d\n", table_f32_f16_pim_ptr.dpu_addr);
  msg_block_builder_op_gemv_q4_q8(&msg_gemv, w_pim_ptr, w->ne[0], w->ne[1] / NR_DPUS, in_q->ne[0], in_q->data, in_q->nb[1], table_f32_f16_pim_ptr);

  msg_buffer buffer;
  msg_buffer_init(&buffer);
  msg_buffer_clear(&buffer);
  msg_buffer_append(&buffer, &msg_gemv);
  msg_buffer_finish(&buffer);
  // msg_buffer_dump_int32(&buffer);
  msg_buffer_send(&buffer, dpu_set);

  ex_tp2 = std::chrono::high_resolution_clock::now();

  dur = ex_tp2 - ex_tp1;

  std::cout << "dpu: in_q传输用时：" << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() << " us" << std::endl;

  ex_tp1 = std::chrono::high_resolution_clock::now();
  dpu_set_launch(dpu_set);
  ex_tp2 = std::chrono::high_resolution_clock::now();

  dur = ex_tp2 - ex_tp1;

  std::cout << "执行用时：" << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() << " us" << std::endl;

  // dpu_set_log_read(dpu_set);
  // Check results
  float *mul_mat_res = (float *)res->data;

  void *dst_ptrs[NR_DPUS];
  for (int i = 0; i < NR_DPUS; i++)
  {
      dst_ptrs[i] = (void *)(mul_mat_res + i * w->ne[1] / NR_DPUS);
  }

  ex_tp1 = std::chrono::high_resolution_clock::now();
  msg_buffer_recv(dpu_set, dst_ptrs, w->ne[1] / NR_DPUS * sizeof(float));
  ex_tp2 = std::chrono::high_resolution_clock::now();

  dur = ex_tp2 - ex_tp1;

  std::cout << "传回结果用时：" << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() << " us" << std::endl;
  return 0;
}
#endif


void gemv_cpu_kernel(struct ggml_tensor * w, struct ggml_tensor * in_q, struct ggml_tensor * res_comp) {

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
  struct dpu_set_t dpu_set;
  DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
  DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));

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

  std::cout << "ts_a: " << std::endl;
  print_tensor(ts_a, stdout);
  std::cout << "ts_b: " << std::endl;
  print_tensor(ts_b, stdout);

  gemv_dpu_kernel(dpu_set, ts_a, ts_bq, ts_c_pim);

  float first_res = mul_add_q4_0_q8_0(ts_a, ts_bq);
  std::cout<<"first element: "<<std::fixed << std::setprecision(6)<<first_res<<std::endl;

  std::cout << "error between c and c_pim:" << std::endl;
  compare_tensor(ts_c, ts_c_pim);

#endif
  return 0;
}
