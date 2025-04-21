#include "trace_driver.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <pthread.h>

extern "C" {
#include "../../PIM-tensorStore/host/pim_llm.h"
}


#define NR_DPUS 512
#define NR_LAYER 2
#define NR_THREADS 3
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

struct param
{
    struct dpu_set_t dpu_set;
    struct ggml_tensor *w;
    remote_ptr table_f32_f16_pim_ptr;
    remote_ptr w_pim_ptr;
    struct ggml_tensor * in_q;
    struct ggml_tensor * res;
};

int gemv_load_weight(struct dpu_set_t dpu_set, struct ggml_tensor *w, remote_ptr* table_f32_f16_pim_ptr, remote_ptr* w_pim_ptr){
    DPU_ASSERT(dpu_broadcast_to(dpu_set, "mul_table_int4_int8", 0, (void *)(mul_table_int4_int8), sizeof(mul_table_int4_int8), DPU_XFER_DEFAULT));
    //ggml_table_f32_f16 tbl is transferred to pim
  
    all_dpu_mm_reset();
    *table_f32_f16_pim_ptr = all_dpu_alloc(sizeof(ggml_table_f32_f16));
    assert((*table_f32_f16_pim_ptr).dpu_id == ALL_DPU && (*table_f32_f16_pim_ptr).dpu_addr == FREE_STORAGE_OFFSET);
    dpu_broadcast_direct(dpu_set, *table_f32_f16_pim_ptr, (void *)(ggml_table_f32_f16), sizeof(ggml_table_f32_f16));
    // DPU_ASSERT(dpu_broadcast_to(dpu_set, "table_f32_f16", 0, (void *)(ggml_table_f32_f16), sizeof(ggml_table_f32_f16), DPU_XFER_DEFAULT));
    std::cout << "ggml_table_f32_f16 len = " << sizeof(ggml_table_f32_f16) << std::endl;
  
    assert(w->ne[1] % NR_DPUS == 0);
  
    *w_pim_ptr = all_dpu_alloc(w->nb[1] * (w->ne[1] / NR_DPUS));
    assert((*w_pim_ptr).dpu_id == ALL_DPU && (*w_pim_ptr).dpu_addr == FREE_STORAGE_OFFSET + sizeof(ggml_table_f32_f16));
  
    void *src_w_ptrs[NR_DPUS];
    for (int i = 0; i < NR_DPUS; i++)
    {
      src_w_ptrs[i] = (void *)((unsigned char *)w->data + i * w->nb[1] * (w->ne[1] / NR_DPUS));
    }
  
    dpu_send_direct(dpu_set, *w_pim_ptr, src_w_ptrs, w->nb[1] * (w->ne[1] / NR_DPUS));
    return 0;
}

void* gemv_dpu_kernel(void *arg) {
  std::chrono::high_resolution_clock::time_point ex_tp1;
  std::chrono::high_resolution_clock::time_point ex_tp2;
  std::chrono::duration<size_t, std::nano> dur;
  struct param *pa = (struct param *)arg;
  struct dpu_set_t dpu_set = pa->dpu_set;
  struct ggml_tensor *w = pa->w;
  remote_ptr table_f32_f16_pim_ptr = pa->table_f32_f16_pim_ptr;
  remote_ptr w_pim_ptr = pa->w_pim_ptr;
  struct ggml_tensor * in_q = pa->in_q;
  struct ggml_tensor * res = pa->res;

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
  return NULL;
}
#endif

int main(int argc, char** argv) {
  // init fp table for fp16 dump
  fp_table_init();
  mul_table_int4_int8_init();

#ifdef PIM_KERNEL
  // WQ-PIM allocate dpu
  param pas[NR_THREADS];
  for(int i=0;i<NR_THREADS;i++){
    struct dpu_set_t& dpu_set = pas[i].dpu_set;
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
  }


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

  for(int i=0;i<NR_THREADS;i++){
    pas[i].w = ts_a;
    pas[i].in_q = ts_bq;
    pas[i].res = ts_c_pim;
  }

  for(int i=0;i<NR_THREADS;i++){
    struct dpu_set_t& dpu_set = pas[i].dpu_set;
    remote_ptr table_f32_f16_pim_ptr;
    remote_ptr w_pim_ptr;
    gemv_load_weight(dpu_set, ts_a, &table_f32_f16_pim_ptr, &w_pim_ptr);
    pas[i].table_f32_f16_pim_ptr = table_f32_f16_pim_ptr;
    pas[i].w_pim_ptr = w_pim_ptr;
  }


  uint64_t start = usec();
  for(int i=0;i<NR_THREADS;i++){
    gemv_dpu_kernel(&(pas[i]));
  }
  uint64_t end = usec();
  std::cout<<"single thread sum time: "<<end - start << " us"<<std::endl;

  start = usec();
  pthread_t pid[NR_THREADS];
  for (int i = 0; i < NR_THREADS; i++)
  {
      pthread_create(&(pid[i]), NULL, gemv_dpu_kernel, &(pas[i]));
  }
  for (int i = 0; i < NR_THREADS; i++)
  {
      pthread_join(pid[i], NULL);
  }
  end = usec();
  std::cout<<"multi thread sum time: "<<end - start << " us"<<std::endl;

  float first_res = mul_add_q4_0_q8_0(ts_a, ts_bq);
  std::cout<<"first element: "<<std::fixed << std::setprecision(6)<<first_res<<std::endl;

  std::cout << "error between c and c_pim:" << std::endl;
  compare_tensor(ts_c, ts_c_pim);

#endif
  return 0;
}
