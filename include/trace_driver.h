#ifndef TRACE_DRIVER_H
#define TRACE_DRIVER_H
#include "ggml.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-cpu-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

  struct tensor_differ{
    float max_abs_diff;
    float diff_sum;
    float diff_abs_sum;
  };

  extern float ggml_table_f32_f16[1 << 16];

  void tensor_export(const struct ggml_tensor * tensor, const char * fname);
  struct ggml_tensor * tensor_import(const char * fname);
  void dump_tensor_first_n(const struct ggml_tensor * tensor, int n, FILE * fout);
  void compare_tensor(const struct ggml_tensor *a, const struct ggml_tensor *b);
  void print_q4_tensor(const struct ggml_tensor *a);
  void print_tensor(const struct ggml_tensor *tensor, FILE *fout);
  void dump_tensor(const struct ggml_tensor * tensor, FILE * fout);

  float mul_add_q4_0_q8_0(struct ggml_tensor * a, struct ggml_tensor * b);
  struct tensor_differ max_diff(struct ggml_tensor * a, struct ggml_tensor * b);

#ifdef __cplusplus
}
#endif

#endif
