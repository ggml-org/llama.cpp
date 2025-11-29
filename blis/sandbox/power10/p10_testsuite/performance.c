/*

    This program is designed to gather the performance data of the POWER10
    GEMM kernels in `blis/sandbox/power10`.

    By default, the performance of the kernels is gather over a set of square
    matrices. The perfromance results are reported in GFLOPS, and outputted in
    CSV format.

*/

#include "performance.h"
#include "blis.h"
#include "../bli_sandbox.h"
#include "common.h"

#include <stdio.h>
// print kernel name
const char* get_kernel_name(int kernel_id)
{
    switch (kernel_id)
    {
        case FLOAT16 : return "bli_shgemm";
        case BFLOAT16: return "bli_sbgemm";
        case INT16   : return "bli_i16gemm";
        case INT8    : return "bli_i8gemm";
        case INT4    : return "bli_i4gemm";
        default: printf("INCORRECT KERNEL ID\n"); exit(-1);
    }
}

// create all the performance gathering functions for each kernel
GET_PERF_API_TEMP(sb, bli_sbgemm, bfloat16, float);
GET_PERF_API_TEMP(sh, bli_shgemm, float16, float);
GET_PERF_API_TEMP(i16, bli_i16gemm, int16_t, int);
GET_PERF_API_TEMP(i8, bli_i8gemm, int8_t, int);
GET_PERF_API_TEMP(i4, bli_i4gemm, nibbles, int);


// using the DATATYPE enum, gather the performance of the respective GEMM kernel
double run_kernel(int kernel_id, int nreps, int m, int n, int k)
{
    switch (kernel_id)
    {
        case FLOAT16 : return test_shapi(nreps, m, n, k);
        case BFLOAT16: return test_sbapi(nreps, m, n, k);
        case INT16   : return test_i16api(nreps, m, n, k);
        case INT8    : return test_i8api(nreps, m, n, k);
        case INT4    : return test_i4api(nreps, m, n, k);
        default: return -1.0;
    }
}

// print the performance data in CSV format
// performance is measured in terms of GFLOPs
void print_perf_data(int m, int n, int k, double best_time)
{
    double GFLOPS = (2.0 * m * n * k) / (1e9 * best_time);
    printf("%d, %d, %d, %.2f\n", m, n, k, GFLOPS);
}

// get performance data
void get_perf(int kernel_id, int nreps, int start, int end, int inc)
{
    // csv header
    printf("%s performance\n", get_kernel_name(kernel_id));
    printf("m, n, k, GFLOPS\n");

    int m,n,k;

    // run over all problem sizes
    for (int p=start; p<=end; p+=inc)
    {
        // change here to adjust problem size
        m = p,
        n = p,
        k = p;

        double best_run_time = run_kernel(kernel_id, nreps, m, n, k);

        print_perf_data(m, n, k, best_run_time);
    }
}

int main(int argc, char *argv[])
{
    // initialize a square problem set range
    int start = 80;
    int end = 4000;
    int inc = 80;
    
    // number of times the kernel will be run
    int nreps = 5;

    // run a respective kernel
    get_perf( FLOAT16, nreps, start, end, inc);
    get_perf(BFLOAT16, nreps, start, end, inc);
    get_perf(   INT16, nreps, start, end, inc);
    get_perf(    INT8, nreps, start, end, inc);
    get_perf(    INT4, nreps, start, end, inc);

    return 0;
}
