/*

    This program is designed to test the correctness of the POWER10 GEMM 
    kernels in `blis/sandbox/power10`.

    By default, the correctness of the kernels is determined by measuring how 
    close the return value of the following function is to zero for square 
    matrix sizes.

    F(A, B, C_orig, C_ans, alpha, beta, t) =

        normf( (C_ans * t) - ((alpha * A * B + beta * C_orig) * t) )

    The function above can only be used to measure correctness if
    A, B, C_orig, and t have been randomized and normalized.

    The correctness is reported by printing the function return value along
    with the matrices' sizes.

*/


#include "blis.h"
#include "cast_funcs.h"
#include "correctness.h"
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

// normalize the vector using the forbenious norm
void normalize_vec(float *t, int n)
{
    // normalize t
    float norm_factor;
    bli_snormfv(n, t, 1, &norm_factor);
    // round up to closest power of 2
    norm_factor = 1 / (pow( 2.0, ceil( log2( norm_factor ) ) ));
    bli_sscalv(BLIS_NO_CONJUGATE, n, &norm_factor, t, 1);
}

	// Pre-conditions:
	// - a is randomized.
	// - b is randomized.
	// - c_orig is randomized.
	// Note:
	// - alpha and beta should have non-zero imaginary components in the
	//   complex cases in order to more fully exercise the implementation.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   C := beta * C_orig + alpha * transa(A) * transb(B)
	//
	// is functioning correctly if
	//
	//   normfv( v - z )
	//
	// is negligible, where
	//
	//   v = C * t
	//   z = ( beta * C_orig + alpha * transa(A) * transb(B) ) * t
	//     = beta * C_orig * t + alpha * transa(A) * transb(B) * t
	//     = beta * C_orig * t + alpha * transa(A) * w
	//     = beta * C_orig * t + z
float get_resid(
    int m, int n, int k,
    float *a, int rsa, int csa,
    float *b, int rsb, int csb,
    float *c, int rsc, int csc,
    float *c_orig,
    float *alpha, float *beta
)
{

    float t[n], v[m], w[k], z[m];
    float one = 1.0, zero = 0.0;

    bli_srandv(n, t, 1);

    // normalize so that the values are at the same precision of the input values
    normalize_vec(t, n);

    // v = C * t
    bli_sgemv(
        BLIS_NO_TRANSPOSE,
        BLIS_NO_CONJUGATE,
        m,
        n,
        &one,
        c, rsc, csc,
        t, 1,
        &zero,
        v, 1
    );

    // w = B * t
    bli_sgemv(
        BLIS_NO_TRANSPOSE,
        BLIS_NO_CONJUGATE,
        k,
        n,
        &one,
        b, rsb, csb,
        t, 1,
        &zero,
        w, 1
    );

    // z = alpha * A * w
    bli_sgemv(
        BLIS_NO_TRANSPOSE,
        BLIS_NO_CONJUGATE,
        m,
        k,
        alpha,
        a, rsa, csa,
        w, 1,
        &zero,
        z, 1
    );

    // z += beta * C_orig * t
    bli_sgemv(
        BLIS_NO_TRANSPOSE,
        BLIS_NO_CONJUGATE,
        m,
        n,
        beta,
        c_orig, rsc, csc,
        t, 1,
        &one,
        z, 1
    );

    // v = v - z
    bli_ssubv ( 
        BLIS_NO_CONJUGATE,
        m,
        z, 1,
        v, 1
    );

    // norm = normfv(v)
    float norm;
    bli_snormfv (
        m,
        v, 1,
        &norm
    );

    return norm;
}


// test to see if the result from a BLIS GEMM kernel is correct for a given m x n x k mat-mul
// assumes the matrices are of type float
// assumes the matrices were randomized and normalized
void correctness_checker(
    int m, int n, int k,
    float *a, int rsa, int csa,
    float *b, int rsb, int csb,
    float *c_orig, int rsc, int csc,
    float *c_ans,
    float alpha, float beta
)
{   
    double start, end;

    start = bli_clock();
    float resid = get_resid (
            m, n, k,
            a, rsa, csa,
            b, rsb, csb,
            c_ans, rsc, csc,
            c_orig,
            &alpha, &beta
    );
    end = bli_clock();

    printf("%d, %d, %d, %8.4le\n", m,n,k, resid);
}


// create all the correctness checking functions for each kernel
GEN_FP_COR_KERNEL(sb, bli_sbgemm, bfloat16, cast_f32_to_bf16m, cast_bf16_to_f32m);
GEN_FP_COR_KERNEL(sh, bli_shgemm, float16, cast_f32_to_f16m, cast_f16_to_f32m);
GEN_I_COR_KERNEL(i16, bli_i16gemm, int16_t, cast_f32_to_i16m, cast_i16_to_f32m);
GEN_I_COR_KERNEL(i8, bli_i8gemm, int8_t, cast_f32_to_i8m, cast_i8_to_f32m);

// correctness template for int types
void i4correctness_kernel (int m, int n, int k) 
{ 
    if(n%2 != 0)
    {
        printf("int4 can't handle odd sizes in the data-order dimension");
        exit(-1);
    }

    int rsa = k, csa = 1, 
        rsb = n, csb = 1, 
        rsc = n, csc = 1; 
        
    nibbles *a, *b; 

    int32_t *c_ans, *c_orig, alpha, beta; 

    float *a_float, *b_float, 
          *c_ans_float, *c_orig_float; 

    /* buffers that will be passed into the kernel */ 
    // int4 buffers only need half the space to store all the elements
    a = (nibbles *) malloc (m * (k/2) * sizeof(nibbles)); 
    b = (nibbles *) malloc (k * (n/2) * sizeof(nibbles)); 

    c_ans = (int32_t *) malloc (m * n * sizeof(int32_t)); 
    c_orig = (int32_t *) malloc (m * n * sizeof(int32_t)); 

    /* std format buffers that will be used by the correctness checker */ 
    a_float = (float *) malloc (m * k * sizeof(float)); 
    b_float = (float *) malloc (k * n * sizeof(float)); 
    c_ans_float  = (float *) malloc (m * n * sizeof(float)); 
    c_orig_float = (float *) malloc (m * n * sizeof(float)); 

    /* randomize matrices with float vals */ 
    bli_srandv(m*k, a_float, 1); 
    bli_srandv(k*n, b_float, 1); 
    bli_srandv(m*n, c_orig_float, 1); 

    /* normalize the matrices */ 
    normalize_vec(a_float, m*k); 
    normalize_vec(b_float, k*n); 
    normalize_vec(c_orig_float, m*n); 

    /* cast the float buffers into the buffers for the kernel */ 
    cast_f32_to_i4m (a_float, a, m*k); 
    cast_f32_to_i4m (b_float, b, k*n); 

    /* cast float buffers to support int values */ 
    cast_f32_to_i32m(c_orig_float, c_orig, m*n); 
    cast_i32_to_f32m(c_orig, c_orig_float, m*n); 

    /* cast the kernel buffers into the float buffers to ensure that the values match */ 
    cast_i4_to_f32m (a, a_float, m*k); 
    cast_i4_to_f32m (b, b_float, k*n); 

    /* init alpha and beta */ 
    alpha = 1; 
    beta  = 1; 

    /* run kernel to get result in c_ans */ 
    // strides need to be adjusted since 1 element stores 2 values
    memcpy(c_ans, c_orig, m * n * sizeof(int)); 
    bli_i4gemm( 
            BLIS_NO_TRANSPOSE, 
            BLIS_NO_TRANSPOSE, 
            m, 
            n, 
            k, 
            &alpha, 
            a, rsa/2, csa, 
            b, rsb/2, csb, 
            &beta, 
            c_ans, rsc, csc 
    ); 

    /* cast integer result into float buffer since float is our std format for correctness checking */ 
    cast_i32_to_f32m(c_ans, c_ans_float, m*n); 

    /* using the BLIS GEMM correctness check method, get the resid */ 
    correctness_checker( 
        m, n, k, 
        a_float, rsa, csa, 
        b_float, rsb, csb, 
        c_orig_float, rsc, csc, 
        c_ans_float, 
        (float) alpha, (float) beta 
    ); 

    free(a); 
    free(b); 
    free(c_ans); 
    free(c_orig); 
    free(a_float); 
    free(b_float); 
    free(c_ans_float); 
    free(c_orig_float); 
} 

// using the DATATYPE enum, gather test the correctness of the respective GEMM kernel
void run_correctness_kernel(int kernel_id, int m, int n, int k)
{
    switch (kernel_id)
    {
        case FLOAT16 : shcorrectness_kernel(m, n, k); break;
        case BFLOAT16: sbcorrectness_kernel(m, n, k); break;
        case INT16   : i16correctness_kernel(m, n, k); break;
        case INT8    : i8correctness_kernel(m, n, k); break;
        case INT4    : i4correctness_kernel(m, n, k); break;
        default: break;
    }
}

void test_correctness(int kernel_id, int start, int end, int inc)
{
    printf("%s correctness test\n", get_kernel_name(kernel_id));
    printf("m, n, k, resid\n");
    int m,n,k;
    for (int p=start; p<=end; p+=inc)
    {
        m=n=k=p;
        run_correctness_kernel(kernel_id, m, n, k);
    }
}

// correctness test for bfloat16 gemm
int main(int argc, char *argv[])
{
    
    test_correctness(FLOAT16, 80, 4000, 80);
    test_correctness(BFLOAT16, 80, 4000, 80);
    test_correctness(INT16, 80, 4000, 80);
    test_correctness(INT8, 80, 4000, 80);
    test_correctness(INT4, 80, 4000, 80);
}
