
// function name template
// each function that will gather perform will be named test_<ch>api
#define GEN_PERF_FUNC_NAME_(ch) test_ ## ch ## api
#define GEN_PERF_FUNC_NAME(ch) GEN_PERF_FUNC_NAME_(ch)

/*
    Macro template for getting the best GEMM kernel runtime out of `num_runs`
    for matrices of size (m x n x k).
*/
#define GET_PERF_API_TEMP(ch, kernel, input_t, output_t) \
double GEN_PERF_FUNC_NAME(ch) ( \
    int num_runs, \
    int m, \
    int n, \
    int k \
) \
{ \
    input_t *A,*B; \
    output_t *C; \
    output_t alpha,beta; \
\
    A = (input_t*) malloc(m*k*sizeof(input_t)); \
    B = (input_t*) malloc(n*k*sizeof(input_t)); \
    C = (output_t*) malloc(m*n*sizeof(output_t)); \
     \
    alpha = 1; \
    beta = 1; \
 \
    double best = 1e9; \
 \
    for (int irep=0; irep<num_runs; irep++) \
    { \
        double start = bli_clock(); \
        kernel( \
            BLIS_NO_TRANSPOSE, \
            BLIS_NO_TRANSPOSE, \
            m, \
            n, \
            k, \
            &alpha, \
            A, k, 1, \
            B, n, 1, \
            &beta, \
            C, n, 1 \
        ); \
        double end = bli_clock(); \
 \
        best = bli_min(best, end-start); \
    } \
 \
    free(A); \
    free(B); \
    free(C); \
 \
    return best; \
} \

