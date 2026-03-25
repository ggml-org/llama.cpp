#include <stdio.h>
#include <math.h>
#include <string.h>

extern void quantize_row_turbo3_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo3_0(const void * x, float * y, long long k);
extern void quantize_row_turbo4_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo4_0(const void * x, float * y, long long k);

int main(void) {
    const int d = 128;
    char buf[256];
    float input[128], output[128];
    float mse, cosv, ni, no;

    printf("=== TurboQuant C Round-Trip Test ===\n\n");

    /* Test 1: basis vector */
    memset(input, 0, sizeof(input));
    input[0] = 1.0f;
    quantize_row_turbo3_0_ref(input, buf, d);
    dequantize_row_turbo3_0(buf, output, d);
    printf("Test 1 (turbo3): e0 = [1, 0, ...]\n");
    printf("  In:  [%.6f, %.6f, %.6f, %.6f]\n", input[0], input[1], input[2], input[3]);
    printf("  Out: [%.6f, %.6f, %.6f, %.6f]\n", output[0], output[1], output[2], output[3]);
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    printf("  MSE=%.8f Cosine=%.6f OutNorm=%.6f\n\n", mse/d, ni > 0 && no > 0 ? cosv/sqrtf(ni)/sqrtf(no) : 0, sqrtf(no));

    /* Test 2: large-norm vector */
    for (int i = 0; i < d; i++) input[i] = sinf(i*0.1f+0.5f) * 10.0f;
    quantize_row_turbo3_0_ref(input, buf, d);
    dequantize_row_turbo3_0(buf, output, d);
    printf("Test 2 (turbo3): sin*10\n");
    printf("  In:  [%.4f, %.4f, %.4f, %.4f]\n", input[0], input[1], input[2], input[3]);
    printf("  Out: [%.4f, %.4f, %.4f, %.4f]\n", output[0], output[1], output[2], output[3]);
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    printf("  MSE=%.8f Cosine=%.6f InNorm=%.2f OutNorm=%.2f\n\n", mse/d, cosv/sqrtf(ni)/sqrtf(no), sqrtf(ni), sqrtf(no));

    /* Test 3: turbo4 */
    for (int i = 0; i < d; i++) input[i] = cosf(i*0.2f) * 5.0f;
    quantize_row_turbo4_0_ref(input, buf, d);
    dequantize_row_turbo4_0(buf, output, d);
    printf("Test 3 (turbo4): cos*5\n");
    printf("  In:  [%.4f, %.4f, %.4f, %.4f]\n", input[0], input[1], input[2], input[3]);
    printf("  Out: [%.4f, %.4f, %.4f, %.4f]\n", output[0], output[1], output[2], output[3]);
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    printf("  MSE=%.8f Cosine=%.6f\n\n", mse/d, cosv/sqrtf(ni)/sqrtf(no));

    printf("=== Done ===\n");
    return 0;
}
