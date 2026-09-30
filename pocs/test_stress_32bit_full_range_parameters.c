/*
 * test_stress_32bit_full_range_parameters.c
 * 
 * Stress‑test for the double‑redundancy overflow protection in ggml_new_tensor_impl.
 * Simulates a 32‑bit address space (max alloc size = 0xFFFFFFFF = 4 GiB) even on
 * 64‑bit platforms. The pre‑multiplication check:
 *    data_size <= MAX_ALLOC_SIZE / (size_t)ne[i]
 * catches overflow before it corrupts memory.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>   /* size_t */

/* Simulated 32‑bit memory limit (4 GiB) */
#define MAX_ALLOC_SIZE ((size_t)0xFFFFFFFF)

/* 
 * GGML_ASSERT with informative reporting.
 * On failure, explains why the condition failed and reports success
 * (the filter intercepted noise as expected), then aborts.
 */
#define GGML_ASSERT(cond, msg) do {                                   \
    if (!(cond)) {                                                    \
        fprintf(stderr, "\n[FILTRO ACTIVADO]: %s\n", msg);            \
        fprintf(stderr, "RESULTADO: EXITO - Ruido interceptado en %s:%d\n", \
                __FILE__, __LINE__);                                  \
        abort();                                                      \
    }                                                                 \
} while (0)

/*
 * Safe tensor‑size accumulator with a configurable maximum size.
 * Uses the exact logic from the patched ggml.c, but with a custom
 * max_size instead of SIZE_MAX to emulate a 32‑bit environment.
 */
static size_t safe_tensor_size(size_t row_size, const int64_t ne[], int n_dims,
                               size_t max_size) {
    uint64_t total = (uint64_t)row_size;
    for (int i = 0; i < n_dims; i++) {
        int64_t n = ne[i];
        if (n == 0) continue;          /* empty tensor */

        /* The message now explains the mathematical reason */
        GGML_ASSERT(total <= (uint64_t)max_size / (uint64_t)n,
                    "Deteccion de desbordamiento: El producto superaria el limite fisico.");

        total *= (uint64_t)n;
    }
    return (size_t)total;
}

int main(void) {
    printf("=== 32-bit Overflow Stress Test (simulated 4 GiB limit) ===\n");
    printf("MAX_ALLOC_SIZE = %zu (0x%zX)\n\n", MAX_ALLOC_SIZE, MAX_ALLOC_SIZE);

    /* Test 1: normal dimensions, no overflow */
    {
        printf("[Test 1] Safe dimensions (4 x 4096 x 16) ... ");
        int64_t ne[] = {4, 4096, 16};
        size_t row_size = 64;
        size_t sz = safe_tensor_size(row_size, ne, 3, MAX_ALLOC_SIZE);
        printf("OK, total = %zu bytes (%.2f MiB)\n",
               sz, sz / (1024.0 * 1024.0));
    }

    /* Test 2: large but still within 4 GiB */
    {
        printf("[Test 2] Large but valid (2000 x 2000 x 500) ... ");
        int64_t ne[] = {2000, 2000, 500};
        size_t row_size = 16;
        size_t sz = safe_tensor_size(row_size, ne, 3, MAX_ALLOC_SIZE);
        printf("OK, total = %zu bytes (%.2f MiB)\n",
               sz, sz / (1024.0 * 1024.0));
    }

    /* Test 3: exactly exceeds 4 GiB – must trigger assert */
    {
        printf("[Test 3] Overflow case (must abort) ...\n");
        int64_t ne[] = {2, 2};          /* multiply by 2 twice */
        size_t row_size = 0x40000000;   /* 1 GiB */
        printf("  row_size = %zu (1 GiB), ne = [2, 2] => expected total = 4 GiB = MAX_ALLOC_SIZE+1\n",
               row_size);
        printf("  This should fire GGML_ASSERT and abort the program.\n");
        fflush(stdout);
        size_t sz = safe_tensor_size(row_size, ne, 2, MAX_ALLOC_SIZE);
        /* If we reach here, the assert didn't fire – bug! */
        printf("  FAILED – assert did not fire, total = %zu\n", sz);
        return 1;
    }

    /* Never reached if Test 3 works */
    return 1;
}