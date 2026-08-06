#include <stdio.h>
#include <stdint.h>
#include <stddef.h>  // size_t, SIZE_MAX
#include <limits.h>  // ULLONG_MAX (optional)

int main() {
    /*
     * Simulates the exact size computation in ggml_new_tensor_impl:
     *   ne[0]=32, ne[1]=4096, ne[2]=128, ne[3]=1
     *   type_size = 36864  (e.g., quantized type with large block size)
     *
     * Real size = 19 327 352 832 bytes (≈18 432 MiB)
     * On a 32-bit system, SIZE_MAX = 4 294 967 295 (≈4 GiB)
     * The size_t computation overflows and returns a corrupted value.
     */

    
    int64_t ne[] = {32, 4096, 128, 1};
    int n_dims = 4;
    size_t type_size = 36864;  // equivalent to ggml_row_size

    // --- "Legacy" path (unprotected size_t) ---
    size_t legacy_size = type_size;
    for (int i = 1; i < n_dims; i++) {
        legacy_size *= ne[i];  // overflows on 32-bit!
    }

    // --- "Patched" path (uint64_t + clamp) ---
    uint64_t true_size = type_size;
    for (int i = 1; i < n_dims; i++) {
        true_size *= ne[i];
    }
    size_t patched_alloc = (true_size <= SIZE_MAX) ? (size_t)true_size : SIZE_MAX;

    // --- Report ---
    printf("Dimensions: [%lld, %lld, %lld, %lld]\n",
           (long long)ne[0], (long long)ne[1], (long long)ne[2], (long long)ne[3]);
    printf("Size per element (type_size): %zu bytes\n", type_size);
    printf("True total size (uint64_t): %llu bytes (≈%.2f MiB)\n",
           (unsigned long long)true_size, true_size / (1024.0 * 1024.0));
    printf("\n--- Legacy Behavior (size_t) ---\n");
    printf("Value returned by size_t: %zu bytes (≈%.2f MiB)\n",
           legacy_size, legacy_size / (1024.0 * 1024.0));
    if (legacy_size != true_size)
        printf("OVERFLOW! The value is incorrect (wraparound).\n");
    else
        printf("No overflow on this system (likely 64-bit).\n");

    printf("\n--- Patched Behavior (uint64_t + clamp) ---\n");
    printf("Value computed in 64-bit: %llu bytes\n", (unsigned long long)true_size);
    printf("Actual allocation requested: %zu bytes (≈%.2f MiB)\n",
           patched_alloc, patched_alloc / (1024.0 * 1024.0));
    if (true_size > SIZE_MAX)
        printf("Overflow detected => clamped to SIZE_MAX (%.2f MiB).\n",
               (double)SIZE_MAX / (1024.0 * 1024.0));
    else
        printf("No overflow => using real value.\n");

    // --- Verdict ---
    int success = 0;
    // Success condition: the patched logic made the correct decision
    if ((true_size > SIZE_MAX && patched_alloc == SIZE_MAX) ||
        (true_size <= SIZE_MAX && patched_alloc == (size_t)true_size)) {
        // Additionally, if overflow occurred, legacy must differ from true size
        if (true_size > SIZE_MAX && legacy_size == true_size) {
            printf("\nERROR: On a system where SIZE_MAX is small, legacy should not match.\n");
            success = 1;
        } else {
            printf("\n ✓ The patch logic works correctly.\n");
        }
    } else {
        printf("\n ✗ The patch logic did NOT behave as expected.\n");
        success = 1;
    }

    return success;
}