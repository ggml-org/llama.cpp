/*
 * CXL device probe - raw register access without context
 */
#include "cxl-device.h"
#include "cxl_gpu_cmd.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif

static inline uint32_t r32(volatile void *base, uint32_t off) {
    return *(volatile uint32_t *)((volatile uint8_t *)base + off);
}
static inline uint64_t r64(volatile void *base, uint32_t off) {
    uint32_t lo = r32(base, off);
    uint32_t hi = r32(base, off + 4);
    return (uint64_t)lo | ((uint64_t)hi << 32);
}
static inline void w32(volatile void *base, uint32_t off, uint32_t val) {
    *(volatile uint32_t *)((volatile uint8_t *)base + off) = val;
    __sync_synchronize();
}
// 64-bit write as two 32-bit stores
static inline void w64(volatile void *base, uint32_t off, uint64_t val) {
    w32(base, off, (uint32_t)(val & 0xFFFFFFFF));
    w32(base, off + 4, (uint32_t)(val >> 32));
}

// Execute command and watch status transitions
static int exec_cmd(volatile void *base, uint32_t cmd, const char *name) {
    printf("  Issuing %s (0x%02x)...\n", name, cmd);
    printf("    Before: CMD_STATUS=%u CMD_RESULT=%u\n",
           r32(base, CXL_GPU_REG_CMD_STATUS), r32(base, CXL_GPU_REG_CMD_RESULT));

    w32(base, CXL_GPU_REG_CMD, cmd);

    int timeout = 1000000;
    uint32_t status;
    int polls = 0;
    while (timeout-- > 0) {
        status = r32(base, CXL_GPU_REG_CMD_STATUS);
        polls++;
        if (polls <= 5) {
            printf("    Poll %d: status=%u\n", polls, status);
        }
        if (status == CXL_GPU_CMD_STATUS_COMPLETE || status == CXL_GPU_CMD_STATUS_ERROR) break;
    }

    uint32_t result = r32(base, CXL_GPU_REG_CMD_RESULT);
    printf("    After: CMD_STATUS=%u CMD_RESULT=%u (polls=%d)\n", status, result, polls);
    return (int)result;
}

int main() {
    // Find device
    struct cxl_device devices[CXL_MAX_DEVICES];
    int n = cxl_device_discover_all(devices, CXL_MAX_DEVICES);
    printf("Discovered %d devices\n", n);
    if (n == 0) return 1;

    const char *pci = devices[0].pci_addr;
    printf("Using device at %s\n", pci);

    // Enable
    char path[512];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/enable", pci);
    FILE *f = fopen(path, "w");
    if (f) { fprintf(f, "1"); fclose(f); }

    // Parse BAR2 size
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource", pci);
    size_t bar2_size = 0;
    f = fopen(path, "r");
    if (f) {
        uint64_t start, end, flags;
        for (int i = 0; i < 2; i++)
            if (fscanf(f, "0x%lx 0x%lx 0x%lx\n", &start, &end, &flags) != 3) break;
        if (fscanf(f, "0x%lx 0x%lx 0x%lx\n", &start, &end, &flags) == 3 && end > start)
            bar2_size = (size_t)(end - start + 1);
        fclose(f);
    }
    if (bar2_size == 0) bar2_size = CXL_GPU_CMD_REG_SIZE;
    printf("BAR2 size: %zu bytes (0x%zx)\n", bar2_size, bar2_size);

    // Map BAR2
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource2", pci);
    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) { perror("open BAR2"); return 1; }
    void *map = mmap(NULL, bar2_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }

    volatile void *base = map;

    // Dump identity registers
    printf("\n=== Identity Registers ===\n");
    printf("MAGIC:      0x%08x (expect 0x%08x)\n", r32(base, CXL_GPU_REG_MAGIC), CXL_GPU_MAGIC);
    printf("VERSION:    0x%08x\n", r32(base, CXL_GPU_REG_VERSION));
    printf("STATUS:     0x%08x (READY=%d, BUSY=%d, ERROR=%d, CTX=%d)\n",
           r32(base, CXL_GPU_REG_STATUS),
           (r32(base, CXL_GPU_REG_STATUS) >> 0) & 1,
           (r32(base, CXL_GPU_REG_STATUS) >> 1) & 1,
           (r32(base, CXL_GPU_REG_STATUS) >> 2) & 1,
           (r32(base, CXL_GPU_REG_STATUS) >> 3) & 1);
    printf("CAPS:       0x%08x\n", r32(base, CXL_GPU_REG_CAPS));

    // Memory info (read as 32-bit pairs)
    printf("\n=== Memory Info (32-bit reads) ===\n");
    printf("TOTAL_MEM lo: 0x%08x  hi: 0x%08x  => %zu MiB\n",
           r32(base, CXL_GPU_REG_TOTAL_MEM), r32(base, CXL_GPU_REG_TOTAL_MEM + 4),
           (size_t)(r64(base, CXL_GPU_REG_TOTAL_MEM) / (1024*1024)));
    printf("FREE_MEM  lo: 0x%08x  hi: 0x%08x  => %zu MiB\n",
           r32(base, CXL_GPU_REG_FREE_MEM), r32(base, CXL_GPU_REG_FREE_MEM + 4),
           (size_t)(r64(base, CXL_GPU_REG_FREE_MEM) / (1024*1024)));

    // Device name
    char dev_name[64] = {0};
    for (int i = 0; i < 64; i += 4) {
        uint32_t val = r32(base, CXL_GPU_REG_DEV_NAME + i);
        memcpy(dev_name + i, &val, 4);
    }
    dev_name[63] = '\0';
    printf("DEV_NAME:   \"%s\"\n", dev_name);

    // Command state
    printf("\n=== Command State ===\n");
    printf("CMD:        0x%08x\n", r32(base, CXL_GPU_REG_CMD));
    printf("CMD_STATUS: 0x%08x\n", r32(base, CXL_GPU_REG_CMD_STATUS));
    printf("CMD_RESULT: 0x%08x\n", r32(base, CXL_GPU_REG_CMD_RESULT));
    printf("RESULT0 lo: 0x%08x  hi: 0x%08x\n",
           r32(base, CXL_GPU_REG_RESULT0), r32(base, CXL_GPU_REG_RESULT0 + 4));

    // Test NOP
    printf("\n=== NOP Command ===\n");
    exec_cmd(base, CXL_GPU_CMD_NOP, "NOP");

    // Test CTX_CREATE
    printf("\n=== CTX_CREATE ===\n");
    int result = exec_cmd(base, CXL_GPU_CMD_CTX_CREATE, "CTX_CREATE");
    printf("  RESULT0 lo: 0x%08x hi: 0x%08x\n",
           r32(base, CXL_GPU_REG_RESULT0), r32(base, CXL_GPU_REG_RESULT0 + 4));

    if (result == 0) {
        printf("  CTX_CREATE succeeded!\n");

        // Test PARAM writes + MEM_ALLOC
        printf("\n=== MEM_ALLOC (4096 bytes) ===\n");
        w64(base, CXL_GPU_REG_PARAM0, (uint64_t)4096);
        printf("  PARAM0 written (split): lo=0x%08x hi=0x%08x\n",
               r32(base, CXL_GPU_REG_PARAM0), r32(base, CXL_GPU_REG_PARAM0 + 4));
        result = exec_cmd(base, CXL_GPU_CMD_MEM_ALLOC, "MEM_ALLOC");
        printf("  RESULT0 lo: 0x%08x hi: 0x%08x\n",
               r32(base, CXL_GPU_REG_RESULT0), r32(base, CXL_GPU_REG_RESULT0 + 4));

        if (result == 0) {
            uint64_t ptr = r64(base, CXL_GPU_REG_RESULT0);
            printf("  Allocated at: 0x%016lx\n", (unsigned long)ptr);

            // Test HtoD
            printf("\n=== MEM_COPY_HTOD ===\n");
            volatile uint8_t *data = (volatile uint8_t *)base + CXL_GPU_DATA_OFFSET;
            uint8_t test_pattern[16] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
                                         0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00};
            memcpy((void *)data, test_pattern, 16);
            w64(base, CXL_GPU_REG_PARAM0, ptr);
            w64(base, CXL_GPU_REG_PARAM1, 16);
            result = exec_cmd(base, CXL_GPU_CMD_MEM_COPY_HTOD, "HTOD");

            // Test DtoH
            printf("\n=== MEM_COPY_DTOH ===\n");
            memset((void *)data, 0, 16);
            w64(base, CXL_GPU_REG_PARAM0, ptr);
            w64(base, CXL_GPU_REG_PARAM1, 16);
            result = exec_cmd(base, CXL_GPU_CMD_MEM_COPY_DTOH, "DTOH");
            printf("  Data readback:");
            for (int i = 0; i < 16; i++) printf(" %02x", data[i]);
            printf("\n  Expected:     ");
            for (int i = 0; i < 16; i++) printf(" %02x", test_pattern[i]);
            printf("\n");

            // Free
            printf("\n=== MEM_FREE ===\n");
            w64(base, CXL_GPU_REG_PARAM0, ptr);
            exec_cmd(base, CXL_GPU_CMD_MEM_FREE, "MEM_FREE");
        }

        // Destroy context
        printf("\n=== CTX_DESTROY ===\n");
        exec_cmd(base, CXL_GPU_CMD_CTX_DESTROY, "CTX_DESTROY");
    } else {
        printf("  CTX_CREATE FAILED (result=%d)\n", result);
        printf("  This usually means the QEMU hetGPU backend is not running\n");
        printf("  or the host GPU is not available.\n");

        // Try INIT command first?
        printf("\n=== Trying INIT command first ===\n");
        exec_cmd(base, CXL_GPU_CMD_INIT, "INIT");
        printf("\n  Retrying CTX_CREATE...\n");
        result = exec_cmd(base, CXL_GPU_CMD_CTX_CREATE, "CTX_CREATE");
        if (result == 0) {
            printf("  CTX_CREATE succeeded after INIT!\n");
        } else {
            printf("  Still failed (result=%d)\n", result);
        }
    }

    munmap(map, bar2_size);
    close(fd);
    return 0;
}
