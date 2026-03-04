#include "cxl-device.h"
#include "cxl_gpu_cmd.h"

// Suppress volatile cast warnings - intentional for MMIO register/buffer access
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Register access helpers (matching guest_libcuda patterns)
static inline uint32_t cxl_reg_read32(struct cxl_device * dev, uint32_t offset) {
    return *(volatile uint32_t *)((volatile uint8_t *)dev->regs + offset);
}

static inline uint64_t cxl_reg_read64(struct cxl_device * dev, uint32_t offset) {
    return *(volatile uint64_t *)((volatile uint8_t *)dev->regs + offset);
}

static inline void cxl_reg_write32(struct cxl_device * dev, uint32_t offset, uint32_t value) {
    *(volatile uint32_t *)((volatile uint8_t *)dev->regs + offset) = value;
    __sync_synchronize();
}

static inline void cxl_reg_write64(struct cxl_device * dev, uint32_t offset, uint64_t value) {
    *(volatile uint64_t *)((volatile uint8_t *)dev->regs + offset) = value;
    __sync_synchronize();
}

// MMIO-safe copy: uses 64-bit aligned volatile accesses.
// Standard memcpy may use AVX/SSE SIMD instructions that cause SIGILL on uncacheable PCI BAR MMIO.
// Both src and dst must be 8-byte aligned. len is rounded up to a multiple of 8.
// A single memory barrier is issued after the full transfer (not per-store).
static void mmio_copy64(void * dst, const volatile void * src, size_t len) {
    volatile uint64_t * s = (volatile uint64_t *)src;
    uint64_t * d = (uint64_t *)dst;
    size_t n = (len + 7) / 8;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
    __sync_synchronize();
}

static void mmio_write64(volatile void * dst, const void * src, size_t len) {
    volatile uint64_t * d = (volatile uint64_t *)dst;
    const uint64_t * s = (const uint64_t *)src;
    size_t n = (len + 7) / 8;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
    __sync_synchronize();
}

// Copy data to/from BAR2 data region (8-byte aligned MMIO-safe)
static inline void cxl_data_write(struct cxl_device * dev, size_t offset, const void * src, size_t len) {
    if (offset + len <= CXL_GPU_DATA_SIZE) {
        mmio_write64(dev->data + offset, src, len);
    }
}

static inline void cxl_data_read(struct cxl_device * dev, size_t offset, void * dst, size_t len) {
    if (offset + len <= CXL_GPU_DATA_SIZE) {
        mmio_copy64(dst, dev->data + offset, len);
    }
}

// Execute a command and wait for completion
// Returns the result code (0 = success)
static int cxl_execute_cmd(struct cxl_device * dev, uint32_t cmd) {
    cxl_reg_write32(dev, CXL_GPU_REG_CMD, cmd);

    int timeout = 10000000;
    while (timeout > 0) {
        uint32_t status = cxl_reg_read32(dev, CXL_GPU_REG_CMD_STATUS);
        if (status == CXL_GPU_CMD_STATUS_COMPLETE) {
            return (int)cxl_reg_read32(dev, CXL_GPU_REG_CMD_RESULT);
        }
        if (status == CXL_GPU_CMD_STATUS_ERROR) {
            return (int)cxl_reg_read32(dev, CXL_GPU_REG_CMD_RESULT);
        }
        timeout--;
    }

    fprintf(stderr, "cxl-device: command 0x%02x timeout on %s\n", cmd, dev->name);
    return -1;
}

// Read PCI config value from sysfs
static int read_sysfs_hex(const char * path, uint32_t * value) {
    FILE * f = fopen(path, "r");
    if (!f) return -1;
    int ret = fscanf(f, "%x", value);
    fclose(f);
    return (ret == 1) ? 0 : -1;
}

// Parse BAR sizes from PCI resource file
static void parse_bar_sizes(const char * pci_addr, size_t * bar2_size, size_t * bar4_size) {
    char path[512];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource", pci_addr);

    *bar2_size = 0;
    *bar4_size = 0;

    FILE * fp = fopen(path, "r");
    if (!fp) return;

    uint64_t start, end, flags;
    // Skip BAR0 and BAR1
    for (int i = 0; i < 2; i++) {
        if (fscanf(fp, "0x%lx 0x%lx 0x%lx\n", &start, &end, &flags) != 3) {
            fclose(fp);
            return;
        }
    }
    // Read BAR2
    if (fscanf(fp, "0x%lx 0x%lx 0x%lx\n", &start, &end, &flags) == 3) {
        if (end > start) {
            *bar2_size = (size_t)(end - start + 1);
        }
    }
    // Skip BAR3
    if (fscanf(fp, "0x%lx 0x%lx 0x%lx\n", &start, &end, &flags) != 3) {
        fclose(fp);
        return;
    }
    // Read BAR4
    if (fscanf(fp, "0x%lx 0x%lx 0x%lx\n", &start, &end, &flags) == 3) {
        if (end > start) {
            *bar4_size = (size_t)(end - start + 1);
        }
    }
    fclose(fp);
}

int cxl_device_discover_all(struct cxl_device devices[], int max_devices) {
    int count = 0;

    DIR * pci_dir = opendir("/sys/bus/pci/devices");
    if (!pci_dir) return 0;

    struct dirent * entry;
    while ((entry = readdir(pci_dir)) != NULL && count < max_devices) {
        if (entry->d_name[0] == '.') continue;

        char path[512];

        uint32_t vendor;
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/vendor", entry->d_name);
        if (read_sysfs_hex(path, &vendor) != 0 || vendor != CXL_GPU_PCI_VENDOR_ID) continue;

        uint32_t device_id;
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/device", entry->d_name);
        if (read_sysfs_hex(path, &device_id) != 0 || device_id != CXL_GPU_PCI_DEVICE_ID) continue;

        struct cxl_device * dev = &devices[count];
        memset(dev, 0, sizeof(*dev));
        dev->index = count;
        snprintf(dev->name, sizeof(dev->name), "CXL%d", count);
        snprintf(dev->pci_addr, sizeof(dev->pci_addr), "%s", entry->d_name);
        dev->bar2_fd = -1;
        dev->bar4_fd = -1;

        count++;
    }

    closedir(pci_dir);
    return count;
}

int cxl_device_map(struct cxl_device * dev) {
    char path[512];

    // Enable the PCI device
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/enable", dev->pci_addr);
    int enable_fd = open(path, O_WRONLY);
    if (enable_fd >= 0) {
        ssize_t nw = write(enable_fd, "1", 1);
        (void)nw;
        close(enable_fd);
    }

    // Parse BAR sizes from resource file
    size_t bar2_size = 0, bar4_size = 0;
    parse_bar_sizes(dev->pci_addr, &bar2_size, &bar4_size);
    if (bar2_size == 0) bar2_size = CXL_GPU_CMD_REG_SIZE;
    dev->bar2_size = bar2_size;

    // Map BAR2 (registers + data buffer)
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource2", dev->pci_addr);
    dev->bar2_fd = open(path, O_RDWR | O_SYNC);
    if (dev->bar2_fd < 0) {
        fprintf(stderr, "cxl-device: failed to open BAR2 for %s: %s\n", dev->name, strerror(errno));
        return -1;
    }

    void * map = mmap(NULL, bar2_size, PROT_READ | PROT_WRITE, MAP_SHARED, dev->bar2_fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "cxl-device: failed to mmap BAR2 for %s: %s\n", dev->name, strerror(errno));
        close(dev->bar2_fd);
        dev->bar2_fd = -1;
        return -1;
    }

    dev->regs = (volatile uint32_t *)map;
    dev->data = (volatile uint8_t *)map + CXL_GPU_DATA_OFFSET;

    // Verify magic
    uint32_t magic = cxl_reg_read32(dev, CXL_GPU_REG_MAGIC);
    if (magic != CXL_GPU_MAGIC) {
        fprintf(stderr, "cxl-device: bad magic 0x%08x for %s (expected 0x%08x)\n",
                magic, dev->name, CXL_GPU_MAGIC);
        cxl_device_unmap(dev);
        return -1;
    }

    // Read capabilities
    dev->caps = cxl_reg_read32(dev, CXL_GPU_REG_CAPS);

    // Check device is ready (non-fatal - device may still be usable for BAR access)
    uint32_t status = cxl_reg_read32(dev, CXL_GPU_REG_STATUS);
    if (!(status & CXL_GPU_STATUS_READY)) {
        fprintf(stderr, "cxl-device: warning: device %s not ready (status=0x%x) - continuing with BAR access\n",
                dev->name, status);
    }

    // Map BAR4 (bulk/coherent region) if available
    // BAR4 is opened without O_SYNC to get Write-Combining (WC) mapping
    // on prefetchable BARs, which yields significantly higher bulk bandwidth.
    if (bar4_size > 0) {
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource4", dev->pci_addr);
        dev->bar4_fd = open(path, O_RDWR);
        if (dev->bar4_fd >= 0) {
            void * bar4_map = mmap(NULL, bar4_size, PROT_READ | PROT_WRITE, MAP_SHARED, dev->bar4_fd, 0);
            if (bar4_map != MAP_FAILED) {
                dev->bar4 = (volatile uint8_t *)bar4_map;
                dev->bar4_size = bar4_size;
            } else {
                close(dev->bar4_fd);
                dev->bar4_fd = -1;
            }
        }
    }

    // Read device memory info
    dev->total_memory = (size_t)cxl_reg_read64(dev, CXL_GPU_REG_TOTAL_MEM);
    dev->free_memory  = (size_t)cxl_reg_read64(dev, CXL_GPU_REG_FREE_MEM);

    // Read device name from register space using 32-bit register reads (safe for MMIO)
    {
        uint32_t name_words[16]; // 64 bytes
        for (int i = 0; i < 16; i++) {
            name_words[i] = cxl_reg_read32(dev, CXL_GPU_REG_DEV_NAME + (uint32_t)(i * 4));
        }
        char dev_name[64];
        memcpy(dev_name, name_words, 64);
        dev_name[63] = '\0';
        if (dev_name[0] != '\0') {
            snprintf(dev->name, sizeof(dev->name), "CXL%d (%s)", dev->index, dev_name);
        }
    }

    // Create a context on the device (non-fatal if backend is not running)
    int result = cxl_execute_cmd(dev, CXL_GPU_CMD_CTX_CREATE);
    if (result != CXL_GPU_SUCCESS) {
        fprintf(stderr, "cxl-device: warning: CTX_CREATE failed on %s (error=%d) - backend may not be running\n",
                dev->name, result);
        // Continue anyway - device is mapped and basic register access works
    } else {
        dev->ctx_active = true;
    }

    return 0;
}

void cxl_device_unmap(struct cxl_device * dev) {
    // Destroy context if active
    if (dev->ctx_active && dev->regs) {
        cxl_execute_cmd(dev, CXL_GPU_CMD_CTX_DESTROY);
        dev->ctx_active = false;
    }

    if (dev->bar4) {
        munmap((void *)dev->bar4, dev->bar4_size);
        dev->bar4 = NULL;
    }
    if (dev->bar4_fd >= 0) {
        close(dev->bar4_fd);
        dev->bar4_fd = -1;
    }
    if (dev->regs) {
        munmap((void *)dev->regs, dev->bar2_size);
        dev->regs = NULL;
        dev->data = NULL;
    }
    if (dev->bar2_fd >= 0) {
        close(dev->bar2_fd);
        dev->bar2_fd = -1;
    }
}

uint64_t cxl_device_alloc(struct cxl_device * dev, size_t size) {
    cxl_reg_write64(dev, CXL_GPU_REG_PARAM0, (uint64_t)size);

    int result = cxl_execute_cmd(dev, CXL_GPU_CMD_MEM_ALLOC);
    if (result != CXL_GPU_SUCCESS) {
        return 0;
    }

    return cxl_reg_read64(dev, CXL_GPU_REG_RESULT0);
}

void cxl_device_free(struct cxl_device * dev, uint64_t ptr) {
    cxl_reg_write64(dev, CXL_GPU_REG_PARAM0, ptr);
    cxl_execute_cmd(dev, CXL_GPU_CMD_MEM_FREE);
}

int cxl_device_htod(struct cxl_device * dev, uint64_t dst, const void * src, size_t size) {
    if (size == 0) return 0;

    // Use BAR4 bulk path when available — WC-mapped, up to 64MB chunks
    // vs BAR2 data region which is UC-mapped and limited to 1MB chunks.
    const bool use_bulk = dev->bar4 && (dev->caps & CXL_GPU_CAP_BULK_TRANSFER);
    const size_t max_chunk = use_bulk ? CXL_GPU_BULK_TRANSFER_SIZE : CXL_GPU_DATA_SIZE;

    size_t offset = 0;
    while (offset < size) {
        size_t chunk = size - offset;
        if (chunk > max_chunk) {
            chunk = max_chunk;
        }

        if (use_bulk) {
            // Copy host data to BAR4 bulk region (WC mapping — fast 64-bit stores)
            mmio_write64(dev->bar4, (const uint8_t *)src + offset, chunk);

            // PARAM0 = device dest, PARAM1 = size
            cxl_reg_write64(dev, CXL_GPU_REG_PARAM0, dst + offset);
            cxl_reg_write64(dev, CXL_GPU_REG_PARAM1, (uint64_t)chunk);

            int result = cxl_execute_cmd(dev, CXL_GPU_CMD_BULK_HTOD);
            if (result != CXL_GPU_SUCCESS) {
                return -1;
            }
        } else {
            // Fallback: copy via BAR2 data region (UC mapping, 1MB max)
            cxl_data_write(dev, 0, (const uint8_t *)src + offset, chunk);

            cxl_reg_write64(dev, CXL_GPU_REG_PARAM0, dst + offset);
            cxl_reg_write64(dev, CXL_GPU_REG_PARAM1, (uint64_t)chunk);

            int result = cxl_execute_cmd(dev, CXL_GPU_CMD_MEM_COPY_HTOD);
            if (result != CXL_GPU_SUCCESS) {
                return -1;
            }
        }

        offset += chunk;
    }

    return 0;
}

int cxl_device_dtoh(struct cxl_device * dev, void * dst, uint64_t src, size_t size) {
    if (size == 0) return 0;

    const bool use_bulk = dev->bar4 && (dev->caps & CXL_GPU_CAP_BULK_TRANSFER);
    const size_t max_chunk = use_bulk ? CXL_GPU_BULK_TRANSFER_SIZE : CXL_GPU_DATA_SIZE;

    size_t offset = 0;
    while (offset < size) {
        size_t chunk = size - offset;
        if (chunk > max_chunk) {
            chunk = max_chunk;
        }

        if (use_bulk) {
            // PARAM0 = device src, PARAM1 = size
            cxl_reg_write64(dev, CXL_GPU_REG_PARAM0, src + offset);
            cxl_reg_write64(dev, CXL_GPU_REG_PARAM1, (uint64_t)chunk);

            int result = cxl_execute_cmd(dev, CXL_GPU_CMD_BULK_DTOH);
            if (result != CXL_GPU_SUCCESS) {
                return -1;
            }

            // Copy from BAR4 bulk region to host (WC mapping — fast 64-bit loads)
            mmio_copy64((uint8_t *)dst + offset, dev->bar4, chunk);
        } else {
            // Fallback: copy via BAR2 data region
            cxl_reg_write64(dev, CXL_GPU_REG_PARAM0, src + offset);
            cxl_reg_write64(dev, CXL_GPU_REG_PARAM1, (uint64_t)chunk);

            int result = cxl_execute_cmd(dev, CXL_GPU_CMD_MEM_COPY_DTOH);
            if (result != CXL_GPU_SUCCESS) {
                return -1;
            }

            cxl_data_read(dev, 0, (uint8_t *)dst + offset, chunk);
        }

        offset += chunk;
    }

    return 0;
}

int cxl_device_memset(struct cxl_device * dev, uint64_t ptr, uint8_t value, size_t size) {
    if (size == 0) return 0;

    const bool use_bulk = dev->bar4 && (dev->caps & CXL_GPU_CAP_BULK_TRANSFER);
    const size_t max_chunk = use_bulk ? CXL_GPU_BULK_TRANSFER_SIZE : CXL_GPU_DATA_SIZE;
    size_t chunk_size = (size < max_chunk) ? size : max_chunk;

    // Fill a host-side template buffer with the value
    std::vector<uint8_t> temp(chunk_size, value);

    size_t offset = 0;
    while (offset < size) {
        size_t to_copy = size - offset;
        if (to_copy > chunk_size) to_copy = chunk_size;

        if (use_bulk) {
            mmio_write64(dev->bar4, temp.data(), to_copy);
            cxl_reg_write64(dev, CXL_GPU_REG_PARAM0, ptr + offset);
            cxl_reg_write64(dev, CXL_GPU_REG_PARAM1, (uint64_t)to_copy);

            int result = cxl_execute_cmd(dev, CXL_GPU_CMD_BULK_HTOD);
            if (result != CXL_GPU_SUCCESS) {
                return -1;
            }
        } else {
            cxl_data_write(dev, 0, temp.data(), to_copy);
            cxl_reg_write64(dev, CXL_GPU_REG_PARAM0, ptr + offset);
            cxl_reg_write64(dev, CXL_GPU_REG_PARAM1, (uint64_t)to_copy);

            int result = cxl_execute_cmd(dev, CXL_GPU_CMD_MEM_COPY_HTOD);
            if (result != CXL_GPU_SUCCESS) {
                return -1;
            }
        }

        offset += to_copy;
    }

    return 0;
}

void cxl_device_get_memory(struct cxl_device * dev, size_t * free_mem, size_t * total_mem) {
    dev->total_memory = (size_t)cxl_reg_read64(dev, CXL_GPU_REG_TOTAL_MEM);
    dev->free_memory  = (size_t)cxl_reg_read64(dev, CXL_GPU_REG_FREE_MEM);

    if (free_mem)  *free_mem  = dev->free_memory;
    if (total_mem) *total_mem = dev->total_memory;
}

const char * cxl_device_get_name(struct cxl_device * dev) {
    return dev->name;
}

int cxl_device_graph_compute(struct cxl_device * dev, const void * graph_data, size_t graph_size) {
    // Determine which buffer to use
    if (graph_size > CXL_GPU_DATA_SIZE) {
        if (!dev->bar4 || graph_size > dev->bar4_size) {
            fprintf(stderr, "cxl-device: graph too large (%zu bytes) for %s\n", graph_size, dev->name);
            return -1;
        }
        // Copy to BAR4 bulk region (8-byte aligned MMIO-safe write)
        mmio_write64(dev->bar4, graph_data, graph_size);
        cxl_reg_write64(dev, CXL_GPU_REG_PARAM0, (uint64_t)graph_size);
        cxl_reg_write64(dev, CXL_GPU_REG_PARAM1, 1);  // use BAR4
    } else {
        // Copy to BAR2 data region
        cxl_data_write(dev, 0, graph_data, graph_size);
        cxl_reg_write64(dev, CXL_GPU_REG_PARAM0, (uint64_t)graph_size);
        cxl_reg_write64(dev, CXL_GPU_REG_PARAM1, 0);  // use BAR2
    }

    return cxl_execute_cmd(dev, CXL_GPU_CMD_GGML_GRAPH_COMPUTE);
}
