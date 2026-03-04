#pragma once

// CXL Type 2 Device Communication Layer
// Handles PCI device discovery, BAR MMIO mapping, and command execution
// for CXL Type 2 devices that proxy to host GPUs.
// Compatible with the QEMU hetGPU device model.

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CXL_MAX_DEVICES 8
#define CXL_DEVICE_NAME_MAX 128

struct cxl_device {
    int index;                          // device index (0, 1, 2, ...)
    char name[CXL_DEVICE_NAME_MAX];     // device name (e.g. "CXL0")
    char pci_addr[64];                  // PCI address (e.g. "0000:01:00.0")

    // BAR2: registers + 1MB data buffer
    volatile uint32_t * regs;           // register base pointer
    volatile uint8_t  * data;           // data region (regs + DATA_OFFSET)
    size_t              bar2_size;
    int                 bar2_fd;

    // BAR4: bulk/coherent region
    volatile uint8_t  * bar4;
    size_t              bar4_size;
    int                 bar4_fd;

    // Device capabilities
    uint32_t caps;

    // Device memory info (cached)
    size_t total_memory;
    size_t free_memory;

    // Context active flag
    bool ctx_active;
};

// Discover all CXL Type 2 devices on the PCI bus
// Returns the number of devices found, fills devices array
int cxl_device_discover_all(struct cxl_device devices[], int max_devices);

// Map device BARs, verify magic, create context
// Returns 0 on success, -1 on failure
int cxl_device_map(struct cxl_device * dev);

// Unmap device BARs and destroy context
void cxl_device_unmap(struct cxl_device * dev);

// Allocate memory on the device
// Returns device pointer, or 0 on failure
uint64_t cxl_device_alloc(struct cxl_device * dev, size_t size);

// Free device memory
void cxl_device_free(struct cxl_device * dev, uint64_t ptr);

// Host-to-device transfer (chunked via data buffer for large transfers)
int cxl_device_htod(struct cxl_device * dev, uint64_t dst, const void * src, size_t size);

// Device-to-host transfer (chunked via data buffer for large transfers)
int cxl_device_dtoh(struct cxl_device * dev, void * dst, uint64_t src, size_t size);

// Device memory set
int cxl_device_memset(struct cxl_device * dev, uint64_t ptr, uint8_t value, size_t size);

// Query device memory
void cxl_device_get_memory(struct cxl_device * dev, size_t * free_mem, size_t * total_mem);

// Get device name
const char * cxl_device_get_name(struct cxl_device * dev);

// Execute a serialized GGML graph on the device
// Returns ggml_status
int cxl_device_graph_compute(struct cxl_device * dev, const void * graph_data, size_t graph_size);

#ifdef __cplusplus
}
#endif
