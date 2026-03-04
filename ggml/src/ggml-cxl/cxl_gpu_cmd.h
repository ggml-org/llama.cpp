#pragma once

// CXL Type 2 Device GPU Command Protocol
// Compatible with the QEMU hetGPU device model and guest_libcuda shim.
// Register layout and command interface for CXL Type 2 devices
// that proxy to host GPUs via PCI BAR MMIO.

#include <stdint.h>

// PCI device identification
#define CXL_GPU_PCI_VENDOR_ID   0x8086
#define CXL_GPU_PCI_DEVICE_ID   0x0d92

// GPU Command Register Offsets (from BAR2 base)
#define CXL_GPU_REG_MAGIC           0x0000  // Magic number: 0x43584C32 "CXL2"
#define CXL_GPU_REG_VERSION         0x0004  // Interface version
#define CXL_GPU_REG_STATUS          0x0008  // Device status
#define CXL_GPU_REG_CAPS            0x000C  // Device capabilities

#define CXL_GPU_REG_CMD             0x0010  // Command register
#define CXL_GPU_REG_CMD_STATUS      0x0014  // Command status
#define CXL_GPU_REG_CMD_RESULT      0x0018  // Command result/error code
#define CXL_GPU_REG_CMD_DATA_LO     0x001C  // Command data low 32 bits
#define CXL_GPU_REG_CMD_DATA_HI     0x0020  // Command data high 32 bits

// 64-bit parameter registers (8-byte aligned)
#define CXL_GPU_REG_PARAM0          0x0040
#define CXL_GPU_REG_PARAM1          0x0048
#define CXL_GPU_REG_PARAM2          0x0050
#define CXL_GPU_REG_PARAM3          0x0058
#define CXL_GPU_REG_PARAM4          0x0060
#define CXL_GPU_REG_PARAM5          0x0068
#define CXL_GPU_REG_PARAM6          0x0070
#define CXL_GPU_REG_PARAM7          0x0078

// 64-bit result registers (8-byte aligned)
#define CXL_GPU_REG_RESULT0         0x0080
#define CXL_GPU_REG_RESULT1         0x0088
#define CXL_GPU_REG_RESULT2         0x0090
#define CXL_GPU_REG_RESULT3         0x0098

// Device info registers
#define CXL_GPU_REG_DEV_NAME        0x0100  // Device name (64 bytes)
#define CXL_GPU_REG_TOTAL_MEM       0x0140  // Total memory (64-bit)
#define CXL_GPU_REG_FREE_MEM        0x0148  // Free memory (64-bit)
#define CXL_GPU_REG_CC_MAJOR        0x0150  // Compute capability major
#define CXL_GPU_REG_CC_MINOR        0x0154  // Compute capability minor
#define CXL_GPU_REG_MP_COUNT        0x0158  // Multiprocessor count
#define CXL_GPU_REG_MAX_THREADS     0x015C  // Max threads per block
#define CXL_GPU_REG_WARP_SIZE       0x0160  // Warp size
#define CXL_GPU_REG_BACKEND         0x0164  // Backend type

// P2P registers
#define CXL_GPU_REG_PEER_COUNT      0x0200
#define CXL_GPU_REG_PEER_BAR_BASE   0x0208
#define CXL_GPU_REG_PEER_BAR_SIZE   0x0210
#define CXL_GPU_REG_LOCAL_GPU_ID    0x0218
#define CXL_GPU_REG_COHERENT_BASE   0x0220
#define CXL_GPU_REG_COHERENT_SIZE   0x0228

// Data transfer region in BAR2
#define CXL_GPU_DATA_OFFSET         0x1000
#define CXL_GPU_DATA_SIZE           0x100000    // 1 MB

// BAR2 total region size (registers + data)
#define CXL_GPU_CMD_REG_SIZE        0x101000    // ~1MB + 4KB registers

// BAR4 coherent/bulk region
#define CXL_GPU_COHERENT_REGION_OFFSET  0x0
#define CXL_GPU_COHERENT_REGION_SIZE    0x10000000  // 256 MB
#define CXL_GPU_BULK_TRANSFER_SIZE      0x4000000   // 64 MB

// Magic and version
#define CXL_GPU_MAGIC               0x43584C32  // "CXL2"
#define CXL_GPU_VERSION             0x00010000  // v1.0.0

// Capability bits
#define CXL_GPU_CAP_BULK_TRANSFER   (1 << 0)
#define CXL_GPU_CAP_CACHE_COHERENT  (1 << 1)
#define CXL_GPU_CAP_DMA_ENGINE      (1 << 2)
#define CXL_GPU_CAP_PEER_P2P        (1 << 3)
#define CXL_GPU_CAP_COHERENT_P2P    (1 << 4)

// Device status bits
#define CXL_GPU_STATUS_READY        (1 << 0)
#define CXL_GPU_STATUS_BUSY         (1 << 1)
#define CXL_GPU_STATUS_ERROR        (1 << 2)
#define CXL_GPU_STATUS_CTX_ACTIVE   (1 << 3)

// Command status values
#define CXL_GPU_CMD_STATUS_IDLE     0
#define CXL_GPU_CMD_STATUS_PENDING  1
#define CXL_GPU_CMD_STATUS_RUNNING  2
#define CXL_GPU_CMD_STATUS_COMPLETE 3
#define CXL_GPU_CMD_STATUS_ERROR    4

// GPU Commands (matching QEMU hetGPU device model)
#define CXL_GPU_CMD_NOP             0x00
#define CXL_GPU_CMD_INIT            0x01
#define CXL_GPU_CMD_GET_DEVICE_COUNT 0x02
#define CXL_GPU_CMD_GET_DEVICE      0x03
#define CXL_GPU_CMD_GET_DEVICE_NAME 0x04
#define CXL_GPU_CMD_GET_DEVICE_PROPS 0x05
#define CXL_GPU_CMD_GET_TOTAL_MEM   0x06

#define CXL_GPU_CMD_CTX_CREATE      0x10
#define CXL_GPU_CMD_CTX_DESTROY     0x11
#define CXL_GPU_CMD_CTX_SYNC        0x12

#define CXL_GPU_CMD_MEM_ALLOC       0x20
#define CXL_GPU_CMD_MEM_FREE        0x21
#define CXL_GPU_CMD_MEM_COPY_HTOD   0x22
#define CXL_GPU_CMD_MEM_COPY_DTOH   0x23
#define CXL_GPU_CMD_MEM_COPY_DTOD   0x24
#define CXL_GPU_CMD_MEM_SET         0x25
#define CXL_GPU_CMD_MEM_GET_INFO    0x26

#define CXL_GPU_CMD_MODULE_LOAD_PTX 0x30
#define CXL_GPU_CMD_MODULE_UNLOAD   0x31
#define CXL_GPU_CMD_FUNC_GET        0x32

#define CXL_GPU_CMD_LAUNCH_KERNEL   0x40

// Bulk transfer commands
#define CXL_GPU_CMD_BULK_HTOD       0x70
#define CXL_GPU_CMD_BULK_DTOH       0x71
#define CXL_GPU_CMD_BULK_DTOD       0x72

// CXL.cache coherency commands
#define CXL_GPU_CMD_CACHE_FLUSH     0x80
#define CXL_GPU_CMD_CACHE_INVALIDATE 0x81
#define CXL_GPU_CMD_CACHE_WRITEBACK 0x82

// P2P commands
#define CXL_GPU_CMD_P2P_ENABLE      0x90
#define CXL_GPU_CMD_P2P_DISABLE     0x91
#define CXL_GPU_CMD_P2P_COPY        0x92
#define CXL_GPU_CMD_PEER_MAP_MEM    0x93
#define CXL_GPU_CMD_PEER_UNMAP_MEM  0x94

// GGML-specific commands (extension)
#define CXL_GPU_CMD_GGML_GRAPH_COMPUTE  0xA0
#define CXL_GPU_CMD_GGML_GRAPH_PLAN     0xA1
#define CXL_GPU_CMD_GGML_GET_CAPS       0xA2

// Error codes (matching CUDA)
#define CXL_GPU_SUCCESS                     0
#define CXL_GPU_ERROR_INVALID_VALUE         1
#define CXL_GPU_ERROR_OUT_OF_MEMORY         2
#define CXL_GPU_ERROR_NOT_INITIALIZED       3
