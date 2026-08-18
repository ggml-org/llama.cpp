#ifndef LLAMA_RCCL_TUNER_V6_H
#define LLAMA_RCCL_TUNER_V6_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
} ncclResult_t;

typedef enum {
    ncclFuncBroadcast = 0,
    ncclFuncReduce = 1,
    ncclFuncAllGather = 2,
    ncclFuncReduceScatter = 3,
    ncclFuncAllReduce = 4,
} ncclFunc_t;

typedef enum {
    NCCL_LOG_NONE = 0,
    NCCL_LOG_VERSION = 1,
    NCCL_LOG_WARN = 2,
    NCCL_LOG_INFO = 3,
    NCCL_LOG_ABORT = 4,
    NCCL_LOG_TRACE = 5,
} ncclDebugLogLevel_t;
typedef unsigned long long ncclDebugLogSubSys_t;
typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel_t, ncclDebugLogSubSys_t, const char *, int, const char *, ...);

typedef struct {
    int nNvlDomains;
    int minRanksPerNvlDomain;
    int maxRanksPerNvlDomain;
} ncclNvlDomainInfo_v5_t;

#define NCCL_NUM_ALGORITHMS_V5 7
#define NCCL_NUM_PROTOCOLS_V5 3
#define NCCL_NUM_HW_LINKS_V5 3
#define NCCL_NUM_COMPCAPS_V5 4
#define NCCL_NUM_TUNING_SCALES_V5 3

typedef struct {
    double baseLatencies[NCCL_NUM_ALGORITHMS_V5][NCCL_NUM_PROTOCOLS_V5];
    double hwLatencies[NCCL_NUM_HW_LINKS_V5][NCCL_NUM_ALGORITHMS_V5][NCCL_NUM_PROTOCOLS_V5];
    double llMaxBws[NCCL_NUM_COMPCAPS_V5][NCCL_NUM_TUNING_SCALES_V5];
    double perChMaxRingLL128Bws[NCCL_NUM_COMPCAPS_V5][NCCL_NUM_TUNING_SCALES_V5];
    double perChMaxTreeLL128Bws[NCCL_NUM_COMPCAPS_V5][NCCL_NUM_TUNING_SCALES_V5];
    double perChMaxTreeBws[NCCL_NUM_COMPCAPS_V5][NCCL_NUM_TUNING_SCALES_V5];
    double perChMaxNVLSTreeBws[NCCL_NUM_COMPCAPS_V5][NCCL_NUM_TUNING_SCALES_V5];
    double bwRatio[2][NCCL_NUM_ALGORITHMS_V5][NCCL_NUM_PROTOCOLS_V5];
} ncclTunerConstants_v6_t;

typedef struct {
    const char * name;
    ncclResult_t (*init)(void **, uint64_t, size_t, size_t, ncclDebugLogger_t, ncclNvlDomainInfo_v5_t *, ncclTunerConstants_v6_t *);
    ncclResult_t (*getCollInfo)(void *, ncclFunc_t, size_t, int, float **, int, int, int, int *);
    ncclResult_t (*finalize)(void *);
    ncclResult_t (*getChunkSize)(void *, ncclFunc_t, size_t, int, int, int, size_t *);
} ncclTuner_v6_t;

#define NCCL_ALGO_RING 1
#define NCCL_PROTO_LL 0
#define NCCL_ALGO_PROTO_IGNORE (-1.0f)
#define LLAMA_RCCL_TUNER_PLUGIN_SYMBOL "ncclTunerPlugin_v6"

#ifdef __cplusplus
}
#endif
#endif
