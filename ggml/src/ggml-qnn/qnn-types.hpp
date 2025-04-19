
#pragma once

#include "QnnCommon.h"
#include "QnnInterface.h"
#include "QnnTypes.h"
#include "Saver/QnnSaver.h"
#include "System/QnnSystemInterface.h"

namespace qnn {

enum qcom_htp_arch {
    NONE = 0,
    V68  = 68,
    V69  = 69,
    V73  = 73,
    V75  = 75,
    V79  = 79,  // SD 8 Gen 4 (SM8750)
};

enum qcom_chipset {
    UNKNOWN_SM = 0,
    SM8350     = 30,  // v68, SD 888/888+
    SM8450     = 36,  // v69, SD 8 Gen 1
    SA8295     = 39,  // v68
    SM8475     = 42,  // v69, SD 8+ Gen 1
    SM8550     = 43,  // v73, SD 8 Gen 2
    SSG2115P   = 46,  // v73
    SM7675     = 70,  // V73, SD 7+ Gen 3
    SM8635     = 68,  // v73, SD 8s Gen 3
    SM8650     = 57,  // v75, SD 8 Gen 3
    SM8750     = 69,  // v79, SD 8 Gen 4
};

struct qcom_socinfo {
    uint32_t soc_model;
    size_t   htp_arch;
    size_t   vtcm_size_in_mb;
};

using pfn_rpc_mem_init   = void (*)(void);
using pfn_rpc_mem_deinit = void (*)(void);
using pfn_rpc_mem_alloc  = void * (*) (int, uint32_t, int);
using pfn_rpc_mem_free   = void (*)(void *);
using pfn_rpc_mem_to_fd  = int (*)(void *);

using pfn_qnnsaver_initialize             = decltype(QnnSaver_initialize);
using pfn_qnninterface_getproviders       = decltype(QnnInterface_getProviders);
using pfn_qnnsysteminterface_getproviders = decltype(QnnSystemInterface_getProviders);
}  // namespace qnn

#define RPCMEM_DEFAULT_FLAGS  1
#define RPCMEM_HEAP_ID_SYSTEM 25

#define DISABLE_COPY(class_name)                 \
    class_name(const class_name &)     = delete; \
    void operator=(const class_name &) = delete

#define DISABLE_MOVE(class_name)            \
    class_name(class_name &&)     = delete; \
    void operator=(class_name &&) = delete
