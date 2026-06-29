#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define CLUSTER_SCHEDULING_AVAILABLE
int pthread_prefer_alternate_cluster_self(void);
int pthread_prefer_alternate_amx_self(void);

#define PTHREAD_MAX_PARALLELISM_PHYSICAL 0x1
#define PTHREAD_MAX_PARALLELISM_CLUSTER 0x2
/*
 * For now consider AMX as a per-cluster resource.
 * There's a PTHREAD_MAX_PARALLELISM_AMX but don't use it currently.
 */

int pthread_qos_max_parallelism(qos_class_t qos, unsigned long flags);

#ifdef __cplusplus
}
#endif

