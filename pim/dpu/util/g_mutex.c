#include "g_mutex.h"
#include <mutex_pool.h>

MUTEX_POOL_INIT(mutex_pool, NR_TASKLETS);

void buckets_mutex_lock(uint32_t bucket_id)
{
    mutex_pool_lock(&mutex_pool, bucket_id);
}

void buckets_mutex_unlock(uint32_t bucket_id)
{
    mutex_pool_unlock(&mutex_pool, bucket_id);
}