#ifndef _G_MUTEX_H
#define _G_MUTEX_H

#include <stdint.h>

void buckets_mutex_lock(uint32_t bucket_id);

void buckets_mutex_unlock(uint32_t bucket_id);

#endif