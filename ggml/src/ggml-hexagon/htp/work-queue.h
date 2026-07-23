#ifndef HTP_WORK_QUEUE_H
#define HTP_WORK_QUEUE_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

typedef void (*work_queue_func_t)(unsigned int n, unsigned int i, void *);

struct work_queue_s;
typedef struct work_queue_s * work_queue_t;

#define WORK_QUEUE_MAX_N_THREADS      10

size_t       work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t stack_size);
size_t       work_queue_alignof(void);
work_queue_t work_queue_init(void * ptr, uint32_t n_threads, uint32_t capacity, uint32_t stack_size);
void         work_queue_free(work_queue_t q);

void work_queue_wakeup(work_queue_t q);
void work_queue_suspend(work_queue_t q);

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, void * data, unsigned int n);

static inline bool work_queue_run(work_queue_t q, work_queue_func_t func, void * data, unsigned int n) {
    if (n <= 1) {
        func(n, 0, data);
        return true;
    }
    return work_queue_run_async(q, func, data, n);
}

// Legacy compatibility
typedef work_queue_func_t     worker_callback_t;
#define worker_pool_run_func  work_queue_run
#define worker_pool           work_queue

#endif  // #ifndef HTP_WORK_QUEUE_H
