#ifndef HTP_WORK_QUEUE_H
#define HTP_WORK_QUEUE_H

#include <stdbool.h>
#include <stdint.h>

typedef void (*work_queue_callback_t)(unsigned int n, unsigned int i, void *);

struct work_queue_s;
typedef struct work_queue_s * work_queue_t;

#define WORK_QUEUE_MAX_N_THREADS      10
#define WORK_QUEUE_THREAD_STACK_SIZE  (2 * 16384)

#define WORK_QUEUE_SIZE               16
#define WORK_QUEUE_MASK               (WORK_QUEUE_SIZE - 1)

#define WORK_QUEUE_POLL_COUNT         2000

bool work_queue_init(work_queue_t * q, uint32_t n_threads);
void work_queue_release(work_queue_t * q);
bool work_queue_run_async(work_queue_t q, work_queue_callback_t func, void * data, unsigned int n);

static inline bool work_queue_run(work_queue_t q, work_queue_callback_t func, void * data, unsigned int n) {
    if (n <= 1) {
        func(n, 0, data);
        return true;
    }
    return work_queue_run_async(q, func, data, n);
}

// Legacy compatibility
typedef work_queue_callback_t worker_callback_t;
#define worker_pool_run_func  work_queue_run
#define worker_pool           work_queue

#endif  // #ifndef HTP_WORK_QUEUE_H
