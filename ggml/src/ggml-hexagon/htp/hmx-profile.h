// Conditional fine-grained profiling macros for HMX operations.
//
// Define ENABLE_PROFILE_TIMERS (via compiler flag or before including this
// header) to instrument sub-operation latencies with HAP qtimer.  When the
// macro is not defined the TIMER_* helpers expand to nothing so there is zero
// overhead.
//
// Usage:
//   TIMER_DEFINE(my_phase);          // declare accumulator variable
//   TIMER_START(my_phase);           // snapshot start time
//   ... work ...
//   TIMER_STOP(my_phase);            // accumulate elapsed ticks
//   FARF(ALWAYS, "my_phase: %lld us", TIMER_US(my_phase));

#ifndef HMX_PROFILE_H
#define HMX_PROFILE_H

#include <HAP_perf.h>

// #define ENABLE_PROFILE_TIMERS

#if defined(ENABLE_PROFILE_TIMERS)
#  define TIMER_DEFINE(name) int64_t name##_ticks = 0
#  define TIMER_START(name)  int64_t name##_t0 = HAP_perf_get_qtimer_count()
#  define TIMER_STOP(name)   name##_ticks += HAP_perf_get_qtimer_count() - name##_t0
#  define TIMER_US(name)     HAP_perf_qtimer_count_to_us(name##_ticks)
#else
#  define TIMER_DEFINE(name)
#  define TIMER_START(name)
#  define TIMER_STOP(name)
#  define TIMER_US(name)     0LL
#endif

#endif // HMX_PROFILE_H
