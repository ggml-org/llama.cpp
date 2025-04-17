#ifndef _TIMER_H
#define _TIMER_H

#include <stdint.h>
#include <sys/time.h>

int64_t usec(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (((int64_t)tv.tv_sec) * 1000000) + tv.tv_usec;
}

#endif