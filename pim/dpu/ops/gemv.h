#ifndef _GEMV_H
#define _GEMV_H

#include <mram.h>

void gemv_prepare();
void gemv_tasklets_run();
void gemv_merge();

#endif