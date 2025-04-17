#ifndef _PIM_DIRECT_COMM_H
#define _PIM_DIRECT_COMM_H

#include <dpu.h>
#include "pim_mm.h"

void dpu_send_direct(struct dpu_set_t dpu_set, remote_ptr dst, void **src, uint32_t size_per_dpu);
void dpu_recv_direct(struct dpu_set_t dpu_set, remote_ptr src, void **dst, uint32_t size_per_dpu);

#endif