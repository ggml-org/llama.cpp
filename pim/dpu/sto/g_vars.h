#ifndef _G_VARS_H
#define _G_VARS_H

#include <mram.h>
#include <stdlib.h>
#include "../../host/msg/msg_buffer.h"

extern __host msg_buffer_header g_msg_header;
extern __host char *g_msg_cache_ptr;
extern __host uint32_t *g_blocks_offset_ptr;

#endif