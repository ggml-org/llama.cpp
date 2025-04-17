#include "g_vars.h"

__host msg_buffer_header g_msg_header;
__host char *g_msg_cache_ptr = NULL;
__host uint32_t *g_blocks_offset_ptr = NULL;