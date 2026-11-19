#if defined(RT_KV_BINDING)
layout (binding = RT_KV_BINDING) buffer RT_KV {
    uint rt_page_size;
    uint rt_pages;
    uint rt_generation[];
};

void rt_kv_write(uint row) {
    const uint page = row / rt_page_size;
    if (page < rt_pages) {
        atomicAdd(rt_generation[page], 1);
    }
}
#endif
