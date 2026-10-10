[New LWP 849269 (id 2)]
[New LWP 849268 (id 3)]
[New LWP 849267 (id 4)]
[New LWP 849266 (id 5)]
[New LWP 849265 (id 6)]
[New LWP 849264 (id 7)]
[New LWP 849263 (id 8)]
[New LWP 849262 (id 9)]
[New LWP 849261 (id 10)]
[New LWP 849260 (id 11)]
[New LWP 849259 (id 12)]
[New LWP 847861 (id 13)]
[New LWP 847847 (id 14)]
Registering SYCL extensions for gdb
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/libthread_db.so.1".
[Switching to thread 1 (Thread 0x771653500e00 (LWP 847839))]
0x0000771655c0a8f2 in ?? () from /usr/lib/libc.so.6
#0  0x0000771655c0a8f2 in ?? () from /usr/lib/libc.so.6
#1  0x0000771655c6f4cb in wait4 () from /usr/lib/libc.so.6
#2  0x00007716561dbd8a in ggml_print_backtrace () from /usr/lib/libggml-base.so.0
#3  0x00007716561daeb9 in ggml_abort () from /usr/lib/libggml-base.so.0
#4  0x0000771656a4cf78 in ?? () from /usr/lib/libggml-sycl.so.0
#5  0x0000771656aaff2e in ggml_sycl_pool_vmm::alloc(unsigned long, unsigned long*) () from /usr/lib/libggml-sycl.so.0
#6  0x0000771656ab3b31 in ggml_sycl_op_mul_mat_sycl(ggml_backend_sycl_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, sycl::_V1::queue* const&) () from /usr/lib/libggml-sycl.so.0
#7  0x0000771656a7a7d5 in ?? () from /usr/lib/libggml-sycl.so.0
#8  0x0000771656a6fbf2 in ?? () from /usr/lib/libggml-sycl.so.0
#9  0x0000771656a67e7c in ?? () from /usr/lib/libggml-sycl.so.0
#10 0x0000771656a64048 in ?? () from /usr/lib/libggml-sycl.so.0
#11 0x0000771656206f15 in ggml_backend_sched_graph_compute_async () from /usr/lib/libggml-base.so.0
#12 0x0000771678f4a1a1 in llama_context::graph_compute(ggml_cgraph*, bool) () from /usr/lib/libllama.so.0
#13 0x0000771678f494c9 in llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) () from /usr/lib/libllama.so.0
#14 0x0000771678f4bd89 in llama_context::decode(llama_batch_ext const&) () from /usr/lib/libllama.so.0
#15 0x0000771678f517c9 in llama_decode () from /usr/lib/libllama.so.0
#16 0x000077167963bf35 in ?? () from /usr/lib/libllama-perplexity-impl.so
#17 0x00007716796306ee in llama_perplexity(int, char**) () from /usr/lib/libllama-perplexity-impl.so
#18 0x0000771655b91781 in ?? () from /usr/lib/libc.so.6
#19 0x0000771655b918b9 in __libc_start_main () from /usr/lib/libc.so.6
#20 0x0000000000406405 in ?? ()
[Inferior 1 (process 847839) detached]
