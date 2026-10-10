[New LWP 843539 (id 2)]
[New LWP 843538 (id 3)]
[New LWP 843537 (id 4)]
[New LWP 843535 (id 5)]
[New LWP 843534 (id 6)]
[New LWP 843533 (id 7)]
[New LWP 843532 (id 8)]
[New LWP 843531 (id 9)]
[New LWP 843530 (id 10)]
[New LWP 843529 (id 11)]
[New LWP 843528 (id 12)]
[New LWP 842655 (id 13)]
[New LWP 842646 (id 14)]
Registering SYCL extensions for gdb
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/libthread_db.so.1".
[Switching to thread 1 (Thread 0x7c071c8a0e00 (LWP 842638))]
0x00007c071efaa8f2 in ?? () from /usr/lib/libc.so.6
#0  0x00007c071efaa8f2 in ?? () from /usr/lib/libc.so.6
#1  0x00007c071f00f4cb in wait4 () from /usr/lib/libc.so.6
#2  0x00007c071f57bd8a in ggml_print_backtrace () from /usr/lib/libggml-base.so.0
#3  0x00007c071f57aeb9 in ggml_abort () from /usr/lib/libggml-base.so.0
#4  0x00007c071fdecf78 in ?? () from /usr/lib/libggml-sycl.so.0
#5  0x00007c071fe54117 in ggml_sycl_op_mul_mat_sycl(ggml_backend_sycl_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, sycl::_V1::queue* const&) () from /usr/lib/libggml-sycl.so.0
#6  0x00007c071fe1a7d5 in ?? () from /usr/lib/libggml-sycl.so.0
#7  0x00007c071fe0fbf2 in ?? () from /usr/lib/libggml-sycl.so.0
#8  0x00007c071fe07e7c in ?? () from /usr/lib/libggml-sycl.so.0
#9  0x00007c071fe04048 in ?? () from /usr/lib/libggml-sycl.so.0
#10 0x00007c071f5a6f15 in ggml_backend_sched_graph_compute_async () from /usr/lib/libggml-base.so.0
#11 0x00007c07422ea1a1 in llama_context::graph_compute(ggml_cgraph*, bool) () from /usr/lib/libllama.so.0
#12 0x00007c07422e94c9 in llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) () from /usr/lib/libllama.so.0
#13 0x00007c07422ebd89 in llama_context::decode(llama_batch_ext const&) () from /usr/lib/libllama.so.0
#14 0x00007c07422f17c9 in llama_decode () from /usr/lib/libllama.so.0
#15 0x00007c07429dbf35 in ?? () from /usr/lib/libllama-perplexity-impl.so
#16 0x00007c07429d06ee in llama_perplexity(int, char**) () from /usr/lib/libllama-perplexity-impl.so
#17 0x00007c071ef31781 in ?? () from /usr/lib/libc.so.6
#18 0x00007c071ef318b9 in __libc_start_main () from /usr/lib/libc.so.6
#19 0x0000000000406405 in ?? ()
[Inferior 1 (process 842638) detached]
