[New LWP 862640 (id 2)]
[New LWP 862639 (id 3)]
[New LWP 862638 (id 4)]
[New LWP 862637 (id 5)]
[New LWP 862636 (id 6)]
[New LWP 862635 (id 7)]
[New LWP 862634 (id 8)]
[New LWP 862633 (id 9)]
[New LWP 862632 (id 10)]
[New LWP 862631 (id 11)]
[New LWP 862630 (id 12)]
[New LWP 861811 (id 13)]
[New LWP 861803 (id 14)]
Registering SYCL extensions for gdb
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/libthread_db.so.1".
[Switching to thread 1 (Thread 0x77a2f50dce00 (LWP 861795))]
0x000077a2f77e68f2 in ?? () from /usr/lib/libc.so.6
#0  0x000077a2f77e68f2 in ?? () from /usr/lib/libc.so.6
#1  0x000077a2f784b4cb in wait4 () from /usr/lib/libc.so.6
#2  0x000077a2f7db7d8a in ggml_print_backtrace () from /usr/lib/libggml-base.so.0
#3  0x000077a2f7db6eb9 in ggml_abort () from /usr/lib/libggml-base.so.0
#4  0x000077a2f8628f78 in ?? () from /usr/lib/libggml-sycl.so.0
#5  0x000077a2f8690117 in ggml_sycl_op_mul_mat_sycl(ggml_backend_sycl_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, sycl::_V1::queue* const&) () from /usr/lib/libggml-sycl.so.0
#6  0x000077a2f86567d5 in ?? () from /usr/lib/libggml-sycl.so.0
#7  0x000077a2f864bbf2 in ?? () from /usr/lib/libggml-sycl.so.0
#8  0x000077a2f8643e7c in ?? () from /usr/lib/libggml-sycl.so.0
#9  0x000077a2f8640048 in ?? () from /usr/lib/libggml-sycl.so.0
#10 0x000077a2f7de2f15 in ggml_backend_sched_graph_compute_async () from /usr/lib/libggml-base.so.0
#11 0x000077a31ab261a1 in llama_context::graph_compute(ggml_cgraph*, bool) () from /usr/lib/libllama.so.0
#12 0x000077a31ab254c9 in llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) () from /usr/lib/libllama.so.0
#13 0x000077a31ab27d89 in llama_context::decode(llama_batch_ext const&) () from /usr/lib/libllama.so.0
#14 0x000077a31ab2d7c9 in llama_decode () from /usr/lib/libllama.so.0
#15 0x000077a31b217f35 in ?? () from /usr/lib/libllama-perplexity-impl.so
#16 0x000077a31b20c6ee in llama_perplexity(int, char**) () from /usr/lib/libllama-perplexity-impl.so
#17 0x000077a2f776d781 in ?? () from /usr/lib/libc.so.6
#18 0x000077a2f776d8b9 in __libc_start_main () from /usr/lib/libc.so.6
#19 0x0000000000406405 in ?? ()
[Inferior 1 (process 861795) detached]
