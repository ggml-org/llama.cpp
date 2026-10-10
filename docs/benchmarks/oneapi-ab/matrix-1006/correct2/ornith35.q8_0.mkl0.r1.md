[New LWP 864938 (id 2)]
[New LWP 864937 (id 3)]
[New LWP 864936 (id 4)]
[New LWP 864935 (id 5)]
[New LWP 864934 (id 6)]
[New LWP 864933 (id 7)]
[New LWP 864932 (id 8)]
[New LWP 864931 (id 9)]
[New LWP 864930 (id 10)]
[New LWP 864929 (id 11)]
[New LWP 864928 (id 12)]
[New LWP 863618 (id 13)]
[New LWP 863609 (id 14)]
Registering SYCL extensions for gdb
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/libthread_db.so.1".
[Switching to thread 1 (Thread 0x794a60aaee00 (LWP 863601))]
0x0000794a630cc8f2 in ?? () from /usr/lib/libc.so.6
#0  0x0000794a630cc8f2 in ?? () from /usr/lib/libc.so.6
#1  0x0000794a631314cb in wait4 () from /usr/lib/libc.so.6
#2  0x0000794a6369dd8a in ggml_print_backtrace () from /usr/lib/libggml-base.so.0
#3  0x0000794a6369ceb9 in ggml_abort () from /usr/lib/libggml-base.so.0
#4  0x0000794a63f0ef78 in ?? () from /usr/lib/libggml-sycl.so.0
#5  0x0000794a63f76117 in ggml_sycl_op_mul_mat_sycl(ggml_backend_sycl_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, sycl::_V1::queue* const&) () from /usr/lib/libggml-sycl.so.0
#6  0x0000794a63f3c7d5 in ?? () from /usr/lib/libggml-sycl.so.0
#7  0x0000794a63f31bf2 in ?? () from /usr/lib/libggml-sycl.so.0
#8  0x0000794a63f29e7c in ?? () from /usr/lib/libggml-sycl.so.0
#9  0x0000794a63f26048 in ?? () from /usr/lib/libggml-sycl.so.0
#10 0x0000794a636c8f15 in ggml_backend_sched_graph_compute_async () from /usr/lib/libggml-base.so.0
#11 0x0000794a8640c1a1 in llama_context::graph_compute(ggml_cgraph*, bool) () from /usr/lib/libllama.so.0
#12 0x0000794a8640b4c9 in llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) () from /usr/lib/libllama.so.0
#13 0x0000794a8640dd89 in llama_context::decode(llama_batch_ext const&) () from /usr/lib/libllama.so.0
#14 0x0000794a864137c9 in llama_decode () from /usr/lib/libllama.so.0
#15 0x0000794a86afdf35 in ?? () from /usr/lib/libllama-perplexity-impl.so
#16 0x0000794a86af26ee in llama_perplexity(int, char**) () from /usr/lib/libllama-perplexity-impl.so
#17 0x0000794a63053781 in ?? () from /usr/lib/libc.so.6
#18 0x0000794a630538b9 in __libc_start_main () from /usr/lib/libc.so.6
#19 0x0000000000406405 in ?? ()
[Inferior 1 (process 863601) detached]
