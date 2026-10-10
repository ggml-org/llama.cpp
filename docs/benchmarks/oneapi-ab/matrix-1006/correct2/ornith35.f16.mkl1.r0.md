[New LWP 847034 (id 2)]
[New LWP 847033 (id 3)]
[New LWP 847032 (id 4)]
[New LWP 847031 (id 5)]
[New LWP 847030 (id 6)]
[New LWP 847029 (id 7)]
[New LWP 847028 (id 8)]
[New LWP 847027 (id 9)]
[New LWP 847026 (id 10)]
[New LWP 847025 (id 11)]
[New LWP 847024 (id 12)]
[New LWP 846279 (id 13)]
[New LWP 846271 (id 14)]
Registering SYCL extensions for gdb
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/libthread_db.so.1".
[Switching to thread 1 (Thread 0x74d4a2ce6e00 (LWP 846263))]
0x000074d4a53f08f2 in ?? () from /usr/lib/libc.so.6
#0  0x000074d4a53f08f2 in ?? () from /usr/lib/libc.so.6
#1  0x000074d4a54554cb in wait4 () from /usr/lib/libc.so.6
#2  0x000074d4a59c1d8a in ggml_print_backtrace () from /usr/lib/libggml-base.so.0
#3  0x000074d4a59c0eb9 in ggml_abort () from /usr/lib/libggml-base.so.0
#4  0x000074d4a6232f78 in ?? () from /usr/lib/libggml-sycl.so.0
#5  0x000074d4a6295f2e in ggml_sycl_pool_vmm::alloc(unsigned long, unsigned long*) () from /usr/lib/libggml-sycl.so.0
#6  0x000074d4a6299b31 in ggml_sycl_op_mul_mat_sycl(ggml_backend_sycl_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, sycl::_V1::queue* const&) () from /usr/lib/libggml-sycl.so.0
#7  0x000074d4a62607d5 in ?? () from /usr/lib/libggml-sycl.so.0
#8  0x000074d4a6255bf2 in ?? () from /usr/lib/libggml-sycl.so.0
#9  0x000074d4a624de7c in ?? () from /usr/lib/libggml-sycl.so.0
#10 0x000074d4a624a048 in ?? () from /usr/lib/libggml-sycl.so.0
#11 0x000074d4a59ecf15 in ggml_backend_sched_graph_compute_async () from /usr/lib/libggml-base.so.0
#12 0x000074d4c87301a1 in llama_context::graph_compute(ggml_cgraph*, bool) () from /usr/lib/libllama.so.0
#13 0x000074d4c872f4c9 in llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) () from /usr/lib/libllama.so.0
#14 0x000074d4c8731d89 in llama_context::decode(llama_batch_ext const&) () from /usr/lib/libllama.so.0
#15 0x000074d4c87377c9 in llama_decode () from /usr/lib/libllama.so.0
#16 0x000074d4c8e21f35 in ?? () from /usr/lib/libllama-perplexity-impl.so
#17 0x000074d4c8e166ee in llama_perplexity(int, char**) () from /usr/lib/libllama-perplexity-impl.so
#18 0x000074d4a5377781 in ?? () from /usr/lib/libc.so.6
#19 0x000074d4a53778b9 in __libc_start_main () from /usr/lib/libc.so.6
#20 0x0000000000406405 in ?? ()
[Inferior 1 (process 846263) detached]
