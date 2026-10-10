| model                          |       size |     params | backend    | ngl |   fit | type_k | type_v |  fa |       fitt |        fitc |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ----: | -----: | -----: | --: | ---------: | ----------: | --------------: | -------------------: |
[New LWP 411039]
[New LWP 411038]
[New LWP 411037]
[New LWP 411036]
[New LWP 411035]
[New LWP 411034]
[New LWP 411033]
[New LWP 411032]
[New LWP 411031]
[New LWP 411030]
[New LWP 411029]
[New LWP 410304]
[New LWP 410120]
Registering SYCL extensions for gdb
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/libthread_db.so.1".
0x00007f823e55f8f2 in ?? () from /usr/lib/libc.so.6
#0  0x00007f823e55f8f2 in ?? () from /usr/lib/libc.so.6
#1  0x00007f823e5c44cb in wait4 () from /usr/lib/libc.so.6
#2  0x00007f823eb30d8a in ggml_print_backtrace () from /usr/lib/libggml-base.so.0
#3  0x00007f823eb2feb9 in ggml_abort () from /usr/lib/libggml-base.so.0
#4  0x00007f823f350818 in ?? () from /usr/lib/libggml-sycl.so.0
#5  0x00007f823f365a28 in ?? () from /usr/lib/libggml-sycl.so.0
#6  0x00007f823eb5bb23 in ggml_backend_sched_graph_compute_async () from /usr/lib/libggml-base.so.0
#7  0x00007f825e0ccc51 in llama_context::graph_compute(ggml_cgraph*, bool) () from /usr/lib/libllama.so.0
#8  0x00007f825e0cbf79 in llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) () from /usr/lib/libllama.so.0
#9  0x00007f825e0ce662 in llama_context::decode(llama_batch_ext const&) () from /usr/lib/libllama.so.0
#10 0x00007f825e0d4049 in llama_decode () from /usr/lib/libllama.so.0
#11 0x00007f825e7cb648 in ?? () from /usr/lib/libllama-bench-impl.so
#12 0x00007f825e7c4bd8 in llama_bench(int, char**) () from /usr/lib/libllama-bench-impl.so
#13 0x00007f823e4e6781 in ?? () from /usr/lib/libc.so.6
#14 0x00007f823e4e68b9 in __libc_start_main () from /usr/lib/libc.so.6
#15 0x0000000000405405 in ?? ()
[Inferior 1 (process 410109) detached]
