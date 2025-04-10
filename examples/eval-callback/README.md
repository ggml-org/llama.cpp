# llama.cpp/examples/eval-callback

A simple example which demonstrates how to use callback during the inference.
It simply prints to the console all operations and tensor data.

Usage:

```shell
llama-eval-callback \
  --hf-repo ggml-org/models \
  --hf-file phi-2/ggml-model-q4_0.gguf \
  --model phi-2-q4_0.gguf \
  --prompt hello \
  --seed 42 \
  -ngl 33
```

Will print:

```shell
llm_load_tensors: offloaded 33/33 layers to GPU
...
llama_new_context_with_model: n_ctx      = 512
...
llama_new_context_with_model:      CUDA0 compute buffer size =   105.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =     6.01 MiB
llama_new_context_with_model: graph nodes  = 1225
llama_new_context_with_model: graph splits = 2
ggml_debug:                 inp_embd = (f32)   GET_ROWS(token_embd.weight{2560, 51200, 1, 1}, inp_tokens{1, 1, 1, 1}}) = {2560, 1, 1, 1}
                                     [
                                      [
                                       [ -0.0181,   0.0272,   0.0272, ...],
                                      ],
                                     ]
ggml_debug:                   norm-0 = (f32)       NORM(CUDA0#inp_embd#0{2560, 1, 1, 1}, }) = {2560, 1, 1, 1}
                                     [
                                      [
                                       [ -0.6989,   1.0636,   1.0636, ...],
                                      ],
                                     ]
ggml_debug:                 norm_w-0 = (f32)        MUL(norm-0{2560, 1, 1, 1}, blk.0.attn_norm.weight{2560, 1, 1, 1}}) = {2560, 1, 1, 1}
                                     [
                                      [
                                       [ -0.1800,   0.2817,   0.2632, ...],
                                      ],
                                     ]
ggml_debug:              attn_norm-0 = (f32)        ADD(norm_w-0{2560, 1, 1, 1}, blk.0.attn_norm.bias{2560, 1, 1, 1}}) = {2560, 1, 1, 1}
                                     [
                                      [
                                       [ -0.1863,   0.2970,   0.2604, ...],
                                      ],
                                     ]
ggml_debug:                   wqkv-0 = (f32)    MUL_MAT(blk.0.attn_qkv.weight{2560, 7680, 1, 1}, attn_norm-0{2560, 1, 1, 1}}) = {7680, 1, 1, 1}
                                     [
                                      [
                                       [ -1.1238,   1.2876,  -1.8086, ...],
                                      ],
                                     ]
ggml_debug:                   bqkv-0 = (f32)        ADD(wqkv-0{7680, 1, 1, 1}, blk.0.attn_qkv.bias{7680, 1, 1, 1}}) = {7680, 1, 1, 1}
                                     [
                                      [
                                       [ -1.1135,   1.4604,  -1.9226, ...],
                                      ],
                                     ]
ggml_debug:            bqkv-0 (view) = (f32)       VIEW(bqkv-0{7680, 1, 1, 1}, }) = {2560, 1, 1, 1}
                                     [
                                      [
                                       [ -1.1135,   1.4604,  -1.9226, ...],
                                      ],
                                     ]
ggml_debug:                   Qcur-0 = (f32)       CONT(bqkv-0 (view){2560, 1, 1, 1}, }) = {2560, 1, 1, 1}
                                     [
                                      [
                                       [ -1.1135,   1.4604,  -1.9226, ...],
                                      ],
                                     ]
ggml_debug:        Qcur-0 (reshaped) = (f32)    RESHAPE(Qcur-0{2560, 1, 1, 1}, }) = {80, 32, 1, 1}
                                     [
                                      [
                                       [ -1.1135,   1.4604,  -1.9226, ...],
                                       [ -0.3608,   0.5076,  -1.8866, ...],
                                       [  1.7643,   0.0273,  -2.1065, ...],
                                       ...
                                      ],
                                     ]
ggml_debug:                   Qcur-0 = (f32)       ROPE(Qcur-0 (reshaped){80, 32, 1, 1}, CUDA0#inp_pos#0{1, 1, 1, 1}}) = {80, 32, 1, 1}
                                     [
                                      [
                                       [ -1.1135,   1.4604,  -1.9226, ...],
                                       [ -0.3608,   0.5076,  -1.8866, ...],
                                       [  1.7643,   0.0273,  -2.1065, ...],
                                       ...
                                      ],
                                     ]
```
