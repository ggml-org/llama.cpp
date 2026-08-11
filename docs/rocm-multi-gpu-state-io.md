# Multi-GPU ROCm pageable state-I/O workaround

## Symptom

On multi-GPU ROCm, restoring a large in-memory prompt or context-checkpoint state can fail with a GPU memory fault or:

```text
ROCm error: an illegal memory access was encountered
current device: -1
```

The error may surface on a later HIP call such as `hipHostFree`, scheduler re-reservation, or backend teardown. Hybrid recurrent models make the failure easier to reproduce because their state save/restore operations transfer large buffers. Larger context, prompt-cache reuse, and multiple devices widen the failing window; FlashAttention, CUDA/HIP graphs, MTP, and split mode are not required causes.

This matches the open ROCm runtime defect [ROCm/rocm-systems#4817](https://github.com/ROCm/rocm-systems/issues/4817) and the corrected analysis in [ggml-org/llama.cpp#26828](https://github.com/ggml-org/llama.cpp/issues/26828). Controlled upstream tests found that:

- the source/destination host allocation remained valid and faults occurred without cache eviction;
- extending its lifetime and adding full synchronization did not prevent the fault;
- the device-setting workaround proposed in llama.cpp PR #21170 did not reliably prevent it;
- temporarily registering the pageable host allocation with `hipHostRegisterPortable | hipHostRegisterReadOnly` prevented the tested failures.

The underlying runtime defect is not fixed here. This branch provides an opt-in workaround for in-memory llama state transfers.

## Workaround

Set this before starting the process:

```bash
export GGML_HIP_SAFE_STATE_IO=1
```

The option is independent of `GGML_HIP_GFX1030_NATIVE` and may be combined with normal FlashAttention, graph, MTP, layer/tensor split, and gfx1030 optimization settings.

For every host-backed llama state save or restore, the context:

1. finds the active backend's host registration callbacks;
2. temporarily registers the entire state allocation;
3. performs all deferred tensor transfers while registration remains active;
4. unregisters only after the transfer object has flushed.

This covers server RAM prompt-cache main/draft states and host-backed context checkpoints through the central `llama_state_*` APIs. `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` remains unchanged. No state copy, cache mutex, global stream synchronization, or per-token synchronization is added.

If a compatible HIP backend is present but registration fails, the state operation returns failure instead of silently returning to the known-unsafe pageable path. Builds without a compatible GPU registration callback ignore the HIP-specific opt-in.

`--cache-ram 0` remains a stock-build mitigation for server prompt-cache restores, but it does not cover every host checkpoint path.

## Validation on four gfx1030 GPUs

The Qwen3.6 35B four-V620 server was run with the production optimization stack, FlashAttention, HIP graphs, layer split, MTP, a 262144-token context, and a 64 GiB RAM prompt-cache limit. Alternating 33K-36K-token prompt families forced twelve partial cache restores (`f_keep` approximately 0.985) and fourteen recurrent shrink/expand cycles.

The run completed all fourteen requests with:

- 50 successful temporary registrations, including approximately 65-71 MiB draft states and 708-767 MiB main states;
- twelve confirmed RAM prompt-cache restores;
- no illegal memory access or failed restore.

A second run kept active MTP drafting and completed two additional long prompt-cache restores. Host sequence save/load regression tests generated the same deterministic tokens with the workaround off and on, and fragmented/equivalence restore tests passed with zero reported bitwise mismatches.

Preserved local artifacts:

- `/tmp/safe-state-server-stress.json`
- `/tmp/safe-state-server-stress.log`
- `/tmp/safe-state-server-mtp.json`
- `/tmp/safe-state-server-mtp.log`
- `/tmp/test-safe-state-io-debug.log`
- `/tmp/test-state-io-stock.log`
- `/tmp/test-state-restore-fragmented-safe.log`
- `/tmp/test-state-restore-equivalence-safe.log`