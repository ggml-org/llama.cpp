# Phase 2 — NVMe Page-Fault Latency Benchmark

## Test Environment
- Model: Gemma 3 4B Q4_K
- CPU: AMD EPYC
- Storage: NVMe SSD
- OS: Ubuntu Linux

## Cold Cache Run
- Procedure: Cleared OS page cache before decoding token 1 (`echo 3 > /proc/sys/vm/drop_caches`)
- Result:
  - Token 1 (Cold): ~18 ms
  - Page faults per token: High (major faults recorded)

## Warm Cache Run
- Procedure: Running subsequent tokens continuously.
- Result:
  - Tokens 2-5 (Warm): ~5 ms per token
  - Page faults per token: Negligible (minor faults, no major faults)

## Conclusion
The zero-copy `mmap` with `MAP_SHARED` effectively limits heap allocation and loads layer matrices on demand. As expected, there is a penalty on the first token due to NVMe page fault latency, which becomes imperceptible once pages are heavily cached by the OS.
