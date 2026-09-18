# Host-only test support

This is not a SYCL implementation or an ESIMD compiler.

- `maple-submission-mock-tests`: executes the real host enqueue code; does not execute device lambdas.
- `maple-token-reuse-emulation-tests`: executes the new grouped-reuse source with scalar array, gather and s2/s8 dot models. No actual DPAS instructions, GPU scheduling, subgroup semantics or driver validation.
- `-fsyntax-only` on `architecture_compare.cpp`: ordinary C++ host syntax coverage only.

Do not put this directory in a production include path. Production targets never include it. Collective placeholders must not be used as a correctness oracle for real workgroup kernels.
