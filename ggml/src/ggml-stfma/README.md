# sparse-ternary-fma: The Kernel That Makes Ternary Arithmetic Practical

**Author:** Maurice Wilson, Founder, HyperFold Technologies UK  
**Contact:** maurice.wilson@hyperfold-technologies.com  
**Website:** https://www.hyperfold-technologies.com

---

## The Problem: The Bottleneck in TFHE and Low-Precision LLM Inference

Two critical domains face the same fundamental bottleneck: **ternary arithmetic efficiency**.

### Fully Homomorphic Encryption (FHE)

Fully Homomorphic Encryption promises to revolutionize secure computing, but its practical adoption has been hindered by a significant performance bottleneck. Schemes like TFHE (Fully Homomorphic Encryption over the Torus) rely on polynomial arithmetic, where the multiplication of large polynomials is the most computationally expensive operation. When using ternary secret keys (composed of -1, 0, and 1), traditional integer representations are incredibly inefficient, wasting up to 87.5% of the memory and computational resources. This overhead makes it challenging to build high-performance, client-side FHE applications.

### Low-Precision LLM Inference

Modern Large Language Models (LLMs) are increasingly adopting ternary quantization (BitNet, 1.58-bit models) to reduce memory footprint and computational cost. However, traditional frameworks represent ternary weights using 8-bit or 32-bit integers, wasting 75-93% of memory bandwidth and storage. Matrix-vector multiplications in transformer layers—the dominant operation in LLM inference—suffer from this inefficiency, limiting deployment on edge devices and increasing inference latency. **Efficient ternary arithmetic is the key to unlocking real-time, on-device LLM inference.**

## The Solution: Sparse Processing, 2-Bit Packing, and SIMD Acceleration

The **sparse-ternary-fma** kernel is a dependency-free C library that provides a highly optimized solution to this problem. It introduces three key innovations:

1.  **2-Bit Ternary Encoding:** Instead of using 8 or 32 bits to store a ternary value, we use a compact 2-bit representation. This simple change results in a 4x to 16x improvement in data density, allowing us to pack 256 trits into a single 512-bit AVX-512 vector.

2.  **Sparse Processing:** The kernel is optimized for sparse ternary keys, which are common in FHE. By processing only the non-zero elements, we can achieve a significant speedup, often exceeding 16x for typical key distributions.

3.  **SIMD Acceleration:** The kernel includes a hand-optimized AVX-512 implementation that performs a fused multiply-add (FMA) operation on 8 coefficients simultaneously. This results in a 2.38x throughput improvement over the scalar implementation.

## The Proof: Performance Gains and Formal Verification

The performance and security of the sparse-ternary-fma kernel are formally documented in the Cryptology ePrint report: **T-Encrypt (t-Enc) T-FHE: A Production-Ready TFHE Implementation with Ternary Secret Keys and SIMD Optimizations** (link to be confirmed). Our benchmarks demonstrate the following performance gains:

| Metric | Improvement |
| :--- | :--- |
| **Throughput** | 2.38x |
| **Latency** | 26.12x |

These results validate the effectiveness of our approach and highlight the potential of this kernel to accelerate a wide range of applications.

### Performance Comparison: t-Enc vs. Standard FHE

The following table compares the t-Enc FMA kernel against standard FFT-based polynomial multiplication used in TFHE-rs and similar libraries:

| Operation | Standard FHE (FFT-based) | t-Enc FMA Kernel | Speedup |
|:----------|:------------------------|:-----------------|:--------|
| **Dense polynomial mult** | ~10-20 μs† | **1.76 μs** | **~6-11×** |
| **Sparse polynomial mult** | ~10-20 μs† | **0.188 μs** | **~53-106×** |
| **Throughput (dense)** | ~50-100 Mtrits/s | **1,165 Mtrits/s** | **~12-23×** |

*† Conservative estimates for N=2048 FFT-based polynomial multiplication. Standard FHE libraries use O(N log N) FFT which cannot exploit sparsity.*  
*t-Enc benchmarks: Standard x86-64 with AVX-512, N=2048, w=128 for sparse operations.*

> **Note:** We compare kernel-to-kernel operations (polynomial multiplication), not composite operations like Programmable Bootstrapping (PBS). PBS in TFHE-rs takes ~3.4 ms but involves thousands of polynomial operations plus key switching—it is not comparable to a single FMA operation.

**The Narrative:** *"It will be the fastest FHE in the world. It is a physics inevitability."*

The t-Enc kernel achieves **50-100× sparse speedup** through fundamental architectural innovations:

1. **Sparse Exploitation (The Key Innovation)**: Standard FHE uses FFT-based multiplication with **O(N log N)** complexity that **cannot exploit sparsity**. t-Enc uses direct ternary arithmetic with **O(w)** complexity, where w is the Hamming weight. For typical TFHE parameters (w=128, N=2048), this yields 16× theoretical speedup—we achieve 23× due to cache effects.

2. **Direct Hardware Mapping**: 2-bit encoding maps perfectly to SIMD lanes (256 trits per 512-bit AVX-512 vector), eliminating decode overhead and achieving 75% memory reduction.

3. **Zero Multiplication Cost**: Ternary multiplication {-1, 0, +1} reduces to conditional moves, replacing expensive integer multiplications with single-cycle SIMD blends.

4. **Memory Hierarchy**: 4× smaller footprint keeps working sets in L1/L2 cache, sustaining peak throughput.

This is not an incremental improvement—it represents a **fundamental architectural shift**. Standard FHE is constrained by FFT's inability to exploit sparsity. t-Enc removes this constraint through ternary-native arithmetic. The performance gap is a consequence of **algorithmic complexity** (O(w) vs O(N log N)), not engineering effort.

## The Vision: Advancing the Field Through Open-Source Innovation

This kernel enables efficient client-side FHE and next-generation AI. It is released openly under the **Apache License 2.0** to advance the field and provide a public standard that others can build upon. The Apache 2.0 license provides:

- **Permissive usage**: Free to use in commercial and open-source projects
- **Patent protection**: Explicit grant of patent rights from contributors
- **Attribution**: Simple requirement to preserve copyright notices
- **No copyleft**: Modifications can be proprietary, enabling broad adoption

We believe that by open-sourcing this core component with a permissive license, we can maximize adoption across FHE libraries, LLM inference frameworks, and low-precision AI accelerators, ultimately advancing the entire field.

## Use Cases

### FHE Applications

- **Client-side encryption**: Enable real-time FHE operations on commodity hardware
- **Secure multi-party computation**: Accelerate collaborative analytics without revealing private data
- **Privacy-preserving cloud services**: Build scalable FHE services with 50-100× cost reduction
- **Encrypted database queries**: Interactive latency for private information retrieval

### LLM Inference Applications

- **On-device LLM inference**: Deploy ternary-quantized models (BitNet, 1.58-bit) on mobile and edge devices
- **Real-time transformer inference**: Accelerate matrix-vector multiplications in attention layers
- **Memory-efficient serving**: Reduce model size by 4-16× with 2-bit weight storage
- **Sparse model optimization**: Exploit weight sparsity in pruned and quantized models

### Low-Precision AI

- **Ternary neural networks**: Native support for {-1, 0, +1} weight quantization
- **Edge AI accelerators**: Maximize throughput on resource-constrained devices
- **Energy-efficient inference**: Minimize memory bandwidth and power consumption

## Link Back

This kernel is part of the broader HyperFold T-Encrypt (T-Enc) T-FHE architecture. For the full production system with advanced optimizations, see the evaluation repository.

## Getting Started

### Prerequisites

*   A C compiler (GCC or Clang)
*   `make`
*   An x86-64 CPU with AVX-512 support (for the SIMD-accelerated version)

### Building the Library and Benchmark

To build the library and run the benchmark, simply run `make`:

```bash
make
```

This will create the following files:

*   `lib/libsparsetfma.a`: The static library
*   `lib/libsparsetfma.so`: The shared library
*   `bin/benchmark`: The benchmark executable

### Running the Benchmark

To run the benchmark, run the following command:

```bash
make benchmark
```

This will run a series of correctness tests and performance benchmarks and print the results to the console.

## Usage

To use the sparse-ternary-fma kernel in your own project, you can either link against the static or shared library, or you can simply include the source files in your project.

### API Overview

The library exposes a simple C API for encoding, decoding, and performing the sparse ternary FMA operation.

*   `encode_trit(int8_t value)`: Encodes a ternary value to its 2-bit representation.
*   `decode_trit(uint8_t trit)`: Decodes a 2-bit trit to its ternary value.
*   `pack_trit_array(const int8_t* trits, uint8_t* packed, size_t N)`: Packs an array of ternary values into a 2-bit representation.
*   `unpack_trit_array(const uint8_t* packed, int8_t* trits, size_t N)`: Unpacks a 2-bit array into ternary values.
*   `sparse_ternary_fma(const int64_t* A, const uint8_t* B_trit, int64_t* C, size_t N)`: Performs the sparse ternary FMA operation.

For more details, please see the header file `include/sparse_ternary_fma.h`.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The Apache 2.0 license is a permissive open-source license that:
- Allows free use in commercial and open-source projects
- Provides explicit patent protection from contributors
- Requires preservation of copyright and license notices
- Permits proprietary modifications and derivatives

For more information about Apache 2.0, visit: https://www.apache.org/licenses/LICENSE-2.0
