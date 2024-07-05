BLIS Installation Manual
------------------------

BLIS is a portable software framework for high-performance BLAS-like dense linear algebra libraries. It has received awards and recognition, including the 2023 James H. Wilkinson Prize for Numerical Software and the 2020 SIAM Activity Group on Supercomputing Best Paper Prize. BLIS provides a new BLAS-like API and a compatibility layer for traditional BLAS routine calls. It offers features such as object-based API, typed API, BLAS and CBLAS compatibility layers.

Project URL: https://github.com/flame/blis

### Prepare:

Compile BLIS:

```bash
git clone https://github.com/flame/blis
cd blis
./configure --enable-cblas -t openmp,pthreads auto
# will install to /usr/local/ by default.
make -j
```

Install BLIS:

```bash
sudo make install
```

We recommend using openmp since it's easier to modify the cores being used.

### llama.cpp compilation

Makefile:

```bash
make GGML_BLIS=1 -j
# make GGML_BLIS=1 llama-benchmark-matmult
```

CMake:

```bash
mkdir build
cd build
cmake -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=FLAME ..
make -j
```

### llama.cpp execution

According to the BLIS documentation, we could set the following
environment variables to modify the behavior of openmp:

```bash
export GOMP_CPU_AFFINITY="0-19"
export BLIS_NUM_THREADS=14
```

And then run the binaries as normal.


### Intel specific issue

Some might get the error message saying that `libimf.so` cannot be found.
Please follow this [stackoverflow page](https://stackoverflow.com/questions/70687930/intel-oneapi-2022-libimf-so-no-such-file-or-directory-during-openmpi-compila).

### Reference:

1. https://github.com/flame/blis#getting-started
2. https://github.com/flame/blis/blob/master/docs/Multithreading.md
