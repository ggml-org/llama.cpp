# Contents
/
* **[Contents](Multithreading.md#contents)**
* **[Introduction](Multithreading.md#introduction)**
* **[Enabling multithreading](Multithreading.md#enabling-multithreading)**
  * [Choosing OpenMP vs pthreads](Multithreading.md#choosing-openmp-vs-pthreads)
  * [Specifying thread-to-core affinity](Multithreading.md#specifying-thread-to-core-affinity)
* **[Specifying multithreading](Multithreading.md#specifying-multithreading)**
  * [Globally via environment variables](Multithreading.md#globally-via-environment-variables)
    * [The automatic way](Multithreading.md#environment-variables-the-automatic-way)
    * [The manual way](Multithreading.md#environment-variables-the-manual-way)
    * [Overriding the default threading implementation](Multithreading.md#environment-variables-overriding-the-default-threading-implementation)
  * [Globally at runtime](Multithreading.md#globally-at-runtime)
    * [The automatic way](Multithreading.md#globally-at-runtime-the-automatic-way)
    * [The manual way](Multithreading.md#globally-at-runtime-the-manual-way)
    * [Overriding the default threading implementation](Multithreading.md#globally-at-runtime-overriding-the-default-threading-implementation)
  * [Locally at runtime](Multithreading.md#locally-at-runtime)
    * [Initializing a rntm_t](Multithreading.md#initializing-a-rntm-t)
    * [The automatic way](Multithreading.md#locally-at-runtime-the-automatic-way)
    * [The manual way](Multithreading.md#locally-at-runtime-the-manual-way)
    * [Overriding the default threading implementation](Multithreading.md#locally-at-runtime-overriding-the-default-threading-implementation)
    * [Using the expert interface](Multithreading.md#locally-at-runtime-using-the-expert-interface)
* **[Known issues](Multithreading.md#known-issues)**
* **[Conclusion](Multithreading.md#conclusion)**


# Introduction

Our paper [Anatomy of High-Performance Many-Threaded Matrix Multiplication](https://github.com/flame/blis#citations), presented at IPDPS'14, identified five loops around the microkernel as opportunities for parallelization within level-3 operations such as `gemm`. Within BLIS, we have enabled parallelism for four of those loops, with the fifth planned for future work. This software architecture extends naturally to all level-3 operations except for `trsm`, where its application is necessarily limited to three of the five loops due to inter-iteration dependencies.

**IMPORTANT**: Multithreading in BLIS is disabled by default. Furthermore, even when multithreading is enabled, BLIS will default to single-threaded execution at runtime. In order to both *allow* and *invoke* parallelism from within BLIS operations, you must both *enable* multithreading at configure-time and *specify* multithreading at runtime.

To summarize: In order to observe multithreaded parallelism within a BLIS operation, you must do *both* of the following:
1. Enable multithreading at configure-time. This is discussed in the [next section](docs/Multithreading.md#enabling-multithreading).
2. Specify multithreading at runtime. This is also discussed [later on](docs/Multithreading.md#specifying-multithreading).

# Enabling multithreading

BLIS disables multithreading by default. In order to allow multithreaded parallelism from BLIS, you must first enable multithreading explicitly at configure-time.

As of this writing, BLIS optionally supports multithreading via OpenMP or POSIX threads(or both).

To enable multithreading via OpenMP, you must provide the `--enable-threading` option to the `configure` script:
```
$ ./configure --enable-threading=openmp auto
```
In this example, we target the `auto` configuration, which is like asking `configure` to choose the most appropriate configuration based on some detection heuristic (e.g. `cpuid` on x86_64 hardware). Similarly, to enable multithreading via POSIX threads (pthreads), specify the threading model as `pthreads` instead of `openmp`:
```
$ ./configure --enable-threading=pthreads auto
```
You can also use the shorthand option for `--enable-threading`, which is `-t`:
```
$ ./configure -t openmp auto
$ ./configure -t pthreads auto
```
You may even combine multiple threading implementations into the same library build. We call this "fat threading." When more than one option is given, the first option acts as the default. Note that no matter what arguments you specify for the `-t` option, the single-threaded implementation will always be available.
```
$ ./configure -t openmp,pthreads auto
```
In the above example, OpenMP will serve as the default threading implementation since it is listed first. This default can be overridden at runtime, though, which is discussed later on.
For more complete and up-to-date information on the `--enable-threading` option, run `configure` with the `--help` (or `-h`) option:
```
$ ./configure --help
```

## Choosing OpenMP vs pthreads

While we provide the ability to implement multithreading in BLIS in terms of either OpenMP or pthreads, we typically encourage users to opt for OpenMP:
```
$ ./configure -t openmp auto
```
The reason mostly comes down to the fact that most OpenMP implementations (most notably GNU) allow the user to conveniently bind threads to cores via an environment variable(s) set prior to running the application. This is important because when the operating system causes a thread to migrate from one core to another, the thread will typically leave behind the data it was using in the L1 and L2 caches. That data may not be present in the caches of the destination core. Once the thread resumes execution from the new core, it will experience a period of frequent cache misses as the data it was previously using is transmitted once again through the cache hierarchy. If migration happens frequently enough, it can pose a significant (and unnecessary) drag on performance.

Note that binding threads to cores is possible in pthreads, but it requires a runtime call to the operating system, such as `sched_setaffinity()`, to convey the thread binding information, and BLIS does not yet implement this behavior for pthreads.

## Specifying thread-to-core affinity

The solution to thread migration is setting *processor affinity*. In this context, affinity refers to the tendency for a thread to remain bound to a particular compute core. There are at least two ways to set affinity in OpenMP. The first way offers more control, but requires you to understand a bit about the processor topology and how core IDs are mapped to physical cores, while the second way is simpler but less powerful.

Let's start with an example. Suppose I have a two-socket system with a total of eight cores, four cores per socket. By setting `GOMP_CPU_AFFINITY` as follows
```
$ export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7"
```
I am communicating to OpenMP that the first thread to be created should be spawned on core 0, from which it should not migrate. The second thread to be created should be spawned on core 1, from which it should not migrate, and so forth. If socket 0 has cores 0-3 and socket 1 has 4-7, this would result in the first four threads on socket 0 and the second four threads on socket 1. (And if more than eight threads are spawned, the mapping wraps back around, staring from the beginning.) So with `GOMP_CPU_AFFINITY`, you are doing more than just preventing threads from migrating once they are spawned--you are specifying the cores on which they will be spawned in the first place.

Another example: Suppose the hardware numbers the cores alternatingly between sockets, such that socket 0 gets even-numbered cores and socket 1 gets odd-numbered cores. In such a scenario, you might want to use `GOMP_CPU_AFFINITY` as follows
```
$ export GOMP_CPU_AFFINITY="0 2 4 6 1 3 5 7"
```
Because the first four entries are `0 2 4 6`, threads 0-3 would be spawned on the first socket, since that is where cores 0, 2, 4, and 6 are located. Similarly, the subsequent `1 3 5 7` would cause threads 4-7 to be spawned on the second socket, since that is where cores 1, 3, 5, and 7 reside. Of course, setting `GOMP_CPU_AFFINITY` in this way implies that BLIS benefits from this kind of grouping of threads--which, generally, it does. As a general rule, you should try to fill up a socket with one thread per core before moving to the next socket.

A second method of specifying affinity is via `OMP_PROC_BIND`, which is much simpler to set:
```
$ export OMP_PROC_BIND=close
```
This binds the threads close to the master thread, in contiguous "place" partitions. (There are other valid values aside from `close`.) Places are specified by another variable, `OMP_PLACES`:
```
$ export OMP_PLACES=cores
```
The `cores` value is most appropriate for BLIS since we usually want to ignore hardware threads (symmetric multithreading, or "hyperthreading" on Intel systems) and instead map threads to physical cores.

Setting these two variables is often enough. However, it obviously does not offer the level of control that `GOMP_CPU_AFFINITY` does. Sometimes, it takes some experimentation to determine whether a particular mapping is performing as expected. If multithreaded performance on eight cores is only twice what it is observed of single-threaded performance, the affinity mapping may be to blame. But if performance is six or seven times higher than sequential execution, then the mapping you chose is probably working fine.

Unfortunately, the topic of thread-to-core affinity is well beyond the scope of this document. (A web search will uncover many [great resources](https://web.archive.org/web/20190130102805/http://www.nersc.gov/users/software/programming-models/openmp/process-and-thread-affinity) discussing the use of [GOMP_CPU_AFFINITY](https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fCPU_005fAFFINITY.html) and [OMP_PROC_BIND](https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fPROC_005fBIND.html#OMP_005fPROC_005fBIND).) It's up to the user to determine an appropriate affinity mapping, and then choose your preferred method of expressing that mapping to the OpenMP implementation.


# Specifying multithreading

There are three broad methods of specifying multithreading in BLIS:
* [Globally via environment variables](Multithreading.md#globally-via-environment-variables)
* [Globally at runtime](Multithreading.md#globally-at-runtime)
* [Locally at runtime](Multithreading.md#locally-at-runtime) (that is, on a per-call, thread-safe basis)

Within each of these three broad methods there are two specific ways of expressing a request for parallelism. First, the user may express a single number--the total number of threads, or ways of parallelism, to use within a single operation such as `gemm`. We call this the "automatic" way. Alternatively, the user may express the number of ways of parallelism to obtain within *each loop* of the level-3 operation. We call this the "manual" way. The latter way is actually what BLIS eventually needs before it can perform its multithreading; the former is viable only because we have a heuristic of determining a reasonable instance of the latter when given the former.
This choice--automatic or manual--must be made regardless of which of the three methods is used.

Regardless of which method is employed, and which specific way within each method, after setting the number of threads, the application may call the desired level-3 operation (via either the [typed API](docs/BLISTypedAPI.md) or the [object API](docs/BLISObjectAPI.md)) and the operation will execute in a multithreaded manner. (When calling BLIS via the BLAS API, only the first two (global) methods are available.)

**Note**: Please be aware of what happens if you try to specify both the automatic and manual ways, as it could otherwise confuse new users. Here are the important points:
 * Regardless of which of the three methods is used, **if multithreading is specified via both the automatic and manual ways, the values set via the manual way will always take precedence.**
 * Specifying parallelism for even *one* loop counts as specifying the manual way (in which case the ways of parallelism for the remaining loops will be assumed to be 1). (Note: Setting the ways of parallelism for a loop to any value less than or equal to 1 does *not* count as specifying parallelism for that loop; in these cases, the default of 1 will silently be used instead.) If you want to switch from using the manual way to automatic way, you must not only set (`export`) the `BLIS_NUM_THREADS` variable, but you must either `unset` all of the `BLIS_*_NT` variables, or make sure they are all set to 1.
 * If you have specified multithreading via *both* the automatic and manual ways, BLIS will **not** complain if the values are inconsistent with one another. (For example, you may request 12 total threads be used while also specifying 2 and 4 ways of parallelism within the JC and IC loops, respectively, for a total of 8 ways. 12 is obviously not equal to 8, and in this case the 8-thread specification will prevail.) Furthermore, you will be able to query these inconsistent values via the runtime API both before and after multithreading executes.
 * If multithreading is disabled, you **may still** specify multithreading values via either the manual or automatic ways. However, BLIS will silently ignore **all** of these values. A BLIS library that is built with multithreading disabled at configure-time will always run sequentially (from the perspective of a single application thread).

Furthermore:
* For small numbers of threads, the number requested will be honored faithfully. However, if you request a larger number of threads that happens to also be prime, BLIS will (by default) reduce the number by one in order to allow more more efficient thread factorizations. This behavior (in which `BLIS_DISABLE_AUTO_PRIME_NUM_THREADS` is set by default) can be overridden by configuring BLIS with the `BLIS_ENABLE_AUTO_PRIME_NUM_THREADS` macro defined in the `bli_family_*.h` file of the relevant target configuration. This `BLIS_ENABLE_*` macro will allow BLIS to use any prime number of threads. Note that the threshold beyond which BLIS will reduce primes by one (assuming `BLIS_DISABLE_AUTO_PRIME_NUM_THREADS` is set) can be set via `BLIS_NT_MAX_PRIME`. This value is ignored if `BLIS_ENABLE_AUTO_PRIME_NUM_THREADS` is defined.

## Globally via environment variables

The most common method of specifying multithreading in BLIS is globally via environment variables. With this method, the user sets one or more environment variables in the shell before launching the BLIS-linked executable.

Regardless of whether you end up using the automatic or manual way of expressing a request for multithreading, note that the environment variables are read (via `getenv()`) by BLIS **only once**, when the library is initialized. Subsequent to library initialization, the global settings for parallelization may only be changed via the [global runtime API](Multithreading.md#globally-at-runtime). If this constraint is not a problem, then environment variables may work fine for you. Otherwise, please consider [local settings](Multithreading.md#locally-at-runtime). (Local settings may used at any time, regardless of whether global settings were explicitly specified, and local settings always override global settings.)

**Note**: Regardless of which way ([automatic](Multithreading.md#environment-variables-the-automatic-way) or [manual](Multithreading.md#environment-variables-the-manual-way)) environment variables are used to specify multithreading, that specification will affect operation of BLIS through **both** the BLAS compatibility layer as well as the native ([typed](docs/BLISTypedAPI.md) and [object](docs/BLISObjectAPI.md)) APIs that are unique to BLIS.

### Environment variables: the automatic way

The automatic way of specifying parallelism entails setting the total number of threads you wish BLIS to employ in its parallelization. This total number of threads is captured by the `BLIS_NUM_THREADS` environment variable. You can set this variable prior to executing your BLIS-linked executable:
```
$ export GOMP_CPU_AFFINITY="0-15"  # optional step when using GNU libgomp.
$ export BLIS_NUM_THREADS=16
$ ./my_blis_program
```
If you don't want or need your environment variable assignments to persist after `my_blis_program` completes, you can instead set the variables only for the duration of the program as follows:
```
$ GOMP_CPU_AFFINITY="0-15" BLIS_NUM_THREADS=16 ./my_blis_program
```
Either of these approaches causes BLIS to automatically determine a reasonable threading strategy based on what is known about the operation and problem size. If `BLIS_NUM_THREADS` is not set, BLIS will attempt to query the value of `BLIS_NT` (a shorthand alternative to `BLIS_NUM_THREADS`). If neither variable is defined, then BLIS will attempt to read `OMP_NUM_THREADS`. If none of these variables is set, the default number of threads is 1.

**Note**: If none of `BLIS_NT`/`BLIS_NUM_THREADS` are defined, BLIS will fall back to use
the standardized `OMP_NUM_THREADS` environment variable.
By having an application specific environment variable one can fine-tune the thread
utilization, e.g. to run OpenMP constructs using 4 threads, and BLIS with 2 threads:
```
$ OMP_NUM_THREADS=4 BLIS_NUM_THREADS=2 ./my_omp_blis_program
```


### Environment variables: the manual way

The manual way of specifying parallelism involves communicating which loops within the matrix multiplication algorithm to parallelize and the degree of parallelism to be obtained from each of those loops.

The below chart describes the five loops used in BLIS's matrix multiplication operations.

| Loop around microkernel  | Environment variable | Direction | Notes                 |
|:-------------------------|:---------------------|:----------|:----------------------|
| 5th loop ("JC loop")     | `BLIS_JC_NT`         | `n`       |                       |
| 4th loop ("PC loop")     | _N/A_                | `k`       | Unavailable; always 1 |
| 3rd loop ("IC loop")     | `BLIS_IC_NT`         | `m`       |                       |
| 2nd loop ("JR loop")     | `BLIS_JR_NT`         | `n`       | Typically <= 8        |
| 1st loop ("IR loop")     | `BLIS_IR_NT`         | `m`       | Typically 1           |

**Note**: Parallelization of the 4th loop is not currently available because each iteration of the loop updates the same part of the output matrix C. Thus, to safely parallelize it requires either a reduction or mutex locks when updating C.

Parallelization in BLIS is hierarchical. So if we parallelize multiple loops, the total number of threads will be the product of the amount of parallelism for each loop. Thus the total number of threads used is the product of all the values:
`BLIS_JC_NT * BLIS_IC_NT * BLIS_JR_NT * BLIS_IR_NT`.
Note that if you set at least one of these loop-specific variables, any others that are unset will default to 1.

In general, the way to choose how to set these environment variables is as follows: The amount of parallelism from the M and N dimensions should be roughly the same. Thus `BLIS_IR_NT * BLIS_IC_NT` should be roughly equal to `BLIS_JR_NT * BLIS_JC_NT`.

Next, which combinations of loops to parallelize depends on which caches are shared. Here are some of the more common scenarios:
 * When compute resources have private L3 caches (example: multi-socket systems), try parallelizing  the `JC` loop. This means threads (or thread groups) will pack and compute with different row panels from matrix B.
 * For compute resources that have private L2 caches but that share an L3 cache (example: cores on a socket), try parallelizing the `IC` loop. In this situation, threads will share the same packed row panel from matrix B, but pack and compute with different blocks of matrix A.
 * If compute resources share an L2 cache but have private L1 caches (example: pairs of cores), try parallelizing the `JR` loop. Here, threads share the same packed block of matrix A but read different packed micropanels of B into their private L1 caches. In some situations, *lightly* parallelizing the `IR` loop may also be effective.

![The primary algorithm for level-3 operations in BLIS](http://www.cs.utexas.edu/users/field/mm_algorithm_color.png)

### Environment variables: overriding the default threading implementation

Just as you may specify the number of threads for BLIS to use by setting environment variables prior to running your BLIS-linked application, you may also specify your preferred threading implementation. Suppose that you configured BLIS as follows:
```
$ ./configure -t openmp,pthreads auto
```
This will result in both OpenMP and pthreads implementations being compiled and included within the BLIS library, with OpenMP serving as the default (since it was listed first to the `-t` option). You can link your program against this BLIS library and force the use of pthreads (instead of OpenMP) via environment variables as follows:
```
$ BLIS_THREAD_IMPL=pthreads BLIS_NUM_THREADS=8 ./my_blis_program
```
You can even disable multithreading altogether by forcing the use of the single-threaded code path:
```
$ BLIS_THREAD_IMPL=single ./my_blis_program
```
Note that if `BLIS_THREAD_IMPL` is assigned to `single`, any other threading-related variables that may be set, such as `BLIS_NUM_THREADS` or any of the `BLIS_*_NT` variables, are ignored.
If `BLIS_THREAD_IMPL` is not set, BLIS will attempt to query its shorthand alternative, `BLIS_TI`. If neither value is set, the configure-time default (in the example shown above, OpenMP) will prevail.

## Globally at runtime

***Note:** If you want to gain access to BLIS API function prototypes, be sure to #include "blis.h" from the relevant source files in your application.*

If you still wish to set the parallelization scheme globally, but you want to do so at runtime, BLIS provides a thread-safe API for specifying multithreading. Think of these functions as a way to modify the same internal data structure into which the environment variables are read. (Recall that the environment variables are only read once, when BLIS is initialized).

**Note**: If you set parallelization globally via environment variables and *then* your application *also* uses the global runtime API to set the ways of parallelism, the global runtime API will prevail.

**Note**: Regardless of which way ([automatic](Multithreading.md#globally-at-runtime-the-automatic-way) or [manual](Multithreading.md#globally-at-runtime-the-manual-way)) the global runtime API is used to specify multithreading, that specification will affect operation of BLIS through **both** the BLAS compatibility layer as well as the native ([typed](docs/BLISTypedAPI.md) and [object](docs/BLISObjectAPI.md)) APIs that are unique to BLIS.

If BLIS is being used by two or more application-level threads, each of those application threads will track their own global state for the purpose of specifying parallelism. We felt this makes sense because each application thread may wish to specify a different parallelization scheme without affecting the scheme for the other application thread(s).

### Globally at runtime: the automatic way

If you simply want to specify an overall number of threads and let BLIS choose a thread factorization automatically, use the following function:
```c
void bli_thread_set_num_threads( dim_t n_threads );
```
This function takes one integer--the total number of threads for BLIS to utilize in any one operation. So, for example, if we call
```c
bli_thread_set_num_threads( 4 );
```
we are requesting that the total number of threads (ways of parallelism) be set to 4. You may also query the number of threads at any time via
```c
dim_t bli_thread_get_num_threads( void );
```
Which may be called in the usual way:
```c
dim_t nt = bli_thread_get_num_threads();
```

### Globally at runtime: the manual way

If you want to specify the number of ways of parallelism to obtain for each loop, use the following function:
```c
void bli_thread_set_ways( dim_t jc, dim_t pc, dim_t ic, dim_t jr, dim_t ir );
```
This function takes one integer for each loop in the level-3 operations. (**Note**: even though the function takes a `pc` argument, it will be ignored until parallelism is supported in the `KC` loop.)
So, for example, if we call
```c
bli_thread_set_ways( 2, 1, 4, 1, 1 );
```
we are requesting 2 ways of parallelism in the `JC` loop and 4 ways of parallelism in the `IC` loop.
Unlike environment variables, which only allow the user to set the parallelization strategy prior to running the executable, `bli_thread_set_ways()` may be called any time during the normal course of the BLIS-linked application's execution.

### Globally at runtime: overriding the default threading implementation

Let's assume that you configured BLIS as follows:
```
$ ./configure -t openmp,pthreads auto
```
This will result in both OpenMP and pthreads implementations being compiled and included within the BLIS library, with OpenMP serving as the default (since it was listed first to the `-t` option). You can link your program against this BLIS library and force the use of pthreads (instead of OpenMP) globally at runtime via the following API:
```c
void bli_thread_set_thread_impl( timpl_t ti );
```
The function takes a `timpl_t`, which is an enumerated type that has four valid values corresponding to the four possible threading implementations: `BLIS_OPENMP`, `BLIS_POSIX`, `BLIS_HPX`, and `BLIS_SINGLE`. Forcing use of pthreads is as simple as calling:
```c
bli_thread_set_thread_impl( BLIS_POSIX )
```
You can even disable multithreading altogether by forcing the use of the single-threaded code path:
```c
bli_thread_set_thread_impl( BLIS_SINGLE )
```
Note that if `BLIS_SINGLE` is specified, any other-related parameters previously set, such as via `bli_thread_set_num_threads()` or `bli_thread_set_ways()`, are ignored.

## Locally at runtime

***Note:** If you want to gain access to BLIS API function prototypes, be sure to #include "blis.h" from the relevant source files in your application.*

In addition to the global methods based on environment variables and runtime function calls, BLIS also offers a local, *per-call* method of requesting parallelism at runtime. This method has the benefit of being thread-safe and flexible; your application can spawn two or more threads at the application level, with each thread requesting different degrees of parallelism from their respective calls to level-3 BLIS operations.

As with environment variables and the global runtime API, there are two ways to specify parallelism: the automatic way and the manual way. Both ways involve allocating a BLIS-specific object, initializing the object and encoding the desired parallelization, and then passing a pointer to the object into one of the expert interfaces of either the [typed](docs/BLISTypedAPI.md) or [object](docs/BLISObjectAPI) APIs. We provide examples of utilizing this threading object below.

**Note**: If you set parallelization globally via environment variables and/or globally via the runtime API, and *then* specify parallelization locally on a per-call basis, the values specified locally will prevail.

**Note**: Neither way ([automatic](Multithreading.md#locally-at-runtime-the-automatic-way) nor [manual](Multithreading.md#locally-at-runtime-the-manual-way)) of specifying multithreading via the local runtime API can be used via the BLAS interfaces. The local runtime API may *only* be used via the native ([typed](docs/BLISTypedAPI.md) and [object](docs/BLISObjectAPI.md)) APIs, which are unique to BLIS. (Furthermore, the expert interfaces of each API must be used. This is demonstrated later on in this section.)

### Initializing a rntm_t

Before specifying the parallelism (automatically or manually), you must first allocate a special BLIS object called a `rntm_t` (runtime). The object is quite small (about 128 bytes), and so we recommend allocating it statically on the function stack:
```c
rntm_t rntm;
```
You **must** initialize the `rntm_t`. This can be done in either of two ways.
If you want to initialize it as part of the declaration, you may do so via the default `BLIS_RNTM_INITIALIZER` macro:
```c
rntm_t rntm = BLIS_RNTM_INITIALIZER;
```
As of this writing, BLIS treats a default-initialized `rntm_t` as a request for single-threaded execution.
If your application needs to know the ways of parallelism that were conveyed via environment variables, then there is an another way by copying the global `rntm_t` object via
```c
void bli_rntm_init_from_global( rntm_t* rntm );
```
Which may be called as:
```c
bli_rntm_init_from_global( &rntm );
```
This way is necessary when running application with multiple BLIS threads.

**Note**: If you choose to **not** initialize the `rntm_t` object and then pass it into a level-3 operation, **you will almost surely observe undefined behavior!** Please don't do this!

### Locally at runtime: the automatic way

Once your `rntm_t` is initialized, you may request automatic parallelization by encoding only the total number of threads into the `rntm_t` via the following function:
```c
void bli_rntm_set_num_threads( dim_t n_threads, rntm_t* rntm );
```
As with `bli_thread_set_num_threads()` [discussed previously](Multithreading.md#globally-at-runtime-the-automatic-way), this function takes a single integer. It also takes the address of the `rntm_t` to modify. So, for example, if (after declaring and initializing a `rntm_t` as discussed above) we call
```c
bli_rntm_set_num_threads( 6, &rntm );
```
the `rntm_t` object will be encoded to use a total of 6 threads.

### Locally at runtime: the manual way

Once your `rntm_t` is initialized, you may manually encode the ways of parallelism for each loop into the `rntm_t` by using the following function:
```c
void bli_rntm_set_ways( dim_t jc, dim_t pc, dim_t ic, dim_t jr, dim_t ir, rntm_t* rntm );
```
As with `bli_thread_set_ways()` [discussed previously](Multithreading.md#globally-at-runtime-the-manual-way), this function takes one integer for each loop in the level-3 operations. It also takes the address of the `rntm_t` to modify.
(**Note**: even though the function takes a `pc` argument, it will be ignored--and assumed to be 1--until parallelism is supported in the `KC` loop.)
So, for example, if we call
```c
bli_rntm_set_ways( 1, 1, 2, 3, 1, &rntm );
```
we are requesting two ways of parallelism in the `IC` loop and three ways of parallelism in the `JR` loop.

### Locally at runtime: overriding the default threading implementation

Let's assume that you configured BLIS as follows:
```
$ ./configure -t openmp,pthreads auto
```
This will result in both OpenMP and pthreads implementations being compiled and included within the BLIS library, with OpenMP serving as the default (since it was listed first to the `-t` option). You can link your program against this BLIS library and force the use of pthreads (instead of OpenMP) at runtime, on a per-call basis, by encoding your choice within your `rntm_t`:
```c
void bli_rntm_set_thread_impl( timpl_t ti, rntm_t* rntm );
```
The function takes a `timpl_t`, which is an enumerated type that has four valid values corresponding to the four possible threading implementations: `BLIS_OPENMP`, `BLIS_POSIX`, `BLIS_HPX`, and `BLIS_SINGLE`. Forcing use of pthreads is as simple as calling:
```c
bli_rntm_set_thread_impl( BLIS_POSIX, &rntm );
```
You can even disable multithreading altogether by forcing the use of the single-threaded code path:
```c
bli_rntm_set_thread_impl( BLIS_SINGLE, &rntm );
```
Note that if `BLIS_SINGLE` is specified, any other-related parameters previously set within the `rntm_t`, such as via `bli_rntm_set_num_threads()` or `bli_rntm_set_ways()`, are ignored.

### Locally at runtime: using the expert interfaces

Regardless of whether you specified parallelism into your `rntm_t` object via the automatic or manual method, eventually you must use the data structure when calling a BLIS operation in order for it to have any effect.

Let's assume you wish to call `gemm`. To so do, use the expert interface, which takes two additional arguments: a `cntx_t` (context) and a `rntm_t`. For the context, you may simply pass in `NULL` and BLIS will select a default context internally (which is exactly what happens for both the `cntx_t*` and `rntm_t*` parameters when you call the basic/non-expert interfaces). Here is an example of such a call:
```c
bli_gemm_ex( &alpha, &a, &b, &beta, &c, NULL, &rntm );
```
This will cause `gemm` to execute and parallelize in the manner encoded by `rntm` (and it will do so using a default `cntx_t*`).

To summarize, using a `rntm_t` involves three steps:
```c
// Declare and initialize a rntm_t object.
rntm_t rntm = BLIS_RNTM_INITIALIZER;

// Call ONE (not both) of the following to encode your parallelization into
// the rntm_t. (These are examples only--use numbers that make sense for your
// application!)
bli_rntm_set_num_threads( 6, &rntm );
bli_rntm_set_ways( 1, 1, 2, 3, 1, &rntm );

// Finally, call BLIS via an expert interface and pass in your rntm_t.
bli_gemm_ex( &alpha, &a, &b, &beta, &c, NULL, &rntm );
```
Note that `rntm_t` objects may be reused over and over again once they are initialized; there is no need to reinitialize them and re-encode their threading values!

Also, you may pass in `NULL` for the `rntm_t*` parameter of an expert interface. This causes the current global settings to be used.

# Known issues

* **Internal transposition and manual parallelism.** BLIS supports both row- and column-stored matrices (and tensor-like general storage). However, typically the `gemm` microkernel prefers to read and write microtiles of matrix C by rows, or by columns. If the storage of the user-provided matrix C does not match that of the microkernel preference, BLIS logically transpose the entire operation so that by the time the microkernel sees matrix C, it will appear to be stored according to its storage preference. If the caller is employing the automatic style of parallelism, whereby only the total number of threads is specified, this transposition happens *before* the the total number of threads is factored into the various loop-specific ways of parallelism and everything works as expected. However, if the caller employs the manual style of parallelism, the transposition must (by definition) happen *after* the thread factorization is done since, in this situation, the caller has taken responsibility for providing that factorization explicitly.

   This situation could lead to unexpectedly low multithreaded performance. Suppose the user calls `gemm` on a problem with a large m dimension and small k and n dimensions, and explicitly requests parallelism only in the IC loop, but also suppose that the storage of C does not match that of the microkernel's preference. After BLIS transposes the operation internally, the *effective* m dimension will no longer be large; instead, it will be small (because the original m and n dimension will have been swapped). The multithreaded implementation will then proceed to parallelize this small m dimension.

   There are currently no good *and* easy solutions to this problem. Eventually, though, we plan to add support for two microkernels per datatype per configuration--one for use with matrices C that are row-stored, and one for those that are column-stored. This will obviate the logic within BLIS that sometimes induces the operation transposition, and the problem will go away.

* **Thread affinity when BLIS and MKL are used together.** Some users have reported that when running a program that links both BLIS (configured with OpenMP) and MKL, **and** when OpenMP thread affinity has been specified (e.g. via `OMP_PROC_BIND` and `OMP_PLACES`), that very poor performance is observed. This may be due to incorrect thread masking, causing all threads to run on one physical core. The exact circumstances leading to this behavior have not been identified, but unsetting the OpenMP thread affinity variables appears to be a solution.

# Conclusion

Please send us feedback if you have any concerns or questions, or [open an issue](http://github.com/flame/blis/issues) if you observe any reproducible behavior that you think is erroneous. (You are welcome to use the issue feature to start any non-trivial dialogue; we don't restrict them only to bug reports!)

Thanks for your interest in BLIS.

