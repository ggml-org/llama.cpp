# custom-op

A custom operator: `saxpy`, `dst = a*x + y`.

The op is described once (see `ggml_custom_op_init()` in `ggml.h`): a list of kernels, one for each backend that can run it, by the name of the backend. Here it has the `CPU` kernel only: a function, as for `ggml_custom_4d()`.

Each kernel also has a callback that tells how it runs a node, from the node and its arguments: the number of threads for the CPU, the number and size of the workgroups for the others.

A node is made with `ggml_custom_op_4d()`. It gets the arguments and a small params struct (copied, every kernel receives it).

The scheduler gives the node to the first backend that can run it. A backend without a kernel leaves it to the CPU.

```
$ ./custom-op
saxpy computed by CPU:
[ 12.00 24.00 36.00 48.00 60.00 72.00 84.00 96.00 ]
```
