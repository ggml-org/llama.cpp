## Contents

* **[Introduction](Addons.md#introduction)**
* **[Enabling addons](Addons.md#enabling-addons)**
* **[Addon rules](Addons.md#addon-rules)**
* **[Caveats](Addons.md#caveats)**
* **[Known issues](Addons.md#known-issues)**
* **[Conclusion](Addons.md#conclusion)**


## Introduction

This file briefly describes the requirements for enabling or creating a
custom BLIS *addon*.

Simply put, an addon in BLIS provides additional APIs, operations, and/or
implementations that may be useful to certain users. An addon can be
thought of as a standalone extension of BLIS that does not depend on any
other addon, although addons may utilize existing functionality or kernels
within the core framework.

By definition, an addon should *never* provide APIs that conflict with
the interfaces that belong to either the [typed API](BLISTypedAPI.md) or the
[object API](BLISObjectAPI.md). Thus, you'll never have to worry about a
properly constructed (and properly functioning) addon interfering with or
otherwise changing core BLIS functionality.

How does an addon differ from a [sandbox](Sandboxes.md)? Great question!
Sometimes you want to include additional BLIS-like functionality that does
not relate directly to `gemm` or any other BLIS operation.
(By contrast, a sandbox requires you to implement `gemm` whether you want
to or not.)
Furthermore, you may wish to enable multiple addons simultaneously.
(By contrast, only one sandbox may be enabled at a time.)
Thus, the addon feature provides additional flexibility to some
users in a way that sandboxes cannot, while still providing many of the
conveniences of sandboxes.

## Enabling an addon

To enable an existing addon at configure-time, you simply specify it as an
option to `configure`. Either of the following usages are accepted:
```
$ ./configure --enable-addon=foobar auto
$ ./configure -a foobar auto
```
Here, we tell `configure` that we want to use the `foobar` addon, which
corresponds to a subdirectory of the `addon` directory named `foobar`.
(Reminder: the `auto` argument is the configuration target and
unrelated to addons.)

You may also enable multiple addons within the same build of BLIS:
```
$ ./configure -a foobar -a thing1 -a thing2 auto
```
Note that the default behavior of `configure` is that no addons are enabled.

As `configure` runs, you should get output that includes lines
similar to:
```
configure: configuring with addons:
configure:   addon/foobar
configure:   addon/thing1
configure:   addon/thing2
```
And when you build BLIS, the addon source code will be among the last files to
be compiled:
```
Compiling obj/haswell/addon/foobar/foobar.o ('haswell' CFLAGS for addons)
Compiling obj/haswell/addon/thing1/thing1.o ('haswell' CFLAGS for addons)
Compiling obj/haswell/addon/thing1/thing1_api.o ('haswell' CFLAGS for addons)
Compiling obj/haswell/addon/thing2/thing2_api.o ('haswell' CFLAGS for addons)
...
```
That's it! After the BLIS library is built, it will contain your chosen
addons. You can always confirm this by using `nm` to confirm the presence
of your API symbols:
```
$ nm lib/haswell/libblis.a | grep foobar
foobar.o:
0000000000000000 T foobar
```

## Addon rules

Please follow these guidelines for the best developer experience when
creating addons.

1. As with sandboxes, you don't need to worry about creating makefiles. The
BLIS build system will take care of this for you. :) By configuring BLIS with
an addon enabled, `make` will scan your addon subdirectory and compile
all of its source code using similar compilation rules as were used for the rest
of the framework. In addition, the compilation command line will automatically
contain one `-I<includepath>` option for every subdirectory in your addon,
so it doesn't matter where in your addon directory hierarchy you place your
header files -- they will be found!

2. We recommend that you write your addon in C99. While you *may* use C++11
to implement your addon, you should provide a C99 wrapper API to your
implementation so that others can interface with it. There is no guarantee
that the end-user will be using a C++11 compiler, and therefore you should
limit the definitions in your addon header to those that are C99 compliant.
If you write your addon in C++11, you must use one of the BLIS-approved file
extensions for your source files (`.cc`, `.cpp`, `.cxx`) and your local
header files (`.hh`, `.hpp`, `.hxx`).
Note that `blis.h` already contains all of its definitions inside of an
`extern "C"` block, so you should be able to `#include "blis.h"` from your
C++11 source code without any issues.

3. All of your code related to the addon should reside within the named
addon directory, or some subdirectory therein. If your addon requires
new kernels, you should add kernel source code to an appropriate
microarchitecture-specific subdirectory within the top-level `kernels`
directory so that they are compiled with the correct
microarchitecture-specific optimization flags.

4. If your addon is named `foobar`, the BLIS build system will expect to
find a header called `foobar.h` somewhere in the `addon/foobar` directory
(or one of its subdirectories). This `foobar.h` header will automatically
be inlined into the monolithic `blis.h` header that is produced by the
BLIS build system. `foobar.h` may `#include` other local headers, each of
which will also (recursively) get inlined into `blis.h`. However, you may
choose to omit some local addon headers from `foobar.h.` You might do this,
for example, because those headers define things that are not needed in
order for the end user to call your addon code.

5. Your addon APIs will always be available within static library builds of
BLIS, but if you want your addon APIs to be exported as public APIs within
*shared* library builds of BLIS, you'll need to annotate the prototypes
accordingly. (BLIS makes its shared library symbols private by default; this
allows us to export only those functions that we consider to be part of the
public APIs.) This annotation can be done by prefixing function prototypes
with the `BLIS_EXPORT_ADDON` macro as follows:
```c
BLIS_EXPORT_ADDON void foobar_calc( void* a, void* b );
```

6. Do not define any symbols in your addon that conflict with any symbols within
the core framework. For example, don't define a function called `bli_copym()`
in your addon since that function is already defined within BLIS.

7. Do not define any symbols in your addon that conflict with any symbols within
the C99 standard libraries/headers. For example, don't define a function called
`printf()` since that function is already defined within the C99 standard library.

8. *Try* to not define any symbols in your addon that conflict with symbols in any
other addon, unless your addon is meant to serve as an alternative to the
conflicting addon, in which case conflicting symbol names is okay (since you
will presumably never build with both addons enabled).

9. When choosing names for your addon files, avoid source filenames that already
exist within BLIS. For example, don't name one of your files `bli_obj.c`
since that file would compile into `bli_obj.o`, which will have already been
placed into the library by the build system.

10. Similarly, avoid header filenames that already exist within BLIS or C99.
For example, don't name one of your header files `bli_obj.h` since that file
already exists in BLIS. Also, don't name one of your header files `math.h`
since that name would conflict with the `math.h` defined by C99. (This also
means you shouldn't name your addon `math` since normally that name would
require that you provide a `math.h` header inside the addon directory.)

If you follow these rules, you will be much more likely to have a pleasant
experience integrating your BLIS addon into the larger framework.

## Caveats

Notice that the BLIS addons are limited in what they can accomplish. Generally
speaking, addons cannot change existing implementations within BLIS. Instead,
addons aim to provide a way to quickly augment BLIS with additional bundles of
code that extend BLIS's set of functionality in some interesting way. If you
want to define new BLAS-like functions, but don't know where to start, creating
a new addon is an appropriate place to start experimenting. If you want to
change or refactor existing BLIS code, an addon is probably not suited for your
needs.

Another important limitation is the fact that the build system currently uses
"framework `CFLAGS`" when compiling the addon source files. These are the same
`CFLAGS` used when compiling general framework source code,
```
# Example framework CFLAGS used by 'haswell' sub-configuration
-O2 -Wall -Wno-unused-function -Wfatal-errors -fPIC -std=c99
-D_POSIX_C_SOURCE=200112L -Iinclude/haswell -I./frame/3/
-I./frame/1m/ -I./frame/1f/ -I./frame/1/ -I./frame/include
-DBLIS_VERSION_STRING=\"0.8.1-195\" -fvisibility=hidden
```
which are likely more general-purpose than the `CFLAGS` used for, say,
optimized kernels or even reference kernels:
```
# Example optimized kernel CFLAGS used by 'haswell' sub-configuration
-O3 -fomit-frame-pointer -mavx2 -mfma -mfpmath=sse -march=haswell -Wall
-Wno-unused-function -Wfatal-errors -fPIC -std=c99 -D_POSIX_C_SOURCE=200112L
-Iinclude/haswell -I./frame/3/ -I./frame/1m/ -I./frame/1f/ -I./frame/1/
-I./frame/include -DBLIS_VERSION_STRING=\"0.8.1-195\" -fvisibility=hidden
```
(To see precisely which flags are being employed for any given file, enable
verbosity at compile-time via `make V=1`.) Compiling addons with these more
versatile `CFLAGS` compiler options means that we only need to compile one
instance of each addon source file, even when targeting multiple
configurations (for example, via `./configure x86_64`). However, it also means
that addons are not ideal for microkernels, as they sometimes need additional
compiler flags in order to
yield the highest performance. If you have a new microkernel you would like to
use within an addon, you can always develop it within that addon. However,
once it is stable and ready for use by others, it's best to move the kernel(s)
to the appropriate microarchitecture-specific subdirectory of the `kernels`
directory the kernel(s). This will allow the kernel to be compiled with the
appropriate microarchitecture-specific compiler flags.
Please see the
[Configuration Guide](ConfigurationHowTo)
for more details, and when in doubt, please don't be shy about seeking
guidance from BLIS developers by opening a
[new issue](https://github.com/flame/blis/issues) or sending a message to the
[blis-devel](http://groups.google.com/d/forum/blis-devel) mailing list.

Notwithstanding these limitations, hopefully you still find BLIS addons
useful!

## Known issues

* None yet.

## Conclusion

If you encounter any problems, please open
a new [issue on GitHub](https://github.com/flame/blis/issues).

If you are unsure about how something works, you can still open an issue. Or, you
can send a message to
[blis-devel](https://groups.google.com/d/forum/blis-devel) mailing list.

