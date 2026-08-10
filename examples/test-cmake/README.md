## cmake-test

This is just for manually testing/developing of a llama.cpp installation to
enable troubleshooting issues and exploration. The idea is that this can be used
after making changes to llama.cpp installation cmake configuration and then
verify it locally.

### Usage
The following will configure, build, and install llama.cpp

Configuring/build/install:
```console
./build-install.sh
```
The above command will create a directory named `install` in the current directory
which will have the follwing files in its lib directory:
```console
(venv) $ ls install/lib/
cmake                   libggml.so          libllama-common.so.0      libllama.so.0.1.0  llama.cpp
libggml-base.so         libggml.so.0        libllama-common.so.0.1.0  libmtmd.so         pkgconfig
libggml-base.so.0       libggml.so.0.19.0   libllama.so               libmtmd.so.0
libggml-base.so.0.19.0  libllama-common.so  libllama.so.0             libmtmd.so.0.1.0

```

Build/run this project using the installation created above:
```console
(venv) $ ./build.sh
-- Configuring done (0.0s)
-- Generating done (0.0s)
-- Build files have been written to: /home/danbev/work/ai/llama.cpp/examples/test-cmake/build
[100%] Built target test-cmake
[test-cmake] Using llama.cpp version 0.1.0-dev-b10335
[test-cmake] Initializing backend...
load_backend: loaded CPU backend from /home/danbev/work/ai/llama.cpp/examples/test-cmake/install/lib/llama.cpp/libggml-cpu-alderlake.so
[test-cmake] Backend initialized.
```
Notice that the llama.cpp version is currently `0.1.0-dev-b10335`.

### Nightly builds
Currently nightly/daily builds use build number if the format `b10333`. This will
change if the changes in here are accepted and the build would be named in the
following format: `0.1.0-dev-b10335`.

### Releases
Releases should now be able to create a tag with the target version, using the
example above `0.1.0-dev-b10335` was the pre-release version so we would tag
the real release with 0.1.0.

After this we need to dump the version in CMakeLists.txt so that the pre-release
is for the next version, for example `0.2.0`. TODO: need to figure this out and
if this can be automated so that there are no builds that get created in the
time between the actual release and a new nightly/daily build.

_wip_
