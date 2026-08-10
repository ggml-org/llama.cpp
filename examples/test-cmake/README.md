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
Notice that the llama.cpp version is currently `0.1.0-dev-b10335`. And also note
that the cpu backend module is not using a semver. I was not sure about this a
it these are used as plugins and we look for `.so` files in this case. But this
migth be an issue if we have multiple versions in the same directory as a new
version would overwrite the old file.

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

```console
$ wget https://github.com/danbev/llama.cpp/releases/download/b10339/llama-b10339-bin-ubuntu-x64.tar.gz
$ mkdir tmp; cd tmp
$ mv ../llama-b10339-bin-ubuntu-x64.tar.gz .
$ tar xvf llama-b10339-bin-ubuntu-x64.tar.gz

(venv) $ ls llama-b10339/libllama.*
llama-b10339/libllama.so  llama-b10339/libllama.so.0  llama-b10339/libllama.so.0.1.0

venv) $ ./llama-b10339/llama-cli --version
version: 0.1.0-dev-b10339 (build 10339, commit c8fe5edf5)
built with GNU 11.4.0 for Linux x86_64

(venv) $ ./llama-b10339/llama-server --version
version: 0.1.0-dev-b10339 (build 10339, commit c8fe5edf5)
built with GNU 11.4.0 for Linux x86_64

```
As mentioned about the ggml-cpu backeds are not using semantic versioning:
```console
(venv) $ ls llama-b10339/libggml-cpu-*
llama-b10339/libggml-cpu-alderlake.so    llama-b10339/libggml-cpu-piledriver.so
llama-b10339/libggml-cpu-cannonlake.so   llama-b10339/libggml-cpu-sandybridge.so
llama-b10339/libggml-cpu-cascadelake.so  llama-b10339/libggml-cpu-sapphirerapids.so
llama-b10339/libggml-cpu-cooperlake.so   llama-b10339/libggml-cpu-skylakex.so
llama-b10339/libggml-cpu-haswell.so      llama-b10339/libggml-cpu-sse42.so
llama-b10339/libggml-cpu-icelake.so      llama-b10339/libggml-cpu-x64.so
llama-b10339/libggml-cpu-ivybridge.so    llama-b10339/libggml-cpu-zen4.so
```
If we were to install a new version, these backends would be overwritten which
will not work. We need a solution for this and I think it would be most consistent
to use the semver for them as well. An alternative would be to place them in
a versioned directory but I'm not sure how well that would work for projects
integrating ggml into their projects. The might be building with a specific
ggml backends directory (GGML_BACKEND_DIR) and I don't think we can force this
upon them. I'll look into this but in the ggml repo as that is where the semver
changes for ggml are taking place. 


_wip_
