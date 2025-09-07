# CI

In addition to [Github Actions](https://github.com/ggml-org/llama.cpp/actions) `llama.cpp` uses a custom CI framework:

https://github.com/ggml-org/ci

It monitors the `master` branch for new commits and runs the
[ci/run.sh](https://github.com/ggml-org/llama.cpp/blob/master/ci/run.sh) script on dedicated cloud instances. This allows us
to execute heavier workloads compared to just using Github Actions. Also with time, the cloud instances will be scaled
to cover various hardware architectures, including GPU and Apple Silicon instances.

Collaborators can optionally trigger the CI run by adding the `ggml-ci` keyword to their commit message.
Only the branches of this repo are monitored for this keyword.

It is a good practice, before publishing changes to execute the full CI locally on your machine:

```bash
mkdir tmp

# CPU-only build
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with CUDA support
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with SYCL support
source /opt/intel/oneapi/setvars.sh
GG_BUILD_SYCL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with MUSA support
GG_BUILD_MUSA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

## Running MUSA CI in a Docker Container

Assuming `$PWD` is the root of the `llama.cpp` repository, follow these steps to set up and run MUSA CI in a Docker container:

### 1. Create a local directory to store cached models, configuration files and venv:

```bash
mkdir -p $HOME/llama.cpp/ci-cache
```

### 2. Create a local directory to store CI run results:

```bash
mkdir -p $HOME/llama.cpp/ci-results
```

### 3. Start a Docker container and run the CI:

```bash
docker run --privileged -it \
    -v $HOME/llama.cpp/ci-cache:/ci-cache \
    -v $HOME/llama.cpp/ci-results:/ci-results \
    -v $PWD:/ws -w /ws \
    mthreads/musa:rc4.2.0-devel-ubuntu22.04-amd64
```

Inside the container, execute the following commands:

```bash
apt update -y && apt install -y bc cmake ccache git python3.10-venv time unzip wget
git config --global --add safe.directory /ws
GG_BUILD_MUSA=1 bash ./ci/run.sh /ci-results /ci-cache
```

This setup ensures that the CI runs within an isolated Docker environment while maintaining cached files and results across runs.
