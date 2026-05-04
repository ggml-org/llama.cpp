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
    mthreads/musa:rc4.3.0-devel-ubuntu22.04-amd64
```

Inside the container, execute the following commands:

```bash
apt update -y && apt install -y bc cmake ccache git python3.10-venv time unzip wget
git config --global --add safe.directory /ws
GG_BUILD_MUSA=1 bash ./ci/run.sh /ci-results /ci-cache
```

This setup ensures that the CI runs within an isolated Docker environment while maintaining cached files and results across runs.
