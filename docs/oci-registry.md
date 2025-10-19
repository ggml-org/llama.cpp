# OCI/Docker Registry Integration

llama.cpp supports pulling models directly from OCI-compliant registries such as Docker Hub. This feature uses the [go-containerregistry](https://github.com/google/go-containerregistry) library to handle registry authentication and image pulling.

## Features

- Pull GGUF models from Docker Hub and other OCI registries
- Automatic authentication using Docker credentials (via `docker login`)
- Support for private registries with authentication
- Caching of downloaded models
- **Docker-style progress bars** showing download progress, speed, and ETA
- **Resumable downloads** - interrupted downloads can be resumed automatically

## Prerequisites

- Go 1.24 or later (for building from source)
- Docker credentials configured (for private registries)

## Usage

### Pulling Public Models

To pull a public model from Docker Hub:

```bash
./llama-cli --docker-repo ai/smollm2:135M-Q4_0
```

By default, models are pulled from the `ai/` namespace on Docker Hub. If no namespace is specified, `ai/` is assumed:

```bash
# These are equivalent:
./llama-cli --docker-repo gemma3
./llama-cli --docker-repo ai/gemma3
```

### Pulling Private Models

For private models or registries requiring authentication, first authenticate using Docker:

```bash
docker login
# Or for a specific registry:
docker login registry.example.com
```

Then pull the model:

```bash
./llama-cli --docker-repo myuser/private-model:Q4_K_M
```

### Custom Registries

You can also pull from custom OCI registries by specifying the full registry URL:

```bash
./llama-cli --docker-repo registry.example.com/namespace/model:tag
```

## How It Works

1. The `--docker-repo` (or `-dr`) flag specifies the OCI image reference
2. llama.cpp uses the Go-based OCI library to:
   - Parse the image reference
   - Authenticate using Docker credentials (if available)
   - Fetch the manifest from the registry
   - Identify and download the GGUF layer with progress tracking
   - Display docker-style progress bars during download
3. The model is cached locally for future use
4. If a download is interrupted, it will automatically resume from where it left off

### Progress Display

During download, you'll see progress information similar to Docker:

```
1234567890ab: Downloading [===================>              ] 39.0% (39.00 MB / 100.00 MB) 20.10 MB/s
```

- **1234567890ab**: Short digest of the layer being downloaded
- **Progress bar**: Visual representation of download progress
- **Percentage**: Completion percentage
- **Size**: Downloaded size / Total size in MB
- **Speed**: Current download speed in MB/s

### Resumable Downloads

If a download is interrupted (e.g., network failure, Ctrl+C), the next download attempt will automatically resume:

```
1234567890ab: Resuming download from 39.00 MB
1234567890ab: Downloading [===================>              ] 41.0% (41.00 MB / 100.00 MB) 18.50 MB/s
```

The download will continue from where it stopped, saving time and bandwidth. The integrity of resumed downloads is verified using layer digests.

## Image Format

Models must be packaged as OCI images with a GGUF layer. The layer should have one of these media types:
- `application/vnd.docker.ai.gguf.v3`
- Any media type containing "gguf"

## Authentication

Authentication is handled automatically using the same credentials as the Docker CLI:
- Credentials are stored in `~/.docker/config.json`
- Use `docker login` to authenticate
- Supports credential helpers and authentication providers

## Caching

Downloaded models are cached in the standard llama.cpp cache directory:
- Linux/macOS: `~/.cache/llama.cpp/`
- Windows: `%LOCALAPPDATA%\llama.cpp\`

Cached models are verified using their digest to ensure integrity. If the cached file matches the expected digest, it will be used instead of re-downloading.

### Partial Downloads

Partial downloads are stored with a `.tmp` extension alongside a `.digest` file for verification. If a download is interrupted:
1. The partial file and digest are preserved
2. On the next attempt, if the digest matches, download resumes
3. If the digest differs (e.g., model was updated), a fresh download starts

## Building with OCI Support

OCI support is automatically enabled if Go is available during build:

```bash
cmake -B build
cmake --build build
```

If Go is not found, a warning will be displayed and OCI functionality will be unavailable.

## Troubleshooting

### Authentication Issues

If you encounter authentication errors:
1. Ensure you're logged in: `docker login`
2. Verify credentials: Check `~/.docker/config.json`
3. For private registries, specify the full registry URL

### Network Issues

If downloads fail or are interrupted:
1. Check your internet connection
2. Verify the registry is accessible
3. Try pulling a test image with Docker: `docker pull <image>`
4. The download will automatically resume on retry if the partial download is valid

Note: Progress bars require a TTY. If running in a non-interactive environment (e.g., CI/CD), progress information will be minimal.

### Build Issues

If OCI support is not available:
1. Ensure Go 1.24 or later is installed: `go version`
2. Rebuild the project: `cmake --build build --clean-first`
3. Check CMake output for Go-related warnings
