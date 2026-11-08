# GGML-VirtGPU Backend Configuration

This document describes the environment variables used by the ggml-virtgpu backend system, covering both the frontend (guest-side) and backend (host-side) components.

## Environment Variables Overview

The ggml-virtgpu backend uses environment variables for configuration across three main components:
- **Frontend (Guest)**: GGML applications running in VMs
- **Hypervisor**: Virglrenderer/APIR system
- **Backend (Host)**: Host-side GGML backend integration

## Frontend (Guest-side) Configuration

### GGML_REMOTING_USE_APIR_CAPSET
- **Location**: `ggml/src/ggml-virtgpu/virtgpu.cpp`
- **Type**: Boolean flag (presence-based)
- **Purpose**: Controls which virtio-gpu capability set to use for communication
- **Values**:
  - Set (any value): Use the APIR capset (long-term setup)
  - Unset: Use the Venus capset (easier for testing with an unmodified hypervisor)
- **Default**: Unset (Venus capset)
- **Usage**:
  ```bash
  export GGML_REMOTING_USE_APIR_CAPSET=1  # Use APIR capset
  # or leave unset for Venus capset
  ```

## Hypervisor (Virglrenderer/APIR) Configuration

These environment variables are used during the transition phase for
running with an unmodified hypervisor (not supporting the
VirglRenderer APIR component). They will be removed in the future, and
the hypervisor will instead configure VirglRenderer with the APIR
_Configuration Key_.

### VIRGL_APIR_BACKEND_LIBRARY
- **Location**: `virglrenderer/src/apir/apir-context.c`
- **Configuration Key**: `apir.load_library.path`
- **Type**: File path string
- **Purpose**: Path to the APIR backend library that virglrenderer should dynamically load
- **Required**: Yes
- **Example**:
  ```bash
  export VIRGL_APIR_BACKEND_LIBRARY="/path/to/libggml-remotingbackend.so"
  ```

### VIRGL_ROUTE_VENUS_TO_APIR
- **Location**: `virglrenderer/src/apir/apir-renderer.h`
- **Type**: Boolean flag (presence-based)
- **Purpose**: Temporary workaround to route Venus capset calls to APIR during hypervisor transition period
- **Status**: will be removed once hypervisors support APIR natively
- **Warning**: Breaks normal Vulkan/Venus functionality
- **Usage**:
  ```bash
  export VIRGL_ROUTE_VENUS_TO_APIR=1  # For testing with an unmodified hypervisor
  ```

### VIRGL_APIR_LOG_TO_FILE
- **Location**: `virglrenderer/src/apir/apir-renderer.c`
- **Environment Variable**: `VIRGL_APIR_LOG_TO_FILE`
- **Type**: File path string
- **Purpose**: Enable debug logging from the VirglRenderer APIR component to specified file
- **Required**: No (optional debugging)
- **Default**: Logging to `stderr`
- **Usage**:
  ```bash
  export VIRGL_APIR_LOG_TO_FILE="/tmp/apir-debug.log"
  ```

## Backend (Host-side) Configuration

These environment variables are used during the transition phase for
running with an unmodified hypervisor (not supporting the
VirglRenderer APIR component). They will be removed in the future, and
the hypervisor will instead configure VirglRenderer with the APIR
_Configuration Key_.

### APIR_LLAMA_CPP_GGML_LIBRARY_PATH
- **Location**: `ggml/src/ggml-virtgpu/backend/backend.cpp`
- **Environment Variable**: `APIR_LLAMA_CPP_GGML_LIBRARY_PATH`
- **Configuration Key**: `ggml.library.path`
- **Type**: File path string
- **Purpose**: Path to the actual GGML backend library (Metal, CUDA, Vulkan, etc.)
- **Required**: **Yes** - backend initialization fails without this
- **Examples**:
  ```bash
  # macOS with Metal backend
  export APIR_LLAMA_CPP_GGML_LIBRARY_PATH="/opt/llama.cpp/lib/libggml-metal.dylib"

  # Linux with CUDA backend
  export APIR_LLAMA_CPP_GGML_LIBRARY_PATH="/opt/llama.cpp/lib/libggml-cuda.so"

  # macOS or Linux with Vulkan backend
  export APIR_LLAMA_CPP_GGML_LIBRARY_PATH="/opt/llama.cpp/lib/libggml-vulkan.so"
  ```

### APIR_LLAMA_CPP_GGML_LIBRARY_REG
- **Location**: `ggml/src/ggml-virtgpu/backend/backend.cpp`
- **Environment Variable**: `APIR_LLAMA_CPP_GGML_LIBRARY_REG`
- **Configuration Key**: `ggml.library.reg`
- **Type**: Function symbol name string
- **Purpose**: Name of the backend registration function to call after loading the library
- **Required**: No (defaults to `ggml_backend_init`)
- **Default**: `ggml_backend_init`
- **Examples**:
  ```bash
  # Metal backend
  export APIR_LLAMA_CPP_GGML_LIBRARY_REG="ggml_backend_metal_reg"

  # CUDA backend
  export APIR_LLAMA_CPP_GGML_LIBRARY_REG="ggml_backend_cuda_reg"

  # Vulkan backend
  export APIR_LLAMA_CPP_GGML_LIBRARY_REG="ggml_backend_vulkan_reg"

  # Generic fallback (default)
  # export APIR_LLAMA_CPP_GGML_LIBRARY_REG="ggml_backend_init"
  ```

### APIR_LLAMA_CPP_LOG_TO_FILE
- **Location**: `ggml/src/ggml-virtgpu/backend/backend.cpp:62`
- **Environment Variable**: `APIR_LLAMA_CPP_LOG_TO_FILE`
- **Type**: File path string
- **Purpose**: Enable debug logging from the GGML backend to specified file
- **Required**: No (optional debugging)
- **Usage**:
  ```bash
  export APIR_LLAMA_CPP_LOG_TO_FILE="/tmp/ggml-backend-debug.log"
  ```

## Configuration Flow

The configuration system works as follows:

1. **Hypervisor Setup**: Virglrenderer loads the APIR backend library specified by `VIRGL_APIR_BACKEND_LIBRARY`

2. **Context Creation**: When an APIR context is created, it populates a configuration table with environment variables:
   - `apir.load_library.path` ← `VIRGL_APIR_BACKEND_LIBRARY`
   - `ggml.library.path` ← `APIR_LLAMA_CPP_GGML_LIBRARY_PATH`
   - `ggml.library.reg` ← `APIR_LLAMA_CPP_GGML_LIBRARY_REG`
   - this step will eventually be performed by the hypervisor itself, with command-line arguments instead of environment variables.

3. **Backend Initialization**: The backend queries the configuration via callbacks:
   - `virgl_cbs->get_config(ctx_id, "ggml.library.path")` returns the library path
   - `virgl_cbs->get_config(ctx_id, "ggml.library.reg")` returns the registration function

4. **Library Loading**: The backend dynamically loads and initializes the specified GGML library

## Error Messages

Common error scenarios and their messages:

- **Missing library path**: `"cannot open the GGML library: env var 'APIR_LLAMA_CPP_GGML_LIBRARY_PATH' not defined"`
- **Missing registration function**: `"cannot register the GGML library: env var 'APIR_LLAMA_CPP_GGML_LIBRARY_REG' not defined"`

## Example Complete Configuration

Here's an example configuration for a macOS host with Metal backend:

```bash
# Hypervisor environment
export VIRGL_APIR_BACKEND_LIBRARY="/opt/llama.cpp/lib/libggml-virtgpu-backend.dylib"

# Backend configuration
export APIR_LLAMA_CPP_GGML_LIBRARY_PATH="/opt/llama.cpp/lib/libggml-metal.dylib"
export APIR_LLAMA_CPP_GGML_LIBRARY_REG="ggml_backend_metal_reg"

# Optional logging
export VIRGL_APIR_LOG_TO_FILE="/tmp/apir.log"
export APIR_LLAMA_CPP_LOG_TO_FILE="/tmp/ggml.log"

# Guest configuration
export GGML_REMOTING_USE_APIR_CAPSET=1
```
