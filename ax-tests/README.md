# Axelera Backend Tests

This directory contains tests for the Axelera AI accelerator backend.

## Files

| File | Description |
|------|-------------|
| `test-backend-axelera.cpp` | Backend functionality test - creates graph, tests compilation and execution |
| `CMakeLists.txt` | Build configuration |
| `build-and-run.sh` | Convenience script for building and running tests |

## Quick Start

### Building

```bash
# From project root build directory
cd /Users/ikolt/Projects/llama.cpp/build
cmake .. -DGGML_AXELERA=ON
cmake --build . --target test-backend-axelera -j8

# Or use convenience script
cd /Users/ikolt/Projects/llama.cpp/ax-tests
./build-and-run.sh
```

### Running Tests

**Backend Test:**
```bash
cd /Users/ikolt/Projects/llama.cpp/build

# Basic run
./bin/test-backend-axelera

# With debug logging
GGML_LOG_LEVEL=info ./bin/test-backend-axelera

# With detailed trace logging
GGML_LOG_LEVEL=debug ./bin/test-backend-axelera
```

Expected output:
- Backend initialization: "Initializing Axelera backend registry with 1 devices"
- Graph structure printing with operations and tensor details
- Compilation pipeline execution (or cache hit on subsequent runs)
- PyTorch code generation in `/tmp/axelera_work/graph_<hash>.py`
- Execution status (SUCCESS with CPU fallback if PyTorch not installed)

## How It Works

The Axelera backend uses a compiler-based execution model:

```
User calls llama.cpp
    ↓
graph_compute() - Executes graph
    ↓
graph_plan_create() - Compiles graph (if needed)
    ↓
  Hash graph → Check cache (/tmp/axelera_cache/<hash>.axe)
    ↓
  If cache MISS:
    • Serialize GGML graph to PyTorch code
    • Generate Python script in /tmp/axelera_work/
    • Execute: python3 script.py → TorchScript (.pt)
    • Compile: axelera-compiler → binary (.axe)
    • Cache compiled binary
    ↓
  If cache HIT:
    • Load pre-compiled binary
    ↓
graph_plan_compute() - Execute on Axelera hardware
    ↓
graph_plan_free() - Cleanup resources
```

## What the Test Does

**test-backend-axelera** performs the following:

1. **Creates a computation graph** with matrix multiplication (MUL_MAT)
   - Matrix A: [4, 3]
   - Matrix B: [4, 2]
   - Result C: [3, 2]

2. **Prints detailed graph information:**
   ```
   === Axelera Graph Compute (Debug) ===
   Node 0: result_C
     Operation: MUL_MAT
     Type: f32
     Shape: [3, 2, 1, 1]
     Src[0]: matrix_A, type=f32, shape=[4, 3, 1, 1]
     Src[1]: matrix_B, type=f32, shape=[4, 2, 1, 1]
   ```

3. **Triggers compilation pipeline:**
   - Generates graph hash (e.g., `38833c27c5713f4e`)
   - Serializes to PyTorch with proper operation translation
   - Creates executable Python script
   - Attempts compilation (gracefully handles missing PyTorch)

4. **Demonstrates caching:**
   - First run: Cache MISS → full compilation
   - Second run: Cache HIT → instant load

5. **Shows execution flow:**
   - Backend initialization with tracing
   - Graph computation with detailed logging
   - Resource cleanup

**Use for:**
- Verifying backend integration
- Understanding GGML operation flow
- Testing compilation pipeline
- Debugging graph transformations

## Generated Files

When you run the test, several files are created:

### Temporary Work Directory (`/tmp/axelera_work/`)
- `graph_<hash>.py` - Generated PyTorch model code
- `ggml_model.pt` - TorchScript binary (if PyTorch installed)
- `graph_<hash>.axe` - Compiled Axelera binary (placeholder)

### Cache Directory (`/tmp/axelera_cache/`)
- `<hash>.axe` - Cached compiled models for fast loading

You can inspect these files to see the PyTorch code generation:
```bash
cat /tmp/axelera_work/graph_*.py
```

## PyTorch Serialization

The backend includes a full PyTorch serializer (integrated in `ggml-axelera-compiler.cpp`) that supports:

### Operations
- **Matrix operations:** MUL_MAT → `torch.matmul()`
- **Arithmetic:** MUL, ADD, SUB, DIV, SQRT, SQR
- **Normalization:** SOFT_MAX, NORM, RMS_NORM
- **Transformations:** RESHAPE, VIEW, PERMUTE, TRANSPOSE, CONT
- **Advanced:** ROPE (rotary position embedding)

### Features
- Graph analysis (identifies inputs vs parameters)
- Type conversion (F32, F16, BF16, I32, I16, I8)
- Name sanitization for Python compatibility
- ONNX export support
- Proper tensor shape handling

### Example Output

For the test's MUL_MAT operation, the serializer generates:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class GGMLModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, matrix_A: torch.Tensor, matrix_B: torch.Tensor):
        # MUL_MAT
        result_C = torch.matmul(matrix_A, matrix_B)
        return result_C

# Create model instance
model = GGMLModel()
model.eval()

# Export to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save('ggml_model.pt')
```

## Environment Variables

### Build Time
- `GGML_AXELERA=ON` - Enable Axelera backend in CMake
- `AXELERA_SDK_PATH` - Path to Axelera SDK (optional)

### Runtime
- `GGML_LOG_LEVEL` - Set to `info` or `debug` for detailed logging
- `AXELERA_COMPILER` - Path to Axelera compiler executable (default: `axelera-compiler`)
- `AXELERA_CACHE_DIR` - Cache directory for compiled graphs (default: `/tmp/axelera_cache`)

## Debug Tracing

The backend includes comprehensive [TRACE] logging. Enable with:
```bash
GGML_LOG_LEVEL=debug ./bin/test-backend-axelera
```

You'll see:
- `[TRACE] backend.get_name(...)` - Backend identification
- `[TRACE] backend.graph_compute(...)` - Graph execution entry
- `[TRACE] graph_plan_create(...)` - Compilation pipeline
- `[TRACE] graph_plan_compute(...)` - Execution on hardware
- `[TRACE] graph_plan_free(...)` - Resource cleanup

Each trace includes parameters and return values for debugging.

## Testing with PyTorch

To test the full compilation pipeline with PyTorch:

1. **Install PyTorch:**
   ```bash
   pip3 install torch
   ```

2. **Run the test:**
   ```bash
   GGML_LOG_LEVEL=info ./bin/test-backend-axelera
   ```

3. **Verify PyTorch model creation:**
   ```bash
   ls -lh /tmp/axelera_work/
   # Should show: graph_<hash>.py, ggml_model.pt
   ```

4. **Test the generated model:**
   ```python
   import torch
   model = torch.jit.load('/tmp/axelera_work/ggml_model.pt')
   # Test with dummy inputs
   result = model(torch.randn(4, 3), torch.randn(4, 2))
   print(result.shape)  # Should be [3, 2]
   ```

## Current Status

✅ **Working:**
- Backend registration and initialization
- Graph debugging output
- PyTorch code generation (15+ operations)
- Compilation pipeline (Python script generation)
- Caching system
- Debug tracing
- CPU fallback

⏳ **Pending:**
- Axelera SDK integration (placeholders ready)
- Hardware execution (returns success for testing)
- PyTorch installation (needed for .pt generation)
- Axelera compiler integration (needs actual compiler)

## Documentation

For more details, see:
- `README_AXELERA.md` - Complete backend overview and user guide
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `NOTES.md` - Quick build and test instructions

## Common Workflows

### Testing Backend Integration
```bash
cd /Users/ikolt/Projects/llama.cpp/build
cmake .. -DGGML_AXELERA=ON
cmake --build . -j8
GGML_LOG_LEVEL=info ./bin/test-backend-axelera
```

### Inspecting Generated Code
```bash
# Run test to generate code
./bin/test-backend-axelera

# View the generated PyTorch model
cat /tmp/axelera_work/graph_*.py
```

### Testing Cache System
```bash
# First run - compiles
./bin/test-backend-axelera
# Output: Cache MISS

# Second run - uses cache
./bin/test-backend-axelera
# Output: Cache HIT
```

### Debugging Compilation Issues
```bash
# Enable debug logging
GGML_LOG_LEVEL=debug ./bin/test-backend-axelera 2>&1 | tee debug.log

# Check what's being generated
cat /tmp/axelera_work/graph_*.py

# Test Python script manually
cd /tmp/axelera_work
python3 graph_*.py
```

## Support

If you encounter issues:
1. Check that `GGML_AXELERA=ON` was set during CMake configuration
2. Verify the backend loads: `GGML_LOG_LEVEL=info ./bin/llama-cli --version`
3. Check for compilation errors in the build output
4. Review generated code in `/tmp/axelera_work/`
5. Enable debug tracing: `GGML_LOG_LEVEL=debug`
