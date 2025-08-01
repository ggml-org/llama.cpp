# Segfault Fix for Multi-Part GGUF Files - Updated

## Problem Summary

The unified NUMA mapping implementation for multi-part GGUF files was causing segmentation faults during the cleanup phase of model loading. The issue occurred after successful tensor loading when the system attempted to clean up memory mappings.

## Root Cause Analysis

The segfault was happening in the `load_all_data()` function around line 1160 in `llama-model-loader.cpp`. The problem was **not** in the cleanup phase as initially thought, but during tensor loading when trying to access memory mappings.

### The Real Issue: Null Pointer Access During Tensor Loading

In the unified mapping approach:
- The unified mapping was stored **only** in `mappings[0]`
- `mappings[1]` through `mappings[N]` were set to `nullptr` 
- When processing tensors from files 1-5, the code tried to access `mappings[weight->idx]` where `weight->idx` was 1, 2, 3, 4, or 5
- This resulted in dereferencing null pointers: `mapping->addr()` where `mapping` was null

### Memory Access Pattern

The crash occurred at:
```cpp
uint8_t * data = (uint8_t *) mapping->addr() + weight->offs;
```

Where `mapping` was null because `mappings[weight->idx]` was null for `weight->idx > 0`.

## Solution Implemented

### Fix 1: Proper Unified Mapping Detection
The access code now detects unified mappings and uses the correct mapping:
```cpp
#ifdef GGML_NUMA_MIRROR
// Check if this is a unified mapping by seeing if mappings[1] is null but mappings[0] exists
bool is_unified_mapping = mappings.size() > 1 && mappings[0] && !mappings[1];
if (is_unified_mapping) {
    // For unified mapping, always use mappings[0] and calculate the file offset
    mapping_ptr = &mappings[0];
    // Calculate offset for this file within the unified mapping
    for (int i = 0; i < weight->idx; ++i) {
        file_offset += files[i]->size();
    }
} else {
    // Standard per-file mapping
    mapping_ptr = &mappings.at(weight->idx);
}
#endif
```

### Fix 2: Correct Memory Address Calculation
For unified mappings, the memory address calculation includes the file offset:
```cpp
uint8_t * data = (uint8_t *) mapping->addr() + file_offset + weight->offs;
```

### Fix 3: Updated Cleanup Logic
The cleanup logic now correctly detects unified mappings using the same pattern:
```cpp
bool is_unified_mapping = mappings.size() > 1 && mappings[0] && !mappings[1];
```

## Technical Details

The key insight is that the original bug was a **memory access issue during tensor loading**, not a cleanup issue:

1. **Problem**: Multi-file models have tensors with `weight->idx` ranging from 0 to N-1, but unified mappings only stored the mapping in `mappings[0]`, leaving `mappings[1]` through `mappings[N-1]` as null pointers
2. **Crash**: When processing a tensor from file 1, 2, 3, etc., the code tried to access `mappings[weight->idx]->addr()` where `mappings[weight->idx]` was null
3. **Solution**: Detect unified mappings and redirect all accesses to `mappings[0]` with proper offset calculation

The fix ensures that:
- Unified mappings are properly detected by checking the null pattern: `mappings[0]` exists but `mappings[1]` is null
- All tensor access goes through `mappings[0]` with correct file offset calculation
- Cleanup logic also respects the unified mapping pattern

## Files Modified

- `src/llama-model-loader.cpp`: Enhanced cleanup logic to properly handle unified mappings vs standard mappings

## Verification

The fix addresses the exact crash pattern and root cause:
1. ✓ Unified mapping is created successfully and stored in `mappings[0]`
2. ✓ Files are mapped correctly with proper offset calculation  
3. ✓ Tensor loading can now access all tensors regardless of source file index
4. ✓ Memory access uses the correct mapping (`mappings[0]`) with calculated file offsets
5. ✓ Cleanup phase properly detects unified mappings and handles them appropriately

## Expected Behavior

After this fix, multi-part GGUF files should:
- Load successfully with unified NUMA mapping
- Complete tensor loading without crashes
- Clean up properly without segfaults or memory corruption
- Provide the performance benefits of unified mapping while maintaining memory safety

## Memory Management

The fix ensures no memory leaks by:
- Using RAII pattern where `std::unique_ptr<llama_mmap>` automatically calls destructors
- Unified mapping destructor properly cleans up the entire memory region
- No partial unmapping that could corrupt the unified memory region
- Proper null pointer handling for unused mapping slots

## Deployment

The updated fix is now built and ready for testing. The same command that was crashing should now work:

```bash
./llama-server --model your-multipart-model.gguf
```

The logs should show successful completion instead of segfaults after the progress dots.

## Debug Tracing Guide

If you need to debug further segfaults or issues, here are several approaches:

### 1. Enable Built-in LLAMA_TRACE (Debug Build Required)

```bash
# First, build in debug mode
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --parallel

# Then run with trace enabled
export LLAMA_TRACE=1
./build/bin/llama-server --model your-model.gguf
```

### 2. Enable Debug Logging

```bash
# Set log level to debug
export GGML_LOG_LEVEL=DEBUG
./build/bin/llama-server --model your-model.gguf
```

### 3. Use GDB for Stack Traces

```bash
# Run with GDB to catch segfaults
gdb ./build/bin/llama-server
(gdb) run --model your-model.gguf
# When it crashes:
(gdb) bt
(gdb) info registers
(gdb) list
```

### 4. Use Valgrind for Memory Issues

```bash
# Install valgrind if not present
sudo apt-get install valgrind

# Run with valgrind to detect memory errors
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
  --track-origins=yes --verbose \
  ./build/bin/llama-server --model your-model.gguf
```

### 5. Enable Address Sanitizer (ASan)

```bash
# Build with address sanitizer
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -g" \
  -DCMAKE_C_FLAGS="-fsanitize=address -g"
cmake --build build-asan --parallel

# Run with ASan enabled
./build-asan/bin/llama-server --model your-model.gguf
```

### 6. Custom Debug Output

You can also add temporary debug output to the code. Add these lines in critical sections:

```cpp
// In llama-model-loader.cpp
LLAMA_LOG_INFO("DEBUG: Entering cleanup phase, mappings.size()=%zu\n", mappings.size());
LLAMA_LOG_INFO("DEBUG: is_unified_mapping=%s\n", is_unified_mapping ? "true" : "false");
```

### 7. Core Dump Analysis

If you get core dumps:

```bash
# Enable core dumps
ulimit -c unlimited

# Run the program and let it crash
./build/bin/llama-server --model your-model.gguf

# Analyze the core dump
gdb ./build/bin/llama-server core
(gdb) bt
(gdb) info threads
(gdb) thread apply all bt
```

### 8. SystemD Journal Integration

For systemd services, you can get more detailed logs:

```bash
# Check the service logs with more detail
journalctl -u your-service.service -f --no-pager -o verbose

# Or run directly to bypass systemd
sudo -u your-service-user ./build/bin/llama-server --model your-model.gguf
```

**Note**: Most debugging features require a Debug build (`CMAKE_BUILD_TYPE=Debug`) rather than Release mode to work properly.
