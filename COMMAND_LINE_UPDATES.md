# Command-Line Argument Updates

## Summary

This document summarizes the changes made to llama.cpp's command-line arguments and environment variables to improve consistency and make the default behavior more user-friendly.

## Changes Made

### 1. Hyperthreading Flag Rename
- **Old**: `--no-hyperthreading`
- **New**: `--cpu-no-hyperthreading`
- **Behavior**: No change - still disables hyperthreading when specified

### 2. Efficiency Cores Logic Inversion  
- **Old**: `--use-efficiency-cores` (disabled by default, enabled when flag present)
- **New**: `--cpu-no-efficiency-cores` (enabled by default, disabled when flag present)
- **Behavior**: **CHANGED** - Efficiency cores are now **enabled by default**

### 3. Environment Variables Updated
- **Old**: `LLAMA_NO_HYPERTHREADING=1` (disable hyperthreading)
- **New**: `LLAMA_CPU_NO_HYPERTHREADING=1` (disable hyperthreading)
- **Old**: `LLAMA_USE_EFFICIENCY_CORES=1` (enable efficiency cores) 
- **New**: `LLAMA_CPU_NO_EFFICIENCY_CORES=1` (disable efficiency cores)

## Migration Guide

### Command Line
```bash
# Old way
./llama-server --no-hyperthreading --use-efficiency-cores

# New way  
./llama-server --cpu-no-hyperthreading
# (no flag needed for efficiency cores - they're enabled by default now)

# To disable efficiency cores (new option):
./llama-server --cpu-no-efficiency-cores
```

### Environment Variables
```bash
# Old way
LLAMA_NO_HYPERTHREADING=1 LLAMA_USE_EFFICIENCY_CORES=1 ./llama-server

# New way
LLAMA_CPU_NO_HYPERTHREADING=1 ./llama-server
# (efficiency cores enabled by default)

# To disable efficiency cores:
LLAMA_CPU_NO_EFFICIENCY_CORES=1 ./llama-server
```

## Rationale

1. **Consistency**: All CPU-related flags now have `--cpu-` prefix
2. **Better Defaults**: Efficiency cores are now enabled by default for better performance on most systems
3. **Clarity**: Flag names clearly indicate what they disable rather than enable
4. **User-Friendly**: Most users get optimal performance without needing to specify flags

## Default Behavior Changes

### Before
- Hyperthreading: **Enabled** (good default)
- Efficiency cores: **Disabled** (conservative but suboptimal)

### After  
- Hyperthreading: **Enabled** (unchanged)
- Efficiency cores: **Enabled** (better performance default)

## Files Updated

### Source Code
- `common/common.h` - Updated struct defaults
- `common/arg.cpp` - Updated command-line argument parsing
- `common/common.cpp` - Updated environment variable logic

### Documentation  
- `.github/copilot-instructions.md`
- `NUMA_IMPROVEMENTS.md`
- `NUMA_OPTIMIZATION_COMPLETE.md`
- `UNIFIED_MAPPING_SUMMARY.md`
- `.devcontainer/README.md`
- `.devcontainer/launch.json`

## Compatibility

### Backward Compatibility
- **Breaking**: Old environment variable names no longer work
- **Breaking**: Old `--use-efficiency-cores` flag no longer exists
- **Breaking**: Old `--no-hyperthreading` flag no longer exists
- **Behavior Change**: Efficiency cores are now enabled by default

### Forward Compatibility
- All new flag names follow consistent `--cpu-*` pattern
- Logic is more intuitive (flags disable features rather than enable them)
