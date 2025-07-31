# DevContainer Configuration Examples

Copy and paste these configurations into your `.devcontainer/devcontainer.json` file, replacing the existing `"build"` section.

## Minimal Setup (Default)
Fastest build time, CPU-only development.
```json
"build": {
    "args": {
        "INSTALL_CUDA": "false",
        "INSTALL_ROCM": "false", 
        "INSTALL_PYTHON_DEPS": "false"
    }
}
```

## CPU + Python Tools
For model conversion and CPU inference.
```json
"build": {
    "args": {
        "INSTALL_CUDA": "false",
        "INSTALL_ROCM": "false", 
        "INSTALL_PYTHON_DEPS": "true"
    }
}
```

## NVIDIA GPU Development
For CUDA acceleration with model tools.
```json
"build": {
    "args": {
        "INSTALL_CUDA": "true",
        "INSTALL_ROCM": "false", 
        "INSTALL_PYTHON_DEPS": "true"
    }
}
```

## AMD GPU Development  
For ROCm acceleration with model tools.
```json
"build": {
    "args": {
        "INSTALL_CUDA": "false",
        "INSTALL_ROCM": "true", 
        "INSTALL_PYTHON_DEPS": "true"
    }
}
```

## Multi-GPU Research Setup
For testing both NVIDIA and AMD GPU paths (large build).
```json
"build": {
    "args": {
        "INSTALL_CUDA": "true",
        "INSTALL_ROCM": "true", 
        "INSTALL_PYTHON_DEPS": "true"
    }
}
```

## Build Time Estimates
- Minimal: 2-3 minutes
- CPU + Python: 3-5 minutes
- NVIDIA GPU: 5-8 minutes
- AMD GPU: 8-12 minutes
- Multi-GPU: 12-15 minutes

## After Changing Configuration
1. Save the `devcontainer.json` file
2. In VS Code: `Ctrl+Shift+P` → "Dev Containers: Rebuild Container"
3. Wait for the build to complete
