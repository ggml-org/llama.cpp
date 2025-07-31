#!/bin/bash

# llama.cpp DevContainer Configuration Script
# This script helps you quickly configure optional components for the development container.

set -e

CONFIG_FILE=".devcontainer/devcontainer.json"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Error: $CONFIG_FILE not found. Are you in the llama.cpp root directory?"
    exit 1
fi

echo "🔧 llama.cpp DevContainer Configuration"
echo "======================================"
echo
echo "This script will help you configure optional components for your development environment."
echo "After making changes, you'll need to rebuild the container in VS Code."
echo

# Function to get current setting
get_current_setting() {
    local component=$1
    local current=$(grep -A 10 '"build"' "$CONFIG_FILE" | grep "\"$component\"" | sed 's/.*"\([^"]*\)".*/\1/')
    echo "${current:-false}"
}

# Function to update setting
update_setting() {
    local component=$1
    local value=$2
    
    # Use a more robust sed command that works across platforms
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/\(\"$component\":\s*\)\"[^\"]*\"/\1\"$value\"/" "$CONFIG_FILE"
    else
        # Linux/WSL
        sed -i "s/\(\"$component\":\s*\)\"[^\"]*\"/\1\"$value\"/" "$CONFIG_FILE"
    fi
}

# Get current settings
cuda_current=$(get_current_setting "INSTALL_CUDA")
rocm_current=$(get_current_setting "INSTALL_ROCM")
python_current=$(get_current_setting "INSTALL_PYTHON_DEPS")

echo "Current configuration:"
echo "  • CUDA support: $cuda_current"
echo "  • ROCm support: $rocm_current"
echo "  • Python dependencies: $python_current"
echo

# CUDA Configuration
echo "🎯 CUDA Support (NVIDIA GPUs)"
echo "   Installs CUDA 12.9 toolkit (~5-8 minutes build time)"
read -p "   Enable CUDA support? [y/N]: " cuda_choice
cuda_choice=${cuda_choice,,} # to lowercase
if [[ $cuda_choice =~ ^(yes|y)$ ]]; then
    cuda_new="true"
else
    cuda_new="false"
fi

# ROCm Configuration  
echo
echo "🎯 ROCm Support (AMD GPUs)"
echo "   Installs ROCm 6.4 for AMD GPU acceleration (~8-12 minutes build time)"
read -p "   Enable ROCm support? [y/N]: " rocm_choice
rocm_choice=${rocm_choice,,}
if [[ $rocm_choice =~ ^(yes|y)$ ]]; then
    rocm_new="true"
else
    rocm_new="false"
fi

# Python Dependencies
echo
echo "🎯 Python Dependencies"
echo "   Installs packages for model conversion: numpy, torch, transformers, etc."
read -p "   Enable Python dependencies? [y/N]: " python_choice
python_choice=${python_choice,,}
if [[ $python_choice =~ ^(yes|y)$ ]]; then
    python_new="true"
else
    python_new="false"
fi

# Summary and confirmation
echo
echo "📋 Configuration Summary:"
echo "   • CUDA support: $cuda_current → $cuda_new"
echo "   • ROCm support: $rocm_current → $rocm_new" 
echo "   • Python dependencies: $python_current → $python_new"
echo

# Estimate build time
build_time="2-3 minutes"
if [[ $cuda_new == "true" ]]; then
    build_time="5-8 minutes"
fi
if [[ $rocm_new == "true" ]]; then
    build_time="8-12 minutes"
fi
if [[ $python_new == "true" && $cuda_new == "false" && $rocm_new == "false" ]]; then
    build_time="3-5 minutes"
fi

echo "⏱️  Estimated build time: $build_time"
echo

read -p "Apply these changes? [Y/n]: " confirm
confirm=${confirm,,}
if [[ ! $confirm =~ ^(no|n)$ ]]; then
    echo
    echo "✅ Applying configuration..."
    
    update_setting "INSTALL_CUDA" "$cuda_new"
    update_setting "INSTALL_ROCM" "$rocm_new"
    update_setting "INSTALL_PYTHON_DEPS" "$python_new"
    
    echo "✅ Configuration updated successfully!"
    echo
    echo "🔄 Next steps:"
    echo "   1. Open VS Code in this directory"
    echo "   2. Press Ctrl+Shift+P and select 'Dev Containers: Rebuild Container'"
    echo "   3. Wait for the container to build with your new configuration"
    echo
else
    echo "❌ Configuration cancelled."
fi
