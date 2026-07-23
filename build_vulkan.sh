set -e

echo "======================================"
echo "=   LLAMA.CPP VULKAN BUILD SCRIPT    ="
echo "======================================"
echo ""

# List of required commands
required_cmds=(cmake curl gcc g++ make)

# Check each command
for cmd in "${required_cmds[@]}"; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: '$cmd' not found. Please install it and try again." >&2
        exit 1
    fi
done

# Check for Vulkan SDK (by looking for vulkaninfo)
if ! command -v vulkaninfo &> /dev/null; then
    echo "Error: Vulkan SDK not found. Please install it (e.g., from https://vulkan.lunarg.com/)." >&2
    exit 1
fi

rm -rf build
mkdir build && cd build
cmake .. -DGGML_VULKAN=ON
make

echo "Build finished. See llama-server at ./bin/"
