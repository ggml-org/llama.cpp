#!/bin/bash
#
# Multi-process Tensor Parallelism launcher for llama.cpp SYCL backend
#
# Usage:
#   ./scripts/mpirun-tp.sh -n 2 ./build/bin/llama-cli -m model.gguf --split-mode tp ...
#
# This script:
# 1. Sets up the Intel oneAPI environment
# 2. Determines MPI rank and sets ONEAPI_DEVICE_SELECTOR per-process
# 3. Runs llama-cli with one GPU per process
#

set -e

# Parse -n argument for number of processes
NUM_PROCS=2
while getopts "n:" opt; do
    case $opt in
        n) NUM_PROCS=$OPTARG ;;
    esac
done
shift $((OPTIND-1))

if [ $# -lt 1 ]; then
    echo "Usage: $0 -n <num_gpus> <llama-cli command...>"
    echo ""
    echo "Example:"
    echo "  $0 -n 2 ./build/bin/llama-cli -m model.gguf --split-mode tp -p 'Hello' -n 10"
    exit 1
fi

# Source oneAPI if not already done
if [ -z "$ONEAPI_ROOT" ]; then
    source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
fi

# Create a wrapper that sets device selector based on MPI rank
WRAPPER_SCRIPT=$(mktemp)
cat > "$WRAPPER_SCRIPT" << 'EOF'
#!/bin/bash
# Get MPI rank
if [ -n "$PMI_RANK" ]; then
    RANK=$PMI_RANK
elif [ -n "$OMPI_COMM_WORLD_RANK" ]; then
    RANK=$OMPI_COMM_WORLD_RANK
else
    RANK=0
fi

# Set device selector for this process
export ONEAPI_DEVICE_SELECTOR="level_zero:$RANK"

# CCL environment variables for multi-process coordination
# Use MPI transport since we're running under mpirun
export CCL_ATL_TRANSPORT=mpi
export CCL_WORKER_COUNT=1
export CCL_SYCL_OUTPUT_EVENT=0

# Debug: show rank and device
echo "[Rank $RANK] Device selector: $ONEAPI_DEVICE_SELECTOR" >&2

# Run the actual command
exec "$@"
EOF
chmod +x "$WRAPPER_SCRIPT"

echo "Launching $NUM_PROCS processes with tensor parallelism..."
echo "Each process will use one GPU (level_zero:0, level_zero:1, ...)"
echo ""

# Run with mpirun
mpirun -n "$NUM_PROCS" -l "$WRAPPER_SCRIPT" "$@"

# Cleanup
rm -f "$WRAPPER_SCRIPT"
