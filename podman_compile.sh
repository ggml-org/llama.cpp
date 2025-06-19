#! /bin/bash


set -o pipefail
set -o errexit
set -o nounset
set -o errtrace

opts=""
opts="$opts --device /dev/dri "
echo "Running with the GPU passthrough"

IMAGE=quay.io/ramalama/remoting:latest

what=${1:-}
if [[ -z "$what" ]]; then
    what=remoting
fi

cmd="bash ./build.$what.sh"

POD_NAME=mac_ai_compiling
podman machine ssh podman rm $POD_NAME --force

set -x
podman run \
--name $POD_NAME \
--user root:root \
--cgroupns host \
--security-opt label=disable \
--env HOME="$HOME" \
--env PERF_MODE="${PERF_MODE:-}" \
--env BENCH_MODE="${BENCH_MODE:-}" \
-v "$HOME":"$HOME":Z \
-w "$PWD" \
-it --rm \
$opts \
$IMAGE \
$cmd
