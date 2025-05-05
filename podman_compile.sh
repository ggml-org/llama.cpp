#! /bin/bash


set -o pipefail
set -o errexit
set -o nounset
set -o errtrace

opts=""
opts="$opts --device /dev/dri "
echo "Running with the GPU passthrough"

image=localhost/pytorch:remoting

what=${1:-}
if [[ -z "$what" ]]; then
    what=remoting
fi

cmd="bash ./build.$what.sh"

set -x
podman run \
--name mac_ai_compiling \
--user root:root \
--cgroupns host \
--security-opt label=disable \
--env HOME="$HOME" \
-v "$HOME":"$HOME":Z \
-w "$PWD" \
-it --rm \
$opts \
$image \
$cmd
