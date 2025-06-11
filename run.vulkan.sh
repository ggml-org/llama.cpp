#! /bin/bash
if [[ ${1:-} == "strace" ]]; then
    prefix="strace"
elif [[ ${1:-} == "gdb" ]]; then
    prefix="gdb --args"
else
    prefix=""
fi

rm -f /usr/lib64/libvulkan_virtio.so

ICD_DIR=/Users/kevinpouget/.local/share/vulkan/icd.d

USE_WORK_MESA=1
if [[ "$USE_WORK_MESA" == 1 ]]; then
    export VK_ICD_FILENAMES=$ICD_DIR/virtio_icd.aarch64.json
else
    export VK_ICD_FILENAMES=$ICD_DIR/virtio_icd.good.aarch64.json
fi

# init result vtest wsi no_abort log_ctx_info cache no_sparse no_gpl
export VN_DEBUG=vtest
$prefix ../build.vulkan/bin/llama-run --verbose ~/models/llama3.2 "say nothing"
