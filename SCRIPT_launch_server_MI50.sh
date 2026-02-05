#!/bin/bash
# shellcheck disable=SC1143,SC2215
cat << 'EOF'

   ██╗     ██╗      █████╗ ███╗   ███╗ █████╗    ██████╗██████╗ ██████╗
   ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗  ██╔════╝██╔══██╗██╔══██╗
   ██║     ██║     ███████║██╔████╔██║███████║  ██║     ██████╔╝██████╔╝
   ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║  ██║     ██╔═══╝ ██╔═══╝
   ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║  ╚██████╗██║     ██║
   ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═════╝╚═╝     ╚═╝
            ██████╗ ███████╗██╗  ██╗ █████╗  ██████╗  ██████╗
           ██╔════╝ ██╔════╝╚██╗██╔╝██╔══██╗██╔═████╗██╔════╝
           ██║  ███╗█████╗   ╚███╔╝ ╚██████║██║██╔██║███████╗
           ██║   ██║██╔══╝   ██╔██╗  ╚═══██║████╔╝██║██╔═══██╗
           ╚██████╔╝██║     ██╔╝ ██╗ █████╔╝╚██████╔╝╚██████╔╝
            ╚═════╝ ╚═╝     ╚═╝  ╚═╝ ╚════╝  ╚═════╝  ╚═════╝            


EOF

export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export GGML_BACKEND_HIP=1
export HCC_AMDGPU_TARGET=gfx906

# Model path 
MODEL_PATH="/path/..."

# Vision projector path (uncomment for multimodal models)
#MMPROJ_PATH="/path/..."

# Model path .................. -m
# Vision projector ............ --mmproj
# GPU layers (99 = all) ....... -ngl
# Flash attention ............. -fa
# KV cache key type ........... -ctk
# KV cache value type ......... -ctv
# Listen interface ............ --host
# Server port ................. --port
# Context size (tokens) ....... -c
# Jinja templating ............ --jinja

#ngram settings:
#        --spec-type ngram-mod \
#        --spec-ngram-size-n 24 \
#        --draft-min 48 \
#        --draft-max 64 \


./build/bin/llama-server \
    -m "$MODEL_PATH" \
        #--spec-type ngram-mod \
        #--spec-ngram-size-n 24 \
        #--draft-min 48 \
        #--draft-max 64 \
    -ngl 99 \
    -fa on \
    -ctk f16 \
    -ctv f16 \
    --host 0.0.0.0 \
    --port 8080 \
    -c 800 \
    -b 2048 \
    -ub 2048 \
    --jinja
    # --mmproj "$MMPROJ_PATH"
    
    
    