#!/usr/bin/env bash

set -e

# Parse command line arguments
SENTENCE_TRANSFORMERS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -st|--sentence-transformers)
            SENTENCE_TRANSFORMERS="--sentence-transformers-dense-modules"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

MODEL_NAME="${MODEL_NAME:-$(basename "$EMBEDDING_MODEL_PATH")}"
OUTPUT_DIR="${OUTPUT_DIR:-../../models}"
TYPE="${OUTTYPE:-f16}"
METADATA_OVERRIDE="${METADATA_OVERRIDE:-}"
CONVERTED_MODEL="${OUTPUT_DIR}/${MODEL_NAME}.gguf"

echo "Model path: ${EMBEDDING_MODEL_PATH}"
echo "Model name: ${MODEL_NAME}"
echo "Data  type: ${TYPE}"
echo "Converted model path:: ${CONVERTED_MODEL}"
python ../../convert_hf_to_gguf.py --verbose \
    ${EMBEDDING_MODEL_PATH} \
    --outfile ${CONVERTED_MODEL} \
    --outtype ${TYPE} \
    ${SENTENCE_TRANSFORMERS}

echo ""
echo "The environment variable CONVERTED_EMBEDDING MODEL can be set to this path using:"
echo "export CONVERTED_EMBEDDING_MODEL=$(realpath ${CONVERTED_MODEL})"
