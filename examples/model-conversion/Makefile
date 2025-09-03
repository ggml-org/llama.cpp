MAKEFLAGS += --no-print-directory

define validate_model_path
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: MODEL_PATH must be provided either as:"; \
		echo "  1. Environment variable: export MODEL_PATH=/path/to/model"; \
		echo "  2. Command line argument: make $(1) MODEL_PATH=/path/to/model"; \
		exit 1; \
	fi
endef

define validate_embedding_model_path
	@if [ -z "$(EMBEDDING_MODEL_PATH)" ]; then \
		echo "Error: EMBEDDING_MODEL_PATH must be provided either as:"; \
		echo "  1. Environment variable: export EMBEDDING_MODEL_PATH=/path/to/model"; \
		echo "  2. Command line argument: make $(1) EMBEDDING_MODEL_PATH=/path/to/model"; \
		exit 1; \
	fi
endef

define quantize_model
	@CONVERTED_MODEL="$(1)" QUANTIZED_TYPE="$(QUANTIZED_TYPE)" \
	TOKEN_EMBD_TYPE="$(TOKEN_EMBD_TYPE)" OUTPUT_TYPE="$(OUTPUT_TYPE)" \
	./scripts/utils/quantize.sh "$(1)" "$(QUANTIZED_TYPE)" "$(TOKEN_EMBD_TYPE)" "$(OUTPUT_TYPE)"
	@echo "Export the quantized model path to $(2) variable in your environment"
endef

###
### Casual Model targets/recipes
###
causal-convert-model-bf16: OUTTYPE=bf16
causal-convert-model-bf16: causal-convert-model

causal-convert-model:
	$(call validate_model_path,causal-convert-model)
	@MODEL_NAME="$(MODEL_NAME)" OUTTYPE="$(OUTTYPE)" MODEL_PATH="$(MODEL_PATH)" \
	METADATA_OVERRIDE="$(METADATA_OVERRIDE)" \
	./scripts/causal/convert-model.sh

causal-convert-mm-model-bf16: OUTTYPE=bf16
causal-convert-mm-model-bf16: MM_OUTTYPE=f16
causal-convert-mm-model-bf16: causal-convert-mm-model

causal-convert-mm-model:
	$(call validate_model_path,causal-convert-mm-model)
	@MODEL_NAME="$(MODEL_NAME)" OUTTYPE="$(OUTTYPE)" MODEL_PATH="$(MODEL_PATH)" \
	METADATA_OVERRIDE="$(METADATA_OVERRIDE)" \
	./scripts/causal/convert-model.sh

	@MODEL_NAME="$(MODEL_NAME)" OUTTYPE="$(MM_OUTTYPE)" MODEL_PATH="$(MODEL_PATH)" \
	METADATA_OVERRIDE="$(METADATA_OVERRIDE)" \
	./scripts/causal/convert-model.sh --mmproj

causal-run-original-model:
	$(call validate_model_path,causal-run-original-model)
	@MODEL_PATH="$(MODEL_PATH)" ./scripts/causal/run-org-model.py

causal-run-converted-model:
	@CONVERTED_MODEL="$(CONVERTED_MODEL)" ./scripts/causal/run-converted-model.sh

causal-verify-logits: causal-run-original-model causal-run-converted-model
	@./scripts/causal/compare-logits.py
	@MODEL_PATH="$(MODEL_PATH)" ./scripts/utils/check-nmse.py -m ${MODEL_PATH}

causal-run-original-embeddings:
	@./scripts/causal/run-casual-gen-embeddings-org.py

causal-run-converted-embeddings:
	@./scripts/causal/run-converted-model-embeddings-logits.sh

causal-verify-embeddings: causal-run-original-embeddings causal-run-converted-embeddings
	@./scripts/causal/compare-embeddings-logits.sh

causal-inspect-original-model:
	@./scripts/utils/inspect-org-model.py

causal-inspect-converted-model:
	@./scripts/utils/inspect-converted-model.sh

causal-start-embedding-server:
	@./scripts/utils/run-embedding-server.sh ${CONVERTED_MODEL}

causal-curl-embedding-endpoint: causal-run-original-embeddings
	@./scripts/utils/curl-embedding-server.sh | ./scripts/causal/compare-embeddings-logits.sh

causal-quantize-Q8_0: QUANTIZED_TYPE = Q8_0
causal-quantize-Q8_0: causal-quantize-model

causal-quantize-Q4_0: QUANTIZED_TYPE = Q4_0
causal-quantize-Q4_0: causal-quantize-model

# For Quantization Aware Trained (QAT) models in Q4_0 we explicitly set the
# token embedding and output types to Q8_0 instead of the default Q6_K.
causal-quantize-qat-Q4_0: QUANTIZED_TYPE = Q4_0
causal-quantize-qat-Q4_0: TOKEN_EMBD_TYPE = Q8_0
causal-quantize-qat-Q4_0: OUTPUT_TYPE = Q8_0
causal-quantize-qat-Q4_0: causal-quantize-model

causal-quantize-model:
	$(call quantize_model,$(CONVERTED_MODEL),QUANTIZED_MODEL)

causal-run-quantized-model:
	@QUANTIZED_MODEL="$(QUANTIZED_MODEL)" ./scripts/causal/run-converted-model.sh ${QUANTIZED_MODEL}


###
### Embedding Model targets/recipes
###

embedding-convert-model-bf16: OUTTYPE=bf16
embedding-convert-model-bf16: embedding-convert-model

embedding-convert-model:
	$(call validate_embedding_model_path,embedding-convert-model)
	@MODEL_NAME="$(MODEL_NAME)" OUTTYPE="$(OUTTYPE)" MODEL_PATH="$(EMBEDDING_MODEL_PATH)" \
	METADATA_OVERRIDE="$(METADATA_OVERRIDE)" \
	./scripts/embedding/convert-model.sh

embedding-run-original-model:
	$(call validate_embedding_model_path,embedding-run-original-model)
	@EMBEDDING_MODEL_PATH="$(EMBEDDING_MODEL_PATH)" ./scripts/embedding/run-original-model.py

embedding-run-converted-model:
	@CONVERTED_EMBEDDING_MODEL="$(CONVERTED_EMBEDDING_MODEL)" ./scripts/embedding/run-converted-model.sh ${CONVERTED_EMBEDDING_MODEL}

embedding-verify-logits: embedding-run-original-model embedding-run-converted-model
	@./scripts/embedding/compare-embeddings-logits.sh

embedding-inspect-original-model:
	$(call validate_embedding_model_path,embedding-inspect-original-model)
	@EMBEDDING_MODEL_PATH="$(EMBEDDING_MODEL_PATH)" ./scripts/utils/inspect-org-model.py -m ${EMBEDDING_MODEL_PATH}

embedding-inspect-converted-model:
	@CONVERTED_EMBEDDING_MODEL="$(CONVERTED_EMBEDDING_MODEL)" ./scripts/utils/inspect-converted-model.sh ${CONVERTED_EMBEDDING_MODEL}

embedding-start-embedding-server:
	@./scripts/utils/run-embedding-server.sh ${CONVERTED_EMBEDDING_MODEL}

embedding-curl-embedding-endpoint:
	@./scripts/utils/curl-embedding-server.sh | ./scripts/embedding/compare-embeddings-logits.sh

embedding-quantize-Q8_0: QUANTIZED_TYPE = Q8_0
embedding-quantize-Q8_0: embedding-quantize-model

embedding-quantize-Q4_0: QUANTIZED_TYPE = Q4_0
embedding-quantize-Q4_0: embedding-quantize-model

# For Quantization Aware Trained (QAT) models in Q4_0 we explicitly set the
# token embedding and output types to Q8_0 instead of the default Q6_K.
embedding-quantize-qat-Q4_0: QUANTIZED_TYPE = Q4_0
embedding-quantize-qat-Q4_0: TOKEN_EMBD_TYPE = Q8_0
embedding-quantize-qat-Q4_0: OUTPUT_TYPE = Q8_0
embedding-quantize-qat-Q4_0: embedding-quantize-model

embedding-quantize-model:
	$(call quantize_model,$(CONVERTED_EMBEDDING_MODEL),QUANTIZED_EMBEDDING_MODEL)

embedding-run-quantized-model:
	@./scripts/embedding/run-converted-model.sh ${QUANTIZED_EMBEDDING_MODEL}

###
### Perplexity targets/recipes
###
perplexity-data-gen:
	CONVERTED_MODEL="$(CONVERTED_MODEL)" ./scripts/utils/perplexity-gen.sh

perplexity-run-full:
	QUANTIZED_MODEL="$(QUANTIZED_MODEL)" LOOGITS_FILE="$(LOGITS_FILE)" \
	./scripts/utils/perplexity-run.sh

perplexity-run:
	QUANTIZED_MODEL="$(QUANTIZED_MODEL)" ./scripts/utils/perplexity-run-simple.sh

###
### HuggingFace targets/recipes
###

hf-create-model:
	@./scripts/utils/hf-create-model.py -m "${MODEL_NAME}" -ns "${NAMESPACE}" -b "${ORIGINAL_BASE_MODEL}"

hf-create-model-dry-run:
	@./scripts/utils/hf-create-model.py -m "${MODEL_NAME}" -ns "${NAMESPACE}" -b "${ORIGINAL_BASE_MODEL}" -d

hf-create-model-embedding:
	@./scripts/utils/hf-create-model.py -m "${MODEL_NAME}" -ns "${NAMESPACE}" -b "${ORIGINAL_BASE_MODEL}" -e

hf-create-model-embedding-dry-run:
	@./scripts/utils/hf-create-model.py -m "${MODEL_NAME}" -ns "${NAMESPACE}" -b "${ORIGINAL_BASE_MODEL}" -e -d

hf-create-model-private:
	@./scripts/utils/hf-create-model.py -m "${MODEL_NAME}" -ns "${NAMESPACE}" -b "${ORIGINAL_BASE_MODEL}" -p

hf-upload-gguf-to-model:
	@./scripts/utils/hf-upload-gguf-model.py -m "${MODEL_PATH}" -r "${REPO_ID}" -o "${NAME_IN_REPO}"

hf-create-collection:
	@./scripts/utils/hf-create-collection.py -n "${NAME}" -d "${DESCRIPTION}" -ns "${NAMESPACE}"

hf-add-model-to-collection:
	@./scripts/utils/hf-add-model-to-collection.py -c "${COLLECTION}" -m "${MODEL}"


.PHONY: clean
clean:
	@${RM} -rf data .converted_embedding_model.txt .converted_model.txt .embedding_model_name.txt .model_name.txt

