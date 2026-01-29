# Model Conversion Example
This directory contains scripts and code to help in the process of converting
HuggingFace PyTorch models to GGUF format.

The motivation for having this is that the conversion process can often be an
iterative process, where the original model is inspected, converted, updates
made to llama.cpp, converted again, etc. Once the model has been converted it
needs to be verified against the original model, and then optionally quantified,
and in some cases perplexity checked of the quantized model. And finally the
model/models need to the ggml-org on Hugging Face. This tool/example tries to
help with this process.

> 📝 **Note:** When adding a new model from an existing family, verify the
> previous version passes logits verification first. Existing models can have
> subtle numerical differences that don't affect generation quality but cause
> logits mismatches. Identifying these upfront whether they exist in llama.cpp,
> the conversion script, or in an upstream implementation, can save significant
> debugging time.

### Overview
The idea is that the makefile targets and scripts here can be used in the
development/conversion process assisting with things like:

* inspect/run the original model to figure out how it works
* convert the original model to GGUF format
* inspect/run the converted model
* verify the logits produced by the original model and the converted model
* quantize the model to GGUF format
* run perplexity evaluation to verify that the quantized model is performing
  as expected
* upload the model to HuggingFace to make it available for others

## Setup
Create virtual python environment
```console
$ python3.11 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```

## Causal Language Model Conversion
This section describes the steps to convert a causal language model to GGUF and
to verify that the conversion was successful.

### Download the original model
First, clone the original model to some local directory:
```console
$ mkdir models && cd models
$ git clone https://huggingface.co/user/model_name
$ cd model_name
$ git lfs install
$ git lfs pull
```

### Set the MODEL_PATH
The path to the downloaded model can be provided in two ways:

**Option 1: Environment variable (recommended for iterative development)**
```console
export MODEL_PATH=~/work/ai/models/some_model
```

**Option 2: Command line argument (for one-off tasks)**
```console
make causal-convert-model MODEL_PATH=~/work/ai/models/some_model
```

Command line arguments take precedence over environment variables when both are provided.

In cases where the transformer implementation for the model has not been released
yet it is possible to set the environment variable `UNRELEASED_MODEL_NAME` which
will then cause the transformer implementation to be loaded explicitely and not
use AutoModelForCausalLM:
```
export UNRELEASED_MODEL_NAME=SomeNewModel
```

### Inspecting the original tensors
```console
# Using environment variable
(venv) $ make causal-inspect-original-model

# Or using command line argument
(venv) $ make causal-inspect-original-model MODEL_PATH=~/work/ai/models/some_model
```

### Running the original model
This is mainly to verify that the original model works, and to compare the output
from the converted model.
```console
# Using environment variable
(venv) $ make causal-run-original-model

# Or using command line argument
(venv) $ make causal-run-original-model MODEL_PATH=~/work/ai/models/some_model
```
This command will save two files to the `data` directory, one is a binary file
containing logits which will be used for comparison with the converted model
later, and the other is a text file which allows for manual visual inspection.

### Model conversion
After updates have been made to [gguf-py](../../gguf-py) to add support for the
new model, the model can be converted to GGUF format using the following command:
```console
# Using environment variable
(venv) $ make causal-convert-model

# Or using command line argument
(venv) $ make causal-convert-model MODEL_PATH=~/work/ai/models/some_model
```

### Inspecting the converted model
The converted model can be inspected using the following command:
```console
(venv) $ make causal-inspect-converted-model
```

### Running the converted model
```console
(venv) $ make causal-run-converted-model
```

### Model logits verfication
The following target will run the original model and the converted model and
compare the logits:
```console
(venv) $ make causal-verify-logits
```

### Quantizing the model
The causal model can be quantized to GGUF format using the following command:
```console
(venv) $ make causal-quantize-Q8_0
Quantized model saved to: /path/to/quantized/model-Q8_0.gguf
Export the quantized model path to QUANTIZED_MODEL variable in your environment
```
This will show the path to the quantized model in the terminal, which can then
be used to set the `QUANTIZED_MODEL` environment variable:
```console
export QUANTIZED_MODEL=/path/to/quantized/model-Q8_0.gguf
```
Then the quantized model can be run using the following command:
```console
(venv) $ make causal-run-quantized-model
```

### Quantizing QAT (Quantization Aware Training) models
When quantizing to `Q4_0`, the default data type for the token embedding weights
will be `Q6_K`. For models that are going to be uploaded to ggml-org it is
recommended to use `Q8_0` instead for the embeddings and output tensors.
The reason is that although `Q6_K` is smaller in size, it requires more compute
to unpack, which can hurt performance during output generation when the entire
embedding matrix must be dequantized to compute vocabulary logits. `Q8_0`
provides practically full quality with better computational efficiency.
```console
(venv) $ make causal-quantize-qat-Q4_0
```


## Embedding Language Model Conversion

### Download the original model
```console
$ mkdir models && cd models
$ git clone https://huggingface.co/user/model_name
$ cd model_name
$ git lfs install
$ git lfs pull
```

The path to the embedding model can be provided in two ways:

**Option 1: Environment variable (recommended for iterative development)**
```console
export EMBEDDING_MODEL_PATH=~/path/to/embedding_model
```

**Option 2: Command line argument (for one-off tasks)**
```console
make embedding-convert-model EMBEDDING_MODEL_PATH=~/path/to/embedding_model
```

Command line arguments take precedence over environment variables when both are provided.

### Running the original model
This is mainly to verify that the original model works and to compare the output
with the output from the converted model.
```console
# Using environment variable
(venv) $ make embedding-run-original-model

# Or using command line argument
(venv) $ make embedding-run-original-model EMBEDDING_MODEL_PATH=~/path/to/embedding_model
```
This command will save two files to the `data` directory, one is a binary
file containing logits which will be used for comparison with the converted
model, and the other is a text file which allows for manual visual inspection.

#### Using SentenceTransformer with numbered layers
For models that have numbered SentenceTransformer layers (01_Pooling, 02_Dense,
03_Dense, 04_Normalize), these will be applied automatically when running the
converted model but currently there is a separate target to run the original
version:

```console
# Run original model with SentenceTransformer (applies all numbered layers)
(venv) $ make embedding-run-original-model-st
```

This will use the SentenceTransformer library to load and run the model, which
automatically applies all the numbered layers in the correct order. This is
particularly useful when comparing with models that should include these
additional transformation layers beyond just the base model output.

The type of normalization can be specified for the converted model but is not
strictly necessary as the verification uses cosine similarity and the magnitude
of the output vectors does not affect this. But the normalization type can be
specified as an argument to the target which might be useful for manual
inspection:
```console
(venv) $ make embedding-verify-logits-st EMBD_NORMALIZE=1
```
The original model will apply the normalization according to the normalization
layer specified in the modules.json configuration file.

### Model conversion
After updates have been made to [gguf-py](../../gguf-py) to add support for the
new model the model can be converted to GGUF format using the following command:
```console
(venv) $ make embedding-convert-model
```

### Run the converted model
```console
(venv) $ make embedding-run-converted-model
```

### Model logits verfication
The following target will run the original model and the converted model (which
was done manually in the previous steps) and compare the logits:
```console
(venv) $ make embedding-verify-logits
```

For models with SentenceTransformer layers, use the `-st` verification target:
```console
(venv) $ make embedding-verify-logits-st
```
This convenience target automatically runs both the original model with SentenceTransformer
and the converted model with pooling enabled, then compares the results.

### llama-server verification
To verify that the converted model works with llama-server, the following
command can be used:
```console
(venv) $ make embedding-start-embedding-server
```
Then open another terminal and set the `EMBEDDINGS_MODEL_PATH` environment
variable as this will not be inherited by the new terminal:
```console
(venv) $ make embedding-curl-embedding-endpoint
```
This will call the `embedding` endpoing and the output will be piped into
the same verification script as used by the target `embedding-verify-logits`.

The causal model can also be used to produce embeddings and this can be verified
using the following commands:
```console
(venv) $ make causal-start-embedding-server
```
Then open another terminal and set the `MODEL_PATH` environment
variable as this will not be inherited by the new terminal:
```console
(venv) $ make casual-curl-embedding-endpoint
```

### Quantizing the model
The embedding model can be quantized to GGUF format using the following command:
```console
(venv) $ make embedding-quantize-Q8_0
Quantized model saved to: /path/to/quantized/model-Q8_0.gguf
Export the quantized model path to QUANTIZED_EMBEDDING_MODEL variable in your environment
```
This will show the path to the quantized model in the terminal, which can then
be used to set the `QUANTIZED_EMBEDDING_MODEL` environment variable:
```console
export QUANTIZED_EMBEDDING_MODEL=/path/to/quantized/model-Q8_0.gguf
```
Then the quantized model can be run using the following command:
```console
(venv) $ make embedding-run-quantized-model
```

### Quantizing QAT (Quantization Aware Training) models
When quantizing to `Q4_0`, the default data type for the token embedding weights
will be `Q6_K`. For models that are going to be uploaded to ggml-org it is
recommended to use `Q8_0` instead for the embeddings and output tensors.
The reason is that although `Q6_K` is smaller in size, it requires more compute
to unpack, which can hurt performance during output generation when the entire
embedding matrix must be dequantized to compute vocabulary logits. `Q8_0`
provides practically full quality with better computational efficiency.
```console
(venv) $ make embedding-quantize-qat-Q4_0
```

## Perplexity Evaluation

### Simple perplexity evaluation
This allows to run the perplexity evaluation without having to generate a
token/logits file:
```console
(venv) $ make perplexity-run QUANTIZED_MODEL=~/path/to/quantized/model.gguf
```
This will use the wikitext dataset to run the perplexity evaluation and
output the perplexity score to the terminal. This value can then be compared
with the perplexity score of the unquantized model.

### Full perplexity evaluation
First use the converted, non-quantized, model to generate the perplexity evaluation
dataset using the following command:
```console
$ make perplexity-data-gen CONVERTED_MODEL=~/path/to/converted/model.gguf
```
This will generate a file in the `data` directory named after the model and with
a `.kld` suffix which contains the tokens and the logits for the wikitext dataset.

After the dataset has been generated, the perplexity evaluation can be run using
the quantized model:
```console
$ make perplexity-run-full QUANTIZED_MODEL=~/path/to/quantized/model-Qxx.gguf LOGITS_FILE=data/model.gguf.ppl
```

> 📝 **Note:** The `LOGITS_FILE` is the file generated by the previous command
> can be very large, so make sure you have enough disk space available.

## HuggingFace utilities
The following targets are useful for creating collections and model repositories
on Hugging Face in the the ggml-org. These can be used when preparing a relase
to script the process for new model releases.

For the following targets a `HF_TOKEN` environment variable is required.

> 📝 **Note:** Don't forget to logout from Hugging Face after running these
> commands, otherwise you might have issues pulling/cloning repositories as
> the token will still be in use:
> $ huggingface-cli logout
> $ unset HF_TOKEN

### Create a new Hugging Face Model (model repository)
This will create a new model repsository on Hugging Face with the specified
model name.
```console
(venv) $ make hf-create-model MODEL_NAME='TestModel' NAMESPACE="danbev" ORIGINAL_BASE_MODEL="some-base-model"
Repository ID:  danbev/TestModel-GGUF
Repository created: https://huggingface.co/danbev/TestModel-GGUF
```
Note that we append a `-GGUF` suffix to the model name to ensure a consistent
naming convention for GGUF models.

An embedding model can be created using the following command:
```console
(venv) $ make hf-create-model-embedding MODEL_NAME='TestEmbeddingModel' NAMESPACE="danbev" ORIGINAL_BASE_MODEL="some-base-model"
```
The only difference is that the model card for an embedding model will be different
with regards to the llama-server command and also how to access/call the embedding
endpoint.

### Upload a GGUF model to model repository
The following target uploads a model to an existing Hugging Face model repository.
```console
(venv) $ make hf-upload-gguf-to-model MODEL_PATH=dummy-model1.gguf REPO_ID=danbev/TestModel-GGUF
📤 Uploading dummy-model1.gguf to danbev/TestModel-GGUF/dummy-model1.gguf
✅ Upload successful!
🔗 File available at: https://huggingface.co/danbev/TestModel-GGUF/blob/main/dummy-model1.gguf
```
This command can also be used to update an existing model file in a repository.

### Create a new Collection
```console
(venv) $ make hf-new-collection NAME=TestCollection DESCRIPTION="Collection for testing scripts" NAMESPACE=danbev
🚀 Creating Hugging Face Collection
Title: TestCollection
Description: Collection for testing scripts
Namespace: danbev
Private: False
✅ Authenticated as: danbev
📚 Creating collection: 'TestCollection'...
✅ Collection created successfully!
📋 Collection slug: danbev/testcollection-68930fcf73eb3fc200b9956d
🔗 Collection URL: https://huggingface.co/collections/danbev/testcollection-68930fcf73eb3fc200b9956d

🎉 Collection created successfully!
Use this slug to add models: danbev/testcollection-68930fcf73eb3fc200b9956d
```

### Add model to a Collection
```console
(venv) $ make hf-add-model-to-collection COLLECTION=danbev/testcollection-68930fcf73eb3fc200b9956d MODEL=danbev/TestModel-GGUF
✅ Authenticated as: danbev
🔍 Checking if model exists: danbev/TestModel-GGUF
✅ Model found: danbev/TestModel-GGUF
📚 Adding model to collection...
✅ Model added to collection successfully!
🔗 Collection URL: https://huggingface.co/collections/danbev/testcollection-68930fcf73eb3fc200b9956d

🎉 Model added successfully!

```
