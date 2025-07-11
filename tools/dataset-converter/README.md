`convert-to-train-gguf` Utility
===============================

This utility is designed to convert text datasets (or pre-tokenized data) into the GGUF format, optimized for training models in `llama.cpp`.

Features
--------

*   **Two-pass processing**: Efficiently handles large datasets that do not fit entirely into RAM, performing a first pass to collect metadata and a second pass to write the actual tensor data.

*   **Flexible input**: Supports reading both raw text (with subsequent tokenization using a provided model) and pre-tokenized data (in the format of space-separated token IDs).

*   **Modular architecture**: The code is divided into several classes (`llama_gguf_file`, `llama_gguf_writer`, `llama_dataset_reader`, `llama_text_dataset_reader`, `llama_gguf_converter`, `llama_gguf_reader`) to improve modularity, extensibility, and testability.

*   **Preview functionality**: Allows you to view metadata and the first few sequences of the generated GGUF file, including optional detokenization.


GGUF Structure for Training Data
--------------------------------

The generated GGUF files follow a specific structure for training data:

*   **Metadata (KV pairs)**: All metadata keys are prefixed with `training.` to avoid conflicts with model metadata.

    *   `training.format.version`: `string` (e.g., "1.0") - Specification version.

    *   `training.dataset.name`: `string` (optional) - Dataset name (e.g., "OpenWebText-ru").

    *   `training.dataset.source`: `string` (optional) - URL or description of the data source.

    *   `training.file.creation_date`: `string` (ISO 8601) - File creation date.

    *   `training.tokenizer.gguf.model`: `string` - Tokenizer model name (e.g., "llama", "gpt2", "bert").

    *   `training.tokenizer.gguf.vocab`: `array[string]` - Tokenizer vocabulary.

    *   `training.tokenizer.gguf.merges`: `array[string]` (optional) - Tokenizer merges (for BPE).

    *   `training.tokenizer.gguf.pre`: `string` (optional) - Architecture of the pre-tokenizer.

    *   `training.sequence.count`: `uint64` - Total number of sequences in the file.

*   **Tensors**: Each training sequence is stored as a separate tensor.

    *   **Naming**: `training.tensor.{index}` (e.g., `training.tensor.0`, `training.tensor.1`, ...). No leading zeros.

    *   **Data type**: `GGML_TYPE_I32` (standard for tokens in `llama.cpp`).

    *   **Shape**: `[sequence_length]` - One-dimensional array. `sequence_length` will vary for each tensor and can be obtained from the tensor's shape.


Building
--------

It is assumed that you have already set up the `llama.cpp` build environment (e.g., using CMake) and installed Arrow and Parquet on your system.

1.  **Clone the `llama.cpp` repository**:

        git clone https://github.com/ggerganov/llama.cpp.git
        cd llama.cpp


2.  **Create a build directory and navigate into it**:

        mkdir build
        cd build


3.  **Configure and build the project using CMake**:

        cmake -DLLAMA_PARQUET=ON ..
        cmake --build . --config Release


    The `convert-to-train-gguf` utility will be built in the `build/bin` directory.


Usage
-----

    ./bin/convert-to-train-gguf [options]


### Command-line Options

*   `-h`, `--help`: Show this help message and exit.

*   `-m <path>, --model <path>` : Path to the GGUF model used for the tokenizer (default: `models/7B/ggml-model-f16.gguf`).

*   `--in-file <path>`: Path to the input dataset file, either a plain text file or a Parquet file (default: `input.txt`).

*   `-o <path>`, `--output <path>`: Path to save the output GGUF file to (default: `output.gguf`).

*   `--max-seq-len <length>`: Maximum sequence length in tokens (default: `2048`). Sequences exceeding this length will be truncated.

*   `--pre-tokenized`: Specifies that the input file contains pre-tokenized data (space-separated token IDs) rather than raw text.

*   `--dataset-format <type>`: Type of input data (`text`, `parquet`). (default: `text`).

*   `--parquet-text-column <name>`: For `parquet` input type, the column name containing raw text data (default: `text`).

*   `--parquet-tokens-column <name>`: For `parquet` input type, the column name containing pre-tokenized data (list of int32) (default: `tokens`).

*   `--preview`: Enables previewing of the generated GGUF file (prints metadata and the first few sequences).

*   `--preview-count <N>`: Number of sequences to preview (default: `1`). Requires `--preview`.

*   `--detokenize-preview`: Detokenize previewed sequences back into text for better readability. Requires `--preview`.


### Usage Examples

1.  **Converting a plain text file**:

        ./bin/convert-to-train-gguf -m models/7B/ggml-model-f16.gguf -i my_dataset.txt -o my_training_data.gguf -l 1024


2.  **Converting a pre-tokenized file**:

        ./bin/convert-to-train-gguf -m models/7B/ggml-model-f16.gguf -i pre_tokenized_data.txt -o pre_tokenized_training_data.gguf -p


    (Assumes `pre_tokenized_data.txt` contains lines like: `101 200 300 102 ...`)

3.  **Converting a Parquet file with raw text**:

        ./bin/convert-to-train-gguf -m models/7B/ggml-model-f16.gguf -i my_parquet_dataset.parquet -o my_training_data.gguf -t parquet --parquet-text-column "document_text"


4.  **Converting a Parquet file with pre-tokenized data**:

        ./bin/convert-to-train-gguf -m models/7B/ggml-model-f16.gguf -i my_tokenized_parquet.parquet -o my_training_data.gguf -t parquet -p --parquet-tokens-column "token_ids"


5.  **Converting with a preview of 5 sequences and detokenization**:

        ./bin/convert-to-train-gguf -m models/7B/ggml-model-f16.gguf -i my_dataset.txt -o my_training_data.gguf --preview --preview-count 5 --detokenize-preview



Future Improvements
-------------------

*   **Improved Error Handling**: More detailed messages and handling of edge cases.

*   **Additional Validation**: Data integrity checks at various stages.

*   **Dataset Statistics**: Ability to output statistics on sequence lengths, token distribution, etc.
