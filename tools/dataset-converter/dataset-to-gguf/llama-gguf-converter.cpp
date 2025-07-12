// Utility for converting a text dataset to the GGUF format for training models in llama.cpp.
//
// Logic:
// 1. Loads the tokenizer model.
// 2. Performs a first pass over the input data to collect metadata (sequence lengths).
// 3. Creates a GGUF file and writes all collected metadata to it.
// 4. Performs a second pass over the input data to add each sequence as a separate tensor to the GGUF file.
//
// This two-pass approach allows processing datasets significantly larger than
// available RAM.

#include "llama-gguf-converter.h"  // Include the new header name for the converter

#include <cinttypes>               // For PRIu64
#include <cstdio>                  // For fprintf, snprintf
#include <memory>                  // For std::unique_ptr
#include <stdexcept>               // For std::runtime_error
#include <vector>                  // For std::vector

// Include the refactored GGUF and data reader headers
#include "common.h"
#include "llama-dataset-reader/llama-dataset-reader.h"
#include "llama-dataset-reader/llama-parquet-data-reader.h"
#include "llama-dataset-reader/llama-text-data-reader.h"
#include "llama-gguf-file.h"
#include "llama-gguf-writer.h"

// Method to execute the conversion process.
bool llama_gguf_converter::llama_gguf_converter_convert(const struct common_params & params,
                                                        const struct llama_model *   model) {
    // --- Create DataReader based on input_type ---
    std::unique_ptr<llama_dataset_reader> reader;
    if (params.dataset_format == "text") {
        reader = std::make_unique<llama_text_dataset_reader>(model, params.max_seq_len, params.pre_tokenized);
#ifdef LLAMA_PARQUET
    } else if (params.dataset_format == "parquet") {
        reader = std::make_unique<llama_parquet_dataset_reader>(model, params.max_seq_len, params.pre_tokenized, params.dataset_column);
#endif
    } else {
        fprintf(stderr, "error: Unsupported input type: %s\n", params.dataset_format.c_str());
        return false;
    }

    // Open the data source
    if (!reader->open(params.in_files[0])) {  //now only first file
        fprintf(stderr, "error: Failed to open data source %s\n", params.in_files[0].c_str());
        return false;
    }

    uint64_t total_sequence_count = 0;
    std::vector<uint32_t> sequence_lengths;  // Will store sequence lengths for text files

    // --- FIRST PASS: Collect sequence lengths or get total count ---
    printf("First pass: Reading input data and getting sequence lengths...\n");

    if (params.dataset_format == "parquet") {
        // For Parquet, get total sequence count from metadata
        total_sequence_count = reader->total_sequences();
        printf("First pass complete. Found %" PRIu64 " sequences (from Parquet metadata).\n\n", total_sequence_count);
    } else {  // For text files
        // For text files, perform a full first pass to count sequences
        // and their lengths (as this is the only way to know the exact token count).
        std::vector<llama_token> tokens;
        while (reader->read_next_sequence(tokens)) {
            sequence_lengths.push_back(tokens.size());
        }
        total_sequence_count = sequence_lengths.size();
        printf("First pass complete. Found %" PRIu64 " sequences.\n\n", total_sequence_count);
    }

    // --- WRITE GGUF FILE ---
    printf("Creating GGUF file...\n");
    // Create a llama_gguf_file instance, which will manage the GGUF context
    std::unique_ptr<llama_gguf_file> gguf_file;
    try {
        gguf_file = std::make_unique<llama_gguf_file>();
    } catch (const std::runtime_error & e) {
        fprintf(stderr, "error: Failed to initialize llama_gguf_file: %s\n", e.what());
        return false;
    }

    // Pass the pointer to gguf_file to llama_gguf_writer
    llama_gguf_writer writer(gguf_file.get());

    // Initialize GGUF file metadata
    writer.llama_gguf_writer_init_metadata(model, params.in_files[0], total_sequence_count);
    printf("Metadata written.\n");

    // --- SECOND PASS: Write tensors ---
    printf("Second pass: Writing tensors to GGUF file...\n");
    if (!reader->reset()) {
        fprintf(stderr, "error: Failed to reset data reader for second pass.\n");
        return false;
    }

    uint64_t current_sequence_idx = 0;
    std::vector<llama_token> tokens;  // Reuse the tokens vector
    while (reader->read_next_sequence(tokens)) {
        if (current_sequence_idx >= total_sequence_count) {
            fprintf(stderr,
                    "error: file ended prematurely on second pass. Expected %" PRIu64
                    " sequences, but reached end of file at %" PRIu64 ".\n",
                    total_sequence_count, current_sequence_idx);
            break;
        }

        uint32_t expected_n_tokens;
        if (params.dataset_format == "text") {
            // For text files, use lengths collected in the first pass
            expected_n_tokens = sequence_lengths[current_sequence_idx];
        } else {
            // For Parquet, we don't know the expected length beforehand,
            // so just use the actual length of the read sequence.
            // If the Parquet file contains empty sequences, they will be handled.
            expected_n_tokens = tokens.size();
        }

        uint32_t actual_n_tokens = tokens.size();

        // If the number of tokens does not match (only for text where we know it beforehand),
        // this is a critical error, as metadata collected in the first pass will be incorrect for this tensor.
        // Abort conversion to avoid creating a corrupted GGUF file.
        if (params.dataset_format == "text" && actual_n_tokens != expected_n_tokens) {
            fprintf(stderr,
                    "error: Tokenization mismatch on second pass for sequence %" PRIu64
                    ". Expected %u tokens, got %u.\n",
                    current_sequence_idx, expected_n_tokens, actual_n_tokens);
            fprintf(
                stderr,
                "This indicates a non-deterministic tokenizer or an issue with input reading. Aborting conversion.\n");
            return false;  // Abort conversion
        }

        // Add tensor only if there are tokens
        if (actual_n_tokens > 0) {
            writer.llama_gguf_writer_add_sequence_tensor(current_sequence_idx, tokens);
        } else {
            // If 0 tokens were expected, but the line was not empty, print a warning
            // (This condition `expected_n_tokens != 0` is only relevant for text files,
            // where we might have gotten 0 tokens in the first pass for a non-empty line.)
            if (params.dataset_format == "text" && expected_n_tokens != 0) {
                fprintf(stderr, "warning: sequence %" PRIu64 " resulted in 0 tokens on second pass, but expected %u.\n",
                        current_sequence_idx, expected_n_tokens);
                // Continue, as this might be acceptable for some datasets,
                // but warn about potential inconsistency.
            }
        }
        current_sequence_idx++;
    }
    reader->close();  // Close DataReader after use
    printf("Second pass complete.\n\n");

    // Save file to disk
    printf("Writing GGUF data to %s...\n", params.out_file.c_str());
    if (!writer.llama_gguf_writer_write_to_file(params.out_file)) {
        fprintf(stderr, "error: Failed to write GGUF file %s\n", params.out_file.c_str());
        return false;
    }

    printf("Conversion successful!\n");
    printf("Output file: %s\n", params.out_file.c_str());

    return true;
}
