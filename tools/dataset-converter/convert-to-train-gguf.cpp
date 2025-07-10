// Main utility for converting a text dataset to the GGUF format for training models in llama.cpp.
//
// Logic:
// 1. Parses command line arguments.
// 2. Loads the tokenizer model.
// 3. Uses the llama_gguf_converter class to perform the entire conversion process:
//    - First pass over the input data to collect metadata (sequence lengths).
//    - Creation of the GGUF file and writing all collected metadata to it.
//    - Second pass over the input data to add each sequence as a separate tensor to the GGUF file.
// 4. After successful conversion, uses llama_gguf_reader to read and print
//    some meta-information and the first record from the created GGUF file.
//
// This two-pass approach allows processing datasets significantly larger than
// available RAM.

#include <algorithm>  // For std::min
#include <array>      // For std::array
#include <cinttypes>  // For PRIu64
#include <iostream>
#include <string>
#include <vector>

#include "arg.h"
#include "common.h"
#include "dataset-to-gguf/llama-gguf-converter.h"
#include "dataset-to-gguf/llama-gguf-reader.h"
#include "llama.h"  // For llama_backend_init, llama_backend_free, llama_model_load_from_file, llama_model_free

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    // Print parameters for verification
    printf("Parameters:\n");
    printf("  Model for tokenizer: %s\n", params.model.path.c_str());
    printf("  Input files: ");
    for (auto & i : params.in_files) {
        printf("%s ", i.c_str());
    }
    printf("\n  Output file: %s\n", params.out_file.c_str());
    printf("  Max sequence length: %d\n", params.max_seq_len);
    printf("  Pre-tokenized input: %s\n", params.pre_tokenized ? "Yes" : "No");
    printf("  Input type: %s\n", params.dataset_format.c_str());
    printf("  Do preview: %s\n", params.do_preview ? "Yes" : "No");
    if (params.do_preview) {
        printf("  Preview count: %d\n", params.preview_count);
        printf("  Detokenize preview: %s\n", params.detokenize_preview ? "Yes" : "No");
    }
#ifdef LLAMA_PARQUET
    if (params.dataset_format == "parquet") {
        printf("  Parquet text column: %s\n", params.parquet_text_column.c_str());
        printf("  Parquet tokens column: %s\n", params.parquet_tokens_column.c_str());
    }
#endif
    printf("\n");

    // Initialize llama.cpp
    llama_backend_init();

    // Load the model for its tokenizer
    llama_model_params model_params = llama_model_default_params();
    llama_model *model        = llama_model_load_from_file(params.model.path.c_str(), model_params);

    if (model == nullptr) {
        fprintf(stderr, "error: failed to load model from %s\n", params.model.path.c_str());
        llama_backend_free();
        return 1;
    }

    // --- Diagnostic Test: Reading tokenizer model GGUF file ---
    printf("--- Diagnostic Test: Reading tokenizer model GGUF file ---\n");
    try {
        llama_gguf_reader tokenizer_model_reader(params.model.path);
        if (tokenizer_model_reader.llama_gguf_reader_is_initialized()) {
            printf("  Tokenizer Model GGUF file opened successfully.\n");
            printf("  Tokenizer Model Name: %s\n",
                   tokenizer_model_reader.llama_gguf_reader_get_metadata_str("general.name", "N/A").c_str());
            printf("  Tokenizer Model Architecture: %s\n",
                   tokenizer_model_reader.llama_gguf_reader_get_metadata_str("general.architecture", "N/A").c_str());
            printf("  Tokenizer Model Tensor Count: %llu\n",
                   static_cast<long long>(tokenizer_model_reader.llama_gguf_reader_get_tensor_count()));
            printf("  Diagnostic Test: Tokenizer Model GGUF read successful.\n");
        } else {
            fprintf(stderr, "error: Diagnostic Test: Tokenizer Model GGUF read failed to initialize.\n");
            llama_model_free(model);  // Free model before exiting
            llama_backend_free();
            return 1;
        }
    } catch (const std::runtime_error & e) {
        fprintf(stderr, "error: Diagnostic Test: Tokenizer Model GGUF read failed: %s\n", e.what());
        llama_model_free(model);  // Free model before exiting
        llama_backend_free();
        return 1;
    }
    printf("--- End of Diagnostic Test ---\n\n");

    // Create and run the converter
    llama_gguf_converter converter;
    bool success = converter.llama_gguf_converter_convert(params, model);

    if (!success) {
        fprintf(stderr, "error: GGUF conversion failed.\n");
        llama_model_free(model); // Free model on conversion failure
        llama_backend_free();
        return 1;
    }

    printf("Conversion successful!\n");
    printf("Output file: %s\n", params.out_file.c_str());

    // --- Preview generated GGUF file (if requested) ---
    if (params.do_preview) {
        printf("\n--- Previewing generated GGUF file ---\n");
        try {
            llama_gguf_reader reader(params.out_file);

            if (!reader.llama_gguf_reader_is_initialized()) {
                fprintf(stderr, "error: llama_gguf_reader failed to initialize for preview.\n");
                llama_model_free(model); // Free model before exiting
                llama_backend_free();
                return 1;
            }

            printf("  Dataset Name: %s\n",
                   reader.llama_gguf_reader_get_metadata_str("training.dataset.name", "N/A").c_str());
            printf("  Sequence Count: %llu\n", static_cast<long long>(reader.llama_gguf_reader_get_metadata_u64("training.sequence.count", 0)));
            printf("  Tokenizer Model: %s\n",
                   reader.llama_gguf_reader_get_metadata_str("training.tokenizer.gguf.model", "N/A").c_str());

            int64_t tensor_count = reader.llama_gguf_reader_get_tensor_count();
            if (tensor_count > 0) {
                // Print N first sequences
                for (int64_t i = 0; i < std::min((int64_t) params.preview_count, tensor_count); ++i) {
                    printf("  Sequence (training.tensor.%" PRId64 "):\n", i);
                    std::vector<llama_token> sequence_tokens;
                    if (reader.llama_gguf_reader_read_tensor_data(i, sequence_tokens)) {
                        printf("    Length: %zu tokens\n", sequence_tokens.size());
                        printf("    Tokens: [");
                        for (size_t j = 0; j < std::min((size_t) 10, sequence_tokens.size());
                             ++j) {  // Print up to 10 tokens
                            printf("%d%s", sequence_tokens[j],
                                   (j == std::min((size_t) 10, sequence_tokens.size()) - 1) ? "" : ", ");
                        }
                        if (sequence_tokens.size() > 10) {
                            printf("...");
                        }
                        printf("]\n");

                        if (params.detokenize_preview) {
                            // Detokenization
                            std::string detokenized_text = "";
                            // Buffer for a single token
                            std::array<char, 256> piece_buf;  // Large enough buffer for a single token
                            // Ensure model is valid before calling llama_model_get_vocab
                            if (model != nullptr) {
                                for (llama_token token : sequence_tokens) {
                                    int n_chars = llama_token_to_piece(llama_model_get_vocab(model), token,
                                                                       piece_buf.data(), piece_buf.size(), 1, false);
                                    if (n_chars > 0) {
                                        detokenized_text.append(piece_buf.data(), n_chars);
                                    }
                                }
                                printf("    Detokenized: \"%s\"\n", detokenized_text.c_str());
                            } else {
                                fprintf(stderr, "    Warning: Cannot detokenize preview, model is null.\n");
                            }
                        }

                    } else {
                        fprintf(stderr, "    Error: Could not read data for sequence %" PRId64 ".\n", i);
                    }
                }
            } else {
                printf("  No sequences found in the GGUF file.\n");
            }

        } catch (const std::runtime_error & e) {
            fprintf(stderr, "error: GGUF preview failed: %s\n", e.what());
            llama_model_free(model); // Free model before exiting
            llama_backend_free();
            return 1;
        }
        printf("--- End of GGUF file preview ---\n");
    }

    // Clean up llama model and backend after all usage
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
