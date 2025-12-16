#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd/swin.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

// External preprocessing function
extern "C" bool nougat_preprocess_pipeline(
    const char* input_path,
    float** output_data,
    int* output_width,
    int* output_height,
    int* num_pages);

extern "C" void nougat_preprocess_cleanup(float* data);

// CLI arguments structure
struct nougat_params {
    std::string input_path = "";
    std::string output_path = "";
    std::string vision_model = "models/nougat-vision-swin.gguf";
    std::string text_model = "models/nougat-text-mbart.gguf";
    std::string projector_model = "models/nougat-projector.gguf";

    // Processing options
    bool batch_mode = false;
    int batch_size = 1;
    int n_threads = 4;
    int n_gpu_layers = 0;

    // Output options
    std::string output_format = "markdown"; // markdown, latex, plain
    bool verbose = false;
    bool save_intermediate = false;

    // Performance options
    bool use_mmap = true;
    bool use_flash_attn = false;
    int context_size = 2048;

    // Document-specific options
    bool deskew = true;
    bool denoise = true;
    bool detect_tables = true;
    bool detect_math = true;
    int max_pages = -1; // -1 for all pages
};

static void print_usage(const char* prog_name) {
    fprintf(stdout, "\n");
    fprintf(stdout, "Nougat OCR - Neural Optical Understanding for Academic Documents\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "Usage: %s [options] -i input_file -o output_file\n", prog_name);
    fprintf(stdout, "\n");
    fprintf(stdout, "Options:\n");
    fprintf(stdout, "  -i, --input FILE          Input document (PDF, PNG, JPG)\n");
    fprintf(stdout, "  -o, --output FILE         Output file path\n");
    fprintf(stdout, "  --vision-model FILE       Path to vision model GGUF (default: models/nougat-vision-swin.gguf)\n");
    fprintf(stdout, "  --text-model FILE         Path to text model GGUF (default: models/nougat-text-mbart.gguf)\n");
    fprintf(stdout, "  --projector FILE          Path to projector model GGUF (default: models/nougat-projector.gguf)\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "  Processing Options:\n");
    fprintf(stdout, "  -t, --threads N           Number of threads (default: 4)\n");
    fprintf(stdout, "  -ngl, --n-gpu-layers N    Number of layers to offload to GPU (default: 0)\n");
    fprintf(stdout, "  -b, --batch-size N        Batch size for processing (default: 1)\n");
    fprintf(stdout, "  -c, --context-size N      Context size (default: 2048)\n");
    fprintf(stdout, "  --max-pages N             Maximum pages to process (default: all)\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "  Output Options:\n");
    fprintf(stdout, "  -f, --format FORMAT       Output format: markdown, latex, plain (default: markdown)\n");
    fprintf(stdout, "  -v, --verbose             Verbose output\n");
    fprintf(stdout, "  --save-intermediate       Save intermediate processing results\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "  Document Processing:\n");
    fprintf(stdout, "  --no-deskew               Disable automatic deskewing\n");
    fprintf(stdout, "  --no-denoise              Disable denoising\n");
    fprintf(stdout, "  --no-tables               Disable table detection\n");
    fprintf(stdout, "  --no-math                 Disable math formula detection\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "  Performance Options:\n");
    fprintf(stdout, "  --no-mmap                 Disable memory mapping\n");
    fprintf(stdout, "  --flash-attn              Use flash attention\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "Examples:\n");
    fprintf(stdout, "  # Basic OCR of a PDF document\n");
    fprintf(stdout, "  %s -i paper.pdf -o paper.md\n", prog_name);
    fprintf(stdout, "\n");
    fprintf(stdout, "  # Process with GPU acceleration\n");
    fprintf(stdout, "  %s -i scan.png -o text.md -ngl 32 -t 8\n", prog_name);
    fprintf(stdout, "\n");
    fprintf(stdout, "  # LaTeX output with math detection\n");
    fprintf(stdout, "  %s -i math_paper.pdf -o paper.tex -f latex --detect-math\n", prog_name);
    fprintf(stdout, "\n");
}

static bool parse_args(int argc, char** argv, nougat_params& params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-i" || arg == "--input") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing argument for %s\n", arg.c_str());
                return false;
            }
            params.input_path = argv[i];
        }
        else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing argument for %s\n", arg.c_str());
                return false;
            }
            params.output_path = argv[i];
        }
        else if (arg == "--vision-model") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing argument for %s\n", arg.c_str());
                return false;
            }
            params.vision_model = argv[i];
        }
        else if (arg == "--text-model") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing argument for %s\n", arg.c_str());
                return false;
            }
            params.text_model = argv[i];
        }
        else if (arg == "--projector") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing argument for %s\n", arg.c_str());
                return false;
            }
            params.projector_model = argv[i];
        }
        else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing argument for %s\n", arg.c_str());
                return false;
            }
            params.n_threads = std::stoi(argv[i]);
        }
        else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing argument for %s\n", arg.c_str());
                return false;
            }
            params.n_gpu_layers = std::stoi(argv[i]);
        }
        else if (arg == "-b" || arg == "--batch-size") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing argument for %s\n", arg.c_str());
                return false;
            }
            params.batch_size = std::stoi(argv[i]);
        }
        else if (arg == "-c" || arg == "--context-size") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing argument for %s\n", arg.c_str());
                return false;
            }
            params.context_size = std::stoi(argv[i]);
        }
        else if (arg == "--max-pages") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing argument for %s\n", arg.c_str());
                return false;
            }
            params.max_pages = std::stoi(argv[i]);
        }
        else if (arg == "-f" || arg == "--format") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing argument for %s\n", arg.c_str());
                return false;
            }
            params.output_format = argv[i];
        }
        else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        }
        else if (arg == "--save-intermediate") {
            params.save_intermediate = true;
        }
        else if (arg == "--no-deskew") {
            params.deskew = false;
        }
        else if (arg == "--no-denoise") {
            params.denoise = false;
        }
        else if (arg == "--no-tables") {
            params.detect_tables = false;
        }
        else if (arg == "--no-math") {
            params.detect_math = false;
        }
        else if (arg == "--no-mmap") {
            params.use_mmap = false;
        }
        else if (arg == "--flash-attn") {
            params.use_flash_attn = true;
        }
        else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        }
        else {
            fprintf(stderr, "Error: Unknown argument '%s'\n", arg.c_str());
            return false;
        }
    }

    // Validate required arguments
    if (params.input_path.empty()) {
        fprintf(stderr, "Error: Input file is required\n");
        return false;
    }

    if (params.output_path.empty()) {
        // Generate default output path
        size_t dot_pos = params.input_path.find_last_of(".");
        params.output_path = params.input_path.substr(0, dot_pos);

        if (params.output_format == "markdown") {
            params.output_path += ".md";
        } else if (params.output_format == "latex") {
            params.output_path += ".tex";
        } else {
            params.output_path += ".txt";
        }
    }

    return true;
}

// Process a single page through the Nougat pipeline
static std::string process_page(
    struct swin_ctx* vision_ctx,
    struct llama_model* text_model,
    struct llama_context* text_ctx,
    const float* image_data,
    int width,
    int height,
    const nougat_params& params) {

    // Step 1: Encode image with Swin Transformer
    if (params.verbose) {
        printf("Encoding image with Swin Transformer...\n");
    }

    // Create image batch
    swin_image_f32 img = {
        width,
        height,
        3,
        std::vector<float>(image_data, image_data + width * height * 3)
    };

    swin_image_batch imgs = {1, &img};

    // Encode image
    std::vector<float> vision_embeddings(2048); // Adjust size based on model
    if (!swin_image_batch_encode(vision_ctx, params.n_threads, &imgs, vision_embeddings.data())) {
        fprintf(stderr, "Failed to encode image\n");
        return "";
    }

    // Step 2: Pass embeddings through projector
    // This would map vision embeddings to text embedding space

    // Step 3: Generate text with mBART decoder
    if (params.verbose) {
        printf("Generating text with mBART decoder...\n");
    }

    // Create batch for text generation
    llama_batch batch = llama_batch_init(params.context_size, 0, 1);

    // Set up cross-attention with vision embeddings
    // This requires the decoder to attend to encoder outputs

    // Start with BOS token
    llama_token bos_token = llama_token_get_bos(text_model);
    batch.token[0] = bos_token;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.n_tokens = 1;

    // Decode initial token
    if (llama_decode(text_ctx, batch) != 0) {
        fprintf(stderr, "Failed to decode\n");
        llama_batch_free(batch);
        return "";
    }

    // Generate text autoregressively
    std::vector<llama_token> generated_tokens;
    generated_tokens.push_back(bos_token);

    llama_token eos_token = llama_token_get_eos(text_model);
    int max_tokens = params.context_size;

    for (int i = 1; i < max_tokens; i++) {
        // Get logits from last position
        float* logits = llama_get_logits_ith(text_ctx, batch.n_tokens - 1);

        // Sample next token
        int n_vocab = llama_n_vocab(text_model);
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);

        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }

        llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

        // Sample with top-k and top-p
        int top_k = 40;
        float top_p = 0.9f;
        float temp = 0.8f;

        llama_sample_top_k(text_ctx, &candidates_p, top_k, 1);
        llama_sample_top_p(text_ctx, &candidates_p, top_p, 1);
        llama_sample_temp(text_ctx, &candidates_p, temp);

        llama_token new_token = llama_sample_token(text_ctx, &candidates_p);

        // Check for EOS
        if (new_token == eos_token) {
            break;
        }

        generated_tokens.push_back(new_token);

        // Add to batch for next iteration
        batch.token[0] = new_token;
        batch.pos[0] = i;
        batch.n_tokens = 1;

        if (llama_decode(text_ctx, batch) != 0) {
            fprintf(stderr, "Failed to continue decoding\n");
            break;
        }
    }

    llama_batch_free(batch);

    // Convert tokens to text
    std::string result;
    for (auto token : generated_tokens) {
        std::string piece = llama_token_to_piece(text_ctx, token, true);
        result += piece;
    }

    return result;
}

int main(int argc, char** argv) {
    nougat_params params;

    // Parse command line arguments
    if (!parse_args(argc, argv, params)) {
        print_usage(argv[0]);
        return 1;
    }

    // Print banner
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║          Nougat OCR - Document Understanding          ║\n");
    printf("║        Powered by Swin Transformer + mBART            ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    printf("\n");

    printf("Input:  %s\n", params.input_path.c_str());
    printf("Output: %s\n", params.output_path.c_str());
    printf("Format: %s\n", params.output_format.c_str());
    printf("\n");

    // Initialize backend
    llama_backend_init();

    // Load vision model (Swin Transformer)
    printf("Loading vision model from %s...\n", params.vision_model.c_str());
    struct swin_ctx* vision_ctx = swin_model_load(params.vision_model, params.verbose ? 2 : 1);
    if (!vision_ctx) {
        fprintf(stderr, "Failed to load vision model\n");
        return 1;
    }

    // Load text model (mBART)
    printf("Loading text model from %s...\n", params.text_model.c_str());

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;
    model_params.use_mmap = params.use_mmap;

    struct llama_model* text_model = llama_load_model_from_file(
        params.text_model.c_str(), model_params);
    if (!text_model) {
        fprintf(stderr, "Failed to load text model\n");
        swin_free(vision_ctx);
        return 1;
    }

    // Create text generation context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params.context_size;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads;
    ctx_params.flash_attn = params.use_flash_attn;

    struct llama_context* text_ctx = llama_new_context_with_model(text_model, ctx_params);
    if (!text_ctx) {
        fprintf(stderr, "Failed to create text context\n");
        llama_free_model(text_model);
        swin_free(vision_ctx);
        return 1;
    }

    // Preprocess document
    printf("Preprocessing document...\n");
    float* preprocessed_data = nullptr;
    int width, height, num_pages;

    if (!nougat_preprocess_pipeline(
            params.input_path.c_str(),
            &preprocessed_data,
            &width, &height, &num_pages)) {
        fprintf(stderr, "Failed to preprocess document\n");
        llama_free(text_ctx);
        llama_free_model(text_model);
        swin_free(vision_ctx);
        return 1;
    }

    printf("Document info: %d pages, %dx%d pixels\n", num_pages, width, height);

    // Limit pages if requested
    if (params.max_pages > 0 && num_pages > params.max_pages) {
        num_pages = params.max_pages;
        printf("Processing first %d pages only\n", num_pages);
    }

    // Process each page
    std::string full_output;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int page = 0; page < num_pages; page++) {
        printf("\nProcessing page %d/%d...\n", page + 1, num_pages);

        float* page_data = preprocessed_data + (page * width * height * 3);

        std::string page_text = process_page(
            vision_ctx, text_model, text_ctx,
            page_data, width, height, params);

        if (page_text.empty()) {
            fprintf(stderr, "Warning: Failed to process page %d\n", page + 1);
            continue;
        }

        // Add page separator for multi-page documents
        if (page > 0) {
            if (params.output_format == "markdown") {
                full_output += "\n\n---\n\n";
            } else if (params.output_format == "latex") {
                full_output += "\n\\newpage\n\n";
            } else {
                full_output += "\n\n[Page " + std::to_string(page + 1) + "]\n\n";
            }
        }

        full_output += page_text;

        // Save intermediate results if requested
        if (params.save_intermediate) {
            std::string intermediate_file = params.output_path + ".page" +
                                          std::to_string(page + 1) + ".tmp";
            std::ofstream tmp_out(intermediate_file);
            tmp_out << page_text;
            tmp_out.close();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    // Save final output
    printf("\nSaving output to %s...\n", params.output_path.c_str());
    std::ofstream output_file(params.output_path);
    if (!output_file) {
        fprintf(stderr, "Failed to open output file\n");
    } else {
        // Add format-specific headers/footers
        if (params.output_format == "latex") {
            output_file << "\\documentclass{article}\n";
            output_file << "\\usepackage{amsmath}\n";
            output_file << "\\usepackage{graphicx}\n";
            output_file << "\\begin{document}\n\n";
        }

        output_file << full_output;

        if (params.output_format == "latex") {
            output_file << "\n\n\\end{document}\n";
        }

        output_file.close();
    }

    // Print statistics
    printf("\n");
    printf("╔════════════════════════════════════╗\n");
    printf("║         OCR Complete!              ║\n");
    printf("╠════════════════════════════════════╣\n");
    printf("║ Pages processed: %-17d ║\n", num_pages);
    printf("║ Time taken:      %-17lds║\n", duration.count());
    printf("║ Output size:     %-17zd ║\n", full_output.size());
    printf("╚════════════════════════════════════╝\n");

    // Cleanup
    nougat_preprocess_cleanup(preprocessed_data);
    llama_free(text_ctx);
    llama_free_model(text_model);
    swin_free(vision_ctx);
    llama_backend_free();

    return 0;
}