#include "clip.h"
#include "swin.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

// External image loading library integration (stb_image)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#endif

// Document-specific preprocessing parameters for Nougat
struct nougat_preprocess_params {
    int target_width = 896;     // Nougat uses different resolution than standard vision models
    int target_height = 1344;   // Optimized for document aspect ratio
    float mean[3] = {0.485f, 0.456f, 0.406f};  // ImageNet normalization
    float std[3] = {0.229f, 0.224f, 0.225f};
    bool center_crop = false;   // Documents should not be center-cropped
    bool maintain_aspect = true; // Important for documents
    int patch_size = 4;         // Swin Transformer patch size
};

// Structure to hold document metadata
struct document_metadata {
    int original_width;
    int original_height;
    int num_pages;
    std::string format; // PDF, PNG, JPG, etc.
    float dpi;
};

// Preprocess a single document image for Nougat
static bool preprocess_document_image(
    const uint8_t* img_data,
    int width,
    int height,
    int channels,
    const nougat_preprocess_params& params,
    std::vector<float>& output) {

    // Calculate scaling to fit target dimensions while maintaining aspect ratio
    float scale_w = static_cast<float>(params.target_width) / width;
    float scale_h = static_cast<float>(params.target_height) / height;
    float scale = params.maintain_aspect ? std::min(scale_w, scale_h) : 1.0f;

    int new_width = static_cast<int>(width * scale);
    int new_height = static_cast<int>(height * scale);

    // Ensure dimensions are divisible by patch size
    new_width = (new_width / params.patch_size) * params.patch_size;
    new_height = (new_height / params.patch_size) * params.patch_size;

    // Resize image using bilinear interpolation
    std::vector<uint8_t> resized_img(new_width * new_height * 3);

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            float src_x = x / scale;
            float src_y = y / scale;

            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, width - 1);
            int y1 = std::min(y0 + 1, height - 1);

            float fx = src_x - x0;
            float fy = src_y - y0;

            for (int c = 0; c < 3; c++) {
                float v00 = img_data[(y0 * width + x0) * channels + c];
                float v10 = img_data[(y0 * width + x1) * channels + c];
                float v01 = img_data[(y1 * width + x0) * channels + c];
                float v11 = img_data[(y1 * width + x1) * channels + c];

                float v0 = v00 * (1 - fx) + v10 * fx;
                float v1 = v01 * (1 - fx) + v11 * fx;
                float v = v0 * (1 - fy) + v1 * fy;

                resized_img[(y * new_width + x) * 3 + c] = static_cast<uint8_t>(v);
            }
        }
    }

    // Pad to target size if needed
    int pad_left = (params.target_width - new_width) / 2;
    int pad_top = (params.target_height - new_height) / 2;

    output.resize(params.target_width * params.target_height * 3);

    // Initialize with padding (white background for documents)
    std::fill(output.begin(), output.end(), 1.0f);

    // Copy resized image to output with normalization
    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            int out_x = x + pad_left;
            int out_y = y + pad_top;

            if (out_x >= 0 && out_x < params.target_width &&
                out_y >= 0 && out_y < params.target_height) {

                for (int c = 0; c < 3; c++) {
                    float pixel = resized_img[(y * new_width + x) * 3 + c] / 255.0f;
                    pixel = (pixel - params.mean[c]) / params.std[c];
                    output[(out_y * params.target_width + out_x) * 3 + c] = pixel;
                }
            }
        }
    }

    return true;
}

// Load and preprocess a document file (supports various formats)
bool nougat_preprocess_document_file(
    const std::string& filename,
    nougat_preprocess_params& params,
    std::vector<std::vector<float>>& page_outputs,
    document_metadata& metadata) {

    // Check file extension
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    metadata.format = ext;

    if (ext == "pdf") {
        // PDF processing would require a PDF library like poppler or mupdf
        // For now, we'll return an error for PDF files
        fprintf(stderr, "PDF processing not yet implemented. Please convert to image format.\n");
        return false;
    }

    // Load image using stb_image
    int width, height, channels;
    unsigned char* img_data = stbi_load(filename.c_str(), &width, &height, &channels, 3);

    if (!img_data) {
        fprintf(stderr, "Failed to load image: %s\n", filename.c_str());
        return false;
    }

    metadata.original_width = width;
    metadata.original_height = height;
    metadata.num_pages = 1; // Single image
    metadata.dpi = 300.0f;  // Assume standard document DPI

    // Preprocess the image
    std::vector<float> output;
    bool success = preprocess_document_image(
        img_data, width, height, 3, params, output);

    if (success) {
        page_outputs.push_back(output);
    }

    stbi_image_free(img_data);
    return success;
}

// Batch preprocessing for multiple document pages
bool nougat_preprocess_document_batch(
    const std::vector<std::string>& filenames,
    nougat_preprocess_params& params,
    std::vector<std::vector<float>>& outputs) {

    outputs.clear();
    outputs.reserve(filenames.size());

    for (const auto& filename : filenames) {
        document_metadata metadata;
        std::vector<std::vector<float>> page_outputs;

        if (!nougat_preprocess_document_file(filename, params, page_outputs, metadata)) {
            fprintf(stderr, "Failed to preprocess: %s\n", filename.c_str());
            continue;
        }

        // Add all pages from this document
        for (auto& page : page_outputs) {
            outputs.push_back(std::move(page));
        }
    }

    return !outputs.empty();
}

// Apply document-specific augmentations
void nougat_augment_document(
    std::vector<float>& image_data,
    int width,
    int height,
    bool random_rotation = false,
    bool deskew = true,
    bool denoise = true) {

    // Document deskewing (straighten tilted scans)
    if (deskew) {
        // Simplified deskew - would need proper implementation
        // using Hough transform or similar technique
    }

    // Denoising for scanned documents
    if (denoise) {
        // Apply median filter or similar denoising
        // Simplified implementation
        std::vector<float> temp = image_data;

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                for (int c = 0; c < 3; c++) {
                    std::vector<float> neighborhood;

                    // Collect 3x3 neighborhood
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int idx = ((y + dy) * width + (x + dx)) * 3 + c;
                            neighborhood.push_back(temp[idx]);
                        }
                    }

                    // Median filter
                    std::sort(neighborhood.begin(), neighborhood.end());
                    image_data[(y * width + x) * 3 + c] = neighborhood[4];
                }
            }
        }
    }

    // Random rotation for augmentation during training
    if (random_rotation) {
        // Apply small random rotation (-5 to +5 degrees)
        // Would need proper rotation implementation
    }
}

// Extract text regions from document for focused processing
struct text_region {
    int x, y, width, height;
    float confidence;
};

std::vector<text_region> nougat_detect_text_regions(
    const std::vector<float>& image_data,
    int width,
    int height) {

    std::vector<text_region> regions;

    // Simple text detection based on connected components
    // This would need a proper implementation using:
    // - Edge detection
    // - Connected component analysis
    // - Text/non-text classification

    // For now, return the whole image as a single region
    text_region full_page;
    full_page.x = 0;
    full_page.y = 0;
    full_page.width = width;
    full_page.height = height;
    full_page.confidence = 1.0f;

    regions.push_back(full_page);

    return regions;
}

// Enhanced preprocessing for mathematical formulas
void nougat_preprocess_math_regions(
    std::vector<float>& image_data,
    int width,
    int height,
    const std::vector<text_region>& math_regions) {

    // Apply special preprocessing for mathematical content
    for (const auto& region : math_regions) {
        // Enhance contrast for mathematical symbols
        for (int y = region.y; y < region.y + region.height; y++) {
            for (int x = region.x; x < region.x + region.width; x++) {
                for (int c = 0; c < 3; c++) {
                    int idx = (y * width + x) * 3 + c;
                    float& pixel = image_data[idx];

                    // Increase contrast
                    pixel = (pixel - 0.5f) * 1.2f + 0.5f;
                    pixel = std::max(0.0f, std::min(1.0f, pixel));
                }
            }
        }
    }
}

// Table detection and preprocessing
struct table_region {
    text_region bounds;
    int rows, cols;
    std::vector<text_region> cells;
};

std::vector<table_region> nougat_detect_tables(
    const std::vector<float>& image_data,
    int width,
    int height) {

    std::vector<table_region> tables;

    // Table detection would require:
    // - Line detection (horizontal and vertical)
    // - Grid structure analysis
    // - Cell boundary detection

    // Placeholder implementation
    return tables;
}

// Main preprocessing pipeline for Nougat OCR
extern "C" bool nougat_preprocess_pipeline(
    const char* input_path,
    float** output_data,
    int* output_width,
    int* output_height,
    int* num_pages) {

    nougat_preprocess_params params;
    std::vector<std::vector<float>> page_outputs;
    document_metadata metadata;

    // Load and preprocess document
    if (!nougat_preprocess_document_file(
            input_path, params, page_outputs, metadata)) {
        return false;
    }

    // Apply document-specific processing
    for (auto& page : page_outputs) {
        // Detect text regions
        auto text_regions = nougat_detect_text_regions(
            page, params.target_width, params.target_height);

        // Apply augmentations
        nougat_augment_document(
            page, params.target_width, params.target_height,
            false, true, true);

        // Detect and process mathematical regions
        // (would need actual math detection)
        // nougat_preprocess_math_regions(page, width, height, math_regions);
    }

    // Prepare output
    if (!page_outputs.empty()) {
        *output_width = params.target_width;
        *output_height = params.target_height;
        *num_pages = page_outputs.size();

        // Allocate and copy data
        size_t total_size = params.target_width * params.target_height * 3 * page_outputs.size();
        *output_data = new float[total_size];

        size_t offset = 0;
        for (const auto& page : page_outputs) {
            std::copy(page.begin(), page.end(), *output_data + offset);
            offset += page.size();
        }

        return true;
    }

    return false;
}

// Cleanup function
extern "C" void nougat_preprocess_cleanup(float* data) {
    delete[] data;
}