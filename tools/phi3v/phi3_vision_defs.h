#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

// Patch size is fixed for Phi-3-Vision's CLIP
const int PHI3V_PATCH_SIZE = 336;
const uint8_t WHITE_PADDING = 255; // Padding color must be white (255)

// Represents an 8-bit RGB image.
struct image_u8 {
    int w = 0;      // width
    int h = 0;      // height
    int n_comp = 3; // number of components (RGB)
    std::vector<uint8_t> data;
};

// Represents the calculated grid size (in 336x336 patches).
struct Phi3Grid {
    int w = 1; // number of patches wide
    int h = 1; // number of patches high
};

// Represents the output batch of images ready for the CLIP encoder.
struct Phi3ImageBatch {
    // Note: The global image is always batch[0].
    std::vector<image_u8> batch;
    Phi3Grid grid;
};