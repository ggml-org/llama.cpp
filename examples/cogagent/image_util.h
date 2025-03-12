#include <stddef.h>
#include <stdint.h>
// Copied from the LLAVA example

struct clip_image_size {
    int width;
    int height;
};

struct clip_image_u8_batch {
    struct clip_image_u8 * data;
    size_t size;
};

struct clip_image_f32_batch {
    struct clip_image_f32 * data;
    size_t size;
};

struct clip_image_size * clip_image_size_init();
struct clip_image_u8  * clip_image_u8_init ();
struct clip_image_f32 * clip_image_f32_init();

void clip_image_u8_free (struct clip_image_u8  * img);
void clip_image_f32_free(struct clip_image_f32 * img);
void clip_image_u8_batch_free (struct clip_image_u8_batch  * batch);
void clip_image_f32_batch_free(struct clip_image_f32_batch * batch);

bool clip_image_load_from_file(const char * fname, struct clip_image_u8 * img);

/** interpret bytes as an image file with length bytes_length, and use the result to populate img */
bool clip_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct clip_image_u8 * img);
bool load_and_stretch_image(const char* path, int output_size, std::vector<float> &output_data,
                            const float mean[3], const float std[3]);