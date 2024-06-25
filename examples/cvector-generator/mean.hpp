#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <string>
#include <vector>
#include <math.h>

namespace mean {

static void run(
        const std::vector<struct ggml_tensor *> & v_input, // shape of v_input[0]: [n_embd, n_samples]
        const std::vector<struct ggml_tensor *> & v_output) {
    printf("%s: Running mean...\n", __func__);
    for (size_t il = 0; il < v_input.size(); ++il) {
        // prepare output vector
        struct ggml_tensor * ctrl_out = v_output[il];
        ggml_format_name(ctrl_out, "direction.%ld", il+1);

        // calculate mean vector
        struct ggml_tensor * t_layer = v_input[il];
        GGML_ASSERT(t_layer->ne[0] == ctrl_out->ne[0]); // == n_embd
        for (int ic = 0; ic < t_layer->ne[0]; ic++) {
            float f = 0.0;
            for (int ir = 0; ir < t_layer->ne[1]; ir++) {
                f += ggml_get_f32_nd(t_layer, ic, ir, 0, 0);
            }
            f /= t_layer->ne[1];
            ggml_set_f32_1d(ctrl_out, ic, f);
        }

        // normalize output vector
        float norm = 0.0;
        for (int i = 0; i < ggml_nelements(ctrl_out); i++) {
            float f = ggml_get_f32_1d(ctrl_out, i);
            norm += f*f;
        }
        norm = sqrt(norm);
        for (int i = 0; i < ggml_nelements(ctrl_out); i++) {
            float f = ggml_get_f32_1d(ctrl_out, i);
            ggml_set_f32_1d(ctrl_out, i, f / norm);
        }

        printf("%s: Done layer %d / %d\n", __func__, (int) il+1, (int) v_input.size());
    }
}

}
