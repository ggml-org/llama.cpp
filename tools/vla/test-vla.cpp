// Standalone numerical-alignment test for a VLA model component.
//
// Built with tools/vla when LLAMA_BUILD_VLA=ON.
//
// Usage:
//   ./build/bin/test-vla <vla.gguf> <reference-dir>
//
// <reference-dir> must contain raw f32 little-endian files produced by the
// matching PyTorch implementation:
//   vl_embs.bin      [S, cross_dim]   (S inferred from file size)
//   state.bin        [state_dim]
//   noise.bin        [horizon, action_dim]
//   actions_ref.bin  [horizon, action_dim]
//
// Pass criteria (CUDA backend): MAE < 1e-4 and max|diff| < 5e-3.

#include "vla.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static bool read_f32_file(const std::string & path, std::vector<float> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz % 4 != 0) {
        std::fprintf(stderr, "bad size %ld for %s\n", sz, path.c_str());
        std::fclose(f);
        return false;
    }
    out.resize((size_t) sz / 4);
    const size_t rd = std::fread(out.data(), 1, (size_t) sz, f);
    std::fclose(f);
    return rd == (size_t) sz;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <vla.gguf> <reference-dir>\n", argv[0]);
        return 1;
    }
    const std::string gguf_path = argv[1];
    const std::string ref_dir   = argv[2];

    vla_context_params params = vla_context_params_default();
    vla_context * ctx = vla_init_from_file(gguf_path.c_str(), nullptr, params);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: model load\n");
        return 1;
    }

    const int64_t cross_dim = vla_conditioning_dim(ctx);
    const int64_t state_dim = vla_state_dim(ctx);
    const int64_t action_dim = vla_action_dim(ctx);
    const int64_t horizon = vla_action_horizon(ctx);

    std::vector<float> vl, state, noise, ref;
    if (!read_f32_file(ref_dir + "/vl_embs.bin", vl) ||
        !read_f32_file(ref_dir + "/state.bin", state) ||
        !read_f32_file(ref_dir + "/noise.bin", noise) ||
        !read_f32_file(ref_dir + "/actions_ref.bin", ref)) {
        vla_free(ctx);
        return 1;
    }

    if ((int64_t) vl.size() % cross_dim != 0) {
        std::fprintf(stderr, "FAIL: vl_embs.bin size %zu not a multiple of cross_dim %lld\n",
                     vl.size(), (long long) cross_dim);
        vla_free(ctx);
        return 1;
    }
    const int64_t S = (int64_t) vl.size() / cross_dim;

    if ((int64_t) state.size() != state_dim ||
        (int64_t) noise.size() != horizon * action_dim ||
        (int64_t) ref.size()   != horizon * action_dim) {
        std::fprintf(stderr, "FAIL: bad input sizes (state=%zu noise=%zu ref=%zu)\n",
                     state.size(), noise.size(), ref.size());
        vla_free(ctx);
        return 1;
    }

    std::printf("test-vla: type=%s S=%lld cross=%lld state=%lld horizon=%lld action=%lld\n",
                vla_model_type(ctx),
                (long long) S, (long long) cross_dim, (long long) state_dim,
                (long long) horizon, (long long) action_dim);

    std::vector<float> out((size_t) horizon * action_dim, 0.0f);
    const int embodiment_id = 0;
    vla_input input = {
        /*.embeddings    =*/ vl.data(),
        /*.n_tokens      =*/ S,
        /*.n_embd        =*/ cross_dim,
        /*.state         =*/ state.data(),
        /*.n_state       =*/ (int64_t) state.size(),
        /*.noise         =*/ noise.data(),
        /*.n_noise       =*/ (int64_t) noise.size(),
        /*.embodiment_id =*/ embodiment_id,
    };
    vla_output output = {
        /*.actions  =*/ out.data(),
        /*.capacity =*/ (int64_t) out.size(),
    };
    if (!vla_predict(ctx, &input, &output)) {
        std::fprintf(stderr, "FAIL: predict\n");
        vla_free(ctx);
        return 1;
    }

    double sum_abs = 0.0, max_abs = 0.0;
    size_t max_idx = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double d = std::fabs((double) out[i] - (double) ref[i]);
        sum_abs += d;
        if (d > max_abs) { max_abs = d; max_idx = i; }
    }
    const double mae = sum_abs / out.size();

    std::printf("MAE      = %.3e\n", mae);
    std::printf("max|diff|= %.3e  (at [%zu][%zu]: got %.6f, ref %.6f)\n",
                max_abs, max_idx / action_dim, max_idx % action_dim,
                out[max_idx], ref[max_idx]);

    // print a few sample values
    std::printf("sample row 0 : got  [%.5f %.5f %.5f %.5f ...]\n", out[0], out[1], out[2], out[3]);
    std::printf("               ref  [%.5f %.5f %.5f %.5f ...]\n", ref[0], ref[1], ref[2], ref[3]);

    const bool pass = mae < 1e-4 && max_abs < 5e-3;
    std::printf("%s\n", pass ? "PASS" : "FAIL");

    vla_free(ctx);
    return pass ? 0 : 1;
}
