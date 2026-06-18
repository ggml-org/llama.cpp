#include "../tools/tts/higgs-sampler.h"

#include <cassert>
#include <random>
#include <vector>

static void test_delay_pattern_roundtrip() {
    const std::vector<std::vector<int>> raw = {
        { 1, 2, 3 },
        { 4, 5, 6 },
    };

    const std::vector<std::vector<int>> expected = {
        { 1, higgs_audio::BOC_ID, higgs_audio::BOC_ID },
        { 4, 2,                  higgs_audio::BOC_ID },
        { higgs_audio::EOC_ID, 5, 3 },
        { higgs_audio::EOC_ID, higgs_audio::EOC_ID, 6 },
    };

    const auto delayed = higgs_audio::apply_delay_pattern(raw, 3);
    assert(delayed == expected);
    assert(higgs_audio::reverse_delay_pattern(delayed, 3) == raw);
}

static void test_sampler_delay_window() {
    higgs_audio::delay_sampler sampler(3);

    assert((sampler.step_from_codes({ 10, 20, 30 }) == std::vector<int>{ 10, higgs_audio::BOC_ID, higgs_audio::BOC_ID }));
    assert((sampler.step_from_codes({ 11, 21, 31 }) == std::vector<int>{ 11, 21, higgs_audio::BOC_ID }));
    assert((sampler.step_from_codes({ 12, 22, 32 }) == std::vector<int>{ 12, 22, 32 }));
    assert(sampler.delay_steps() == 3);
    assert(!sampler.done());
}

static void test_sampler_eoc_winddown() {
    higgs_audio::delay_sampler sampler(3);

    (void) sampler.step_from_codes({ 10, 20, 30 });
    (void) sampler.step_from_codes({ 11, 21, 31 });
    (void) sampler.step_from_codes({ 12, 22, 32 });

    assert((sampler.step_from_codes({ higgs_audio::EOC_ID, 40, 41 }) == std::vector<int>{ higgs_audio::EOC_ID, 40, 41 }));
    assert(sampler.pending_eoc_steps() == 1);
    assert(!sampler.done());

    assert((sampler.step_from_codes({ 50, 51, 52 }) == std::vector<int>{ 50, 51, 52 }));
    assert(sampler.done());

    assert((sampler.step_from_codes({ 60, 61, 62 }) == std::vector<int>{ higgs_audio::STOP_CODE, higgs_audio::STOP_CODE, higgs_audio::STOP_CODE }));
}

static void test_sampler_greedy() {
    higgs_audio::delay_sampler sampler(2);

    const std::vector<std::vector<float>> logits = {
        { -1.0f, 3.0f, 2.0f },
        { 0.0f, -2.0f, 5.0f },
    };

    assert((sampler.step_greedy(logits) == std::vector<int>{ 1, higgs_audio::BOC_ID }));
}

static void test_codebook_frame_embedding() {
    const higgs_audio::codebook_layout layout {
        /* num_codebooks = */ 2,
        /* codebook_size = */ 3,
        /* n_embd        = */ 2,
    };

    const std::vector<float> embedding_weight = {
        1.0f, 0.0f,
        2.0f, 0.0f,
        3.0f, 0.0f,
        0.0f, 10.0f,
        0.0f, 20.0f,
        0.0f, 30.0f,
    };

    const auto embd = higgs_audio::embed_codebook_frame(layout, embedding_weight, { 2, 1 });
    assert((embd == std::vector<float>{ 3.0f, 20.0f }));
}

static void test_codebook_head_projection() {
    const higgs_audio::codebook_layout layout {
        /* num_codebooks = */ 2,
        /* codebook_size = */ 3,
        /* n_embd        = */ 2,
    };

    const std::vector<float> head_weight = {
        1.0f, 0.0f,
        2.0f, 0.0f,
        3.0f, 0.0f,
        0.0f, 10.0f,
        0.0f, 20.0f,
        0.0f, 30.0f,
    };

    const auto logits = higgs_audio::project_codebook_logits(layout, head_weight, { 1.0f, 2.0f });

    assert((logits[0] == std::vector<float>{ 1.0f, 2.0f, 3.0f }));
    assert((logits[1] == std::vector<float>{ 20.0f, 40.0f, 60.0f }));
}

static void test_greedy_codebook_codes_from_hidden() {
    const higgs_audio::codebook_layout layout {
        /* num_codebooks = */ 2,
        /* codebook_size = */ 3,
        /* n_embd        = */ 2,
    };

    const std::vector<float> head_weight = {
        1.0f, 0.0f,
        2.0f, 0.0f,
        3.0f, 0.0f,
        0.0f, 10.0f,
        0.0f, 20.0f,
        0.0f, 30.0f,
    };

    higgs_audio::delay_sampler sampler(2);
    const auto codes = higgs_audio::step_greedy_from_hidden(sampler, layout, head_weight, { 1.0f, 2.0f });

    assert((codes == std::vector<int>{ 2, higgs_audio::BOC_ID }));
}

static void test_sampling_temperature_zero_is_greedy() {
    const higgs_audio::codebook_layout layout {
        /* num_codebooks = */ 2,
        /* codebook_size = */ 3,
        /* n_embd        = */ 2,
    };

    const std::vector<std::vector<float>> logits = {
        { 0.0f, 5.0f, 1.0f },
        { 8.0f, 4.0f, 2.0f },
    };

    higgs_audio::codebook_sampling_params params;
    params.temperature = 0.0f;
    params.top_k = 1;
    std::mt19937 rng(1234);

    const auto codes = higgs_audio::sample_codebook_codes(layout, logits, params, rng);
    assert((codes == std::vector<int>{ 1, 0 }));
}

static void test_sampling_top_k_one_is_argmax() {
    const higgs_audio::codebook_layout layout {
        /* num_codebooks = */ 2,
        /* codebook_size = */ 4,
        /* n_embd        = */ 2,
    };

    const std::vector<float> flat_logits = {
        -2.0f, 1.0f, 7.0f, 3.0f,
        0.0f, 9.0f, 2.0f, 1.0f,
    };

    higgs_audio::codebook_sampling_params params;
    params.temperature = 0.8f;
    params.top_k = 1;
    std::mt19937 rng(4321);

    const auto codes = higgs_audio::sample_codebook_codes(layout, flat_logits, params, rng);
    assert((codes == std::vector<int>{ 2, 1 }));
}

int main() {
    test_delay_pattern_roundtrip();
    test_sampler_delay_window();
    test_sampler_eoc_winddown();
    test_sampler_greedy();
    test_codebook_frame_embedding();
    test_codebook_head_projection();
    test_greedy_codebook_codes_from_hidden();
    test_sampling_temperature_zero_is_greedy();
    test_sampling_top_k_one_is_argmax();

    return 0;
}
