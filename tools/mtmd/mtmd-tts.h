/**
 * mtmd-tts.h - Text-to-Speech support for mtmd (multimodal) library
 *
 * This library provides TTS output capability for Qwen3-Omni and similar models.
 * It integrates with mtmd-cli to enable speech output from any input modality
 * (text, audio, image).
 *
 * Architecture:
 *   Input embeddings (from Thinker layer 18)
 *       -> Text Projection MLP (2048 -> 1024)
 *       -> Talker (20L MoE, generates codec tokens)
 *       -> Code Predictor (5L, expands 1 -> 16 codebooks)
 *       -> Code2Wav (HiFi-GAN vocoder, generates 24kHz audio)
 *       -> WAV output
 *
 * Usage:
 *   mtmd_tts_context * ctx = mtmd_tts_init(thinker_model, talker_model, params);
 *   mtmd_tts_generate(ctx, embeddings, n_tokens, samples, max_samples);
 *   mtmd_tts_write_wav("output.wav", samples, n_samples, 24000);
 *   mtmd_tts_free(ctx);
 */

#ifndef MTMD_TTS_H
#define MTMD_TTS_H

#include "llama.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque TTS context
struct mtmd_tts_context;
typedef struct mtmd_tts_context mtmd_tts_context;

// TTS parameters
struct mtmd_tts_params {
    // Sampling parameters for Talker
    float temperature;      // Sampling temperature (default: 0.9)
    int   top_k;            // Top-k sampling (default: 50)
    float top_p;            // Top-p (nucleus) sampling (default: 0.9)

    // Generation limits
    int   max_codec_tokens; // Maximum codec tokens to generate (default: 500)

    // Speaker selection (Qwen3-Omni speaker IDs)
    // 2301 = Chelsie, 2302 = Ethan, 2303 = Aiden
    int   speaker_id;

    // Output format
    int   sample_rate;      // Output sample rate (default: 24000)

    // Debugging
    bool  verbose;          // Enable verbose logging
    bool  cpu_only;         // Force CPU for Code2Wav (workaround for CUDA issues)
};

// Get default TTS parameters
struct mtmd_tts_params mtmd_tts_params_default(void);

/**
 * Initialize TTS context
 *
 * @param thinker_model  The main LLM model (Thinker)
 * @param talker_model   The TTS model containing Talker, Code Predictor, and Code2Wav weights
 * @param params         TTS parameters
 * @return               TTS context, or NULL on failure
 */
mtmd_tts_context * mtmd_tts_init(
    const struct llama_model * thinker_model,
    const struct llama_model * talker_model,
    struct mtmd_tts_params params);

/**
 * Free TTS context
 */
void mtmd_tts_free(mtmd_tts_context * ctx);

/**
 * Generate speech from Thinker embeddings
 *
 * The input embeddings should be from Thinker layer 18 (the "talker accept layer").
 * Shape: [n_tokens, n_embd] where n_embd is typically 2048.
 *
 * The pipeline:
 *   1. Apply text projection MLP (2048 -> 1024)
 *   2. Run Talker autoregressively to generate codec tokens
 *   3. Run Code Predictor to expand 1 -> 16 codebooks
 *   4. Run Code2Wav vocoder to synthesize audio
 *
 * @param ctx             TTS context
 * @param embeddings      Input embeddings from Thinker [n_tokens * n_embd floats]
 * @param n_tokens        Number of input tokens
 * @param output_samples  Output buffer for audio samples (caller allocates)
 * @param max_samples     Size of output buffer
 * @return                Number of samples generated, or -1 on error
 */
int mtmd_tts_generate(
    mtmd_tts_context * ctx,
    const float * embeddings,
    int n_tokens,
    float * output_samples,
    int max_samples);

/**
 * Generate speech from text (convenience function)
 *
 * This tokenizes the text, runs it through Thinker to get layer-18 embeddings,
 * then generates speech.
 *
 * @param ctx             TTS context
 * @param llama_ctx       Llama context for tokenization and Thinker inference
 * @param text            Input text to synthesize
 * @param output_samples  Output buffer for audio samples
 * @param max_samples     Size of output buffer
 * @return                Number of samples generated, or -1 on error
 */
int mtmd_tts_generate_from_text(
    mtmd_tts_context * ctx,
    struct llama_context * llama_ctx,
    const char * text,
    float * output_samples,
    int max_samples);

/**
 * Estimate output samples from codec token count
 *
 * Code2Wav upsamples by 480x (8*5*4*3), so n_codec_tokens * 480 = n_samples
 *
 * @param n_codec_tokens  Number of codec tokens
 * @return                Estimated number of audio samples
 */
int mtmd_tts_estimate_samples(int n_codec_tokens);

/**
 * Write audio samples to WAV file
 *
 * @param path         Output file path
 * @param samples      Audio samples in [-1, 1] range
 * @param n_samples    Number of samples
 * @param sample_rate  Sample rate (typically 24000)
 * @return             true on success
 */
bool mtmd_tts_write_wav(
    const char * path,
    const float * samples,
    int n_samples,
    int sample_rate);

/**
 * Check if model supports TTS output
 *
 * Looks for Talker and Code2Wav tensors in the model.
 *
 * @param model  Model to check
 * @return       true if TTS is supported
 */
bool mtmd_tts_supported(const struct llama_model * model);

#ifdef __cplusplus
}
#endif

#endif // MTMD_TTS_H
