// Depth heads maths shared by the engine (m5_engine.cpp) and the stage
// subprocess (backbone.cpp --depth-chain).
//
// One audio frame is sixteen code books sampled as a serial chain: book 0
// comes from a projection of the backbone hidden, and each later book sees a
// transform row of the previous code plus one depth-core decode step. The
// parent used to drive that chain one book per pipe round trip; with
// `--depth-chain` the whole loop runs inside the stage process and crosses
// the pipe once per frame instead of sixteen times.
//
// The maths is bit-for-bit the one the Java engine performs, including the
// JavaRandom noise stream, which the caller owns and passes in.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>

// The engine build always compiles the f16 kernel (targets armv8.2-a+fp16);
// stage builds opt in the same way. Host builds fall back to the f32 kernel.
#ifndef M5_HEADS_FP16
#define M5_HEADS_FP16 1
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#else
typedef uint16_t float16_t;  // opaque placeholder; the f16 kernel is compiled out
#endif

namespace m5 {
namespace heads {

constexpr int kWidth = 512;
constexpr int kVocab = 2051;
constexpr int kBooks = 16;
constexpr int kNoiseBooks = kBooks;

// Offsets (in floats) into the depth-weights.f32 mapping:
//   [0, projection)                        book-0 logits projection
//   [projection, projected)                projected hidden block
//   [projected, audio)                     per-book transform rows
//   [audio, audio + projection)            code -> next-hidden embedding rows
struct Layout {
    size_t projection;
    size_t projected;
    size_t audio;
};

inline Layout layout() {
    Layout l;
    l.projection = (size_t) kBooks * kVocab * kWidth;
    l.projected = l.projection + (size_t) kWidth * kWidth;
    l.audio = l.projected + (size_t) kBooks * kVocab * kWidth;
    return l;
}

// __restrict matters here: without it the compiler assumes the output may
// alias the weights and refuses to vectorise the dot products. `matrix16`
// mirrors `matrix` as f16 halves; when present the dot runs on the f16 copy
// because the weights are streamed, not the inputs.
inline void project(const float * __restrict matrix, const float * __restrict hidden,
                    float * __restrict out, int rows,
                    const float16_t * __restrict matrix16 = nullptr) {
#if defined(__aarch64__) && defined(M5_HEADS_FP16) && defined(__ARM_FEATURE_FP16)
    if (matrix16) {
        float16_t input[kWidth];
        for (int j = 0; j < kWidth; j += 4) {
            vst1_f16(input + j, vcvt_f16_f32(vld1q_f32(hidden + j)));
        }
        for (int row = 0; row < rows; row++) {
            const float16_t * w = matrix16 + (size_t) row * kWidth;
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int col = 0; col < kWidth; col += 16) {
                const float16x8_t w0 = vld1q_f16(w + col);
                const float16x8_t h0 = vld1q_f16(input + col);
                const float16x8_t w1 = vld1q_f16(w + col + 8);
                const float16x8_t h1 = vld1q_f16(input + col + 8);
#if defined(__ARM_FEATURE_FP16FML) || defined(__ARM_FEATURE_FP16_FML)
                acc = vfmlalq_low_f16(acc, w0, h0);
                acc = vfmlalq_high_f16(acc, w0, h0);
                acc = vfmlalq_low_f16(acc, w1, h1);
                acc = vfmlalq_high_f16(acc, w1, h1);
#else
                acc = vfmaq_f32(acc, vcvt_f32_f16(vget_low_f16(w0)), vcvt_f32_f16(vget_low_f16(h0)));
                acc = vfmaq_f32(acc, vcvt_f32_f16(vget_high_f16(w0)), vcvt_f32_f16(vget_high_f16(h0)));
                acc = vfmaq_f32(acc, vcvt_f32_f16(vget_low_f16(w1)), vcvt_f32_f16(vget_low_f16(h1)));
                acc = vfmaq_f32(acc, vcvt_f32_f16(vget_high_f16(w1)), vcvt_f32_f16(vget_high_f16(h1)));
#endif
            }
            out[row] = vaddvq_f32(acc);
        }
        return;
    }
#else
    (void) matrix16;
#endif
    for (int row = 0; row < rows; row++) {
        const float * w = matrix + (size_t) row * kWidth;
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int col = 0; col < kWidth; col += 4) {
            s0 += hidden[col + 0] * w[col + 0];
            s1 += hidden[col + 1] * w[col + 1];
            s2 += hidden[col + 2] * w[col + 2];
            s3 += hidden[col + 3] * w[col + 3];
        }
        out[row] = (s0 + s1) + (s2 + s3);
    }
}

// Per-row symmetric int8 mirror of the streamed project blocks (same coverage
// as the f16 mirror: kBooks*kVocab + kWidth rows). Halves the weight traffic
// versus f16; only the weights are quantised, the input stays f32.
inline size_t int8_rows() {
    return (size_t) kBooks * kVocab + kWidth;
}

inline void build_int8(const float * w, int8_t * dst, float * scales, size_t rows) {
    for (size_t r = 0; r < rows; r++) {
        const float * src = w + r * kWidth;
        float m = 0;
        for (int j = 0; j < kWidth; j++) {
            const float a = std::fabs(src[j]);
            if (a > m) m = a;
        }
        const float s = m > 0 ? m / 127.0f : 1.0f;
        scales[r] = s;
        const float inv = 1.0f / s;
        for (int j = 0; j < kWidth; j++) {
            float q = std::lrintf(src[j] * inv);
            if (q > 127.0f) q = 127.0f;
            if (q < -127.0f) q = -127.0f;
            dst[r * kWidth + j] = (int8_t) q;
        }
    }
}

inline void project8(const int8_t * __restrict matrix, const float * __restrict scales,
                     const float * __restrict hidden, float * __restrict out, int rows) {
#if defined(__aarch64__)
    for (int row = 0; row < rows; row++) {
        const int8_t * w = matrix + (size_t) row * kWidth;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int col = 0; col < kWidth; col += 16) {
            const int8x16_t v = vld1q_s8(w + col);
            const int16x8_t lo = vmovl_s8(vget_low_s8(v));
            const int16x8_t hi = vmovl_s8(vget_high_s8(v));
            acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vld1q_f32(hidden + col));
            acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vld1q_f32(hidden + col + 4));
            acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), vld1q_f32(hidden + col + 8));
            acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vld1q_f32(hidden + col + 12));
        }
        out[row] = vaddvq_f32(acc) * scales[row];
    }
#else
    (void) matrix; (void) scales; (void) hidden; (void) out; (void) rows;
    for (int row = 0; row < rows; row++) {
        const int8_t * w = matrix + (size_t) row * kWidth;
        float acc = 0;
        for (int col = 0; col < kWidth; col++) acc += (float) w[col] * hidden[col];
        out[row] = acc * scales[row];
    }
#endif
}

// Top-k filtering without the full sort the Java engine performs: only the
// 10th largest logit is needed to build the cutoff. `noise` is the caller's
// JavaRandom stream, kBooks * kVocab large; the divide-by-noise argmax is the
// Gumbel-style trick the Java engine uses.
inline long sample(const float * source, const float * noise, int book, bool allow_eos) {
    float scaled[kVocab];
    for (int i = 0; i < kVocab; i++) scaled[i] = source[i] / 0.9f;
    if (!allow_eos) scaled[0] = -1e5f / 0.9f;

    float top[10];
    for (int i = 0; i < 10; i++) top[i] = -INFINITY;
    for (int i = 0; i < kVocab; i++) {
        float v = scaled[i];
        if (v <= top[9]) continue;
        int j = 9;
        while (j > 0 && top[j - 1] < v) { top[j] = top[j - 1]; j--; }
        top[j] = v;
    }
    const float cutoff = top[9];
    const float max = top[0];
    float sum = 0;
    float probs[kVocab];
    for (int i = 0; i < kVocab; i++) {
        float p = scaled[i] >= cutoff ? (float) exp((double) (scaled[i] - max)) : 0.0f;
        probs[i] = p;
        sum += p;
    }
    long selected = 0;
    float best = -1;
    for (int i = 0; i < kVocab; i++) {
        float score = (probs[i] / sum) / noise[(size_t) book * kVocab + i];
        if (score > best) { best = score; selected = i; }
    }
    return selected;
}

// Runs the sixteen-book chain for one frame. `decode(in, rows, out, ctx)`
// performs one depth-core step (1-2 rows in, kWidth out) and returns false on
// transport failure; the caller resets the core's state before each frame.
// Mirrors NativeDepth.run exactly. `w8/w8s` (optional, per-row scaled int8
// mirror of the project blocks) take precedence over the f16 mirror.
template <typename Decode>
inline bool chain(const float * w, const float16_t * w16, const float * hidden,
                  const float * noise, bool allow_eos, long codes[kBooks],
                  float * next, Decode && decode,
                  const int8_t * w8 = nullptr, const float * w8s = nullptr) {
    const Layout off = layout();
    const bool have16 = w16 != nullptr;
    float logits[kVocab];
    auto proj = [&](size_t f32_off, const float * h, float * out, int rows) {
        if (w8) {
            project8(w8 + f32_off, w8s + f32_off / kWidth, h, out, rows);
        } else {
            project(w + f32_off, h, out, rows, have16 ? w16 + f32_off : nullptr);
        }
    };
    proj(0, hidden, logits, kVocab);
    codes[0] = sample(logits, noise, 0, allow_eos);

    // Book 1 consumes the projected hidden plus the embedding of code 0;
    // every later book only sees the transform of the previous code, which
    // is what the Java engine's rolling buffer ends up holding.
    float local[2 * kWidth];
    float projected_hidden[kWidth];
    proj(off.projection, hidden, projected_hidden, kWidth);
    // M5_ZERO_TRANSFORM: ablation experiment - replace the code-conditioned
    // transform rows with zeros to measure how much the serial chain actually
    // matters. If the sampled codes stay the same, the dependency is inert and
    // the 15 decode calls can be batched with dummy rows.
    static const bool kZeroTransform = getenv("M5_ZERO_TRANSFORM") != nullptr;
    memcpy(local, projected_hidden, sizeof(float) * kWidth);
    if (kZeroTransform) {
        memset(local + kWidth, 0, sizeof(float) * kWidth);
    } else {
        memcpy(local + kWidth, w + off.projected + codes[0] * kWidth, sizeof(float) * kWidth);
    }

    float decoded[kWidth];
    for (int book = 1; book < kBooks; book++) {
        const int rows = book == 1 ? 2 : 1;
        if (book > 1) {
            if (kZeroTransform) {
                memset(local, 0, sizeof(float) * kWidth);
            } else {
                const float * transform =
                    w + off.projected + ((size_t) (book - 1) * kVocab + codes[book - 1]) * kWidth;
                memcpy(local, transform, sizeof(float) * kWidth);
            }
        }
        if (!decode(local, rows, decoded)) return false;
        proj((size_t) book * kVocab * kWidth, decoded, logits, kVocab);
        codes[book] = sample(logits, noise, book, true);
    }

    for (int j = 0; j < kWidth; j++) next[j] = 0;
    for (int book = 0; book < kBooks; book++) {
        const float * row = w + off.audio + ((size_t) book * kVocab + codes[book]) * kWidth;
        for (int j = 0; j < kWidth; j++) next[j] += row[j];
    }
    return true;
}

// Builds the f16 mirror of the two projected blocks (the streamed ones).
// `halves` floats, matching the parent's w16_ layout: [0, projection) of the
// projection block plus the kWidth*kWidth projected block.
inline size_t f16_halves() {
    const Layout off = layout();
    return (size_t) kBooks * kVocab * kWidth + (size_t) kWidth * kWidth;
}

inline void build_f16(const float * w, float16_t * dst, size_t halves) {
#if defined(__aarch64__) && defined(M5_HEADS_FP16) && defined(__ARM_FEATURE_FP16)
    const float * src = w;
    size_t i = 0;
    for (; i + 4 <= halves; i += 4) {
        vst1_f16(dst + i, vcvt_f16_f32(vld1q_f32(src + i)));
    }
    for (; i < halves; i++) dst[i] = (float16_t) src[i];
#else
    for (size_t i = 0; i < halves; i++) dst[i] = (float16_t) w[i];
#endif
}

}  // namespace heads
}  // namespace m5
