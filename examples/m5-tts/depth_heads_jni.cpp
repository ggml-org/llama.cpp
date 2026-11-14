#include <jni.h>

#include <cstddef>

extern "C" JNIEXPORT void JNICALL Java_com_m5_offline_NativeDepth_project(
        JNIEnv * env, jclass, jobject weights, jint offset, jfloatArray input, jfloatArray output) {
    constexpr int width = 512;
    const auto * matrix = static_cast<const float *>(env->GetDirectBufferAddress(weights));
    if (!matrix) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Expected mapped depth weights");
        return;
    }
    float hidden[width];
    env->GetFloatArrayRegion(input, 0, width, hidden);
    if (env->ExceptionCheck()) return;
    constexpr int max_outputs = 2051;
    float result[max_outputs];
    const auto rows = env->GetArrayLength(output);
    if (rows > max_outputs) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Invalid depth projection size");
        return;
    }
    for (int row = 0; row < rows; ++row) {
        const float * w = matrix + offset + static_cast<size_t>(row) * width;
        float sum = 0;
        for (int col = 0; col < width; ++col) sum += hidden[col] * w[col];
        result[row] = sum;
    }
    env->SetFloatArrayRegion(output, 0, rows, result);
}
