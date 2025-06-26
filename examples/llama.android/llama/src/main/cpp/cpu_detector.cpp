#include <jni.h>
#include <cpuinfo_aarch64.h>
#include <android/log.h>
#include <string>

using namespace cpu_features;

#define LOG_TAG "CpuDetector"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

static const Aarch64Info info = GetAarch64Info();
static const Aarch64Features features = info.features;

extern "C" JNIEXPORT jint JNICALL
Java_android_llama_cpp_InferenceEngineLoader_getOptimalTier(
        JNIEnv* env,
        jclass clazz) {
    int tier = 0;  // Default to T0 (baseline)

    // Check features in reverse order (highest tier first)
    // TODO-han.yin: implement T4 once obtaining an Android device with SME!
    if (features.sve && features.sve2) {
        tier = 3;  // T3: ARMv9-a with SVE/SVE2
        LOGI("Detected SVE/SVE2 support - selecting T3");
    }
    else if (features.i8mm) {
        tier = 2;  // T2: ARMv8.6-a with i8mm
        LOGI("Detected i8mm support - selecting T2");
    }
    else if (features.asimddp) {
        tier = 1;  // T1: ARMv8.2-a with dotprod
        LOGI("Detected dotprod support - selecting T1");
    }
    else if (features.asimd) {
        tier = 0;  // T0: baseline ARMv8-a with SIMD
        LOGI("Detected basic ASIMD support - selecting T0");
    }
    else {
        // Fallback - this shouldn't happen on arm64-v8a devices
        tier = 0;
        LOGI("No expected features detected - falling back to T0");
    }

    return tier;
}

// Optional: Keep a feature string function for debugging
extern "C" JNIEXPORT jstring JNICALL
Java_android_llama_cpp_InferenceEngineLoader_getCpuFeaturesString(
        JNIEnv* env,
        jclass clazz) {
    std::string text;

    if (features.asimd) text += "ASIMD ";
    if (features.asimddp) text += "ASIMDDP ";
    if (features.i8mm) text += "I8MM ";
    if (features.sve) text += "SVE ";
    if (features.sve2) text += "SVE2 ";
    if (features.sme) text += "SME ";

    return env->NewStringUTF(text.c_str());
}
