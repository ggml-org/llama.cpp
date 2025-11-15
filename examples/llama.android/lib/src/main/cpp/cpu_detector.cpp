#include <jni.h>
#include <android/log.h>
#include <string>

#if defined(__aarch64__)
#include <cpuinfo_aarch64.h>
using namespace cpu_features;
static const Aarch64Info info = GetAarch64Info();
static const Aarch64Features features = info.features;
#endif

#define LOG_TAG "CpuDetector"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_TierDetectionImpl_getOptimalTier(
        JNIEnv*  /*env*/,
        jobject  /*clazz*/) {
    int tier = 0;  // Default to T0 (baseline)

#if defined(__aarch64__)
    // Check features in reverse order (highest tier first)
    if (features.sme) {
        tier = 5;  // T5: ARMv9-a with SVE/SVE2
        LOGI("Detected SME support - selecting T5");
    }
    else if (features.sve && features.sve2) {
        tier = 4;  // T4: ARMv9-a with SVE/SVE2
        LOGI("Detected SVE/SVE2 support - selecting T4");
    }
    else if (features.i8mm) {
        tier = 3;  // T3: ARMv8.6-a with i8mm
        LOGI("Detected i8mm support - selecting T3");
    }
    else if (features.asimddp) {
        tier = 2;  // T2: ARMv8.2-a with dotprod
        LOGI("Detected dotprod support - selecting T2");
    }
    else if (features.asimd) {
        tier = 1;  // T1: baseline ARMv8-a with ASIMD
        LOGI("Detected basic ASIMD support - selecting T1");
    }
    else {
        // Fallback - this shouldn't happen on arm64-v8a devices
        tier = 1;
        LOGI("No expected features detected - falling back to T1");
    }
#else
    LOGI("non aarch64 architecture detected - defaulting to T0");
#endif

    return tier;
}

// Optional: Keep a feature string function for debugging
extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_TierDetectionImpl_getCpuFeaturesString(
        JNIEnv* env,
        jobject  /*clazz*/) {
    std::string text;

#if defined(__aarch64__)
    if (features.asimd) text += "ASIMD ";
    if (features.asimddp) text += "ASIMDDP ";
    if (features.i8mm) text += "I8MM ";
    if (features.sve) text += "SVE ";
    if (features.sve2) text += "SVE2 ";
    if (features.sme) text += "SME ";
#else
    LOGI("non aarch64 architecture detected");
#endif

    return env->NewStringUTF(text.c_str());
}
