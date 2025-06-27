package android.llama.cpp

/**
 * Public interface for [LLamaTier] detection information.
 */
interface TierDetection {
    val detectedTier: LLamaTier?
    fun clearCache()
}

/**
 * ARM optimization tiers supported by the Kleidi-Llama library.
 * Higher tiers provide better performance on supported hardware.
 */
enum class LLamaTier(val rawValue: Int, val libraryName: String, val description: String) {
    T0(0, "llama_android_t0", "ARMv8-a baseline with SIMD"),
    T1(1, "llama_android_t1", "ARMv8.2-a with DotProd"),
    T2(2, "llama_android_t2", "ARMv8.6-a with DotProd + I8MM"),
    T3(3, "llama_android_t3", "ARMv9-a with DotProd + I8MM + SVE/SVE2");
    // TODO-han.yin: implement T4 once obtaining an Android device with SME!

    companion object {
        fun fromRawValue(value: Int): LLamaTier? = entries.find { it.rawValue == value }

        val maxSupportedTier = T3
    }
}
