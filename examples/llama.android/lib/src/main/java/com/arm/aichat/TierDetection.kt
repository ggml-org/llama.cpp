package com.arm.aichat

/**
 * Public interface for [ArmCpuTier] detection information.
 */
interface TierDetection {
    fun getDetectedTier(): ArmCpuTier?
    fun clearCache()
}

/**
 * ARM optimization tiers supported by this library.
 * Higher tiers provide better performance on supported hardware.
 */
enum class ArmCpuTier(val rawValue: Int, val description: String) {
    NONE(0, "No valid Arm® optimization available!"),
    T1(1, "ARMv8-a baseline with ASIMD"),
    T2(2, "ARMv8.2-a with DotProd"),
    T3(3, "ARMv8.6-a with DotProd + I8MM"),
    T4(4, "ARMv9-a with DotProd + I8MM + SVE/SVE2"),
    T5(5, "ARMv9.2-a with DotProd + I8MM + SVE/SVE2 + SME/SME2");

    companion object {
        fun fromRawValue(value: Int): ArmCpuTier? = entries.find { it.rawValue == value }

        val maxSupportedTier = T5
    }
}
