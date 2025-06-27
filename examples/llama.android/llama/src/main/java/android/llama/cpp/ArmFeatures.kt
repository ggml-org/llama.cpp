package android.llama.cpp

/**
 * Represents an ARM CPU feature with its metadata.
 */
data class ArmFeature(
    val name: String,
    val displayName: String,
    val description: String,
    val armDocUrl: String
)

/**
 * Helper class to map LLamaTier to supported ARM features.
 */
object ArmFeaturesMapper {

    /**
     * UI display item combining feature info with support status.
     */
    data class DisplayItem(
        val feature: ArmFeature,
        val isSupported: Boolean
    )

    /**
     * All ARM features supported by the library, in order of introduction.
     *
     * TODO-han.yin: fix the broken hyperlinks above!
     */
    val allFeatures = listOf(
        ArmFeature(
            name = "ASIMD",
            displayName = "ASIMD",
            description = "Advanced SIMD (NEON) - baseline vectorization",
            armDocUrl = "https://developer.arm.com/architectures/instruction-sets/simd-isas/neon"
        ),
        ArmFeature(
            name = "DOTPROD",
            displayName = "DOTPROD",
            description = "Dot Product instructions for neural networks",
            armDocUrl = "https://developer.arm.com/architectures/instruction-sets/intrinsics/dotprod"
        ),
        ArmFeature(
            name = "I8MM",
            displayName = "I8MM",
            description = "Integer 8-bit Matrix Multiplication",
            armDocUrl = "https://developer.arm.com/documentation/102107/latest/Matrix-multiplication-instructions"
        ),
        ArmFeature(
            name = "SVE",
            displayName = "SVE",
            description = "Scalable Vector Extension",
            armDocUrl = "https://developer.arm.com/architectures/instruction-sets/simd-isas/sve"
        ),
        ArmFeature(
            name = "SME",
            displayName = "SME",
            description = "Scalable Matrix Extension",
            armDocUrl = "https://developer.arm.com/architectures/instruction-sets/simd-isas/sme"
        )
    )

    /**
     * Maps a LLamaTier to its supported ARM features.
     * Returns a list of booleans where each index corresponds to allFeatures.
     */
    fun getSupportedFeatures(tier: LLamaTier?): List<Boolean> =
        when (tier) {
            LLamaTier.T0 -> listOf(true, false, false, false, false)  // ASIMD only
            LLamaTier.T1 -> listOf(true, true, false, false, false)   // ASIMD + DOTPROD
            LLamaTier.T2 -> listOf(true, true, true, false, false)    // ASIMD + DOTPROD + I8MM
            LLamaTier.T3 -> listOf(true, true, true, true, false)     // ASIMD + DOTPROD + I8MM + SVE
            // TODO-han.yin: implement T4 once obtaining an Android device with SME!
            null -> listOf(false, false, false, false, false)         // No tier detected
        }

    /**
     * Gets the feature support data for UI display.
     */
    fun getFeatureDisplayData(tier: LLamaTier?): List<DisplayItem> =
        getSupportedFeatures(tier).let { flags ->
            allFeatures.mapIndexed { index, feature ->
                DisplayItem(
                    feature = feature,
                    isSupported = flags.getOrElse(index) { false }
                )
            }
        }
}
