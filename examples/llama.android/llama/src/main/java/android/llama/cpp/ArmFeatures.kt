package android.llama.cpp

/**
 * Represents an Arm® CPU feature with its metadata.
 */
data class ArmFeature(
    val name: String,
    val displayName: String,
    val description: String,
    val armDocUrl: String
)

/**
 * Helper class to map LLamaTier to supported Arm® features.
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
     * All Arm® features supported by the library, in order of introduction.
     */
    val allFeatures = listOf(
        ArmFeature(
            name = "ASIMD",
            displayName = "ASIMD",
            description = "Advanced SIMD (NEON) - baseline vectorization",
            armDocUrl = "https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/matrix-matrix-multiplication-neon-sve-and-sme-compared"
        ),
        ArmFeature(
            name = "DOTPROD",
            displayName = "DOTPROD",
            description = "Dot Product instructions for neural networks",
            armDocUrl = "https://community.arm.com/arm-community-blogs/b/tools-software-ides-blog/posts/exploring-the-arm-dot-product-instructions"
        ),
        ArmFeature(
            name = "I8MM",
            displayName = "I8MM",
            description = "Integer 8-bit Matrix Multiplication",
            armDocUrl = "https://community.arm.com/arm-community-blogs/b/ai-blog/posts/optimize-llama-cpp-with-arm-i8mm-instruction"
        ),
        ArmFeature(
            name = "SVE",
            displayName = "SVE",
            description = "Scalable Vector Extension",
            armDocUrl = "https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/sve2"
        ),
        ArmFeature(
            name = "SME",
            displayName = "SME",
            description = "Scalable Matrix Extension",
            armDocUrl = "https://newsroom.arm.com/blog/scalable-matrix-extension"
        )
    )

    /**
     * Gets the feature support data for UI display.
     */
    fun getFeatureDisplayData(tier: LLamaTier?): List<DisplayItem>? =
        getSupportedFeatures(tier).let { optFlags ->
            optFlags?.let { flags ->
                allFeatures.mapIndexed { index, feature ->
                    DisplayItem(
                        feature = feature,
                        isSupported = flags.getOrElse(index) { false }
                    )
                }
            }
        }

    /**
     * Maps a LLamaTier to its supported Arm® features.
     * Returns a list of booleans where each index corresponds to allFeatures.
     */
    private fun getSupportedFeatures(tier: LLamaTier?): List<Boolean>? =
        when (tier) {
            LLamaTier.NONE, null -> null                              // No tier detected
            LLamaTier.T0 -> listOf(true, false, false, false, false)  // ASIMD only
            LLamaTier.T1 -> listOf(true, true, false, false, false)   // ASIMD + DOTPROD
            LLamaTier.T2 -> listOf(true, true, true, false, false)    // ASIMD + DOTPROD + I8MM
            LLamaTier.T3 -> listOf(true, true, true, true, false)     // ASIMD + DOTPROD + I8MM + SVE/2
            LLamaTier.T4 -> listOf(true, true, true, true, true)     // ASIMD + DOTPROD + I8MM + SVE/2 + SME/2
        }
}
