package com.arm.aiplayground.util

import java.util.Locale


@Deprecated("Use GgufMetadataReader instead!")
class NaiveMetadataExtractor private constructor() {
    /**
     * Try to extract parameters by looking for patterns like 7B, 13B, etc.
     */
    fun extractParametersFromFilename(filename: String): String? =
        Regex("([0-9]+(\\.[0-9]+)?)[bB]").find(filename)?.value?.uppercase()

    /**
     * Try to extract quantization by looking for patterns like Q4_0, Q5_K_M, etc.
     */
    fun extractQuantizationFromFilename(filename: String) =
        listOf(
            Regex("[qQ][0-9]_[0-9]"),
            Regex("[qQ][0-9]_[kK]_[mM]"),
            Regex("[qQ][0-9]_[kK]"),
            Regex("[qQ][0-9][fF](16|32)")
        ).firstNotNullOfOrNull {
            it.find(filename)?.value?.uppercase()
        }

    /**
     * Try to extract model type (Llama, Mistral, etc.)
     */
    fun extractModelTypeFromFilename(filename: String): String? =
        filename.lowercase().let { lowerFilename ->
            listOf("llama", "mistral", "phi", "qwen", "falcon", "mpt")
                .firstNotNullOfOrNull { type ->
                    if (lowerFilename.contains(type)) {
                        type.replaceFirstChar {
                            if (it.isLowerCase()) it.titlecase(Locale.ROOT) else it.toString()
                        }
                    } else { null }
                }
        }
}
