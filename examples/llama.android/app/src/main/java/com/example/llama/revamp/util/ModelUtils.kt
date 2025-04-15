package com.example.llama.revamp.util

import java.util.Locale


/**
 * Convert bytes into human readable sizes
 */
fun formatSize(sizeInBytes: Long) = when {
    sizeInBytes >= 1_000_000_000 -> {
        val sizeInGb = sizeInBytes / 1_000_000_000.0
        String.format(Locale.getDefault(), "%.2f GB", sizeInGb)
    }
    sizeInBytes >= 1_000_000 -> {
        val sizeInMb = sizeInBytes / 1_000_000.0
        String.format(Locale.getDefault(), "%.2f MB", sizeInMb)
    }
    else -> {
        val sizeInKb = sizeInBytes / 1_000.0
        String.format(Locale.getDefault(), "%.2f KB", sizeInKb)
    }
}

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
 *
 * TODO-han.yin: Replace with GGUF parsing, also to be moved into the util object
 */
fun extractModelTypeFromFilename(filename: String): String? {
    val lowerFilename = filename.lowercase()
    return listOf("llama", "mistral", "phi", "qwen", "falcon", "mpt")
        .firstNotNullOfOrNull { type ->
            if (lowerFilename.contains(type)) {
                type.replaceFirstChar {
                    if (it.isLowerCase()) it.titlecase(Locale.ROOT) else it.toString()
                }
            } else { null }
        }
}
