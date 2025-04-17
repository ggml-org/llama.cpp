package com.example.llama.revamp.data.model

import com.example.llama.revamp.util.formatSize

/**
 * Data class containing information about an LLM model.
 */
data class ModelInfo(
    val id: String,
    val name: String,
    val path: String,
    val sizeInBytes: Long,
    val parameters: String?,
    val quantization: String?,
    val type: String?,
    val contextLength: Int?,
    val lastUsed: Long? = null
) {
    val formattedSize: String
        get() = formatSize(sizeInBytes)
}
