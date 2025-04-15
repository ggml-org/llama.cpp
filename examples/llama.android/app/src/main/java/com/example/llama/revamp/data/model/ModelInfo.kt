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

    companion object {
        /**
         * Creates a list of sample models for development and testing.
         */
        fun getSampleModels(): List<ModelInfo> {
            return listOf(
                ModelInfo(
                    id = "mistral-7b",
                    name = "Mistral 7B",
                    path = "/storage/models/mistral-7b-q4_0.gguf",
                    sizeInBytes = 4_000_000_000,
                    parameters = "7B",
                    quantization = "Q4_K_M",
                    type = "Mistral",
                    contextLength = 8192,
                    lastUsed = System.currentTimeMillis() - 86400000 // 1 day ago
                ),
                ModelInfo(
                    id = "llama2-13b",
                    name = "Llama 2 13B",
                    path = "/storage/models/llama2-13b-q5_k_m.gguf",
                    sizeInBytes = 8_500_000_000,
                    parameters = "13B",
                    quantization = "Q5_K_M",
                    type = "Llama",
                    contextLength = 4096,
                    lastUsed = System.currentTimeMillis() - 259200000 // 3 days ago
                ),
                ModelInfo(
                    id = "phi-2",
                    name = "Phi-2",
                    path = "/storage/models/phi-2.gguf",
                    sizeInBytes = 2_800_000_000,
                    parameters = "2.7B",
                    quantization = "Q4_0",
                    type = "Phi",
                    contextLength = 2048,
                    lastUsed = null
                )
            )
        }
    }
}
