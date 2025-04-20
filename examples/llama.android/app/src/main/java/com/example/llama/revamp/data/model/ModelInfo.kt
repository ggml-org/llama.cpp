package com.example.llama.revamp.data.model

import com.example.llama.revamp.util.FileType
import com.example.llama.revamp.util.GgufMetadata
import com.example.llama.revamp.util.formatContextLength
import com.example.llama.revamp.util.formatFileByteSize


/**
 * Data class containing information about an LLM model.
 */
data class ModelInfo(
    val id: String,
    val name: String,
    val path: String,
    val sizeInBytes: Long,
    val metadata: GgufMetadata,
    val dateAdded: Long,
    val dateLastUsed: Long? = null,
) {
    val formattedFullName: String
        get() = metadata.fullModelName ?: name

    val formattedFileSize: String
        get() = formatFileByteSize(sizeInBytes)

    val formattedArchitecture: String
        get() = metadata.architecture?.architecture ?: "-"

    val formattedParamSize: String
        get() = metadata.basic.sizeLabel ?: "-"

    val formattedContextLength: String
        get() = metadata.dimensions?.contextLength?.let { formatContextLength(it) } ?: "-"

    val formattedQuantization: String
        get() = metadata.architecture?.fileType?.let { FileType.fromCode(it).label } ?: "-"

    val tags: List<String>? = metadata.additional?.tags?.takeIf { it.isNotEmpty() }

    val languages: List<String>? = metadata.additional?.languages?.takeIf { it.isNotEmpty() }
}
