package com.arm.aiplayground.data.source.remote

import android.net.Uri
import androidx.core.net.toUri
import com.arm.aiplayground.di.HUGGINGFACE_HOST
import java.util.Date

internal const val FILE_EXTENSION_GGUF = ".GGUF"
internal val QUANTIZATION_Q4_0 = arrayOf("Q4_0", "Q4-0")

data class HuggingFaceModel(
    val _id: String,
    val id: String,
    val modelId: String,

    val author: String,
    val createdAt: Date,
    val lastModified: Date,

    val pipeline_tag: String,
    val tags: List<String>,

    val private: Boolean,
    val gated: Boolean,

    val likes: Int,
    val downloads: Int,

    val sha: String,
    val siblings: List<Sibling>,

    val library_name: String?,
) {
    fun getGgufFilename(keywords: Array<String> = QUANTIZATION_Q4_0): String? =
        siblings.map { it.rfilename }
            .filter {
                it.endsWith(FILE_EXTENSION_GGUF, ignoreCase = true) }
            .firstOrNull { filename ->
                keywords.any { filename.contains(it, ignoreCase = true) }
            }

    fun anyFilenameContains(keywords: Array<String>): Boolean =
        siblings.map { it.rfilename }
            .any { filename ->
                keywords.any { filename.contains(it, ignoreCase = true) }
            }

    fun toDownloadInfo() = getGgufFilename()?.let {
        HuggingFaceDownloadInfo(_id, modelId, it)
    }
}

data class HuggingFaceDownloadInfo(
    val _id: String,
    val modelId: String,
    val filename: String,
) {
    val uri: Uri
        get() = "$HUGGINGFACE_HOST${modelId}/resolve/main/$filename".toUri()
}
