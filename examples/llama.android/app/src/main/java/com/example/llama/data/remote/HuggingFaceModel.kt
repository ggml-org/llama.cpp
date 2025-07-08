package com.example.llama.data.remote

import android.net.Uri
import androidx.core.net.toUri
import com.example.llama.di.HUGGINGFACE_HOST
import java.util.Date

private const val FILE_EXTENSION_GGUF = ".gguf"

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
    data class Sibling(
        val rfilename: String,
    )

    fun getGgufFilename(): String? =
        siblings.map { it.rfilename }.firstOrNull { it.endsWith(FILE_EXTENSION_GGUF) }

    fun toDownloadInfo() = getGgufFilename()?.let { HuggingFaceDownloadInfo(_id, modelId, it) }
}

data class HuggingFaceDownloadInfo(
    val _id: String,
    val modelId: String,
    val filename: String,
) {
    val uri: Uri
        get() = "$HUGGINGFACE_HOST${modelId}/resolve/main/$filename".toUri()
}
