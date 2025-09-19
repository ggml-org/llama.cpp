package com.arm.aiplayground.data.source.remote

import java.util.Date
import kotlin.String

data class HuggingFaceModelDetails(
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
    val disabled: Boolean?,

    val likes: Int,
    val downloads: Int,

    val usedStorage: Long?,

    val sha: String,
    val siblings: List<Sibling>,
    val cardData: CardData?,
    val widgetData: List<WidgetData>?,

    val gguf: Gguf?,

    val library_name: String?,
) {
    fun getGgufFilename(keywords: Array<String> = QUANTIZATION_Q4_0): String? =
        siblings.map { it.rfilename }
            .filter {
                it.endsWith(FILE_EXTENSION_GGUF, ignoreCase = true) }
            .firstOrNull { filename ->
                keywords.any { filename.contains(it, ignoreCase = true) }
            }

    fun toModel() = HuggingFaceModel(
        _id = this._id,
        id = this.id,
        modelId = this.modelId,
        author = this.author,
        createdAt = this.createdAt,
        lastModified = this.lastModified,
        pipeline_tag = this.pipeline_tag,
        tags = this.tags,
        private = this.private,
        gated = this.gated,
        likes = this.likes,
        downloads = this.downloads,
        sha = this.sha,
        siblings = this.siblings.map { Sibling(it.rfilename) },
        library_name = this.library_name,
    )

    fun toDownloadInfo() = getGgufFilename()?.let {
        HuggingFaceDownloadInfo(_id, modelId, it)
    }
}

data class Sibling(
    val rfilename: String,
)

data class Gguf(
    val total: Long?,
    val architecture: String?,
    val context_length: Int?,
    val chat_template: String?,
    val bos_token: String?,
    val eos_token: String?,
)

data class CardData(
    val base_model: String?,
    val language: List<String>?,
    val license: String?,
    val pipeline_tag: String?,
    val tags: List<String>?,
)

data class WidgetData(
    val text: String
)
