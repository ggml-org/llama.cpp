package com.example.llama.data.remote

import java.util.Date

data class HuggingFaceModelDetails(
    val _id: String,
    val id: String,
    val modelId: String,

    val author: String,
    val createdAt: Date?,
    val lastModified: Date?,

    val library_name: String?,
    val pipeline_tag: String?,
    val tags: List<String>?,

    val private: Boolean?,
    val disabled: Boolean?,
    val gated: Boolean?,

    val likes: Int?,
    val downloads: Int?,

    val usedStorage: Long?,
    val sha: String?,

    val cardData: CardData?,
    val siblings: List<Sibling>?,
    val widgetData: List<WidgetData>?,

    val gguf: Gguf?,
) {
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
}
