package com.example.llama.data.remote

import java.util.Date

data class HuggingFaceModel(
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
    val gated: Boolean?,

    val likes: Int?,
    val downloads: Int?,

    val sha: String?,

    val siblings: List<Sibling>?,
) {
    data class Sibling(
        val rfilename: String,
    )
}
