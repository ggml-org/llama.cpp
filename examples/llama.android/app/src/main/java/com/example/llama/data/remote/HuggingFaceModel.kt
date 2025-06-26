package com.example.llama.data.remote

data class HuggingFaceModel(
    val id: String,
    val modelId: String,
    val likes: Int?,
    val trendingScore: Int?,
    val private: Boolean?,
    val downloads: Int?,
    val tags: List<String>?,
    val pipeline_tag: String?,
    val library_name: String?,
    val createdAt: String?
)
