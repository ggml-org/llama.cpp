package com.arm.aiplayground.data.source.remote

import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.GET
import retrofit2.http.HEAD
import retrofit2.http.Path
import retrofit2.http.Query
import retrofit2.http.Streaming

interface HuggingFaceApiService {
    @GET("api/models")
    suspend fun getModels(
        @Query("search") search: String? = null,
        @Query("author") author: String? = null,
        @Query("filter") filter: String? = null,
        @Query("sort") sort: String? = null,
        @Query("direction") direction: String? = null,
        @Query("limit") limit: Int? = null,
        @Query("full") full: Boolean? = null,
    ): List<HuggingFaceModel>

    @GET("api/models/{modelId}")
    suspend fun getModelDetails(@Path("modelId") modelId: String): HuggingFaceModelDetails

    @HEAD("{modelId}/resolve/main/{filename}")
    suspend fun getModelFileHeader(
        @Path("modelId", encoded = true) modelId: String,
        @Path("filename", encoded = true) filename: String
    ): Response<Void>

    @Deprecated("Use DownloadManager instead!")
    @GET("{modelId}/resolve/main/{filename}")
    @Streaming
    suspend fun downloadModelFile(
        @Path("modelId") modelId: String,
        @Path("filename") filename: String
    ): ResponseBody
}
