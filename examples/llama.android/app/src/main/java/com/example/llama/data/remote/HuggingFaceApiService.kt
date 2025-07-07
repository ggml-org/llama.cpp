package com.example.llama.data.remote

import okhttp3.ResponseBody
import retrofit2.http.GET
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

    @GET("{modelId}/resolve/main/{filePath}")
    @Streaming
    suspend fun downloadModelFile(
        @Path("modelId") modelId: String,
        @Path("filePath") filePath: String
    ): ResponseBody
}
