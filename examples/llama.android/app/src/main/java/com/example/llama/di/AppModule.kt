package com.example.llama.di

import android.content.Context
import android.llama.cpp.InferenceEngine
import android.llama.cpp.KleidiLlama
import android.llama.cpp.TierDetection
import android.llama.cpp.gguf.GgufMetadataReader
import com.example.llama.data.db.AppDatabase
import com.example.llama.data.repo.ModelRepository
import com.example.llama.data.repo.ModelRepositoryImpl
import com.example.llama.data.repo.SystemPromptRepository
import com.example.llama.data.repo.SystemPromptRepositoryImpl
import com.example.llama.data.source.local.LocalFileDataSource
import com.example.llama.data.source.local.LocalFileDataSourceImpl
import com.example.llama.data.source.remote.GatedTypeAdapter
import com.example.llama.data.source.remote.HuggingFaceApiService
import com.example.llama.data.source.remote.HuggingFaceRemoteDataSource
import com.example.llama.data.source.remote.HuggingFaceRemoteDataSourceImpl
import com.example.llama.engine.BenchmarkService
import com.example.llama.engine.ConversationService
import com.example.llama.engine.InferenceService
import com.example.llama.engine.InferenceServiceImpl
import com.example.llama.engine.ModelLoadingService
import com.example.llama.engine.StubInferenceEngine
import com.example.llama.engine.StubTierDetection
import com.example.llama.monitoring.PerformanceMonitor
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import dagger.Binds
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import javax.inject.Singleton

const val HUGGINGFACE_HOST = "https://huggingface.co/"
const val HUGGINGFACE_DATETIME_FORMAT = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"

@Module
@InstallIn(SingletonComponent::class)
internal abstract class AppModule {

    @Binds
    abstract fun bindInferenceService(impl: InferenceServiceImpl) : InferenceService

    @Binds
    abstract fun bindModelLoadingService(impl: InferenceServiceImpl) : ModelLoadingService

    @Binds
    abstract fun bindBenchmarkService(impl: InferenceServiceImpl) : BenchmarkService

    @Binds
    abstract fun bindConversationService(impl: InferenceServiceImpl) : ConversationService

    @Binds
    abstract fun bindModelsRepository(impl: ModelRepositoryImpl): ModelRepository

    @Binds
    abstract fun bindSystemPromptRepository(impl: SystemPromptRepositoryImpl): SystemPromptRepository

    @Binds
    abstract fun bindLocalFileDataSource(impl: LocalFileDataSourceImpl) : LocalFileDataSource

    @Binds
    abstract fun bindHuggingFaceRemoteDataSource(impl: HuggingFaceRemoteDataSourceImpl): HuggingFaceRemoteDataSource

    companion object {
        const val USE_STUB_ENGINE = false

        @Provides
        fun provideInferenceEngine(@ApplicationContext context: Context): InferenceEngine {
            return if (USE_STUB_ENGINE) {
                StubInferenceEngine()
            } else {
                KleidiLlama.createInferenceEngine(context)
                    ?: throw InstantiationException("Cannot instantiate InferenceEngine!")
            }
        }

        @Provides
        fun provideTierDetection(@ApplicationContext context: Context): TierDetection {
            return if (USE_STUB_ENGINE) {
                StubTierDetection
            } else {
                KleidiLlama.getTierDetection(context)
            }
        }

        @Provides
        fun providePerformanceMonitor(@ApplicationContext context: Context) = PerformanceMonitor(context)

        @Provides
        fun provideAppDatabase(@ApplicationContext context: Context) = AppDatabase.getDatabase(context)

        @Provides
        fun providesModelDao(appDatabase: AppDatabase) = appDatabase.modelDao()

        @Provides
        fun providesSystemPromptDao(appDatabase: AppDatabase) = appDatabase.systemPromptDao()

        @Provides
        @Singleton
        fun providesGgufMetadataReader(): GgufMetadataReader = GgufMetadataReader.create()

        @Provides
        @Singleton
        fun provideOkhttpClient() = OkHttpClient.Builder()
            .addInterceptor(HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BODY
            }).build()

        @Provides
        @Singleton
        fun provideGson(): Gson = GsonBuilder()
            .setDateFormat(HUGGINGFACE_DATETIME_FORMAT)
            .registerTypeAdapter(Boolean::class.java, GatedTypeAdapter())
            .create()

        @Provides
        @Singleton
        fun provideHuggingFaceApiService(
            okHttpClient: OkHttpClient,
            gson: Gson,
        ): HuggingFaceApiService =
            Retrofit.Builder()
                .baseUrl(HUGGINGFACE_HOST)
                .client(okHttpClient)
                .addConverterFactory(GsonConverterFactory.create(gson))
                .build()
                .create(HuggingFaceApiService::class.java)
    }
}
