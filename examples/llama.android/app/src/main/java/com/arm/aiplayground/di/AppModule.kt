package com.arm.aiplayground.di

import android.content.Context
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.arm.aichat.TierDetection
import com.arm.aichat.gguf.GgufMetadataReader
import com.arm.aiplayground.data.db.AppDatabase
import com.arm.aiplayground.data.repo.ModelRepository
import com.arm.aiplayground.data.repo.ModelRepositoryImpl
import com.arm.aiplayground.data.repo.SystemPromptRepository
import com.arm.aiplayground.data.repo.SystemPromptRepositoryImpl
import com.arm.aiplayground.data.source.local.LocalFileDataSource
import com.arm.aiplayground.data.source.local.LocalFileDataSourceImpl
import com.arm.aiplayground.data.source.remote.GatedTypeAdapter
import com.arm.aiplayground.data.source.remote.HuggingFaceApiService
import com.arm.aiplayground.data.source.remote.HuggingFaceRemoteDataSource
import com.arm.aiplayground.data.source.remote.HuggingFaceRemoteDataSourceImpl
import com.arm.aiplayground.engine.BenchmarkService
import com.arm.aiplayground.engine.ConversationService
import com.arm.aiplayground.engine.InferenceService
import com.arm.aiplayground.engine.InferenceServiceImpl
import com.arm.aiplayground.engine.ModelLoadingService
import com.arm.aiplayground.engine.StubInferenceEngine
import com.arm.aiplayground.engine.StubTierDetection
import com.arm.aiplayground.monitoring.PerformanceMonitor
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import dagger.Binds
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import okhttp3.OkHttpClient
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
                AiChat.getInferenceEngine(context)
            }
        }

        @Provides
        fun provideTierDetection(@ApplicationContext context: Context): TierDetection {
            return if (USE_STUB_ENGINE) {
                StubTierDetection
            } else {
                AiChat.getTierDetection(context)
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
        fun provideOkhttpClient() = OkHttpClient.Builder().build()

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
