package com.example.llama.revamp.di

import android.content.Context
import android.llama.cpp.InferenceEngine
import android.llama.cpp.LLamaAndroid
import com.example.llama.revamp.data.local.AppDatabase
import com.example.llama.revamp.data.remote.HuggingFaceApiService
import com.example.llama.revamp.data.remote.HuggingFaceRemoteDataSource
import com.example.llama.revamp.data.remote.HuggingFaceRemoteDataSourceImpl
import com.example.llama.revamp.data.repository.ModelRepository
import com.example.llama.revamp.data.repository.ModelRepositoryImpl
import com.example.llama.revamp.data.repository.SystemPromptRepository
import com.example.llama.revamp.data.repository.SystemPromptRepositoryImpl
import com.example.llama.revamp.engine.BenchmarkService
import com.example.llama.revamp.engine.ConversationService
import com.example.llama.revamp.engine.InferenceService
import com.example.llama.revamp.engine.InferenceServiceImpl
import com.example.llama.revamp.engine.ModelLoadingService
import com.example.llama.revamp.engine.StubInferenceEngine
import com.example.llama.revamp.monitoring.PerformanceMonitor
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
    abstract fun bindsModelsRepository(impl: ModelRepositoryImpl): ModelRepository

    @Binds
    abstract fun bindsSystemPromptRepository(impl: SystemPromptRepositoryImpl): SystemPromptRepository

    @Binds
    abstract fun bindHuggingFaceRemoteDataSource(
        impl: HuggingFaceRemoteDataSourceImpl
    ): HuggingFaceRemoteDataSource

    companion object {
        @Provides
        fun provideInferenceEngine(): InferenceEngine {
            val useRealEngine = true
            return if (useRealEngine) LLamaAndroid.instance() else StubInferenceEngine()
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
        fun provideOkhttpClient() = OkHttpClient.Builder()
            .addInterceptor(HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BODY
            }).build()

        @Provides
        @Singleton
        fun provideHuggingFaceApiService(okHttpClient: OkHttpClient): HuggingFaceApiService =
            Retrofit.Builder()
                .baseUrl("https://huggingface.co/")
                .client(okHttpClient)
                .addConverterFactory(GsonConverterFactory.create())
                .build()
                .create(HuggingFaceApiService::class.java)
    }
}
