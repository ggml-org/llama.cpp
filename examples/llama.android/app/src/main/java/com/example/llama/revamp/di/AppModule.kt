package com.example.llama.revamp.di

import android.content.Context
import com.example.llama.revamp.data.local.AppDatabase
import com.example.llama.revamp.data.repository.ModelRepository
import com.example.llama.revamp.data.repository.ModelRepositoryImpl
import com.example.llama.revamp.data.repository.SystemPromptRepository
import com.example.llama.revamp.data.repository.SystemPromptRepositoryImpl
import com.example.llama.revamp.engine.BenchmarkService
import com.example.llama.revamp.engine.ConversationService
import com.example.llama.revamp.engine.InferenceEngine
import com.example.llama.revamp.engine.InferenceService
import com.example.llama.revamp.engine.InferenceServiceImpl
import com.example.llama.revamp.engine.ModelLoadingService
import com.example.llama.revamp.monitoring.PerformanceMonitor
import dagger.Binds
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
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

    companion object {
        @Provides
        @Singleton
        fun provideInferenceEngine() = InferenceEngine()

        @Provides
        @Singleton
        fun providePerformanceMonitor(@ApplicationContext context: Context) = PerformanceMonitor(context)

        @Provides
        fun provideAppDatabase(@ApplicationContext context: Context) = AppDatabase.getDatabase(context)

        @Provides
        fun providesModelDao(appDatabase: AppDatabase) = appDatabase.modelDao()

        @Provides
        fun providesSystemPromptDao(appDatabase: AppDatabase) = appDatabase.systemPromptDao()
    }
}
